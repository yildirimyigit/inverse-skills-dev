"""Steps 1-4 of the framework, run end-to-end on ManiSkill3 PushCube.

The PlanRob experiment executed these steps individually: the symbolic prefix
was a hand-scripted pick-and-place and the active/fence partition was a
hardcoded list. This script runs them for real and checks whether the planner
reproduces what the paper claims:

  1. Extract the PUSH operator from N scripted forward-push demonstrations.
  2. Derive the inverse target  I(o) = Pre u Del u not-Add.
  3. BFS-plan the inverse prefix over the scripted primitive library.
  4. Execute the planned prefix, then partition I(o) into fences F and the
     active residual A from the *real* handoff scene.

Success criteria:
  SC1  BFS returns  pick(cube) -> place(src).
  SC2  The post-execution partition yields A = {in_region(cube,src)} and
       F = {tcp_near(cube), gripper_open(), not in_region(cube,goal)}.

`src` and `goal` are treated as environment-provided regions. They are carried
inside each scene, so one predicate registry grounds correctly across demos
with different cube initialisations.

Run (from repo root):
    python scripts/planrob_pushcube_symbolic_pipeline.py
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np

# Scripted-only pipeline: no policy is loaded, so stub SB3 if it is absent
# (the reused demo module imports SAC / BaseCallback at import time).
try:
    import stable_baselines3  # noqa: F401
except ModuleNotFoundError:
    _sb3 = types.ModuleType("stable_baselines3")
    _sb3.__path__ = []
    _sb3.SAC = object
    _common = types.ModuleType("stable_baselines3.common")
    _common.__path__ = []
    _callbacks = types.ModuleType("stable_baselines3.common.callbacks")

    class _BaseCallback:
        pass

    _callbacks.BaseCallback = _BaseCallback
    _callbacks.CallbackList = object
    _common.callbacks = _callbacks
    _sb3.common = _common
    sys.modules["stable_baselines3"] = _sb3
    sys.modules["stable_baselines3.common"] = _common
    sys.modules["stable_baselines3.common.callbacks"] = _callbacks

_SPEC = importlib.util.spec_from_file_location(
    "pushcube_full", "scripts/planrob_inverse_rl_pushcube_full_demo_2d_action.py"
)
pushcube_full = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pushcube_full)
demo = pushcube_full.demo
pc = pushcube_full.pc

import gymnasium as gym  # noqa: E402

from inverse_skills.core import Region  # noqa: E402
from inverse_skills.logging.rollout import ForwardRollout  # noqa: E402
from inverse_skills.operators.extractor import OperatorExtractor  # noqa: E402
from inverse_skills.operators.restoration import RestorationObjective  # noqa: E402
from inverse_skills.operators.toy_planner import ToyInversePlanner  # noqa: E402
from inverse_skills.predicates import (  # noqa: E402
    GripperOpenPredicate,
    InRegionPredicate,
    PredicateRegistry,
    TcpNearObjectPredicate,
)
from inverse_skills.toy.primitives import PrimitiveLibrary  # noqa: E402

# Environment-provided region geometry (the user supplies these; the framework
# does not infer them). `src` half-extent is the paper's 1 cm restoration
# tolerance — that tightness is what leaves a precision residual for RL. `goal`
# is looser because the scripted push only lands the cube to ~2 cm. z is
# generous so the vertical axis never dominates the box margin.
_SRC_HALF_XY = 0.01
_GOAL_HALF_XY = 0.04
_REGION_HALF_Z = 0.05
# Sharper than the InRegion default (0.02) so a cube at the centre of a 1 cm
# box saturates past the 0.80 precondition threshold instead of stalling at
# 0.73 — otherwise in_region(cube,src) never qualifies as a precondition.
_IN_REGION_TEMP = 0.01
# theta: V_p(s_h) >= theta => fence. Matched to the extractor's precondition
# threshold so "counts as restored" means the same thing in both phases.
_FENCE_THRESHOLD = 0.80


def _regions_for(cube_init: np.ndarray, push_dx: float) -> dict[str, Region]:
    """src/goal as environment-provided boxes around the pre- and post-push
    cube positions."""
    src_half = np.array([_SRC_HALF_XY, _SRC_HALF_XY, _REGION_HALF_Z], dtype=np.float32)
    goal_half = np.array([_GOAL_HALF_XY, _GOAL_HALF_XY, _REGION_HALF_Z], dtype=np.float32)
    src_c = np.asarray(cube_init, dtype=np.float32)
    goal_c = src_c + np.array([push_dx, 0.0, 0.0], dtype=np.float32)
    return {
        "src": Region("src", src_c - src_half, src_c + src_half),
        "goal": Region("goal", goal_c - goal_half, goal_c + goal_half),
    }


def _registry() -> PredicateRegistry:
    return PredicateRegistry([
        InRegionPredicate("cube", "src", temperature=_IN_REGION_TEMP),
        InRegionPredicate("cube", "goal", temperature=_IN_REGION_TEMP),
        GripperOpenPredicate(
            min_width=pushcube_full._GRIPPER_OPEN_MIN_WIDTH,
            temperature=pushcube_full._GRIPPER_OPEN_TEMP_TRAIN,
        ),
        TcpNearObjectPredicate(
            object_name="cube",
            distance_threshold=pushcube_full._TCP_NEAR_THRESHOLD_M,
            temperature=pushcube_full._TCP_NEAR_TEMP,
        ),
    ])


def _cube(obs) -> np.ndarray:
    return obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()


def _tcp(obs) -> np.ndarray:
    return obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy().copy()


def _push_dx(obs, cube_init: np.ndarray, fallback: float, use_env_goal: bool) -> float:
    goal = obs.get("extra", {}).get("goal_pos")
    if use_env_goal and goal is not None:
        return float(goal.squeeze()[:3].cpu().numpy()[0] - cube_init[0])
    return float(fallback)


# --- forward skill -----------------------------------------------------------


def _forward_demo(env, seed: int, fallback_dx: float, use_env_goal: bool):
    """One forward-push demonstration.

    The rollout spans the whole forward skill: first scene = the world before
    push(cube) runs, last scene = the world it leaves behind (after the stroke,
    the lift that breaks contact, and the gripper opening). Cutting at the push
    *stroke* instead would sample the gripper mid-close and the TCP mid-contact,
    which makes GRIPPER_OPEN drop out of Pre and turns TCP_NEAR into a spurious
    add effect. Returns (rollout, post_forward_obs, regions, cube_init, dx).
    """
    obs, _ = env.reset(seed=seed)
    obs = pc._step_in_place(env, obs, 2, gripper_cmd=1.0)  # clear stale grasp flag

    cube_init = _cube(obs)
    dx = _push_dx(obs, cube_init, fallback_dx, use_env_goal)
    regions = _regions_for(cube_init, dx)
    scene_start = demo._obs_to_scene(obs, regions)

    behind = cube_init.copy()
    behind[0] -= 0.06
    behind[2] = cube_init[2] + 0.005
    obs = pc._step_toward_scaled(env, obs, behind, 30, 0.012, gripper_cmd=-1.0, scale=1.0)

    push_tgt = cube_init.copy()
    push_tgt[0] += dx
    push_tgt[2] += 0.005
    obs = pc._step_toward_scaled(env, obs, push_tgt, 30, 0.005, gripper_cmd=-1.0,
                                 scale=pushcube_full._FORWARD_PUSH_SCALE)

    # Lift clear + open: still part of the forward skill's execution, and the
    # world state the inverse actually starts from.
    up = _tcp(obs)
    up[2] += 0.10
    obs = pc._step_toward_scaled(env, obs, up, 20, 0.012, gripper_cmd=-1.0, scale=1.0)
    obs = pc._step_in_place(env, obs, 8, gripper_cmd=1.0)
    scene_end = demo._obs_to_scene(obs, regions)

    rollout = ForwardRollout(
        skill_name="push", demo_id=f"demo_{seed}", scenes=[scene_start, scene_end]
    )
    return rollout, obs, regions, cube_init, dx


# --- primitive executor (step 3 output -> real robot) ------------------------


def _execute_pick(env, obs, _regions):
    cube_now = _cube(obs)
    above = cube_now.copy()
    above[2] += 0.10
    obs = pc._step_toward_scaled(env, obs, above, 20, 0.012, gripper_cmd=1.0, scale=1.0)
    obs = pc._step_toward_scaled(env, obs, cube_now.copy(), 20, 0.012, gripper_cmd=1.0, scale=1.0)
    obs = pc._step_in_place(env, obs, 10, gripper_cmd=-1.0)
    obs = pc._step_in_place(env, obs, 5, gripper_cmd=-1.0)
    return obs


def _execute_place(env, obs, regions, region_name: str, perturbation: np.ndarray):
    target = np.asarray(regions[region_name].center, dtype=np.float32)
    cube_now = _cube(obs)
    lift = cube_now.copy()
    lift[2] += 0.15
    obs = pc._step_toward_scaled(env, obs, lift, 25, 0.02, gripper_cmd=-1.0, scale=0.5)
    over = np.array([target[0], target[1], lift[2]], dtype=np.float32)
    obs = pc._step_toward_scaled(env, obs, over, 80, 0.012, gripper_cmd=-1.0, scale=0.5)
    drop = target.copy()
    drop[0] += float(perturbation[0])
    drop[1] += float(perturbation[1])
    drop[2] += 0.005
    obs = pc._step_toward_scaled(env, obs, drop, 30, 0.012, gripper_cmd=-1.0, scale=0.5)
    obs = pc._step_in_place(env, obs, 10, gripper_cmd=1.0)
    obs = pc._step_in_place(env, obs, 10, gripper_cmd=1.0)
    return obs


def _execute_plan(env, obs, regions, actions: list[str], perturbation: np.ndarray):
    """Run the BFS-planned primitive sequence on the real robot.

    Returns (obs, commanded_drop) where `commanded_drop` is the xyz the last
    place primitive was told to release at — the reference the handoff validity
    gate measures against.
    """
    commanded_drop = None
    for action in actions:
        if action == "pick(cube)":
            obs = _execute_pick(env, obs, regions)
        elif action.startswith("place("):
            region_name = action[len("place("):-1]
            obs = _execute_place(env, obs, regions, region_name, perturbation)
            commanded_drop = np.asarray(regions[region_name].center, dtype=np.float32).copy()
            commanded_drop[0] += float(perturbation[0])
            commanded_drop[1] += float(perturbation[1])
        elif action == "noop":
            pass
        else:
            raise NotImplementedError(f"No executor bound for planned primitive {action!r}")
    return obs, commanded_drop


# --- report helpers ----------------------------------------------------------


def _fmt_terms(objective, scene) -> str:
    rows = []
    for term in objective.terms:
        raw = objective.predicates.get(term.key).evaluate(scene)
        rows.append(
            f"    {'NOT ' if term.polarity == 'negative' else '':<4}{term.key:<26}"
            f" margin={raw.margin * 1000:8.1f} mm"
            f"   V_p={objective.term_value(term, scene):.4f}"
            f"   weighted={objective.term_score(term, scene):.4f}"
        )
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo-seeds", type=lambda s: [int(x) for x in s.split(",")],
                    default=[0, 1, 2])
    ap.add_argument("--plan-seed", type=int, default=1000)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-attempts", type=int, default=pushcube_full._MAX_SCRIPTED_ATTEMPTS,
                    help="handoff validity gate: retries with a fresh seed")
    ap.add_argument("--handoff-tolerance-m", type=float,
                    default=pushcube_full._SYMBOLIC_HANDOFF_TOLERANCE_M,
                    help="max |cube - commanded drop point| for a valid handoff")
    ap.add_argument("--perturbation-range-m", type=float,
                    default=pushcube_full._PERTURBATION_RANGE_M)
    ap.add_argument("--no-env-goal-push", action="store_true")
    ap.add_argument("--push-displacement-m", type=float,
                    default=pushcube_full._PUSH_DISPLACEMENT_M)
    ap.add_argument("--out", type=Path, default=Path("artifacts/planrob_symbolic_pipeline.json"))
    args = ap.parse_args()

    use_env_goal = not args.no_env_goal_push
    env = gym.make("PushCube-v1", obs_mode="state_dict",
                   control_mode="pd_ee_delta_pos", max_episode_steps=600)
    registry = _registry()

    # ---- Step 1: extract the forward operator from demonstrations ----------
    print("=" * 74)
    print("STEP 1  Operator extraction from forward-push demonstrations")
    print("=" * 74)
    rollouts = []
    for seed in args.demo_seeds:
        rollout, _, _, cube_init, dx = _forward_demo(
            env, seed, args.push_displacement_m, use_env_goal)
        rollouts.append(rollout)
        print(f"  demo seed={seed:<5} cube_init=({cube_init[0]:+.3f},{cube_init[1]:+.3f})"
              f"  push_dx={dx * 100:.1f} cm")

    result = OperatorExtractor(registry).extract("push", rollouts)
    operator = result.operator
    print("\n  per-predicate scores (mean over demos):")
    print(f"    {'predicate':<26} {'start':>8} {'end':>8} {'delta':>8}")
    for key, s in sorted(result.scores.items()):
        print(f"    {key:<26} {s['start_mean']:8.4f} {s['end_mean']:8.4f} {s['delta']:+8.4f}")
    print("\n  extracted operator PUSH:")
    print(f"    Pre: {sorted(t.key for t in operator.preconditions)}")
    print(f"    Add: {sorted(t.key for t in operator.add_effects)}")
    print(f"    Del: {sorted(t.key for t in operator.delete_effects)}")

    # ---- Step 2: inverse target -------------------------------------------
    print("\n" + "=" * 74)
    print("STEP 2  Inverse target  I(o) = Pre u Del u not-Add")
    print("=" * 74)
    objective = RestorationObjective(operator, registry)
    for term in objective.terms:
        print(f"    [{'-' if term.polarity == 'negative' else '+'}] {term.key}")

    # ---- Steps 3-4: plan, execute, partition (with handoff validity gate) --
    print("\n" + "=" * 74)
    print("STEP 3  BFS inverse planning   +   STEP 4  execute -> partition")
    print("=" * 74)
    primitives = PrimitiveLibrary(object_name="cube", source_name="src", target_name="goal")
    planner = ToyInversePlanner(primitives, objective)
    rng = np.random.default_rng(args.plan_seed)

    accepted = None
    attempt_log: list[dict] = []
    for attempt in range(args.max_attempts):
        try_seed = args.plan_seed if attempt == 0 else int(rng.integers(0, 2**31 - 1))
        _, obs, regions, cube_init, dx = _forward_demo(
            env, try_seed, args.push_displacement_m, use_env_goal)

        # Gate 1: did the forward push actually reach the goal?
        fwd_target = np.array([cube_init[0] + dx, cube_init[1]], dtype=np.float32)
        fwd_err = float(np.linalg.norm(_cube(obs)[:2] - fwd_target))
        fwd_tol = max(pushcube_full._FORWARD_PUSH_TOLERANCE_MIN_M,
                      pushcube_full._FORWARD_PUSH_TOLERANCE_FRAC * abs(dx))
        if fwd_err >= fwd_tol:
            attempt_log.append({"seed": try_seed, "reject": "forward_push",
                                "forward_err_mm": fwd_err * 1000.0})
            print(f"  attempt {attempt + 1:>2} seed={try_seed:<11} REJECT forward push "
                  f"({fwd_err * 1000:.1f} mm >= {fwd_tol * 1000:.1f} mm)")
            continue

        post_forward_scene = demo._obs_to_scene(obs, regions)
        plan = planner.plan(post_forward_scene, max_depth=args.max_depth)
        if not plan.actions:
            attempt_log.append({"seed": try_seed, "reject": "empty_plan"})
            print(f"  attempt {attempt + 1:>2} seed={try_seed:<11} REJECT empty plan")
            continue

        perturbation = (rng.uniform(-args.perturbation_range_m, args.perturbation_range_m, size=2)
                        if args.perturbation_range_m > 0 else np.zeros(2))
        obs, commanded_drop = _execute_plan(env, obs, regions, plan.actions, perturbation)
        handoff_scene = demo._obs_to_scene(obs, regions)

        # Gate 2: did the executed prefix release the cube where it was told?
        sym_err = (float(np.linalg.norm(_cube(obs) - commanded_drop))
                   if commanded_drop is not None else float("nan"))
        if not (sym_err < args.handoff_tolerance_m):
            attempt_log.append({"seed": try_seed, "reject": "symbolic_execution",
                                "symbolic_err_mm": sym_err * 1000.0})
            print(f"  attempt {attempt + 1:>2} seed={try_seed:<11} REJECT prefix execution "
                  f"({sym_err * 1000:.1f} mm >= {args.handoff_tolerance_m * 1000:.1f} mm)")
            continue

        attempt_log.append({"seed": try_seed, "reject": None,
                            "forward_err_mm": fwd_err * 1000.0,
                            "symbolic_err_mm": sym_err * 1000.0})
        print(f"  attempt {attempt + 1:>2} seed={try_seed:<11} ACCEPT "
              f"(forward {fwd_err * 1000:.1f} mm, prefix {sym_err * 1000:.1f} mm)")
        accepted = {
            "seed": try_seed, "plan": plan, "regions": regions,
            "post_forward_scene": post_forward_scene, "handoff_scene": handoff_scene,
            "perturbation": perturbation, "cube_final": _cube(obs),
            "forward_err_m": fwd_err, "symbolic_err_m": sym_err,
        }
        break

    if accepted is None:
        raise RuntimeError(
            f"No valid handoff after {args.max_attempts} attempts; last: {attempt_log[-1]}")

    plan = accepted["plan"]
    regions = accepted["regions"]
    handoff_scene = accepted["handoff_scene"]
    perturbation = accepted["perturbation"]

    print(f"\n  accepted seed={accepted['seed']}  post-forward scene:")
    print(_fmt_terms(objective, accepted["post_forward_scene"]))
    print(f"\n  BFS: success={plan.success}  expanded={plan.expanded_nodes}"
          f"  V_initial={plan.initial_potential:.4f} -> V_model={plan.final_potential:.4f}")
    print(f"  planned prefix: {' -> '.join(plan.actions)}")
    print("\n  term_max_scores across the search (model prediction, V_p):")
    for key, v in sorted(plan.term_max_scores.items()):
        print(f"    {key:<26} {v:.4f}")

    sc1 = plan.actions == ["pick(cube)", "place(src)"]
    print(f"\n  SC1  prefix == pick(cube) -> place(src):  {'PASS' if sc1 else 'FAIL'}")

    err_mm = float(np.linalg.norm(
        accepted["cube_final"] - np.asarray(regions["src"].center))) * 1000.0
    print(f"\n  handoff perturbation = ({perturbation[0] * 1000:+.1f}, "
          f"{perturbation[1] * 1000:+.1f}) mm")
    print(f"  cube -> src center   = {err_mm:.1f} mm")
    print(f"\n  handoff scene term scores  (theta={_FENCE_THRESHOLD}):")
    print(_fmt_terms(objective, handoff_scene))

    fences, active = [], []
    for term in objective.terms:
        label = ("NOT " if term.polarity == "negative" else "") + term.key
        (fences if objective.term_value(term, handoff_scene) >= _FENCE_THRESHOLD
         else active).append(label)

    print(f"\n  V(handoff) = {objective.potential(handoff_scene):.4f}")
    print(f"  Fence  F = {sorted(fences)}")
    print(f"  Active A = {sorted(active)}")

    sc2 = sorted(active) == ["in_region(cube,src)"] and len(fences) == len(objective.terms) - 1
    print(f"\n  SC2  A == {{in_region(cube,src)}}, every other term fenced:"
          f"  {'PASS' if sc2 else 'FAIL'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "demo_seeds": args.demo_seeds,
        "plan_seed": args.plan_seed,
        "accepted_seed": accepted["seed"],
        "attempt_log": attempt_log,
        "operator": {
            "pre": sorted(t.key for t in operator.preconditions),
            "add": sorted(t.key for t in operator.add_effects),
            "del": sorted(t.key for t in operator.delete_effects),
        },
        "extraction_scores": result.scores,
        "inverse_target": [
            {"key": t.key, "polarity": t.polarity} for t in objective.terms
        ],
        "plan": {
            "success": plan.success,
            "actions": plan.actions,
            "expanded_nodes": plan.expanded_nodes,
            "v_initial": plan.initial_potential,
            "v_model": plan.final_potential,
            "term_max_scores": plan.term_max_scores,
        },
        "handoff": {
            "perturbation_xy_m": perturbation.tolist(),
            "cube_to_src_center_mm": err_mm,
            "forward_err_m": accepted["forward_err_m"],
            "symbolic_err_m": accepted["symbolic_err_m"],
            "v_handoff": objective.potential(handoff_scene),
            "fences": sorted(fences),
            "active": sorted(active),
        },
        "sc1_prefix_matches": bool(sc1),
        "sc2_partition_matches": bool(sc2),
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out}")
    env.close()


if __name__ == "__main__":
    main()
