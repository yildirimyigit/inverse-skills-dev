"""Direct baseline, step 2: the checks before any training.

On every training (0-31) and validation (500-509) start, never on the seeds
kept for evaluation:

  1. same starts      DrawerDirectEnv and the grasp hole's DrawerHoleEnv reset to
                      the same physical state (TCP, gripper, drawer), offsets
                      included, and the start condition holds.
  2. horizon          a temporary reference controller, acting only through the
                      same 4-D actions and seeing only the policy's observation,
                      descends onto the fin, closes, and pulls straight back. It
                      must open the drawer within the episode on every start. It
                      exists only for this check: never training data, never in RL.
  3. measurement      at every step of those episodes: the opening the
                      environment reports equals the drawer joint's position;
                      drawer_open's score is 0.5 (1 + tanh((opening - 8 cm) / 2 cm));
                      the reward is 2 V(drawer_open) - 1; goal_met is
                      drawer_open and not drawer_closed.
  4. nothing scripted every episode step is exactly one simulator step, and no
                      scripted motion runs during an episode (they are made to
                      raise for the duration).

Also measured, not gated: what uniformly random actions, which SAC explores
with first, reach on the training starts: grasps, how far the drawer opens, and
the best drawer_open score, the only reward signal.

Writes artifacts/drawer/direct/precheck.json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from inverse_skills.envs import drawer_direct_env as direct_env
from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import drawer_env as geo
from inverse_skills.envs import drawer_hole_env as hole_env
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.envs import motion
from inverse_skills.envs.drawer_direct_env import DrawerDirectEnv
from inverse_skills.envs.drawer_hole_env import SATISFIED, DrawerHoleEnv, load_spec

TRAIN_SEEDS = tuple(range(32))
VALIDATION_SEEDS = tuple(range(500, 510))
_TOL = 1e-5
# Scripted motions that must never run while a policy is in control.
_GUARDED = [(prim, "push_close"), (prim, "approach_handle"), (prim, "test_grasp"), (prim, "pull_open"),
            (prim, "go_home"), (motion, "step_toward"), (motion, "step_in_place"),
            (hole_env, "step_toward"), (hole_env, "step_in_place"), (direct_env, "step_in_place")]


class SimSteps(gym.Wrapper):
    """Counts simulator steps."""

    def __init__(self, env):
        super().__init__(env)
        self.count = 0

    def step(self, action):
        self.count += 1
        return self.env.step(action)


@contextlib.contextmanager
def nothing_scripted():
    def forbidden(*_args, **_kwargs):
        raise RuntimeError("a scripted motion ran inside an episode")
    saved = [(module, name, getattr(module, name)) for module, name in _GUARDED]
    saved_prefix = dict(hole_env._PRIMITIVES)
    for module, name in _GUARDED:
        setattr(module, name, forbidden)
    hole_env._PRIMITIVES.update({k: forbidden for k in hole_env._PRIMITIVES})
    try:
        yield
    finally:
        for module, name, original in saved:
            setattr(module, name, original)
        hole_env._PRIMITIVES.update(saved_prefix)


def reference_policy():
    """Descend onto the fin and close (Stage 3's hand-coded reference), then pull straight
    back with the gripper closed until the drawer is 15 cm open, then hold. Sees only the
    policy's 9 observation numbers."""
    pulling = False

    def act(obs):
        nonlocal pulling
        pulling = pulling or 0.5 * (1.0 + np.tanh(obs[0])) >= SATISFIED   # handle_grasped holds
        if not pulling:
            height_cm = obs[3] * 5.0
            action = [-2.0 * obs[1], -2.0 * obs[2], -height_cm, 1.0 if height_cm > 0.3 else -1.0]
        elif obs[8] * geo.MAX_OPENING < 0.15:
            action = [-1.0, 0.0, 0.0, -1.0]                                 # toward the robot: opening
        else:
            action = [0.0, 0.0, 0.0, -1.0]
        return np.clip(action, -1.0, 1.0)
    return act


def physical_state(env) -> np.ndarray:
    obs = env._obs
    return np.concatenate([prim.tcp_pos(obs), [prim.gripper_width(obs), prim.opening(obs)], prim.grasp_point(obs)])


def episode(env: DrawerDirectEnv, sim: SimSteps, seed: int, policy, measure: list | None = None) -> dict:
    obs, info = env.reset(seed=seed)
    before = sim.count
    with nothing_scripted():
        for _ in range(env.max_steps):
            obs, reward, _terminated, truncated, step_info = env.step(policy(obs))
            if measure is not None:
                scores = step_info["scores"]
                opening = prim.opening(env._obs)
                joint = float(env._env.unwrapped.cabinet.get_qpos()[0, 0])
                formula = 0.5 * (1.0 + np.tanh((opening - 0.08) / 0.02))
                measure.append({
                    "opening_is_joint": bool(abs(opening - joint) < _TOL),
                    "drawer_open_formula": bool(abs(scores[env.achieve] - formula) < _TOL),
                    "reward": bool(abs(reward - (2.0 * scores[env.achieve] - 1.0)) < _TOL),
                    "goal_met": step_info["goal_met"] == (scores["drawer_open(drawer)"] >= SATISFIED
                                                          and scores["drawer_closed(drawer)"] < SATISFIED),
                })
            if truncated:
                break
    record = dict(env.episodes[-1])
    record.update(seed=seed, handoff_valid=info["handoff_valid"], sim_steps=sim.count - before)
    record.update(env.settle())
    return record


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--operator", type=Path, default=Path("artifacts/drawer/stage1_operator.json"))
    ap.add_argument("--grasp-spec", type=Path, default=Path("artifacts/drawer/hole_spec.json"))
    ap.add_argument("--spec", type=Path, default=Path("artifacts/drawer/direct/hole_spec_direct.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/drawer/direct/precheck.json"))
    args = ap.parse_args()

    goal = dd.goal_from_operator(args.operator)
    env = DrawerDirectEnv(load_spec(args.spec), goal)
    sim = SimSteps(env._env)
    env._env = sim
    hole = DrawerHoleEnv(load_spec(args.grasp_spec))
    starts = TRAIN_SEEDS + VALIDATION_SEEDS
    print(f"episode length: {env.max_steps} steps")

    # 1. same starts
    same = []
    for seed in starts:
        _, a = hole.reset(seed=seed)
        _, b = env.reset(seed=seed)
        diff = float(np.max(np.abs(physical_state(hole) - physical_state(env))))
        same.append({"seed": seed, "max_difference": diff, "offset_mm": b["start_offset_mm"],
                     "valid": a["handoff_valid"] and b["handoff_valid"]})
    hole.close()
    same_ok = all(s["max_difference"] == 0.0 and s["valid"] for s in same)
    print(f"1. same starts as the grasp hole: {sum(s['max_difference'] == 0.0 for s in same)}/{len(same)} identical, "
          f"start condition holds on {sum(s['valid'] for s in same)}/{len(same)}")

    # 2-4. the reference controller, measured and guarded
    measure: list[dict] = []
    reference = [episode(env, sim, seed, reference_policy(), measure) for seed in starts]
    opened = [r for r in reference if r["goal_met"]]
    first = sorted(r["first_satisfied_step"] for r in opened)
    horizon_ok = len(opened) == len(reference) and all(r["settled_goal_met"] for r in reference)
    print(f"2. reference controller: goal met at the end on {len(opened)}/{len(reference)} starts, still met after "
          f"settling on {sum(r['settled_goal_met'] for r in reference)}; grasp at step "
          f"{min(r['first_grasp_step'] for r in reference)}-{max(r['first_grasp_step'] for r in reference)}, "
          f"drawer open (score >= 0.8) at step {first[0]}-{first[-1]} of {env.max_steps}; "
          f"final opening {min(r['end_opening_mm'] for r in reference)}-{max(r['end_opening_mm'] for r in reference)} mm")
    checks = {k: sum(m[k] for m in measure) for k in measure[0]}
    measure_ok = all(v == len(measure) for v in checks.values())
    print(f"3. measurement, over {len(measure)} steps: " + ", ".join(f"{k} {v}/{len(measure)}" for k, v in checks.items()))

    # random actions, also under the guard
    rng = np.random.default_rng(0)
    random_runs = [episode(env, sim, seed, lambda _obs: rng.uniform(-1.0, 1.0, size=4)) for seed in TRAIN_SEEDS]
    every = reference + random_runs
    scripted_ok = all(r["sim_steps"] == env.max_steps for r in every)
    print(f"4. nothing scripted: {len(every)} guarded episodes ran, each exactly {env.max_steps} simulator steps: "
          f"{scripted_ok}")
    print(f"random actions on the training starts: grasped in {sum(r['ever_grasped'] for r in random_runs)}/"
          f"{len(random_runs)} episodes, drawer opened at most {max(r['max_opening_mm'] for r in random_runs)} mm, "
          f"best drawer_open score {max(r['best_score'] for r in random_runs):.4f}, "
          f"goal met {sum(r['goal_met'] for r in random_runs)}/{len(random_runs)}")
    env.close()

    passed = same_ok and horizon_ok and measure_ok and scripted_ok
    print("\nPRECHECK:", "PASS" if passed else "FAIL")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "passed": passed, "episode_length": env.max_steps,
        "checks": {"same_starts": same_ok, "horizon": horizon_ok, "measurement": measure_ok,
                   "nothing_scripted": scripted_ok},
        "measurement_counts": checks, "measured_steps": len(measure),
        "same_starts": same, "reference": reference, "random_actions": random_runs,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
