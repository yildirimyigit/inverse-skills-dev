"""Stage 0 for the signed goal: is the existing retreat motion a reliable library action?

The StowCube library is to gain a known action, retreat_to_safe, whose desired effects are
not tcp_near(cube_a) and not tcp_near(cube_b), and nothing else. Before it is registered, this
gate measures what the existing `retreat()` primitive really does when run as that action would
run it (`stow_executive.retreat_to_safe`: the gripper kept as it is).

The retreat starts from states the existing inverse execution really reaches: the forward stow,
then the first k steps of the historical plan, run by the executive with the historical learned
drag policy, as the published Stage 4 ran them:

  k = 0  after the forward skill
  k = 2  after place(cube_b,src_b)
  k = 3  after approach(cube_a)              the hand seated on A, gripper closed
  k = 4  after the learned drag              the hand on A, out of the pocket, gripper closed
  k = 6  after place(cube_a,src_a)           the end of the historical plan, gripper open

For each run it records, before and after the retreat, every predicate's score, both cubes'
positions, the TCP and the gripper width. During the rise it records, step by step, the largest
contact force between any robot link and each cube and pocket wall.

A run passes when all of these hold, judged robustly (p means a score of at least 0.8, not p a
score of at most 0.2):
  - both tcp_near literals are robustly false after the retreat;
  - every other predicate, the gripper included, keeps its robust value: robustly true stays
    robustly true, robustly false stays robustly false;
  - neither cube moves more than 1 mm;
  - no robot link touches a cube or a wall after the first step of the rise. A press the state
    started with (after approach or the drag) is released during the first step.

A state whose prefix fails (the executive refuses or cannot complete a step) is recorded as not
reached; that is the prefix's failure, not the retreat's.

The first run of this gate (commit 7279aad, before the action was registered) measured the motion
from every state above. From the closed-gripper states, releasing the press displaced cube A, so
the registered action (`stow_domain.retreat_to_safe`) needs the gripper open. Every state is
still measured and reported; the gate's verdict covers the runs whose start satisfies the
registered preconditions, robustly for scored literals (`applicable`).

    PYTHONNOUSERSITE=1 python scripts/stow_retreat_gate.py --out <new path>.json

Nothing here changes the planning domain. The output is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from inverse_skills.envs import stow_domain as sd
from inverse_skills.envs import stow_executive as ex
from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs import stow_primitives as prim
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.hole_planner import Action

torch.set_num_threads(1)

STATES = {0: "after the forward skill", 2: "after place(cube_b,src_b)", 3: "after approach(cube_a)",
          4: "after the learned drag", 6: "after place(cube_a,src_a)"}
SEEDS = tuple(range(1000, 1010)) + tuple(range(2000, 2010))
ROBUST_TRUE, ROBUST_FALSE = 0.8, 0.2
MAX_DISPLACEMENT_MM = 1.0
CONTACT_N = 1e-3          # a force above this counts as touching
TCP_NEAR = ("tcp_near(cube_a)", "tcp_near(cube_b)")
_MAX_FORWARD_ATTEMPTS = 5


def robust_value(score: float) -> str:
    return "true" if score >= ROBUST_TRUE else "false" if score <= ROBUST_FALSE else "uncertain"


class ContactRecorder:
    """Passes every step to the environment, then records the largest contact force between any
    robot link and each named body. The motion under test is not changed."""

    def __init__(self, env, bodies: dict):
        self.env = env
        self.unwrapped = env.unwrapped
        self.links = env.unwrapped.agent.robot.get_links()
        self.bodies = bodies
        self.forces: list[dict[str, float]] = []

    def step(self, action):
        result = self.env.step(action)
        scene = self.env.unwrapped.scene
        row = {}
        for name, body in self.bodies.items():
            row[name] = max(float(torch.linalg.norm(scene.get_pairwise_contact_forces(link, body), dim=-1).max())
                            for link in self.links)
        self.forces.append(row)
        return result


def measure(obs, regions, registry) -> dict:
    scene = prim.obs_to_scene(obs, regions)
    qpos = obs["agent"]["qpos"].squeeze().cpu().numpy()
    return {"scores": {key: float(result.score) for key, result in registry.evaluate_all(scene).items()},
            "holding": scene.robot.holding,
            "cube_a": prim.cube_pos(obs, "cube_a").tolist(), "cube_b": prim.cube_pos(obs, "cube_b").tolist(),
            "tcp": prim.tcp_pos(obs).tolist(), "gripper_width_mm": float((qpos[-2] + qpos[-1]) * 1000)}


def applicable(before: dict, action) -> bool:
    """Does the start satisfy the registered action's preconditions? Scored literals robustly; the
    unscored holding(cube) fluents from the scene's record of what the gripper holds."""
    scores = before["scores"]
    for atom in action.pre:
        if atom in scores and scores[atom] < ROBUST_TRUE:
            return False
    for atom in action.pre_absent:
        if atom in scores and scores[atom] > ROBUST_FALSE:
            return False
        if atom.startswith("holding(") and before["holding"] == atom[len("holding("):-1]:
            return False
    return True


def stowed(env, seed: int, expected: frozenset[str], registry):
    """The forward skill, re-rolled until it leaves the state the plan starts from (as Stage 4)."""
    for attempt in range(_MAX_FORWARD_ATTEMPTS):
        obs = prim.reset(env, seed + attempt * 10_000)
        regions = prim.regions(obs)
        obs = prim.forward_stow(env, obs)
        scores = {k: float(r.score) for k, r in registry.evaluate_all(prim.obs_to_scene(obs, regions)).items()}
        if hp.symbolic_state(scores) == expected:
            return obs, regions, attempt
    raise RuntimeError(f"seed {seed}: the forward skill never produced the planned start")


def judge(before: dict, after: dict, forces: list[dict]) -> dict:
    checks = {f"{key} robustly false after": after["scores"][key] <= ROBUST_FALSE for key in TCP_NEAR}
    changed = [key for key in before["scores"] if key not in TCP_NEAR
               and robust_value(before["scores"][key]) != robust_value(after["scores"][key])]
    checks["every other predicate keeps its robust value"] = not changed
    displacement = {cube: float(np.linalg.norm(np.subtract(after[cube], before[cube])) * 1000)
                    for cube in ("cube_a", "cube_b")}
    checks[f"no cube moves more than {MAX_DISPLACEMENT_MM} mm"] = max(displacement.values()) <= MAX_DISPLACEMENT_MM
    later = {name: max((row[name] for row in forces[1:]), default=0.0) for name in (forces[0] if forces else {})}
    touched = sorted(name for name, force in later.items() if force > CONTACT_N)
    checks["no contact after the first step"] = not touched
    return {"passed": all(checks.values()), "checks": checks, "predicates_changed": changed,
            "displacement_mm": displacement, "touched_after_first_step": touched,
            "first_step_contact_n": forces[0] if forces else {}, "largest_later_contact_n": later,
            "rise_steps": len(forces)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--spec", type=Path, default=Path("artifacts/stow/hole_spec.json"))
    ap.add_argument("--checkpoint", type=Path, default=Path("artifacts/stow/stage3/1790788534/sac_hole_best"))
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; results are never overwritten")

    spec = json.loads(args.spec.read_text())
    target = hp.spec_target(spec)          # the saved spec predates signs: its target is positive
    hole = Action(name=spec["hole"], pre=frozenset(spec["start"]), add=frozenset({target.atom}),
                  hole_for=target)
    steps = [Action.from_dict(a) for a in spec["prefix"]] + [hole] + [Action.from_dict(a) for a in spec["suffix"]]
    model = SAC.load(args.checkpoint, device="cuda")
    registry = sp.registry()
    registered = sd.retreat_to_safe()
    env = prim.make_env()
    unwrapped = env.unwrapped
    bodies = {"cube_a": unwrapped.cube_a, "cube_b": unwrapped.cube_b, "back_wall": unwrapped.back_wall,
              **{arm.name: arm for arm in unwrapped.arms}}
    print("historical plan:", " > ".join(a.name for a in steps))

    runs = []
    for k, label in STATES.items():
        print(f"\n{label} (first {k} steps)")
        for seed in SEEDS:
            obs, regions, rerolls = stowed(env, seed, sd.stowed_state(), registry)
            obs, report = ex.execute_plan(env, obs, steps[:k], model=model, regions=regions)
            row = {"state": label, "steps_before": k, "seed": seed, "forward_rerolls": rerolls}
            if len(report.steps) != k or not all(s.ok for s in report.steps):
                failed = next((s for s in report.steps if not s.ok), None)
                row.update(reached=False, reason=f"{failed.action}: {failed.detail}" if failed else "prefix stopped")
                runs.append(row)
                print(f"  seed {seed}: state not reached ({row['reason']})")
                continue
            before = measure(obs, regions, registry)
            recorder = ContactRecorder(env, bodies)
            obs = ex.retreat_to_safe(recorder, obs, regions, registry)
            after = measure(obs, regions, registry)
            verdict = judge(before, after, recorder.forces)
            row.update(reached=True, applicable=applicable(before, registered), before=before, after=after,
                       contact_by_step=recorder.forces, **verdict)
            runs.append(row)
            print(f"  seed {seed}: {'pass' if verdict['passed'] else 'FAIL'}"
                  f"{'' if row['applicable'] else ' (outside the preconditions)'}  "
                  f"tcp_near A {before['scores']['tcp_near(cube_a)']:.3f}->{after['scores']['tcp_near(cube_a)']:.3f}  "
                  f"B {before['scores']['tcp_near(cube_b)']:.3f}->{after['scores']['tcp_near(cube_b)']:.3f}  "
                  f"gripper {before['scores']['gripper_open()']:.3f}->{after['scores']['gripper_open()']:.3f}  "
                  f"moved A {verdict['displacement_mm']['cube_a']:.2f} B {verdict['displacement_mm']['cube_b']:.2f} mm  "
                  f"rise {verdict['rise_steps']} steps"
                  + (f"  touched {verdict['touched_after_first_step']}" if verdict["touched_after_first_step"] else "")
                  + (f"  changed {verdict['predicates_changed']}" if verdict["predicates_changed"] else ""))
    env.close()

    summary = {}
    for k, label in STATES.items():
        rows = [r for r in runs if r["steps_before"] == k]
        reached = [r for r in rows if r["reached"]]
        summary[label] = {
            "runs": len(rows), "reached": len(reached), "passed": sum(r["passed"] for r in reached),
            "applicable": sum(r["applicable"] for r in reached),
            "applicable_passed": sum(r["passed"] for r in reached if r["applicable"]),
            "failed_checks": {check: sum(not r["checks"][check] for r in reached)
                              for check in (reached[0]["checks"] if reached else {})},
            "tcp_near_after_max": {key: max((r["after"]["scores"][key] for r in reached), default=None)
                                   for key in TCP_NEAR},
            "gripper_open_before_after": sorted({(robust_value(r["before"]["scores"]["gripper_open()"]),
                                                  robust_value(r["after"]["scores"]["gripper_open()"])) for r in reached}),
            "largest_displacement_mm": {cube: max((r["displacement_mm"][cube] for r in reached), default=None)
                                        for cube in ("cube_a", "cube_b")},
            "largest_later_contact_n": max((max(r["largest_later_contact_n"].values(), default=0.0) for r in reached),
                                           default=None),
            "tcp_z_after_mm": [min((r["after"]["tcp"][2] * 1000 for r in reached), default=None),
                               max((r["after"]["tcp"][2] * 1000 for r in reached), default=None)],
            "not_reached": [{"seed": r["seed"], "reason": r["reason"]} for r in rows if not r["reached"]]}
    applicable_runs = [r for r in runs if r["reached"] and r["applicable"]]
    passed = bool(applicable_runs) and all(r["passed"] for r in applicable_runs)

    print("\nsummary")
    for label, s in summary.items():
        print(f"  {label:28s} reached {s['reached']:2d}/{s['runs']}  passed {s['passed']:2d}  "
              f"applicable {s['applicable']:2d} (passed {s['applicable_passed']:2d})  "
              f"tcp_near after <= {max(v for v in s['tcp_near_after_max'].values() if v is not None) if s['reached'] else '-'}  "
              f"moved <= {max(v for v in s['largest_displacement_mm'].values() if v is not None) if s['reached'] else '-'} mm")
    print("\nRETREAT GATE:", "PASS" if passed else "FAIL")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "tree_clean": not subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                         capture_output=True, text=True).stdout.strip(),
        "spec": str(args.spec), "checkpoint": str(args.checkpoint), "seeds": list(SEEDS),
        "criteria": {"robust_true": ROBUST_TRUE, "robust_false": ROBUST_FALSE,
                     "max_displacement_mm": MAX_DISPLACEMENT_MM, "contact_n": CONTACT_N},
        "registered_action": registered.to_dict(), "plan": [a.name for a in steps], "summary": summary,
        "passed": passed, "verdict_covers": "runs whose start satisfies the registered preconditions",
        "runs": runs}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
