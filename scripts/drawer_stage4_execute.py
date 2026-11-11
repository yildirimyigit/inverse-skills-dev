"""Stage 4 for the drawer: the whole INVERSE pipeline, executed.

For each start configuration (seeds 2000-2019, never used in training,
checkpoint selection or earlier evaluations):

  1. reset the scene and run the demonstrated forward skill, push_close;
  2. measure the predicate scores, read them as a Boolean symbolic state (true at
     a score of at least 0.5), and check that the forward operator's extracted
     signed effects robustly hold (at least 0.8 for p, at most 0.2 for not p).
     An inverse is only defined when the forward happened: if they do not, that
     is reported and nothing more is run;
  3. take the signed restoration goal of the Stage 1 operator, G = Pre;
  4. plan from the measured state with the existing planner and library;
  5. execute the plan with the drawer executive. A hole is filled by the
     learned policy trained for the hole spec, stopped as soon as the literal
     it owes robustly holds. The goal is judged robustly, every literal of
     both signs; the Boolean verdict is recorded beside it.

The script never names the expected inverse plan: it executes whatever the
planner returns. Conditions, all on the same 20 configurations:

  learned, SAC seed k   each Stage 3 run's selected checkpoint, unchanged
                        (its SHA-256 is checked against stage3/checkpoints.json)
  library_only          baseline: with no holes allowed there is no plan; the
                        planned steps without the hole are then executed, to
                        show where execution stops
  scripted_grasp        baseline: Stage 0's scripted grasp added to the library
                        as a known action. Only this baseline is given its
                        controller.

Writes artifacts/drawer/stage4_execution.json and prints the comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import drawer_executive as ex
from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.envs.drawer_hole_env import load_spec
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.hole_planner import spec_target
from inverse_skills.operators.literals import join, robustly_unmet

torch.set_num_threads(1)

SEEDS = tuple(range(2000, 2020))
FORWARD = "forward: push_close"
LIBRARY = {"approach_handle": prim.approach_handle, "pull_open": prim.pull_open}
LIBRARY_WITH_SCRIPTED_GRASP = {**LIBRARY, "grasp_handle": prim.test_grasp}


class StepCounter(gym.Wrapper):
    """Counts control steps under the name of the step being run (`current`)."""

    def __init__(self, env):
        super().__init__(env)
        self.current = ""
        self.counts: dict[str, int] = {}

    def step(self, action):
        self.counts[self.current] = self.counts.get(self.current, 0) + 1
        return self.env.step(action)


def load_checkpoints(stage3_dir: Path) -> dict[str, object]:
    """Each Stage 3 run's selected checkpoint, after checking it is the recorded file."""
    manifest = json.loads((stage3_dir / "checkpoints.json").read_text())
    policies = {}
    for run, entry in sorted(manifest["runs"].items(), key=lambda item: item[1]["sac_seed"]):
        path = stage3_dir / run / entry["selected"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["files"][entry["selected"]]["sha256"]:
            raise SystemExit(f"{path} is not the recorded checkpoint")
        policies[f"learned, SAC seed {entry['sac_seed']}"] = SAC.load(path.with_suffix(""), device="cpu")
    return policies


def forward_effects(operator_path: Path):
    """The extracted forward operator's signed effects."""
    operator = json.loads(operator_path.read_text())["operator"]
    return join((t["key"] for t in operator["add_effects"]), (t["key"] for t in operator["delete_effects"]))


def episode(env, seed: int, forward, goal, condition: str, policy, spec) -> dict:
    registry = dp.registry()
    obs = prim.reset(env, seed)
    env.counts, env.current = {}, FORWARD
    row = {"seed": seed, "start_opening_mm": round(prim.opening(obs) * 1000, 1)}

    # The demonstrated forward skill, then what the robot now measures.
    obs = prim.push_close(env, obs)
    measured_scores = ex.scores(obs, registry)
    state = hp.symbolic_state(measured_scores)
    row["after_forward_opening_mm"] = round(prim.opening(obs) * 1000, 1)
    row["measured_state"] = sorted(state)
    row["forward_effects_unmet"] = [str(l) for l in robustly_unmet(forward, measured_scores)]
    row["forward_effects_measured"] = not row["forward_effects_unmet"]
    row["forward_control_steps"] = env.counts[FORWARD]
    # Until the inverse runs and reports otherwise:
    row.update(ok=False, goal_met=False, steps=[], inverse_control_steps=0,
               final_opening_mm=row["after_forward_opening_mm"])
    if not row["forward_effects_measured"]:
        row["stopped"] = "the forward skill's effects were not measured, so there is nothing to invert"
        return row

    # Plan from the measured state.
    learned, controllers, domain = {}, LIBRARY, dd.domain()
    if condition == "scripted_grasp":
        domain = dd.domain(extra_actions=[dd.scripted_grasp()])
        controllers = LIBRARY_WITH_SCRIPTED_GRASP
    elif condition != "library_only":
        learned = {spec_target(spec): policy}
    result = hp.plan(domain, state, *goal)
    row["plan"] = list(result.actions) if result else None
    if not result:
        row["stopped"] = "no plan"
        return row
    steps = list(result.steps)
    if condition == "library_only":
        no_holes = hp.plan(domain, state, *goal, max_delegated=0)
        row["plan_without_holes"] = list(no_holes.actions) if no_holes else None
        row["unmet_needs"] = [l.to_dict() for l in no_holes.unmet_needs]
        steps = [s for s in steps if s.hole_for is None]
    row["executed_steps"] = [s.name for s in steps]
    row["grasp_step"] = next((s.name for s in steps if spec_target(spec) in s.effect_literals), None)

    # Execute; every control step is counted under the plan step that took it.
    def count_as(name: str, attempt: int) -> None:
        env.current = name

    obs, report = ex.execute_plan(env, obs, steps, domain, goal, controllers, learned, on_action=count_as)
    row.update(report.to_dict())
    for step in row["steps"]:
        step["control_steps"] = env.counts.get(step["action"], 0)
    row["inverse_control_steps"] = sum(n for name, n in env.counts.items() if name != FORWARD)
    return row


def step_named(row: dict, name: str | None) -> dict | None:
    return next((s for s in row["steps"] if s["action"] == name), None)


def moved_before_pull(row: dict) -> float:
    """How far the drawer moved, from where the forward skill left it, before pull_open ran."""
    moved = 0.0
    for step in row["steps"]:
        if step["action"].startswith("pull_open"):
            break
        moved = max(moved, abs(step["opening_mm"] - row["after_forward_opening_mm"]))
    return round(moved, 1)


def summarize(rows: list[dict]) -> dict:
    grasps = [step_named(r, r.get("grasp_step")) for r in rows]
    grasps = [g for g in grasps if g is not None]
    pulls = [s for r in rows for s in r["steps"] if s["action"].startswith("pull_open")]
    failed = [f"{s['action']}: {s['detail']}" for r in rows for s in r["steps"] if not s["ok"]]
    failed += [r["stopped"] for r in rows if "stopped" in r]
    inverse_steps = sorted(r["inverse_control_steps"] for r in rows)
    return {
        "forward_effects_measured": sum(r["forward_effects_measured"] for r in rows),
        "restored": sum(r["ok"] for r in rows),
        "grasp_effect_achieved": sum(g["ok"] for g in grasps),
        "grasp_control_steps": sorted(g["control_steps"] for g in grasps),
        "pull_open_succeeded": sum(p["ok"] for p in pulls),
        "goal_met": sum(bool(r["goal_met"]) for r in rows),
        "goal_met_boolean": sum(bool(r.get("goal_met_boolean")) for r in rows),
        "goal_literals_ever_unmet": sorted({l for r in rows for l in r.get("goal_unmet", [])}),
        "failed_transitions": {f: failed.count(f) for f in sorted(set(failed))},
        "recovery_attempts": sum(max(s["attempts"] - 1, 0) for r in rows for s in r["steps"]),
        "model_vs_world_differences": sum(bool(s["divergence"]) for r in rows for s in r["steps"]),
        "max_drawer_moved_before_pull_mm": max(moved_before_pull(r) for r in rows),
        "max_tilt_after_grasp_deg": max((g["tilt_deg"] for g in grasps), default=None),
        "final_opening_mm": [min(r["final_opening_mm"] for r in rows), max(r["final_opening_mm"] for r in rows)],
        "inverse_control_steps": [inverse_steps[0], int(np.median(inverse_steps)), inverse_steps[-1]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=list(SEEDS))
    ap.add_argument("--operator", type=Path, default=Path("artifacts/drawer/stage1_operator.json"))
    ap.add_argument("--spec", type=Path, default=Path("artifacts/drawer/hole_spec.json"))
    ap.add_argument("--stage3", type=Path, default=Path("artifacts/drawer/stage3"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/drawer/stage4_execution.json"))
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; saved results are never overwritten. Choose another --out.")

    forward = forward_effects(args.operator)
    goal = dd.goal_from_operator(args.operator)
    spec = load_spec(args.spec)
    conditions = {**load_checkpoints(args.stage3), "library_only": None, "scripted_grasp": None}
    print("signed restoration goal of the Stage 1 operator:", ", ".join(str(l) for l in sorted(join(*goal))))
    print(f"hole spec: {spec['hole']}, starts from {spec['start']}")

    env = StepCounter(prim.make_env())
    results, summary = {}, {}
    for condition, policy in conditions.items():
        rows = [episode(env, seed, forward, goal, condition, policy, spec) for seed in args.seeds]
        results[condition], summary[condition] = rows, summarize(rows)
        print(f"\n{condition}: plan {rows[0].get('plan')}")
        for row in rows:   # only the episodes that were not clean
            failed = next((s for s in row["steps"] if not s["ok"]), None)
            retries = sum(max(s["attempts"] - 1, 0) for s in row["steps"])
            diverged = [f"{s['action']}: {', '.join(s['divergence'])}" for s in row["steps"] if s["divergence"]]
            if row["ok"] and not retries and not diverged:
                continue
            status = row.get("stopped") or (f"stopped at {failed['action']} ({failed['detail']})"
                                            if failed else "goal not met")
            print(f"  seed {row['seed']}: {status} | final {row['final_opening_mm']:.1f} mm"
                  + (f" | retries {retries}" if retries else "")
                  + (f" | model != world: {'; '.join(diverged)}" if diverged else ""))
    env.close()

    n = len(args.seeds)
    print(f"\n{'condition':22s} {'fwd':>5} {'restored':>8} {'grasp':>6} {'pull':>5} {'goal':>5} {'failed':>6} "
          f"{'retries':>7} {'model!=world':>12} {'moved mm':>8} {'tilt':>5} {'final mm':>13} {'steps':>12}")
    for condition, s in summary.items():
        tilt = "-" if s["max_tilt_after_grasp_deg"] is None else f"{s['max_tilt_after_grasp_deg']:.1f}"
        final = f"{s['final_opening_mm'][0]:.1f}-{s['final_opening_mm'][1]:.1f}"
        steps = "/".join(str(k) for k in s["inverse_control_steps"])
        print(f"{condition:22s} {s['forward_effects_measured']:>5} {s['restored']:>5}/{n:<2} "
              f"{s['grasp_effect_achieved']:>6} {s['pull_open_succeeded']:>5} {s['goal_met']:>5} "
              f"{sum(s['failed_transitions'].values()):>6} {s['recovery_attempts']:>7} "
              f"{s['model_vs_world_differences']:>12} {s['max_drawer_moved_before_pull_mm']:>8.1f} "
              f"{tilt:>5} {final:>13} {steps:>12}")
    print("fwd: forward effects measured; moved mm: max drawer movement before pull_open; "
          "tilt: max hand tilt after the grasp (deg); steps: inverse control steps min/median/max")

    args.out.write_text(json.dumps({"seeds": args.seeds,
                                    "inputs": {"operator": str(args.operator), "spec": str(args.spec),
                                               "stage3": str(args.stage3)},
                                    "goal": [l.to_dict() for l in sorted(join(*goal))],
                                    "forward_effects": [l.to_dict() for l in sorted(forward)],
                                    "hole_spec": spec, "summary": summary, "results": results}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
