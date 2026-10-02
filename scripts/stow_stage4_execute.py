"""Stage 4: run the whole inverse on the robot, and the ablations beside it.

Before the conditions run, the learned skill joins the library the same way the
forward skill entered the framework: its own executions are handed to the
operator extractor, and what comes back — preconditions, add and delete
effects — becomes a library action. Nothing about it is written by hand.

Conditions, on the same seeds:

  full                  the planned inverse, the hole filled by the learned skill
  library_only          no plan exists; the hole is skipped to show where it breaks
  without_interference  the ordering a planner without the interference model
                        cannot tell apart from the chosen one — the two tie
  relearned             re-planned once the learned skill is a library action

Every run is monitored: the executive reports each scored literal where the
measured scene disagrees with what that condition's domain predicts.

An inverse is only defined when the forward skill actually happened, so each
episode must start in the state the plan was made for. When the forward push
falls short (it occasionally leaves A outside the pocket's seat), the scene is
re-rolled, as the hole environment does for its handoffs, and the re-roll is
reported rather than hidden.

Writes artifacts/stow/learned_operator.json and artifacts/stow/stage4_execution.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from inverse_skills.envs import stow_domain as sd
from inverse_skills.envs import stow_executive as ex
from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs import stow_primitives as prim
from inverse_skills.envs.stow_hole_env import load_spec
from inverse_skills.logging.rollout import ForwardRollout
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.extractor import OperatorExtractor
from inverse_skills.operators.hole_planner import Action

torch.set_num_threads(1)

EXTRACTION_SEEDS = tuple(range(2000, 2010))   # disjoint from the evaluation seeds
_MAX_FORWARD_ATTEMPTS = 5


def measured(obs, regions) -> frozenset[str]:
    registry = sp.registry()
    scene = prim.obs_to_scene(obs, regions)
    return hp.symbolic_state({k: float(r.score) for k, r in registry.evaluate_all(scene).items()})


def stowed(env, seed: int, expected: frozenset[str]):
    """The scene the forward skill leaves for this seed, re-rolled until it is the
    state the plan starts from. Returns the observation, regions and re-rolls."""
    for attempt in range(_MAX_FORWARD_ATTEMPTS):
        obs = prim.reset(env, seed + attempt * 10_000)
        regions = prim.regions(obs)
        obs = prim.forward_stow(env, obs)
        if measured(obs, regions) == expected:
            return obs, regions, attempt
    raise RuntimeError(f"seed {seed}: the forward skill never produced the planned start")


def learn_operator(env, model, spec: dict, seeds, planned_start) -> tuple[Action, dict]:
    """Model the learned skill from its own executions, as a demonstration would be."""
    prefix = [Action.from_dict(a) for a in spec["prefix"]]
    registry = sp.registry()
    name = f"learned:{spec['achieve']}"
    rollouts = []
    for seed in seeds:
        obs, regions, _ = stowed(env, seed, planned_start)
        obs, report = ex.execute_plan(env, obs, prefix, regions=regions)
        if len(report.steps) != len(prefix) or not all(s.ok for s in report.steps):
            continue
        before = prim.obs_to_scene(obs, regions)
        obs = ex.run_learned(env, obs, model, spec["achieve"], regions, registry)
        after = prim.obs_to_scene(obs, regions, timestep=1)
        rollouts.append(ForwardRollout(skill_name=name, demo_id=f"run_{seed}", scenes=[before, after]))
    extracted = OperatorExtractor(registry).extract(name, rollouts)
    action = hp.learned_action(spec["achieve"], extracted.operator)
    return action, {"rollouts": len(rollouts), "seeds": list(seeds),
                    "operator": extracted.operator.to_dict(), "scores": extracted.scores,
                    "action": action.to_dict()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")],
                    default=list(range(1000, 1010)))
    ap.add_argument("--spec", type=Path, default=Path("artifacts/stow/hole_spec.json"))
    ap.add_argument("--operator", type=Path, default=Path("artifacts/stow/stage1_operator.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/stow/stage4_execution.json"))
    ap.add_argument("--learned-out", type=Path, default=Path("artifacts/stow/learned_operator.json"))
    args = ap.parse_args()

    model = SAC.load(args.checkpoint, device="cuda")
    spec = load_spec(args.spec)
    goal = sd.goal_from_operator(args.operator)
    start = sd.stowed_state()
    env = prim.make_env()

    domain = sd.domain()
    planned = hp.plan(domain, start, *goal)
    if list(planned.actions) != spec["plan"]:
        raise SystemExit("the plan differs from the one the policy was trained for; rerun stage 2")
    blind_domain = sd.domain(interference=False)
    blind = hp.plan(blind_domain, start, *goal)
    other = next(alt for alt in (blind.steps, *blind.alternatives)
                 if tuple(a.name for a in alt) != planned.actions)

    print(f"learning the model of LEARNED[{spec['achieve']}] from its executions "
          f"on seeds {EXTRACTION_SEEDS[0]}-{EXTRACTION_SEEDS[-1]}")
    learned, learned_record = learn_operator(env, model, spec, EXTRACTION_SEEDS, start)
    print(f"  from {learned_record['rollouts']} runs:")
    print(f"    pre   {', '.join(sorted(learned.pre))}")
    print(f"    add   {', '.join(sorted(learned.add))}")
    print(f"    del   {', '.join(sorted(learned.delete))}")
    args.learned_out.write_text(json.dumps(learned_record, indent=2))
    learned_domain = sd.domain(extra_actions=[learned])
    relearned = hp.plan(learned_domain, start, *goal, max_delegated=0)

    conditions = {
        "full": (list(planned.steps), domain),
        "library_only": ([s for s in planned.steps if s.hole_for is None], domain),
        "without_interference": (list(other), blind_domain),
        "relearned": (list(relearned.steps) if relearned else [], learned_domain),
    }
    print("\nplans")
    for name, (steps, _) in conditions.items():
        print(f"  {name:21s}", " -> ".join(s.name for s in steps) or "no plan")

    results = {name: [] for name in conditions}
    for name, (steps, cond_domain) in conditions.items():
        print(f"\n{name}")
        for seed in args.seeds:
            obs, regions, rerolls = stowed(env, seed, start)
            obs, report = ex.execute_plan(env, obs, steps, model=model, regions=regions,
                                          goal=goal, domain=cond_domain)
            results[name].append({"seed": seed, "forward_rerolls": rerolls, **report.to_dict()})
            failed = next((s for s in report.steps if not s.ok), None)
            status = "goal met" if report.goal_met else (
                f"failed at {failed.action} ({failed.detail})" if failed else "goal not met")
            diverged = [f"{s.action}: {', '.join(s.divergence)}" for s in report.steps if s.divergence]
            retries = sum(max(s.attempts - 1, 0) for s in report.steps)
            print(f"  seed {seed}: {status}; A {report.cube_a_err_mm:5.1f}mm "
                  f"B {report.cube_b_err_mm:5.1f}mm; retries {retries}"
                  + (f"; forward skill re-rolled {rerolls}x" if rerolls else ""))
            for line in diverged:
                print(f"      model != world after {line}")
    env.close()

    print(f"\n{'condition':22s} {'goal met':>9} {'mean A err':>11} {'mean B err':>11} {'diverging steps':>16}")
    summary = {}
    for name, rows in results.items():
        met = float(np.mean([bool(r["goal_met"]) for r in rows]))
        a_err = float(np.mean([r["cube_a_err_mm"] for r in rows]))
        b_err = float(np.mean([r["cube_b_err_mm"] for r in rows]))
        diverging = sum(bool(s["divergence"]) for r in rows for s in r["steps"])
        summary[name] = {"goal_met_rate": met, "mean_a_err_mm": a_err, "mean_b_err_mm": b_err,
                         "diverging_steps": diverging}
        print(f"{name:22s} {met:>8.0%} {a_err:>10.1f}mm {b_err:>10.1f}mm {diverging:>16}")

    passed = (summary["full"]["goal_met_rate"] >= 0.8
              and summary["library_only"]["goal_met_rate"] == 0.0
              and summary["relearned"]["goal_met_rate"] >= 0.8
              and spec["achieve"] in learned.add)
    print("\nSTAGE 4:", "PASS" if passed else "FAIL")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "plans": {name: [s.name for s in steps] for name, (steps, _) in conditions.items()},
        "learned_action": learned.to_dict(),
        "summary": summary, "results": results, "passed": passed}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
