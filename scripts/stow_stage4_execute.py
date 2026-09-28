"""Stage 4: run the whole inverse on the robot, and the ablations beside it.

Four conditions on the same seeds:

  full          the planned inverse, hole filled by the learned operator
  library_only  the best the library can do: no plan exists, so the hole is
                simply skipped and the executive reports where it breaks
  hole_first    the ordering the late-hole tie-break rejected, at equal cost
  relearned     re-plan once the learned operator joins the library: no holes

Writes artifacts/stow/stage4_execution.json.
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
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.hole_planner import Action

torch.set_num_threads(1)

# The learned operator, as it enters the library: the predicate it establishes,
# the precondition Stage 3 measured, and the gripper state it leaves behind.
LEARNED_DRAG = Action(
    name="drag(cube_a)",
    pre=frozenset({"tcp_near(cube_a)"}),
    add=frozenset({"clear_of_walls(cube_a)"}),
    delete=frozenset({"gripper_open()"}),
)


def stowed(env, seed: int):
    obs = prim.reset(env, seed)
    regions = prim.regions(obs)
    return prim.forward_stow(env, obs), regions


def symbolic(obs, regions) -> frozenset[str]:
    registry = sp.registry()
    scene = prim.obs_to_scene(obs, regions)
    return hp.symbolic_state({k: float(r.score) for k, r in registry.evaluate_all(scene).items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")],
                    default=[1000, 1001, 1002, 1003, 1004])
    ap.add_argument("--operator", type=Path, default=Path("artifacts/stow/stage1_operator.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/stow/stage4_execution.json"))
    args = ap.parse_args()

    model = SAC.load(args.checkpoint, device="cuda")
    goal_pos, goal_neg = sd.goal_from_operator(args.operator)
    domain = sd.domain()
    env = prim.make_env()

    planned = hp.plan(domain, sd.stowed_state(), goal_pos, goal_neg)
    full_plan = list(planned.actions)
    hole_first = list(planned.alternatives[0])
    library_only = [a for a in full_plan if not a.startswith("HOLE")]
    relearned = hp.plan(sd.domain(extra_actions=[LEARNED_DRAG]), sd.stowed_state(),
                        goal_pos, goal_neg, max_delegated=0)

    print("plans")
    print("  full        ", " -> ".join(full_plan))
    print("  library_only", " -> ".join(library_only), " (no plan exists; the hole is skipped)")
    print("  hole_first  ", " -> ".join(hole_first))
    print("  relearned   ", " -> ".join(relearned.actions),
          f" (0 holes, cost {relearned.cost})")

    learned_domain = sd.domain(extra_actions=[LEARNED_DRAG])
    conditions = {"full": full_plan, "library_only": library_only,
                  "hole_first": hole_first, "relearned": list(relearned.actions),
                  "relearned_replan": list(relearned.actions)}
    results = {name: [] for name in conditions}
    for name, plan in conditions.items():
        print(f"\n{name}")
        for seed in args.seeds:
            obs, regions = stowed(env, seed)
            obs, report = ex.execute_plan(env, obs, plan, model=model, regions=regions)
            # The contract's recovery path: a plan whose steps all passed their
            # own postconditions can still miss the goal, because a later step
            # disturbs an earlier one. The executive measures the world and
            # plans again from what it finds.
            rounds = 1
            if name.endswith("replan"):
                while not report.goal_met and rounds < 3:
                    again = hp.plan(learned_domain, symbolic(obs, regions),
                                    goal_pos, goal_neg, max_delegated=0)
                    if again is None or not again.actions:
                        break
                    print(f"      replanning: {' -> '.join(again.actions)}")
                    obs, report = ex.execute_plan(env, obs, list(again.actions),
                                                  model=model, regions=regions)
                    rounds += 1
            results[name].append({"seed": seed, "rounds": rounds, **report.to_dict()})
            failed = next((s for s in report.steps if not s.ok), None)
            status = "goal met" if report.goal_met else (
                f"failed at {failed.action} ({failed.detail})" if failed else "goal not met")
            retries = sum(s.attempts - 1 for s in report.steps)
            print(f"  seed {seed}: {status}; A {report.cube_a_err_mm:5.1f}mm "
                  f"B {report.cube_b_err_mm:5.1f}mm; retries {retries}")
            if not report.goal_met:
                for s in report.steps:
                    print(f"      {'ok ' if s.ok else 'BAD'} {s.action:28s} x{s.attempts} {s.detail}")
    env.close()

    print(f"\n{'condition':14s} {'goal met':>9} {'mean A err':>11} {'mean B err':>11}")
    summary = {}
    for name, rows in results.items():
        met = float(np.mean([r["goal_met"] for r in rows]))
        a_err = float(np.mean([r["cube_a_err_mm"] for r in rows]))
        b_err = float(np.mean([r["cube_b_err_mm"] for r in rows]))
        summary[name] = {"goal_met_rate": met, "mean_a_err_mm": a_err, "mean_b_err_mm": b_err}
        print(f"{name:14s} {met:>8.0%} {a_err:>10.1f}mm {b_err:>10.1f}mm")

    passed = summary["full"]["goal_met_rate"] >= 0.8 and summary["library_only"]["goal_met_rate"] == 0.0
    print("\nSTAGE 4:", "PASS" if passed else "FAIL")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"plans": conditions, "summary": summary,
                                    "results": results, "passed": passed}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
