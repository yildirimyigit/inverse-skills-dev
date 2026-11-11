"""Stage 1 for the drawer: extract the forward operator "push_close" from demonstrations.

Each demonstration spans the whole forward skill: the first scene is the world
before the push (drawer open, arm at its start pose, gripper open), the last is
the world it leaves behind (drawer shut, arm back home, gripper open). The
existing OperatorExtractor, with its default min_max rule, turns these scenes
into signed preconditions and signed effects, with no drawer-specific rule: every
predicate reliably true or reliably false before every demonstration is a
precondition. The restoration goal is G = Pre u complement(Eff), which the strict
rule makes Pre itself: the inverse must restore every reliably observed initial
condition, false ones included.

The goal is symbolic. drawer_open holds for any opening that scores at least
0.8 (about 94 mm or more); the inverse must make that predicate true again. It
does not have to return the drawer to the opening it started with (125-179 mm
across these seeds).

Prints the per-predicate score table, the extracted operator and its restoration
goal, checks the effects against what the forward skill is designed to do (a
check, never an input; see operators/stage1_checks.py), and writes the operator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.logging.rollout import ForwardRollout
from inverse_skills.operators.extractor import OperatorExtractor
from inverse_skills.operators.literals import neg, pos
from inverse_skills.operators.stage1_checks import signed_summary, stage1_checks, stage1_table, uncertain_at_start

# What the forward skill is meant to do. Only compared with the result.
# What the forward skill is designed to do, as signed effects. Only compared with the result;
# the preconditions, and so the restoration goal, are whatever the demonstrations show.
DESIGNED_EFFECTS = {pos("drawer_closed(drawer)"), neg("drawer_open(drawer)")}


def demonstrate(env, seed: int) -> tuple[ForwardRollout, float]:
    obs = prim.reset(env, seed)
    start_opening = prim.opening(obs)
    start = prim.obs_to_scene(obs)
    obs = prim.push_close(env, obs)
    end = prim.obs_to_scene(obs, timestep=1)
    return ForwardRollout(skill_name="push_close", demo_id=f"demo_{seed}", scenes=[start, end]), start_opening


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")],
                    default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=Path("artifacts/drawer/stage1_operator.json"))
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; saved results are never overwritten. Choose another --out.")

    env = prim.make_env()
    demos = [demonstrate(env, seed) for seed in args.seeds]
    env.close()
    rollouts = [rollout for rollout, _ in demos]
    start_openings_mm = [round(opening * 1000, 1) for _, opening in demos]

    registry = dp.registry()
    result = OperatorExtractor(registry).extract("push_close", rollouts)
    operator = result.operator

    print(f"{len(rollouts)} demonstrations, seeds {args.seeds}, drawer starting "
          f"{min(start_openings_mm)}-{max(start_openings_mm)} mm open\n")
    for line in stage1_table(result.scores):
        print(line)

    summary = signed_summary(operator)
    print("\noperator 'push_close'")
    print("  Pre:", ", ".join(summary["preconditions"]) or "-")
    print("  Eff:", ", ".join(summary["effects"]) or "-")
    print("  restoration goal G = Pre u complement(Eff):", ", ".join(summary["restoration_goal"]))
    uncertain = uncertain_at_start(result.scores)
    print("  neither reliably true nor false at the start (not in G):", ", ".join(uncertain) or "none")

    checks = stage1_checks(operator, DESIGNED_EFFECTS, set(registry.robot_relative_keys()))
    print("\nchecks:", json.dumps(checks))
    for name, ok in checks.items():
        if not ok:
            print(f"  {name}: FAILED")
    print("STAGE 1:", "PASS" if all(checks.values()) else "FAIL")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "seeds": args.seeds,
        "start_openings_mm": start_openings_mm,
        "demonstrations": {"skill": "push_close", "seeds": args.seeds,
                           "boundaries": "the whole skill: the scene before it starts and the scene it leaves"},
        "extraction": {"mode": "min_max", "true_threshold": 0.8, "false_threshold": 0.2},
        "operator": operator.to_dict(),
        "signed": summary,
        "uncertain_at_start": uncertain,
        "scores": result.scores,
        "checks": checks,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
