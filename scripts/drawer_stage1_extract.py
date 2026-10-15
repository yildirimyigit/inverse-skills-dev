"""Stage 1 for the drawer: extract the forward operator "push_close" from demonstrations.

Each demonstration spans the whole forward skill: the first scene is the world
before the push (drawer open, arm at its start pose, gripper open), the last is
the world it leaves behind (drawer shut, arm back home, gripper open). The
existing OperatorExtractor turns these scenes into preconditions, add effects
and delete effects, with no drawer-specific rule; the generic inversion rule
(preconditions + delete effects + the negation of each add effect) then gives
the goal the inverse must restore.

The goal is symbolic. drawer_open holds for any opening that scores at least
0.8 (about 94 mm or more); the inverse must make that predicate true again. It
does not have to return the drawer to the opening it started with (125-179 mm
across these seeds).

Prints the per-predicate score table, the extracted operator and its inverse
goal, compares them with what we expect (a check, never an input), and writes
artifacts/drawer/stage1_operator.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.logging.rollout import ForwardRollout
from inverse_skills.operators.extractor import OperatorExtractor

# What the forward skill is meant to do. Only compared with the result.
_EXPECTED_PRE = {"drawer_open(drawer)"}
_EXPECTED_ADD = {"drawer_closed(drawer)"}
_EXPECTED_DEL = {"drawer_open(drawer)"}
_ROBOT_TERMS = {"at_handle_standoff(drawer)", "handle_grasped(drawer)"}


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
    print(f"{'predicate':28s} {'start':>7} {'end':>7} {'delta':>7}")
    for key in registry.keys():
        s = result.scores[key]
        print(f"{key:28s} {s['start_mean']:7.3f} {s['end_mean']:7.3f} {s['delta']:+7.3f}")

    pre = {t.key for t in operator.preconditions}
    add = {t.key for t in operator.add_effects}
    dele = {t.key for t in operator.delete_effects}
    goal = [f"{'not ' if t.polarity == 'negative' else ''}{t.key}"
            for t in operator.inverse_target_terms()]
    print("\noperator 'push_close'")
    print("  Pre:", ", ".join(sorted(pre)) or "-")
    print("  Add:", ", ".join(sorted(add)) or "-")
    print("  Del:", ", ".join(sorted(dele)) or "-")
    print("  inverse goal:", ", ".join(goal))
    print("  (symbolic: drawer_open must score >= 0.8, about 94 mm or more; the starting opening is not a target)")

    checks = {
        "pre_matches": pre == _EXPECTED_PRE,
        "add_matches": add == _EXPECTED_ADD,
        "del_matches": dele == _EXPECTED_DEL,
        "no_robot_terms": not (_ROBOT_TERMS & (pre | add | dele)),
    }
    print("\nchecks:", json.dumps(checks))
    print("STAGE 1:", "PASS" if all(checks.values()) else "FAIL")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "seeds": args.seeds,
        "start_openings_mm": start_openings_mm,
        "operator": operator.to_dict(),
        "scores": result.scores,
        "checks": checks,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
