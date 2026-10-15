"""Stage 2 for the drawer: plan the inverse from the scene the push leaves, and
write the hole's learning problem.

Runs the demonstrated push, reads the predicate scores off the real scene,
thresholds them into a symbolic state, and plans against the inverse goal that
the generic inversion rule derived from the Stage 1 operator. The planner is
unchanged: which predicate to delegate, where the learned skill starts and what
it must preserve are all derived, and printed so they can be checked.

Ablations, from the same measured state:
  library_only    the known actions, no holes allowed
  frontier        the normal planner: holes only at capability gaps
  unrestricted    any unreachable scored fact may become a hole (experiments only)
  with_grasp      Stage 0's scripted grasp added as a known action

Writes artifacts/drawer/stage2_plan.json and artifacts/drawer/hole_spec.json,
the input RL will train against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.operators import hole_planner as hp

# The plan we expect. A check on the planner's output, never an input to it.
_EXPECTED = ("approach_handle(drawer)", "HOLE[handle_grasped(drawer)]", "pull_open(drawer)")


def measured_state(env, seed: int, registry) -> tuple[frozenset[str], dict[str, float]]:
    obs = prim.push_close(env, prim.reset(env, seed))
    scene = prim.obs_to_scene(obs)
    scores = {k: float(r.score) for k, r in registry.evaluate_all(scene).items()}
    return hp.symbolic_state(scores), scores


def summary(result) -> dict:
    row = {"capability_gaps": list(result.capability_gaps),
           "synthesizable_gaps": list(result.synthesizable_gaps)}
    if not result:
        row["plan"] = None
        return row
    row.update(plan=list(result.actions), cost=list(result.cost), delegated=list(result.delegated),
               hole_indices=list(result.hole_indices), candidates=list(result.candidates),
               ties=[list(a) for a in result.alternative_names], expanded=result.expanded)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")],
                    default=[0, 1, 2, 3, 4])
    ap.add_argument("--operator", type=Path, default=Path("artifacts/drawer/stage1_operator.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/drawer/stage2_plan.json"))
    ap.add_argument("--spec-out", type=Path, default=Path("artifacts/drawer/hole_spec.json"))
    args = ap.parse_args()

    goal_pos, goal_neg = dd.goal_from_operator(args.operator)
    domain = dd.domain()
    print("goal (generic inversion of the Stage 1 operator)")
    print("  make true: ", ", ".join(sorted(goal_pos)))
    print("  make false:", ", ".join(sorted(goal_neg)))

    registry = dp.registry()
    env = prim.make_env()
    rows, states = [], []
    for seed in args.seeds:
        state, scores = measured_state(env, seed, registry)
        states.append(state)
        result = hp.plan(domain, state, goal_pos, goal_neg)
        row = {"seed": seed, "state": sorted(state), "scores": scores, **summary(result)}
        row["matches_expected"] = bool(result and result.actions == _EXPECTED and not result.alternatives)
        rows.append(row)
        print(f"\nseed {seed}: measured state = {', '.join(sorted(state))}")
        print("  plan:", " -> ".join(row["plan"]) if row["plan"] else "none",
              f"  cost {row.get('cost')}")
    env.close()

    start = states[0]
    reachable = hp.relaxed_reachable(domain, start)
    frontier = hp.plan(domain, start, goal_pos, goal_neg)
    spec = hp.hole_spec(domain, start, frontier, goal_pos, goal_neg)
    robot_facts = sorted(p for p in domain.robot_relative if hp.predicate_args(p)[:1] == ("drawer",))

    print("\nderived, not given")
    print("  reachable by the library: ", ", ".join(sorted(reachable)))
    print("  capability gaps:          ", ", ".join(frontier.capability_gaps))
    print("  synthesizable (holes):    ", ", ".join(frontier.synthesizable_gaps))
    print("  hole starts from:         ", ", ".join(spec["start"]),
          f"  (robot-relative facts about the drawer: {', '.join(robot_facts)};"
          f" of these the library can reach: {', '.join(p for p in robot_facts if p in reachable)})")
    print("  must preserve:            ", ", ".join(spec["preserve"]) or "nothing")
    print("  must keep false:          ", ", ".join(spec["preserve_absent"]) or "nothing")

    ablations = {
        "library_only": summary(hp.plan(domain, start, goal_pos, goal_neg, max_delegated=0)),
        "frontier": summary(frontier),
        "unrestricted": summary(hp.plan(domain, start, goal_pos, goal_neg, unrestricted_holes=True)),
        "with_grasp": summary(hp.plan(dd.domain(extra_actions=[dd.scripted_grasp()]),
                                      start, goal_pos, goal_neg)),
    }
    print("\nablations")
    for name, row in ablations.items():
        plan = " -> ".join(row["plan"]) if row["plan"] else "no plan"
        print(f"  {name:13s} {plan:75s} cost {row.get('cost')}  gaps {row['capability_gaps']}")

    ok = all(row["matches_expected"] for row in rows)
    print("\nSTAGE 2:", "PASS" if ok else "FAIL")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "goal_positive": sorted(goal_pos),
        "goal_negative": sorted(goal_neg),
        "expected_plan": list(_EXPECTED),
        "all_seeds_match": ok,
        "reachable": sorted(reachable),
        "ablations": ablations,
        "seeds": rows,
    }, indent=2))
    args.spec_out.write_text(json.dumps(spec, indent=2))
    print(f"wrote {args.out} and {args.spec_out}")


if __name__ == "__main__":
    main()
