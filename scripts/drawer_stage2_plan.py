"""Stage 2 for the drawer: plan the inverse from the scene the push leaves, and
write the hole's learning problem.

Runs the demonstrated push, reads the predicate scores off the real scene as a
Boolean symbolic state (true when a predicate scores at least 0.5), and plans
against the signed restoration goal of the Stage 1 operator, G = Pre. Which
signed literal to delegate, where the learned skill starts and what it must
preserve are all derived, and printed so they can be checked.

Every seed's plan is replayed with the transition model. The plan the published
experiments used is compared with, never imposed. STAGE 2 passes when every seed
has a plan within the learning budget, every seed has the same plan, and each
plan replays to the goal.

Ablations, from the same measured state:
  library_only    the known actions, no holes allowed
  frontier        the planner as it stands, in its default exact mode
  unrestricted    any unreachable scored fact may become a hole: the relaxed mode's ablation,
                  named as such, because the exact mode has no such option (experiments only).
                  The relaxed mode is the historical comparison mode: its holes are positive only
  with_grasp      Stage 0's scripted grasp added as a known action

Writes artifacts/drawer/stage2_plan.json and artifacts/drawer/hole_spec.json,
the input RL will train against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.exact_hole_planner import ExactNoPlan, ExactPlanResult
from inverse_skills.operators.literals import join, robust_value

# The plan the published experiments used, under the positive-only goal. Compared with, never
# an input to the planner.
HISTORICAL_PLAN = ("approach_handle(drawer)", "HOLE[handle_grasped(drawer)]", "pull_open(drawer)")


def literal_dicts(literals) -> list[dict]:
    return [literal.to_dict() for literal in literals]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measured_state(env, seed: int, registry) -> tuple[frozenset[str], dict[str, float]]:
    obs = prim.push_close(env, prim.reset(env, seed))
    scene = prim.obs_to_scene(obs)
    scores = {k: float(r.score) for k, r in registry.evaluate_all(scene).items()}
    return hp.symbolic_state(scores), scores


def summary(result) -> dict:
    """A plan, or why there is none, in the terms of the mode that made it: the exact mode reports
    each hole's context or the needs left unmet, the relaxed mode its capability gaps."""
    row = {"plan": list(result.actions) if result else None}
    if result:
        row.update(cost=list(result.cost), delegated=literal_dicts(result.delegated),
                   hole_indices=list(result.hole_indices), candidates=literal_dicts(result.candidates),
                   ties=[list(a) for a in result.alternative_names], expanded=result.expanded)
    if isinstance(result, ExactPlanResult):
        row["planning_cost"] = list(result.planning_cost)
        row["hole_contexts"] = [context.to_dict() for context in result.hole_contexts]
    elif isinstance(result, ExactNoPlan):
        row.update(unmet_needs=literal_dicts(result.unmet_needs), feasible_with_all_holes=result.feasible_with_all_holes)
    else:
        row.update(capability_gaps=list(result.capability_gaps), synthesizable_gaps=list(result.synthesizable_gaps))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")],
                    default=[0, 1, 2, 3, 4])
    ap.add_argument("--operator", type=Path, default=Path("artifacts/drawer/stage1_operator.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/drawer/stage2_plan.json"))
    ap.add_argument("--spec-out", type=Path, default=Path("artifacts/drawer/hole_spec.json"))
    args = ap.parse_args()
    for path in (args.out, args.spec_out):
        if path.exists():
            raise SystemExit(f"{path} exists; saved results are never overwritten. Choose another --out and --spec-out.")

    goal_pos, goal_neg = dd.goal_from_operator(args.operator)
    goal = join(goal_pos, goal_neg)
    domain = dd.domain()
    registry = dp.registry()
    unmeasured = sorted(hp.domain_atoms(domain) - set(registry.keys()))
    print("restoration goal G (from the Stage 1 operator):", ", ".join(str(l) for l in sorted(goal)))
    print("atoms the domain uses but nothing measures:", ", ".join(unmeasured) or "none")

    env = prim.make_env()
    rows, states = [], []
    for seed in args.seeds:
        state, scores = measured_state(env, seed, registry)
        states.append(state)
        result = hp.plan(domain, state, goal_pos, goal_neg)
        row = {"seed": seed, "state": sorted(state), "scores": scores,
               "uncertain": sorted(k for k, v in scores.items() if robust_value(v) == "uncertain"),
               **summary(result)}
        row["replay_problems"] = (hp.replay_problems(domain, state, result.steps, goal_pos, goal_neg)
                                  if result else ["no plan"])
        row["same_as_historical"] = bool(result and result.actions == HISTORICAL_PLAN)
        rows.append(row)
        print(f"\nseed {seed}: measured state = {', '.join(sorted(state))}")
        if row["uncertain"]:
            print(f"  between 0.2 and 0.8 (Boolean and robust readings differ): {', '.join(row['uncertain'])}")
        print("  plan:", " -> ".join(row["plan"]) if row["plan"] else "none",
              f"  planning cost {row.get('planning_cost')}",
              "  replay:", "reaches the goal" if not row["replay_problems"] else "; ".join(row["replay_problems"]))
    env.close()

    start = states[0]
    frontier = hp.plan(domain, start, goal_pos, goal_neg)
    spec = hp.hole_spec(domain, start, frontier, goal_pos, goal_neg)
    robot_facts = sorted(p for p in domain.robot_relative if hp.predicate_args(p)[:1] == ("drawer",))

    print("\nderived, not given")
    print("  delegated to learning:    ", ", ".join(map(str, frontier.delegated)))
    for context in frontier.hole_contexts:
        print(f"    {context.literal}: missing for {', '.join(context.waiting)}; first used by {context.used_by}")
    print("  hole starts from:         ", ", ".join(spec["start"]) or "nothing",
          f"  (robot-relative facts about the drawer: {', '.join(robot_facts)})")
    print("  must preserve:            ", ", ".join(spec["preserve"]) or "nothing")
    print("  must keep false:          ", ", ".join(spec["preserve_absent"]) or "nothing")
    print("  handoff conditions:       ", json.dumps(spec["handoff_conditions"]))

    ablations = {
        "library_only": summary(hp.plan(domain, start, goal_pos, goal_neg, max_delegated=0)),
        "frontier": summary(frontier),
        "unrestricted": summary(hp.plan(domain, start, goal_pos, goal_neg, mode="relaxed", unrestricted_holes=True)),
        "with_grasp": summary(hp.plan(dd.domain(extra_actions=[dd.scripted_grasp()]),
                                      start, goal_pos, goal_neg)),
    }
    print("\nablations")
    for name, row in ablations.items():
        plan = " -> ".join(row["plan"]) if row["plan"] else "no plan"
        print(f"  {name:13s} {plan:75s} cost {row.get('cost')}")

    plans = {tuple(row["plan"]) for row in rows if row["plan"] is not None}
    ok = (all(row["plan"] is not None and not row["replay_problems"] for row in rows) and len(plans) == 1)
    same = all(row["same_as_historical"] for row in rows)
    print(f"\nthe published plan ({' -> '.join(HISTORICAL_PLAN)}): " + ("the same" if same else "changed"))
    print("STAGE 2:", "PASS" if ok else "FAIL")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "inputs": {"operator": str(args.operator), "operator_sha256": sha256(args.operator)},
        "goal": literal_dicts(sorted(goal)),
        "goal_positive": sorted(goal_pos),
        "goal_negative": sorted(goal_neg),
        "unmeasured_atoms": {"atoms": unmeasured,
                             "assumed": "false at the start, by the closed world; no predicate measures them"},
        "historical_plan": list(HISTORICAL_PLAN),
        "same_as_historical": same,
        "all_seeds_same_plan": len(plans) == 1,
        "passed": ok,
        "ablations": ablations,
        "seeds": rows,
    }, indent=2))
    args.spec_out.write_text(json.dumps(spec, indent=2))
    print(f"wrote {args.out} and {args.spec_out}")


if __name__ == "__main__":
    main()
