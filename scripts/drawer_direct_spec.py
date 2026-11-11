"""Direct baseline, step 1: the learning problem of the historical one-policy baseline.

The direct baseline is a protocol, not a planner answer: one RL policy takes over
where the targeted grasp policy takes over (after approach_handle, from the
standoff) and must restore the whole goal itself, with no symbolic decomposition
after it. Its hole is HOLE[drawer_open], the goal's top-level positive literal;
its plan is approach_handle -> HOLE[drawer_open], fixed here by the protocol.

Under the positive-only goal, the relaxed planner's unrestricted ablation used to
produce exactly this plan. Under the signed goal it no longer does: the plan does
not reach `not at_handle_standoff` in the model, because a hole only makes its
own literal true and the hand stays at the standoff there. So the plan is stated
here, and both planners' own answers are recorded beside it.

Everything else is derived as for every other learned skill: the hole's start is
the exact planner's start rule (the largest set of robot-relative facts the
library holds together: the standoff), and `hole_spec` reads the fences off the
plan's causal structure under the signed goal. Under that goal the plan must keep
`not handle_grasped`: the goal needs the handle released at the end and nothing
after the hole releases it.

Success is measured physically, against the complete signed goal (the direct
environment and the final comparison); the model does not predict it here. The
targeted method has one scripted action after its learned step, pull_open, which
also releases the handle; the direct policy has none, and must end released and
away from the standoff by itself. The record states this difference.

Checks: the stated plan runs under the transition model on every seed, and the
direct hole starts where the grasp hole starts (same start condition, same
prefix), so both learners take over from the same states.

Writes hole_spec_direct.json and plan_direct.json into --out-dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.envs.drawer_hole_env import load_spec
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.exact_hole_planner import hole_starts, library_states
from inverse_skills.operators.literals import join, pos

PREFIX = "approach_handle(drawer)"


def protocol_plan(domain, start) -> hp.PlanResult:
    """approach_handle -> HOLE[drawer_open], with the hole's start by the exact planner's rule,
    and the cost read off the transitions."""
    approach = next(a for a in domain.actions if a.name == PREFIX)
    holes = hole_starts(domain, pos(dd.OPEN), library_states(domain, start))
    if len(holes) != 1:
        raise SystemExit(f"expected one start for HOLE[{dd.OPEN}], got {len(holes)}")
    steps = (approach, holes[0])
    state, displaced, changed = frozenset(start), 0, 0
    for step in steps:
        if not domain.enabled(state, step):
            raise SystemExit(f"{step.name} cannot run where the protocol places it")
        nxt, collateral = domain.transition(state, step)
        displaced += len(collateral)
        if step.hole_for is not None:
            changed += len((nxt ^ state) - collateral)
        state = nxt
    return hp.PlanResult(actions=tuple(s.name for s in steps), steps=steps, delegated=(pos(dd.OPEN),),
                         collateral=displaced, hole_literals=changed, hole_indices=(1,), expanded=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2, 3, 4])
    ap.add_argument("--operator", type=Path, default=Path("artifacts/drawer/stage1_operator.json"))
    ap.add_argument("--grasp-spec", type=Path, default=Path("artifacts/drawer/hole_spec.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/drawer/direct"))
    args = ap.parse_args()
    for name in ("plan_direct.json", "hole_spec_direct.json"):
        if (args.out_dir / name).exists():
            raise SystemExit(f"{args.out_dir / name} exists; saved results are never overwritten. Choose another --out-dir.")

    goal_pos, goal_neg = dd.goal_from_operator(args.operator)
    goal = join(goal_pos, goal_neg)
    domain = dd.domain()
    unscored = dd.domain()
    unscored.scored = unscored.scored - {dd.GRASPED}
    registry = dp.registry()
    env = prim.make_env()
    rows, states = [], []
    for seed in args.seeds:
        obs = prim.push_close(env, prim.reset(env, seed))
        scores = {k: float(r.score) for k, r in registry.evaluate_all(prim.obs_to_scene(obs)).items()}
        state = hp.symbolic_state(scores)
        states.append(state)
        stated = protocol_plan(domain, state)
        exact = hp.plan(domain, state, goal_pos, goal_neg)
        without_grasp = hp.plan(unscored, state, goal_pos, goal_neg)
        relaxed = hp.plan(domain, state, goal_pos, goal_neg, mode="relaxed", unrestricted_holes=True)
        rows.append({"seed": seed, "state": sorted(state),
                     "protocol_plan": list(stated.actions),
                     "protocol_plan_reaches_the_goal_in_the_model": not hp.replay_problems(
                         domain, state, stated.steps, goal_pos, goal_neg),
                     "model_misses": hp.replay_problems(domain, state, stated.steps, goal_pos, goal_neg),
                     "exact_planner": list(exact.actions) if exact else None,
                     "exact_planner_without_a_learnable_grasp": list(without_grasp.actions) if without_grasp else None,
                     "relaxed_unrestricted_ablation": list(relaxed.actions) if relaxed else None})
        print(f"seed {seed}: measured {', '.join(sorted(state))}\n"
              f"  protocol: {' -> '.join(stated.actions)}; in the model it misses: "
              f"{'; '.join(rows[-1]['model_misses']) or 'nothing'}\n"
              f"  the exact planner would plan: {' -> '.join(rows[-1]['exact_planner'] or ['no plan'])}\n"
              f"  with the grasp not learnable: {' -> '.join(rows[-1]['exact_planner_without_a_learnable_grasp'] or ['no plan'])}")
    env.close()

    start = states[0]
    spec = hp.hole_spec(domain, start, protocol_plan(domain, start), goal_pos, goal_neg)
    grasp_spec = load_spec(args.grasp_spec)
    same_start = spec["start"] == grasp_spec["start"] and spec["prefix"] == grasp_spec["prefix"]
    ok = same_start and len({tuple(row["state"]) for row in rows}) == 1

    print(f"\ndirect hole: {spec['hole']}, target {hp.spec_target(spec)}")
    print(f"  starts from: {', '.join(spec['start'])} after {[a['name'] for a in spec['prefix']]}"
          f"  (grasp hole: {', '.join(grasp_spec['start'])} after {[a['name'] for a in grasp_spec['prefix']]})")
    print(f"  must preserve: {spec['preserve'] or 'nothing'}; must keep false: {spec['preserve_absent'] or 'nothing'}"
          f"; after the hole: {[a['name'] for a in spec['suffix']] or 'nothing'}")
    print("\nDIRECT SPEC:", "PASS" if ok else "FAIL")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "plan_direct.json").write_text(json.dumps({
        "goal": [l.to_dict() for l in sorted(goal)], "goal_positive": sorted(goal_pos), "goal_negative": sorted(goal_neg),
        "protocol": "one policy after approach_handle, from the standoff: HOLE[drawer_open], nothing scripted after it",
        "difference_from_the_targeted_method": "the targeted plan runs the scripted pull_open after its learned "
                                               "grasp, which also releases the handle; the direct policy has no "
                                               "scripted action after it",
        "same_start_as_grasp_hole": same_start, "passed": ok, "seeds": rows}, indent=2))
    (args.out_dir / "hole_spec_direct.json").write_text(json.dumps(spec, indent=2))
    print(f"wrote {args.out_dir}/plan_direct.json and hole_spec_direct.json")


if __name__ == "__main__":
    main()
