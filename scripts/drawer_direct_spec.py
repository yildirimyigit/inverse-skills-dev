"""Direct baseline, step 1: the learning problem, derived by the existing planner.

From the scene the forward push leaves (seeds 0-4, as in Stage 2), the planner
is run with `unrestricted_holes=True`, its ablation option that lets any
unreachable scored fact become a hole. It delegates the whole goal,
drawer_open, after approach_handle. hp.hole_spec turns that plan into the
learning problem, exactly as Stage 2 did for the grasp hole. Nothing in the
spec is written by hand.

Checks: the unrestricted plan is approach_handle -> HOLE[drawer_open] on every
seed, and the direct hole starts where the grasp hole starts (same start
condition, same prefix), so both learners take over from the same states.

Writes artifacts/drawer/direct/hole_spec_direct.json and plan_direct.json.
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

# A check on the planner's output, never an input to it.
_EXPECTED = ("approach_handle(drawer)", "HOLE[drawer_open(drawer)]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2, 3, 4])
    ap.add_argument("--operator", type=Path, default=Path("artifacts/drawer/stage1_operator.json"))
    ap.add_argument("--grasp-spec", type=Path, default=Path("artifacts/drawer/hole_spec.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/drawer/direct"))
    args = ap.parse_args()

    goal_pos, goal_neg = dd.goal_from_operator(args.operator)
    domain = dd.domain()
    registry = dp.registry()
    env = prim.make_env()
    rows, states = [], []
    for seed in args.seeds:
        obs = prim.push_close(env, prim.reset(env, seed))
        scores = {k: float(r.score) for k, r in registry.evaluate_all(prim.obs_to_scene(obs)).items()}
        state = hp.symbolic_state(scores)
        states.append(state)
        direct = hp.plan(domain, state, goal_pos, goal_neg, unrestricted_holes=True)
        frontier = hp.plan(domain, state, goal_pos, goal_neg)
        rows.append({"seed": seed, "state": sorted(state),
                     "unrestricted_plan": list(direct.actions) if direct else None,
                     "unrestricted_cost": list(direct.cost) if direct else None,
                     "frontier_plan": list(frontier.actions) if frontier else None,
                     "matches_expected": bool(direct and direct.actions == _EXPECTED and not direct.alternatives)})
        print(f"seed {seed}: measured {', '.join(sorted(state))} | unrestricted: "
              f"{' -> '.join(rows[-1]['unrestricted_plan'] or ['no plan'])} | "
              f"frontier: {' -> '.join(rows[-1]['frontier_plan'] or ['no plan'])}")
    env.close()

    start = states[0]
    direct = hp.plan(domain, start, goal_pos, goal_neg, unrestricted_holes=True)
    spec = hp.hole_spec(domain, start, direct, goal_pos, goal_neg)
    grasp_spec = load_spec(args.grasp_spec)
    same_start = spec["start"] == grasp_spec["start"] and spec["prefix"] == grasp_spec["prefix"]
    ok = all(row["matches_expected"] for row in rows) and same_start

    print(f"\ndirect hole: {spec['hole']}, achieve {spec['achieve']}")
    print(f"  starts from: {', '.join(spec['start'])} after {[a['name'] for a in spec['prefix']]}"
          f"  (grasp hole: {', '.join(grasp_spec['start'])} after {[a['name'] for a in grasp_spec['prefix']]})")
    print(f"  must preserve: {spec['preserve'] or 'nothing'}; must keep false: {spec['preserve_absent'] or 'nothing'}"
          f"; after the hole: {[a['name'] for a in spec['suffix']] or 'nothing'}")
    print("\nDIRECT SPEC:", "PASS" if ok else "FAIL")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "plan_direct.json").write_text(json.dumps({
        "goal_positive": sorted(goal_pos), "goal_negative": sorted(goal_neg),
        "expected_plan": list(_EXPECTED), "same_start_as_grasp_hole": same_start,
        "all_seeds_match": ok, "seeds": rows}, indent=2))
    (args.out_dir / "hole_spec_direct.json").write_text(json.dumps(spec, indent=2))
    print(f"wrote {args.out_dir}/plan_direct.json and hole_spec_direct.json")


if __name__ == "__main__":
    main()
