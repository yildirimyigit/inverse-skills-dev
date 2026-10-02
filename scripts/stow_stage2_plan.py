"""Stage 2: plan the inverse from the scene the forward skill leaves, and write
the hole's learning problem.

Runs the demonstrated skill, reads the predicate scores off the real scene,
thresholds them into a symbolic state, and plans against the inverse target
extracted in Stage 1. Everything the planner uses about holes — which predicate
to delegate, where the skill starts, what it must preserve — is derived, and
printed so it can be checked.

Writes artifacts/stow/stage2_plan.json and artifacts/stow/hole_spec.json, the
input Stage 3 trains against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inverse_skills.envs import stow_domain as sd
from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs import stow_primitives as prim
from inverse_skills.operators import hole_planner as hp

# The plan this scene is designed to need. A check on the planner's output,
# never an input to it.
_EXPECTED = (
    "pick(cube_b)", "place(cube_b,src_b)", "approach(cube_a)",
    "HOLE[clear_of_walls(cube_a)]", "pick(cube_a)", "place(cube_a,src_a)",
)


def measured_state(env, seed: int, registry) -> tuple[frozenset[str], dict[str, float]]:
    obs = prim.reset(env, seed)
    regions = prim.regions(obs)
    obs = prim.forward_stow(env, obs)
    scene = prim.obs_to_scene(obs, regions)
    scores = {k: float(r.score) for k, r in registry.evaluate_all(scene).items()}
    return hp.symbolic_state(scores), scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")],
                    default=[0, 1, 2, 3, 4])
    ap.add_argument("--operator", type=Path, default=Path("artifacts/stow/stage1_operator.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/stow/stage2_plan.json"))
    ap.add_argument("--spec-out", type=Path, default=Path("artifacts/stow/hole_spec.json"))
    args = ap.parse_args()

    goal_pos, goal_neg = sd.goal_from_operator(args.operator)
    domain = sd.domain()
    print("goal (from the extracted operator)")
    print("  restore:", ", ".join(sorted(goal_pos)))
    print("  remove: ", ", ".join(sorted(goal_neg)))
    print("world facts the domain declares")
    for d in domain.disturbances:
        print(f"  changing {d.trigger} displaces {', '.join(sorted(d.clobbers))}")

    env = prim.make_env()
    rows, states = [], []
    for seed in args.seeds:
        state, scores = measured_state(env, seed, sp.registry())
        states.append(state)
        result = hp.plan(domain, state, goal_pos, goal_neg)
        rows.append({
            "seed": seed,
            "state": sorted(state),
            "scores": scores,
            "plan": list(result.actions) if result else None,
            "candidates": list(result.candidates) if result else None,
            "delegated": list(result.delegated) if result else None,
            "hole_indices": list(result.hole_indices) if result else None,
            "cost": list(result.cost) if result else None,
            "ties": [list(a) for a in result.alternative_names] if result else None,
            "expanded": result.expanded if result else None,
            "matches_expected": bool(result and result.actions == _EXPECTED
                                     and not result.alternatives),
        })
        print(f"\nseed {seed}: measured state = {', '.join(sorted(state))}")
        if result is None:
            print("  no plan")
            continue
        for i, action in enumerate(result.actions):
            mark = "  <- learned" if action.startswith("HOLE") else ""
            print(f"  {i + 1}. {action}{mark}")
        print(f"  cost (delegated, collateral, hole literals, holes, length) = {result.cost}"
              f"   ties {len(result.alternatives)}   expanded {result.expanded} states")
    env.close()

    start = states[0]
    reachable = hp.relaxed_reachable(domain, start)
    result = hp.plan(domain, start, goal_pos, goal_neg)
    spec = hp.hole_spec(domain, start, result, goal_pos, goal_neg)
    blind = hp.plan(sd.domain(interference=False), start, goal_pos, goal_neg)

    print("\nderived, not given")
    print("  unreachable by the library:", ", ".join(sorted(goal_pos - reachable)))
    print("  hole candidates:          ", ", ".join(result.candidates))
    print("  hole starts from:         ", ", ".join(spec["start"]))
    print("  must preserve:            ", ", ".join(spec["preserve"]))
    print("  must keep false:          ", ", ".join(spec["preserve_absent"]))
    print("\nablations")
    print("  library only (max_delegated=0):",
          hp.plan(domain, start, goal_pos, goal_neg, max_delegated=0) or "no plan")
    print(f"  without the interference model: {1 + len(blind.alternatives)} orderings tie")
    for alt in (blind.actions, *blind.alternative_names):
        print("    ", " -> ".join(alt))

    ok = all(row["matches_expected"] for row in rows)
    print("\nSTAGE 2:", "PASS" if ok else "FAIL")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "goal_positive": sorted(goal_pos),
        "goal_negative": sorted(goal_neg),
        "expected_plan": list(_EXPECTED),
        "all_seeds_match": ok,
        "unreachable": sorted(goal_pos - reachable),
        "without_interference_model": [list(blind.actions)]
                                      + [list(a) for a in blind.alternative_names],
        "seeds": rows,
    }, indent=2))
    args.spec_out.write_text(json.dumps(spec, indent=2))
    print(f"wrote {args.out} and {args.spec_out}")


if __name__ == "__main__":
    main()
