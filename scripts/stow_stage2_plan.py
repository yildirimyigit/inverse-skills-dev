"""Stage 2: plan the inverse from the scene the forward skill leaves, and write
the hole's learning problem.

Runs the demonstrated skill, reads the predicate scores off the real scene as a
Boolean symbolic state (a predicate is true when it scores at least 0.5), and
plans against the signed restoration goal extracted in Stage 1, G = Pre.
Everything the planner uses about holes — which signed literal to delegate,
where the skill starts, what it must preserve — is derived, and printed so it
can be checked. The library includes retreat_to_safe; the planner decides where,
if anywhere, it is needed.

Every seed's plan is replayed with the transition model. The plan the published
experiments used is compared with, never imposed. STAGE 2 passes when every seed
has a plan within the learning budget, every seed has the same plan, and each
plan replays to the goal.

Atoms the domain uses but no predicate measures (what the gripper holds, slots
no predicate scores) are false at the start under the closed world; the record
lists them. Scores left between 0.2 and 0.8 are listed too: there the Boolean
state and robust evidence differ.

Writes the plan record and the hole spec, the input Stage 3 trains against.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from inverse_skills.envs import stow_domain as sd
from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs import stow_primitives as prim
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.literals import join, robust_value

# The plan the published experiments used, under the positive-only goal. Compared with, never
# an input to the planner.
HISTORICAL_PLAN = (
    "pick(cube_b)", "place(cube_b,src_b)", "approach(cube_a)",
    "HOLE[clear_of_walls(cube_a)]", "pick(cube_a)", "place(cube_a,src_a)",
)


def literal_dicts(literals) -> list[dict]:
    return [literal.to_dict() for literal in literals]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    for path in (args.out, args.spec_out):
        if path.exists():
            raise SystemExit(f"{path} exists; saved results are never overwritten. Choose another --out and --spec-out.")

    goal_pos, goal_neg = sd.goal_from_operator(args.operator)
    goal = join(goal_pos, goal_neg)
    domain = sd.domain()
    registry = sp.registry()
    unmeasured = sorted(hp.domain_atoms(domain) - set(registry.keys()))
    print("restoration goal G (from the extracted operator):", ", ".join(str(l) for l in sorted(goal)))
    print("world facts the domain declares")
    for d in domain.disturbances:
        print(f"  changing {d.trigger} displaces {', '.join(sorted(d.clobbers))}")
    print("atoms the domain uses but nothing measures (false at the start, closed world):", ", ".join(unmeasured))

    env = prim.make_env()
    rows, states = [], []
    for seed in args.seeds:
        state, scores = measured_state(env, seed, registry)
        states.append(state)
        result = hp.plan(domain, state, goal_pos, goal_neg)
        problems = hp.replay_problems(domain, state, result.steps, goal_pos, goal_neg) if result else ["no plan"]
        rows.append({
            "seed": seed,
            "state": sorted(state),
            "scores": scores,
            "uncertain": sorted(k for k, v in scores.items() if robust_value(v) == "uncertain"),
            "plan": list(result.actions) if result else None,
            "candidates": literal_dicts(result.candidates),
            "delegated": literal_dicts(result.delegated) if result else None,
            "hole_indices": list(result.hole_indices) if result else None,
            "cost": list(result.cost) if result else None,
            "planning_cost": list(result.planning_cost) if result else None,
            "ties": [list(a) for a in result.alternative_names] if result else None,
            "expanded": result.expanded,
            "hole_contexts": [context.to_dict() for context in result.hole_contexts] if result else None,
            "unmet_needs": None if result else literal_dicts(result.unmet_needs),
            "replay_problems": problems,
            "same_as_historical": bool(result and result.actions == HISTORICAL_PLAN),
        })
        print(f"\nseed {seed}: measured state = {', '.join(sorted(state))}")
        if rows[-1]["uncertain"]:
            print(f"  between 0.2 and 0.8 (Boolean and robust readings differ): {', '.join(rows[-1]['uncertain'])}")
        if not result:
            print(f"  no plan; unmet needs {', '.join(map(str, result.unmet_needs)) or 'none'}, "
                  f"of which lack a soft score: {', '.join(map(str, result.unsynthesizable_needs)) or 'none'}")
            continue
        for i, action in enumerate(result.actions):
            mark = "  <- learned" if i in result.hole_indices else ""
            print(f"  {i + 1}. {action}{mark}")
        print(f"  planning cost (holes, displaced, length) = {result.planning_cost}"
              f"   ties {len(result.alternatives)}   expanded {result.expanded} states"
              f"   replay: {'reaches the goal' if not problems else '; '.join(problems)}")
    env.close()

    start = states[0]
    result = hp.plan(domain, start, goal_pos, goal_neg)
    spec = hp.hole_spec(domain, start, result, goal_pos, goal_neg)
    blind = hp.plan(sd.domain(interference=False), start, goal_pos, goal_neg)
    library_only = hp.plan(domain, start, goal_pos, goal_neg, max_delegated=0)

    print("\nderived, not given")
    print("  unmet by the library alone:", ", ".join(map(str, library_only.unmet_needs)) if not library_only else "-")
    print("  delegated to learning:    ", ", ".join(map(str, result.delegated)))
    for context in result.hole_contexts:
        print(f"    {context.literal}: missing for {', '.join(context.waiting)}; first used by {context.used_by}")
    print("  hole starts from:         ", ", ".join(spec["start"]) or "nothing")
    print("  must preserve:            ", ", ".join(spec["preserve"]) or "nothing")
    print("  must keep false:          ", ", ".join(spec["preserve_absent"]) or "nothing")
    print("  handoff conditions:       ", json.dumps(spec["handoff_conditions"]))
    print("\nablations")
    print("  library only (max_delegated=0):", "no plan" if not library_only else " -> ".join(library_only.actions))
    print(f"  without the interference model: {1 + len(blind.alternatives)} orderings tie")
    for alt in (blind.actions, *blind.alternative_names):
        print("    ", " -> ".join(alt))

    plans = {tuple(row["plan"]) for row in rows if row["plan"] is not None}
    ok = (all(row["plan"] is not None and not row["replay_problems"] for row in rows) and len(plans) == 1)
    same = all(row["same_as_historical"] for row in rows)
    print(f"\nthe published plan ({' -> '.join(HISTORICAL_PLAN)}): "
          + ("the same" if same else "changed"))
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
        "library_only": {"plan": list(library_only.actions) if library_only else None,
                         "unmet_needs": None if library_only else literal_dicts(library_only.unmet_needs)},
        "delegated": literal_dicts(result.delegated),
        "hole_contexts": [context.to_dict() for context in result.hole_contexts],
        "without_interference_model": [list(blind.actions)]
                                      + [list(a) for a in blind.alternative_names],
        "seeds": rows,
    }, indent=2))
    args.spec_out.write_text(json.dumps(spec, indent=2))
    print(f"wrote {args.out} and {args.spec_out}")


if __name__ == "__main__":
    main()
