"""Default-mode validation, Phase 2: all four extraction and planning combinations.

Every plan names its planning mode, and every goal comes from an operator Phase 1 extracted with
a named mode, so no result depends on a default. The start states are measured here with Stage 2's
own measured_state(), on Stage 2's seeds, and compared with the saved ones.

The scenarios are Stage 2's, and Stage 4's replanning with the learned action:
  stow     frontier on each seed, no interference model, library only, relearned
  drawer   frontier on each seed, library only, scripted grasp, unrestricted (relaxed only)

Each combination is compared with mean + relaxed, in three groups of fields:
  plan         whether a plan exists, its actions, hole positions, delegated literals, cost, ties
  learning     hole_spec(): what RL trains on (achieve, object, start, preserve, preserve_absent,
               prefix, suffix), then the rest of it (hole, plan, hole_index, cost)
  diagnostics  search and reporting fields, which never reach RL (expanded, candidates, gaps,
               limits_reached, hole_contexts and the no-plan report)

Run from the repository root, after Phase 1, with PYTHONNOUSERSITE=1:
  PYTHONNOUSERSITE=1 python scripts/validate_modes_planning.py stow
  PYTHONNOUSERSITE=1 python scripts/validate_modes_planning.py drawer

Writes under artifacts/default_mode_validation/phase2/ (or --out-dir), never overwriting:
  <domain>.json                                 every plan, spec and comparison
  <domain>_<extraction>_<planner>_hole_spec.json  each combination's Stage 2 learning problem
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import stow_domain as sd
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.hole_planner import Action

PHASE1 = Path("artifacts/default_mode_validation/phase1")
OUT_DIR = Path("artifacts/default_mode_validation/phase2")
CONFIGS = (("mean", "relaxed"), ("min_max", "relaxed"), ("mean", "exact"), ("min_max", "exact"))
BASELINE = "mean+relaxed"
PLAN_FIELDS = ("found", "actions", "hole_positions", "delegated", "hole_starts", "cost", "tied_plans")
BEHAVIOURAL_SPEC_FIELDS = ("achieve", "object", "start", "preserve", "preserve_absent", "prefix", "suffix")
OTHER_SPEC_FIELDS = ("hole", "plan", "hole_index", "cost")


def config_name(extraction: str, planner: str) -> str:
    return f"{extraction}+{planner}"


def replays(domain, start, steps, goal_positive, goal_negative) -> bool:
    """Every step can run in the state before it, with real transitions, and the goal holds."""
    state = start
    for step in steps:
        if not domain.applicable(state, step):
            return False
        state = domain.apply(state, step)
    return goal_positive <= state and not (goal_negative & state)


def describe(domain, start, goal, result) -> dict:
    """A plan or no-plan report as plain data, split into the groups the comparison uses."""
    goal_positive, goal_negative = goal
    diagnostics = {"result_type": type(result).__name__, "candidates": list(result.candidates),
                   "capability_gaps": list(result.capability_gaps),
                   "synthesizable_gaps": list(result.synthesizable_gaps)}
    for field in ("expanded", "limits_reached", "unmet_needs", "unsynthesizable_needs",
                  "unnecessary_holes", "replaceable_holes", "feasible_with_all_holes"):
        if hasattr(result, field):
            value = getattr(result, field)
            diagnostics[field] = list(value) if isinstance(value, tuple) else value
    if not result:
        return {"found": False, "actions": None, "hole_positions": None, "delegated": None, "hole_starts": None,
                "cost": None, "tied_plans": None, "specs": [], "diagnostics": diagnostics}
    if hasattr(result, "hole_contexts"):
        diagnostics["hole_contexts"] = [context.to_dict() for context in result.hole_contexts]
    specs = [hp.hole_spec(domain, start, result, goal_positive, goal_negative, index=i)
             for i in range(len(result.hole_indices))]
    return {"found": True, "actions": list(result.actions), "hole_positions": list(result.hole_indices),
            "delegated": list(result.delegated),
            "hole_starts": {result.steps[i].hole_for: sorted(result.steps[i].pre) for i in result.hole_indices},
            "cost": list(result.cost), "tied_plans": [list(names) for names in result.alternative_names],
            "replays_to_goal": replays(domain, start, result.steps, goal_positive, goal_negative),
            "specs": specs, "diagnostics": diagnostics}


def compare(baseline: dict, other: dict) -> dict:
    """What differs from the baseline, field by field, in each group."""
    plan = {f: {"baseline": baseline[f], "this": other[f]} for f in PLAN_FIELDS if baseline[f] != other[f]}
    behavioural, other_spec = {}, {}
    if len(baseline["specs"]) != len(other["specs"]):
        behavioural["number_of_specs"] = {"baseline": len(baseline["specs"]), "this": len(other["specs"])}
    for index, (a, b) in enumerate(zip(baseline["specs"], other["specs"])):
        for f in BEHAVIOURAL_SPEC_FIELDS:
            if a[f] != b[f]:
                behavioural[f"{index}.{f}"] = {"baseline": a[f], "this": b[f]}
        for f in OTHER_SPEC_FIELDS:
            if a[f] != b[f]:
                other_spec[f"{index}.{f}"] = {"baseline": a[f], "this": b[f]}
    left, right = baseline["diagnostics"], other["diagnostics"]
    diagnostics = {f: {"baseline": left.get(f, "<absent>"), "this": right.get(f, "<absent>")}
                   for f in sorted(set(left) | set(right)) if left.get(f, "<absent>") != right.get(f, "<absent>")}
    return {"same_plan": not plan, "same_learning_problem": not behavioural and not other_spec,
            "plan_differences": plan, "behavioural_spec_differences": behavioural,
            "other_spec_differences": other_spec, "diagnostic_fields_differing": sorted(diagnostics),
            "diagnostic_differences": diagnostics}


def goal_dict(goal) -> dict:
    return {"positive": sorted(goal[0]), "negative": sorted(goal[1])}


def measure(names_and_seeds, measured_state, make_env, registry, saved_rows) -> tuple[list, list[dict]]:
    env = make_env()
    states, rows = [], []
    saved = {row["seed"]: row for row in saved_rows}
    for seed in names_and_seeds:
        state, scores = measured_state(env, seed, registry)
        states.append(state)
        stored = saved[seed]
        differences = {k: scores[k] - stored["scores"][k] for k in scores}
        rows.append({"seed": seed, "state": sorted(state), "scores": scores,
                     "saved_state_same": sorted(state) == stored["state"],
                     "largest_score_difference_from_saved": max(abs(d) for d in differences.values())})
    env.close()
    return states, rows


def stow_scenarios(states, learned: Action) -> list[tuple]:
    rows = [(f"frontier, seed {seed}", sd.domain, state, {}) for seed, state in states]
    start = states[0][1]
    rows += [("no interference model", lambda: sd.domain(interference=False), start, {}),
             ("library only", sd.domain, start, {"max_delegated": 0}),
             ("relearned", lambda: sd.domain(extra_actions=[learned]), sd.stowed_state(), {"max_delegated": 0})]
    return rows


def drawer_scenarios(states) -> list[tuple]:
    rows = [(f"frontier, seed {seed}", dd.domain, state, {}) for seed, state in states]
    start = states[0][1]
    rows += [("library only", dd.domain, start, {"max_delegated": 0}),
             ("scripted grasp", lambda: dd.domain(extra_actions=[dd.scripted_grasp()]), start, {})]
    return rows


def stage4_other_ordering(goal, planner: str) -> dict:
    """The ordering stow_stage4_execute.py runs as its `without_interference` condition."""
    planned = hp.plan(sd.domain(), sd.stowed_state(), *goal, mode=planner)
    blind = hp.plan(sd.domain(interference=False), sd.stowed_state(), *goal, mode=planner)
    other = next(alt for alt in (blind.steps, *blind.alternatives)
                 if tuple(a.name for a in alt) != planned.actions)
    return {"actions": [a.name for a in other], "steps": [a.to_dict() for a in other]}


def library_around_the_hole(described: dict) -> list[dict]:
    """The exact planner's intent, read off its plan: library steps on both sides of the hole."""
    rows = []
    for spec in described["specs"]:
        rows.append({"hole": spec["hole"],
                     "library_before": [a["name"] for a in spec["prefix"] if a["hole_for"] is None],
                     "library_after": [a["name"] for a in spec["suffix"] if a["hole_for"] is None],
                     "prefix_and_suffix_are_all_library": all(a["hole_for"] is None for a in spec["prefix"] + spec["suffix"])})
    return rows


def run_domain(name: str) -> dict:
    if name == "stow":
        from stow_stage2_plan import measured_state

        from inverse_skills.envs import stow_predicates as predicates
        from inverse_skills.envs import stow_primitives as primitives
        domain_module, operator_part = sd, "stow"
        saved_plan = Path("artifacts/stow/stage2_plan.json")
        saved_spec = Path("artifacts/stow/hole_spec.json")
        saved_operator = Path("artifacts/stow/stage1_operator.json")
    else:
        from drawer_stage2_plan import measured_state

        from inverse_skills.envs import drawer_predicates as predicates
        from inverse_skills.envs import drawer_primitives as primitives
        domain_module, operator_part = dd, "drawer_original"
        saved_plan = Path("artifacts/drawer/stage2_plan.json")
        saved_spec = Path("artifacts/drawer/hole_spec.json")
        saved_operator = Path("artifacts/drawer/stage1_operator.json")

    saved_rows = json.loads(saved_plan.read_text())["seeds"]
    seeds = [row["seed"] for row in saved_rows]
    states, measured = measure(seeds, measured_state, primitives.make_env, predicates.registry(), saved_rows)
    seeded_states = list(zip(seeds, states))

    goals = {mode: domain_module.goal_from_operator(PHASE1 / f"{operator_part}_{mode}_operator.json")
             for mode in ("mean", "min_max")}
    stored_goal = domain_module.goal_from_operator(saved_operator)
    goal_record = {"mean": goal_dict(goals["mean"]), "min_max": goal_dict(goals["min_max"]),
                   "saved": goal_dict(stored_goal),
                   "all_same": goals["mean"] == goals["min_max"] == stored_goal}
    if name == "drawer":
        current = {mode: dd.goal_from_operator(PHASE1 / f"drawer_current_{mode}_operator.json")
                   for mode in ("mean", "min_max")}
        goal_record["from_todays_push"] = {mode: goal_dict(goal) for mode, goal in current.items()}
        goal_record["all_same"] = goal_record["all_same"] and current["mean"] == current["min_max"] == stored_goal

    learned = {}
    if name == "stow":
        record = json.loads((PHASE1 / "stow_learned.json").read_text())
        learned = {mode: Action.from_dict(record["actions"][mode]) for mode in ("mean", "min_max")}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs, spec_files = {}, {}
    for extraction, planner in CONFIGS:
        goal = goals[extraction]
        combination = config_name(extraction, planner)
        scenarios = stow_scenarios(seeded_states, learned[extraction]) if name == "stow" else drawer_scenarios(seeded_states)
        configs[combination] = {}
        for scenario, make_domain, start, limits in scenarios:
            domain = make_domain()
            result = hp.plan(domain, start, *goal, mode=planner, **limits)
            configs[combination][scenario] = describe(domain, start, goal, result)

        # Stage 2's learning problem: the frontier plan from the first measured state.
        domain = domain_module.domain()
        frontier = hp.plan(domain, states[0], *goal, mode=planner)
        spec = hp.hole_spec(domain, states[0], frontier, *goal)
        path = OUT_DIR / f"{name}_{extraction}_{planner}_hole_spec.json"
        path.write_text(json.dumps(spec, indent=2))
        spec_files[combination] = {"file": str(path), "bytes_equal_saved": path.read_text() == saved_spec.read_text()}

    comparisons = {combination: {scenario: compare(configs[BASELINE][scenario], configs[combination][scenario])
                                 for scenario in configs[BASELINE]}
                   for combination in configs if combination != BASELINE}

    output = {
        "made_by": f"PYTHONNOUSERSITE=1 python scripts/validate_modes_planning.py {name}",
        "seeds": seeds, "measured": measured, "goals": goal_record, "configs": configs,
        "comparisons_with_mean_relaxed": comparisons, "stage2_spec_files": spec_files,
        "saved_spec": str(saved_spec),
        "exact_keeps_library_around_the_hole": {
            combination: library_around_the_hole(configs[combination][f"frontier, seed {seeds[0]}"])
            for combination in configs if combination.endswith("exact")},
    }
    if name == "stow":
        output["stage4_without_interference_ordering"] = {
            config_name(e, p): stage4_other_ordering(goals[e], p) for e, p in CONFIGS}
        output["learned_actions"] = {mode: action.to_dict() for mode, action in learned.items()}
    else:
        output["direct_baseline"] = direct_baseline(goals, states[0])
    return output


def direct_baseline(goals, start) -> dict:
    """The unrestricted ablation, which only the relaxed mode has, under each extraction."""
    saved = Path("artifacts/drawer/direct/hole_spec_direct.json")
    rows = {}
    for mode, goal in goals.items():
        domain = dd.domain()
        direct = hp.plan(domain, start, *goal, mode="relaxed", unrestricted_holes=True)
        spec = hp.hole_spec(domain, start, direct, *goal)
        rows[mode] = {"plan": list(direct.actions), "cost": list(direct.cost),
                      "spec_bytes_equal_saved": json.dumps(spec, indent=2) == saved.read_text()}
    try:
        hp.plan(dd.domain(), start, *goals["min_max"], mode="exact", unrestricted_holes=True)
        rows["exact_mode"] = "accepted"
    except ValueError as error:
        rows["exact_mode"] = f"refused: {error}"
    return rows


def show(name: str, output: dict) -> None:
    print(f"\n{name}: seeds {output['seeds']}")
    for row in output["measured"]:
        print(f"  seed {row['seed']}: {', '.join(row['state'])} | same as saved {row['saved_state_same']}, "
              f"largest score difference {row['largest_score_difference_from_saved']:.3g}")
    print(f"  goals all the same: {output['goals']['all_same']}: {output['goals']['mean']}")
    baseline = output["configs"][BASELINE]
    for scenario, row in baseline.items():
        print(f"\n  [{scenario}] {' > '.join(row['actions']) if row['found'] else 'no plan'}  cost {row['cost']}")
        for combination, comparisons in output["comparisons_with_mean_relaxed"].items():
            c = comparisons[scenario]
            print(f"    {combination:16s} plan same {c['same_plan']!s:5}  learning same {c['same_learning_problem']!s:5}  "
                  f"diagnostics differ: {', '.join(c['diagnostic_fields_differing']) or 'none'}")
            for group in ("plan_differences", "behavioural_spec_differences", "other_spec_differences"):
                for field, values in c[group].items():
                    print(f"      {group} {field}: {values['baseline']} -> {values['this']}")
    print("\n  Stage 2 spec files, against the saved one:")
    for combination, row in output["stage2_spec_files"].items():
        print(f"    {combination:16s} bytes equal saved: {row['bytes_equal_saved']}")
    print("\n  library around the hole (exact):")
    for combination, rows in output["exact_keeps_library_around_the_hole"].items():
        print(f"    {combination:16s} {rows}")
    if "stage4_without_interference_ordering" in output:
        print("\n  Stage 4 without_interference ordering:")
        for combination, row in output["stage4_without_interference_ordering"].items():
            print(f"    {combination:16s} {' > '.join(row['actions'])}")
    if "direct_baseline" in output:
        print(f"\n  direct baseline: {output['direct_baseline']}")


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("domain", choices=("stow", "drawer"))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    OUT_DIR = args.out_dir
    target = OUT_DIR / f"{args.domain}.json"
    if target.exists():
        raise SystemExit(f"{target} exists; outputs are never overwritten")
    output = run_domain(args.domain)
    target.write_text(json.dumps(output, indent=2))
    show(args.domain, output)
    print(f"\nwrote {target}")


if __name__ == "__main__":
    main()
