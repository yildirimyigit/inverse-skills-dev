"""Both planning modes on the same problems, side by side. Planning only: no simulation, no RL.

Each scenario hands both modes the same domain, start, goal and limits. For each mode it reports
whether a plan was found, the holes and their positions, the plan, its cost, the search nodes
expanded and the planning time. The time is the median of repeated runs, and it is the only value
that differs from one run to the next. Every plan is also replayed with the domain's own
transition model.

It also says, for each scenario, whether the two modes would give a hole the same start. Where
they would not, the comparison is not perfectly controlled, and the scenario says so.

Stow and the drawer use the goals extracted in Stage 1 and the start states Stage 2 measured.

Run from the repository root, with PYTHONNOUSERSITE=1 (stray ~/.local packages break ManiSkill):
  PYTHONNOUSERSITE=1 python scripts/compare_hole_planners.py [--out comparison.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import stow_domain as sd
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.exact_hole_planner import hole_starts, library_states
from inverse_skills.operators.hole_planner import Action, Domain

REPEATS = 7
STOW_OPERATOR = "artifacts/stow/stage1_operator.json"
DRAWER_OPERATOR = "artifacts/drawer/stage1_operator.json"


def act(name, pre=(), add=(), delete=(), absent=()) -> Action:
    return Action(name, pre=frozenset(pre), add=frozenset(add), delete=frozenset(delete),
                  pre_absent=frozenset(absent))


def scenarios() -> list[dict]:
    """name, domain, start, goal literals and any limits. Each domain is built fresh."""
    stow_goal = sd.goal_from_operator(STOW_OPERATOR)
    drawer_goal = dd.goal_from_operator(DRAWER_OPERATOR)
    shut = frozenset({dd.CLOSED})

    def stow(name, domain, **limits):
        return {"name": name, "domain": domain, "start": sd.stowed_state(), "goal": stow_goal, "limits": limits}

    def drawer(name, domain, **limits):
        return {"name": name, "domain": domain, "start": shut, "goal": drawer_goal, "limits": limits}

    def small(name, actions, scored, start=(), positive=("g",), negative=(), **fields):
        return {"name": name, "domain": Domain(actions=actions, scored=frozenset(scored), **fields),
                "start": frozenset(start), "goal": (set(positive), set(negative)), "limits": {}}

    stow_unscored = sd.domain()
    stow_unscored.scored = stow_unscored.scored - {"clear_of_walls(cube_a)"}
    drawer_no_pull = dd.domain()
    drawer_no_pull.actions = [a for a in drawer_no_pull.actions if a.name != "pull_open(drawer)"]
    make_q_use = [act("make_q", {"p"}, {"q"}, {"p"}), act("use", {"p", "q"}, {"g"})]

    return [
        stow("Stow", sd.domain()),
        stow("Stow, no interference model", sd.domain(interference=False)),
        stow("Stow, library alone", sd.domain(), max_delegated=0),
        stow("Stow, clear_of_walls has no soft score", stow_unscored),
        drawer("Drawer", dd.domain()),
        drawer("Drawer, library alone", dd.domain(), max_delegated=0),
        drawer("Drawer, scripted grasp in the library", dd.domain(extra_actions=[dd.scripted_grasp()])),
        drawer("Drawer, no pull_open in the library", drawer_no_pull),
        small("Delete interaction (make_q, use)", make_q_use, {"p", "q", "g"}, {"p"}),
        small("Delete interaction, p has no soft score", make_q_use, {"q", "g"}, {"p"}),
        small("Delete interaction, q has no soft score", make_q_use, {"p", "g"}, {"p"}),
        small("Delete interaction, only g has a soft score", make_q_use, {"g"}, {"p"}),
        small("Consumed shared token",
              [act("make_p", {"token"}, {"p"}, {"token"}), act("make_q", {"token"}, {"q"}, {"token"})],
              {"token", "p", "q"}, {"token"}, ("p", "q")),
        small("Producer destroys part of the goal",
              [act("use_p", {"p"}, {"g"}), act("make_p", (), {"p"}, {"z"})],
              {"p", "g"}, {"z"}, ("g", "z")),
        small("Chain of prerequisites", [act("use_p", {"p"}, {"g"}), act("make_p", {"r"}, {"p"})], {"p", "r", "g"}),
        small("Conjunctive prerequisites", [act("make", {"q", "r"}, {"g"})], {"g", "q", "r"}),
        small("Pure cycle", [act("g_from_q", {"q"}, {"g"}), act("q_from_g", {"g"}, {"q"})], {"g", "q"}),
        small("Cycle with a way in",
              [act("g_from_q", {"q"}, {"g"}), act("q_from_g", {"g"}, {"q"}), act("q_from_r", {"r"}, {"q"})],
              {"g", "q", "r"}),
        small("Negative goal needs a remover", [act("clear_x", {"k"}, (), {"x"})], {"k", "x"}, {"x"}, (), ("x",)),
        small("Restoring a consumed fact needs r",
              [*make_q_use, act("make_p", {"r"}, {"p"})], {"p", "q", "r", "g"}, {"p"}),
        small("Restoring a consumed fact cannot be learned",
              [*make_q_use, act("make_p", {"r"}, {"p"})], {"q", "g"}, {"p"}),
        small("Goal producers destroy each other",
              [act("make_g", {"s"}, {"g"}, {"s", "h"}), act("make_h", {"s"}, {"h"}, {"s", "g"})],
              {"g", "h", "s"}, {"s"}, ("g", "h")),
        small("A prerequisite nothing can establish",
              [act("use_p", {"p"}, {"g"}), act("make_p", {"r"}, {"p"})], {"p"}),
        small("Prerequisites never held together",
              [act("make_m", (), {"m"}, {"n"}), act("make_p", {"m", "n"}, {"p"}), act("use", {"p"}, {"g"})],
              {"p"}, {"n"}),
        small("Hole start: facts that exclude each other",
              [act("go_near", add={"near(o)"}), act("go_far", add={"far(o)"})], {"g(o)"}, (), ("g(o)",),
              robot_relative=frozenset({"near(o)", "far(o)"}),
              mutex_groups=[frozenset({"near(o)", "far(o)"})]),
        small("Hole start: a fact only the relaxation reaches",
              [act("make_b", {"a"}, {"b"}, {"a"}), act("prep", {"a", "b"}, {"near(o)"})], {"g(o)"}, {"a"}, ("g(o)",),
              robot_relative=frozenset({"near(o)"})),
    ]


def plan_replays_to_goal(domain, start, steps, goal_positive, goal_negative) -> bool:
    state = start
    for step in steps:
        if not domain.applicable(state, step):
            return False
        state = domain.apply(state, step)
    return goal_positive <= state and not (goal_negative & state)


def run_mode(scenario: dict, mode: str) -> dict:
    domain, start = scenario["domain"], scenario["start"]
    positive, negative = scenario["goal"]
    times = []
    for _ in range(REPEATS):
        began = time.perf_counter()
        result = hp.plan(domain, start, positive, negative, mode=mode, **scenario["limits"])
        times.append((time.perf_counter() - began) * 1000)
    row = {"found": bool(result), "runtime_ms": statistics.median(times)}
    if result:
        row.update(
            actions=list(result.actions),
            holes=[{"literal": result.steps[i].hole_for, "position": i, "start": sorted(result.steps[i].pre)}
                   for i in result.hole_indices],
            cost=list(result.cost),
            expanded=result.expanded,
            tied_plans_listed=len(result.alternatives),
            replays_to_goal=plan_replays_to_goal(domain, start, result.steps, positive, negative),
        )
        if mode == "exact":
            row["hole_contexts"] = [context.to_dict() for context in result.hole_contexts]
            row["limits_reached"] = list(result.limits_reached)
    elif mode == "exact":
        row.update(expanded=result.expanded, limits_reached=list(result.limits_reached),
                   unmet_needs=list(result.unmet_needs),
                   unsynthesizable_needs=list(result.unsynthesizable_needs),
                   unnecessary_holes=list(result.unnecessary_holes),
                   replaceable_holes=list(result.replaceable_holes),
                   feasible_with_all_holes=result.feasible_with_all_holes)
    else:
        row.update(expanded=None, capability_gaps=list(result.capability_gaps),
                   unsynthesizable_gaps=list(result.unsynthesizable_gaps))
    return row


def hole_start_differences(scenario: dict) -> dict:
    """Every scored literal whose hole would start differently in the two modes."""
    domain, start = scenario["domain"], scenario["start"]
    reachable = hp.relaxed_reachable(domain, start)
    states = library_states(domain, start)
    differences = {}
    for key in sorted(domain.scored):
        relaxed = sorted(domain.hole(key, reachable).pre)
        exact = [sorted(hole.pre) for hole in hole_starts(domain, key, states)]
        if exact != [relaxed]:
            differences[key] = {"relaxed": relaxed, "exact": exact}
    return differences


def hole_start_note(differences: dict, relaxed: dict, exact: dict) -> str:
    if not differences:
        return "hole starts: same in both modes"
    # A plan that failed may have failed because of the start it would have had.
    in_a_plan = {hole["literal"] for row in (relaxed, exact) for hole in row.get("holes", ())}
    if in_a_plan & set(differences) or not (relaxed["found"] and exact["found"]):
        return f"hole starts: DIFFER, so this is not a perfectly controlled comparison: {differences}"
    return f"hole starts: differ only for holes neither plan uses: {differences}"


def describe(row: dict) -> str:
    if row["found"]:
        holes = ", ".join(f"{h['literal']}@{h['position']}" for h in row["holes"]) or "none"
        return (f"{' > '.join(row['actions']) or '(already satisfied)'}\n"
                f"           cost {tuple(row['cost'])}  holes {holes}  ties listed {row['tied_plans_listed']}  "
                f"replays {row['replays_to_goal']}")
    if "limits_reached" in row:
        return (f"NO PLAN within the limits  limits reached {row['limits_reached'] or 'none'}  "
                f"no soft score {row['unsynthesizable_needs'] or 'none'}  "
                f"not needed {row['unnecessary_holes'] or 'none'}  "
                f"refused by the causal preference {row['replaceable_holes'] or 'none'}  "
                f"solvable with every hole {row['feasible_with_all_holes']}")
    return (f"NO PLAN  capability gaps {row['capability_gaps'] or 'none'}  "
            f"no soft score {row['unsynthesizable_gaps'] or 'none'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="also write the results as JSON here; never overwrites")
    args = ap.parse_args()
    if args.out and args.out.exists():
        raise SystemExit(f"{args.out} exists; outputs are never overwritten")

    rows = []
    for scenario in scenarios():
        relaxed, exact = run_mode(scenario, "relaxed"), run_mode(scenario, "exact")
        differences = hole_start_differences(scenario)
        same_plan = relaxed["found"] and exact["found"] and relaxed["actions"] == exact["actions"]
        rows.append({"scenario": scenario["name"], "limits": scenario["limits"],
                     "start": sorted(scenario["start"]),
                     "goal_positive": sorted(scenario["goal"][0]), "goal_negative": sorted(scenario["goal"][1]),
                     "relaxed": relaxed, "exact": exact, "same_plan": same_plan,
                     "hole_start_differences": differences})

        print(f"\n{scenario['name']}" + (f"  {scenario['limits']}" if scenario["limits"] else ""))
        for mode, row in (("relaxed", relaxed), ("exact", exact)):
            print(f"  {mode:8s} {describe(row)}")
            print(f"           expanded {row['expanded'] if row['expanded'] is not None else 'n/a'}  "
                  f"{row['runtime_ms']:.2f} ms")
        print(" ", hole_start_note(differences, relaxed, exact))

    only_exact = [r["scenario"] for r in rows if r["exact"]["found"] and not r["relaxed"]["found"]]
    only_relaxed = [r["scenario"] for r in rows if r["relaxed"]["found"] and not r["exact"]["found"]]
    both_differ = [r["scenario"] for r in rows if r["relaxed"]["found"] and r["exact"]["found"] and not r["same_plan"]]
    uneven = [r["scenario"] for r in rows if r["hole_start_differences"]]
    print(f"\n{len(rows)} scenarios: {sum(r['same_plan'] for r in rows)} with the same plan, "
          f"{sum(not r['relaxed']['found'] and not r['exact']['found'] for r in rows)} with no plan in either")
    print("  a plan only in exact mode:  ", only_exact or "none")
    print("  a plan only in relaxed mode:", only_relaxed or "none")
    print("  plans differ:               ", both_differ or "none")
    print("  hole starts differ:         ", uneven or "none")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"made_by": "python scripts/compare_hole_planners.py", "repeats": REPEATS,
                                        "scenarios": rows}, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
