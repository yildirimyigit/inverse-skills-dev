"""Run the planner on a complete library and an incomplete one, and explain both.

Same scene, same goal, two libraries:

  incomplete   approach / pick / place / push — nothing pulls a cube toward the
               robot, so `clear_of_walls(cube_a)` cannot be established and the
               planner must delegate it to a hole
  complete     the same library plus an un-wedging `drag(cube_a)` primitive,
               which is exactly the operator the hole becomes once it is learned

For each, prints the delete-relaxed reachability proof, the plan, its cost, and
a step-by-step trace of the symbolic state. Writes artifacts/stow/planner_demo.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inverse_skills.envs import stow_domain as sd
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.hole_planner import Action, Domain

DRAG = Action(
    name="drag(cube_a)",
    pre=frozenset({"tcp_near(cube_a)"}),
    add=frozenset({"clear_of_walls(cube_a)"}),
    delete=frozenset({"gripper_open()"}),
)


def relaxed_reachable(domain: Domain, start: frozenset[str]) -> set[str]:
    """Every literal reachable ignoring delete effects.

    If a goal literal is missing from this set it is unreachable outright, since
    dropping deletes only ever makes more things possible. That is what makes
    "this library cannot do it" a proof rather than a failed search.
    """
    reached = set(start)
    changed = True
    while changed:
        changed = False
        for action in domain.actions:
            if action.pre <= reached:
                for literal in action.add:
                    if literal not in reached:
                        reached.add(literal)
                        reached |= domain.implications.get(literal, set())
                        changed = True
    return reached


def adders(domain: Domain, literal: str) -> list[Action]:
    return [a for a in domain.actions if literal in a.add
            or literal in set().union(*(domain.implications.get(x, set()) for x in a.add))
            if a.add]


def trace(domain: Domain, start: frozenset[str], plan: list[str],
          goal_pos: set[str], goal_neg: set[str]) -> list[dict]:
    """Replay the plan, recording what each step needed and what it changed."""
    by_name = {a.name: a for a in domain.actions}
    for key in domain.hole_candidates:
        hole = domain.hole(key)
        by_name[hole.name] = hole
    state = start
    rows = []
    for name in plan:
        action = by_name[name]
        nxt = domain.apply(state, action)
        remaining = sorted((goal_pos - nxt) | (goal_neg & nxt))
        rows.append({
            "action": name,
            "is_hole": action.hole_for is not None,
            "requires": sorted(action.pre),
            "requires_absent": sorted(action.pre_absent),
            "adds": sorted(nxt - state),
            "removes": sorted(state - nxt),
            "state_after": sorted(nxt),
            "goal_remaining": remaining,
        })
        state = nxt
    return rows


def describe(name: str, domain: Domain, start: frozenset[str],
             goal_pos: set[str], goal_neg: set[str]) -> dict:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"library: {len(domain.actions)} ground actions "
          f"({', '.join(sorted({a.name.split('(')[0] for a in domain.actions}))})")
    print(f"start:   {', '.join(sorted(start))}")
    print(f"goal:    restore {', '.join(sorted(goal_pos))}")
    print(f"         remove  {', '.join(sorted(goal_neg))}")

    reachable = relaxed_reachable(domain, start)
    missing = sorted(goal_pos - reachable)
    print("\ndelete-relaxed reachability (what the library could ever establish):")
    if missing:
        print(f"  unreachable: {', '.join(missing)}")
        for literal in missing:
            print(f"  nothing in the library adds {literal} from a reachable state:")
            for action in adders(domain, literal):
                unmet = sorted(action.pre - reachable)
                print(f"    {action.name:28s} needs {', '.join(sorted(action.pre)) or '-'}"
                      + (f"   (unreachable: {', '.join(unmet)})" if unmet else ""))
    else:
        print("  every goal literal is reachable — no hole is needed")

    hole_free = hp.plan(domain, start, goal_pos, goal_neg, max_delegated=0)
    print(f"\nwith no holes allowed: {' -> '.join(hole_free.actions) if hole_free else 'NO PLAN'}")

    result = hp.plan(domain, start, goal_pos, goal_neg)
    rows = trace(domain, start, list(result.actions), goal_pos, goal_neg)
    print(f"\nplan ({len(result.actions)} steps, cost "
          f"[delegated {result.cost[0]}, hole literals {result.cost[1]}, "
          f"holes {result.cost[2]}, length {result.cost[3]}], "
          f"{result.expanded} states expanded):")
    for i, row in enumerate(rows, 1):
        tag = "  <- learned" if row["is_hole"] else ""
        print(f"\n  {i}. {row['action']}{tag}")
        if row["requires"]:
            print(f"       needs:   {', '.join(row['requires'])}")
        if row["adds"]:
            print(f"       adds:    {', '.join(row['adds'])}")
        if row["removes"]:
            print(f"       removes: {', '.join(row['removes'])}")
        print(f"       goal still missing: "
              f"{', '.join(row['goal_remaining']) if row['goal_remaining'] else 'nothing'}")

    return {
        "library_size": len(domain.actions),
        "start": sorted(start),
        "unreachable_goal_literals": missing,
        "hole_free_plan": list(hole_free.actions) if hole_free else None,
        "plan": list(result.actions),
        "cost": list(result.cost),
        "delegated": list(result.delegated),
        "hole_indices": list(result.hole_indices),
        "expanded": result.expanded,
        "alternatives": [list(a) for a in result.alternatives],
        "trace": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--operator", type=Path, default=Path("artifacts/stow/stage1_operator.json"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/stow/planner_demo.json"))
    args = ap.parse_args()

    goal_pos, goal_neg = sd.goal_from_operator(args.operator)
    start = sd.stowed_state()

    cases = {
        "incomplete": describe("INCOMPLETE LIBRARY — nothing can free the wedged cube",
                               sd.domain(), start, goal_pos, goal_neg),
        "complete": describe("COMPLETE LIBRARY — an un-wedging primitive exists",
                             sd.domain(extra_actions=[DRAG]), start, goal_pos, goal_neg),
    }

    print(f"\n{'=' * 78}\nside by side\n{'=' * 78}")
    print(f"{'':14s} {'steps':>6} {'holes':>6} {'delegated':>10} {'expanded':>9}")
    for name, case in cases.items():
        print(f"{name:14s} {case['cost'][3]:>6} {case['cost'][2]:>6} "
              f"{case['cost'][0]:>10} {case['expanded']:>9}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"goal_positive": sorted(goal_pos),
                                    "goal_negative": sorted(goal_neg),
                                    "cases": cases}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
