"""The worked example of the 2026-11-03 explainer page ("the key and the tool"), and its table of
what can be learned, planned again with the minimal planner. Also the upstream sets behind the
Stow and Drawer holes. Planning only. The page's answers are copied in as `old_page` so the two
can be compared; they are the legacy rules' answers, not inputs.

Writes worked_example_check.json next to this file, never overwriting it.
"""

from __future__ import annotations

import json
from pathlib import Path

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import stow_domain as sd
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.exact_hole_planner import upstream
from inverse_skills.operators.hole_planner import Action, Domain

OUT = Path("artifacts/final_verification_fe374f0/worked_example_check.json")


def act(name, pre=(), add=(), delete=()) -> Action:
    return Action(name, pre=frozenset(pre), add=frozenset(add), delete=frozenset(delete))


KEY_AND_TOOL = [act("make_q", {"p"}, {"q"}, {"p"}), act("use", {"p", "q"}, {"g"})]
RESTORE = act("make_p", {"r"}, {"p"})

ROWS = [   # learnable literals, extra actions, and what the old page said the exact mode returned
    ({"p", "q", "g"}, [], "make_q > HOLE[p] > use"),
    ({"q", "g"}, [], "HOLE[q] > use"),
    ({"p", "g"}, [], "make_q > HOLE[p] > use"),
    ({"g"}, [], "no plan (Waiting)"),
    (set(), [], "no plan (no holes offered)"),
    ({"p", "q", "r", "g"}, [RESTORE], "make_q > HOLE[r] > make_p > use"),
]


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; results are never overwritten")
    rows = []
    for scored, extra, old in ROWS:
        domain = Domain(actions=KEY_AND_TOOL + extra, scored=frozenset(scored))
        result = hp.plan(domain, {"p"}, {"g"})
        answer = " > ".join(result.actions) if result else (
            "no plan, limit: " + ", ".join(result.limits_reached) if result.limits_reached else "no plan")
        rows.append({"learnable": sorted(scored), "extra_actions": [a.name for a in extra], "old_page": old,
                     "minimal_planner": answer,
                     "planning_cost": list(result.planning_cost) if result else None,
                     "feasible_with_all_holes": None if result else result.feasible_with_all_holes,
                     "changed": answer.split(",")[0] != old.split(" (")[0]})
        print(f"{str(sorted(scored)):22s} {str([a.name for a in extra]):12s} old: {old:34s} now: {answer}")

    stow_goal = sd.goal_from_operator("artifacts/final_verification_fe374f0/stow_stage1_operator.json")
    drawer_goal = dd.goal_from_operator("artifacts/final_verification_fe374f0/drawer_stage1_operator.json")
    domains = {"stow": (sd.domain(), stow_goal), "drawer": (dd.domain(), drawer_goal)}
    ups = {}
    for name, (domain, goal) in domains.items():
        needs = set(goal[0]) | {p for a in domain.actions for p in a.pre}
        candidates = needs & domain.scored
        ups[name] = {literal: sorted(upstream(domain, literal) & candidates) for literal in sorted(candidates)}
        print(name, json.dumps(ups[name], indent=1))
    OUT.write_text(json.dumps({"key_and_tool": rows, "upstream_within_candidates": ups}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
