from __future__ import annotations

import json
from pathlib import Path

from inverse_skills.operators import OperatorExtractor, RestorationObjective, ToyInversePlanner
from inverse_skills.toy import PrimitiveLibrary, build_predicate_registry, make_pick_place_rollouts, make_push_rollouts


def run_case(skill_name: str, rollouts) -> dict:
    registry = build_predicate_registry()
    extraction = OperatorExtractor(registry).extract(skill_name, rollouts)
    objective = RestorationObjective(extraction.operator, registry)
    planner = ToyInversePlanner(PrimitiveLibrary(), objective)
    result = planner.plan(rollouts[0].last(), max_depth=3)

    summary = {
        "skill_name": skill_name,
        "operator": extraction.operator.to_dict(),
        "inverse_potential_at_forward_final": objective.potential(rollouts[0].last()),
        "inverse_potential_at_forward_start": objective.potential(rollouts[0].first()),
        "plan_success": result.success,
        "plan_actions": result.actions,
        "plan_final_potential": result.final_potential,
        "expanded_nodes": result.expanded_nodes,
    }
    return summary


def main() -> None:
    outputs = {
        "pick_place": run_case("pick_place", make_pick_place_rollouts()),
        "push_to_target": run_case("push_to_target", make_push_rollouts()),
    }

    print(json.dumps(outputs, indent=2))
    out = Path("artifacts/planrob_toy_demo.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
