from __future__ import annotations

from inverse_skills.operators import OperatorExtractor, RestorationObjective, ToyInversePlanner
from inverse_skills.toy import PrimitiveLibrary, build_predicate_registry, make_push_rollouts


def test_push_inverse_can_be_realized_with_pick_place_primitives() -> None:
    rollouts = make_push_rollouts(num_rollouts=2)
    registry = build_predicate_registry()

    extracted = OperatorExtractor(registry).extract("push_to_target", rollouts)
    objective = RestorationObjective(extracted.operator, registry)
    planner = ToyInversePlanner(PrimitiveLibrary(), objective)

    result = planner.plan(rollouts[0].last(), max_depth=3)

    assert result.success
    assert result.actions == ["pick(cube)", "place(source)"]
    assert result.final_potential >= 0.98
