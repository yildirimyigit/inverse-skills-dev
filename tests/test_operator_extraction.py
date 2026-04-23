from __future__ import annotations

from inverse_skills.operators import OperatorExtractor, RestorationObjective
from inverse_skills.toy import build_predicate_registry, make_pick_place_rollouts


def test_operator_extraction_for_pick_place_transition() -> None:
    rollouts = make_pick_place_rollouts(num_rollouts=2)
    registry = build_predicate_registry()

    result = OperatorExtractor(registry).extract("pick_place", rollouts)
    op = result.operator

    pre = {term.key for term in op.preconditions}
    add = {term.key for term in op.add_effects}
    delete = {term.key for term in op.delete_effects}

    assert "in_region(cube,source)" in pre
    assert "in_region(cube,target)" in add
    assert "in_region(cube,source)" in delete

    objective = RestorationObjective(op, registry)
    assert objective.potential(rollouts[0].first()) > objective.potential(rollouts[0].last())
