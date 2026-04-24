from inverse_skills.operators import OperatorExtractor, RestorationObjective, ToyInversePlanner
from inverse_skills.toy import (
    PrimitiveLibrary,
    build_predicate_registry,
    make_pick_place_rollouts_executable,
    make_push_rollouts_executable,
)


def test_executable_pick_place_rollouts_capture_action_trace() -> None:
    rollouts = make_pick_place_rollouts_executable(num_rollouts=2)
    assert len(rollouts[0].scenes) == 3
    assert rollouts[0].metadata["action_trace"] == ["pick(cube)", "place(target)"]


def test_executable_push_inverse_uses_pick_place_back() -> None:
    registry = build_predicate_registry()
    rollouts = make_push_rollouts_executable(num_rollouts=3)
    extraction = OperatorExtractor(registry).extract("push_to_target", rollouts)
    objective = RestorationObjective(extraction.operator, registry)
    planner = ToyInversePlanner(PrimitiveLibrary(), objective)
    result = planner.plan(rollouts[0].last(), max_depth=3)

    assert result.success
    assert result.actions == ["pick(cube)", "place(source)"]
