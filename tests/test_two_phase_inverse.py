from inverse_skills.operators import (
    OperatorExtractor,
    ResidualInverseObjective,
    RestorationObjective,
    ToyInversePlanner,
    two_phase_inverse,
)
from inverse_skills.toy.domains import build_predicate_registry
from inverse_skills.toy.generators import (
    make_pick_place_rollouts_executable,
    make_push_rollouts_executable,
)
from inverse_skills.toy.primitives import PrimitiveLibrary


def _planner_for(rollouts):
    skill = rollouts[0].skill_name
    registry = build_predicate_registry()
    operator = OperatorExtractor(registry).extract(skill, rollouts).operator
    objective = RestorationObjective(operator, registry)
    return ToyInversePlanner(PrimitiveLibrary(), objective)


def test_two_phase_fully_solved_has_empty_residual() -> None:
    rollouts = make_pick_place_rollouts_executable()
    planner = _planner_for(rollouts)
    result = two_phase_inverse(planner, rollouts[0].last(), max_depth=3)

    assert result.fully_solved is True
    assert result.residual_term_keys == []
    assert result.symbolic_prefix == ["pick(cube)", "place(source)"]
    assert result.gap_remaining_for_rl < 0.01
    assert result.gap_closed_by_symbolic > 0.5


def test_two_phase_depth_limited_partial_recovery() -> None:
    """At max_depth=1, BFS cannot complete the 2-step inverse for push.
    Residual must contain the in_region terms BFS cannot satisfy."""
    rollouts = make_push_rollouts_executable()
    planner = _planner_for(rollouts)
    result = two_phase_inverse(planner, rollouts[0].last(), max_depth=1)

    assert result.fully_solved is False
    assert "in_region(cube,source)" in result.residual_term_keys
    assert "in_region(cube,target)" in result.residual_term_keys
    assert result.gap_closed_by_symbolic < 0.01
    assert result.gap_remaining_for_rl > 0.5


def test_v_gap_decomposition_is_consistent() -> None:
    rollouts = make_push_rollouts_executable()
    planner = _planner_for(rollouts)
    result = two_phase_inverse(planner, rollouts[0].last(), max_depth=1)

    assert abs(
        result.gap_closed_by_symbolic + result.gap_remaining_for_rl - result.gap_total
    ) < 1e-6


def test_residual_objective_filters_terms() -> None:
    rollouts = make_push_rollouts_executable()
    planner = _planner_for(rollouts)
    result = two_phase_inverse(planner, rollouts[0].last(), max_depth=1)

    assert isinstance(result.residual_objective, ResidualInverseObjective)
    base_terms = {t.key for t in planner.objective.terms}
    residual_terms = {t.key for t in result.residual_objective.terms}
    assert residual_terms.issubset(base_terms)
    assert residual_terms == set(result.residual_term_keys)


def test_residual_potential_at_handoff_is_nontrivial() -> None:
    """The residual reward at the handoff state should be < 1.0 — there is
    real work remaining for an RL agent."""
    rollouts = make_push_rollouts_executable()
    planner = _planner_for(rollouts)
    result = two_phase_inverse(planner, rollouts[0].last(), max_depth=1)

    residual_v = result.residual_objective.potential(result.handoff_scene)
    assert residual_v < 0.5


def test_pose_precise_partial_recovery_realistic_case() -> None:
    """The discrete primitive `place(source)` teleports the cube to the region
    center; if the inverse target requires a precise pose offset from the
    center, BFS provably cannot satisfy it at any depth.  The residual must
    surface the at_pose predicate as the structurally unreachable gap."""
    from inverse_skills.planrob_bundle import _build_pose_precise_push_case

    skill, rollouts, registry = _build_pose_precise_push_case()
    operator = OperatorExtractor(registry).extract(skill, rollouts).operator
    objective = RestorationObjective(operator, registry)
    planner = ToyInversePlanner(PrimitiveLibrary(), objective)
    result = two_phase_inverse(planner, rollouts[0].last(), max_depth=3)

    assert result.fully_solved is False
    assert result.symbolic_prefix == ["pick(cube)", "place(source)"]
    assert "at_pose(cube,target_pose)" in result.residual_term_keys
    assert result.term_max_scores["at_pose(cube,target_pose)"] < 0.01
    assert result.gap_closed_by_symbolic > 0.4
    assert result.gap_remaining_for_rl > 0.2
