from __future__ import annotations

import numpy as np
import pytest

from inverse_skills.core.scene import RobotState, SceneGraph
from inverse_skills.logging.rollout import ForwardRollout
from inverse_skills.operators import (
    OperatorExtractionConfig,
    OperatorExtractor,
    RestorationObjective,
)
from inverse_skills.predicates.base import Predicate, PredicateRegistry, PredicateResult
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


# --- Mean versus min/max extraction, on demonstrations whose scores the test states ---


class _Scripted(Predicate):
    """A predicate named `key` that reports whatever score the test wrote into the scene."""

    def __init__(self, key: str):
        self.name, self.args, self.weight = key, (), 1.0

    @property
    def key(self) -> str:
        return self.name

    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        score = scene.metadata["scores"][self.key]
        # The score is 0.5 * (1 + tanh(margin / T)); with T = 1 this margin gives it back.
        return PredicateResult(self.name, self.args, margin=float(np.arctanh(2 * score - 1)), temperature=1.0)


def _scene(scores: dict[str, float]) -> SceneGraph:
    return SceneGraph(timestep=0, robot=RobotState(q=np.zeros(7), gripper_width=0.0), objects={},
                      metadata={"scores": scores})


def _extract(table: dict[str, list[tuple[float, float]]], mode: str = "mean"):
    """Extract from one demonstration per column: table[p][i] = (start score, end score)
    of predicate p in demonstration i. Returns the Pre, Add and Del keys and the result."""
    count = len(next(iter(table.values())))
    rollouts = [
        ForwardRollout(skill_name="skill", demo_id=f"demo_{i}", scenes=[
            _scene({p: pairs[i][0] for p, pairs in table.items()}),
            _scene({p: pairs[i][1] for p, pairs in table.items()}),
        ])
        for i in range(count)
    ]
    registry = PredicateRegistry(_Scripted(p) for p in table)
    result = OperatorExtractor(registry, OperatorExtractionConfig(mode=mode)).extract("skill", rollouts)
    op = result.operator
    pre = {t.key for t in op.preconditions}
    add = {t.key for t in op.add_effects}
    delete = {t.key for t in op.delete_effects}
    return pre, add, delete, result


def test_a_predicate_true_at_the_start_of_every_demonstration_is_a_precondition_in_both_modes() -> None:
    table = {"p": [(0.95, 0.95), (0.9, 0.9), (0.99, 0.99), (0.85, 0.85), (0.97, 0.97)]}
    for mode in ("mean", "min_max"):
        pre, add, delete, _ = _extract(table, mode)
        assert (pre, add, delete) == ({"p"}, set(), set()), mode


def test_a_predicate_false_at_the_start_of_one_demonstration_is_a_precondition_only_by_the_mean() -> None:
    # Average 0.812 passes 0.8, but the last demonstration starts with p false.
    table = {"p": [(0.99, 0.99)] * 4 + [(0.1, 0.1)]}
    assert _extract(table, "mean")[0] == {"p"}
    assert _extract(table, "min_max")[0] == set()


def test_a_consistent_false_to_true_transition_is_an_add_effect_in_both_modes() -> None:
    table = {"p": [(0.05, 0.95), (0.1, 0.9), (0.0001, 0.99), (0.15, 0.85), (0.02, 0.97)]}
    for mode in ("mean", "min_max"):
        pre, add, delete, _ = _extract(table, mode)
        assert (pre, add, delete) == (set(), {"p"}, set()), mode


def test_a_consistent_true_to_false_transition_is_a_precondition_and_a_delete_effect_in_both_modes() -> None:
    table = {"p": [(0.95, 0.05), (0.9, 0.1), (0.99, 0.0001), (0.85, 0.15), (0.97, 0.02)]}
    for mode in ("mean", "min_max"):
        pre, add, delete, _ = _extract(table, mode)
        assert (pre, add, delete) == ({"p"}, set(), {"p"}), mode


def test_a_transition_one_demonstration_leaves_uncertain_is_an_effect_only_by_the_mean() -> None:
    # In the last demonstration, p and q both end at 0.5: neither true nor false.
    # So not every demonstration shows p added or q deleted.
    table = {"p": [(0.05, 0.95)] * 4 + [(0.05, 0.5)],
             "q": [(0.95, 0.05)] * 4 + [(0.95, 0.5)]}
    _, add, delete, _ = _extract(table, "mean")
    assert (add, delete) == ({"p"}, {"q"})
    _, add, delete, result = _extract(table, "min_max")
    assert (add, delete) == (set(), set())
    assert result.scores["p"]["end_min"] == pytest.approx(0.5)
    assert result.scores["q"]["end_max"] == pytest.approx(0.5)


def test_an_add_effect_that_already_held_before_one_demonstration_is_dropped_by_min_max() -> None:
    # The STRIPS case the strict rule cannot express: p was already true before the
    # last demonstration, so not every demonstration shows the false-to-true change.
    table = {"p": [(0.05, 0.95)] * 4 + [(0.9, 0.95)]}
    assert _extract(table, "mean")[1] == {"p"}
    _, add, _, result = _extract(table, "min_max")
    assert add == set()
    assert result.scores["p"]["start_max"] == pytest.approx(0.9)


def test_min_max_terms_carry_the_weakest_evidence_and_their_own_extractor_name() -> None:
    table = {"p": [(0.9, 0.1), (0.95, 0.05)], "q": [(0.2, 0.8), (0.1, 0.9)]}
    _, _, _, result = _extract(table, "min_max")
    op = result.operator
    assert [(t.key, t.weight) for t in op.preconditions] == [("p", pytest.approx(0.9))]
    assert [(t.key, t.weight) for t in op.delete_effects] == [("p", pytest.approx(0.9 - 0.1))]
    assert [(t.key, t.weight) for t in op.add_effects] == [("q", pytest.approx(0.8 - 0.2))]
    assert op.metadata == {"num_rollouts": 2, "extractor": "first_last_score_min_max_v0"}


def test_the_mean_mode_is_the_default_and_unchanged() -> None:
    assert OperatorExtractionConfig().mode == "mean"
    # Hand-computed from the averaging rule: Pre if the start average >= 0.8;
    # Add / Del if the average rises / falls by >= 0.35.
    table = {"p": [(0.99, 0.99)] * 4 + [(0.1, 0.1)],    # start average 0.812
             "q": [(0.05, 0.95)] * 4 + [(0.9, 0.95)],   # 0.22 -> 0.95
             "r": [(0.95, 0.05)] * 4 + [(0.95, 0.5)]}   # 0.95 -> 0.14
    pre, add, delete, result = _extract(table)
    assert (pre, add, delete) == ({"p", "r"}, {"q"}, {"r"})
    assert result.scores["q"] == pytest.approx({"start_mean": 0.22, "end_mean": 0.95, "delta": 0.73, "abs_delta": 0.73})
    op = result.operator
    assert [(t.key, t.weight) for t in op.preconditions] == [("p", pytest.approx(0.812)), ("r", pytest.approx(0.95))]
    assert [(t.key, t.weight) for t in op.add_effects] == [("q", pytest.approx(0.73))]
    assert [(t.key, t.weight) for t in op.delete_effects] == [("r", pytest.approx(0.81))]
    assert op.metadata == {"num_rollouts": 5, "extractor": "first_last_score_delta_v0"}


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="Unknown extraction mode"):
        OperatorExtractor(PredicateRegistry([]), OperatorExtractionConfig(mode="minmax"))
