"""Signed literals, their two kinds of truth, and the signed extraction rule.

Extraction tests state their demonstrations as tables of scores: table[p][i] = (start score, end
score) of predicate p in demonstration i. The scores sit on both sides of 0.2, 0.5 and 0.8.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from inverse_skills.core.scene import RobotState, SceneGraph
from inverse_skills.logging.rollout import ForwardRollout
from inverse_skills.operators import OperatorExtractionConfig, OperatorExtractor
from inverse_skills.operators.literals import (
    Literal,
    MeasurementError,
    boolean_unmet,
    join,
    literal_set,
    neg,
    pos,
    robust_value,
    robustly_unmet,
    satisfied,
    split,
    symbolic_state,
    validate_scores,
)
from inverse_skills.operators.schema import LearnedOperator, PredicateTerm
from inverse_skills.predicates.base import Predicate, PredicateRegistry, PredicateResult


# --- the literal type -----------------------------------------------------------------------


def test_a_literal_and_its_complement_hold_in_opposite_states() -> None:
    p, not_p = pos("p"), neg("p")
    assert p.complement() == not_p and not_p.complement() == p
    assert p.holds({"p"}) and not p.holds(set())
    assert not_p.holds(set()) and not not_p.holds({"p"})      # closed world: missing means false
    assert satisfied({p, neg("q")}, {"p"}) and not satisfied({p, neg("q")}, {"p", "q"})


def test_the_sign_is_a_field_not_part_of_the_name() -> None:
    for name in ("not p", "¬p", "!p", ""):
        with pytest.raises(ValueError):
            Literal(name)
    assert str(neg("p")) == "not p" and str(pos("p")) == "p"


def test_the_signed_score_is_2v_minus_1_for_p_and_1_minus_2v_for_not_p() -> None:
    for v in (0.0, 0.2, 0.5, 0.7, 0.8, 1.0):
        assert pos("p").signed_score(v) == pytest.approx(2 * v - 1)
        assert neg("p").signed_score(v) == pytest.approx(1 - 2 * v)


def test_boolean_truth_puts_0_5_on_the_positive_side() -> None:
    assert pos("p").boolean_holds(0.5) and not neg("p").boolean_holds(0.5)
    assert neg("p").boolean_holds(0.49) and not pos("p").boolean_holds(0.49)
    assert symbolic_state({"p": 0.5, "q": 0.49, "r": 0.79}) == {"p", "r"}


def test_robust_truth_needs_0_8_and_robust_falsity_0_2() -> None:
    assert pos("p").robustly_holds(0.8) and not pos("p").robustly_holds(0.79)
    assert neg("p").robustly_holds(0.2) and not neg("p").robustly_holds(0.21)
    # A score of 0.7 or 0.75 makes p true in the Boolean sense, but is robust evidence of neither
    # truth nor falsity: `not p` is not robustly satisfied there, and neither is p.
    for v in (0.7, 0.75):
        assert not neg("p").robustly_holds(v) and not pos("p").robustly_holds(v)
        assert robust_value(v) == "uncertain"
    assert (robust_value(0.8), robust_value(0.2), robust_value(0.5)) == ("true", "false", "uncertain")
    assert robustly_unmet({pos("p"), neg("q")}, {"p": 0.9, "q": 0.75}) == [neg("q")]
    assert boolean_unmet({pos("p"), neg("q")}, {"p": 0.9, "q": 0.75}) == [neg("q")]
    assert boolean_unmet({neg("q")}, {"q": 0.3}) == [] and robustly_unmet({neg("q")}, {"q": 0.3}) == [neg("q")]


def test_contradictory_literal_sets_are_rejected() -> None:
    with pytest.raises(ValueError, match="contradictory"):
        literal_set({pos("p"), neg("p")})
    with pytest.raises(ValueError, match="contradictory"):
        join(["p"], ["p"])
    assert literal_set([pos("p"), pos("p")]) == {pos("p")}          # duplicates are merged


def test_split_and_join_are_inverse() -> None:
    literals = {pos("a"), neg("b"), pos("c")}
    positive, negative = split(literals)
    assert (positive, negative) == ({"a", "c"}, {"b"})
    assert join(positive, negative) == literals


def test_a_missing_or_non_finite_score_is_never_read_as_false() -> None:
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(MeasurementError):
            validate_scores({"p": bad})
        with pytest.raises(MeasurementError):
            symbolic_state({"p": bad})
    with pytest.raises(MeasurementError, match="no score for q"):
        robustly_unmet({pos("q")}, {"p": 0.9})
    with pytest.raises(MeasurementError):
        validate_scores({"p": 0.5}, keys=["p", "q"])


def test_a_literal_round_trips_and_a_bare_name_reads_as_positive() -> None:
    for literal in (pos("in_region(cube_a,src_a)"), neg("tcp_near(cube_a)")):
        assert Literal.from_dict(json.loads(json.dumps(literal.to_dict()))) == literal
    assert Literal.from_dict("handle_grasped(drawer)") == pos("handle_grasped(drawer)")


# --- signed extraction ------------------------------------------------------------------------


class _Scripted(Predicate):
    """A predicate named `key` that reports whatever score the test wrote into the scene."""

    def __init__(self, key: str):
        self.name, self.args, self.weight = key, (), 1.0

    @property
    def key(self) -> str:
        return self.name

    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        score = min(max(scene.metadata["scores"][self.key], 1e-12), 1 - 1e-12)
        return PredicateResult(self.name, self.args, margin=float(np.arctanh(2 * score - 1)), temperature=1.0)


def _scene(scores: dict[str, float]) -> SceneGraph:
    return SceneGraph(timestep=0, robot=RobotState(q=np.zeros(7), gripper_width=0.0), objects={},
                      metadata={"scores": scores})


def _extract(table: dict[str, list[tuple[float, float]]], mode: str = "min_max"):
    count = len(next(iter(table.values())))
    rollouts = [ForwardRollout(skill_name="skill", demo_id=f"demo_{i}", scenes=[
        _scene({p: pairs[i][0] for p, pairs in table.items()}),
        _scene({p: pairs[i][1] for p, pairs in table.items()})]) for i in range(count)]
    registry = PredicateRegistry(_Scripted(p) for p in table)
    return OperatorExtractor(registry, OperatorExtractionConfig(mode=mode)).extract("skill", rollouts)


def _pre_eff(table, mode="min_max"):
    op = _extract(table, mode).operator
    return op.precondition_literals(), op.effect_literals()


def test_a_reliably_true_start_gives_p_and_a_reliably_false_start_gives_not_p() -> None:
    table = {"p": [(0.95, 0.95), (0.85, 0.9)], "q": [(0.05, 0.05), (0.15, 0.1)]}
    assert _pre_eff(table) == ({pos("p"), neg("q")}, set())


def test_an_inconsistent_start_gives_no_precondition() -> None:
    table = {"p": [(0.9, 0.9), (0.1, 0.1)], "q": [(0.6, 0.6), (0.6, 0.6)]}    # 0.6: neither reliably
    assert _pre_eff(table) == (set(), set())


@pytest.mark.parametrize("start, expected", [
    (0.8, {pos("p")}), (0.79, set()), (0.6, set()), (0.5, set()), (0.4, set()),
    (0.21, set()), (0.2, {neg("p")}), (0.0, {neg("p")}),
])
def test_the_robust_thresholds_are_inclusive(start, expected) -> None:
    assert _pre_eff({"p": [(start, start)] * 3})[0] == expected


def test_every_demonstration_must_agree() -> None:
    table = {"p": [(0.1, 0.1)] * 4 + [(0.25, 0.25)], "q": [(0.95, 0.95)] * 4 + [(0.79, 0.79)]}
    assert _pre_eff(table) == (set(), set())


def test_a_false_to_true_transition_is_an_effect_p_with_not_p_before() -> None:
    table = {"p": [(0.05, 0.95), (0.2, 0.8), (0.1, 0.9)]}
    pre, eff = _pre_eff(table)
    assert (pre, eff) == ({neg("p")}, {pos("p")})
    op = _extract(table).operator
    assert [t.key for t in op.add_effects] == ["p"] and op.delete_effects == []


def test_a_true_to_false_transition_is_an_effect_not_p_with_p_before() -> None:
    table = {"p": [(0.95, 0.05), (0.8, 0.2), (0.9, 0.1)]}
    pre, eff = _pre_eff(table)
    assert (pre, eff) == ({pos("p")}, {neg("p")})
    op = _extract(table).operator
    assert [t.key for t in op.delete_effects] == ["p"] and op.add_effects == []


def test_an_unchanged_condition_is_a_precondition_and_no_effect() -> None:
    table = {"stays_true": [(0.9, 0.95), (0.85, 0.99)], "stays_false": [(0.05, 0.01), (0.1, 0.15)]}
    assert _pre_eff(table) == ({pos("stays_true"), neg("stays_false")}, set())


def test_a_weak_or_inconsistent_transition_is_no_effect() -> None:
    table = {"weak_rise": [(0.1, 0.7), (0.05, 0.75)],             # never robustly true at the end
             "weak_fall": [(0.9, 0.3), (0.95, 0.25)],             # never robustly false at the end
             "one_demo_only": [(0.05, 0.95), (0.05, 0.05)]}       # changes in one demonstration only
    pre, eff = _pre_eff(table)
    assert eff == set()
    assert pre == {neg("weak_rise"), pos("weak_fall"), neg("one_demo_only")}


def test_every_extracted_effect_has_its_complement_among_the_preconditions_so_the_goal_is_pre() -> None:
    rng = random.Random(7)
    levels = (0.0, 0.05, 0.19, 0.2, 0.21, 0.4, 0.5, 0.6, 0.79, 0.8, 0.81, 0.95, 1.0)
    for _ in range(200):
        demos = rng.randint(1, 4)
        table = {f"p{i}": [(rng.choice(levels), rng.choice(levels)) for _ in range(demos)] for i in range(5)}
        op = _extract(table).operator
        assert op.effects_complement_in_preconditions()
        goal = {t.literal for t in op.restoration_goal()}
        assert goal == op.precondition_literals()


def test_the_extraction_records_its_evidence_for_every_predicate() -> None:
    result = _extract({"p": [(0.05, 0.95), (0.1, 0.9)], "q": [(0.9, 0.9), (0.5, 0.5)]})
    p, q = result.scores["p"], result.scores["q"]
    assert (p["start_min"], p["start_max"], p["end_min"], p["end_max"]) == pytest.approx((0.05, 0.1, 0.9, 0.95))
    assert p["support"] == {"demonstrations": 2, "true_at_start": 0, "false_at_start": 2,
                            "true_at_end": 2, "false_at_end": 0}
    assert p["precondition"] == neg("p").to_dict() and p["effect"] == pos("p").to_dict()
    assert p["changed_by_the_skill"]
    assert q["precondition"] is None and q["effect"] is None and not q["changed_by_the_skill"]


def test_the_legacy_mean_rule_still_extracts_no_negative_precondition() -> None:
    table = {"p": [(0.05, 0.95), (0.1, 0.9)], "q": [(0.05, 0.05), (0.1, 0.1)]}
    pre, eff = _pre_eff(table, mode="mean")
    assert pre == set() and eff == {pos("p")}


# --- the operator schema ----------------------------------------------------------------------


def test_contradictory_operators_are_rejected() -> None:
    with pytest.raises(ValueError, match="contradictory"):
        LearnedOperator("x", preconditions=[PredicateTerm("p"), PredicateTerm("p", polarity="negative")])
    with pytest.raises(ValueError, match="contradictory effects"):
        LearnedOperator("x", add_effects=[PredicateTerm("p")], delete_effects=[PredicateTerm("p")])
    with pytest.raises(ValueError, match="positive"):
        LearnedOperator("x", delete_effects=[PredicateTerm("p", polarity="negative")])
    with pytest.raises(ValueError):
        PredicateTerm("p", polarity="maybe")
    with pytest.raises(ValueError):
        PredicateTerm("not p")


def test_the_restoration_goal_is_pre_with_the_complement_of_every_effect() -> None:
    op = LearnedOperator("x", preconditions=[PredicateTerm("p"), PredicateTerm("q", polarity="negative")],
                         add_effects=[PredicateTerm("r")], delete_effects=[PredicateTerm("p")])
    # r made true, so `not r` held before; that is not among this hand-written Pre.
    assert {t.literal for t in op.restoration_goal()} == {pos("p"), neg("q"), neg("r")}
    assert not op.effects_complement_in_preconditions()
    assert op.effect_literals() == {pos("r"), neg("p")}
    contradicting = LearnedOperator("y", preconditions=[PredicateTerm("r")], add_effects=[PredicateTerm("r")])
    with pytest.raises(ValueError, match="contradicts itself"):
        contradicting.restoration_goal()


def test_serialization_keeps_every_sign() -> None:
    op = _extract({"p": [(0.05, 0.95)] * 2, "q": [(0.95, 0.05)] * 2, "r": [(0.1, 0.1)] * 2}).operator
    data = json.loads(json.dumps(op.to_dict()))
    assert data["schema_version"] == "signed_literals_v1"
    assert {(t["key"], t["polarity"]) for t in data["preconditions"]} == {("p", "negative"), ("q", "positive"),
                                                                          ("r", "negative")}
    assert {(t["key"], t["polarity"]) for t in data["effects"]} == {("p", "positive"), ("q", "negative")}
    assert data["restoration_goal"] == data["inverse_target_terms"] == data["preconditions"]
    back = LearnedOperator.from_dict(data)
    assert back.precondition_literals() == op.precondition_literals()
    assert back.effect_literals() == op.effect_literals()


@pytest.mark.parametrize("path", ["artifacts/stow/stage1_operator.json", "artifacts/drawer/stage1_operator.json",
                                  "artifacts/final_verification_fe374f0/stow_stage1_operator.json"])
def test_a_positive_only_artifact_reads_back_with_its_goal_unchanged(path) -> None:
    saved = json.loads(Path(path).read_text())["operator"]
    op = LearnedOperator.from_dict(saved)
    assert all(t.polarity == "positive" for t in op.preconditions)
    assert {(t.key, t.polarity) for t in op.restoration_goal()} == {
        (t["key"], t["polarity"]) for t in saved["inverse_target_terms"]}
