"""The handoff check on learned-operator demonstrations, and the failure it was added for.

At seed 2004, lifting cube B drags cube A partly out of the pocket. Every step of the known prefix
still reports success, because each step only measures what it promises. An execution from there
is no demonstration of the learned operator, and it made min_max extraction lose the operator's
only add effect. The scores below are the ones recorded on the real pipeline, to four decimals.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

from inverse_skills.core.scene import RobotState, SceneGraph
from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import stow_domain as sd
from inverse_skills.logging.rollout import ForwardRollout
from inverse_skills.operators import handoff
from inverse_skills.operators.extractor import OperatorExtractionConfig, OperatorExtractor
from inverse_skills.operators.handoff import (
    MAX_ATTEMPTS,
    HandoffConditions,
    HandoffsExhausted,
    Prepared,
    collect_demonstrations,
    handoff_conditions,
    handoff_violations,
    summarize_attempts,
)
from inverse_skills.operators.hole_planner import Action, Domain, hole_spec, learned_action, plan
from inverse_skills.predicates.base import Predicate, PredicateRegistry, PredicateResult

CLEAR, POCKET, TCP, SRC_B = ("clear_of_walls(cube_a)", "in_region(cube_a,pocket)", "tcp_near(cube_a)",
                             "in_region(cube_b,src_b)")
ELSEWHERE = ("gripper_open()", "in_region(cube_a,src_a)", "in_region(cube_b,mouth)", "tcp_near(cube_b)")
STOW_GOAL = ({CLEAR, "gripper_open()", "in_region(cube_a,src_a)", SRC_B}, {POCKET, "in_region(cube_b,mouth)"})
STOW_CONDITIONS = HandoffConditions(frozenset({POCKET, SRC_B, TCP}), frozenset({CLEAR, "in_region(cube_b,mouth)"}))
EXPECTED = {"pre": [POCKET, SRC_B, TCP], "add": [CLEAR], "delete": [POCKET]}

# scene seed: scores of (clear, pocket, tcp_near(A), src_b) when the policy takes over, then when it stops.
RECORDED = {
    2000: ((0.1201, 0.9549, 0.8752, 0.9789), (0.8808, 0.0180, 0.8784, 0.9789)),
    12001: ((0.1198, 0.9579, 0.8608, 0.9811), (0.8695, 0.0220, 0.8592, 0.9811)),
    2002: ((0.1199, 0.9719, 0.8731, 0.9762), (0.8728, 0.0208, 0.8747, 0.9762)),
    2003: ((0.1196, 0.9668, 0.8735, 0.9814), (0.8796, 0.0184, 0.8768, 0.9814)),
    2004: ((0.4148, 0.6656, 0.8702, 0.9563), (0.8364, 0.0368, 0.8557, 0.9563)),
    12004: ((0.1201, 0.9708, 0.8790, 0.9813), (0.8695, 0.0220, 0.8721, 0.9813)),
    2005: ((0.1200, 0.9672, 0.8742, 0.9752), (0.8750, 0.0200, 0.8739, 0.9752)),
    2006: ((0.1196, 0.9738, 0.8628, 0.9809), (0.8311, 0.0397, 0.8604, 0.9809)),
    2007: ((0.1199, 0.9639, 0.8737, 0.9765), (0.8772, 0.0192, 0.8749, 0.9765)),
    2008: ((0.1194, 0.9643, 0.8678, 0.9813), (0.8793, 0.0185, 0.8699, 0.9813)),
    2009: ((0.1196, 0.9756, 0.8653, 0.9812), (0.8738, 0.0204, 0.8667, 0.9812)),
}
ORIGINAL_SEEDS = tuple(range(2000, 2010))
# The scenes the original collection used: the forward skill had re-rolled 2001 to 12001, and 2004 ran.
SCENES_WITHOUT_THE_CHECK = (2000, 12001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009)
FORWARD_FELL_SHORT = {2001}


def scores_of(values: tuple[float, float, float, float]) -> dict[str, float]:
    scores = dict(zip((CLEAR, POCKET, TCP, SRC_B), values))
    scores.update(dict.fromkeys(ELSEWHERE, 1e-6))
    return scores


class Scripted(Predicate):
    """A predicate named `key` that reports whatever score the test wrote into the scene."""

    def __init__(self, key: str):
        self.name, self.args, self.weight = key, (), 1.0

    @property
    def key(self) -> str:
        return self.name

    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        score = scene.metadata["scores"][self.key]
        return PredicateResult(self.name, self.args, margin=float(np.arctanh(2 * score - 1)), temperature=1.0)


def scene_with(scores: dict[str, float]) -> SceneGraph:
    return SceneGraph(timestep=0, robot=RobotState(q=np.zeros(7), gripper_width=0.0), objects={},
                      metadata={"scores": scores})


def rollout_of(scene_seed: int, ends=None) -> ForwardRollout:
    start, end = RECORDED[scene_seed]
    return ForwardRollout(skill_name="skill", demo_id=f"run_{scene_seed}",
                          scenes=[scene_with(scores_of(start)), scene_with(scores_of(ends or end))])


def learned_from(rollouts, mode: str) -> Action:
    """Extract in the named mode from the rollouts, and make the library action Stage 4 makes."""
    registry = PredicateRegistry(Scripted(p) for p in (CLEAR, POCKET, TCP, SRC_B, *ELSEWHERE))
    result = OperatorExtractor(registry, OperatorExtractionConfig(mode=mode)).extract("skill", rollouts)
    return learned_action(CLEAR, result.operator)


def as_expected(action: Action) -> dict:
    return {"pre": sorted(action.pre), "add": sorted(action.add), "delete": sorted(action.delete)}


def stow_spec() -> dict:
    domain, start = sd.domain(), sd.stowed_state()
    return hole_spec(domain, start, plan(domain, start, *STOW_GOAL, mode="exact"), *STOW_GOAL)


def recorded_prepare(forward_short=FORWARD_FELL_SHORT):
    """The collection's scenes as the real pipeline measured them, with no simulator."""
    def prepare(scene_seed: int) -> Prepared:
        if scene_seed in forward_short:
            return Prepared(failed_stage="forward", reason="differs from the planned start")
        return Prepared(scores=scores_of(RECORDED[scene_seed][0]), context=scene_seed)
    return prepare


# --- What a valid handoff is, read off the plan ---


def test_the_stow_conditions_come_from_the_plan_and_its_world_facts():
    conditions = handoff_conditions(sd.domain(), sd.stowed_state(), stow_spec())
    assert conditions == STOW_CONDITIONS


def test_the_drawer_needs_no_code_of_its_own_and_is_not_asked_for_what_its_plan_does_not_use():
    domain, start = dd.domain(), frozenset({dd.CLOSED})
    goal = ({dd.OPEN}, {dd.CLOSED})
    spec = hole_spec(domain, start, plan(domain, start, *goal, mode="exact"), *goal)
    conditions = handoff_conditions(domain, start, spec)
    assert conditions.hold == {dd.STANDOFF} and conditions.absent == {dd.GRASPED}
    # The drawer is predicted shut at the handoff, but nothing needs that: Stage 0 pulled it open from ajar.
    assert dd.CLOSED not in conditions.hold | conditions.absent


def test_each_kind_of_condition_on_a_small_domain():
    """A start, a mutex partner the hole removes, a fence each way. Unscored and unrelated: not asked."""
    domain = Domain(
        actions=[Action("reach", add=frozenset({"near(o)"})),
                 Action("use", pre=frozenset({"goal_part(o)", "fence"}), pre_absent=frozenset({"bad"}), add=frozenset({"done"}))],
        scored=frozenset({"near(o)", "goal_part(o)", "partner", "fence", "bad", "unrelated", "done"}),
        robot_relative=frozenset({"near(o)"}),
        mutex_groups=[frozenset({"goal_part(o)", "partner"}), frozenset({"goal_part(o)", "unscored"})])
    start = frozenset({"partner", "fence", "unrelated", "unscored"})
    result = plan(domain, start, {"done"}, mode="exact")
    assert result.actions == ("reach", "HOLE[goal_part(o)]", "use")
    conditions = handoff_conditions(domain, start, hole_spec(domain, start, result, {"done"}))
    assert conditions.hold == {"near(o)", "partner", "fence"}
    assert conditions.absent == {"goal_part(o)", "bad"}


def test_seed_2004_is_invalid_for_exactly_the_literal_the_prefix_disturbed():
    reasons = handoff_violations(STOW_CONDITIONS, scores_of(RECORDED[2004][0]))
    assert reasons == [f"{POCKET} should hold, score 0.6656"]


def test_every_other_recorded_handoff_is_valid():
    for scene_seed, (start, _) in RECORDED.items():
        if scene_seed != 2004:
            assert handoff_violations(STOW_CONDITIONS, scores_of(start)) == [], scene_seed


def test_predicates_the_plan_does_not_rely_on_are_not_asked_for():
    scores = scores_of(RECORDED[2000][0])
    scores.update({"gripper_open()": 0.99, "tcp_near(cube_b)": 0.95, "in_region(cube_a,src_a)": 0.9})
    assert handoff_violations(STOW_CONDITIONS, scores) == []


def test_every_condition_is_checked_with_the_symbolic_threshold():
    good = scores_of(RECORDED[2000][0])
    for literal in STOW_CONDITIONS.hold:
        assert handoff_violations(STOW_CONDITIONS, {**good, literal: 0.8}) == []          # true at 0.8
        assert handoff_violations(STOW_CONDITIONS, {**good, literal: 0.7999}) != []
    for literal in STOW_CONDITIONS.absent:
        assert handoff_violations(STOW_CONDITIONS, {**good, literal: 0.7999}) == []       # false below it
        assert handoff_violations(STOW_CONDITIONS, {**good, literal: 0.8}) != []


# --- Bounded retries, recorded ---


def test_an_invalid_handoff_is_replaced_by_the_next_scene_and_the_skill_never_runs_from_it():
    ran = []

    def demonstrate(scene_seed, context):
        ran.append(scene_seed)
        return scene_seed

    kept, attempts = collect_demonstrations([2004], STOW_CONDITIONS, recorded_prepare(), demonstrate)
    assert kept == ran == [12004]
    assert [(a.scene_seed, a.number, a.stage) for a in attempts] == [(2004, 0, "handoff"), (12004, 1, "valid")]
    assert attempts[0].reasons == (f"{POCKET} should hold, score 0.6656",)
    assert dict(attempts[0].scores)[POCKET] == pytest.approx(0.6656)


def test_retries_are_bounded_and_running_out_is_an_error_that_names_every_scene():
    tried = []

    def prepare(scene_seed):
        tried.append(scene_seed)
        return Prepared(scores=scores_of(RECORDED[2004][0]))

    def demonstrate(scene_seed, context):
        raise AssertionError("the skill must not run from an invalid handoff")

    with pytest.raises(HandoffsExhausted) as caught:
        collect_demonstrations([7], STOW_CONDITIONS, prepare, demonstrate)
    assert tried == [7 + k * 10_000 for k in range(MAX_ATTEMPTS)]
    assert MAX_ATTEMPTS == 5
    assert len(caught.value.attempts) == MAX_ATTEMPTS and caught.value.seed == 7
    assert all(a.stage == "handoff" for a in caught.value.attempts)
    assert "scene 40007" in str(caught.value) and POCKET in str(caught.value)


def test_a_demonstration_is_never_silently_dropped():
    """One per seed, in order, or an error: the old loop's `continue` is gone."""
    kept, _ = collect_demonstrations(ORIGINAL_SEEDS, STOW_CONDITIONS, recorded_prepare(), lambda seed, context: seed)
    assert len(kept) == len(ORIGINAL_SEEDS)
    with pytest.raises(HandoffsExhausted):
        collect_demonstrations([2003], STOW_CONDITIONS, lambda seed: Prepared(failed_stage="prefix", reason="dropped"),
                               lambda seed, context: seed)


def test_forward_and_prefix_failures_are_retried_and_recorded_apart_from_the_handoff_rate():
    def prepare(scene_seed):
        if scene_seed == 2000:
            return Prepared(failed_stage="forward", reason="fell short")
        if scene_seed == 12000:
            return Prepared(failed_stage="prefix", reason="pick(cube_b): cube z 12mm")
        if scene_seed == 22000:
            return Prepared(scores=scores_of(RECORDED[2004][0]))
        return Prepared(scores=scores_of(RECORDED[2003][0]))

    kept, attempts = collect_demonstrations([2000], STOW_CONDITIONS, prepare, lambda seed, context: seed)
    assert kept == [32000]
    assert [a.stage for a in attempts] == ["forward", "prefix", "handoff", "valid"]
    assert attempts[1].reasons == ("pick(cube_b): cube z 12mm",)
    summary = summarize_attempts(attempts)
    assert summary == {"attempts": 4, "failed_forward": 1, "failed_prefix": 1, "reached_the_handoff": 2,
                       "invalid_handoffs": 1, "valid_handoffs": 1, "raw_handoff_validity": 0.5}


def test_the_raw_rate_of_the_real_collection():
    _, attempts = collect_demonstrations(ORIGINAL_SEEDS, STOW_CONDITIONS, recorded_prepare(), lambda seed, context: seed)
    summary = summarize_attempts(attempts)
    assert summary["attempts"] == 12 and summary["failed_forward"] == 1 and summary["invalid_handoffs"] == 1
    assert summary["valid_handoffs"] == 10 and summary["raw_handoff_validity"] == pytest.approx(10 / 11)
    with_the_reason = [a.to_dict() for a in attempts if a.stage != "valid"]
    assert [row["scene_seed"] for row in with_the_reason] == [2001, 2004]


# --- The failure seed 2004 caused, and the recovery ---


def test_without_the_check_seed_2004_costs_min_max_the_add_effect_and_the_plan():
    rollouts = [rollout_of(scene) for scene in SCENES_WITHOUT_THE_CHECK]
    assert as_expected(learned_from(rollouts, "mean")) == EXPECTED
    broken = learned_from(rollouts, "min_max")
    assert as_expected(broken) == {"pre": [SRC_B, TCP], "add": [], "delete": []}
    for planner in ("relaxed", "exact"):
        assert not plan(sd.domain(extra_actions=[broken]), sd.stowed_state(), *STOW_GOAL, max_delegated=0, mode=planner)


def test_with_the_check_min_max_recovers_the_operator_and_exact_replanning_works():
    rollouts, attempts = collect_demonstrations(ORIGINAL_SEEDS, STOW_CONDITIONS, recorded_prepare(),
                                                lambda seed, context: rollout_of(context))
    assert [a.scene_seed for a in attempts if a.stage == "valid"] == [2000, 12001, 2002, 2003, 12004, 2005, 2006, 2007, 2008, 2009]
    assert as_expected(learned_from(rollouts, "mean")) == EXPECTED
    recovered = learned_from(rollouts, "min_max")
    assert as_expected(recovered) == EXPECTED
    assert CLEAR in recovered.add

    replanned = plan(sd.domain(extra_actions=[recovered]), sd.stowed_state(), *STOW_GOAL, max_delegated=0, mode="exact")
    assert replanned.actions == ("pick(cube_b)", "place(cube_b,src_b)", "approach(cube_a)",
                                 f"LEARNED[{CLEAR}]", "pick(cube_a)", "place(cube_a,src_a)")


def test_a_run_is_never_selected_by_its_outcome():
    """A valid handoff from which the skill then fails is kept, and min_max reports what it shows.
    The check looks only at the state before the skill, so it cannot be used to force agreement."""
    def demonstrate(scene_seed, context):
        if context == 2006:                                     # the policy runs out of steps
            return rollout_of(context, ends=(0.3, 0.9, 0.87, 0.98))
        return rollout_of(context)

    rollouts, attempts = collect_demonstrations(ORIGINAL_SEEDS, STOW_CONDITIONS, recorded_prepare(), demonstrate)
    assert len(rollouts) == 10 and all(a.stage == "valid" for a in attempts if a.scene_seed == 2006)
    weakened = learned_from(rollouts, "min_max")
    assert weakened.add == frozenset() and weakened.delete == frozenset()


def test_the_handoff_code_names_no_domain_vocabulary():
    """Everything domain-specific must reach it as data. Docstrings may use examples."""
    tree = ast.parse(inspect.getsource(handoff))
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
                  and node.body and isinstance(node.body[0], ast.Expr)}
    words = {"tcp_near", "in_region", "clear_of_walls", "holding", "gripper", "cube", "pocket", "mouth", "drawer", "handle"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            found |= {w for w in words if w in node.value}
        if isinstance(node, ast.Name):
            found |= {w for w in words if w in node.id}
    assert not found
