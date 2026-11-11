"""Regression tests for the guarantees of the "Minimal Hole Planner" formulation.

Each section pins one claim the paper makes about the exact planner: the learning-budget
counterexample, what NoPlan guarantees, subset-minimality without minimum cardinality, the
upstream preference and its limits, the paper's learning problems, and the assumption that the
library is complete under the transition model. Random checks use the `Oracle` of
tests/test_exact_hole_planner.py, which is the formulation written out again and shares no code
with the planner.
"""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

from test_exact_hole_planner import Oracle, layered_problem, random_problem

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import stow_domain as sd
from inverse_skills.operators.exact_hole_planner import ExactNoPlan, plan_exact, upstream
from inverse_skills.operators.handoff import handoff_conditions
from inverse_skills.operators.hole_planner import Action, Domain, hole_spec

from inverse_skills.operators.literals import Literal, neg, pos


def names(literals) -> tuple[str, ...]:
    """Signed literals as a person writes them: "p" and "not p"."""
    return tuple(str(literal) for literal in literals)


def strs(literals) -> set[str]:
    return {str(literal) for literal in literals}


ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"


def act(name, pre=(), add=(), delete=(), absent=()) -> Action:
    return Action(name, pre=frozenset(pre), add=frozenset(add), delete=frozenset(delete),
                  pre_absent=frozenset(absent))


def random_problems(count, seed):
    rng = random.Random(seed)
    for number in range(count):
        yield (random_problem if number % 2 else layered_problem)(rng)


# The learning-budget counterexample

def counterexample() -> Domain:
    """Empty start, goal g, one library action that needs p and q and adds g. All three scored."""
    return Domain(actions=[act("make", {"p", "q"}, {"g"})], scored=frozenset({"p", "q", "g"}))


def test_a_hole_the_budget_can_pay_for_is_not_refused_for_a_replacement_it_cannot():
    """HOLE[g] reaches the goal. {p, q} is upstream of g and reaches it too, but needs two holes,
    which a budget of one cannot pay for. So it does not refuse HOLE[g]."""
    result = plan_exact(counterexample(), (), {"g"}, max_delegated=1)
    assert result.actions == ("HOLE[g]",)


def test_with_a_budget_for_both_prerequisites_they_are_learned_instead():
    for budget in (2, 3):
        result = plan_exact(counterexample(), (), {"g"}, max_delegated=budget)
        assert result.actions == ("HOLE[p]", "HOLE[q]", "make")


def test_with_no_budget_there_is_no_plan_and_the_budget_is_named():
    result = plan_exact(counterexample(), (), {"g"}, max_delegated=0)
    assert isinstance(result, ExactNoPlan)
    assert result.limits_reached == ("max_delegated",) and result.feasible_with_all_holes


# What NoPlan guarantees

def test_no_plan_exactly_when_no_set_within_the_budget_is_feasible():
    """NoPlan is exact relative to the candidates, the budget and the transition model, for every
    budget. When holes for every candidate would work, raising the budget to the fewest that
    suffice gives a plan."""
    for domain, start, goal_pos, goal_neg in random_problems(400, seed=11):
        oracle = Oracle(domain, start, goal_pos, goal_neg)
        sizes = [len(chosen) for chosen in oracle.subsets(len(oracle.candidates())) if oracle.solves(chosen)]
        for budget in range(4):
            result = plan_exact(domain, start, goal_pos, goal_neg, max_delegated=budget)
            assert bool(result) == oracle.feasible_within(budget)
            if not result:
                assert result.feasible_with_all_holes == bool(sizes)
                assert result.unnecessary_holes == result.replaceable_holes == ()
        if sizes:
            assert plan_exact(domain, start, goal_pos, goal_neg, max_delegated=min(sizes))


def test_no_plan_without_a_named_limit_means_no_candidate_holes_can_do_it():
    """k has no score, so `use` never runs, whatever the budget."""
    domain = Domain(actions=[act("use", {"k"}, {"g"})], scored=frozenset())
    result = plan_exact(domain, (), {"g"}, max_delegated=5)
    assert isinstance(result, ExactNoPlan)
    assert result.limits_reached == () and not result.feasible_with_all_holes
    assert names(result.unsynthesizable_needs) == ("g", "k")


def test_no_plan_is_relative_to_the_candidate_set():
    """LIMITATION, stated in the paper. x is scored and implies p, so a hole for x would let `use`
    run. But x is neither a goal literal nor a library precondition, so it is no candidate, and
    the answer is NoPlan: exact for the candidates, not for every scored literal."""
    domain = Domain(actions=[act("use", {"p"}, {"g"})], scored=frozenset({"x"}),
                    implications={"x": frozenset({"p"})})
    result = plan_exact(domain, (), {"g"})
    assert isinstance(result, ExactNoPlan) and names(result.candidates) == ()
    assert Oracle(domain, (), {"g"}).solves({"x"})


# Subset-minimality, not minimum cardinality

def test_every_plan_is_subset_minimal():
    """No proper subset of the plan's holes reaches the goal at any length, for every budget."""
    for domain, start, goal_pos, goal_neg in random_problems(400, seed=12):
        oracle = Oracle(domain, start, goal_pos, goal_neg)
        for budget in (1, 2, 3):
            result = plan_exact(domain, start, goal_pos, goal_neg, max_delegated=budget)
            if result:
                assert oracle.is_minimal(result.delegated) and len(result.delegated) <= budget


def test_a_plan_can_use_more_holes_than_the_fewest_that_suffice():
    """The preference trades HOLE[g] for its two upstream prerequisites when the budget allows.
    The plan is subset-minimal, but it is not of minimum size."""
    result = plan_exact(counterexample(), (), {"g"}, max_delegated=2)
    oracle = Oracle(counterexample(), (), {"g"})
    assert len(result.delegated) == 2 and oracle.is_minimal(result.delegated)
    assert oracle.solves({"g"})


# The upstream preference

def test_upstream_follows_preconditions_of_both_signs():
    """Superseded 2026-11-11 by the signed formulation (before: positive preconditions only).
    `make_p` needs r and needs busy absent, and deletes z. So r and `not busy` are upstream of p;
    what it deletes is not something it needs."""
    domain = Domain(actions=[act("make_p", {"r"}, {"p"}, {"z"}, {"busy"}), act("use", {"p"}, {"g"})])
    assert strs(upstream(domain, pos("p"))) == {"r", "not busy"}
    assert strs(upstream(domain, pos("g"))) == {"p", "r", "not busy"}


def test_upstream_is_strict_inside_a_cycle():
    """p and q each establish the other, so neither is upstream of the other; r feeds both."""
    domain = Domain(actions=[act("p_from_q", {"q"}, {"p"}), act("q_from_p", {"p"}, {"q"}),
                             act("q_from_r", {"r"}, {"q"})])
    assert strs(upstream(domain, pos("p"))) == {"r"} and strs(upstream(domain, pos("q"))) == {"r"}


def test_upstream_counts_what_an_action_establishes_through_an_implication():
    domain = Domain(actions=[act("make_x", {"m"}, {"x"})], implications={"x": frozenset({"g"})})
    assert strs(upstream(domain, pos("g"))) == {"m"}


def test_a_replacement_must_really_reach_the_goal():
    """r is upstream of p, but `make_p` deletes the q it restores p for, so {r} reaches nothing,
    and p is kept."""
    domain = Domain(actions=[act("make_q", {"p"}, {"q"}, {"p"}), act("use", {"p", "q"}, {"g"}),
                             act("make_p", {"r"}, {"p"}, {"q"})], scored=frozenset({"p", "q", "r", "g"}))
    assert "r" in strs(upstream(domain, pos("p"))) and not Oracle(domain, {"p"}, {"g"}).solves({"r"})
    assert names(plan_exact(domain, {"p"}, {"g"}).delegated) == ("p",)


def test_the_preference_decides_the_drawer():
    """{grasped} and {open} are both feasible and minimal. The grasp is upstream of the opening,
    so the opening is refused; with the grasp unscored, the opening is what is learned."""
    oracle = Oracle(dd.domain(), {dd.CLOSED}, {dd.OPEN}, {dd.CLOSED})
    assert {frozenset({dd.OPEN}), frozenset({dd.GRASPED})} <= oracle.minimal_sets()
    assert strs(upstream(dd.domain(), pos(dd.OPEN))) == {dd.GRASPED}
    assert names(plan_exact(dd.domain(), {dd.CLOSED}, {dd.OPEN}, {dd.CLOSED}).delegated) == (dd.GRASPED,)
    unscored = dd.domain()
    unscored.scored = unscored.scored - {dd.GRASPED}
    assert names(plan_exact(unscored, {dd.CLOSED}, {dd.OPEN}, {dd.CLOSED}).delegated) == (dd.OPEN,)


# The paper's learning problems

def measured_start(name):
    return frozenset(json.loads((ARTIFACTS / name / "stage2_plan.json").read_text())["seeds"][0]["state"])


def saved(path):
    return json.loads((ARTIFACTS / path).read_text())


def assert_reproduces(spec: dict, trained: dict) -> None:
    """Every field of the spec RL was trained on comes back unchanged. Specs now also carry the
    signed fields (2026-11-11); they must agree with the trained fields: the target is the
    trained `achieve`, positive, the start has no negative part, and the fences are the same."""
    assert {key: spec[key] for key in trained} == trained
    assert spec["target"] == {"atom": trained["achieve"], "polarity": "positive"}
    assert spec["start_absent"] == []
    assert {(f["atom"], f["polarity"]) for f in spec["fences"]} == (
        {(a, "positive") for a in trained["preserve"]} | {(a, "negative") for a in trained["preserve_absent"]})


def test_the_trained_stow_and_drawer_learning_problems_are_reproduced():
    """The hole specs RL was trained on, every field, from the first measured Stage 2 state."""
    for name, domain, goal in (
            ("stow", sd.domain(), sd.goal_from_operator(ARTIFACTS / "stow" / "stage1_operator.json")),
            ("drawer", dd.domain(), dd.goal_from_operator(ARTIFACTS / "drawer" / "stage1_operator.json"))):
        start = measured_start(name)
        assert_reproduces(hole_spec(domain, start, plan_exact(domain, start, *goal), *goal),
                          saved(f"{name}/hole_spec.json"))


def test_the_direct_baseline_is_what_the_planner_learns_without_the_grasp():
    """The targeted-versus-direct ablation, from the planner's side: with the grasp unscored, the
    spec is byte for byte the direct baseline the experiment trained."""
    goal = dd.goal_from_operator(ARTIFACTS / "drawer" / "stage1_operator.json")
    start = measured_start("drawer")
    unscored = dd.domain()
    unscored.scored = unscored.scored - {dd.GRASPED}
    spec = hole_spec(unscored, start, plan_exact(unscored, start, *goal), *goal)
    assert_reproduces(spec, saved("drawer/direct/hole_spec_direct.json"))


# The library is complete under the transition model

def written_out(domain: Domain) -> Domain:
    """Every effect the world constraints give a library action, written into the action."""
    return replace(domain, actions=[replace(a, add=frozenset(domain.closure(a.add)),
                                            delete=frozenset(domain.removes(a)) - frozenset(domain.closure(a.add)))
                                    for a in domain.actions])


def test_writing_the_library_effects_out_changes_nothing_the_planner_or_rl_uses():
    """The world constraints only abbreviate library effects: written out, they give the same
    plans, holes, starts, fences and handoff conditions. Only the add and delete lists of the
    prefix and suffix steps a spec records grow, and the RL environments run those steps by name
    and precondition."""
    for name, domain, goal in (
            ("stow", sd.domain(), sd.goal_from_operator(ARTIFACTS / "stow" / "stage1_operator.json")),
            ("drawer", dd.domain(), dd.goal_from_operator(ARTIFACTS / "drawer" / "stage1_operator.json"))):
        start = measured_start(name)
        result, explicit = plan_exact(domain, start, *goal), plan_exact(written_out(domain), start, *goal)
        assert (result.actions, result.delegated, result.cost) == (explicit.actions, explicit.delegated, explicit.cost)
        spec = hole_spec(domain, start, result, *goal)
        spec_explicit = hole_spec(written_out(domain), start, explicit, *goal)
        assert {k for k in spec if spec[k] != spec_explicit[k]} <= {"prefix", "suffix"}
        assert handoff_conditions(domain, start, spec) == handoff_conditions(written_out(domain), start, spec_explicit)
