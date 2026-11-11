"""Planning with signed literals: negative preconditions, negative goals and negative holes.

Every expected plan below was worked out by hand from the formulation before the planner ran,
and each named problem is also checked against the independent oracle.
"""

from __future__ import annotations

import random

import pytest
from test_exact_hole_planner import Oracle, act, agrees_with_the_oracle, assert_valid, exact, names

from inverse_skills.operators.exact_hole_planner import ExactNoPlan, plan_exact, upstream
from inverse_skills.operators.hole_planner import Action, Disturbance, Domain, hole_spec, learned_action
from inverse_skills.operators.literals import neg, pos
from inverse_skills.operators.schema import LearnedOperator, PredicateTerm


def strs(literals) -> set[str]:
    return {str(literal) for literal in literals}


# --- negative preconditions and negative goals, met by the library --------------------------


def test_a_negative_precondition_blocks_an_action_while_its_atom_is_true():
    make_g = act("make_g", (), {"g"}, (), {"busy"})
    domain = Domain(actions=[make_g])
    assert not domain.applicable(frozenset({"busy"}), make_g)
    assert domain.applicable(frozenset(), make_g)
    assert exact(domain, (), {"g"}).actions == ("make_g",)
    assert isinstance(exact(domain, {"busy"}, {"g"}), ExactNoPlan)        # nothing scored, no hole


@pytest.mark.parametrize("domain, cost", [
    (Domain(actions=[act("clear", (), (), {"x"})], scored=frozenset({"x"})), (0, 0, 1)),            # a delete
    (Domain(actions=[act("put_y", (), {"y"})], scored=frozenset({"x"}),
            mutex_groups=[frozenset({"x", "y"})]), (0, 0, 1)),                                     # a mutex
    (Domain(actions=[act("move", (), {"t"})], scored=frozenset({"x"}),
            disturbances=[Disturbance("t", frozenset({"x"}))]), (0, 1, 1)),                        # a disturbance
], ids=["delete", "mutex partner", "disturbance"])
def test_a_negative_goal_is_met_by_the_library_without_a_hole(domain, cost):
    """The library makes x false, so `not x` needs no hole: holes come first in the cost. The
    disturbance displaces x, which counts as one displaced literal."""
    result = exact(domain, {"x"}, (), {"x"})
    assert result.delegated == () and result.planning_cost == cost
    assert_valid(domain, {"x"}, result, (), {"x"})


# --- negative holes ------------------------------------------------------------------------------


def test_a_negative_prerequisite_is_learned_to_enable_a_positive_operator():
    """`make_p` needs busy false, and nothing clears busy. Both {p} and {not busy} reach g, but
    `not busy` is upstream of p (make_p needs it), so {p} has a replacement and the preference
    learns `not busy`, keeping make_p."""
    domain = Domain(actions=[act("make_p", (), {"p"}, (), {"busy"}), act("use", {"p"}, {"g"})],
                    scored=frozenset({"busy", "p"}))
    assert strs(upstream(domain, pos("p"))) == {"not busy"}
    result = exact(domain, {"busy"}, {"g"})
    assert result.actions == ("HOLE[not busy]", "make_p", "use") and result.planning_cost == (1, 0, 3)
    assert result.steps[0].delete == {"busy"} and result.steps[0].add == frozenset()
    assert_valid(domain, {"busy"}, result, {"g"})
    context = result.hole_contexts[0]
    assert context.literal == neg("busy") and context.used_by == "make_p" and context.waiting == ("make_p",)


def test_a_positive_prerequisite_is_learned_to_enable_an_operator_that_makes_a_goal_false():
    """`clear_x` deletes x given key. {not x} reaches the goal, but key is upstream of `not x`
    (its remover needs key), and {key} reaches it too, so key is learned and clear_x kept."""
    domain = Domain(actions=[act("clear_x", {"key"}, (), {"x"})], scored=frozenset({"x", "key"}))
    assert strs(upstream(domain, neg("x"))) == {"key"}
    result = exact(domain, {"x"}, (), {"x"})
    assert result.actions == ("HOLE[key]", "clear_x") and result.planning_cost == (1, 0, 2)
    assert_valid(domain, {"x"}, result, (), {"x"})


def test_a_mixed_goal_takes_one_positive_and_one_negative_hole():
    domain = Domain(actions=[], scored=frozenset({"g", "x"}))
    result = exact(domain, {"x"}, {"g"}, {"x"})
    assert names(result.delegated) == ("g", "not x") and result.planning_cost == (2, 0, 2)
    reported = {tuple(s.name for s in steps) for steps in (result.steps, *result.alternatives)}
    assert reported == {("HOLE[g]", "HOLE[not x]"), ("HOLE[not x]", "HOLE[g]")}
    assert_valid(domain, {"x"}, result, {"g"}, {"x"})
    over = exact(domain, {"x"}, {"g"}, {"x"}, max_delegated=1)
    assert isinstance(over, ExactNoPlan) and over.limits_reached == ("max_delegated",)
    assert over.feasible_with_all_holes


def test_a_goal_asking_for_both_p_and_not_p_is_rejected():
    with pytest.raises(ValueError, match="contradictory"):
        plan_exact(Domain(actions=[]), (), {"p"}, {"p"})


def test_a_negative_condition_is_re_established_after_an_intermediate_step():
    """The lid is shut at the start and must be shut at the end, but taking the item needs it
    open, and nothing in the library shuts it. So the plan opens it, takes the item, and learns
    to shut it again."""
    domain = Domain(actions=[act("open_lid", (), {"lid_open"}), act("take", {"lid_open"}, {"item"})],
                    scored=frozenset({"lid_open"}))
    result = exact(domain, (), {"item"}, {"lid_open"})
    assert result.actions == ("open_lid", "take", "HOLE[not lid_open]")
    assert_valid(domain, (), result, {"item"}, {"lid_open"})


def test_a_negative_hole_obeys_the_disturbances_like_any_action():
    """Making t false sweeps b away, for a hole as for a library action."""
    domain = Domain(actions=[], scored=frozenset({"t"}), disturbances=[Disturbance("t", frozenset({"b"}))])
    result = exact(domain, {"t", "b"}, (), {"t"})
    assert result.actions == ("HOLE[not t]",) and result.collateral == 1
    assert domain.transition(frozenset({"t", "b"}), result.steps[0]) == (frozenset(), frozenset({"b"}))


def test_deleting_an_implied_atom_leaves_what_implies_it():
    """LIMITATION, kept as it is for library deletes: implications are applied to what an action
    adds, not in reverse to what it deletes. A hole for `not p` while q holds, with q implying p,
    leaves q true and p false."""
    domain = Domain(actions=[], scored=frozenset({"p"}), implications={"q": frozenset({"p"})})
    result = exact(domain, {"q", "p"}, (), {"p"})
    assert result.actions == ("HOLE[not p]",)
    assert domain.apply(frozenset({"q", "p"}), result.steps[0]) == {"q"}


# --- budget-aware replacement, signed ------------------------------------------------------------


def test_a_negative_hole_is_replaced_by_its_upstream_prerequisites_only_within_the_budget():
    """`clear_x` deletes x given a and b. With k = 1, no replacement of `not x` fits, so `not x`
    is learned. With k = 2, {a, b} replaces it: two holes, chosen over one, because the set is
    subset-minimal and the preference keeps the remover. Subset-minimality is not minimum
    cardinality, and the choice depends on k."""
    domain = Domain(actions=[act("clear_x", {"a", "b"}, (), {"x"})], scored=frozenset({"x", "a", "b"}))
    one = exact(domain, {"x"}, (), {"x"}, max_delegated=1)
    assert one.actions == ("HOLE[not x]",)
    two = exact(domain, {"x"}, (), {"x"}, max_delegated=2)
    assert names(two.delegated) == ("a", "b") and two.actions[-1] == "clear_x" and two.planning_cost == (2, 0, 3)
    oracle = Oracle(domain, {"x"}, (), {"x"})
    assert oracle.admissible({neg("x")}, 1) and not oracle.admissible({neg("x")}, 2)
    assert oracle.admissible({pos("a"), pos("b")}, 2)


def test_a_negative_hole_ends_a_long_plan_with_no_length_bound():
    chain = [act(f"step{i}", {f"s{i}"}, {f"s{i + 1}"}, {f"s{i}"}) for i in range(14)]
    domain = Domain(actions=[*chain, act("mark", {"s0"}, {"x"})], scored=frozenset({"x"}))
    result = exact(domain, {"s0"}, {"s14"}, {"x"})
    assert result.delegated == () and len(result.actions) == 14 and "mark" not in result.actions  # x never true
    marked = exact(domain, {"s0", "x"}, {"s14"}, {"x"})
    assert names(marked.delegated) == ("not x",) and len(marked.actions) == 15
    assert_valid(domain, {"s0", "x"}, marked, {"s14"}, {"x"})


# --- the learning problem of a negative hole -----------------------------------------------------


def test_the_spec_of_a_negative_hole_keeps_its_sign_its_fences_and_its_pre_state():
    """Only x must change. f must stay true and n must stay false while it does: one fence of
    each sign. At the handoff x must hold (the hole's pre-state) and so must f; n must not."""
    domain = Domain(actions=[], scored=frozenset({"x", "f", "n"}))
    start = frozenset({"x", "f"})
    result = exact(domain, start, {"f"}, {"x", "n"})
    assert result.actions == ("HOLE[not x]",)
    spec = hole_spec(domain, start, result, {"f"}, {"x", "n"})
    assert spec["target"] == {"atom": "x", "polarity": "negative"} and spec["achieve"] == "x"
    assert (spec["preserve"], spec["preserve_absent"]) == (["f"], ["n"])
    assert spec["fences"] == [{"atom": "f", "polarity": "positive"}, {"atom": "n", "polarity": "negative"}]
    assert spec["handoff_state"] == ["f", "x"]
    assert spec["handoff_conditions"] == {"hold": ["f", "x"], "absent": ["n"]}


# --- learned operators keep their signs in the library ----------------------------------------


def test_a_learned_operator_with_a_negative_precondition_runs_only_while_its_atom_is_false():
    """Its executions began with p true and q false. Read with its signs, it needs q false; read
    as before (every precondition positive), it would wrongly need q true and send the plan to
    make_q first."""
    operator = LearnedOperator("skill", preconditions=[PredicateTerm("p"), PredicateTerm("q", polarity="negative")],
                               add_effects=[PredicateTerm("r")])
    learned = learned_action(pos("r"), operator)
    assert (learned.pre, learned.pre_absent) == ({"p"}, {"q"})
    domain = Domain(actions=[act("make_p", (), {"p"}), act("make_q", (), {"q"}), learned])
    assert exact(domain, (), {"r"}, max_delegated=0).actions == ("make_p", "LEARNED[r]")
    assert isinstance(exact(domain, {"q"}, {"r"}, max_delegated=0), ExactNoPlan)   # q never becomes false


# --- the oracle, on problems weighted to negative literals ------------------------------------


def signed_problem(rng):
    """Few atoms, many deletes and negative preconditions, goals of both signs."""
    pool = rng.sample(["a(o)", "b(o)", "c(o)", "d", "e", "f"], rng.randint(4, 6))

    def some(low, high):
        return frozenset(rng.sample(pool, min(rng.randint(low, high), len(pool))))

    actions = []
    for i in range(rng.randint(2, 6)):
        add = some(0, 2)
        delete = some(0, 2) - add
        pre = some(0, 2)
        absent = some(0, 2) - pre
        actions.append(Action(f"act{i}", pre=pre, pre_absent=absent, add=add, delete=delete))
    domain = Domain(
        actions=actions,
        scored=some(2, len(pool)),
        robot_relative=frozenset(p for p in pool if "(o)" in p and rng.random() < 0.5),
        mutex_groups=[frozenset(rng.sample(pool, 2))] if rng.random() < 0.3 else [],
        implications={rng.choice(pool): frozenset({rng.choice(pool)})} if rng.random() < 0.2 else {},
        disturbances=([Disturbance(rng.choice(pool), frozenset({rng.choice(pool)}))]
                      if rng.random() < 0.2 else []))
    start = some(0, 3)
    goal_pos = set(some(0, 2))
    goal_neg = set(some(1, 2)) - goal_pos
    return domain, start, goal_pos, goal_neg


def test_the_planner_agrees_with_the_oracle_on_signed_problems():
    rng = random.Random(11)
    with_a_negative_hole = with_a_hole = 0
    for _ in range(1200):
        domain, start, goal_pos, goal_neg = signed_problem(rng)
        budget = rng.choice([1, 2])
        if agrees_with_the_oracle(domain, start, goal_pos, goal_neg, budget, 5):
            with_a_hole += 1
            result = plan_exact(domain, start, goal_pos, goal_neg, max_delegated=budget)
            with_a_negative_hole += any(not literal.positive for literal in result.delegated)
    assert with_a_hole > 200 and with_a_negative_hole > 100     # the sample really exercises negative holes


@pytest.mark.parametrize("name, domain, start, goal_pos, goal_neg, budget", [
    ("negative prerequisite", Domain(actions=[act("make_p", (), {"p"}, (), {"busy"}), act("use", {"p"}, {"g"})],
                                     scored=frozenset({"busy", "p"})), {"busy"}, {"g"}, set(), 2),
    ("remover's prerequisite", Domain(actions=[act("clear_x", {"key"}, (), {"x"})], scored=frozenset({"x", "key"})),
     {"x"}, set(), {"x"}, 2),
    ("mixed goal", Domain(actions=[], scored=frozenset({"g", "x"})), {"x"}, {"g"}, {"x"}, 2),
    ("lid", Domain(actions=[act("open_lid", (), {"lid_open"}), act("take", {"lid_open"}, {"item"})],
                   scored=frozenset({"lid_open"})), set(), {"item"}, {"lid_open"}, 2),
    ("two removers' prerequisites, k=1", Domain(actions=[act("clear_x", {"a", "b"}, (), {"x"})],
                                                scored=frozenset({"x", "a", "b"})), {"x"}, set(), {"x"}, 1),
    ("two removers' prerequisites, k=2", Domain(actions=[act("clear_x", {"a", "b"}, (), {"x"})],
                                                scored=frozenset({"x", "a", "b"})), {"x"}, set(), {"x"}, 2),
], ids=lambda v: v if isinstance(v, str) else "")
def test_the_named_signed_problems_agree_with_the_oracle(name, domain, start, goal_pos, goal_neg, budget):
    assert agrees_with_the_oracle(domain, start, goal_pos, goal_neg, budget, 6)
