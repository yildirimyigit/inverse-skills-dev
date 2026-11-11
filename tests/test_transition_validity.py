"""Round 1 (2026-11-11): a transition is valid only if its result is a valid state.

The domain's implications and mutex groups are constraints on states. What an action adds brings
what it implies and displaces its mutex partners, and the disturbances its changes trigger are
applied; then the result must keep every constraint, hold every atom the action adds and none it
deletes. Otherwise the action, or hole, cannot run there. Nothing is deleted to repair a broken
constraint, and a start that breaks one is refused. Every case below was worked out by hand.
"""

from __future__ import annotations

import random

import pytest
from test_exact_hole_planner import Oracle, act, agrees_with_the_oracle, assert_valid, exact

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import stow_domain as sd
from inverse_skills.operators.exact_hole_planner import ExactNoPlan, library_states, reachable_states
from inverse_skills.operators.hole_planner import (
    Action,
    Disturbance,
    Domain,
    InconsistentState,
    InvalidTransition,
    plan,
)
from inverse_skills.operators.literals import neg, pos

# --- the check itself ------------------------------------------------------------------------------


def test_a_state_is_checked_against_every_implication_and_mutex_group():
    domain = Domain(actions=[], implications={"q": frozenset({"p"})}, mutex_groups=[frozenset({"a", "b"})])
    assert domain.violations({"q", "p", "a"}) == []
    assert domain.violations({"q", "a", "b"}) == ["q implies p, but p is false", "a and b exclude each other"]


def test_deleting_an_atom_that_a_true_atom_implies_is_not_a_valid_transition():
    """q implies p. A library action that deletes p while q holds would leave q without p. Its own
    precondition holds, but its result is invalid, so it cannot run; without q it can."""
    drop_p = act("drop_p", (), (), {"p"})
    domain = Domain(actions=[drop_p], implications={"q": frozenset({"p"})})
    state = frozenset({"q", "p"})
    assert domain.can_run(state, drop_p) and not domain.enabled(state, drop_p)
    assert domain.transition_problems(state, drop_p) == ["q implies p, but p is false"]
    assert domain.successor(state, drop_p) is None
    with pytest.raises(InvalidTransition):
        domain.transition(state, drop_p)
    assert domain.successor(frozenset({"p"}), drop_p) == (frozenset(), frozenset())


def test_an_action_that_adds_two_atoms_excluding_each_other_is_invalid():
    both = act("both", (), {"a", "b"})
    domain = Domain(actions=[both], mutex_groups=[frozenset({"a", "b"})])
    assert domain.transition_problems(frozenset(), both) == ["a is added but false after", "b is added but false after"]
    assert not domain.enabled(frozenset(), both)


def test_an_action_whose_additions_imply_an_atom_it_deletes_is_invalid():
    """It adds q and deletes p, but q implies p: its own effects contradict the constraint."""
    swap = act("swap", (), {"q"}, {"p"})
    domain = Domain(actions=[swap], implications={"q": frozenset({"p"})})
    assert domain.transition_problems(frozenset({"p"}), swap) == ["p is deleted but true after"]


def test_what_an_addition_implies_and_displaces_is_applied_and_valid():
    """The declared model the Drawer's pull_open relies on: adding `open` displaces `shut`, its
    mutex partner, and adding q brings the p it implies. These are the additions' modelled
    consequences, not repairs, and the results are valid states."""
    open_ = act("open", (), {"open"})
    make_q = act("make_q", (), {"q"})
    domain = Domain(actions=[open_, make_q], mutex_groups=[frozenset({"open", "shut"})],
                    implications={"q": frozenset({"p"})})
    assert domain.successor(frozenset({"shut"}), open_) == (frozenset({"open"}), frozenset())
    assert domain.successor(frozenset(), make_q) == (frozenset({"q", "p"}), frozenset())


def test_a_disturbance_that_would_break_an_implication_makes_the_transition_invalid():
    """Changing t sweeps b away, and c implies b. With c true, deleting t would leave c without b;
    without c, b is simply displaced."""
    drop_t = act("drop_t", (), (), {"t"})
    domain = Domain(actions=[drop_t], implications={"c": frozenset({"b"})},
                    disturbances=[Disturbance("t", frozenset({"b"}))])
    assert domain.transition_problems(frozenset({"t", "b", "c"}), drop_t) == ["c implies b, but b is false"]
    assert domain.successor(frozenset({"t", "b"}), drop_t) == (frozenset(), frozenset({"b"}))


# --- holes of both signs obey the same check --------------------------------------------------------


def test_a_positive_hole_whose_displacement_would_break_an_implication_cannot_run():
    """s implies r, and p excludes r. A hole for p from {s, r} would displace r and leave s without
    r, so it cannot run; once s can be dropped, it can."""
    stuck = Domain(actions=[], scored=frozenset({"p"}), implications={"s": frozenset({"r"})},
                   mutex_groups=[frozenset({"p", "r"})])
    assert isinstance(exact(stuck, {"s", "r"}, {"p"}), ExactNoPlan)
    assert not agrees_with_the_oracle(stuck, {"s", "r"}, {"p"}, set(), 2, 4)
    freed = Domain(actions=[act("drop_s", (), (), {"s"})], scored=frozenset({"p"}),
                   implications={"s": frozenset({"r"})}, mutex_groups=[frozenset({"p", "r"})])
    result = exact(freed, {"s", "r"}, {"p"})
    assert result.actions == ("drop_s", "HOLE[p]")
    assert_valid(freed, {"s", "r"}, result, {"p"})
    assert agrees_with_the_oracle(freed, {"s", "r"}, {"p"}, set(), 2, 4)


def test_a_negative_hole_and_a_library_delete_are_judged_alike():
    """q implies p. From {q, p}, the hole for `not p` and a library action deleting p are both
    invalid, for the same reason."""
    drop_p = act("drop_p", (), (), {"p"})
    domain = Domain(actions=[drop_p], scored=frozenset({"p"}), implications={"q": frozenset({"p"})})
    state = frozenset({"q", "p"})
    hole = domain.hole(neg("p"), ())
    assert domain.transition_problems(state, hole) == domain.transition_problems(state, drop_p)
    assert domain.successor(state, hole) is None and domain.successor(state, drop_p) is None


# --- starts that break a constraint are refused --------------------------------------------------------


def test_a_start_that_breaks_a_constraint_is_refused_by_every_entry_point():
    domain = Domain(actions=[], scored=frozenset({"p"}), implications={"q": frozenset({"p"})},
                    mutex_groups=[frozenset({"a", "b"})])
    for start in ({"q"}, {"a", "b", "p"}):
        with pytest.raises(InconsistentState):
            exact(domain, start, {"p"})
        with pytest.raises(InconsistentState):
            plan(domain, frozenset(start), {"p"}, mode="relaxed")
        with pytest.raises(InconsistentState):
            library_states(domain, frozenset(start))


# --- the real domains ---------------------------------------------------------------------------------


def test_the_stow_and_drawer_starts_and_every_state_they_reach_are_consistent():
    """Every state the library and holes for every scored literal reach, from the start each task
    planned from, keeps the declared constraints."""
    drawer_start = frozenset({"drawer_closed(drawer)"})
    for domain, start in ((sd.domain(), sd.stowed_state()), (dd.domain(), drawer_start)):
        assert domain.violations(start) == []
        holes = [domain.hole(literal, domain.robot_relative) for key in sorted(domain.scored)
                 for literal in (pos(key), neg(key))]
        for state in reachable_states(domain, start, holes):
            assert domain.violations(state) == [], sorted(state)


# --- random problems: the planner and the oracle agree, with invalid transitions common ----------------


def constrained_problem(rng):
    """Few atoms, most of them tied by implications and mutex groups, so that many transitions
    are invalid."""
    pool = rng.sample(["a(o)", "b(o)", "c(o)", "d", "e", "f"], rng.randint(4, 6))

    def some(low, high):
        return frozenset(rng.sample(pool, min(rng.randint(low, high), len(pool))))

    actions = []
    for i in range(rng.randint(2, 6)):
        add = some(0, 2)
        delete = some(0, 2) - add
        pre = some(0, 1)
        absent = some(0, 1) - pre
        actions.append(Action(f"act{i}", pre=pre, pre_absent=absent, add=add, delete=delete))
    implications = {}
    for _ in range(rng.randint(1, 3)):
        source, implied = rng.sample(pool, 2)
        implications.setdefault(source, set()).add(implied)
    domain = Domain(
        actions=actions,
        scored=some(2, len(pool)),
        robot_relative=frozenset(p for p in pool if "(o)" in p and rng.random() < 0.5),
        mutex_groups=[frozenset(rng.sample(pool, 2))] if rng.random() < 0.7 else [],
        implications={source: frozenset(implied) for source, implied in implications.items()},
        disturbances=([Disturbance(rng.choice(pool), frozenset({rng.choice(pool)}))]
                      if rng.random() < 0.3 else []))
    return domain, some(0, 3), set(some(0, 2)), set(some(0, 2))


def invalid_somewhere(domain, start) -> bool:
    """Some library action can run in some library state, but its result there is invalid."""
    return any(domain.can_run(state, action) and domain.transition_problems(state, action)
               for state in library_states(domain, start) for action in domain.actions)


def test_the_planner_agrees_with_the_oracle_where_invalid_transitions_are_common():
    rng = random.Random(21)
    planned = refused = with_invalid = 0
    for _ in range(1000):
        domain, start, goal_pos, goal_neg = constrained_problem(rng)
        goal_neg -= goal_pos
        if not Oracle(domain, start, goal_pos, goal_neg).consistent(start):
            with pytest.raises(InconsistentState):
                exact(domain, start, goal_pos, goal_neg)
            refused += 1
            continue
        with_invalid += invalid_somewhere(domain, start)
        planned += agrees_with_the_oracle(domain, start, goal_pos, goal_neg, rng.choice([1, 2]), 5)
    # The sample really exercises invalid transitions, refused starts and holes.
    assert with_invalid > 150 and refused > 150 and planned > 100
