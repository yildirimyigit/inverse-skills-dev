"""Round 1 (2026-11-11): where a hole may start.

A hole for the signed literal l starts from an initiation context: a largest set of robot-relative
facts about l's object that hold together in some library-reachable state where l does not hold
yet. A state that already satisfies l cannot be a source, since the hole could not run there. If
l holds in every library-reachable state, there is no context and no hole, and the result says
so. The rule is the same for both signs.
"""

from __future__ import annotations

import random

from test_exact_hole_planner import agrees_with_the_oracle, assert_valid, exact, layered_problem, random_problem
from test_signed_planning import signed_problem

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import stow_domain as sd
from inverse_skills.operators.exact_hole_planner import ExactNoPlan, hole_starts, library_states
from inverse_skills.operators.hole_planner import Action, Domain
from inverse_skills.operators.literals import neg, pos


def test_learning_not_p_does_not_wait_for_a_context_reached_only_after_p_is_gone():
    """Start {p(o), z}; goal z and not p(o). The library's one action needs p(o), adds near(o)
    and deletes p(o) and z. near(o) is robot-relative; only p(o) is scored.

    The library reaches near(o) only in a state where not p(o) already holds and z is lost. That
    state cannot be a source for a hole for not p(o), so the context is empty, and the hole runs
    from the start. Under the earlier rule the context was {near(o)}, and there was no plan."""
    leave = Action("leave", pre=frozenset({"p(o)"}), add=frozenset({"near(o)"}), delete=frozenset({"p(o)", "z"}))
    domain = Domain(actions=[leave], scored=frozenset({"p(o)"}), robot_relative=frozenset({"near(o)"}))
    start = frozenset({"p(o)", "z"})
    library = library_states(domain, start)
    assert library == {start, frozenset({"near(o)"})}
    assert [hole.pre for hole in hole_starts(domain, neg("p(o)"), library)] == [frozenset()]

    result = exact(domain, start, {"z"}, {"p(o)"})
    assert result.actions == ("HOLE[not p(o)]",) and result.steps[0].pre == frozenset()
    assert_valid(domain, start, result, {"z"}, {"p(o)"})
    assert agrees_with_the_oracle(domain, start, {"z"}, {"p(o)"}, 2, 4)

    # The earlier rule took the largest facts held in any library state, here {near(o)}.
    earlier = max((state & {"near(o)"} for state in library), key=len)
    assert earlier == {"near(o)"} and all(not pos("p(o)").holds(state) for state in library if earlier <= state)


def test_both_signs_draw_their_contexts_from_the_states_where_they_do_not_hold():
    """The library reaches {p(o), near(o)} and {far(o)}, and p(o) holds only where near(o) does.
    So a hole for `not p(o)` starts from near(o), and a hole for `p(o)` from far(o): each from
    the states it can run in, by the same rule."""
    domain = Domain(actions=[Action("go_far", add=frozenset({"far(o)"}), delete=frozenset({"p(o)", "near(o)"}))],
                    scored=frozenset({"p(o)"}), robot_relative=frozenset({"near(o)", "far(o)"}))
    library = library_states(domain, frozenset({"p(o)", "near(o)"}))
    assert library == {frozenset({"p(o)", "near(o)"}), frozenset({"far(o)"})}
    assert [hole.pre for hole in hole_starts(domain, neg("p(o)"), library)] == [{"near(o)"}]
    assert [hole.pre for hole in hole_starts(domain, pos("p(o)"), library)] == [{"far(o)"}]


def test_a_literal_the_library_keeps_true_everywhere_has_no_context_and_is_reported():
    """g holds in every library state, so no state could start a hole for g: none is invented,
    and both a plan and a no-plan report name it."""
    domain = Domain(actions=[Action("use", pre=frozenset({"g"}), add=frozenset({"x"}))],
                    scored=frozenset({"g", "x", "y"}))
    library = library_states(domain, frozenset({"g"}))
    assert hole_starts(domain, pos("g"), library) == []
    result = exact(domain, {"g"}, {"g", "y"})
    assert result.actions == ("HOLE[y]",) and result.no_context == (pos("g"),)
    stuck = exact(domain, {"g"}, {"g", "y"}, max_delegated=0)
    assert isinstance(stuck, ExactNoPlan) and stuck.no_context == (pos("g"),)


def test_every_context_is_held_in_a_library_state_where_its_literal_does_not_hold():
    """On random problems of every kind, for every scored literal of either sign: each context is
    exactly the robot-relative facts of some library state where the literal does not hold, and
    no larger such set exists. A literal with no such state has no context."""
    checked = without = 0
    for make, seed in ((random_problem, 31), (layered_problem, 32), (signed_problem, 33)):
        rng = random.Random(seed)
        for _ in range(150):
            domain, start, _, _ = make(rng)
            if domain.violations(start):
                continue
            library = library_states(domain, frozenset(start))
            for key in sorted(domain.scored):
                for literal in (pos(key), neg(key)):
                    about = domain.about_object(literal)
                    eligible = {state & about for state in library if not literal.holds(state)}
                    contexts = [hole.pre for hole in hole_starts(domain, literal, library)]
                    assert set(contexts) == {c for c in eligible if not any(c < other for other in eligible)}
                    for context in contexts:
                        assert any(not literal.holds(state) and state & about == context for state in library)
                    checked += len(contexts)
                    without += not contexts
    assert checked > 1000 and without > 100


def test_the_stow_and_drawer_holes_keep_their_contexts():
    """The contexts of the holes the tasks use are unchanged: the drag starts at tcp_near(A), the
    grasp, and the direct baseline's opening, at the standoff."""
    stow, drawer = sd.domain(), dd.domain()
    stow_library = library_states(stow, sd.stowed_state())
    drawer_library = library_states(drawer, frozenset({"drawer_closed(drawer)"}))
    assert [h.pre for h in hole_starts(stow, pos("clear_of_walls(cube_a)"), stow_library)] == [{"tcp_near(cube_a)"}]
    for key in ("handle_grasped(drawer)", "drawer_open(drawer)"):
        assert [h.pre for h in hole_starts(drawer, pos(key), drawer_library)] == [{"at_handle_standoff(drawer)"}]
