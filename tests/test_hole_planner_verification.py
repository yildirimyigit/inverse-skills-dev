"""Verification of the hole planner on small symbolic problems, edge cases included.

Every scenario states its start, library, goal and expected result, runs the real
planner, and replays the returned plan with the domain's own transition model:
each step must apply and change the state, the goal must hold at the end, and the
cost recomputed from the transitions must equal the cost the planner reported.

Tests that pin a limitation of the formulation say so in their docstring; they
assert what the planner does today, not what it should do. Covered elsewhere and
not repeated here (tests/test_hole_planner.py): the shorter path kept under the
step limit, paths that used different holes kept apart, cycles, conjunctive
prerequisites, a negated goal that needs a remover, and the delete relaxation.
"""

from __future__ import annotations

import pytest

from inverse_skills.operators.hole_planner import (
    Action,
    Disturbance,
    Domain,
    NoPlan,
    hole_spec,
    plan,
    relaxed_reachable,
)


def replay(domain: Domain, start, steps) -> tuple[frozenset[str], tuple[int, ...]]:
    """Re-apply `steps` from `start`. Returns the final state and the cost recomputed
    from the transitions: (delegated predicates, collateral, literals holes change,
    holes, length)."""
    state, delegated, collateral, changed, holes = frozenset(start), set(), 0, 0, 0
    for action in steps:
        assert domain.applicable(state, action), f"{action.name} is not applicable"
        nxt, displaced = domain.transition(state, action)
        assert nxt != state, f"{action.name} changes nothing"
        collateral += len(displaced)
        if action.hole_for is not None:
            delegated.add(action.hole_for)
            holes += 1
            changed += len((nxt ^ state) - displaced)
        state = nxt
    return state, (len(delegated), collateral, changed, holes, len(steps))


def assert_valid(domain: Domain, start, result, goal_pos, goal_neg=()) -> frozenset[str]:
    """The plan, and every tied alternative, replays to the goal at the reported cost."""
    for steps in (result.steps, *result.alternatives):
        final, cost = replay(domain, start, steps)
        assert set(goal_pos) <= final and not (set(goal_neg) & final)
        assert cost == result.cost
    return replay(domain, start, result.steps)[0]


def optimal(domain: Domain, start, goal_pos, goal_neg, candidates, max_delegated=2, max_length=10):
    """Exhaustive enumeration of every sequence within the same bounds, over the
    library and the holes the planner offered: the cheapest cost and every plan with
    it. No-op steps and steps past the goal are skipped (they only add cost)."""
    holes = [domain.hole(key, relaxed_reachable(domain, start)) for key in candidates]
    actions = list(domain.actions) + holes
    best, plans = None, []

    def extend(state, taken):
        nonlocal best, plans
        if set(goal_pos) <= state and not (set(goal_neg) & state):
            cost = replay(domain, start, taken)[1]
            if best is None or cost < best:
                best, plans = cost, [taken]
            elif cost == best:
                plans.append(taken)
            return
        if len(taken) == max_length:
            return
        for action in actions:
            delegated = {a.hole_for for a in (*taken, action) if a.hole_for}
            if (domain.applicable(state, action) and domain.apply(state, action) != state
                    and len(delegated) <= max_delegated):
                extend(domain.apply(state, action), (*taken, action))

    extend(frozenset(start), ())
    return best, plans


# Small domains, used by several scenarios.

def no_library() -> Domain:
    return Domain(actions=[], scored=frozenset({"p", "q", "r"}))


def two_part_assembly() -> Domain:
    """`assemble` needs both parts fixed, and nothing fixes a part. The robot can be
    near one part at a time, and being near a part is what a hole for it starts from."""
    return Domain(
        actions=[Action("approach(o1)", add=frozenset({"near(o1)"})),
                 Action("approach(o2)", add=frozenset({"near(o2)"})),
                 Action("assemble", pre=frozenset({"fixed(o1)", "fixed(o2)"}), add=frozenset({"done"}))],
        scored=frozenset({"fixed(o1)", "fixed(o2)", "near(o1)", "near(o2)", "done"}),
        robot_relative=frozenset({"near(o1)", "near(o2)"}),
        mutex_groups=[frozenset({"near(o1)", "near(o2)"})])


# A. No holes required

def test_a_satisfied_goal_needs_an_empty_plan():
    domain = Domain(actions=[Action("make_g", pre=frozenset({"k"}), add=frozenset({"g"}))],
                    scored=frozenset({"g", "k"}))
    for goal_neg in ((), {"x"}):
        result = plan(domain, frozenset({"g"}), {"g"}, goal_neg)
        assert result.actions == () and result.cost == (0, 0, 0, 0, 0)
        assert result.capability_gaps == () and result.candidates == ()


def test_a_goal_the_library_reaches_needs_no_hole():
    domain = Domain(actions=[Action("get_k", add=frozenset({"k"})),
                             Action("make_g", pre=frozenset({"k"}), add=frozenset({"g"}))],
                    scored=frozenset({"g", "k"}))
    result = plan(domain, frozenset(), {"g"})
    assert result.actions == ("get_k", "make_g") and result.delegated == ()
    assert result.candidates == ()                      # no hole is even offered
    assert_valid(domain, frozenset(), result, {"g"})


# B. Empty action library

def test_b_empty_library_delegates_the_goal_itself():
    """Symbolically feasible, nothing more: the hole has no start condition, because
    no library action establishes anything about its object. Whether a policy can
    learn p from arbitrary states is not something the planner checks; its only
    test is that p has a soft score to train against."""
    domain = no_library()
    result = plan(domain, frozenset(), {"p"})
    assert result.actions == ("HOLE[p]",) and result.cost == (1, 0, 1, 1, 1)
    assert_valid(domain, frozenset(), result, {"p"})
    spec = hole_spec(domain, frozenset(), result, {"p"})
    assert spec["achieve"] == "p" and spec["start"] == [] and spec["prefix"] == []


def test_b_each_missing_goal_predicate_gets_its_own_hole():
    """Both orders cost the same, so the other one is reported as a tie."""
    domain = no_library()
    result = plan(domain, frozenset(), {"p", "q"})
    assert sorted(result.actions) == ["HOLE[p]", "HOLE[q]"] and result.cost == (2, 0, 2, 2, 2)
    assert len(result.alternatives) == 1
    assert_valid(domain, frozenset(), result, {"p", "q"})


def test_b_more_missing_capabilities_than_the_limit_gives_no_plan():
    """The limit counts distinct delegated predicates. Every gap is still reported."""
    domain = no_library()
    three = plan(domain, frozenset(), {"p", "q", "r"})                 # default limit 2
    assert isinstance(three, NoPlan) and three.synthesizable_gaps == ("p", "q", "r")
    two = plan(domain, frozenset(), {"p", "q"}, max_delegated=1)
    assert isinstance(two, NoPlan) and two.synthesizable_gaps == ("p", "q")


def test_b_a_gap_without_a_soft_score_is_reported_not_delegated():
    unscored = Domain(actions=[], scored=frozenset())
    for unrestricted in (False, True):
        result = plan(unscored, frozenset(), {"p"}, unrestricted_holes=unrestricted)
        assert isinstance(result, NoPlan) and result.unsynthesizable_gaps == ("p",)
    # One scored and one unscored goal predicate: no partial plan, both gaps reported.
    mixed = plan(Domain(actions=[], scored=frozenset({"p"})), frozenset(), {"p", "q"})
    assert isinstance(mixed, NoPlan)
    assert mixed.synthesizable_gaps == ("p",) and mixed.unsynthesizable_gaps == ("q",)


# C. Multi-literal effects

def test_c_a_hole_and_its_mutex_partner_both_change():
    """HOLE[p] adds p, and the mutex group removes q: the transition, the
    falsified-literal set the executive checks, and the cost all show both."""
    domain = Domain(actions=[], scored=frozenset({"p", "q"}), mutex_groups=[frozenset({"p", "q"})])
    result = plan(domain, frozenset({"q"}), {"p"}, {"q"})
    assert result.actions == ("HOLE[p]",)
    assert assert_valid(domain, frozenset({"q"}), result, {"p"}, {"q"}) == {"p"}
    assert domain.removes(result.steps[0]) == {"q"}
    assert result.hole_literals == 2                     # p added and q removed


def test_c_one_hole_satisfies_what_its_predicate_implies():
    """With p -> q declared, a single HOLE[p] meets a goal that needs both."""
    domain = Domain(actions=[], scored=frozenset({"p", "q"}), implications={"p": frozenset({"q"})})
    result = plan(domain, frozenset(), {"p", "q"})
    assert result.actions == ("HOLE[p]",) and result.cost == (1, 0, 2, 1, 1)
    assert_valid(domain, frozenset(), result, {"p", "q"})


def test_c_two_independent_predicates_need_two_holes():
    """LIMITATION. A hole establishes exactly one predicate: the planner cannot
    create one hole whose add effects hold two independent predicates. Unrelated
    p and q take two holes, and count as two against the delegation limit."""
    domain = no_library()
    reach = relaxed_reachable(domain, frozenset())
    assert all(domain.hole(key, reach).add == {key} for key in ("p", "q"))
    result = plan(domain, frozenset(), {"p", "q"})
    assert result.delegated == ("p", "q") and len(result.hole_indices) == 2
    assert isinstance(plan(domain, frozenset(), {"p", "q"}, max_delegated=1), NoPlan)


# D. Multiple holes inside a plan

def test_d_two_missing_prerequisites_take_two_holes_each_with_its_own_learning_problem():
    domain = two_part_assembly()
    result = plan(domain, frozenset(), {"done"})
    assert result.delegated == ("fixed(o1)", "fixed(o2)") and result.cost == (2, 0, 2, 2, 5)
    assert result.actions[-1] == "assemble"
    assert_valid(domain, frozenset(), result, {"done"})
    # Each hole starts where the robot is near its own part.
    holes = {result.steps[i].hole_for: result.steps[i] for i in result.hole_indices}
    assert holes["fixed(o1)"].pre == {"near(o1)"} and holes["fixed(o2)"].pre == {"near(o2)"}
    # One learning problem per hole: the second must keep the first part fixed.
    first, second = (hole_spec(domain, frozenset(), result, {"done"}, index=i) for i in (0, 1))
    assert (first["achieve"], first["preserve"]) == ("fixed(o1)", [])
    assert (second["achieve"], second["preserve"]) == ("fixed(o2)", ["fixed(o1)"])
    assert [a["name"] for a in second["prefix"]] == list(result.actions[:result.hole_indices[1]])
    # Two capabilities are missing, so a limit of one leaves no plan.
    assert isinstance(plan(domain, frozenset(), {"done"}, max_delegated=1), NoPlan)


# E. Negative restoration objectives

def test_e_library_actions_remove_what_must_become_false():
    swap = Domain(actions=[Action("swap", pre=frozenset({"x"}), add=frozenset({"y"}), delete=frozenset({"x"}))],
                  scored=frozenset({"x", "y"}))
    assert plan(swap, frozenset({"x"}), {"y"}, {"x"}).actions == ("swap",)
    # Removed through a mutex partner the library adds.
    via_mutex = Domain(actions=[Action("set_p", add=frozenset({"p"}))], scored=frozenset({"p", "x"}),
                       mutex_groups=[frozenset({"p", "x"})])
    assert plan(via_mutex, frozenset({"x"}), set(), {"x"}).actions == ("set_p",)
    # Making y also makes x, which must then be cleared.
    readds = Domain(actions=[Action("make_y", add=frozenset({"y", "x"})), Action("clear_x", delete=frozenset({"x"}))],
                    scored=frozenset({"x", "y"}))
    result = plan(readds, frozenset(), {"y"}, {"x"})
    assert result.actions == ("make_y", "clear_x")
    assert_valid(readds, frozenset(), result, {"y"}, {"x"})


def test_e_a_hole_removes_a_negative_goal_through_its_mutex_partner():
    """The drawer's pattern: the goal is p and not x, with p and x exclusive."""
    domain = Domain(actions=[], scored=frozenset({"p", "x"}), mutex_groups=[frozenset({"p", "x"})])
    result = plan(domain, frozenset({"x"}), {"p"}, {"x"})
    assert result.actions == ("HOLE[p]",)
    assert_valid(domain, frozenset({"x"}), result, {"p"}, {"x"})


def test_e_nothing_can_delete_a_literal_with_an_empty_library():
    """LIMITATION. Holes only establish literals. A goal that only needs x to become
    false, with no library action that removes x, gets no hole, not even through
    x's scored mutex partner p, and the report names no gap."""
    domain = Domain(actions=[], scored=frozenset({"x", "p"}), mutex_groups=[frozenset({"p", "x"})])
    for unrestricted in (False, True):
        result = plan(domain, frozenset({"x"}), set(), {"x"}, unrestricted_holes=unrestricted)
        assert isinstance(result, NoPlan)
        assert result.capability_gaps == () and result.candidates == ()


# F. Search correctness and limits

def test_f_the_plan_length_limit_is_respected():
    """A four-step chain: no plan within three steps, found within four. The report
    does not say the limit was the reason (no gap is named either way)."""
    chain = Domain(actions=[Action(f"s{i}", pre=frozenset({f"c{i}"}), add=frozenset({f"c{i + 1}"}),
                                   delete=frozenset({f"c{i}"})) for i in range(4)])
    start = frozenset({"c0"})
    assert isinstance(plan(chain, start, {"c4"}, max_length=3), NoPlan)
    assert isinstance(plan(chain, start, {"c4"}, max_length=0), NoPlan)
    result = plan(chain, start, {"c4"}, max_length=4)
    assert result.actions == ("s0", "s1", "s2", "s3")
    assert_valid(chain, start, result, {"c4"})


def shorter_way() -> Domain:
    """The regression domain of test_a_shorter_way_to_an_explored_state_is_still_explored."""
    long_way = [Action("long1", pre=frozenset({"start"}), add=frozenset({"step1"}),
                       delete=frozenset({"start", "object_in_place"}))]
    long_way += [Action(f"long{i}", pre=frozenset({f"step{i - 1}"}), add=frozenset({f"step{i}"}),
                        delete=frozenset({f"step{i - 1}"})) for i in range(2, 8)]
    long_way.append(Action("long8", pre=frozenset({"step7"}), add=frozenset({"middle"}), delete=frozenset({"step7"})))
    finish = [Action("finish1", pre=frozenset({"middle"}), add=frozenset({"end1"}), delete=frozenset({"middle"})),
              Action("finish2", pre=frozenset({"end1"}), add=frozenset({"end2"}), delete=frozenset({"end1"})),
              Action("finish3", pre=frozenset({"end2"}), add=frozenset({"goal"}), delete=frozenset({"end2"}))]
    return Domain(actions=[Action("shortcut", pre=frozenset({"start"}), add=frozenset({"middle"}),
                                  delete=frozenset({"start"}))] + long_way + finish,
                  disturbances=[Disturbance(trigger="middle", clobbers=frozenset({"object_in_place"}))])


def reused_hole() -> Domain:
    """The regression domain of test_paths_that_used_different_holes_are_kept_apart."""
    return Domain(actions=[Action("use_p", pre=frozenset({"p"}), add=frozenset({"m"}), delete=frozenset({"p", "x"})),
                           Action("use_q", pre=frozenset({"q"}), add=frozenset({"m"}), delete=frozenset({"q"})),
                           Action("finish", pre=frozenset({"m", "p"}), add=frozenset({"g"}))],
                  scored=frozenset({"p", "q"}), implications={"p": frozenset({"x"})})


def drawer_like(with_grasp: bool = False) -> Domain:
    """The drawer's vocabulary: open or closed; at the standoff or on the handle."""
    actions = [Action("approach", add=frozenset({"standoff(d)"})),
               Action("pull", pre=frozenset({"grasped(d)"}), add=frozenset({"open(d)"}),
                      delete=frozenset({"grasped(d)"}))]
    if with_grasp:
        actions.append(Action("grasp", pre=frozenset({"standoff(d)"}), add=frozenset({"grasped(d)"})))
    return Domain(actions=actions, scored=frozenset({"open(d)", "closed(d)", "standoff(d)", "grasped(d)"}),
                  robot_relative=frozenset({"standoff(d)", "grasped(d)"}),
                  mutex_groups=[frozenset({"open(d)", "closed(d)"}), frozenset({"standoff(d)", "grasped(d)"})])


CLOSED_DRAWER = ({"closed(d)"}, {"open(d)"}, {"closed(d)"})
SMALL_PROBLEMS = {   # name: (domain, start, goal_pos, goal_neg, planner options)
    "library only": (Domain(actions=[Action("get_k", add=frozenset({"k"})),
                                     Action("make_g", pre=frozenset({"k"}), add=frozenset({"g"}))]),
                     set(), {"g"}, set(), {}),
    "two holes, no library": (no_library(), set(), {"p", "q"}, set(), {}),
    "three gaps over the limit": (no_library(), set(), {"p", "q", "r"}, set(), {}),
    "hole with mutex": (Domain(actions=[], scored=frozenset({"p", "q"}), mutex_groups=[frozenset({"p", "q"})]),
                        {"q"}, {"p"}, {"q"}, {}),
    "hole with implication": (Domain(actions=[], scored=frozenset({"p", "q"}), implications={"p": frozenset({"q"})}),
                              set(), {"p", "q"}, set(), {}),
    "two-part assembly": (two_part_assembly(), set(), {"done"}, set(), {}),
    "two-part assembly, limit 1": (two_part_assembly(), set(), {"done"}, set(), {"max_delegated": 1}),
    "negative goal re-added": (Domain(actions=[Action("make_y", add=frozenset({"y", "x"})),
                                               Action("clear_x", delete=frozenset({"x"}))]),
                               set(), {"y"}, {"x"}, {}),
    "shorter way, limit 10": (shorter_way(), {"start", "object_in_place"}, {"goal"}, set(), {}),
    "shorter way, limit 11": (shorter_way(), {"start", "object_in_place"}, {"goal"}, set(), {"max_length": 11}),
    "reused hole": (reused_hole(), set(), {"g"}, set(), {}),
    "drawer, frontier": (drawer_like(), *CLOSED_DRAWER, {}),
    "drawer, unrestricted": (drawer_like(), *CLOSED_DRAWER, {"unrestricted_holes": True}),
    "drawer with a grasp": (drawer_like(with_grasp=True), *CLOSED_DRAWER, {}),
}


@pytest.mark.parametrize("name", list(SMALL_PROBLEMS))
def test_f_the_chosen_plan_is_optimal_against_exhaustive_enumeration(name):
    """Over the same library, holes and bounds, enumeration finds no cheaper plan, and
    no plan exactly when the planner finds none."""
    domain, start, goal_pos, goal_neg, options = SMALL_PROBLEMS[name]
    start = frozenset(start)
    result = plan(domain, start, goal_pos, goal_neg, **options)
    best, plans = optimal(domain, start, goal_pos, goal_neg, result.candidates,
                          options.get("max_delegated", 2), options.get("max_length", 10))
    if not result:
        assert best is None
        return
    assert_valid(domain, start, result, goal_pos, goal_neg)
    assert result.cost == best
    assert result.steps in plans
    assert len(result.alternatives) + 1 <= len(plans)


def test_f_tied_orderings_through_a_shared_state_are_under_reported():
    """LIMITATION, known and accepted at the planner freeze. HOLE[q] then HOLE[r], or
    the reverse, reach the same state at the same cost; the search keeps one of the
    two paths, so the other ordering is not listed among the alternatives. The chosen
    plan is still optimal."""
    domain = Domain(actions=[Action("make", pre=frozenset({"q", "r"}), add=frozenset({"p"}))],
                    scored=frozenset({"p", "q", "r"}))
    result = plan(domain, frozenset(), {"p"})
    best, plans = optimal(domain, frozenset(), {"p"}, set(), result.candidates)
    assert result.cost == best and len(plans) == 2 and result.alternatives == ()


# A finding of this verification

def test_mutually_exclusive_robot_relative_starts_make_a_hole_inapplicable():
    """LIMITATION, found here. A hole starts from every robot-relative fact about its
    object that the library can reach. If two of those facts exclude each other, the
    start can never hold: the gap is reported as synthesizable, yet no plan is found.
    It fails safe (no wrong plan). The drawer has such a pair (standoff, grasped), but
    no hole in the experiments starts from both: grasped is the hole itself, or
    unreachable, or, with the scripted grasp in the library, no hole is needed. A hole
    for open(d) next to a library grasp would hit it."""
    domain = Domain(actions=[Action("go_near", add=frozenset({"near(o)"})),
                             Action("go_far", add=frozenset({"far(o)"}))],
                    scored=frozenset({"done(o)"}), robot_relative=frozenset({"near(o)", "far(o)"}),
                    mutex_groups=[frozenset({"near(o)", "far(o)"})])
    result = plan(domain, frozenset(), {"done(o)"})
    assert isinstance(result, NoPlan) and result.synthesizable_gaps == ("done(o)",)
    assert domain.hole("done(o)", relaxed_reachable(domain, frozenset())).pre == {"near(o)", "far(o)"}
    # The drawer's holes start from the standoff alone, with and without a scripted grasp.
    for with_grasp in (False, True):
        drawer = drawer_like(with_grasp)
        reach = relaxed_reachable(drawer, frozenset({"closed(d)"}))
        assert drawer.hole("grasped(d)", reach).pre == {"standoff(d)"}
    assert drawer_like().hole("open(d)", relaxed_reachable(drawer_like(), frozenset({"closed(d)"}))).pre \
        == {"standoff(d)"}
    with_grasp = drawer_like(with_grasp=True)
    reach = relaxed_reachable(with_grasp, frozenset({"closed(d)"}))
    assert with_grasp.hole("open(d)", reach).pre == {"standoff(d)", "grasped(d)"}   # the latent case
    assert plan(with_grasp, frozenset({"closed(d)"}), {"open(d)"}, {"closed(d)"}).delegated == ()  # never planned
