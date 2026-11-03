"""The optional exact-transition planner, and proof that it leaves the default planner alone.

The sections A to J follow the task that asked for this mode. Later sections pin the rules
for holes: exact necessity, and the causal preference that is separate from it. Every plan is
replayed with `Domain.transition`, and `test_the_search_agrees_with_enumerating_every_plan`
checks the search against `Oracle`, which enumerates every plan and shares no code with the planner.

Tests that pin a limit of the formulation say so in their docstring: they assert what the
planner does today, not what it ideally should do.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import itertools
import random
from typing import NamedTuple

import pytest

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import stow_domain as sd
from inverse_skills.operators import exact_hole_planner, hole_planner
from inverse_skills.operators.exact_hole_planner import (
    ExactNoPlan,
    ExactPlanResult,
    hole_starts,
    library_states,
    plan_exact,
    relevant_actions,
)
from inverse_skills.operators.hole_planner import (
    Action,
    Disturbance,
    Domain,
    NoPlan,
    PlanResult,
    hole_spec,
    plan,
    relaxed_reachable,
)

STOW_GOAL = ({"clear_of_walls(cube_a)", "gripper_open()", "in_region(cube_a,src_a)",
              "in_region(cube_b,src_b)"}, {"in_region(cube_a,pocket)", "in_region(cube_b,mouth)"})
STOW_PLAN = ("pick(cube_b)", "place(cube_b,src_b)", "approach(cube_a)",
             "HOLE[clear_of_walls(cube_a)]", "pick(cube_a)", "place(cube_a,src_a)")
OPEN, CLOSED = "drawer_open(drawer)", "drawer_closed(drawer)"
STANDOFF, GRASPED = "at_handle_standoff(drawer)", "handle_grasped(drawer)"
DRAWER_START = frozenset({CLOSED})
DRAWER_GOAL = ({OPEN}, {CLOSED})


def act(name, pre=(), add=(), delete=(), absent=()) -> Action:
    return Action(name, pre=frozenset(pre), add=frozenset(add), delete=frozenset(delete),
                  pre_absent=frozenset(absent))


def exact(domain, start, goal_pos, goal_neg=(), **limits):
    return plan(domain, frozenset(start), set(goal_pos), set(goal_neg), mode="exact", **limits)


def replay(domain, start, steps):
    """Re-apply `steps` with the real transition. Returns every state along the way and
    the cost recomputed from the transitions."""
    states = [frozenset(start)]
    delegated, collateral, changed, holes = set(), 0, 0, 0
    for action in steps:
        assert domain.applicable(states[-1], action), f"{action.name} is not applicable"
        nxt, displaced = domain.transition(states[-1], action)
        assert nxt != states[-1], f"{action.name} changes nothing"
        collateral += len(displaced)
        if action.hole_for is not None:
            assert action.hole_for in domain.scored, f"{action.name} has no soft score"
            delegated.add(action.hole_for)
            holes += 1
            changed += len((nxt ^ states[-1]) - displaced)
        states.append(nxt)
    return states, (len(delegated), collateral, changed, holes, len(steps))


def assert_valid(domain, start, result, goal_pos, goal_neg=()):
    """The plan and every tied alternative replay to the goal at the reported cost, the
    hole positions are right, and each hole's context agrees with the trace."""
    for steps in (result.steps, *result.alternatives):
        states, cost = replay(domain, start, steps)
        assert set(goal_pos) <= states[-1] and not (set(goal_neg) & states[-1])
        assert cost == result.cost

    states, _ = replay(domain, start, result.steps)
    assert result.actions == tuple(step.name for step in result.steps)
    assert result.hole_indices == tuple(i for i, s in enumerate(result.steps) if s.hole_for)
    assert len(result.hole_contexts) == len(result.hole_indices)
    for context, i in zip(result.hole_contexts, result.hole_indices):
        hole = result.steps[i]
        assert context.literal == hole.hole_for and context.index == i
        assert context.state_before == states[i] and context.literal not in states[i]
        assert hole.pre <= states[i]
        assert context.prefix == result.actions[:i] and context.suffix == result.actions[i + 1:]
        if context.used_at is None:
            assert context.used_by == "goal"
            assert states[-1] & domain.closure(hole.add) & set(goal_pos)
        else:
            assert context.used_by == result.actions[context.used_at] and context.used_at > i
            assert result.steps[context.used_at].pre & domain.closure(hole.add)
        assert context.waiting, "nothing was waiting for this hole"
    return states


def delete_interaction(scored=("p", "q", "g")) -> Domain:
    """`use` needs p and q together, but making q consumes p."""
    return Domain(actions=[act("make_q", {"p"}, {"q"}, {"p"}), act("use", {"p", "q"}, {"g"})],
                  scored=frozenset(scored))


def drawer_without_pull() -> Domain:
    domain = dd.domain()
    domain.actions = [a for a in domain.actions if a.name != "pull_open(drawer)"]
    return domain


# A. The default planner is unchanged

def test_a_the_relaxed_planner_is_the_default():
    for domain, start, goal in ((sd.domain(), sd.stowed_state(), STOW_GOAL),
                                (dd.domain(), DRAWER_START, DRAWER_GOAL)):
        default = plan(domain, start, *goal)
        assert default == plan(domain, start, *goal, mode="relaxed")
        assert type(default) is PlanResult
    assert type(plan(delete_interaction(), frozenset({"p"}), {"g"})) is NoPlan


def test_a_the_default_result_types_gain_no_fields():
    assert [f.name for f in dataclasses.fields(PlanResult)] == [
        "actions", "steps", "delegated", "collateral", "hole_literals", "hole_indices", "expanded",
        "candidates", "alternatives", "capability_gaps", "synthesizable_gaps"]
    assert [f.name for f in dataclasses.fields(NoPlan)] == [
        "capability_gaps", "synthesizable_gaps", "candidates"]


def test_a_a_mode_is_checked():
    with pytest.raises(ValueError, match="Unknown planning mode"):
        plan(delete_interaction(), frozenset({"p"}), {"g"}, mode="exactly")
    with pytest.raises(ValueError, match="relaxed mode only"):
        plan(delete_interaction(), frozenset({"p"}), {"g"}, mode="exact", unrestricted_holes=True)


def test_a_exact_results_still_work_as_the_results_callers_know():
    result = exact(dd.domain(), DRAWER_START, *DRAWER_GOAL)
    assert isinstance(result, ExactPlanResult) and isinstance(result, PlanResult) and result
    failed = exact(delete_interaction(scored=("g",)), {"p"}, {"g"})
    assert isinstance(failed, ExactNoPlan) and isinstance(failed, NoPlan) and not failed
    assert plan_exact(dd.domain(), DRAWER_START, *DRAWER_GOAL) == result


# B. The delete interaction the relaxed analysis cannot see

def test_b_the_hidden_gap_gets_a_causally_justified_hole():
    domain = delete_interaction()
    assert "g" in relaxed_reachable(domain, frozenset({"p"}))
    assert not plan(domain, frozenset({"p"}), {"g"})        # default: no plan, as before

    result = exact(domain, {"p"}, {"g"})
    assert result.actions == ("make_q", "HOLE[p]", "use")    # keeps make_q, which HOLE[q] replaces
    assert result.cost == (1, 0, 1, 1, 3)
    assert result.alternatives == ()
    assert_valid(domain, {"p"}, result, {"g"})

    context = result.hole_contexts[0]
    assert context.literal == "p" and context.used_by == "use" and context.waiting == ("make_q", "use")
    assert context.library_alternatives == ()               # nothing in the library establishes p


def test_b_the_exact_mode_is_not_the_unrestricted_ablation():
    domain = delete_interaction()
    # With every open condition on offer, the shortest plan just delegates the goal.
    assert plan(domain, frozenset({"p"}), {"g"}, unrestricted_holes=True).actions == ("HOLE[g]",)
    assert "HOLE[g]" not in exact(domain, {"p"}, {"g"}).actions


def test_b_either_prerequisite_is_learned_if_the_other_cannot_be():
    """The two plans the task names. Which one depends on which literal can be learned."""
    no_score_for_p = delete_interaction(scored=("q", "g"))
    learn_q = exact(no_score_for_p, {"p"}, {"g"})
    assert learn_q.actions == ("HOLE[q]", "use")
    # make_q establishes q, and the context says why it cannot do this hole's job here.
    assert learn_q.hole_contexts[0].library_alternatives == (("make_q", "the plan then fails at use"),)
    no_score_for_q = delete_interaction(scored=("p", "g"))
    result = exact(no_score_for_q, {"p"}, {"g"})
    assert result.actions == ("make_q", "HOLE[p]", "use")
    assert_valid(no_score_for_q, {"p"}, result, {"g"})


def test_b_the_goal_hole_is_not_a_fallback():
    """Neither prerequisite can be learned, so the only hole left is the goal itself. `use`
    would run if they could be, so the planner reports it rather than work around an action the
    library has, as the default planner does for its own unsynthesizable gaps."""
    result = exact(delete_interaction(scored=("g",)), {"p"}, {"g"})
    assert isinstance(result, ExactNoPlan)
    assert result.replaceable_holes == ("g",) and result.limits_reached == ()
    assert result.unnecessary_holes == () and result.feasible_with_all_holes


def test_b_a_consumed_shared_resource():
    """Both actions consume the one token. The default finds no gap; here the token is what
    is missing, and the library does the rest."""
    domain = Domain(actions=[act("make_p", {"token"}, {"p"}, {"token"}),
                             act("make_q", {"token"}, {"q"}, {"token"})],
                    scored=frozenset({"token", "p", "q"}))
    assert not plan(domain, frozenset({"token"}), {"p", "q"})
    result = exact(domain, {"token"}, {"p", "q"})
    assert result.delegated == ("token",) and result.cost == (1, 0, 1, 1, 3)
    assert set(result.alternative_names) | {result.actions} == {
        ("make_p", "HOLE[token]", "make_q"), ("make_q", "HOLE[token]", "make_p")}
    assert_valid(domain, {"token"}, result, {"p", "q"})


def test_b_a_producer_that_destroys_part_of_the_goal_is_unusable():
    """`make_p` would serve `use_p`, but it destroys z, which the goal needs and nothing can
    restore. So a hole for p is allowed, although a library action establishes p."""
    domain = Domain(actions=[act("use_p", {"p"}, {"g"}), act("make_p", (), {"p"}, {"z"})],
                    scored=frozenset({"p", "g"}))
    result = exact(domain, {"z"}, {"g", "z"})
    assert result.actions == ("HOLE[p]", "use_p")
    assert_valid(domain, {"z"}, result, {"g", "z"})


def test_a_the_cost_order_is_the_defaults():
    """Plans here delegate two literals, displace nothing and change three literals. The next
    term decides: two holes win over three, although they take a longer plan. c and e form a
    cycle, so the causal preference leaves all these plans to the cost. No random problem in
    300,000 reached this term, so it needs a test of its own."""
    domain = Domain(actions=[act("make_d", {"e"}, {"d"}, {"e"}), act("make_e", {"c"}, {"e"}),
                             act("make_c", {"e"}, {"c"})],
                    scored=frozenset({"c", "e", "f"}), mutex_groups=[frozenset({"c", "f"})])
    hole_e, hole_f = domain.hole("e", ()), domain.hole("f", ())
    three_holes = [hole_e, domain.actions[0], hole_e, hole_f]
    assert replay(domain, (), three_holes)[1] == (2, 0, 3, 3, 4)    # shorter, and it reaches the goal

    result = exact(domain, (), {"d", "e", "f"})
    assert result.cost == (2, 0, 3, 2, 5)
    assert_valid(domain, (), result, {"d", "e", "f"})
    assert result.cost == plan(domain, frozenset(), {"d", "e", "f"}).cost


# C. A library that can already do it

def test_c_a_satisfied_goal_needs_an_empty_plan():
    result = exact(delete_interaction(), {"g"}, {"g"})
    assert result.actions == () and result.cost == (0, 0, 0, 0, 0) and result.hole_contexts == ()


def test_c_a_library_plan_is_preferred_over_any_hole():
    domain = dd.domain(extra_actions=[dd.scripted_grasp()])
    result = exact(domain, DRAWER_START, *DRAWER_GOAL)
    assert result.delegated == () and result.hole_indices == ()
    assert result.actions == ("approach_handle(drawer)", "grasp_handle(drawer)", "pull_open(drawer)")
    assert_valid(domain, DRAWER_START, result, *DRAWER_GOAL)


def test_c_a_long_library_plan_beats_a_short_hole():
    chain = [act(f"step{i}", {f"s{i}"}, {f"s{i + 1}"}, {f"s{i}"}) for i in range(5)]
    domain = Domain(actions=chain, scored=frozenset({"s5"}))
    assert exact(domain, {"s0"}, {"s5"}).actions == tuple(f"step{i}" for i in range(5))


# D. A missing primitive capability

def test_d_the_drawer_learns_the_grasp_and_reuses_pull_open():
    domain = dd.domain()
    result = exact(domain, DRAWER_START, *DRAWER_GOAL)
    assert result.actions == ("approach_handle(drawer)", f"HOLE[{GRASPED}]", "pull_open(drawer)")
    assert result.alternatives == ()
    assert_valid(domain, DRAWER_START, result, *DRAWER_GOAL)
    assert result.hole_contexts[0].used_by == "pull_open(drawer)"
    assert result.hole_contexts[0].library_alternatives == ()    # nothing in the library grasps


def test_d_stow_gets_the_plan_the_default_gets():
    domain = sd.domain()
    result = exact(domain, sd.stowed_state(), *STOW_GOAL)
    assert result.actions == STOW_PLAN
    assert result.cost == plan(domain, sd.stowed_state(), *STOW_GOAL).cost
    assert result.alternatives == ()
    assert_valid(domain, sd.stowed_state(), result, *STOW_GOAL)


@pytest.mark.parametrize("domain, start, goal", [
    (sd.domain(), sd.stowed_state(), STOW_GOAL),
    (dd.domain(), DRAWER_START, DRAWER_GOAL),
], ids=["stow", "drawer"])
def test_d_both_modes_give_rl_the_same_learning_problem(domain, start, goal):
    assert (hole_spec(domain, start, exact(domain, start, *goal), *goal)
            == hole_spec(domain, start, plan(domain, start, *goal), *goal))


def test_d_a_chain_is_repaired_at_its_root():
    """`make_p` can serve `use_p` once it has r, so r is what to learn. The shorter
    HOLE[p] -> use_p would learn a way around `make_p`."""
    domain = Domain(actions=[act("use_p", {"p"}, {"g"}), act("make_p", {"r"}, {"p"})],
                    scored=frozenset({"p", "r", "g"}))
    assert exact(domain, (), {"g"}).actions == ("HOLE[r]", "make_p", "use_p")
    assert exact(domain, (), {"g"}).actions == plan(domain, frozenset(), {"g"}).actions


def test_d_conjunctive_prerequisites_are_learned_not_their_product():
    domain = Domain(actions=[act("make", {"q", "r"}, {"p"})], scored=frozenset({"p", "q", "r"}))
    result = exact(domain, (), {"p"})
    assert result.delegated == ("q", "r") and result.actions[-1] == "make"
    assert_valid(domain, (), result, {"p"})


# E. A goal nothing establishes

def test_e_a_goal_nothing_establishes_gets_a_direct_hole():
    domain = Domain(actions=[], scored=frozenset({"p"}))
    result = exact(domain, (), {"p"})
    assert result.actions == ("HOLE[p]",)
    assert result.hole_contexts[0].used_by == "goal" and result.hole_contexts[0].waiting == ("goal",)


def test_e_the_drawer_learns_the_opening_when_nothing_can_pull():
    domain = drawer_without_pull()
    result = exact(domain, DRAWER_START, *DRAWER_GOAL)
    assert result.actions == ("approach_handle(drawer)", f"HOLE[{OPEN}]")
    assert_valid(domain, DRAWER_START, result, *DRAWER_GOAL)
    assert (hole_spec(domain, DRAWER_START, result, *DRAWER_GOAL)
            == hole_spec(domain, DRAWER_START, plan(domain, DRAWER_START, *DRAWER_GOAL), *DRAWER_GOAL))


def test_e_a_declared_implication_counts_as_a_producer():
    """x implies the goal, so `make_x` establishes it; its prerequisite is what to learn."""
    domain = Domain(actions=[act("make_x", {"m"}, {"x"})], scored=frozenset({"g", "m", "x"}),
                    implications={"x": frozenset({"g"})})
    assert exact(domain, (), {"g"}).actions == ("HOLE[m]", "make_x")


def test_e_one_hole_can_satisfy_what_its_literal_implies():
    domain = Domain(actions=[], scored=frozenset({"p", "q"}), implications={"p": frozenset({"q"})})
    assert exact(domain, (), {"p", "q"}).actions == ("HOLE[p]",)


# F. Circular dependencies

def test_f_a_pure_cycle_gets_a_hole():
    domain = Domain(actions=[act("p_from_q", {"q"}, {"p"}), act("q_from_p", {"p"}, {"q"})],
                    scored=frozenset({"p", "q"}))
    result = exact(domain, (), {"p"})
    assert result.actions == plan(domain, frozenset(), {"p"}).actions == ("HOLE[p]",)
    # q_from_p needs p, and p_from_q needs q: neither can replace a hole for the other.
    reasons = dict(result.hole_contexts[0].library_alternatives)
    assert "cannot run here" in reasons["p_from_q"]


def test_f_a_cycle_with_a_way_in_is_entered_from_outside():
    domain = Domain(actions=[act("p_from_q", {"q"}, {"p"}), act("q_from_p", {"p"}, {"q"}),
                             act("q_from_r", {"r"}, {"q"})], scored=frozenset({"p", "q", "r"}))
    assert exact(domain, (), {"p"}).actions == ("HOLE[r]", "q_from_r", "p_from_q")


def test_f_a_cycle_that_leads_into_another_cycle_is_broken_at_the_last():
    domain = Domain(actions=[act("p_from_q", {"q"}, {"p"}), act("q_from_p", {"p"}, {"q"}),
                             act("q_from_x", {"x"}, {"q"}), act("x_from_y", {"y"}, {"x"}),
                             act("y_from_x", {"x"}, {"y"})], scored=frozenset({"p", "q", "x", "y"}))
    assert exact(domain, (), {"p"}).actions == plan(domain, frozenset(), {"p"}).actions


def test_f_stows_cycle_runs_through_a_literal_that_cannot_be_learned():
    """clear_of_walls(A) is established only by placing A, and A is picked only once clear.
    So it stays the hole, although three library actions establish it."""
    result = exact(sd.domain(), sd.stowed_state(), *STOW_GOAL)
    context = result.hole_contexts[0]
    assert context.used_by == "pick(cube_a)" and "goal" in context.waiting
    assert {name for name, _ in context.library_alternatives} == {
        "place(cube_a,src_a)", "place(cube_a,src_b)", "place(cube_a,mouth)"}
    assert all("cannot run here" in why for _, why in context.library_alternatives)


def test_f_an_action_that_needs_its_own_effect_does_not_establish_it():
    """Regression. `keep_b` needs b and adds b, so it cannot be what establishes b. Counted as
    a way to get b, it made b and f each look reducible to the other, and no plan passed."""
    domain = Domain(actions=[act("keep_b", {"b"}, {"b", "f"}), act("use_f", {"c", "f"}, {"b", "f"})],
                    scored=frozenset({"b", "f"}), implications={"b": frozenset({"f"})})
    result = exact(domain, {"c"}, {"b", "c"})
    assert result.actions == ("HOLE[f]", "use_f")
    assert_valid(domain, {"c"}, result, {"b", "c"})


def test_f_an_action_that_needs_its_own_effect_is_not_a_cycle():
    """A difference between the modes. `make_f` needs f, so nothing can establish f. The default
    planner sees d and f as a cycle and delegates d. Here `make_d` is waiting for f, which cannot be
    learned, so it reports no plan, as for any other goal producer waiting on a prerequisite."""
    domain = Domain(actions=[act("make_f", {"d", "f"}, {"f"}), act("make_d", {"f"}, {"d"})],
                    scored=frozenset({"d"}))
    assert plan(domain, frozenset(), {"d"}).actions == ("HOLE[d]",)
    result = exact(domain, (), {"d"})
    assert isinstance(result, ExactNoPlan) and result.replaceable_holes == ("d",)


# G. Literals without a soft score

def test_g_a_literal_without_a_score_is_never_delegated():
    domain = Domain(actions=[act("use", {"k"}, {"g"})], scored=frozenset({"g"}))
    result = exact(domain, (), {"g"})
    assert isinstance(result, ExactNoPlan)
    assert result.unsynthesizable_needs == ("k",)
    assert result.unmet_needs == ("g", "k")


def test_g_a_goal_without_a_score_gets_no_hole():
    result = exact(Domain(actions=[], scored=frozenset()), (), {"g"})
    assert isinstance(result, ExactNoPlan) and result.candidates == ()
    assert result.unsynthesizable_needs == ("g",)
    assert not result.feasible_with_all_holes    # exact: these holes cannot solve this at any length


def test_g_stow_without_the_score_reports_it_and_delegates_nothing_else():
    """As in the default planner: no hole for clear_of_walls(A) means no plan, not a hole for
    another literal that the library has an action for."""
    domain = sd.domain()
    domain.scored = domain.scored - {"clear_of_walls(cube_a)"}
    result = exact(domain, sd.stowed_state(), *STOW_GOAL)
    assert isinstance(result, ExactNoPlan)
    assert result.unsynthesizable_needs == ("clear_of_walls(cube_a)", "holding(cube_a)")
    assert "in_region(cube_a,src_a)" in result.candidates


# H. Independence from the relaxed analysis

@pytest.mark.parametrize("domain, start, goal, expected", [
    (sd.domain(), sd.stowed_state(), STOW_GOAL, STOW_PLAN),
    (dd.domain(), DRAWER_START, DRAWER_GOAL,
     ("approach_handle(drawer)", f"HOLE[{GRASPED}]", "pull_open(drawer)")),
    (delete_interaction(), frozenset({"p"}), ({"g"}, set()), ("make_q", "HOLE[p]", "use")),
], ids=["stow", "drawer", "delete_interaction"])
def test_h_the_exact_mode_never_calls_the_relaxed_analysis(monkeypatch, domain, start, goal, expected):
    def forbidden(*args, **kwargs):
        raise AssertionError("the relaxed analysis was called")

    monkeypatch.setattr(hole_planner, "relaxed_reachable", forbidden)
    monkeypatch.setattr(hole_planner, "capability_gaps", forbidden)
    with pytest.raises(AssertionError, match="relaxed analysis was called"):
        plan(domain, start, *goal)          # the patch is live: the default mode needs them
    assert plan(domain, start, *goal, mode="exact").actions == expected


def _names_in_code(module) -> set[str]:
    """Every name and string the module's code uses, docstrings aside."""
    tree = ast.parse(inspect.getsource(module))
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
                  and node.body and isinstance(node.body[0], ast.Expr)}
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            names.add(node.value)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name)
    return names


def test_h_the_exact_module_does_not_even_name_the_relaxed_analysis():
    names = _names_in_code(exact_hole_planner)
    assert "relaxed_reachable" not in names and "capability_gaps" not in names


def test_h_the_exact_module_names_no_domain_vocabulary():
    words = {"tcp_near", "in_region", "clear_of_walls", "holding", "gripper", "cube",
             "pocket", "mouth", "drawer", "handle"}
    assert not {w for w in words for name in _names_in_code(exact_hole_planner) if w in name}


# I. Execution and provenance

def two_part_assembly() -> Domain:
    """`assemble` needs both parts fixed, and nothing fixes a part. The robot can be near one
    part at a time, and being near a part is what a hole for it starts from."""
    return Domain(
        actions=[act("approach(o1)", add={"near(o1)"}), act("approach(o2)", add={"near(o2)"}),
                 act("assemble", {"fixed(o1)", "fixed(o2)"}, {"done"})],
        scored=frozenset({"fixed(o1)", "fixed(o2)", "near(o1)", "near(o2)", "done"}),
        robot_relative=frozenset({"near(o1)", "near(o2)"}),
        mutex_groups=[frozenset({"near(o1)", "near(o2)"})])


def test_i_two_holes_each_get_their_own_context_and_learning_problem():
    domain = two_part_assembly()
    result = exact(domain, (), {"done"})
    assert result.delegated == ("fixed(o1)", "fixed(o2)")
    assert_valid(domain, (), result, {"done"})
    assert [c.used_by for c in result.hole_contexts] == ["assemble", "assemble"]
    first, second = (hole_spec(domain, frozenset(), result, {"done"}, index=i) for i in (0, 1))
    assert first["start"] == ["near(o1)"] and second["start"] == ["near(o2)"]
    assert second["preserve"] == [first["achieve"]]


def test_i_a_negative_goal_needs_a_remover_and_its_prerequisite():
    domain = Domain(actions=[act("clear_x", {"k"}, (), {"x"})], scored=frozenset({"k", "x"}))
    assert not relevant_actions(Domain(actions=domain.actions), set(), set()), "nothing is wanted"
    result = exact(domain, {"x"}, (), {"x"})
    assert result.actions == ("HOLE[k]", "clear_x")
    states = assert_valid(domain, {"x"}, result, (), {"x"})
    assert "x" not in states[-1]
    assert result.hole_contexts[0].used_by == "clear_x"


def test_i_a_hole_can_remove_a_negative_goal_through_a_mutex():
    domain = dd.domain()
    result = exact(domain, DRAWER_START, *DRAWER_GOAL)
    assert CLOSED in result.hole_contexts[0].state_before
    assert CLOSED not in replay(domain, DRAWER_START, result.steps)[0][-1]


def test_i_a_negative_only_goal_with_no_library_gets_no_hole():
    """LIMITATION, shared with the default planner: a hole only makes a literal true, and a
    hole for a mutex partner would be a delete in disguise, so none is offered."""
    domain = Domain(actions=[], scored=frozenset({"x", "y"}), mutex_groups=[frozenset({"x", "y"})])
    assert isinstance(exact(domain, {"x"}, (), {"x"}), ExactNoPlan)


def test_i_a_disturbance_orders_the_plan_as_in_the_default():
    """Freeing A sweeps through the mouth, so B is moved out of the way first."""
    domain = sd.domain()
    states = assert_valid(domain, sd.stowed_state(),
                          exact(domain, sd.stowed_state(), *STOW_GOAL), *STOW_GOAL)
    assert exact(domain, sd.stowed_state(), *STOW_GOAL).collateral == 0
    hole_index = STOW_PLAN.index("HOLE[clear_of_walls(cube_a)]")
    assert "in_region(cube_b,mouth)" not in states[hole_index]
    blind = exact(sd.domain(interference=False), sd.stowed_state(), *STOW_GOAL)
    assert blind.alternatives != ()     # without the model, the ordering is undetermined


def test_i_a_disturbance_a_plan_cannot_avoid_is_counted():
    domain = Domain(actions=[act("use", {"p"}, {"g"})], scored=frozenset({"p"}),
                    disturbances=[Disturbance("p", frozenset({"object_in_place"}))])
    result = exact(domain, {"object_in_place"}, {"g"})
    assert result.actions == ("HOLE[p]", "use") and result.collateral == 1
    assert_valid(domain, {"object_in_place"}, result, {"g"})
    assert "object_in_place" not in result.hole_contexts[0].state_before - {"object_in_place"}


@pytest.mark.parametrize("name, domain, start, goal_pos, goal_neg", [
    ("stow", sd.domain(), sd.stowed_state(), *STOW_GOAL),
    ("drawer", dd.domain(), DRAWER_START, *DRAWER_GOAL),
    ("two parts", two_part_assembly(), frozenset(), {"done"}, set()),
    ("delete interaction", delete_interaction(scored=("p", "g")), frozenset({"p"}), {"g"}, set()),
], ids=lambda v: v if isinstance(v, str) else "")
def test_i_specs_and_fences_hold_along_the_replayed_plan(name, domain, start, goal_pos, goal_neg):
    result = exact(domain, start, goal_pos, goal_neg)
    states, _ = replay(domain, start, result.steps)
    for index, i in enumerate(result.hole_indices):
        spec = hole_spec(domain, start, result, goal_pos, goal_neg, index=index)
        assert spec["hole_index"] == i and spec["achieve"] == result.steps[i].hole_for
        assert set(spec["start"]) <= states[i]
        assert set(spec["preserve"]) <= states[i] and set(spec["preserve"]) <= states[i + 1]
        assert not (set(spec["preserve_absent"]) & states[i + 1])
        assert [Action.from_dict(a).name for a in spec["prefix"]] == list(result.actions[:i])
        assert [Action.from_dict(a).name for a in spec["suffix"]] == list(result.actions[i + 1:])


def test_i_a_context_serializes_for_inspection():
    context = exact(delete_interaction(scored=("q", "g")), {"p"}, {"g"}).hole_contexts[0]
    assert context.to_dict() == {
        "literal": "q", "index": 0, "state_before": ["p"], "used_by": "use", "used_at": 1,
        "waiting": ["use"], "prefix": [], "suffix": ["use"],
        "library_alternatives": [["make_q", "the plan then fails at use"]]}


def test_i_a_hole_used_only_for_its_side_effect_is_rejected():
    """Only a hole for p can remove x, through the mutex, and p is offered because `alt_g`
    would like it. But `alt_g` can never run, and `side` is no use to the goal, so nothing
    relevant uses p. A hole is not a way to delete a literal."""
    domain = Domain(actions=[act("make_g", (), {"g"}), act("alt_g", {"p", "z"}, {"g"}),
                             act("side", {"p"}, {"w"})],
                    scored=frozenset({"p"}), mutex_groups=[frozenset({"p", "x"})])
    assert isinstance(exact(domain, {"x"}, {"g"}, {"x"}), ExactNoPlan)


# J. Search bounds

def test_j_the_length_limit_is_respected_and_named():
    domain = dd.domain()
    assert exact(domain, DRAWER_START, *DRAWER_GOAL, max_length=3).cost[4] == 3
    result = exact(domain, DRAWER_START, *DRAWER_GOAL, max_length=2)
    assert isinstance(result, ExactNoPlan)
    assert result.limits_reached == ("max_length",) and result.max_length == 2


def test_j_the_delegation_limit_is_respected_and_named():
    domain = two_part_assembly()
    assert exact(domain, (), {"done"}, max_delegated=2).delegated == ("fixed(o1)", "fixed(o2)")
    result = exact(domain, (), {"done"}, max_delegated=1)
    assert isinstance(result, ExactNoPlan)
    assert "max_delegated" in result.limits_reached and result.max_delegated == 1


def test_j_no_holes_allowed_means_the_library_alone():
    result = exact(sd.domain(), sd.stowed_state(), *STOW_GOAL, max_delegated=0)
    assert isinstance(result, ExactNoPlan) and "max_delegated" in result.limits_reached
    domain = dd.domain(extra_actions=[dd.scripted_grasp()])
    assert exact(domain, DRAWER_START, *DRAWER_GOAL, max_delegated=0).delegated == ()


def test_j_a_failure_says_when_no_limit_was_involved():
    """Nothing cut this search short. It still only says no plan passes the rules for holes."""
    result = exact(delete_interaction(scored=("g",)), {"p"}, {"g"})
    assert result.limits_reached == () and result.candidates == ("g",)


def test_j_a_plan_says_whether_the_step_limit_could_have_hidden_a_cheaper_one():
    """Learning c alone takes four steps. Within three, the planner can only learn a and b, and
    it says the limit cut something off. Without the cut, the cost holds at any length."""
    domain = Domain(actions=[act("use_ab", {"a", "b"}, {"g"}), act("s1", {"c"}, {"d"}),
                             act("s2", {"d"}, {"e"}), act("s3", {"e"}, {"g"})],
                    scored=frozenset({"a", "b", "c"}))
    tight = exact(domain, (), {"g"}, max_length=3)
    assert tight.delegated == ("a", "b") and "max_length" in tight.limits_reached
    wide = exact(domain, (), {"g"}, max_length=4)
    assert wide.delegated == ("c",) and "max_length" not in wide.limits_reached
    assert exact(domain, (), {"g"}, max_length=20).cost == wide.cost


def test_j_a_shorter_way_to_an_explored_state_is_still_explored():
    """The default planner's step-limit regression, in this mode too."""
    long_way = [act("long1", {"start"}, {"step1"}, {"start", "object_in_place"})]
    long_way += [act(f"long{i}", {f"step{i - 1}"}, {f"step{i}"}, {f"step{i - 1}"}) for i in range(2, 8)]
    long_way.append(act("long8", {"step7"}, {"middle"}, {"step7"}))
    finish = [act("finish1", {"middle"}, {"end1"}, {"middle"}), act("finish2", {"end1"}, {"end2"}, {"end1"}),
              act("finish3", {"end2"}, {"goal"}, {"end2"})]
    domain = Domain(actions=[act("shortcut", {"start"}, {"middle"}, {"start"}), *long_way, *finish],
                    disturbances=[Disturbance("middle", frozenset({"object_in_place"}))])
    start = {"start", "object_in_place"}
    assert exact(domain, start, {"goal"}).actions == ("shortcut", "finish1", "finish2", "finish3")
    assert exact(domain, start, {"goal"}, max_length=11).cost == (0, 0, 0, 0, 11)


# Hole starts, without the relaxation

def test_hole_starts_are_the_defaults_on_stow_and_the_drawer():
    for domain, start in ((sd.domain(), sd.stowed_state()), (dd.domain(), DRAWER_START)):
        for key in sorted(domain.scored):
            default = domain.hole(key, relaxed_reachable(domain, start))
            assert hole_starts(domain, key, library_states(domain, start)) == [default], key


def test_an_approach_cannot_be_skipped():
    """The hole starts from the standoff the library can reach, so the plan must go there."""
    result = exact(dd.domain(), DRAWER_START, *DRAWER_GOAL)
    assert result.steps[1].pre == {STANDOFF} and result.actions[0] == "approach_handle(drawer)"


def test_facts_that_exclude_each_other_give_one_start_each():
    """The default planner's limitation: its start asks for both near and far, which can
    never hold, so it finds no plan. Here each is a start of its own, and neither is skipped."""
    domain = Domain(actions=[act("go_near", add={"near(o)"}), act("go_far", add={"far(o)"})],
                    scored=frozenset({"done(o)"}), robot_relative=frozenset({"near(o)", "far(o)"}),
                    mutex_groups=[frozenset({"near(o)", "far(o)"})])
    assert isinstance(plan(domain, frozenset(), {"done(o)"}), NoPlan)
    assert domain.hole("done(o)", relaxed_reachable(domain, frozenset())).pre == {"near(o)", "far(o)"}

    starts = hole_starts(domain, "done(o)", library_states(domain, frozenset()))
    assert [hole.pre for hole in starts] == [{"far(o)"}, {"near(o)"}]
    result = exact(domain, (), {"done(o)"})
    assert result.cost == (1, 0, 1, 1, 2)
    assert set(result.alternative_names) | {result.actions} == {
        ("go_near", "HOLE[done(o)]"), ("go_far", "HOLE[done(o)]")}
    assert_valid(domain, (), result, {"done(o)"})


def test_facts_the_library_holds_only_apart_are_not_a_conjunction():
    """No mutex is declared here, but going near one side loses the other, so the library
    never holds both. Reading the real states finds that without being told."""
    domain = Domain(actions=[act("left", (), {"l(o)"}, {"r(o)"}), act("right", (), {"r(o)"}, {"l(o)"})],
                    scored=frozenset({"done(o)"}), robot_relative=frozenset({"l(o)", "r(o)"}))
    starts = hole_starts(domain, "done(o)", library_states(domain, frozenset()))
    assert [hole.pre for hole in starts] == [{"l(o)"}, {"r(o)"}]


def test_a_fact_only_the_relaxation_can_reach_is_not_a_start():
    """A difference between the modes, on purpose. The relaxation says `prep` can run, so the
    default's hole must wait for near(o), which never comes, and it finds no plan. The
    library really cannot get there, so here the hole does not wait for it."""
    domain = Domain(actions=[act("make_b", {"a"}, {"b"}, {"a"}), act("prep", {"a", "b"}, {"near(o)"})],
                    scored=frozenset({"fixed(o)"}), robot_relative=frozenset({"near(o)"}))
    start = frozenset({"a"})
    assert domain.hole("fixed(o)", relaxed_reachable(domain, start)).pre == {"near(o)"}
    assert isinstance(plan(domain, start, {"fixed(o)"}), NoPlan)
    result = exact(domain, start, {"fixed(o)"})
    assert result.actions == ("HOLE[fixed(o)]",) and result.steps[0].pre == frozenset()


def test_a_tight_length_limit_does_not_let_a_hole_skip_its_preparation():
    """Regression. The standoff takes three library steps. A limit too short to get there must
    mean no plan, not a hole that starts without it: what a hole starts from does not depend
    on the search's limits."""
    domain = Domain(actions=[act("go1", (), {"s1"}), act("go2", {"s1"}, {"s2"}), act("go3", {"s2"}, {"near(o)"})],
                    scored=frozenset({"fixed(o)"}), robot_relative=frozenset({"near(o)"}))
    assert exact(domain, (), {"fixed(o)"}).actions == ("go1", "go2", "go3", "HOLE[fixed(o)]")
    tight = exact(domain, (), {"fixed(o)"}, max_length=2)
    assert isinstance(tight, ExactNoPlan) and tight.limits_reached == ("max_length",)


def test_a_hole_is_usable_only_while_its_literal_is_missing():
    """p implies q, but the start holds p without q. Running a hole for p anyway would add q. It
    is not offered in the search, and the exact checks must agree."""
    domain = Domain(actions=[act("make_q", {"p", "r"}, {"q"})], scored=frozenset({"p"}),
                    implications={"p": frozenset({"q"})})
    result = exact(domain, {"p"}, {"q"})
    assert isinstance(result, ExactNoPlan) and not result.feasible_with_all_holes


def test_a_hole_starts_only_from_facts_about_its_object():
    domain = two_part_assembly()
    assert hole_starts(domain, "fixed(o1)", library_states(domain, frozenset()))[0].pre == {"near(o1)"}
    assert hole_starts(domain, "done", library_states(domain, frozenset()))[0].pre == frozenset()


# Library actions that can still do the job

def test_a_pointless_step_cannot_save_a_hole():
    """Regression. `junk` runs only at the standoff, which `pull` leaves. Had the replay been
    forced to run every step, junk would fail there, and that would excuse a hole for the
    opening, at the same cost as the hole for the grasp."""
    domain = Domain(
        actions=[act("approach", add={"standoff(d)"}), act("pull", {"grasped(d)"}, {"open(d)"}, {"grasped(d)"}),
                 act("junk", {"standoff(d)"}, {"w"})],
        scored=frozenset({"open(d)", "grasped(d)", "standoff(d)", "closed(d)"}),
        robot_relative=frozenset({"standoff(d)", "grasped(d)"}),
        mutex_groups=[frozenset({"open(d)", "closed(d)"}), frozenset({"standoff(d)", "grasped(d)"})])
    result = exact(domain, {"closed(d)"}, {"open(d)"}, {"closed(d)"})
    assert result.actions == ("approach", "HOLE[grasped(d)]", "pull") and result.alternatives == ()


def test_a_library_action_may_take_over_after_later_steps():
    """Regression. `make_p` destroys the z that `x` needs, so it cannot sit where a first hole
    for p would. That does not make it unusable: it can run after x. So r is what to learn."""
    domain = Domain(actions=[act("x", {"z"}, {"w"}), act("make_p", {"r"}, {"p"}, {"z"}),
                             act("use", {"p", "w"}, {"g"})], scored=frozenset({"p", "r"}))
    result = exact(domain, {"z"}, {"g"})
    assert result.delegated == ("r",) and result.actions[-2:] == ("make_p", "use")
    assert_valid(domain, {"z"}, result, {"g"})


def test_several_holes_for_a_literal_are_one_thing_to_learn():
    """Regression. `make_p` could replace the first hole for p here, but not the second: x
    consumes p, and making it again would destroy the z `finish` needs. p must be learned
    anyway, so the plan stands. Asking about each hole on its own would reject it."""
    domain = Domain(actions=[act("use_p", {"p"}, {"m"}), act("x", {"m"}, {"z"}, {"p"}),
                             act("finish", {"p", "z"}, {"g"}), act("make_p", {"r"}, {"p"}, {"z"})],
                    scored=frozenset({"p"}))
    result = exact(domain, (), {"g"})
    assert result.actions == ("HOLE[p]", "use_p", "x", "HOLE[p]", "finish")
    assert result.delegated == ("p",) and result.cost == (1, 0, 2, 2, 5)
    assert_valid(domain, (), result, {"g"})


def test_a_library_action_that_is_only_blocked_still_counts():
    """`make_p` cannot run while busy holds, and a hole for p would be shorter. But being given
    what it needs includes clearing what it needs absent, so it still counts, and the plan
    learns what clears the way."""
    domain = Domain(actions=[act("use", {"p"}, {"g"}), act("make_p", (), {"p"}, (), {"busy"}),
                             act("free", {"key"}, (), {"busy"})],
                    scored=frozenset({"p", "key"}))
    result = exact(domain, {"busy"}, {"g"})
    assert result.actions == ("HOLE[key]", "free", "make_p", "use")
    assert_valid(domain, {"busy"}, result, {"g"})


# Alternative action orderings

def test_ordering_a_library_action_can_run_before_the_hole_that_would_block_it():
    """`make_a` destroys d, so it looks useless once a hole for d has run. It works before that
    hole, so f is what to learn, not a. The cheaper HOLE[d] -> HOLE[a] -> finish replaces it."""
    domain = Domain(actions=[act("finish", {"a", "d"}, {"g"}), act("make_a", {"f"}, {"a"}, {"d"})],
                    scored=frozenset({"a", "d", "f"}))
    result = exact(domain, (), {"g"})
    assert result.actions == ("HOLE[f]", "make_a", "HOLE[d]", "finish")
    assert result.actions == plan(domain, frozenset(), {"g"}).actions
    assert_valid(domain, (), result, {"g"})


def test_ordering_no_hole_when_another_order_of_the_library_works():
    """`make_p` destroys the z that `x` needs. Run in that order the library is stuck, and a
    hole for p would look needed. Run the other way it needs no hole at all."""
    domain = Domain(actions=[act("make_p", (), {"p"}, {"z"}), act("x", {"z"}, {"w"}),
                             act("use", {"p", "w"}, {"g"})], scored=frozenset({"p"}))
    result = exact(domain, {"z"}, {"g"})
    assert result.actions == ("x", "make_p", "use") and result.delegated == ()


def test_ordering_a_tight_step_limit_does_not_manufacture_a_hole():
    """The library reaches the goal in four steps, so p is not necessary at any length. A limit
    of three fits HOLE[p] -> use_p, but the answer must be no plan and the limit named."""
    domain = Domain(actions=[act("use_p", {"p"}, {"g"}), act("s1", (), {"a"}), act("s2", {"a"}, {"b"}),
                             act("s3", {"b"}, {"c"}), act("s4", {"c"}, {"g"})], scored=frozenset({"p"}))
    assert exact(domain, (), {"g"}, max_length=4).actions == ("s1", "s2", "s3", "s4")
    tight = exact(domain, (), {"g"}, max_length=3)
    assert isinstance(tight, ExactNoPlan) and tight.limits_reached == ("max_length",)
    assert tight.unnecessary_holes == ("p",) and tight.feasible_with_all_holes


# Restoration of consumed facts

def restoration(scored, restorer_deletes=()) -> Domain:
    """`make_q` consumes the p that `use` also needs. `make_p` restores p, but needs r."""
    return Domain(actions=[act("make_q", {"p"}, {"q"}, {"p"}), act("use", {"p", "q"}, {"g"}),
                           act("make_p", {"r"}, {"p"}, restorer_deletes)], scored=frozenset(scored))


def test_restoration_by_the_library_alone_needs_no_hole():
    domain = Domain(actions=[act("make_q", {"p"}, {"q"}, {"p"}), act("use", {"p", "q"}, {"g"}),
                             act("make_p", (), {"p"})], scored=frozenset({"p", "q", "g"}))
    assert exact(domain, {"p"}, {"g"}).actions == ("make_q", "make_p", "use")


def test_restoration_is_preferred_when_what_it_needs_can_be_learned():
    """Learning r keeps every action in the library, so q, p and g are all refused."""
    domain = restoration({"p", "q", "r", "g"})
    result = exact(domain, {"p"}, {"g"})
    assert result.actions == ("make_q", "HOLE[r]", "make_p", "use")
    assert_valid(domain, {"p"}, result, {"g"})


def test_restoration_that_cannot_be_learned_falls_back_to_the_consumed_fact():
    """r cannot be learned, so `make_p` can never run. Learning p itself still keeps `make_q`."""
    assert exact(restoration({"p", "q", "g"}), {"p"}, {"g"}).actions == ("make_q", "HOLE[p]", "use")


def test_restoration_that_cannot_really_happen_is_not_assumed():
    """Nothing can restore p here, by real transitions, so `make_q` really is unusable and q is
    what to learn. Assuming `make_p` could be given r would have refused this hole and found nothing."""
    assert exact(restoration({"q", "g"}), {"p"}, {"g"}).actions == ("HOLE[q]", "use")


def test_restoration_that_destroys_what_it_restores_for_is_not_a_repair():
    """`make_p` deletes the q it restores p for, so learning r solves nothing. Real transitions
    show that. p is what to learn."""
    domain = restoration({"p", "q", "r", "g"}, restorer_deletes={"q"})
    assert exact(domain, {"p"}, {"g"}).actions == ("make_q", "HOLE[p]", "use")
    assert not Oracle(domain, {"p"}, {"g"}).solves({"r"})


def test_restoration_of_a_goal_literal_a_producer_destroys():
    """`make_p` destroys z, which the goal needs, and `make_z` restores it from y. So y is what
    to learn, and p, which `make_p` establishes, is not."""
    domain = Domain(actions=[act("use_p", {"p"}, {"g"}), act("make_p", (), {"p"}, {"z"}),
                             act("make_z", {"y"}, {"z"})], scored=frozenset({"p", "y"}))
    result = exact(domain, {"z"}, {"g", "z"})
    assert result.delegated == ("y",) and set(result.actions) == {"make_p", "use_p", "HOLE[y]", "make_z"}
    assert_valid(domain, {"z"}, result, {"g", "z"})


# Mutually destructive goal effects

def test_destructive_goal_producers_get_a_hole_that_resolves_the_interference():
    """`make_g` destroys h and `make_h` destroys g, so the library can never hold both. Learning
    s, what they share, does not help: real transitions show they would still fight. So either
    goal literal may be learned, and the cost finds them equal."""
    domain = Domain(actions=[act("make_g", {"s"}, {"g"}, {"s", "h"}),
                             act("make_h", {"s"}, {"h"}, {"s", "g"})],
                    scored=frozenset({"g", "h", "s"}))
    assert not Oracle(domain, {"s"}, {"g", "h"}).solves({"s"})
    result = exact(domain, {"s"}, {"g", "h"})
    assert set(result.alternative_names) | {result.actions} == {
        ("make_g", "HOLE[h]"), ("make_h", "HOLE[g]")}
    assert_valid(domain, {"s"}, result, {"g", "h"})


def test_destructive_goal_holes_only_if_the_shared_prerequisite_cannot_be_learned():
    """If the token can be learned, that repair solves it, and goal holes are refused (see
    test_b_a_consumed_shared_resource). If it cannot, a goal hole resolves the interference."""
    domain = Domain(actions=[act("make_p", {"token"}, {"p"}, {"token"}),
                             act("make_q", {"token"}, {"q"}, {"token"})], scored=frozenset({"p", "q"}))
    result = exact(domain, {"token"}, {"p", "q"})
    assert len(result.delegated) == 1 and result.cost == (1, 0, 1, 1, 2)
    assert_valid(domain, {"token"}, result, {"p", "q"})


def test_destructive_a_repair_for_the_other_goal_literal_counts_too():
    """`make_g` and `make_h` use up the one token, so they cannot both run. `safe_h` makes h without
    it, given k. k has nothing to do with g, but learning it keeps every action, so no goal hole
    is allowed, although make_h -> HOLE[g] is cheaper."""
    domain = Domain(actions=[act("make_g", {"t"}, {"g"}, {"t"}), act("make_h", {"t"}, {"h"}, {"t"}),
                             act("safe_h", {"k"}, {"h"})], scored=frozenset({"g", "h", "k"}))
    result = exact(domain, {"t"}, {"g", "h"})
    assert result.delegated == ("k",) and result.actions[-1] == "safe_h"
    assert_valid(domain, {"t"}, result, {"g", "h"})


def test_destructive_the_destroyed_literal_is_learned_rather_than_the_producer_replaced():
    """Learning z keeps `make_g`, which HOLE[g] replaces, although HOLE[g] is cheaper. If z
    cannot be learned, `make_g` really runs and really destroys it, so HOLE[g] is allowed."""
    actions = [act("make_g", (), {"g"}, {"z"})]
    assert exact(Domain(actions=actions, scored=frozenset({"g", "z"})), {"z"}, {"g", "z"}).actions == (
        "make_g", "HOLE[z]")
    assert exact(Domain(actions=actions, scored=frozenset({"g"})), {"z"}, {"g", "z"}).actions == ("HOLE[g]",)


def test_destructive_a_circle_of_destroyers_has_no_unique_repair():
    """Each action destroys the next one's effect. Regression: if each literal counted as
    upstream of the one before it, all three would be refused and there would be no plan."""
    domain = Domain(actions=[act("make_a", (), {"a"}, {"b"}), act("make_b", (), {"b"}, {"c"}),
                             act("make_c", (), {"c"}, {"a"})], scored=frozenset({"a", "b", "c"}))
    result = exact(domain, (), {"a", "b", "c"})
    assert result.cost == (1, 0, 1, 1, 3)
    assert_valid(domain, (), result, {"a", "b", "c"})
    reported = {frozenset(s.hole_for for s in steps if s.hole_for) for steps in (result.steps, *result.alternatives)}
    assert reported == {frozenset({"a"}), frozenset({"b"}), frozenset({"c"})}


def test_destructive_interference_is_not_the_same_as_a_missing_prerequisite():
    """Both problems have one goal-achieving action per goal literal, and only goal literals are
    scored. Here the actions run but fight, so a goal hole is allowed. In
    test_b_the_goal_hole_is_not_a_fallback `use` never runs at all, so it is refused."""
    fight = Domain(actions=[act("make_g", (), {"g"}, {"h"}), act("make_h", (), {"h"}, {"g"})],
                   scored=frozenset({"g", "h"}))
    assert exact(fight, (), {"g", "h"}).delegated in (("g",), ("h",))
    assert isinstance(exact(delete_interaction(scored=("g",)), {"p"}, {"g"}), ExactNoPlan)


# Pure dependency cycles

def cycle(scored, implications=None) -> Domain:
    return Domain(actions=[act("p_from_q", {"q"}, {"p"}), act("q_from_p", {"p"}, {"q"})],
                  scored=frozenset(scored), implications=implications or {})


def test_cycle_every_member_is_a_valid_hole_and_the_cost_chooses():
    domain = cycle({"p", "q"})
    oracle = Oracle(domain, (), {"p"})
    assert oracle.minimal_sets() == {frozenset({"p"}), frozenset({"q"})}
    assert oracle.admissible({"p"}) and oracle.admissible({"q"})
    assert exact(domain, (), {"p"}).actions == ("HOLE[p]",)


def test_cycle_the_cost_can_choose_the_other_member():
    """p implies z, so a hole for p changes two literals and a hole for q changes one."""
    domain = cycle({"p", "q"}, implications={"p": frozenset({"z"})})
    assert exact(domain, (), {"p"}).actions == ("HOLE[q]", "p_from_q")


def test_cycle_a_member_without_a_score_leaves_the_other():
    assert exact(cycle({"q"}), (), {"p"}).actions == ("HOLE[q]", "p_from_q")


def test_cycle_of_three_can_be_broken_anywhere():
    domain = Domain(actions=[act("a_from_b", {"b"}, {"a"}), act("b_from_c", {"c"}, {"b"}),
                             act("c_from_a", {"a"}, {"c"})], scored=frozenset({"a", "b", "c"}))
    assert Oracle(domain, (), {"a"}).minimal_sets() == {frozenset({"a"}), frozenset({"b"}), frozenset({"c"})}
    assert exact(domain, (), {"a"}).actions == ("HOLE[a]",)
    assert exact(domain, (), {"a"}).actions == plan(domain, frozenset(), {"a"}).actions


# False rejection caused by counterfactual prerequisite supply

def test_false_rejection_prerequisites_the_library_never_holds_together():
    """`a` needs m and n, and making m destroys n, so `a` can never run. Handed m for nothing it
    would run, and that argued against a hole for p. The library has m: what fails is the delete."""
    domain = Domain(actions=[act("make_m", (), {"m"}, {"n"}), act("a", {"m", "n"}, {"p"}),
                             act("use", {"p"}, {"g"})], scored=frozenset({"p"}))
    result = exact(domain, {"n"}, {"g"})
    assert result.actions == ("HOLE[p]", "use")
    assert_valid(domain, {"n"}, result, {"g"})


def test_false_rejection_a_prerequisite_nothing_can_establish():
    """r can never be established, so `make_p` can never run, and p is really needed. The default
    planner reports no plan here: it allows holes only at r. A difference between the modes."""
    domain = Domain(actions=[act("use_p", {"p"}, {"g"}), act("make_p", {"r"}, {"p"})], scored=frozenset({"p"}))
    assert exact(domain, (), {"g"}).actions == ("HOLE[p]", "use_p")
    assert isinstance(plan(domain, frozenset(), {"g"}), NoPlan)


def test_false_rejection_what_learning_a_prerequisite_would_really_take():
    """A hole for r must start from near(o), and going near destroys the x `make_p` also needs. So
    learning r cannot work, although r is scored. Supplying it directly would hide that."""
    domain = Domain(actions=[act("go_near", (), {"near(o)"}, {"x"}), act("make_p", {"r(o)", "x"}, {"p(o)"}),
                             act("use", {"p(o)"}, {"g"})],
                    scored=frozenset({"r(o)", "p(o)"}), robot_relative=frozenset({"near(o)"}))
    assert not Oracle(domain, {"x"}, {"g"}).solves({"r(o)"})
    assert exact(domain, {"x"}, {"g"}).actions == ("go_near", "HOLE[p(o)]", "use")


def test_false_rejection_a_blocker_nothing_removes():
    """`make_p` is blocked for good, so it establishes nothing. A hole cannot clear the way, since
    holes only make things true, and nothing else can, so p is what to learn."""
    domain = Domain(actions=[act("use", {"p"}, {"g"}), act("make_p", (), {"p"}, (), {"busy"})],
                    scored=frozenset({"p"}))
    assert exact(domain, {"busy"}, {"g"}).actions == ("HOLE[p]", "use")


def test_false_rejection_a_goal_literal_whose_producer_is_blocked_for_good():
    """The same for a goal literal: this producer would not run even if everything it needs could
    be learned, so nothing establishes the literal."""
    domain = Domain(actions=[act("make_g", {"m"}, {"g"}, (), {"busy"})], scored=frozenset({"g"}))
    assert exact(domain, {"busy"}, {"g"}).actions == ("HOLE[g]",)


def test_false_rejection_an_order_that_cannot_help_without_its_prerequisite():
    """As in test_ordering_a_library_action_can_run_before_the_hole_that_would_block_it, but f
    cannot be learned, so `make_a` can never run and a really is needed."""
    domain = Domain(actions=[act("finish", {"a", "d"}, {"g"}), act("make_a", {"f"}, {"a"}, {"d"})],
                    scored=frozenset({"a", "d"}))
    result = exact(domain, (), {"g"})
    assert result.delegated == ("a", "d") and result.actions[-1] == "finish"
    assert isinstance(plan(domain, frozenset(), {"g"}), NoPlan)


# Subset-minimal holes and alternative causal repairs

def test_minimal_a_chain_has_three_minimal_sets_of_one_literal():
    """Removing one hole at a time cannot choose among these. The causal preference does."""
    domain = Domain(actions=[act("use_p", {"p"}, {"g"}), act("make_p", {"r"}, {"p"})],
                    scored=frozenset({"p", "r", "g"}))
    oracle = Oracle(domain, (), {"g"})
    assert oracle.minimal_sets() == {frozenset({"g"}), frozenset({"p"}), frozenset({"r"})}
    assert [literal for literal in ("g", "p", "r") if oracle.admissible({literal})] == ["r"]
    assert exact(domain, (), {"g"}).delegated == ("r",)


def test_minimal_the_drawer_has_two_and_the_cost_alone_would_choose_wrongly():
    domain = dd.domain()
    oracle = Oracle(domain, DRAWER_START, *DRAWER_GOAL)
    assert {frozenset({OPEN}), frozenset({GRASPED})} <= oracle.minimal_sets()
    result = exact(domain, DRAWER_START, *DRAWER_GOAL)
    assert result.delegated == (GRASPED,)
    # The plan the preference refused is shorter, so the cost alone would have taken it.
    assert plan(drawer_without_pull(), DRAWER_START, *DRAWER_GOAL).cost < result.cost


def test_minimal_a_larger_upstream_set_beats_a_smaller_one():
    """{p} has one literal and {q, r} has two, and both are minimal. The preference comes before
    the cost, whose first term is the number of literals."""
    domain = Domain(actions=[act("make", {"q", "r"}, {"p"})], scored=frozenset({"p", "q", "r"}))
    oracle = Oracle(domain, (), {"p"})
    assert oracle.minimal_sets() == {frozenset({"p"}), frozenset({"q", "r"})}
    assert exact(domain, (), {"p"}).delegated == ("q", "r")


def test_minimal_a_prerequisite_stays_upstream_if_making_it_destroys_its_user():
    """`make_x` needs y, and `make_y` destroys x, so the two depend on each other only through that
    delete. y is still what `make_x` needs, so it is what to learn. r cannot be learned."""
    domain = Domain(actions=[act("make_x", {"y"}, {"x"}), act("make_y", {"r"}, {"y"}, {"x"})],
                    scored=frozenset({"x", "y"}))
    assert exact(domain, (), {"x"}).actions == ("HOLE[y]", "make_x")


def test_minimal_the_cost_chooses_between_repairs_neither_of_which_is_upstream():
    """Two independent ways in. Both repairs are minimal and admissible, and the cost finds them
    equal, so both are reported."""
    domain = Domain(actions=[act("alt1", {"a"}, {"g"}), act("alt2", {"b"}, {"g"})],
                    scored=frozenset({"a", "b"}))
    result = exact(domain, (), {"g"})
    assert {result.actions, *result.alternative_names} == {("HOLE[a]", "alt1"), ("HOLE[b]", "alt2")}


def test_minimal_the_preference_can_overrule_the_cost():
    """Both repairs of the delete interaction are minimal. HOLE[q] is cheaper, but p is upstream
    of q, so it keeps `make_q`, which HOLE[q] would replace. This pins a design choice: it does
    not claim that p is the easier skill to learn."""
    domain = delete_interaction()
    oracle = Oracle(domain, {"p"}, {"g"})
    assert {frozenset({"p"}), frozenset({"q"})} <= oracle.minimal_sets()
    assert exact(domain, {"p"}, {"g"}).delegated == ("p",)
    assert exact(delete_interaction(scored=("q", "g")), {"p"}, {"g"}).cost < exact(domain, {"p"}, {"g"}).cost


def test_minimal_a_necessary_hole_can_still_be_refused():
    """Necessity and preference are separate. With only g scored, {g} is minimal: the goal cannot
    be reached without it. The preference still refuses it, and the report keeps the two apart."""
    domain = delete_interaction(scored=("g",))
    assert Oracle(domain, {"p"}, {"g"}).minimal_sets() == {frozenset({"g"})}
    result = exact(domain, {"p"}, {"g"})
    assert result.unnecessary_holes == () and result.replaceable_holes == ("g",)


# The search: paths may merge only if they can finish in the same ways

def test_paths_that_used_a_hole_differently_are_kept_apart():
    """Regression. Only a hole for p can remove x. After it, `a` (which needs x gone) and `c`
    (which needs p) lead to the same state, but only `c` uses p, so only that plan passes.
    Treated as one node, `a`, which comes first, hid `c`: no plan."""
    domain = Domain(actions=[act("a", (), {"m"}, (), {"x"}), act("c", {"p"}, {"m"}),
                             act("finish", {"m"}, {"g"})],
                    scored=frozenset({"p"}), mutex_groups=[frozenset({"p", "x"})])
    result = exact(domain, {"x"}, {"g"})
    assert result.actions == ("HOLE[p]", "c", "finish") and result.alternatives == ()


def test_each_set_of_holes_is_judged_once(monkeypatch):
    """Regression. A verdict depends only on the set of literals. Judged again at every goal
    arrival, stow's no-plan case took five seconds, nearly all of it in the dependency walk."""
    calls = []
    walk = exact_hole_planner.upstream

    def counted(*args):
        calls.append(args)
        return walk(*args)

    monkeypatch.setattr(exact_hole_planner, "upstream", counted)
    domain = sd.domain()
    domain.scored = domain.scored - {"clear_of_walls(cube_a)"}
    assert isinstance(exact(domain, sd.stowed_state(), *STOW_GOAL), ExactNoPlan)
    assert len(calls) < 40


def test_tied_plans_through_one_node_can_be_under_reported():
    """LIMITATION, shared with the default planner: equally cheap plans that merge at a node
    are reported as one. The chosen plan and its cost are unaffected. Here two orders tie and
    one is listed; on random problems about 7 percent of tied plans went unreported."""
    domain = Domain(actions=[act("a", (), {"x"}), act("b", (), {"y"}), act("c", {"x", "y"}, {"g"})])
    result = exact(domain, (), {"g"})
    assert result.cost == (0, 0, 0, 0, 3) and result.alternatives == ()
    assert len(Oracle(domain, (), {"g"}).best_plans(2, 5)[1]) == 2


def test_tied_orderings_are_reported_as_by_the_default():
    """Without the interference model, stow has three tied orderings, and both modes list two."""
    domain = sd.domain(interference=False)
    default = plan(domain, sd.stowed_state(), *STOW_GOAL)
    result = exact(domain, sd.stowed_state(), *STOW_GOAL)
    assert result.cost == default.cost and result.actions == default.actions
    assert len(default.alternatives) == len(result.alternatives) == 1
    assert len(Oracle(domain, sd.stowed_state(), *STOW_GOAL).best_plans(2, 6)[1]) == 3


# The search against an oracle that shares no code with it

class Step(NamedTuple):
    """One step of a plan the oracle is enumerating."""

    name: str
    hole: str | None
    pre: frozenset[str]
    before: frozenset[str]
    after: frozenset[str]
    displaced: frozenset[str]
    user: bool                # would this step count as using what a hole established?


class Oracle:
    """The planner's rules written out again from their text.

    It reads a Domain's fields as data and calls none of its methods, and it imports nothing
    from the exact planner, so a fault in a helper there cannot hide here. It shares only
    one reading of the rules. `solves` and `minimal_sets` involve no causal rule at all: they are
    plain searches of the real transitions.
    """

    def __init__(self, domain, start, goal_pos, goal_neg=()):
        self.domain = domain
        self.start = frozenset(start)
        self.goal_pos = set(goal_pos)
        self.goal_neg = set(goal_neg)
        self.memo = {}
        self.library = self.reach(())

    def closure(self, literals):
        closed, pending = set(), list(literals)
        while pending:
            literal = pending.pop()
            if literal not in closed:
                closed.add(literal)
                pending.extend(self.domain.implications.get(literal, ()))
        return closed

    def step(self, state, add, delete=()):
        """The next state, and what was displaced on the way."""
        added = self.closure(add)
        after = (set(state) - set(delete)) | added
        for literal in added:
            for group in self.domain.mutex_groups:
                if literal in group:
                    after -= group - {literal}
        changed = set(state) ^ after
        displaced = set()
        for disturbance in self.domain.disturbances:
            if disturbance.trigger in changed:
                displaced |= (set(disturbance.clobbers) & after) - added
        return frozenset(after - displaced), frozenset(displaced)

    def removes(self, action):
        removed = set(action.delete)
        for literal in self.closure(action.add):
            for group in self.domain.mutex_groups:
                if literal in group:
                    removed |= group - {literal}
        return removed

    @staticmethod
    def runs(state, action):
        return set(action.pre) <= state and not (set(action.pre_absent) & state)

    def starts(self, key):
        """The largest sets of robot-relative facts about key's object held in a library state."""
        def first_arg(literal):
            inside = literal[literal.index("(") + 1:literal.rindex(")")] if "(" in literal else ""
            return inside.split(",")[0]

        about = {p for p in self.domain.robot_relative
                 if first_arg(key) and p != key and first_arg(p) == first_arg(key)}
        held = {frozenset(state & about) for state in self.library}
        return [facts for facts in held if not any(facts < other for other in held)]

    def reach(self, literals):
        """Every state reachable with the library and holes for `literals`. No step limit."""
        literals = frozenset(literals)
        if literals not in self.memo:
            starts = {key: self.starts(key) for key in literals}
            seen, pending = {self.start}, [self.start]
            while pending:
                state = pending.pop()
                following = [self.step(state, a.add, a.delete)[0] for a in self.domain.actions
                             if self.runs(state, a)]
                following += [self.step(state, {key})[0] for key in literals
                              if key not in state and any(s <= state for s in starts[key])]
                for reached in following:
                    if reached not in seen:
                        seen.add(reached)
                        pending.append(reached)
            self.memo[literals] = seen
        return self.memo[literals]

    def satisfied(self, state):
        return self.goal_pos <= state and not (self.goal_neg & state)

    def solves(self, literals):
        return any(self.satisfied(state) for state in self.reach(literals))

    def is_minimal(self, literals):
        literals = set(literals)
        return self.solves(literals) and not any(self.solves(literals - {x}) for x in literals)

    def relevant(self):
        wanted, unwanted, relevant = set(self.goal_pos), set(self.goal_neg), []
        grew = True
        while grew:
            grew = False
            for action in self.domain.actions:
                if action not in relevant and (self.closure(action.add) & wanted or self.removes(action) & unwanted):
                    relevant.append(action)
                    wanted |= set(action.pre)
                    unwanted |= set(action.pre_absent)
                    grew = True
        return relevant

    def learnable(self):
        needs = set(self.goal_pos) | {literal for action in self.relevant() for literal in action.pre}
        return needs & set(self.domain.scored)

    def minimal_sets(self):
        """Every subset-minimal set of learnable literals that solves the problem."""
        found = set()
        literals = sorted(self.learnable())
        for size in range(len(literals) + 1):
            for chosen in itertools.combinations(literals, size):
                if self.is_minimal(chosen):
                    found.add(frozenset(chosen))
        return found

    def makers(self, literal):
        return [a for a in self.domain.actions if literal in self.closure(a.add) and literal not in a.pre]

    def depends(self, literal, used_up):
        """What establishing `literal` depends on, directly or not."""
        found, pending = set(), [literal]
        while pending:
            ways, unseen = [], self.makers(pending.pop())
            while unseen:            # what clears the way for a way is a way too
                way = unseen.pop()
                ways.append(way)
                unseen += [b for b in self.domain.actions if b not in ways and b not in unseen
                           and set(way.pre_absent) & self.removes(b)]
            for way in ways:
                for fact in set(way.pre) | (set(way.delete) if used_up else set()):
                    if fact not in found:
                        found.add(fact)
                        pending.append(fact)
        return found

    def strict(self, literal, used_up):
        return {x for x in self.depends(literal, used_up)
                if x != literal and literal not in self.depends(x, used_up)}

    def admissible(self, literals):
        """Exact necessity, and then the three rules of the causal preference."""
        literals = frozenset(literals)
        if not self.is_minimal(literals):
            return False
        learnable = self.learnable()
        last_resort = {g for g in self.goal_pos if self.makers(g)}
        for literal in literals:
            others = literals - {literal}
            needed = self.strict(literal, False)
            above = (needed | self.strict(literal, True)) & learnable
            if literal in last_resort:
                above |= {x for x in learnable - last_resort if literal not in self.depends(x, True)}
            else:
                above -= last_resort
            if self.solves(others | above):
                return False
            if literal in self.goal_pos:
                actual = self.reach(others | (needed & learnable))
                wished = self.reach(others | needed)
                for maker in self.makers(literal):
                    if any(self.runs(s, maker) for s in wished) and not any(self.runs(s, maker) for s in actual):
                        return False
        return True

    def best_plans(self, max_delegated, max_length):
        """Every sequence within the limits that passes the rules: the cheapest cost, and every
        plan at that cost, as (step names, set of delegated literals)."""
        relevant = self.relevant()
        learnable = sorted(self.learnable())
        starts = {key: self.starts(key) for key in learnable}
        verdicts = {}
        best, plans = None, []

        def finish(trace):
            nonlocal best, plans
            delegated = frozenset(step.hole for step in trace if step.hole)
            for i, step in enumerate(trace):
                if step.hole:
                    established = (self.closure({step.hole}) - step.before) & step.after
                    used = False
                    for later in trace[i + 1:]:
                        if established & set(later.pre) and later.user:
                            used = True
                            break
                        established &= later.after
                    if not used and not (established & self.goal_pos):
                        return
            if delegated not in verdicts:
                verdicts[delegated] = self.admissible(delegated)
            if not verdicts[delegated]:
                return
            cost = (len(delegated), sum(len(s.displaced) for s in trace),
                    sum(len((s.after ^ s.before) - s.displaced) for s in trace if s.hole),
                    sum(1 for s in trace if s.hole), len(trace))
            entry = (tuple(s.name for s in trace), delegated)
            if best is None or cost < best:
                best, plans = cost, [entry]
            elif cost == best:
                plans.append(entry)

        def extend(state, trace, delegated):
            if self.satisfied(state):
                finish(trace)
                return
            if len(trace) == max_length:
                return
            for action in self.domain.actions:
                if self.runs(state, action):
                    after, displaced = self.step(state, action.add, action.delete)
                    if after != state:
                        step = Step(action.name, None, action.pre, state, after, displaced, action in relevant)
                        extend(after, [*trace, step], delegated)
            for key in learnable:
                if key not in state and len(delegated | {key}) <= max_delegated:
                    for facts in starts[key]:
                        if facts <= state:
                            after, displaced = self.step(state, {key})
                            if after != state:
                                step = Step(f"HOLE[{key}]", key, facts, state, after, displaced, True)
                                extend(after, [*trace, step], delegated | {key})

        if self.satisfied(self.start):
            return (0, 0, 0, 0, 0), [((), frozenset())]
        extend(self.start, [], frozenset())
        return best, plans


def agrees_with_the_oracle(domain, start, goal_pos, goal_neg, max_delegated, max_length) -> bool:
    """The search returns the oracle's best cost. Everything it reports is a best plan, and every
    best set of holes is reported. What it returns also passes checks that involve no causal rule."""
    oracle = Oracle(domain, start, goal_pos, goal_neg)
    best, plans = oracle.best_plans(max_delegated, max_length)
    result = plan_exact(domain, start, goal_pos, goal_neg, max_delegated=max_delegated, max_length=max_length)
    assert (result.cost if result else None) == best
    if result:
        reported = [tuple(result.steps), *map(tuple, result.alternatives)]
        names = {tuple(step.name for step in steps) for steps in reported}
        assert names <= {entry[0] for entry in plans}
        assert {frozenset(s.hole_for for s in steps if s.hole_for) for steps in reported} == {entry[1] for entry in plans}
        if result.steps:
            assert_valid(domain, start, result, goal_pos, goal_neg)
        assert oracle.solves(result.delegated) and oracle.is_minimal(result.delegated)
        if "max_length" not in result.limits_reached:
            longer = plan_exact(domain, start, goal_pos, goal_neg, max_delegated=max_delegated, max_length=max_length + 4)
            assert longer.cost == result.cost
    else:
        assert result.feasible_with_all_holes == oracle.solves(oracle.learnable())
        if not result.limits_reached:
            assert not plan_exact(domain, start, goal_pos, goal_neg, max_delegated=max_delegated + 2,
                                  max_length=max_length + 4)
    return bool(result and result.hole_indices)


def random_problem(rng):
    """Few literals and many actions, so that effects interfere and paths merge."""
    pool = rng.sample(["a(o)", "b(o)", "c(o)", "d", "e", "f"], rng.randint(4, 6))

    def some(low, high):
        return frozenset(rng.sample(pool, min(rng.randint(low, high), len(pool))))

    actions = []
    for i in range(rng.randint(3, 7)):
        add = some(1, 2)
        actions.append(Action(f"act{i}", pre=some(0, 2), add=add, delete=some(0, 2) - add,
                              pre_absent=some(0, 1) if rng.random() < 0.15 else frozenset()))
    domain = Domain(
        actions=actions,
        scored=some(2, len(pool)),
        robot_relative=frozenset(p for p in pool if "(o)" in p and rng.random() < 0.5),
        mutex_groups=[frozenset(rng.sample(pool, 2))] if rng.random() < 0.4 else [],
        implications={rng.choice(pool): frozenset({rng.choice(pool)})} if rng.random() < 0.3 else {},
        disturbances=([Disturbance(rng.choice(pool), frozenset({rng.choice(pool)}))]
                      if rng.random() < 0.3 else []))
    goal_neg = set(some(0, 1)) if rng.random() < 0.4 else set()
    return domain, some(0, 3), set(some(1, 3)), goal_neg


def layered_problem(rng):
    """Chains of prerequisites and a small goal, with deletes and blockers that cross the chains."""
    count = rng.randint(6, 8)
    literals = [f"l{i}" if rng.random() < 0.5 else f"l{i}(o)" for i in range(count)]
    actions = []
    for i in range(rng.randint(3, 7)):
        target = rng.randint(1, count - 1)
        below = literals[:target]
        pre = frozenset(rng.sample(below, min(rng.randint(0, 2), len(below))))
        add = {literals[target]} | ({rng.choice(literals)} if rng.random() < 0.2 else set())
        delete = frozenset(rng.sample(literals, rng.randint(0, 2))) - add if rng.random() < 0.6 else frozenset()
        absent = frozenset({rng.choice(literals)}) - pre - add if rng.random() < 0.25 else frozenset()
        actions.append(Action(f"act{i}", pre=pre, add=frozenset(add), delete=delete, pre_absent=absent))
    domain = Domain(
        actions=actions,
        scored=frozenset(x for x in literals if rng.random() < 0.7),
        robot_relative=frozenset(x for x in literals if "(o)" in x and rng.random() < 0.3),
        mutex_groups=[frozenset(rng.sample(literals, 2))] if rng.random() < 0.3 else [],
        implications={rng.choice(literals): frozenset({rng.choice(literals)})} if rng.random() < 0.15 else {},
        disturbances=([Disturbance(rng.choice(literals), frozenset({rng.choice(literals)}))]
                      if rng.random() < 0.15 else []))
    goal = set(rng.sample(literals[count // 2:], rng.randint(1, 2)))
    start = frozenset(rng.sample(literals[:3], rng.randint(0, 2)))
    goal_neg = {rng.choice(literals)} - goal - start if rng.random() < 0.15 else set()
    return domain, start, goal, goal_neg


@pytest.mark.parametrize("make_problem, count", [(random_problem, 1500), (layered_problem, 700)],
                         ids=["few literals", "chains"])
def test_the_search_agrees_with_the_oracle_on_random_problems(make_problem, count):
    """(Run once at larger scale, on other seeds: 109,000 problems of up to seven steps, no
    disagreement. There, the repair tests alone never left a solvable problem without a plan.)"""
    rng = random.Random(0)
    needed_a_hole = 0
    for _ in range(count):
        domain, start, goal_pos, goal_neg = make_problem(rng)
        needed_a_hole += agrees_with_the_oracle(domain, start, goal_pos, goal_neg, rng.choice([1, 2]), 5)
    assert needed_a_hole > 200    # the sample really does exercise holes


def test_the_order_of_the_action_list_does_not_change_the_cost():
    rng = random.Random(1)
    for _ in range(300):
        domain, start, goal_pos, goal_neg = random_problem(rng)
        before = plan_exact(domain, start, goal_pos, goal_neg, max_length=5)
        shuffled = rng.sample(domain.actions, len(domain.actions))
        after = plan_exact(dataclasses.replace(domain, actions=shuffled), start, goal_pos, goal_neg, max_length=5)
        assert (after.cost if after else None) == (before.cost if before else None)


@pytest.mark.parametrize("name, domain, start, goal_pos, goal_neg, limits", [
    ("stow", sd.domain(), sd.stowed_state(), *STOW_GOAL, (2, 6)),
    ("drawer", dd.domain(), DRAWER_START, *DRAWER_GOAL, (2, 5)),
    ("drawer with grasp", dd.domain(extra_actions=[dd.scripted_grasp()]), DRAWER_START, *DRAWER_GOAL, (2, 5)),
    ("drawer without pull", drawer_without_pull(), DRAWER_START, *DRAWER_GOAL, (2, 5)),
    ("delete interaction", delete_interaction(), frozenset({"p"}), {"g"}, set(), (2, 5)),
    ("two parts", two_part_assembly(), frozenset(), {"done"}, set(), (2, 6)),
    ("restoration", restoration({"p", "q", "r", "g"}), frozenset({"p"}), {"g"}, set(), (2, 5)),
], ids=lambda v: v if isinstance(v, str) else "")
def test_the_search_agrees_with_the_oracle_on_the_named_problems(name, domain, start, goal_pos, goal_neg, limits):
    agrees_with_the_oracle(domain, start, goal_pos, goal_neg, *limits)
