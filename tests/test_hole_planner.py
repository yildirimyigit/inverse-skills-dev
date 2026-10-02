import ast
import inspect

from inverse_skills.envs import stow_domain as sd
from inverse_skills.operators import hole_planner
from inverse_skills.operators.hole_planner import (
    Action,
    Domain,
    derive_fences,
    hole_spec,
    learned_action,
    plan,
    relaxed_reachable,
    symbolic_state,
)
from inverse_skills.operators.schema import LearnedOperator, PredicateTerm

STOW_GOAL_POS = {"clear_of_walls(cube_a)", "gripper_open()",
                 "in_region(cube_a,src_a)", "in_region(cube_b,src_b)"}
STOW_GOAL_NEG = {"in_region(cube_a,pocket)", "in_region(cube_b,mouth)"}
B_FIRST = ("pick(cube_b)", "place(cube_b,src_b)", "approach(cube_a)",
           "HOLE[clear_of_walls(cube_a)]", "pick(cube_a)", "place(cube_a,src_a)")

DRAG = Action(
    name="drag(cube_a)",
    pre=frozenset({"tcp_near(cube_a)"}),
    add=frozenset({"clear_of_walls(cube_a)"}),
    delete=frozenset({"gripper_open()"}),
)


def stow_plan(domain=None, **kwargs):
    return plan(domain or sd.domain(), sd.stowed_state(), STOW_GOAL_POS, STOW_GOAL_NEG, **kwargs)


def test_stow_needs_one_hole_placed_mid_plan():
    result = stow_plan()
    assert result.actions == B_FIRST
    assert result.delegated == ("clear_of_walls(cube_a)",)
    assert result.hole_indices == (3,)          # neither first nor last
    assert result.alternatives == ()            # the model determines the ordering


def test_library_alone_cannot_free_the_stowed_cube():
    assert stow_plan(max_delegated=0) is None
    reach = relaxed_reachable(sd.domain(), sd.stowed_state())
    assert "clear_of_walls(cube_a)" not in reach


def test_extracting_a_through_an_occupied_mouth_displaces_b():
    domain = sd.domain()
    hole = domain.hole("clear_of_walls(cube_a)", {"tcp_near(cube_a)"})
    state = frozenset({"in_region(cube_a,pocket)", "in_region(cube_b,mouth)", "tcp_near(cube_a)"})
    nxt, collateral = domain.transition(state, hole)
    assert collateral == {"in_region(cube_b,mouth)"}
    assert "in_region(cube_b,mouth)" not in nxt
    cleared = frozenset(state - {"in_region(cube_b,mouth)"} | {"in_region(cube_b,src_b)"})
    assert domain.transition(cleared, hole)[1] == frozenset()


def test_without_the_interference_model_the_ordering_is_undetermined():
    """Hole-first costs the same unless the model knows the drag sweeps the mouth."""
    blind = stow_plan(sd.domain(interference=False))
    orderings = {blind.actions} | set(blind.alternative_names)
    hole_first = ("approach(cube_a)", "HOLE[clear_of_walls(cube_a)]", "pick(cube_a)",
                  "place(cube_a,src_a)", "pick(cube_b)", "place(cube_b,src_b)")
    assert {B_FIRST, hole_first} <= orderings


def test_a_complete_library_is_ordered_by_collateral_not_by_holes():
    """With no hole in the plan, only the interference model keeps B first."""
    with_model = stow_plan(sd.domain(extra_actions=[DRAG]), max_delegated=0)
    assert with_model.delegated == ()
    assert with_model.actions.index("place(cube_b,src_b)") < with_model.actions.index("drag(cube_a)")
    assert with_model.alternatives == ()
    blind = stow_plan(sd.domain(extra_actions=[DRAG], interference=False), max_delegated=0)
    assert blind.alternatives != ()


def test_a_hole_that_swallows_the_task_is_rejected_for_changing_more():
    """Delegating the whole relocation is one predicate too, but a bigger one."""
    result = stow_plan()
    assert result.hole_literals == 2            # frees A, and so empties the pocket slot
    assert "HOLE[in_region(cube_a,src_a)]" not in result.actions


def test_hole_candidates_are_the_open_conditions_the_library_cannot_reach():
    result = stow_plan()
    assert result.candidates == ("clear_of_walls(cube_a)", "in_region(cube_a,src_a)")
    # holding(cube_a) is unreachable too, but has no soft score to learn against
    assert all(not c.startswith("holding") for c in result.candidates)


def test_hole_start_comes_from_robot_relative_predicates():
    result = stow_plan()
    assert result.steps[3].pre == {"tcp_near(cube_a)"}
    untyped = sd.domain()
    untyped.robot_relative = frozenset()
    bare = stow_plan(untyped)
    assert "approach(cube_a)" not in bare.actions    # nothing asks for it any more


def test_fences_are_the_causal_links_spanning_the_hole():
    domain = sd.domain()
    result = stow_plan(domain)
    fences, negative = derive_fences(domain, sd.stowed_state(), result.steps, 3,
                                     STOW_GOAL_POS, STOW_GOAL_NEG)
    assert fences == ["in_region(cube_b,src_b)"]
    assert negative == ["in_region(cube_b,mouth)"]


def test_hole_spec_is_the_learning_problem():
    domain = sd.domain()
    result = stow_plan(domain)
    spec = hole_spec(domain, sd.stowed_state(), result, STOW_GOAL_POS, STOW_GOAL_NEG)
    assert spec["achieve"] == "clear_of_walls(cube_a)"
    assert spec["object"] == "cube_a"
    assert spec["start"] == ["tcp_near(cube_a)"]
    assert spec["preserve"] == ["in_region(cube_b,src_b)"]
    assert [Action.from_dict(a).name for a in spec["prefix"]] == list(B_FIRST[:3])


def test_learned_operator_enters_the_library_with_its_measured_precondition():
    extracted = LearnedOperator(
        skill_name="learned",
        preconditions=[PredicateTerm("tcp_near(cube_a)"), PredicateTerm("in_region(cube_a,pocket)"),
                       PredicateTerm("in_region(cube_b,src_b)")],
        add_effects=[PredicateTerm("clear_of_walls(cube_a)"), PredicateTerm("in_region(cube_a,mouth)")],
        delete_effects=[PredicateTerm("in_region(cube_a,pocket)")],
    )
    action = learned_action("clear_of_walls(cube_a)", extracted)
    assert action.name == "LEARNED[clear_of_walls(cube_a)]"
    result = stow_plan(sd.domain(extra_actions=[action]), max_delegated=0)
    assert result.actions == tuple(a.replace("HOLE", "LEARNED") for a in B_FIRST)


def test_pushcube_coarse_place_yields_a_trailing_hole():
    """The paper's partition, derived at plan time from an honest place()."""
    domain = Domain(
        actions=[
            Action(name="pick(cube)", pre_absent=frozenset({"holding(cube)"}),
                   add=frozenset({"holding(cube)", "tcp_near(cube)"}),
                   delete=frozenset({"gripper_open()", "in_region(cube,goal)",
                                     "in_region(cube,src)"})),
            # Honest: the scripted place lands inside the coarse slot, not on the pose.
            Action(name="place(cube,src)", pre=frozenset({"holding(cube)"}),
                   add=frozenset({"in_region(cube,src)", "gripper_open()"}),
                   delete=frozenset({"holding(cube)"})),
        ],
        scored=frozenset({"at_pose(cube,src)", "in_region(cube,src)", "in_region(cube,goal)",
                          "gripper_open()", "tcp_near(cube)"}),
        robot_relative=frozenset({"tcp_near(cube)"}),
        mutex_groups=[frozenset({"in_region(cube,src)", "in_region(cube,goal)"})],
    )
    result = plan(domain, frozenset({"in_region(cube,goal)", "gripper_open()"}),
                  {"at_pose(cube,src)", "in_region(cube,src)", "gripper_open()",
                   "tcp_near(cube)"},
                  {"in_region(cube,goal)"})
    assert result.actions == ("pick(cube)", "place(cube,src)", "HOLE[at_pose(cube,src)]")
    assert result.delegated == ("at_pose(cube,src)",)


def test_relaxation_that_hides_a_gap_falls_back_to_every_open_condition():
    """`use` needs p and q together, but making q consumes p: the relaxation
    says g is reachable, the real problem has no hole-free plan."""
    domain = Domain(
        actions=[
            Action(name="make_q", pre=frozenset({"p"}), add=frozenset({"q"}),
                   delete=frozenset({"p"})),
            Action(name="use", pre=frozenset({"p", "q"}), add=frozenset({"g"})),
        ],
        scored=frozenset({"p", "q", "g"}),
    )
    assert "g" in relaxed_reachable(domain, frozenset({"p"}))
    result = plan(domain, frozenset({"p"}), {"g"})
    assert result is not None and result.delegated


def test_learned_steps_name_the_predicate_they_owe():
    assert hole_planner.delegated_predicate("HOLE[clear_of_walls(cube_a)]") == "clear_of_walls(cube_a)"
    assert hole_planner.delegated_predicate("LEARNED[p(x,y)]") == "p(x,y)"
    assert hole_planner.delegated_predicate("pick(cube_a)") is None
    assert hole_planner.predicate_args("in_region(cube_a,pocket)") == ("cube_a", "pocket")
    assert hole_planner.predicate_args("gripper_open()") == ()


def test_action_round_trips_through_a_dict():
    action = Action("x(a)", pre=frozenset({"p(a)"}), pre_absent=frozenset({"q(a)"}),
                    add=frozenset({"r(a)"}), delete=frozenset({"p(a)"}), hole_for="r(a)")
    assert Action.from_dict(action.to_dict()) == action


def test_symbolic_state_thresholds_scores():
    assert symbolic_state({"a": 0.81, "b": 0.79, "c": 1.0}) == frozenset({"a", "c"})


def test_the_planner_code_names_no_domain_vocabulary():
    """Everything domain-specific must reach the planner as data. Docstrings may
    use examples; code and string literals may not."""
    tree = ast.parse(inspect.getsource(hole_planner))
    docstrings = {id(node.body[0].value) for node in ast.walk(tree)
                  if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef))
                  and node.body and isinstance(node.body[0], ast.Expr)}
    words = {"tcp_near", "in_region", "clear_of_walls", "holding", "gripper", "cube",
             "pocket", "mouth", "drawer", "handle"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            found |= {w for w in words if w in node.value}
        if isinstance(node, ast.Name):
            found |= {w for w in words if w in node.id}
    assert not found
