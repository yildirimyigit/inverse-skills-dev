import ast
import inspect

from inverse_skills.envs import stow_domain as sd
from inverse_skills.operators import hole_planner
from inverse_skills.operators.hole_planner import (
    Action,
    Disturbance,
    Domain,
    NoPlan,
    capability_gaps,
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
    blocked = stow_plan(max_delegated=0)
    assert not blocked and isinstance(blocked, NoPlan)
    assert blocked.capability_gaps == ("clear_of_walls(cube_a)", "holding(cube_a)")
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
    """Offered every unreachable open condition, the planner could delegate the
    whole relocation: one predicate too, but a bigger one. The gap rule does not
    even offer it (see the next test)."""
    result = stow_plan(unrestricted_holes=True)
    assert "in_region(cube_a,src_a)" in result.candidates
    assert result.hole_literals == 2            # frees A, and so empties the pocket slot
    assert "HOLE[in_region(cube_a,src_a)]" not in result.actions


def test_holes_are_offered_only_at_synthesizable_capability_gaps():
    """clear_of_walls(A) and holding(A) can each be established only through the
    other: an irreducible cycle, so both are capability gaps. Only
    clear_of_walls(A) has the soft score a learning objective needs, so only it
    is synthesizable. in_region(A,src_a) is unreachable too, but place(A,src_a)
    establishes it once A is held, so it is no gap at all."""
    gaps = capability_gaps(sd.domain(), sd.stowed_state(), STOW_GOAL_POS, STOW_GOAL_NEG)
    assert gaps == {"clear_of_walls(cube_a)", "holding(cube_a)"}
    result = stow_plan()
    assert result.capability_gaps == ("clear_of_walls(cube_a)", "holding(cube_a)")
    assert result.synthesizable_gaps == ("clear_of_walls(cube_a)",)
    assert result.unsynthesizable_gaps == ("holding(cube_a)",)
    assert result.candidates == ("clear_of_walls(cube_a)",)
    assert stow_plan(unrestricted_holes=True).candidates == ("clear_of_walls(cube_a)",
                                                             "in_region(cube_a,src_a)")


def test_unsynthesizable_gaps_give_no_plan_and_are_reported():
    """Take clear_of_walls(A)'s soft score away: the gaps are unchanged, but none
    can become a hole. The planner says so rather than delegate another literal;
    only the unrestricted ablation still finds something to delegate."""
    domain = sd.domain()
    domain.scored = domain.scored - {"clear_of_walls(cube_a)"}
    result = stow_plan(domain)
    assert not result
    assert result.capability_gaps == ("clear_of_walls(cube_a)", "holding(cube_a)")
    assert result.synthesizable_gaps == ()
    assert result.unsynthesizable_gaps == ("clear_of_walls(cube_a)", "holding(cube_a)")
    assert stow_plan(domain, unrestricted_holes=True).delegated == ("in_region(cube_a,src_a)",)


OPEN, CLOSED = "drawer_open(drawer)", "drawer_closed(drawer)"
STANDOFF, GRASPED = "at_handle_standoff(drawer)", "handle_grasped(drawer)"


def drawer_domain(pull_open: bool = True) -> Domain:
    """A drawer the robot can push closed, approach and, perhaps, pull open, with
    both exclusions a faithful model declares: open or closed; at the standoff or
    on the handle."""
    actions = [
        Action("push_close(drawer)", pre=frozenset({OPEN}), add=frozenset({CLOSED}),
               delete=frozenset({OPEN})),
        Action("approach_handle(drawer)", pre_absent=frozenset({GRASPED}),
               add=frozenset({STANDOFF})),
    ]
    if pull_open:
        actions.append(Action("pull_open(drawer)", pre=frozenset({GRASPED}),
                              add=frozenset({OPEN}), delete=frozenset({CLOSED, GRASPED})))
    return Domain(actions=actions, scored=frozenset({OPEN, CLOSED, STANDOFF, GRASPED}),
                  robot_relative=frozenset({STANDOFF, GRASPED}),
                  mutex_groups=[frozenset({OPEN, CLOSED}), frozenset({STANDOFF, GRASPED})])


def drawer_plan(domain: Domain, **kwargs):
    return plan(domain, frozenset({CLOSED}), {OPEN}, {CLOSED}, **kwargs)


def test_drawer_gap_is_the_grasp_when_the_library_can_pull():
    """pull_open establishes drawer_open once the handle is grasped, and nothing
    establishes the grasp: the gap is the grasp, not the opening."""
    domain = drawer_domain()
    assert capability_gaps(domain, frozenset({CLOSED}), {OPEN}, {CLOSED}) == {GRASPED}
    result = drawer_plan(domain)
    assert result.actions == ("approach_handle(drawer)", f"HOLE[{GRASPED}]", "pull_open(drawer)")
    assert result.steps[1].pre == {STANDOFF}
    # Offered every unreachable open condition, the planner would learn the whole
    # pull: both holes change two literals, and the shorter plan wins.
    assert drawer_plan(domain, unrestricted_holes=True).actions == ("approach_handle(drawer)",
                                                           f"HOLE[{OPEN}]")


def test_drawer_gap_is_the_opening_when_nothing_can_pull():
    """The same drawer without pull_open: nothing establishes drawer_open, so the
    opening itself is the gap. The rule follows the library, not a preference."""
    domain = drawer_domain(pull_open=False)
    assert capability_gaps(domain, frozenset({CLOSED}), {OPEN}, {CLOSED}) == {OPEN}
    assert drawer_plan(domain).actions == ("approach_handle(drawer)", f"HOLE[{OPEN}]")


def test_conjunctive_prerequisites_are_the_gaps_not_their_product():
    """`make` establishes p from q and r together. Neither q nor r alone makes p
    reachable, yet p is no gap: the library has the action, it lacks q and r."""
    domain = Domain(actions=[Action("make", pre=frozenset({"q", "r"}), add=frozenset({"p"}))],
                    scored=frozenset({"p", "q", "r"}))
    assert capability_gaps(domain, frozenset(), {"p"}) == {"q", "r"}
    result = plan(domain, frozenset(), {"p"})
    assert result.delegated == ("q", "r") and result.actions[-1] == "make"
    assert plan(domain, frozenset(), {"p"}, unrestricted_holes=True).actions == ("HOLE[p]",)


def test_circular_support_is_not_a_resolver():
    """p is established only from q, and q only from p. An action that needs p
    back does not resolve p: both are gaps, and the cheaper hole is chosen."""
    domain = Domain(actions=[Action("p_from_q", pre=frozenset({"q"}), add=frozenset({"p"})),
                             Action("q_from_p", pre=frozenset({"p"}), add=frozenset({"q"}))],
                    scored=frozenset({"p", "q"}))
    assert capability_gaps(domain, frozenset(), {"p"}) == {"p", "q"}
    assert plan(domain, frozenset(), {"p"}).actions == ("HOLE[p]",)


def test_a_cycle_with_a_way_in_is_resolved_through_it():
    """Give q a second establisher that needs only r, which nothing establishes:
    the cycle is broken from outside it, so r is the only gap."""
    domain = Domain(actions=[Action("p_from_q", pre=frozenset({"q"}), add=frozenset({"p"})),
                             Action("q_from_p", pre=frozenset({"p"}), add=frozenset({"q"})),
                             Action("q_from_r", pre=frozenset({"r"}), add=frozenset({"q"}))],
                    scored=frozenset({"p", "q", "r"}))
    assert capability_gaps(domain, frozenset(), {"p"}) == {"r"}
    assert plan(domain, frozenset(), {"p"}).actions == ("HOLE[r]", "q_from_r", "p_from_q")


def test_a_negated_goal_needs_a_remover_and_its_prerequisites():
    domain = Domain(actions=[Action("clear_x", pre=frozenset({"k"}), delete=frozenset({"x"}))],
                    scored=frozenset({"k", "x"}))
    assert capability_gaps(domain, frozenset({"x"}), set(), {"x"}) == {"k"}
    assert plan(domain, frozenset({"x"}), set(), {"x"}).actions == ("HOLE[k]", "clear_x")


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


def test_a_gap_the_relaxation_hides_gives_no_plan():
    """`use` needs p and q together, but making q consumes p. Ignoring deletes,
    g looks reachable, so the gap analysis finds no gap, and the search, which
    sees the delete, finds no plan. The planner reports that rather than guess a
    hole: a known limit of the delete relaxation, left alone on purpose. The
    unrestricted ablation still falls back to every open condition."""
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
    assert not result and result.capability_gaps == ()
    assert plan(domain, frozenset({"p"}), {"g"}, unrestricted_holes=True).delegated


def test_a_shorter_way_to_an_explored_state_is_still_explored():
    """Two ways reach the state {middle}: eight cheap steps, or one shortcut that
    displaces an object. The cheap way gets there first. From there the goal is
    three more steps, and 8 + 3 is over the limit of 10, so only the shortcut
    (1 + 3 = 4 steps) can finish. It must not be skipped as a duplicate."""
    # The long way also clears object_in_place, by a normal effect, so both ways
    # reach exactly the same state.
    long_way = [Action("long1", pre=frozenset({"start"}), add=frozenset({"step1"}),
                       delete=frozenset({"start", "object_in_place"}))]
    for i in range(2, 8):
        long_way.append(Action(f"long{i}", pre=frozenset({f"step{i - 1}"}),
                               add=frozenset({f"step{i}"}), delete=frozenset({f"step{i - 1}"})))
    long_way.append(Action("long8", pre=frozenset({"step7"}), add=frozenset({"middle"}),
                           delete=frozenset({"step7"})))
    shortcut = Action("shortcut", pre=frozenset({"start"}), add=frozenset({"middle"}),
                      delete=frozenset({"start"}))
    finish = [
        Action("finish1", pre=frozenset({"middle"}), add=frozenset({"end1"}), delete=frozenset({"middle"})),
        Action("finish2", pre=frozenset({"end1"}), add=frozenset({"end2"}), delete=frozenset({"end1"})),
        Action("finish3", pre=frozenset({"end2"}), add=frozenset({"goal"}), delete=frozenset({"end2"})),
    ]
    domain = Domain(actions=[shortcut] + long_way + finish,
                    disturbances=[Disturbance(trigger="middle", clobbers=frozenset({"object_in_place"}))])
    start = frozenset({"start", "object_in_place"})

    result = plan(domain, start, {"goal"})
    assert result.actions == ("shortcut", "finish1", "finish2", "finish3")
    assert result.cost == (0, 1, 0, 0, 4)

    # With room for eleven steps the long way fits, and it wins: it displaces nothing.
    roomy = plan(domain, start, {"goal"}, max_length=11)
    assert roomy.cost == (0, 0, 0, 0, 11)


def test_paths_that_used_different_holes_are_kept_apart():
    """Two half-plans reach the same state {m}: one after a hole for q, the other
    after a hole for p. The goal needs p again. After the hole for p it is already
    counted, so the plan keeps one delegated fact; after the hole for q it would
    need a second. The search must keep the two apart, or it returns the worse plan."""
    domain = Domain(
        actions=[
            Action("use_p", pre=frozenset({"p"}), add=frozenset({"m"}), delete=frozenset({"p", "x"})),
            Action("use_q", pre=frozenset({"q"}), add=frozenset({"m"}), delete=frozenset({"q"})),
            Action("finish", pre=frozenset({"m", "p"}), add=frozenset({"g"})),
        ],
        scored=frozenset({"p", "q"}),
        # p brings x along, so a hole for p changes two facts and the q path looks
        # cheaper at first: it reaches {m} first.
        implications={"p": frozenset({"x"})},
    )
    result = plan(domain, frozenset(), {"g"})
    assert result.actions == ("HOLE[p]", "use_p", "HOLE[p]", "finish")
    assert result.cost == (1, 0, 4, 2, 4)


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
