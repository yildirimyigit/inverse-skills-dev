from inverse_skills.envs import stow_domain as sd
from inverse_skills.operators.hole_planner import Action, Domain, plan, symbolic_state

STOW_GOAL_POS = {"clear_of_walls(cube_a)", "gripper_open()",
                 "in_region(cube_a,src_a)", "in_region(cube_b,src_b)"}
STOW_GOAL_NEG = {"in_region(cube_a,pocket)", "in_region(cube_b,mouth)"}

DRAG = Action(
    name="drag(cube_a)",
    pre=frozenset({"tcp_near(cube_a)"}),
    add=frozenset({"clear_of_walls(cube_a)"}),
    delete=frozenset({"gripper_open()"}),
)


def stow_plan(**kwargs):
    return plan(sd.domain(kwargs.pop("extra_actions", None)), sd.stowed_state(),
                STOW_GOAL_POS, STOW_GOAL_NEG, **kwargs)


def test_stow_needs_one_hole_placed_mid_plan():
    result = stow_plan()
    assert result is not None
    assert result.actions == (
        "pick(cube_b)", "place(cube_b,src_b)", "approach(cube_a)",
        "HOLE[clear_of_walls(cube_a)]", "pick(cube_a)", "place(cube_a,src_a)",
    )
    assert result.delegated == ("clear_of_walls(cube_a)",)
    assert result.hole_indices == (3,)          # neither first nor last


def test_library_alone_cannot_free_the_stowed_cube():
    assert stow_plan(max_delegated=0) is None


def test_a_drag_primitive_removes_the_need_for_a_hole():
    result = stow_plan(max_delegated=0, extra_actions=[DRAG])
    assert result is not None
    assert result.delegated == ()
    assert "drag(cube_a)" in result.actions


def test_without_the_tie_break_the_hole_moves_to_the_front():
    """The hole-first plan costs the same; only the tie-break rejects it."""
    late = stow_plan().actions
    early = stow_plan(latest_holes=False).actions
    assert late != early
    assert len(late) == len(early)
    assert early.index("HOLE[clear_of_walls(cube_a)]") < late.index("HOLE[clear_of_walls(cube_a)]")


def test_a_hole_that_swallows_the_task_is_rejected_for_changing_more():
    """Delegating the whole relocation is one predicate too, but a bigger one."""
    result = stow_plan()
    assert result.hole_literals == 2            # frees A, and so empties the pocket slot
    assert "HOLE[in_region(cube_a,src_a)]" not in result.actions


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
        hole_candidates=["at_pose(cube,src)"],
        mutex_groups=[frozenset({"in_region(cube,src)", "in_region(cube,goal)"})],
    )
    result = plan(domain, frozenset({"in_region(cube,goal)", "gripper_open()"}),
                  {"at_pose(cube,src)", "in_region(cube,src)", "gripper_open()",
                   "tcp_near(cube)"},
                  {"in_region(cube,goal)"})
    assert result.actions == ("pick(cube)", "place(cube,src)", "HOLE[at_pose(cube,src)]")
    assert result.delegated == ("at_pose(cube,src)",)


def test_symbolic_state_thresholds_scores():
    assert symbolic_state({"a": 0.81, "b": 0.79, "c": 1.0}) == frozenset({"a", "c"})
