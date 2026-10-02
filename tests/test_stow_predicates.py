import numpy as np
import pytest

from inverse_skills.core import ObjectState, Pose, Region, RobotState, SceneGraph
from inverse_skills.envs import stow_cube as geo
from inverse_skills.envs import stow_predicates as sp

_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def scene_with_cube_a_at(x: float) -> SceneGraph:
    return SceneGraph(
        timestep=0,
        robot=RobotState(q=np.zeros(7, dtype=np.float32), gripper_width=0.08),
        objects={"cube_a": ObjectState(name="cube_a", semantic_class="cube",
                                       pose=Pose(position=[x, 0.0, geo.CUBE_HALF],
                                                 quat_xyzw=_QUAT))},
        regions={"pocket": Region.from_bounds("pocket", [0.11, -0.02, -0.03], [0.15, 0.02, 0.07])},
    )


def test_clear_of_walls_scores_half_at_the_threshold():
    # Positions are stored as float32, so the margin lands within rounding of 0.
    predicate = sp.ClearOfWallsPredicate("cube_a")
    result = predicate.evaluate(scene_with_cube_a_at(geo.X_CLEAR))
    assert result.margin == pytest.approx(0.0, abs=1e-6)
    assert result.score == pytest.approx(0.5, abs=1e-4)


def test_clear_of_walls_separates_source_from_pocket():
    predicate = sp.ClearOfWallsPredicate("cube_a")
    at_source = predicate.evaluate(scene_with_cube_a_at(0.0))
    in_pocket = predicate.evaluate(scene_with_cube_a_at(geo.POCKET_A_XY[0]))
    assert at_source.score > 0.95 and at_source.truth
    assert in_pocket.score < 0.2 and not in_pocket.truth


def test_threshold_sits_inside_the_measured_grasp_limit():
    """X_CLEAR is the measured limit (pick lifts A to 110 mm, never at 115 mm),
    so the framework's 0.8 threshold must be conservative with respect to it."""
    predicate = sp.ClearOfWallsPredicate("cube_a")
    assert predicate.evaluate(scene_with_cube_a_at(0.110)).score < 0.8   # limit, not yet trusted
    assert predicate.evaluate(scene_with_cube_a_at(0.115)).truth is False  # measured failure
    established = predicate.evaluate(scene_with_cube_a_at(0.096))
    assert established.score >= 0.8 and established.truth


def test_tcp_near_holds_in_the_state_approach_produces():
    """Seated on the cube, the TCP is 30.6-32.0 mm from its centre (measured)."""
    scene = scene_with_cube_a_at(0.0)
    near = sp.registry().get("tcp_near(cube_a)")
    for distance in (0.030, 0.032):
        scene.robot.ee_pose = Pose(position=[0.0, 0.0, geo.CUBE_HALF + distance], quat_xyzw=_QUAT)
        assert near.evaluate(scene).score >= 0.8
    scene.robot.ee_pose = Pose(position=[0.0, 0.0, 0.18], quat_xyzw=_QUAT)   # home, far away
    assert near.evaluate(scene).score < 0.05


def test_gripper_open_separates_open_from_holding_a_cube():
    """Fully open is 80 mm; holding a 4 cm cube leaves 36-37 mm (measured)."""
    scene = scene_with_cube_a_at(0.0)
    open_pred = sp.registry().get("gripper_open()")
    scene.robot.gripper_width = 0.080
    assert open_pred.evaluate(scene).score >= 0.95
    scene.robot.gripper_width = 0.037
    assert open_pred.evaluate(scene).score <= 0.05


def test_robot_relative_predicates_are_typed_by_the_vocabulary():
    assert sp.registry().robot_relative_keys() == ["tcp_near(cube_a)", "tcp_near(cube_b)"]


def test_registry_keys():
    assert sp.registry().keys() == [
        "clear_of_walls(cube_a)",
        "gripper_open()",
        "in_region(cube_a,pocket)",
        "in_region(cube_a,src_a)",
        "in_region(cube_b,mouth)",
        "in_region(cube_b,src_b)",
        "tcp_near(cube_a)",
        "tcp_near(cube_b)",
    ]
