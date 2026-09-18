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
    assert in_pocket.score < 0.05 and not in_pocket.truth


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
