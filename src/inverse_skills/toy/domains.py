from __future__ import annotations

import numpy as np

from inverse_skills.core import ObjectState, Pose, Region, RobotState, SceneGraph
from inverse_skills.logging import ForwardRollout
from inverse_skills.predicates import GripperOpenPredicate, HoldingPredicate, InRegionPredicate, PredicateRegistry


IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def base_regions() -> dict[str, Region]:
    return {
        "source": Region.from_bounds("source", lower=[-0.15, -0.15, -0.05], upper=[0.15, 0.15, 0.10]),
        "target": Region.from_bounds("target", lower=[0.35, -0.15, -0.05], upper=[0.65, 0.15, 0.10]),
    }


def make_scene(
    timestep: int,
    object_position: list[float],
    *,
    holding: str | None = None,
    gripper_width: float = 0.08,
    skill_name: str | None = None,
) -> SceneGraph:
    cube = ObjectState(
        name="cube",
        semantic_class="box",
        pose=Pose(position=np.asarray(object_position, dtype=np.float32), quat_xyzw=IDENTITY_QUAT),
    )
    robot = RobotState(q=np.zeros(7, dtype=np.float32), gripper_width=gripper_width, holding=holding)
    return SceneGraph(
        timestep=timestep,
        robot=robot,
        objects={"cube": cube},
        regions=base_regions(),
        metadata={} if skill_name is None else {"skill_name": skill_name},
    )


def make_pick_place_rollouts(num_rollouts: int = 3) -> list[ForwardRollout]:
    rollouts: list[ForwardRollout] = []
    for i in range(num_rollouts):
        y = 0.02 * i
        start = make_scene(0, [0.0, y, 0.02], skill_name="pick_place")
        end = make_scene(10, [0.50, y, 0.02], skill_name="pick_place")
        rollouts.append(ForwardRollout(skill_name="pick_place", demo_id=f"pick_{i:03d}", scenes=[start, end]))
    return rollouts


def make_push_rollouts(num_rollouts: int = 3) -> list[ForwardRollout]:
    rollouts: list[ForwardRollout] = []
    for i in range(num_rollouts):
        y = 0.015 * i
        start = make_scene(0, [0.0, y, 0.02], skill_name="push_to_target")
        end = make_scene(10, [0.50, y, 0.02], skill_name="push_to_target")
        rollouts.append(ForwardRollout(skill_name="push_to_target", demo_id=f"push_{i:03d}", scenes=[start, end]))
    return rollouts


def build_predicate_registry() -> PredicateRegistry:
    return PredicateRegistry([
        InRegionPredicate("cube", "source"),
        InRegionPredicate("cube", "target"),
        GripperOpenPredicate(min_width=0.04),
        HoldingPredicate("cube"),
    ])
