from __future__ import annotations

import numpy as np

from inverse_skills.core import ObjectState, Pose, Region, RobotState, SceneGraph
from inverse_skills.logging import ForwardRollout
from inverse_skills.predicates import GripperOpenPredicate, HoldingPredicate, InRegionPredicate, PredicateRegistry


IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def named_regions(source_name: str = "source", target_name: str = "target") -> dict[str, Region]:
    return {
        source_name: Region.from_bounds(source_name, lower=[-0.15, -0.15, -0.05], upper=[0.15, 0.15, 0.10]),
        target_name: Region.from_bounds(target_name, lower=[0.35, -0.15, -0.05], upper=[0.65, 0.15, 0.10]),
    }


def base_regions() -> dict[str, Region]:
    return named_regions("source", "target")


def make_scene_named(
    timestep: int,
    object_name: str,
    object_position: list[float],
    *,
    source_name: str = "source",
    target_name: str = "target",
    holding: str | None = None,
    gripper_width: float = 0.08,
    skill_name: str | None = None,
) -> SceneGraph:
    obj = ObjectState(
        name=object_name,
        semantic_class="box",
        pose=Pose(position=np.asarray(object_position, dtype=np.float32), quat_xyzw=IDENTITY_QUAT),
    )
    robot = RobotState(q=np.zeros(7, dtype=np.float32), gripper_width=gripper_width, holding=holding)
    return SceneGraph(
        timestep=timestep,
        robot=robot,
        objects={object_name: obj},
        regions=named_regions(source_name, target_name),
        metadata={} if skill_name is None else {"skill_name": skill_name},
    )


def make_scene(
    timestep: int,
    object_position: list[float],
    *,
    holding: str | None = None,
    gripper_width: float = 0.08,
    skill_name: str | None = None,
) -> SceneGraph:
    return make_scene_named(
        timestep=timestep,
        object_name="cube",
        object_position=object_position,
        source_name="source",
        target_name="target",
        holding=holding,
        gripper_width=gripper_width,
        skill_name=skill_name,
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


def build_predicate_registry(
    object_name: str = "cube",
    source_name: str = "source",
    target_name: str = "target",
) -> PredicateRegistry:
    return PredicateRegistry([
        InRegionPredicate(object_name, source_name),
        InRegionPredicate(object_name, target_name),
        GripperOpenPredicate(min_width=0.04),
        HoldingPredicate(object_name),
    ])


def make_scene_named_with_distractor(
    timestep: int,
    object_name: str,
    object_position: list[float],
    *,
    distractor_name: str,
    distractor_position: list[float],
    source_name: str = "source",
    target_name: str = "target",
    holding: str | None = None,
    gripper_width: float = 0.08,
    skill_name: str | None = None,
) -> SceneGraph:
    obj = ObjectState(
        name=object_name,
        semantic_class="box",
        pose=Pose(position=np.asarray(object_position, dtype=np.float32), quat_xyzw=IDENTITY_QUAT),
    )
    distractor = ObjectState(
        name=distractor_name,
        semantic_class="box",
        pose=Pose(position=np.asarray(distractor_position, dtype=np.float32), quat_xyzw=IDENTITY_QUAT),
    )
    robot = RobotState(q=np.zeros(7, dtype=np.float32), gripper_width=gripper_width, holding=holding)
    return SceneGraph(
        timestep=timestep,
        robot=robot,
        objects={
            object_name: obj,
            distractor_name: distractor,
        },
        regions=named_regions(source_name, target_name),
        metadata={} if skill_name is None else {"skill_name": skill_name},
    )


def build_predicate_registry_grasp_hold(
    object_name: str = "cube",
    source_name: str = "source",
) -> PredicateRegistry:
    return PredicateRegistry([
        InRegionPredicate(object_name, source_name),
        GripperOpenPredicate(min_width=0.04),
        HoldingPredicate(object_name),
    ])


def build_predicate_registry_with_distractor(
    object_name: str,
    source_name: str,
    target_name: str,
    distractor_name: str,
) -> PredicateRegistry:
    return PredicateRegistry([
        InRegionPredicate(object_name, source_name),
        InRegionPredicate(object_name, target_name),
        InRegionPredicate(distractor_name, source_name),
        InRegionPredicate(distractor_name, target_name),
        GripperOpenPredicate(min_width=0.04),
        HoldingPredicate(object_name),
        HoldingPredicate(distractor_name),
    ])
