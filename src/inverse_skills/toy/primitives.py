from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inverse_skills.core import SceneGraph


@dataclass(frozen=True)
class PrimitiveAction:
    name: str

    def __str__(self) -> str:
        return self.name


def _move_tcp(scene: SceneGraph, position) -> None:
    """Put the end-effector at the point it just acted on.

    Every primitive here ends with the gripper at the object it picked, placed
    or pushed. Without modelling that, the TCP never moves during search, so
    TCP-relative predicates (tcp_near, near) are scored against a stale
    end-effector pose and can look permanently unsatisfiable — the planner then
    reports them as residual when the real executor would satisfy them.
    """
    if scene.robot.ee_pose is not None:
        scene.robot.ee_pose.position = np.asarray(position, dtype=np.float32).copy()


class PrimitiveLibrary:
    def __init__(self, object_name: str = "cube", source_name: str = "source", target_name: str = "target"):
        self.object_name = object_name
        self.source_name = source_name
        self.target_name = target_name

    def available_actions(self) -> list[PrimitiveAction]:
        return [
            PrimitiveAction(f"pick({self.object_name})"),
            PrimitiveAction(f"place({self.source_name})"),
            PrimitiveAction(f"place({self.target_name})"),
            PrimitiveAction(f"push({self.target_name})"),
            PrimitiveAction("noop"),
        ]

    def apply(self, scene: SceneGraph, action: PrimitiveAction) -> SceneGraph:
        next_scene = scene.copy(timestep=scene.timestep + 1)
        obj = next_scene.objects[self.object_name]
        source = next_scene.regions[self.source_name]
        target = next_scene.regions[self.target_name]

        if action.name == f"pick({self.object_name})":
            if next_scene.robot.holding is None:
                next_scene.robot.holding = self.object_name
                next_scene.robot.gripper_width = 0.0
                _move_tcp(next_scene, obj.pose.position)
        elif action.name == f"place({self.source_name})":
            if next_scene.robot.holding == self.object_name:
                obj.pose.position = source.center.copy()
                next_scene.robot.holding = None
                next_scene.robot.gripper_width = 0.08
                _move_tcp(next_scene, obj.pose.position)
        elif action.name == f"place({self.target_name})":
            if next_scene.robot.holding == self.object_name:
                obj.pose.position = target.center.copy()
                next_scene.robot.holding = None
                next_scene.robot.gripper_width = 0.08
                _move_tcp(next_scene, obj.pose.position)
        elif action.name == f"push({self.target_name})":
            if next_scene.robot.holding is None:
                obj.pose.position = target.center.copy()
                next_scene.robot.gripper_width = 0.08
                _move_tcp(next_scene, obj.pose.position)
        elif action.name == "noop":
            pass
        else:
            raise ValueError(f"Unknown primitive action: {action.name}")
        return next_scene
