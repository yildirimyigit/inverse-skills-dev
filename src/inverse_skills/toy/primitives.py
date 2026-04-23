from __future__ import annotations

from dataclasses import dataclass

from inverse_skills.core import SceneGraph


@dataclass(frozen=True)
class PrimitiveAction:
    name: str

    def __str__(self) -> str:
        return self.name


class PrimitiveLibrary:
    def available_actions(self) -> list[PrimitiveAction]:
        return [
            PrimitiveAction("pick(cube)"),
            PrimitiveAction("place(source)"),
            PrimitiveAction("place(target)"),
            PrimitiveAction("noop"),
        ]

    def apply(self, scene: SceneGraph, action: PrimitiveAction) -> SceneGraph:
        next_scene = scene.copy(timestep=scene.timestep + 1)
        cube = next_scene.objects["cube"]
        source = next_scene.regions["source"]
        target = next_scene.regions["target"]

        if action.name == "pick(cube)":
            if next_scene.robot.holding is None:
                next_scene.robot.holding = "cube"
                next_scene.robot.gripper_width = 0.0
        elif action.name == "place(source)":
            if next_scene.robot.holding == "cube":
                cube.pose.position = source.center.copy()
                next_scene.robot.holding = None
                next_scene.robot.gripper_width = 0.08
        elif action.name == "place(target)":
            if next_scene.robot.holding == "cube":
                cube.pose.position = target.center.copy()
                next_scene.robot.holding = None
                next_scene.robot.gripper_width = 0.08
        elif action.name == "noop":
            pass
        else:
            raise ValueError(f"Unknown primitive action: {action.name}")
        return next_scene
