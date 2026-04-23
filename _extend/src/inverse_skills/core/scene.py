from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from inverse_skills.core.geometry import Pose, Region, as_float_array


@dataclass
class ObjectState:
    name: str
    pose: Pose
    semantic_class: str | None = None
    size: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "pose": self.pose.to_dict(),
            "semantic_class": self.semantic_class,
            "size": None if self.size is None else self.size.tolist(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ObjectState":
        size = None if data.get("size") is None else as_float_array(data["size"], 3)
        return cls(
            name=data["name"],
            pose=Pose.from_dict(data["pose"]),
            semantic_class=data.get("semantic_class"),
            size=size,
            metadata=data.get("metadata", {}),
        )


@dataclass
class RobotState:
    q: np.ndarray
    gripper_width: float
    ee_pose: Pose | None = None
    holding: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "q": self.q.tolist(),
            "gripper_width": float(self.gripper_width),
            "ee_pose": None if self.ee_pose is None else self.ee_pose.to_dict(),
            "holding": self.holding,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RobotState":
        ee_pose = None if data.get("ee_pose") is None else Pose.from_dict(data["ee_pose"])
        return cls(
            q=np.asarray(data["q"], dtype=np.float32),
            gripper_width=float(data["gripper_width"]),
            ee_pose=ee_pose,
            holding=data.get("holding"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SceneGraph:
    timestep: int
    robot: RobotState
    objects: dict[str, ObjectState]
    regions: dict[str, Region] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_object(self, name: str) -> ObjectState:
        try:
            return self.objects[name]
        except KeyError as exc:
            raise KeyError(f"Object '{name}' is not present in scene") from exc

    def get_region(self, name: str) -> Region:
        try:
            return self.regions[name]
        except KeyError as exc:
            raise KeyError(f"Region '{name}' is not present in scene") from exc

    def to_dict(self) -> dict:
        return {
            "timestep": int(self.timestep),
            "robot": self.robot.to_dict(),
            "objects": {name: obj.to_dict() for name, obj in self.objects.items()},
            "regions": {name: region.to_dict() for name, region in self.regions.items()},
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SceneGraph":
        return cls(
            timestep=int(data["timestep"]),
            robot=RobotState.from_dict(data["robot"]),
            objects={name: ObjectState.from_dict(obj) for name, obj in data["objects"].items()},
            regions={name: Region.from_dict(region) for name, region in data.get("regions", {}).items()},
            metadata=data.get("metadata", {}),
        )
