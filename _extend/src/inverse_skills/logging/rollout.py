from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inverse_skills.core.scene import SceneGraph


@dataclass
class ForwardRollout:
    skill_name: str
    demo_id: str
    scenes: list[SceneGraph]
    metadata: dict[str, Any] = field(default_factory=dict)

    def first(self) -> SceneGraph:
        if not self.scenes:
            raise ValueError("Rollout contains no scenes")
        return self.scenes[0]

    def last(self) -> SceneGraph:
        if not self.scenes:
            raise ValueError("Rollout contains no scenes")
        return self.scenes[-1]

    def to_dict(self) -> dict:
        return {
            "skill_name": self.skill_name,
            "demo_id": self.demo_id,
            "metadata": self.metadata,
            "scenes": [scene.to_dict() for scene in self.scenes],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ForwardRollout":
        return cls(
            skill_name=data["skill_name"],
            demo_id=data["demo_id"],
            metadata=data.get("metadata", {}),
            scenes=[SceneGraph.from_dict(item) for item in data["scenes"]],
        )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


class ForwardRolloutLogger:
    def __init__(self, skill_name: str, demo_id: str, metadata: dict[str, Any] | None = None):
        self.skill_name = skill_name
        self.demo_id = demo_id
        self.metadata = metadata or {}
        self.scenes: list[SceneGraph] = []

    def append(self, scene: SceneGraph) -> None:
        self.scenes.append(scene)

    def as_rollout(self) -> ForwardRollout:
        return ForwardRollout(
            skill_name=self.skill_name,
            demo_id=self.demo_id,
            scenes=self.scenes,
            metadata=self.metadata,
        )

    def save_json(self, path: str | Path) -> None:
        self.as_rollout().save_json(path)


def load_rollout(path: str | Path) -> ForwardRollout:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ForwardRollout.from_dict(data)
