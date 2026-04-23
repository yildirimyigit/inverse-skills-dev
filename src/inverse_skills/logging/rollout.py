from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from inverse_skills.core.scene import SceneGraph


@dataclass
class ForwardRollout:
    skill_name: str
    demo_id: str
    scenes: list[SceneGraph]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenes:
            raise ValueError("ForwardRollout must contain at least one scene")

    def first(self) -> SceneGraph:
        return self.scenes[0]

    def last(self) -> SceneGraph:
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
            scenes=[SceneGraph.from_dict(scene) for scene in data["scenes"]],
            metadata=data.get("metadata", {}),
        )

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "ForwardRollout":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
