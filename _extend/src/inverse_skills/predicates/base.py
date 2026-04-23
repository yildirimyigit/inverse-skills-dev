from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from inverse_skills.core.scene import SceneGraph


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


@dataclass(frozen=True)
class PredicateResult:
    name: str
    args: tuple[str, ...]
    margin: float
    temperature: float = 0.05

    @property
    def key(self) -> str:
        return f"{self.name}({','.join(self.args)})"

    @property
    def truth(self) -> bool:
        return self.margin >= 0.0

    @property
    def score(self) -> float:
        temp = max(float(self.temperature), 1e-6)
        return sigmoid(self.margin / temp)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "args": list(self.args),
            "key": self.key,
            "margin": float(self.margin),
            "score": float(self.score),
            "truth": bool(self.truth),
            "temperature": float(self.temperature),
        }


class Predicate(ABC):
    name: str
    args: tuple[str, ...]
    weight: float

    @property
    def key(self) -> str:
        return f"{self.name}({','.join(self.args)})"

    @abstractmethod
    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        raise NotImplementedError


class PredicateRegistry:
    """Small helper that evaluates and retrieves predicate instances by key."""

    def __init__(self, predicates: Iterable[Predicate]):
        self._predicates = {predicate.key: predicate for predicate in predicates}

    def keys(self) -> list[str]:
        return sorted(self._predicates.keys())

    def get(self, key: str) -> Predicate:
        try:
            return self._predicates[key]
        except KeyError as exc:
            raise KeyError(f"Predicate '{key}' is not registered") from exc

    def evaluate_all(self, scene: SceneGraph) -> dict[str, PredicateResult]:
        return {key: predicate.evaluate(scene) for key, predicate in self._predicates.items()}
