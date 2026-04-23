from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def as_float_array(values: Iterable[float], expected_len: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.shape != (expected_len,):
        raise ValueError(f"Expected shape ({expected_len},), got {arr.shape}")
    return arr


@dataclass
class Pose:
    position: np.ndarray
    quat_xyzw: np.ndarray

    def __post_init__(self) -> None:
        self.position = as_float_array(self.position, 3)
        self.quat_xyzw = as_float_array(self.quat_xyzw, 4)

    def weighted_distance(self, other: "Pose", quat_weight: float = 0.2) -> float:
        pos_dist = float(np.linalg.norm(self.position - other.position))
        qa = self.quat_xyzw / max(np.linalg.norm(self.quat_xyzw), 1e-8)
        qb = other.quat_xyzw / max(np.linalg.norm(other.quat_xyzw), 1e-8)
        quat_alignment = float(abs(np.dot(qa, qb)))
        quat_dist = 1.0 - quat_alignment
        return pos_dist + quat_weight * quat_dist

    def to_dict(self) -> dict:
        return {
            "position": self.position.tolist(),
            "quat_xyzw": self.quat_xyzw.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Pose":
        return cls(position=data["position"], quat_xyzw=data["quat_xyzw"])


@dataclass
class Region:
    name: str
    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        self.lower = as_float_array(self.lower, 3)
        self.upper = as_float_array(self.upper, 3)
        if np.any(self.lower >= self.upper):
            raise ValueError("Region lower bounds must be strictly smaller than upper bounds")

    @property
    def center(self) -> np.ndarray:
        return (self.lower + self.upper) / 2.0

    def signed_margin(self, point: np.ndarray) -> float:
        point = as_float_array(point, 3)
        lower_margin = point - self.lower
        upper_margin = self.upper - point
        if np.all(lower_margin >= 0.0) and np.all(upper_margin >= 0.0):
            return float(np.min(np.minimum(lower_margin, upper_margin)))
        outside = np.maximum(self.lower - point, 0.0) + np.maximum(point - self.upper, 0.0)
        return -float(np.linalg.norm(outside))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lower": self.lower.tolist(),
            "upper": self.upper.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Region":
        return cls(name=data["name"], lower=data["lower"], upper=data["upper"])

    @classmethod
    def from_bounds(cls, name: str, lower: Iterable[float], upper: Iterable[float]) -> "Region":
        return cls(name=name, lower=lower, upper=upper)
