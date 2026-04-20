from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def as_float_array(values: Iterable[float], expected_dim: int | None = None) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if expected_dim is not None and arr.shape != (expected_dim,):
        raise ValueError(f"Expected shape ({expected_dim},), got {arr.shape}")
    return arr


def normalize_quat_xyzw(quat: Iterable[float]) -> np.ndarray:
    q = as_float_array(quat, 4)
    n = float(np.linalg.norm(q))
    if n < 1e-8:
        raise ValueError("Quaternion norm is too close to zero")
    return q / n


def quat_alignment_distance_xyzw(q1: Iterable[float], q2: Iterable[float]) -> float:
    """Sign-invariant quaternion distance in [0, 1].

    Returns 0 for identical orientations and approaches 1 for orthogonal quaternions.
    Quaternions are expected in xyzw convention.
    """
    qa = normalize_quat_xyzw(q1)
    qb = normalize_quat_xyzw(q2)
    return float(1.0 - abs(np.dot(qa, qb)))


@dataclass(frozen=True)
class Pose:
    """Rigid pose with position and xyzw quaternion."""

    position: np.ndarray
    quat_xyzw: np.ndarray

    @classmethod
    def identity(cls) -> "Pose":
        return cls(position=np.zeros(3, dtype=np.float32), quat_xyzw=np.array([0, 0, 0, 1], dtype=np.float32))

    @classmethod
    def from_vector(cls, vector: Iterable[float]) -> "Pose":
        arr = as_float_array(vector, 7)
        return cls(position=arr[:3], quat_xyzw=normalize_quat_xyzw(arr[3:]))

    def as_vector(self) -> np.ndarray:
        return np.concatenate([self.position.astype(np.float32), self.quat_xyzw.astype(np.float32)])

    def position_distance(self, other: "Pose") -> float:
        return float(np.linalg.norm(self.position - other.position))

    def orientation_distance(self, other: "Pose") -> float:
        return quat_alignment_distance_xyzw(self.quat_xyzw, other.quat_xyzw)

    def weighted_distance(self, other: "Pose", quat_weight: float = 0.2) -> float:
        return self.position_distance(other) + quat_weight * self.orientation_distance(other)

    def to_dict(self) -> dict:
        return {"position": self.position.tolist(), "quat_xyzw": self.quat_xyzw.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "Pose":
        return cls(position=as_float_array(data["position"], 3), quat_xyzw=normalize_quat_xyzw(data["quat_xyzw"]))


@dataclass(frozen=True)
class Region:
    """Axis-aligned box region used for source, target, support, or container slots."""

    name: str
    center: np.ndarray
    half_extents: np.ndarray

    @classmethod
    def from_bounds(cls, name: str, lower: Iterable[float], upper: Iterable[float]) -> "Region":
        lo = as_float_array(lower, 3)
        hi = as_float_array(upper, 3)
        if np.any(hi <= lo):
            raise ValueError("Every upper bound must be larger than the lower bound")
        return cls(name=name, center=(lo + hi) / 2.0, half_extents=(hi - lo) / 2.0)

    def signed_margin(self, point: Iterable[float]) -> float:
        """Positive inside, zero on boundary, negative outside."""
        p = as_float_array(point, 3)
        return float(np.min(self.half_extents - np.abs(p - self.center)))

    def contains(self, point: Iterable[float]) -> bool:
        return self.signed_margin(point) >= 0.0

    def to_dict(self) -> dict:
        return {"name": self.name, "center": self.center.tolist(), "half_extents": self.half_extents.tolist()}

    @classmethod
    def from_dict(cls, data: dict) -> "Region":
        return cls(name=data["name"], center=as_float_array(data["center"], 3), half_extents=as_float_array(data["half_extents"], 3))
