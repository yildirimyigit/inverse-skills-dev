from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inverse_skills.core.geometry import Pose
from inverse_skills.core.scene import SceneGraph
from inverse_skills.predicates.base import Predicate, PredicateResult


@dataclass(frozen=True)
class InRegionPredicate(Predicate):
    object_name: str
    region_name: str
    temperature: float = 0.01
    weight: float = 1.0

    @property
    def name(self) -> str:
        return "in_region"

    @property
    def args(self) -> tuple[str, ...]:
        return (self.object_name, self.region_name)

    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        obj = scene.get_object(self.object_name)
        region = scene.get_region(self.region_name)
        margin = region.signed_margin(obj.pose.position)
        return PredicateResult(self.name, self.args, margin=margin, temperature=self.temperature)


@dataclass(frozen=True)
class NearPredicate(Predicate):
    object_a: str
    object_b: str
    distance_threshold: float = 0.05
    temperature: float = 0.02
    weight: float = 1.0

    @property
    def name(self) -> str:
        return "near"

    @property
    def args(self) -> tuple[str, ...]:
        return (self.object_a, self.object_b)

    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        a = scene.get_object(self.object_a)
        b = scene.get_object(self.object_b)
        dist = float(np.linalg.norm(a.pose.position - b.pose.position))
        return PredicateResult(self.name, self.args, margin=self.distance_threshold - dist, temperature=self.temperature)


@dataclass(frozen=True)
class AtPosePredicate(Predicate):
    object_name: str
    target_pose: Pose
    slot_name: str = "target_pose"
    distance_threshold: float = 0.05
    quat_weight: float = 0.2
    temperature: float = 0.02
    weight: float = 1.0

    @property
    def name(self) -> str:
        return "at_pose"

    @property
    def args(self) -> tuple[str, ...]:
        return (self.object_name, self.slot_name)

    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        obj = scene.get_object(self.object_name)
        dist = obj.pose.weighted_distance(self.target_pose, quat_weight=self.quat_weight)
        return PredicateResult(self.name, self.args, margin=self.distance_threshold - dist, temperature=self.temperature)


@dataclass(frozen=True)
class GripperOpenPredicate(Predicate):
    min_width: float = 0.04
    temperature: float = 0.005
    weight: float = 1.0

    @property
    def name(self) -> str:
        return "gripper_open"

    @property
    def args(self) -> tuple[str, ...]:
        return tuple()

    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        margin = float(scene.robot.gripper_width - self.min_width)
        return PredicateResult(self.name, self.args, margin=margin, temperature=self.temperature)


@dataclass(frozen=True)
class TcpNearObjectPredicate(Predicate):
    """Robot end-effector is near the named object.

    This is the natural representation of preconditions like
    "in pusher contact" or "ready to grasp" without elevating the TCP to a
    first-class scene object. Margin = distance_threshold − ||ee − obj||.
    """

    object_name: str
    distance_threshold: float = 0.05
    temperature: float = 0.02
    weight: float = 1.0

    @property
    def name(self) -> str:
        return "tcp_near"

    @property
    def args(self) -> tuple[str, ...]:
        return (self.object_name,)

    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        obj = scene.get_object(self.object_name)
        if scene.robot.ee_pose is None:
            margin = -float(self.distance_threshold)
        else:
            dist = float(np.linalg.norm(obj.pose.position - scene.robot.ee_pose.position))
            margin = float(self.distance_threshold - dist)
        return PredicateResult(self.name, self.args, margin=margin, temperature=self.temperature)


@dataclass(frozen=True)
class HoldingPredicate(Predicate):
    object_name: str
    temperature: float = 0.25
    weight: float = 1.0

    @property
    def name(self) -> str:
        return "holding"

    @property
    def args(self) -> tuple[str, ...]:
        return (self.object_name,)

    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        margin = 1.0 if scene.robot.holding == self.object_name else -1.0
        return PredicateResult(self.name, self.args, margin=margin, temperature=self.temperature)
