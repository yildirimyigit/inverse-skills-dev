"""The predicate vocabulary for SlidingDrawer-v1. It shares nothing with the stow domain.

  drawer_open(drawer)          the drawer is pulled out far enough to count as open
  drawer_closed(drawer)        the drawer is shut
  at_handle_standoff(drawer)   the gripper waits above the handle (robot-relative)
  handle_grasped(drawer)       the fingers pinch the handle (robot-relative)

The two drawer predicates leave a band between them (ajar) where neither holds.

handle_grasped needs physical evidence. Its margin is the smaller of two terms:
how close the TCP is to the grasp point, and the weaker of the two fingers'
pinch forces on the handle. Far from the handle the distance term decides, so the
score still rises as the gripper comes closer; at the handle only a two-sided
pinch can push it over the threshold. Proximity alone never can.

Thresholds are set from the Stage 0 measurements (scripts/drawer_stage0_gate.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from inverse_skills.core.scene import SceneGraph
from inverse_skills.envs import drawer_env as geo
from inverse_skills.predicates.base import Predicate, PredicateRegistry, PredicateResult

DRAWER = "drawer"

# Measured in Stage 0 (10 seeds): push_close leaves the drawer 1-3 mm open;
# the drawer starts 125-179 mm open and pull_open leaves it at about 147 mm.
_CLOSED_BELOW = 0.01       # metres of opening at which drawer_closed scores 0.5
_CLOSED_TEMP = 0.004       # 0.8 at 7 mm
_OPEN_ABOVE = 0.08         # metres of opening at which drawer_open scores 0.5
_OPEN_TEMP = 0.02          # 0.8 at 94 mm
_STANDOFF_TOL = 0.02       # largest per-axis offset from the standoff that scores 0.5
_STANDOFF_TEMP = 0.005
# Real grasps, ones that pull the drawer open, were measured with the TCP up to
# 21 mm from the grasp point: a shallow pinch near the fin's top edge still
# holds. The distance term must not reject them; the pinch term is what rejects
# near misses.
_REACH_TOL = 0.04          # TCP distance from the grasp point that scores 0.5
_REACH_TEMP = 0.01
# A real pinch measured 15-17 N on each finger; near misses 0.5 N or less, or
# negative on one finger (pressing the fin from one side).
_PINCH_MIN = 2.0           # newtons, on the weaker finger, that score 0.5
_PINCH_TEMP = 2.0          # 0.8 at 3.4 N


@dataclass(frozen=True)
class DrawerOpenPredicate(Predicate):
    """Margin = opening − _OPEN_ABOVE."""

    weight: float = 1.0

    @property
    def name(self) -> str:
        return "drawer_open"

    @property
    def args(self) -> tuple[str, ...]:
        return (DRAWER,)

    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        opening = scene.get_object(DRAWER).metadata["opening"]
        return PredicateResult(self.name, self.args, margin=opening - _OPEN_ABOVE,
                               temperature=_OPEN_TEMP)


@dataclass(frozen=True)
class DrawerClosedPredicate(Predicate):
    """Margin = _CLOSED_BELOW − opening."""

    weight: float = 1.0

    @property
    def name(self) -> str:
        return "drawer_closed"

    @property
    def args(self) -> tuple[str, ...]:
        return (DRAWER,)

    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        opening = scene.get_object(DRAWER).metadata["opening"]
        return PredicateResult(self.name, self.args, margin=_CLOSED_BELOW - opening,
                               temperature=_CLOSED_TEMP)


@dataclass(frozen=True)
class AtHandleStandoffPredicate(Predicate):
    """Margin = _STANDOFF_TOL − the largest per-axis offset of the TCP from the standoff."""

    weight: float = 1.0
    robot_relative = True

    @property
    def name(self) -> str:
        return "at_handle_standoff"

    @property
    def args(self) -> tuple[str, ...]:
        return (DRAWER,)

    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        standoff = scene.get_object("handle").pose.position + np.array([0.0, 0.0, geo.STANDOFF_HEIGHT])
        offset = float(np.max(np.abs(scene.robot.ee_pose.position - standoff)))
        return PredicateResult(self.name, self.args, margin=_STANDOFF_TOL - offset,
                               temperature=_STANDOFF_TEMP)


@dataclass(frozen=True)
class HandleGraspedPredicate(Predicate):
    """Margin = min(reach, pinch), each in units of its own temperature:
    reach = (_REACH_TOL − TCP distance to the grasp point) / _REACH_TEMP
    pinch = (weaker finger's pinch force − _PINCH_MIN) / _PINCH_TEMP
    """

    weight: float = 1.0
    robot_relative = True

    @property
    def name(self) -> str:
        return "handle_grasped"

    @property
    def args(self) -> tuple[str, ...]:
        return (DRAWER,)

    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        distance = float(np.linalg.norm(scene.robot.ee_pose.position
                                        - scene.get_object("handle").pose.position))
        reach = (_REACH_TOL - distance) / _REACH_TEMP
        pinch = (min(scene.robot.metadata["pinch_force"]) - _PINCH_MIN) / _PINCH_TEMP
        return PredicateResult(self.name, self.args, margin=min(reach, pinch), temperature=1.0)


def registry() -> PredicateRegistry:
    """Every predicate the drawer domain can talk about."""
    return PredicateRegistry([
        DrawerOpenPredicate(),
        DrawerClosedPredicate(),
        AtHandleStandoffPredicate(),
        HandleGraspedPredicate(),
    ])
