"""The predicate vocabulary for StowCube-v1.

One new class, `ClearOfWallsPredicate`. Everything else is the existing
geometric library, grounded on the slots that `stow_primitives.regions()`
puts in each scene.

Source slots use `in_region` rather than `at_pose` so that one registry works
across demonstrations: a region is looked up in the scene, so each episode's
own ±5 mm sampled source is used, while an `at_pose` target would be frozen
into the predicate at construction time. The slot half-width is the 2 cm
restoration tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

from inverse_skills.core.scene import SceneGraph
from inverse_skills.envs import stow_cube as geo
from inverse_skills.predicates.base import Predicate, PredicateRegistry, PredicateResult
from inverse_skills.predicates.geometric import (
    GripperOpenPredicate,
    InRegionPredicate,
    TcpNearObjectPredicate,
)

CUBE_A = "cube_a"
CUBE_B = "cube_b"

_SLOT_TEMP = 0.01          # slot scores saturate a centimetre inside the region
_CLEAR_TEMP = 0.02         # clear_of_walls ramps over the 7.5 cm the drag covers
_TCP_NEAR_THRESHOLD = 0.05
# Calibrated to the state `approach` actually produces: seated on a cube the TCP
# sits 30.6–32.0 mm from its centre, which scores 0.86 at this temperature but
# only 0.72 at 40 mm — below the 0.8 at which the planner counts it as holding.
_TCP_NEAR_TEMP = 0.02
# Open means wider than any cube in the scene. Holding a 4 cm cube leaves the
# fingers 36-37 mm apart (measured), which the 40 mm default scored 0.34 — close
# enough to the threshold that a slightly rotated grasp read as "open".
# Fully open is 80 mm.
_GRIPPER_OPEN_WIDTH = 0.06


@dataclass(frozen=True)
class ClearOfWallsPredicate(Predicate):
    """The object has been pulled far enough out of the pocket to be grasped.

    Margin = x_threshold − x_object, so the score crosses 0.5 exactly at the
    threshold and rises as the object comes further out toward the robot.
    """

    object_name: str
    x_threshold: float = geo.X_CLEAR
    temperature: float = _CLEAR_TEMP
    weight: float = 1.0

    @property
    def name(self) -> str:
        return "clear_of_walls"

    @property
    def args(self) -> tuple[str, ...]:
        return (self.object_name,)

    def evaluate(self, scene: SceneGraph) -> PredicateResult:
        x = float(scene.get_object(self.object_name).pose.position[0])
        return PredicateResult(self.name, self.args, margin=self.x_threshold - x,
                               temperature=self.temperature)


def registry() -> PredicateRegistry:
    """Every predicate the stow domain can talk about."""
    return PredicateRegistry([
        InRegionPredicate(CUBE_A, "src_a", temperature=_SLOT_TEMP),
        InRegionPredicate(CUBE_B, "src_b", temperature=_SLOT_TEMP),
        InRegionPredicate(CUBE_A, "pocket", temperature=_SLOT_TEMP),
        InRegionPredicate(CUBE_B, "mouth", temperature=_SLOT_TEMP),
        TcpNearObjectPredicate(CUBE_A, distance_threshold=_TCP_NEAR_THRESHOLD,
                               temperature=_TCP_NEAR_TEMP),
        TcpNearObjectPredicate(CUBE_B, distance_threshold=_TCP_NEAR_THRESHOLD,
                               temperature=_TCP_NEAR_TEMP),
        GripperOpenPredicate(min_width=_GRIPPER_OPEN_WIDTH),
        ClearOfWallsPredicate(CUBE_A),
    ])
