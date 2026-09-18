"""The symbolic library for StowCube-v1, with honest preconditions.

Honest means the model admits what the scripted primitives can and cannot do:

  * `pick(cube_a)` needs `clear_of_walls(cube_a)`, because the pocket's arms
    block both grasp directions while A is seated. `pick(cube_b)` carries no
    such precondition: the arms stop short of the mouth.
  * `push` only pushes away from the robot, which is the demonstrated motion,
    so it is written as a directed action per source slot. Nothing in the
    library pulls a cube back toward the robot — that is the gap.
  * grasping and pushing close the gripper; placing opens it.

`holding` is the planner's bookkeeping fluent rather than a scored predicate:
the executive knows what the gripper holds, so the registry never needs it.
"""

from __future__ import annotations

import json
from pathlib import Path

from inverse_skills.envs import stow_cube as geo
from inverse_skills.operators.hole_planner import Action, Domain

CUBES = ("cube_a", "cube_b")
SLOT_X = {
    "src_a": geo.SRC_A_XY[0],
    "src_b": geo.SRC_B_XY[0],
    "mouth": geo.MOUTH_B_XY[0],
    "pocket": geo.POCKET_A_XY[0],
}
# Only A can be wedged, so only A carries a clearance predicate — as in the registry.
CLEAR_OBJECTS = ("cube_a",)
PUSHES = (("cube_a", "src_a", "pocket"), ("cube_b", "src_b", "mouth"))

_HOLDING = frozenset(f"holding({cube})" for cube in CUBES)


def _in(cube: str, slot: str) -> str:
    return f"in_region({cube},{slot})"


def _clear(cube: str) -> str:
    return f"clear_of_walls({cube})"


def _slot_is_clear(slot: str) -> bool:
    return SLOT_X[slot] <= geo.X_CLEAR


def library() -> list[Action]:
    actions: list[Action] = []
    for cube in CUBES:
        others = frozenset(f"tcp_near({o})" for o in CUBES if o != cube)
        grasp_pre = frozenset({_clear(cube)}) if cube in CLEAR_OBJECTS else frozenset()
        actions.append(Action(
            name=f"approach({cube})",
            pre_absent=_HOLDING,
            add=frozenset({f"tcp_near({cube})"}),
            delete=frozenset({"gripper_open()"}) | others,
        ))
        actions.append(Action(
            name=f"pick({cube})",
            pre=grasp_pre,
            pre_absent=_HOLDING,
            add=frozenset({f"holding({cube})", f"tcp_near({cube})"}),
            delete=frozenset({"gripper_open()"}) | others
                   | frozenset(_in(cube, slot) for slot in SLOT_X),
        ))
        for slot in SLOT_X:
            actions.append(Action(
                name=f"place({cube},{slot})",
                pre=frozenset({f"holding({cube})"}),
                add=frozenset({_in(cube, slot), "gripper_open()"}),
                delete=frozenset({f"holding({cube})"}),
            ))
    for cube, from_slot, to_slot in PUSHES:
        actions.append(Action(
            name=f"push({cube},{from_slot}->{to_slot})",
            pre=frozenset({_in(cube, from_slot)}),
            pre_absent=_HOLDING,
            add=frozenset({_in(cube, to_slot), f"tcp_near({cube})"}),
            delete=frozenset({"gripper_open()"}),
        ))
    return actions


def domain(extra_actions: list[Action] | None = None) -> Domain:
    implications = {
        _in(cube, slot): frozenset({_clear(cube)})
        for cube in CLEAR_OBJECTS for slot in SLOT_X if _slot_is_clear(slot)
    }
    mutex = [frozenset(_in(cube, slot) for slot in SLOT_X) for cube in CUBES]
    mutex += [frozenset({_clear(cube), _in(cube, "pocket")}) for cube in CLEAR_OBJECTS]
    candidates = [_clear(cube) for cube in CLEAR_OBJECTS]
    candidates += [_in(cube, slot) for cube in CUBES for slot in SLOT_X]
    candidates += [f"tcp_near({cube})" for cube in CUBES]
    return Domain(
        actions=library() + list(extra_actions or []),
        hole_candidates=candidates,
        mutex_groups=mutex,
        implications=implications,
    )


def stowed_state() -> frozenset[str]:
    """The symbolic state the forward skill leaves behind."""
    return frozenset({_in("cube_a", "pocket"), _in("cube_b", "mouth"), "gripper_open()"})


def goal_from_operator(path: str | Path) -> tuple[set[str], set[str]]:
    """Positive and negative goal literals from an extracted operator's inverse target."""
    terms = json.loads(Path(path).read_text())["operator"]["inverse_target_terms"]
    positive = {t["key"] for t in terms if t["polarity"] == "positive"}
    negative = {t["key"] for t in terms if t["polarity"] == "negative"}
    return positive, negative
