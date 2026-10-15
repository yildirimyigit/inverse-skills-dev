"""The symbolic library for SlidingDrawer-v1. It shares nothing with the stow domain.

The robot's known actions for the inverse, with what Stage 0 measured them to do:

  approach_handle  brings the open gripper to the standoff above the handle. It
                   works whether the drawer is shut, ajar or open, so it has no
                   precondition.
  pull_open        needs the handle grasped, leaves the drawer open, and lets go
                   of the handle at the end. It does not need the drawer to be
                   shut: Stage 0 pulled the drawer open from 4 cm ajar on every seed.

No action grasps the handle. That is the missing capability.

Two facts hold for every action, so the planner applies them instead of each
schema repeating them: the drawer is never both open and shut, and the gripper
is never both at the standoff and on the handle.

`scripted_grasp()` writes Stage 0's test grasp as an action. It is not part of
the library; it exists only for the ablation that adds it.
"""

from __future__ import annotations

import json
from pathlib import Path

from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.operators.hole_planner import Action, Domain

OPEN = "drawer_open(drawer)"
CLOSED = "drawer_closed(drawer)"
STANDOFF = "at_handle_standoff(drawer)"
GRASPED = "handle_grasped(drawer)"


def library() -> list[Action]:
    return [
        Action("approach_handle(drawer)", add=frozenset({STANDOFF})),
        Action("pull_open(drawer)", pre=frozenset({GRASPED}), add=frozenset({OPEN}),
               delete=frozenset({GRASPED})),
    ]


def scripted_grasp() -> Action:
    """Stage 0's test grasp as an action: from the standoff, grasp the handle."""
    return Action("grasp_handle(drawer)", pre=frozenset({STANDOFF}), add=frozenset({GRASPED}))


def domain(extra_actions: list[Action] | None = None) -> Domain:
    registry = dp.registry()
    return Domain(
        actions=library() + list(extra_actions or []),
        scored=frozenset(registry.keys()),
        robot_relative=frozenset(registry.robot_relative_keys()),
        mutex_groups=[frozenset({OPEN, CLOSED}), frozenset({STANDOFF, GRASPED})],
    )


def goal_from_operator(path: str | Path) -> tuple[set[str], set[str]]:
    """Positive and negative goal literals from an extracted operator's inverse target."""
    terms = json.loads(Path(path).read_text())["operator"]["inverse_target_terms"]
    positive = {t["key"] for t in terms if t["polarity"] == "positive"}
    negative = {t["key"] for t in terms if t["polarity"] == "negative"}
    return positive, negative
