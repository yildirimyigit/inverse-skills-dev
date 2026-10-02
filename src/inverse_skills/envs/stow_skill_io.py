"""What a learned stow skill observes and how it acts.

Shared by training (the hole environment) and execution (the executive), so a
deployed policy sees exactly what it was trained on. Nothing here names the
skill: the observation is built from whichever predicate the planner asked the
skill to achieve, and from the object that predicate is about.

The action space is the one chosen for contact skills in this scene: a stroke
along -x and a press along -z with the gripper closed. The Stage 0 sweep showed
that balance is the whole of the drag, and keeping the RL problem this small
was a deliberate choice.
"""

from __future__ import annotations

import numpy as np

from inverse_skills.envs import stow_cube as geo
from inverse_skills.envs import stow_primitives as prim

ACTION_STROKE = 0.3        # max per-step stroke command, from the Stage 0 sweep
ACTION_PRESS = 0.15        # max per-step press command


def observe(obs, regions, achieve, obj: str) -> np.ndarray:
    """The achieved predicate's margin over its temperature, and the TCP's offset
    from the object in cube half-widths."""
    result = achieve.evaluate(prim.obs_to_scene(obs, regions))
    cube = prim.cube_pos(obs, obj)
    tcp = prim.tcp_pos(obs)
    return np.array([
        result.margin / result.temperature,
        (tcp[0] - cube[0]) / geo.CUBE_HALF,
        (tcp[1] - cube[1]) / geo.CUBE_HALF,
        (tcp[2] - cube[2]) / geo.CUBE_HALF,
    ], dtype=np.float32)


def command(action) -> np.ndarray:
    """Map the 2D policy action to the arm's 4D delta command, gripper closed."""
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    return np.array([a[0] * ACTION_STROKE, 0.0, a[1] * ACTION_PRESS, -1.0], dtype=np.float32)
