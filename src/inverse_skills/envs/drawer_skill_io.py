"""What the learned drawer skill observes and how it acts.

Shared by training (the hole environment) and, later, execution, so a deployed
policy sees exactly what it was trained on. Nothing here names the skill: the
observation starts with the margin of whichever predicate the planner asked the
skill to achieve.

Observation, 7 numbers:
  the achieved predicate's margin over its temperature;
  the TCP's offset from the handle's grasp point along x, y and z, in units of
  5 cm (the standoff is 5 cm above the grasp point);
  the gripper's opening, in units of its 8 cm maximum;
  each finger's pinch force on the handle, in units of 20 N (a firm pinch
  measured about 15 N).

Action, 4 numbers in [-1, 1]: the arm's own 4-D command. The first three move
the TCP along x, y and z, scaled so a step moves it at most about 1 cm; the
fourth is the gripper command (-1 closes, +1 opens).
"""

from __future__ import annotations

import numpy as np

from inverse_skills.envs import drawer_primitives as prim

ACTION_SCALE = 0.25        # one unit of the arm's command moves the TCP about 4 cm per step
_POSITION_UNIT = 0.05
_WIDTH_UNIT = 0.08
_FORCE_UNIT = 20.0


def observe(obs, achieve) -> np.ndarray:
    result = achieve.evaluate(prim.obs_to_scene(obs))
    offset = (prim.tcp_pos(obs) - prim.grasp_point(obs)) / _POSITION_UNIT
    left, right = prim.pinch_force(obs)
    return np.array([
        result.margin / result.temperature,
        offset[0], offset[1], offset[2],
        prim.gripper_width(obs) / _WIDTH_UNIT,
        left / _FORCE_UNIT,
        right / _FORCE_UNIT,
    ], dtype=np.float32)


def command(action) -> np.ndarray:
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    return np.array([a[0] * ACTION_SCALE, a[1] * ACTION_SCALE, a[2] * ACTION_SCALE, a[3]],
                    dtype=np.float32)
