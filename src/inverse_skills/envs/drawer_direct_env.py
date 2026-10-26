"""The direct baseline's learning problem: open the drawer from the standoff.

The targeted method (drawer_hole_env.py) learns only the missing grasp and
leaves the pull to the known pull_open controller. The direct baseline asks one
policy for the whole remaining inverse: the planner's unrestricted hole,
HOLE[drawer_open], whose spec (scripts/drawer_direct_spec.py) has the same start
condition and prefix as the grasp hole, and no fences.

Everything except the observation and the episode length is the grasp hole's
environment, inherited unchanged: the same handoff (the forward push, then
approach_handle), the same random start offset of up to 1 cm around the
standoff, the same 4-D action scaling, and the framework's reward, here
2 V(drawer_open) - 1 with no shaping. A step sends the policy's action to the
arm and nothing else: no scripted grasp or pull runs inside an episode.

Observation, 9 numbers: the grasp policy's own 7 (the margin of handle_grasped,
the TCP's offset from the grasp point, the gripper's opening, the two finger
forces), so the direct learner sees everything the targeted one did; then the
margin of the predicate it must achieve, drawer_open, and the drawer's opening
in units of its 20 cm travel.

Each finished episode is summarised in `episodes`, with the reason it failed:
  opened_then_lost  the goal held at some step, but not at the end
  out_of_time       the drawer was still opening over the last 10 steps
  never_grasped     handle_grasped never held
  lost_grasp        the grasp held at some step, but not at the end
  grasped_no_pull   the grasp held at the end and the drawer was not opening
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from inverse_skills.envs import drawer_env as geo
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.envs import drawer_skill_io as skill_io
from inverse_skills.envs.drawer_hole_env import SATISFIED, DrawerHoleEnv
from inverse_skills.envs.motion import step_in_place

MAX_STEPS = 100            # checked in scripts/drawer_direct_precheck.py
GRASPED = "handle_grasped(drawer)"
SETTLE_STEPS = 10          # the settling check after an episode
_OPENING_WINDOW = 10       # "still opening" looks this many steps back ...
_STILL_OPENING = 0.005     # ... for at least this much more opening


def observe(obs, achieve, grasp) -> np.ndarray:
    """The direct policy's 9 numbers: the grasp policy's 7, then achieve's margin and the opening."""
    target = achieve.evaluate(prim.obs_to_scene(obs))
    return np.concatenate([
        skill_io.observe(obs, grasp),
        [target.margin / target.temperature, prim.opening(obs) / geo.MAX_OPENING],
    ]).astype(np.float32)


class DrawerDirectEnv(DrawerHoleEnv):
    """From the standoff, open the drawer: grasp and pull, all learned."""

    def __init__(self, spec: dict, goal: tuple[set[str], set[str]], max_steps: int = MAX_STEPS, **kwargs):
        super().__init__(spec, max_steps=max_steps, **kwargs)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        self.goal_positive, self.goal_negative = goal
        self._grasp = self.registry.get(GRASPED)
        self._openings: list[float] = []         # this episode's opening after each step
        self._grasped: list[bool] = []           # ... and whether the grasp held
        self._gripper = 1.0                      # the last gripper command (the handoff leaves it open)

    def observe(self, obs) -> np.ndarray:
        return observe(obs, self._achieve, self._grasp)

    def goal_met(self, scores) -> bool:
        return (all(scores[p] >= SATISFIED for p in self.goal_positive)
                and all(scores[q] < SATISFIED for q in self.goal_negative))

    def reset(self, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        self._openings, self._grasped, self._gripper = [], [], 1.0
        return self.observe(self._obs), info

    def step(self, action):
        _, reward, terminated, truncated, info = super().step(action)
        self._gripper = float(np.asarray(action, dtype=np.float32).reshape(-1)[3])
        scores = info["scores"]
        self._openings.append(prim.opening(self._obs))
        self._grasped.append(scores[GRASPED] >= SATISFIED)
        info["goal_met"] = self.goal_met(scores)
        if truncated:
            self.episodes[-1].update(self.diagnose(info["goal_met"]))
        return self.observe(self._obs), reward, terminated, truncated, info

    def diagnose(self, goal_met: bool) -> dict:
        """What this episode did, and why it failed if it did."""
        openings, grasped = self._openings, self._grasped
        back = openings[max(0, len(openings) - 1 - _OPENING_WINDOW)]
        still_opening = openings[-1] - back >= _STILL_OPENING
        if goal_met:
            failure = None
        elif self._first_satisfied is not None:
            failure = "opened_then_lost"
        elif still_opening:
            failure = "out_of_time"
        elif not any(grasped):
            failure = "never_grasped"
        elif not grasped[-1]:
            failure = "lost_grasp"
        else:
            failure = "grasped_no_pull"
        return {"goal_met": goal_met, "failure": failure,
                "ever_grasped": any(grasped), "grasped_at_end": grasped[-1],
                "first_grasp_step": grasped.index(True) + 1 if any(grasped) else None,
                "max_opening_mm": round(max(openings) * 1000, 1),
                "end_opening_mm": round(openings[-1] * 1000, 1)}

    def settle(self, steps: int = SETTLE_STEPS) -> dict:
        """After an episode: hold the arm still, gripper as last commanded, and measure again."""
        self._obs = step_in_place(self._env, self._obs, steps, gripper_cmd=self._gripper)
        return {"settled_opening_mm": round(prim.opening(self._obs) * 1000, 1),
                "settled_goal_met": self.goal_met(self._scores(self._obs))}
