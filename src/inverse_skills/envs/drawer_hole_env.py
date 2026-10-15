"""The drawer's hole as a gym environment, built from the planner's hole spec.

As for the stow hole, the spec Stage 2 wrote is the whole learning problem:

  achieve          the predicate the skill must establish: the reward's only active term
  preserve         fences: literals to keep true (none for the drawer)
  preserve_absent  fences: literals to keep false (none for the drawer)
  start            what must hold when the skill takes over: checked at the handoff
  prefix           the plan up to the hole, run in reset()

Each episode starts where the executive will hand over: the forward push has
shut the drawer, and approach_handle has brought the open gripper to the
standoff above the handle. A small random offset of the TCP, up to 1 cm on each
axis, varies the start while keeping the start condition true. The policy then
learns only what comes next, the short reach and the grasp: it never approaches
the handle and never pulls the drawer.

Reward: the framework's signed score, 2 V(achieve) - 1, plus the one-sided
fences (none here), with no shaping terms. Episodes always run 40 steps: with
this reward, ending an episode on success would make stalling just short of
success the best policy, as the stow task showed.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from inverse_skills.envs import drawer_env as geo
from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.envs import drawer_skill_io as skill_io
from inverse_skills.envs.motion import step_in_place, step_toward

MAX_STEPS = 40
START_OFFSET = 0.01        # the TCP starts up to 1 cm from the standoff on each axis
SATISFIED = 0.8            # the framework's truth threshold
# The library steps a prefix can contain, by name.
_PRIMITIVES = {"approach_handle": prim.approach_handle}


class DrawerHoleEnv(gym.Env):
    """One hole: from the standoff, establish the handle grasp."""

    metadata = {"render_modes": []}

    def __init__(self, spec: dict, max_steps: int = MAX_STEPS,
                 seed_pool: tuple[int, ...] = tuple(range(32)),
                 start_offset: float = START_OFFSET, render_mode: str | None = None):
        super().__init__()
        self._env = prim.make_env(render_mode=render_mode)
        self.registry = dp.registry()
        self.spec = spec
        self.achieve = spec["achieve"]
        self._achieve = self.registry.get(self.achieve)
        self._prefix = [a["name"].split("(")[0] for a in spec["prefix"]]
        self.max_steps = max_steps
        self.seed_pool = tuple(seed_pool)        # training episodes draw their scene from this pool
        self.start_offset = start_offset
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self._obs = None
        self._steps = 0
        self._first_satisfied: int | None = None
        self._best_score = 0.0
        # Push and approach take ~0.3 s of simulation and are deterministic per
        # seed, so each seed's handoff is simulated once and restored afterwards.
        self._handoffs: dict[int, dict] = {}
        self.episodes: list[dict] = []           # one summary per finished episode
        self.total_steps = 0                     # steps taken since this environment was created

    # -- handoff --------------------------------------------------------------

    def _handoff(self, episode_seed: int) -> dict:
        """The simulator state after the push and the prefix, for this seed."""
        if episode_seed not in self._handoffs:
            obs = prim.push_close(self._env, prim.reset(self._env, episode_seed))
            for name in self._prefix:
                obs = _PRIMITIVES[name](self._env, obs)
            self._handoffs[episode_seed] = self._env.unwrapped.get_state_dict()
        return self._handoffs[episode_seed]

    # -- scoring --------------------------------------------------------------

    def _scores(self, obs) -> dict[str, float]:
        scene = prim.obs_to_scene(obs)
        return {k: float(r.score) for k, r in self.registry.evaluate_all(scene).items()}

    def _reward(self, scores) -> float:
        """Active term plus one-sided fences: the framework's signed scores."""
        reward = 2.0 * scores[self.achieve] - 1.0
        for key in self.spec["preserve"]:
            reward += min(0.0, 2.0 * scores[key] - 1.0)
        for key in self.spec["preserve_absent"]:
            reward += min(0.0, 1.0 - 2.0 * scores[key])
        return float(reward)

    def _fences_held(self, scores) -> bool:
        return (all(scores[k] >= SATISFIED for k in self.spec["preserve"])
                and all(scores[k] < SATISFIED for k in self.spec["preserve_absent"]))

    # -- gym API --------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            episode_seed = int(seed)
        else:
            episode_seed = int(self.np_random.choice(self.seed_pool))
        self._env.unwrapped.set_state_dict(self._handoff(episode_seed))
        obs = self._env.unwrapped.get_obs()
        offset = self.np_random.uniform(-self.start_offset, self.start_offset, size=3)
        standoff = prim.grasp_point(obs) + np.array([0.0, 0.0, geo.STANDOFF_HEIGHT])
        obs = step_toward(self._env, obs, standoff + offset, 15, 0.001, 1.0, 0.5)
        obs = step_in_place(self._env, obs, 1, gripper_cmd=1.0)   # so contact readings are this episode's
        self._obs = obs
        self._steps = 0
        self._first_satisfied = None
        scores = self._scores(obs)
        self._best_score = scores[self.achieve]
        start_ok = all(scores[p] >= SATISFIED for p in self.spec["start"])
        info = {"episode_seed": episode_seed, "start_offset_mm": (offset * 1000).round(1).tolist(),
                "handoff_valid": bool(start_ok and scores[self.achieve] < SATISFIED), "scores": scores}
        return skill_io.observe(obs, self._achieve), info

    def step(self, action):
        self._obs, *_ = self._env.step(torch.tensor(skill_io.command(action)))
        self._steps += 1
        self.total_steps += 1
        scores = self._scores(self._obs)
        satisfied = scores[self.achieve] >= SATISFIED
        self._best_score = max(self._best_score, scores[self.achieve])
        if satisfied and self._first_satisfied is None:
            self._first_satisfied = self._steps
        truncated = self._steps >= self.max_steps
        info = {
            "scores": scores,
            "postcondition": satisfied,
            "first_satisfied_step": self._first_satisfied,
            "fences_held": self._fences_held(scores),
        }
        if truncated:
            self.episodes.append({"success": satisfied, "best_score": self._best_score,
                                  "first_satisfied_step": self._first_satisfied,
                                  "end_step": self.total_steps})
        return skill_io.observe(self._obs, self._achieve), self._reward(scores), False, truncated, info

    def close(self):
        self._env.close()


def load_spec(path) -> dict:
    return json.loads(Path(path).read_text())
