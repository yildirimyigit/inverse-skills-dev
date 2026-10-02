"""A hole as a gym environment, built from the planner's hole spec.

The spec (`hole_planner.hole_spec`, written by Stage 2) is the whole learning
problem, and nothing about it is chosen here:

  achieve          the predicate the skill must establish — the reward's active term
  preserve         literals the plan relies on across the hole — one-sided fences
  preserve_absent  literals the plan needs to stay false — one-sided fences too
  start            what must hold when the skill takes over — checked at handoff
  prefix           the plan up to the hole — run by the executive in `reset()`

so the episode starts in exactly the state the executive will hand over.

Deliberately small, because the RL is the part most likely to fail: a 2D
action and a 4D observation (`stow_skill_io`), and the framework's reward with
no additions. Episodes run a fixed horizon: the active reward is positive once
the predicate holds, so ending on success would make stalling just short of it
the optimal policy.

A start-state curriculum loosens the handoff from a firm seat on the object to
a light touch, so the first episodes see reward gradient immediately.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs import stow_primitives as prim
from inverse_skills.envs import stow_skill_io as skill_io
from inverse_skills.envs.stow_executive import SATISFIED, execute_plan
from inverse_skills.operators.hole_planner import Action

MAX_STEPS = 40
# The episode starts seated on the object, because `approach` leaves it there.
# The curriculum varies how firmly: a deep press target gives a large PD error
# and a strong grip, a shallow one barely bites.
_SEAT_FIRM = 0.02
_SEAT_LIGHT = 0.005
_MAX_PREFIX_ATTEMPTS = 5


class StowHoleEnv(gym.Env):
    """One hole, one predicate, one contact skill."""

    metadata = {"render_modes": []}

    def __init__(self, spec: dict, max_steps: int = MAX_STEPS, curriculum: float = 0.0,
                 seed_pool: tuple[int, ...] = tuple(range(32)),
                 render_mode: str | None = None):
        super().__init__()
        self._env = prim.make_env(render_mode=render_mode)
        self.registry = sp.registry()
        self.spec = spec
        self.achieve = spec["achieve"]
        self.object = spec["object"]
        self._achieve = self.registry.get(self.achieve)
        self._prefix = [Action.from_dict(a) for a in spec["prefix"]]
        if any(a.hole_for for a in self._prefix):
            raise ValueError("the prefix contains another hole; learn that one first")
        self.max_steps = max_steps
        self.curriculum = float(curriculum)     # 0 = firm seat, 1 = light touch
        # Episodes draw from a fixed pool so the cached handoffs stay bounded.
        self.seed_pool = tuple(seed_pool)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,),
                                            dtype=np.float32)
        self._obs = None
        self._regions = None
        self._steps = 0
        self._first_satisfied: int | None = None
        # The prefix costs ~0.4 s of simulation and is deterministic per seed,
        # so each seed's handoff is simulated once and restored thereafter.
        self._handoffs: dict[int, tuple] = {}

    # -- handoff --------------------------------------------------------------

    def set_curriculum(self, value: float) -> None:
        self.curriculum = float(np.clip(value, 0.0, 1.0))

    def _run_prefix(self, seed: int):
        """The forward skill, then the plan up to the hole, checked step by step."""
        obs = prim.reset(self._env, seed)
        self._regions = prim.regions(obs)
        obs = prim.forward_stow(self._env, obs)
        obs, report = execute_plan(self._env, obs, self._prefix, regions=self._regions)
        prefix_ok = len(report.steps) == len(self._prefix) and all(s.ok for s in report.steps)
        return obs, prefix_ok

    def _handoff(self, episode_seed: int) -> tuple:
        """The cached post-prefix simulator state for this seed, computed once.

        A handoff is valid when every prefix step met its postcondition and the
        hole's start literals hold as measured. Invalid ones are re-rolled with a
        fresh scene: a prefix that failed is the executive's problem, not
        something the policy should be asked to learn from.
        """
        if episode_seed not in self._handoffs:
            attempts, valid = 0, False
            for attempt in range(_MAX_PREFIX_ATTEMPTS):
                attempts = attempt + 1
                obs, prefix_ok = self._run_prefix(episode_seed + attempt * 10_000)
                scores = self._scores(obs)
                valid = prefix_ok and all(scores[p] >= SATISFIED for p in self.spec["start"])
                if valid:
                    break
            state = self._env.unwrapped.get_state_dict()
            self._handoffs[episode_seed] = (state, self._regions, attempts, valid)
        return self._handoffs[episode_seed]

    def _descend_to_start(self, obs):
        """Seat the fingertips on the object, as firmly as the curriculum says."""
        depth = _SEAT_FIRM + self.curriculum * (_SEAT_LIGHT - _SEAT_FIRM)
        return prim.seat_on(self._env, obs, self.object, depth=depth)

    # -- scoring --------------------------------------------------------------

    def _scores(self, obs) -> dict[str, float]:
        scene = prim.obs_to_scene(obs, self._regions)
        return {k: float(r.score) for k, r in self.registry.evaluate_all(scene).items()}

    def _observation(self, obs) -> np.ndarray:
        return skill_io.observe(obs, self._regions, self._achieve, self.object)

    def _reward(self, scores) -> float:
        """Active term plus one-sided fences — the framework's signed scores."""
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
        state, regions, attempts, valid = self._handoff(episode_seed)
        self._env.unwrapped.set_state_dict(state)
        self._regions = regions
        obs = self._descend_to_start(self._env.unwrapped.get_obs())
        self._obs = obs
        self._steps = 0
        self._first_satisfied = None
        scores = self._scores(obs)
        info = {"prefix_attempts": attempts, "curriculum": self.curriculum, "scores": scores,
                "handoff_valid": bool(valid and scores[self.achieve] < SATISFIED)}
        return self._observation(obs), info

    def step(self, action):
        self._obs, *_ = self._env.step(torch.tensor(skill_io.command(action)))
        self._steps += 1

        scores = self._scores(self._obs)
        satisfied = scores[self.achieve] >= SATISFIED
        if satisfied and self._first_satisfied is None:
            self._first_satisfied = self._steps
        # The episode is never cut short on success: see the module docstring.
        # The executive stops the skill when its postcondition holds; the
        # training horizon is a separate matter.
        truncated = self._steps >= self.max_steps
        info = {
            "scores": scores,
            "postcondition": satisfied,
            "first_satisfied_step": self._first_satisfied,
            "object_x": float(prim.cube_pos(self._obs, self.object)[0]),
            "fences_held": self._fences_held(scores),
        }
        return self._observation(self._obs), self._reward(scores), False, truncated, info

    def close(self):
        self._env.close()


def load_spec(path) -> dict:
    return json.loads(Path(path).read_text())
