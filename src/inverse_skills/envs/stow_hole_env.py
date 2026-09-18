"""The hole as a gym environment: learn `clear_of_walls(cube_a)`, nothing else.

`reset()` runs everything the plan puts before the hole — the forward skill,
then `pick(B)`, `place(B,src_b)`, `approach(A)` — and hands over at exactly the
state the executive will hand over at. The policy is then responsible for one
predicate.

Deliberately small, because the RL is the part most likely to fail:

  * 2D action (stroke along -x, press along -z). The gripper stays closed and
    y is fixed: the Stage 0 sweep showed the whole skill is the balance
    between stroke and press, and the drag's lateral drift never mattered.
  * 4 observations, all predicate-grounded: the active predicate's margin over
    its temperature, and the TCP's offset from the cube over the same scale.
  * The reward is the framework's, with no additions — the active term is
    2*V-1 for `clear_of_walls`, the fences are one-sided penalties on what the
    prefix already established. The only number chosen here is the predicate's
    temperature, which Stage 1 already fixed.

A start-state curriculum widens the handoff from "fingertips on the cube" to
the full hover, so the first episodes see reward gradient immediately.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from inverse_skills.envs import stow_cube as geo
from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs import stow_primitives as prim

MAX_STEPS = 40
POSTCONDITION_THRESHOLD = 0.8      # the executive's "done" test, = the fence threshold
FENCE_THRESHOLD = 0.8

_ACTION_STROKE = 0.3               # max per-step stroke command, from the Stage 0 sweep
_ACTION_PRESS = 0.15               # max per-step press command
_START_PRESS_DEPTH = 0.02          # curriculum start: fingertips already on the cube
_START_HOVER = geo.CUBE_HALF + prim._FINGERTIP_OFFSET + prim._HOVER_CLEARANCE
_PREFIX_TOLERANCE = 0.02           # B must land within this of src_b for a valid handoff
_HANDOFF_TCP_NEAR = 0.6            # the approach landed on the cube (measured at the start state)
_MAX_PREFIX_ATTEMPTS = 5

# The fence: what the prefix established and the hole must not break. `tcp_near`
# is the hole's precondition, not a fence — the fingers press on the cube all
# episode, so its score sits near the 0.8 threshold and would add noise to the
# reward for a predicate the policy cannot usefully trade against. The executive
# re-checks it before a retry instead.
_FENCES = ("in_region(cube_b,src_b)",)
_ACTIVE = "clear_of_walls(cube_a)"


class StowHoleEnv(gym.Env):
    """One hole, one predicate, one contact skill."""

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = MAX_STEPS, curriculum: float = 0.0,
                 seed_pool: tuple[int, ...] = tuple(range(32)),
                 render_mode: str | None = None):
        super().__init__()
        self._env = prim.make_env(render_mode=render_mode)
        self.registry = sp.registry()
        self.max_steps = max_steps
        self.curriculum = float(curriculum)     # 0 = fingertips on the cube, 1 = full hover
        # Episodes draw from a fixed pool so the cached handoffs stay bounded.
        self.seed_pool = tuple(seed_pool)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4,),
                                            dtype=np.float32)
        self._obs = None
        self._regions = None
        self._steps = 0
        # The prefix costs ~0.35 s of simulation and is deterministic per seed,
        # so each seed's handoff is simulated once and restored thereafter.
        self._handoffs: dict[int, tuple] = {}

    # -- prefix ---------------------------------------------------------------

    def set_curriculum(self, value: float) -> None:
        self.curriculum = float(np.clip(value, 0.0, 1.0))

    def _run_prefix(self, seed: int):
        """Everything the plan puts before the hole, with the handoff checked."""
        obs = prim.reset(self._env, seed)
        self._regions = prim.regions(obs)
        obs = prim.forward_stow(self._env, obs)
        obs = prim.pick(self._env, obs, "cube_b")
        obs = prim.place(self._env, obs, "cube_b", prim.src_pos(obs, "cube_b"))
        b_error = float(np.linalg.norm(
            prim.cube_pos(obs, "cube_b")[:2] - prim.src_pos(obs, "cube_b")[:2]))
        obs = prim.approach(self._env, obs, "cube_a")
        return obs, b_error

    def _descend_to_start(self, obs):
        """Place the fingertips where the curriculum says the episode begins."""
        cube = prim.cube_pos(obs, "cube_a")
        contact_z = cube[2] + geo.CUBE_HALF + prim._FINGERTIP_OFFSET
        z = contact_z - _START_PRESS_DEPTH + self.curriculum * (
            _START_HOVER + cube[2] - contact_z + _START_PRESS_DEPTH)
        return prim.step_toward(self._env, obs, [cube[0], cube[1], z], 20, 0.002, -1.0, 0.25)

    # -- scoring --------------------------------------------------------------

    def _scores(self, obs) -> dict[str, float]:
        scene = prim.obs_to_scene(obs, self._regions)
        return {k: float(r.score) for k, r in self.registry.evaluate_all(scene).items()}

    def _observation(self, obs, scores) -> np.ndarray:
        cube = prim.cube_pos(obs, "cube_a")
        tcp = prim.tcp_pos(obs)
        margin = geo.X_CLEAR - float(cube[0])
        return np.array([
            margin / sp._CLEAR_TEMP,
            (tcp[0] - cube[0]) / geo.CUBE_HALF,
            (tcp[1] - cube[1]) / geo.CUBE_HALF,
            (tcp[2] - cube[2]) / geo.CUBE_HALF,
        ], dtype=np.float32)

    def _reward(self, scores) -> float:
        """Active term plus one-sided fences — the framework's signed scores."""
        reward = 2.0 * scores[_ACTIVE] - 1.0
        for key in _FENCES:
            reward += min(0.0, 2.0 * scores[key] - 1.0)
        return float(reward)

    # -- gym API --------------------------------------------------------------

    def _handoff(self, episode_seed: int) -> tuple:
        """The cached post-prefix simulator state for this seed, computed once."""
        if episode_seed not in self._handoffs:
            attempts = 0
            for attempt in range(_MAX_PREFIX_ATTEMPTS):
                attempts = attempt + 1
                obs, b_error = self._run_prefix(episode_seed + attempt * 10_000)
                if b_error <= _PREFIX_TOLERANCE:
                    break
            state = self._env.unwrapped.get_state_dict()
            self._handoffs[episode_seed] = (state, self._regions, b_error, attempts)
        return self._handoffs[episode_seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            episode_seed = int(seed)
        else:
            episode_seed = int(self.np_random.choice(self.seed_pool))
        state, regions, b_error, attempts = self._handoff(episode_seed)
        self._env.unwrapped.set_state_dict(state)
        self._regions = regions
        obs = self._descend_to_start(self._env.unwrapped.get_obs())
        self._obs = obs
        self._steps = 0
        scores = self._scores(obs)
        info = {"prefix_attempts": attempts, "b_error": b_error,
                "curriculum": self.curriculum, "scores": scores,
                "handoff_valid": (b_error <= _PREFIX_TOLERANCE
                                  and scores[_ACTIVE] < POSTCONDITION_THRESHOLD
                                  and scores["tcp_near(cube_a)"] >= _HANDOFF_TCP_NEAR)}
        return self._observation(obs, scores), info

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        command = np.array([a[0] * _ACTION_STROKE, 0.0, a[1] * _ACTION_PRESS, -1.0],
                           dtype=np.float32)
        self._obs, *_ = self._env.step(torch.tensor(command))
        self._steps += 1

        scores = self._scores(self._obs)
        reward = self._reward(scores)
        satisfied = scores[_ACTIVE] >= POSTCONDITION_THRESHOLD
        truncated = self._steps >= self.max_steps and not satisfied
        info = {
            "scores": scores,
            "postcondition": satisfied,
            "cube_a_x": float(prim.cube_pos(self._obs, "cube_a")[0]),
            "fences_held": all(scores[k] >= FENCE_THRESHOLD for k in _FENCES),
        }
        return self._observation(self._obs, scores), reward, satisfied, truncated, info

    def close(self):
        self._env.close()
