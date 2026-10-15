"""Gripper motions shared by the project's ManiSkill environments.

Every environment drives a Panda arm with the `pd_ee_delta_pos` controller: an
action is a small displacement of the tool centre point (x, y, z) plus a gripper
command, each in [-1, 1]. Nothing here knows about any task; each task builds
its own primitives (pick, push, ...) from these helpers.

The two stepping helpers come from `scripts/planrob_inverse_rl_pushcube_demo.py`,
which is not an importable package, with two later fixes described in
`step_toward`.
"""

from __future__ import annotations

import numpy as np
import torch

_UNIT_TRAVEL = 0.04        # a unit pd_ee_delta_pos action moves the TCP about this far in one step
_STALL_TRAVEL = 0.0003     # a step moving less than this counts as no progress
_STALL_STEPS = 3           # consecutive no-progress steps that end a motion


def tcp_pos(obs) -> np.ndarray:
    return obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy().astype(np.float64)


def step_toward(env, obs, target_xyz, n_steps: int, tol: float, gripper_cmd: float,
                scale: float = 1.0):
    """Move the EEF toward target with optional action scaling; early-exits at tol.

    Unlike the PushCube helper this one is proportional inside the last
    `_UNIT_TRAVEL`: a full-magnitude command near the target overshoots a
    tight tolerance and the arm twitches back and forth until it lands.

    It also stops once the TCP has stopped making progress, which happens
    either because an object blocks it (a push whose object is already seated
    against a wall) or because the joint controller has settled at its
    steady-state error. Without this the loop keeps commanding a target it
    cannot reach and the arm visibly strains in place.
    """
    target_xyz = np.asarray(target_xyz, dtype=np.float64)
    stalled = 0
    for _ in range(n_steps):
        tcp = tcp_pos(obs)
        delta = (target_xyz - tcp).astype(np.float64)
        norm = float(np.linalg.norm(delta))
        if norm < tol:
            break
        action = np.zeros(4, dtype=np.float32)
        action[:3] = (np.clip(delta / _UNIT_TRAVEL, -1.0, 1.0) * scale).astype(np.float32)
        action[3] = gripper_cmd
        obs, *_ = env.step(torch.tensor(action))
        stalled = stalled + 1 if np.linalg.norm(tcp_pos(obs) - tcp) < _STALL_TRAVEL else 0
        if stalled >= _STALL_STEPS:
            break
    return obs


def step_in_place(env, obs, n_steps: int, gripper_cmd: float):
    """No translation, just send a gripper command for n steps (open/close)."""
    for _ in range(n_steps):
        a = torch.tensor(np.array([0.0, 0.0, 0.0, gripper_cmd], dtype=np.float32))
        obs, *_ = env.step(a)
    return obs


def reset(env, seed: int, options: dict | None = None):
    # A plain reset leaves simulator state behind (contact-solver caches) and
    # the same seed then plays out differently depending on the previous
    # episode. Reconfiguring costs ~80 ms and makes every episode a pure
    # function of its seed. `options` passes environment-specific settings.
    obs, _ = env.reset(seed=seed, options={"reconfigure": True, **(options or {})})
    return step_in_place(env, obs, 2, gripper_cmd=1.0)  # clear stale grasp flag


def lift(env, obs, height: float = 0.10, gripper_cmd: float = -1.0):
    up = tcp_pos(obs)
    up[2] += height
    return step_toward(env, obs, up, 20, 0.012, gripper_cmd, 1.0)


def retreat(env, obs, gripper_cmd: float, safe_z: float):
    """Rise vertically to the travel height `safe_z` if below it, without moving sideways.

    After a release the open fingers still straddle the object, and a diagonal
    departure would clip it. Each task chooses its own travel height.
    """
    tcp = tcp_pos(obs)
    if tcp[2] >= safe_z:
        return obs
    return step_toward(env, obs, [tcp[0], tcp[1], safe_z], 20, 0.012, gripper_cmd, 1.0)
