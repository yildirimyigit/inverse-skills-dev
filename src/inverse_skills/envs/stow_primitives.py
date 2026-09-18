"""Scripted primitives on StowCube-v1 and the observation -> SceneGraph adapter.

The scripted motions follow the PushCube pipeline scripts step for step
(`scripts/planrob_pushcube_symbolic_pipeline.py`, `_execute_pick`,
`_execute_place`, `_forward_demo`). The two stepping helpers are copied from
`scripts/planrob_inverse_rl_pushcube_demo.py` because the scripts are not an
importable package.

Every primitive takes and returns the ManiSkill observation dict so calls
chain: ``obs = pick(env, obs, "cube_a")``.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from inverse_skills.core import ObjectState, Pose, Region, RobotState, SceneGraph
from inverse_skills.envs import stow_cube as geo

CUBE_NAMES = ("cube_a", "cube_b")
_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

_UNIT_TRAVEL = 0.04        # a unit pd_ee_delta_pos action moves the TCP about this far in one step
_STALL_TRAVEL = 0.0003     # a step moving less than this counts as no progress
_STALL_STEPS = 3           # consecutive no-progress steps that end a motion
_SAFE_Z = 0.12             # travel height: clear of the 6 cm walls and of a held cube
_PUSH_SCALE = 0.2          # action scaling during pushes, as in the PushCube scripts
_PUSH_STANDOFF = 0.02      # TCP target sits 1 cm past where the cube seats: the wall (or A) stops
                           # the cube, so the push ends firmly seated rather than at the loop's tolerance
_FINGERTIP_OFFSET = 0.01   # the TCP frame sits this far above the closed fingertips' contact surface
_HOVER_CLEARANCE = 0.01    # approach() leaves this gap between fingertips and the cube's top face
_DESCENT_SCALE = 0.25      # final descent onto a cube: a full-scale step (~4 cm) overshoots into it
_PRESS_DEPTH = 0.02        # drag_out() first commands the fingertips this far below the top face
_DRAG_SCALE = 0.2          # per-step stroke command while dragging
_PRESS_CMD = 0.05          # per-step press command while dragging: lighter and the fingers skate
                           # over the cube, heavier (0.25) and the press stalls the arm's stroke
_REGION_HALF_XY = 0.02     # slot half-width = the 2 cm restoration tolerance
_REGION_HALF_Z = 0.05
_HOLDING_MAX_DIST = 0.03   # a closed gripper this close to a cube counts as holding it
_HOLDING_MAX_WIDTH = 0.04


def make_env(max_episode_steps: int = 3000, render_mode: str | None = None):
    return gym.make(
        "StowCube-v1", obs_mode="state_dict", control_mode="pd_ee_delta_pos",
        render_mode=render_mode, max_episode_steps=max_episode_steps,
    )


# ── stepping helpers (copied from the PushCube demo script) ─────────────────


def step_toward(env, obs, target_xyz, n_steps: int, tol: float, gripper_cmd: float,
                scale: float = 1.0):
    """Move the EEF toward target with optional action scaling; early-exits at tol.

    Unlike the PushCube helper this one is proportional inside the last
    `_UNIT_TRAVEL`: a full-magnitude command near the target overshoots a
    tight tolerance and the arm twitches back and forth until it lands.

    It also stops once the TCP has stopped making progress, which happens
    either because an object blocks it (a push whose cube is already seated
    against the wall) or because the joint controller has settled at its
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


# ── observation accessors ───────────────────────────────────────────────────


def tcp_pos(obs) -> np.ndarray:
    return obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy().astype(np.float64)


def cube_pos(obs, name: str) -> np.ndarray:
    return obs["extra"][f"{name}_pose"].squeeze()[:3].cpu().numpy().astype(np.float64)


def src_pos(obs, name: str) -> np.ndarray:
    key = "src_a" if name == "cube_a" else "src_b"
    return obs["extra"][key].squeeze().cpu().numpy().astype(np.float64)


def reset(env, seed: int):
    # A plain reset leaves simulator state behind (contact-solver caches) and
    # the same seed then stows differently depending on the previous episode.
    # Reconfiguring costs ~80 ms and makes every episode a pure function of its seed.
    obs, _ = env.reset(seed=seed, options={"reconfigure": True})
    return step_in_place(env, obs, 2, gripper_cmd=1.0)  # clear stale grasp flag


# ── scene graph ─────────────────────────────────────────────────────────────


def _box(name: str, xy, half_xy: float) -> Region:
    return Region.from_bounds(
        name,
        lower=[xy[0] - half_xy, xy[1] - half_xy, geo.CUBE_HALF - _REGION_HALF_Z],
        upper=[xy[0] + half_xy, xy[1] + half_xy, geo.CUBE_HALF + _REGION_HALF_Z],
    )


def regions(obs) -> dict[str, Region]:
    """Slots the predicates refer to: the two sampled sources, the pocket seat and the mouth."""
    return {
        "src_a": _box("src_a", src_pos(obs, "cube_a"), _REGION_HALF_XY),
        "src_b": _box("src_b", src_pos(obs, "cube_b"), _REGION_HALF_XY),
        "pocket": _box("pocket", geo.POCKET_A_XY, _REGION_HALF_XY),
        "mouth": _box("mouth", geo.MOUTH_B_XY, _REGION_HALF_XY),
    }


def obs_to_scene(obs, regions: dict[str, Region], timestep: int = 0) -> SceneGraph:
    qpos = obs["agent"]["qpos"].squeeze().cpu().numpy()
    gw = float(qpos[-2] + qpos[-1])
    tcp = tcp_pos(obs).astype(np.float32)
    objects = {
        name: ObjectState(
            name=name, semantic_class="cube",
            pose=Pose(position=cube_pos(obs, name).astype(np.float32), quat_xyzw=_IDENTITY_QUAT),
        )
        for name in CUBE_NAMES
    }
    holding = None
    if gw < _HOLDING_MAX_WIDTH:
        nearest = min(CUBE_NAMES, key=lambda n: np.linalg.norm(cube_pos(obs, n) - tcp))
        if np.linalg.norm(cube_pos(obs, nearest) - tcp) < _HOLDING_MAX_DIST:
            holding = nearest
    return SceneGraph(
        timestep=timestep,
        robot=RobotState(q=qpos[:7].astype(np.float32), gripper_width=gw,
                         ee_pose=Pose(position=tcp, quat_xyzw=_IDENTITY_QUAT), holding=holding),
        objects=objects,
        regions=regions,
    )


# ── forward skill ───────────────────────────────────────────────────────────


def _push_along_x(env, obs, name: str, target_center_x: float, n_steps: int):
    """Closed-gripper push of `name` along +x until its centre would reach target_center_x."""
    start = cube_pos(obs, name)
    z = geo.CUBE_HALF + 0.005
    obs = step_toward(env, obs, [start[0] - 0.06, start[1], z + 0.10], 30, 0.012, -1.0, 1.0)
    obs = step_toward(env, obs, [start[0] - 0.06, start[1], z], 30, 0.012, -1.0, 1.0)
    obs = step_toward(env, obs, [target_center_x - _PUSH_STANDOFF, start[1], z],
                      n_steps, 0.005, -1.0, _PUSH_SCALE)
    up = tcp_pos(obs)
    up[2] += 0.10
    return step_toward(env, obs, up, 20, 0.012, -1.0, 1.0)


def forward_stow(env, obs):
    """The demonstrated skill: push A into the pocket, then B into the mouth, lift, open."""
    obs = _push_along_x(env, obs, "cube_a", geo.POCKET_A_XY[0], n_steps=40)
    obs = _push_along_x(env, obs, "cube_b", geo.MOUTH_B_XY[0], n_steps=50)
    return step_in_place(env, obs, 8, gripper_cmd=1.0)


# ── library primitives ──────────────────────────────────────────────────────


def retreat(env, obs, gripper_cmd: float):
    """Rise vertically to the travel height if below it, without moving sideways.

    After a place() the open fingers still straddle the released cube, and a
    diagonal departure would clip it.
    """
    tcp = tcp_pos(obs)
    if tcp[2] >= _SAFE_Z:
        return obs
    return step_toward(env, obs, [tcp[0], tcp[1], _SAFE_Z], 20, 0.012, gripper_cmd, 1.0)


def pick(env, obs, name: str):
    obs = retreat(env, obs, gripper_cmd=1.0)
    c = cube_pos(obs, name)
    obs = step_toward(env, obs, [c[0], c[1], c[2] + 0.10], 20, 0.012, 1.0, 1.0)
    # Half-scale descent: at full scale the arm overshoots ~2 cm in x on the
    # way down, enough for a finger to catch the pocket's arm tip beside B.
    obs = step_toward(env, obs, c, 40, 0.008, 1.0, 0.5)
    obs = step_in_place(env, obs, 10, gripper_cmd=-1.0)
    return step_in_place(env, obs, 5, gripper_cmd=-1.0)


def place(env, obs, name: str, target_xyz):
    target = np.asarray(target_xyz, dtype=np.float64)
    c = cube_pos(obs, name)
    lift = [c[0], c[1], c[2] + 0.15]
    obs = step_toward(env, obs, lift, 25, 0.02, -1.0, 0.5)
    obs = step_toward(env, obs, [target[0], target[1], lift[2]], 80, 0.012, -1.0, 0.5)
    obs = step_toward(env, obs, [target[0], target[1], target[2] + 0.005], 30, 0.012, -1.0, 0.5)
    obs = step_in_place(env, obs, 10, gripper_cmd=1.0)
    return step_in_place(env, obs, 10, gripper_cmd=1.0)


def approach(env, obs, name: str):
    """Closed gripper brought to just above the cube's top face: the hole's start pose."""
    obs = retreat(env, obs, gripper_cmd=1.0)  # clear the last released cube before closing
    c = cube_pos(obs, name)
    obs = step_toward(env, obs, [c[0], c[1], c[2] + 0.10], 30, 0.012, -1.0, 1.0)
    hover_z = c[2] + geo.CUBE_HALF + _FINGERTIP_OFFSET + _HOVER_CLEARANCE
    return step_toward(env, obs, [c[0], c[1], hover_z], 40, 0.003, -1.0, _DESCENT_SCALE)


def lift(env, obs, height: float = 0.10, gripper_cmd: float = -1.0):
    up = tcp_pos(obs)
    up[2] += height
    return step_toward(env, obs, up, 20, 0.012, gripper_cmd, 1.0)


# ── physics check only (stands in for the learned hole in Stage 0) ──────────


def drag_out(env, obs, name: str, target_x: float):
    """Press the closed fingertips onto the cube's top and drag it along -x by friction.

    Closed-loop on the cube, not the TCP: the cube slips behind the fingertips.
    The delta-position controller is relative to the current pose, so the
    press has to be commanded on every step or the normal force, and with it
    the friction, disappears.
    """
    c = cube_pos(obs, name)
    press_z = c[2] + geo.CUBE_HALF + _FINGERTIP_OFFSET - _PRESS_DEPTH
    obs = step_toward(env, obs, [c[0], c[1], press_z], 10, 0.002, -1.0, _DESCENT_SCALE)
    for _ in range(120):
        if cube_pos(obs, name)[0] <= target_x:
            break
        action = np.array([-_DRAG_SCALE, 0.0, -_PRESS_CMD, -1.0], dtype=np.float32)
        obs, *_ = env.step(torch.tensor(action))
    return lift(env, obs)
