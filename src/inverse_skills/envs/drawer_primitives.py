"""Scripted primitives on SlidingDrawer-v1 and the observation -> SceneGraph adapter.

The robot's known actions:

  push_close       the forward skill: push the open drawer shut with the closed
                   gripper, beside the handle
  approach_handle  bring the open gripper to a standoff above the handle
  pull_open        pull the drawer open. It never opens or closes the gripper on
                   the way, so it only moves the drawer if the handle is already
                   held. It lets go at the end.

For Stage 0 only, never part of the planner's library:

  test_grasp       descend from the standoff and close on the fin. It exists to
                   check the physics and calibrate handle_grasped; a learned skill
                   will take its place.

The gripper motions shared with the stow scene come from `motion.py`. Every
primitive takes and returns the ManiSkill observation dict.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from inverse_skills.core import ObjectState, Pose, RobotState, SceneGraph
from inverse_skills.envs import drawer_env as geo
from inverse_skills.envs import motion
from inverse_skills.envs.motion import step_in_place, step_toward, tcp_pos

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

HOME_TCP = (-0.10, 0.0, 0.25)  # where scripted episodes start and end: clear of everything
TRAVEL_Z = 0.22                # travel height: the hand clears the 12 cm cabinet
OPEN_TARGET = 0.15             # pull_open pulls the drawer to this opening
# push_close pushes beside the handle, at a height where the hand stays above the cabinet top.
_PUSH_Y = 0.07
_PUSH_TCP_Z = 0.085
_FINGER_HALF_DEPTH = 0.01      # the closed fingers reach about this far ahead of the TCP along x
_SHUT = 0.003                  # push_close stops once the drawer is this close to shut
_HELD_BELOW = 0.05             # a gripper narrower than this is treated as closed


def make_env(max_episode_steps: int = 3000, render_mode: str | None = None):
    return gym.make(
        "SlidingDrawer-v1", obs_mode="state_dict", control_mode="pd_ee_delta_pos",
        render_mode=render_mode, max_episode_steps=max_episode_steps,
    )


def reset(env, seed: int, opening: float | None = None):
    """A fresh episode. `opening` fixes how far open the drawer starts; by
    default it is drawn from drawer_env.OPENING_RANGE."""
    return motion.reset(env, seed, options={"opening": opening})


# ── observation accessors ───────────────────────────────────────────────────


def opening(obs) -> float:
    return float(obs["extra"]["drawer_opening"].squeeze())


def grasp_point(obs) -> np.ndarray:
    return obs["extra"]["grasp_point"].squeeze().cpu().numpy().astype(np.float64)


def pinch_force(obs) -> tuple[float, float]:
    left, right = obs["extra"]["pinch_force"].squeeze().cpu().numpy().tolist()
    return left, right


def gripper_width(obs) -> float:
    qpos = obs["agent"]["qpos"].squeeze().cpu().numpy()
    return float(qpos[-2] + qpos[-1])


def tilt_deg(obs) -> float:
    """Angle between the fingers' pointing direction and straight down."""
    _, x, y, _ = obs["extra"]["tcp_pose"].squeeze()[3:7].cpu().numpy()   # quaternion w, x, y, z
    cos_tilt = 2 * (x * x + y * y) - 1   # from the TCP's z-axis, which points along the fingers
    return round(float(np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))), 1)


def obs_to_scene(obs, timestep: int = 0) -> SceneGraph:
    qpos = obs["agent"]["qpos"].squeeze().cpu().numpy()
    tcp = tcp_pos(obs).astype(np.float32)
    point = grasp_point(obs)
    # The drawer's frame is the middle of its front, at table height.
    drawer_origin = point - np.array(geo.GRASP_POINT)
    return SceneGraph(
        timestep=timestep,
        robot=RobotState(q=qpos[:7].astype(np.float32), gripper_width=gripper_width(obs),
                         ee_pose=Pose(position=tcp, quat_xyzw=_IDENTITY_QUAT),
                         metadata={"pinch_force": pinch_force(obs)}),
        objects={
            "drawer": ObjectState(name="drawer", semantic_class="drawer",
                                  pose=Pose(position=drawer_origin.astype(np.float32),
                                            quat_xyzw=_IDENTITY_QUAT),
                                  metadata={"opening": opening(obs)}),
            "handle": ObjectState(name="handle", semantic_class="handle",
                                  pose=Pose(position=point.astype(np.float32),
                                            quat_xyzw=_IDENTITY_QUAT)),
        },
    )


# ── motions ─────────────────────────────────────────────────────────────────


def go_home(env, obs):
    """Rise to the travel height, then move to HOME_TCP with the gripper open."""
    obs = motion.retreat(env, obs, gripper_cmd=1.0, safe_z=TRAVEL_Z)
    return step_toward(env, obs, HOME_TCP, 60, 0.01, 1.0, 1.0)


def push_close(env, obs):
    """The forward skill: push the drawer shut with the closed gripper, beside the handle."""
    point = grasp_point(obs)
    front_x = point[0] - geo.GRASP_POINT[0]           # the drawer front's x right now
    closed_front_x = front_x + opening(obs)           # where the front is once closed
    y = point[1] + _PUSH_Y
    obs = motion.retreat(env, obs, gripper_cmd=-1.0, safe_z=TRAVEL_Z)
    obs = step_toward(env, obs, [front_x - 0.06, y, TRAVEL_Z], 60, 0.01, -1.0, 1.0)
    obs = step_toward(env, obs, [front_x - 0.06, y, _PUSH_TCP_Z], 40, 0.005, -1.0, 0.5)
    # Push one step at a time and stop as soon as the drawer is shut. Pushing on
    # against the closed cabinet only tips the hand over, and the position-only
    # controller never straightens it again: the hand would stay tilted for the
    # rest of the episode, approach and grasp included.
    target_x = closed_front_x - _FINGER_HALF_DEPTH + 0.01
    for _ in range(120):
        if opening(obs) <= _SHUT:
            break
        obs = step_toward(env, obs, [target_x, y, _PUSH_TCP_Z], 1, 0.003, -1.0, 0.3)
    tcp = tcp_pos(obs)
    obs = step_toward(env, obs, [tcp[0] - 0.04, tcp[1], tcp[2]], 20, 0.005, -1.0, 0.5)
    return go_home(env, obs)


def approach_handle(env, obs):
    """Bring the open gripper to the standoff above the handle's grasp point."""
    obs = motion.retreat(env, obs, gripper_cmd=1.0, safe_z=TRAVEL_Z)
    point = grasp_point(obs)
    obs = step_toward(env, obs, [point[0], point[1], TRAVEL_Z], 60, 0.01, 1.0, 1.0)
    standoff = point + np.array([0.0, 0.0, geo.STANDOFF_HEIGHT])
    return step_toward(env, obs, standoff, 40, 0.003, 1.0, 0.5)


def pull_open(env, obs, target_opening: float = OPEN_TARGET):
    """Pull straight toward the robot until the drawer would be `target_opening`
    open, then let go and rise.

    The gripper is held exactly as it is: closed if it is closed, open if it is
    open. Pulling therefore never creates a grasp; the drawer moves only if the
    fingers already hold the handle.
    """
    hold = -1.0 if gripper_width(obs) < _HELD_BELOW else 1.0
    tcp = tcp_pos(obs)
    distance = target_opening - opening(obs)
    obs = step_toward(env, obs, [tcp[0] - distance, tcp[1], tcp[2]], 120, 0.003, hold, 0.4)
    obs = step_in_place(env, obs, 10, gripper_cmd=1.0)
    return motion.retreat(env, obs, gripper_cmd=1.0, safe_z=TRAVEL_Z)


def test_grasp(env, obs):
    """Stage 0 only: descend from the standoff with the gripper open, then close on the fin."""
    obs = step_toward(env, obs, grasp_point(obs), 40, 0.003, 1.0, 0.5)
    return step_in_place(env, obs, 15, gripper_cmd=-1.0)
