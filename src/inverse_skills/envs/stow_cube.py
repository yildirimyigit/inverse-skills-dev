"""StowCube-v1: PushCube's table with a three-walled pocket and two cubes.

Fixed geometry, chosen so that every library primitive fails on the stowed
cube A (see the "Wedged Cube Scene" design page):

  * the pocket is open toward the robot (-x) and its walls are taller than
    the cube, so both top-down grasp directions on A collide with a wall;
  * A's +x and ±y faces are against walls, so no push can move it;
  * B, pushed into the pocket's mouth, blocks A's only exit but is itself
    graspable because the side arms stop short of it.

The forward skill "stow A" pushes A into the pocket and B into the mouth.
All lengths are metres in the table frame (z = 0 is the table surface).
"""

from __future__ import annotations

import numpy as np
import sapien
import torch

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose

CUBE_HALF = 0.02
WALL_HEIGHT = 0.06
WALL_THICKNESS = 0.01
POCKET_BACK_X = 0.15          # inner face of the back wall
POCKET_HALF_WIDTH = 0.025     # inner half-width: 5 cm for the 4 cm cube
ARM_LENGTH = 0.035            # side arms run from x = 0.115 to the back wall
POCKET_A_XY = (POCKET_BACK_X - CUBE_HALF, 0.0)       # A seated against the back wall
MOUTH_B_XY = (POCKET_A_XY[0] - 2 * CUBE_HALF, 0.0)   # B touching A's -x face
SRC_A_XY = (0.0, 0.0)
SRC_B_XY = (-0.10, 0.0)
SRC_NOISE = 0.005             # +-5 mm xy noise on both source poses
X_CLEAR = 0.095               # clear_of_walls(A) holds when x_A <= X_CLEAR (6 cm from the arm tips)
SLOT_TOLERANCE = 0.01         # "in pocket" / "in mouth" radius used by evaluate()


@register_env("StowCube-v1", max_episode_steps=200)
class StowCubeEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        # Front-left of the pocket, so the camera looks into its open mouth.
        pose = sapien_utils.look_at([-0.17, 0.32, 0.26], [0.04, 0.0, 0.02])
        return CameraConfig(
            "render_camera", pose=pose, width=512, height=512, fov=1, near=0.01, far=100
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.cube_a = actors.build_cube(
            self.scene, half_size=CUBE_HALF, color=np.array([12, 42, 160, 255]) / 255,
            name="cube_a", body_type="dynamic",
            initial_pose=sapien.Pose(p=[SRC_A_XY[0], SRC_A_XY[1], CUBE_HALF]),
        )
        self.cube_b = actors.build_cube(
            self.scene, half_size=CUBE_HALF, color=np.array([40, 140, 80, 255]) / 255,
            name="cube_b", body_type="dynamic",
            initial_pose=sapien.Pose(p=[SRC_B_XY[0], SRC_B_XY[1], CUBE_HALF]),
        )

        wall_color = np.array([120, 116, 108, 255]) / 255
        self.back_wall = actors.build_box(
            self.scene,
            half_sizes=[WALL_THICKNESS / 2, POCKET_HALF_WIDTH + WALL_THICKNESS, WALL_HEIGHT / 2],
            color=wall_color, name="back_wall", body_type="static",
            initial_pose=sapien.Pose(p=[POCKET_BACK_X + WALL_THICKNESS / 2, 0.0, WALL_HEIGHT / 2]),
        )
        # Each arm spans from the arm tip to the back wall's outer face.
        arm_half_x = (ARM_LENGTH + WALL_THICKNESS) / 2
        arm_x = POCKET_BACK_X + WALL_THICKNESS - arm_half_x
        arm_y = POCKET_HALF_WIDTH + WALL_THICKNESS / 2
        self.arms = [
            actors.build_box(
                self.scene, half_sizes=[arm_half_x, WALL_THICKNESS / 2, WALL_HEIGHT / 2],
                color=wall_color, name=name, body_type="static",
                initial_pose=sapien.Pose(p=[arm_x, sign * arm_y, WALL_HEIGHT / 2]),
            )
            for name, sign in (("arm_left", 1.0), ("arm_right", -1.0))
        ]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            sources = []
            for cube, src in ((self.cube_a, SRC_A_XY), (self.cube_b, SRC_B_XY)):
                xyz = torch.zeros((b, 3))
                xyz[..., 0] = src[0]
                xyz[..., 1] = src[1]
                xyz[..., :2] += (torch.rand((b, 2)) * 2 - 1) * SRC_NOISE
                xyz[..., 2] = CUBE_HALF
                cube.set_pose(Pose.create_from_pq(p=xyz, q=[1, 0, 0, 0]))
                sources.append(xyz)
            self.src_a, self.src_b = sources

    def evaluate(self):
        a_xy = self.cube_a.pose.p[..., :2]
        b_xy = self.cube_b.pose.p[..., :2]
        pocket = torch.tensor(POCKET_A_XY, device=self.device)
        mouth = torch.tensor(MOUTH_B_XY, device=self.device)
        a_in_pocket = torch.linalg.norm(a_xy - pocket, dim=1) < SLOT_TOLERANCE
        b_in_mouth = torch.linalg.norm(b_xy - mouth, dim=1) < SLOT_TOLERANCE
        return {
            "a_in_pocket": a_in_pocket,
            "b_in_mouth": b_in_mouth,
            "success": a_in_pocket & b_in_mouth,
        }

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            obs.update(
                cube_a_pose=self.cube_a.pose.raw_pose,
                cube_b_pose=self.cube_b.pose.raw_pose,
                src_a=self.src_a,
                src_b=self.src_b,
            )
        return obs

    # The scene carries no task reward of its own: the inverse-skill wrappers
    # build their rewards from predicate scores.
    def compute_dense_reward(self, obs, action, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info: dict):
        return self.compute_dense_reward(obs, action, info)
