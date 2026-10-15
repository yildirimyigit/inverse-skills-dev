"""SlidingDrawer-v1: a cabinet with one sliding drawer and a vertical fin handle.

The drawer slides along x and opens toward the robot (-x). Its handle is a thin
vertical fin sticking out of the drawer front. The fin is thin along y, so the
Panda can pinch it from above with its fingers closing along y, under the same
4-D control (pd_ee_delta_pos) as the stow scene.

The cabinet is the articulation's fixed root, the drawer is a prismatic link,
and the handle is a third link fixed to the drawer. Keeping the handle separate
lets contact with the handle be told apart from contact with the drawer front.

Nothing moves the drawer except contact: there is no joint drive.

All lengths are metres in the table frame (z = 0 is the table surface).
"""

from __future__ import annotations

import sapien
import torch
from sapien import physx

from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose

# Cabinet, open toward the robot. Its local frame sits at the middle of the
# front face, at table height; a closed drawer is flush with that face.
CABINET_FRONT_X = 0.08
CABINET_DEPTH = 0.25           # inner depth along +x
CABINET_HALF_WIDTH = 0.12      # inner half-width along y
CABINET_HEIGHT = 0.10          # inner height
WALL = 0.01                    # wall thickness
CABINET_POS_NOISE = 0.005      # +-5 mm on the cabinet's xy position per episode

# Drawer
CLEARANCE = 0.003              # gap between drawer and cabinet on every side
FRONT_THICKNESS = 0.015
BOX_WALL = 0.008               # bottom, sides and back of the drawer box
BOX_HEIGHT = 0.06
MAX_OPENING = 0.20             # joint limit
OPENING_RANGE = (0.12, 0.18)   # how far open the drawer starts; the forward skill closes it
JOINT_FRICTION = 0.3           # physx joint friction coefficient
JOINT_DAMPING = 10.0           # N per (m/s)

# Handle: a vertical fin on the middle of the drawer front, thin along y.
FIN_LENGTH = 0.05              # how far it sticks out toward the robot (x)
FIN_THICKNESS = 0.012          # along y: the fingers pinch it from both sides
FIN_HEIGHT = 0.04              # along z
FIN_TOP_Z = WALL + CABINET_HEIGHT - CLEARANCE   # level with the top of the drawer front
FIN_FRICTION = 1.0

# Where a good pinch puts the TCP, in the drawer's frame: 3 cm out from the
# drawer front and 1 cm below the fin's top edge. The TCP sits about 1 cm above
# the fingertips, so the finger pads then cover the fin's top 2 cm.
GRASP_POINT = (-0.03, 0.0, FIN_TOP_Z - 0.01)
STANDOFF_HEIGHT = 0.05         # approach_handle waits this far above the grasp point


def _box(builder, center, half_size, color, material=None):
    builder.add_box_collision(pose=sapien.Pose(center), half_size=half_size, material=material)
    builder.add_box_visual(pose=sapien.Pose(center), half_size=half_size,
                           material=sapien.render.RenderMaterial(base_color=color))


@register_env("SlidingDrawer-v1", max_episode_steps=200)
class SlidingDrawerEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda"]
    agent: Panda

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_human_render_camera_configs(self):
        # Front-left of the cabinet, looking at the drawer front and the handle.
        pose = sapien_utils.look_at([-0.30, 0.42, 0.34], [0.02, 0.0, 0.07])
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

        builder = self.scene.create_articulation_builder()
        builder.disable_self_collisions = True   # cabinet, drawer and handle never touch each other
        builder.initial_pose = sapien.Pose(p=[CABINET_FRONT_X, 0, 0])

        # Cabinet: bottom, top, two sides and a back, open at local x = 0.
        cabinet = builder.create_link_builder()
        cabinet.set_name("cabinet")
        grey = [0.72, 0.70, 0.66, 1]
        depth = CABINET_DEPTH + WALL
        half_w = CABINET_HALF_WIDTH + WALL
        inner_mid_z = WALL + CABINET_HEIGHT / 2
        _box(cabinet, [depth / 2, 0, WALL / 2], [depth / 2, half_w, WALL / 2], grey)
        _box(cabinet, [depth / 2, 0, WALL + CABINET_HEIGHT + WALL / 2], [depth / 2, half_w, WALL / 2], grey)
        for side in (1, -1):
            _box(cabinet, [depth / 2, side * (CABINET_HALF_WIDTH + WALL / 2), inner_mid_z],
                 [depth / 2, WALL / 2, CABINET_HEIGHT / 2], grey)
        _box(cabinet, [CABINET_DEPTH + WALL / 2, 0, inner_mid_z],
             [WALL / 2, CABINET_HALF_WIDTH, CABINET_HEIGHT / 2], grey)

        # Drawer: a front panel flush with the opening, and a shallow box behind it.
        # The joint frame is turned 180 degrees about z, so a positive joint
        # position slides the drawer toward the robot (-x): it is the opening in metres.
        drawer = builder.create_link_builder(cabinet)
        drawer.set_name("drawer")
        drawer.set_joint_name("drawer_slide")
        turned = sapien.Pose(q=[0, 0, 0, 1])
        drawer.set_joint_properties("prismatic", limits=[[0.0, MAX_OPENING]],
                                    pose_in_parent=turned, pose_in_child=turned,
                                    damping=JOINT_DAMPING)
        wood = [0.55, 0.42, 0.30, 1]
        half_w = CABINET_HALF_WIDTH - CLEARANCE
        floor_z = WALL + CLEARANCE
        _box(drawer, [FRONT_THICKNESS / 2, 0, inner_mid_z],
             [FRONT_THICKNESS / 2, half_w, CABINET_HEIGHT / 2 - CLEARANCE], wood)
        box_back = CABINET_DEPTH - CLEARANCE
        box_mid_x = (FRONT_THICKNESS + box_back) / 2
        box_half_x = (box_back - FRONT_THICKNESS) / 2
        _box(drawer, [box_mid_x, 0, floor_z + BOX_WALL / 2], [box_half_x, half_w, BOX_WALL / 2], wood)
        for side in (1, -1):
            _box(drawer, [box_mid_x, side * (half_w - BOX_WALL / 2), floor_z + BOX_HEIGHT / 2],
                 [box_half_x, BOX_WALL / 2, BOX_HEIGHT / 2], wood)
        _box(drawer, [box_back - BOX_WALL / 2, 0, floor_z + BOX_HEIGHT / 2],
             [BOX_WALL / 2, half_w, BOX_HEIGHT / 2], wood)

        # Handle: the fin, on its own link fixed to the drawer.
        handle = builder.create_link_builder(drawer)
        handle.set_name("handle")
        handle.set_joint_name("handle_mount")
        handle.set_joint_properties("fixed", limits=[], pose_in_parent=sapien.Pose(),
                                    pose_in_child=sapien.Pose())
        _box(handle, [-FIN_LENGTH / 2, 0, FIN_TOP_Z - FIN_HEIGHT / 2],
             [FIN_LENGTH / 2, FIN_THICKNESS / 2, FIN_HEIGHT / 2], [0.15, 0.15, 0.17, 1],
             material=physx.PhysxMaterial(FIN_FRICTION, FIN_FRICTION, 0.0))

        self.cabinet = builder.build(name="cabinet", fix_root_link=True)
        # ManiSkill's builder applies the joint damping but not the friction.
        self.cabinet.joints_map["drawer_slide"].set_friction(JOINT_FRICTION)
        self.handle_link = self.cabinet.links_map["handle"]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = CABINET_FRONT_X
            xyz[:, :2] += (torch.rand((b, 2)) * 2 - 1) * CABINET_POS_NOISE
            self.cabinet.set_pose(Pose.create_from_pq(p=xyz))
            if options.get("opening") is not None:
                opening = torch.full((b, 1), float(options["opening"]))
            else:
                low, high = OPENING_RANGE
                opening = low + torch.rand((b, 1)) * (high - low)
            self.cabinet.set_qpos(opening)
            self.cabinet.set_qvel(torch.zeros((b, 1)))

    def pinch_forces(self) -> torch.Tensor:
        """Each finger's contact force on the handle, along that finger's closing
        direction (the convention of Panda.is_grasping). Shape (num_envs, 2)."""
        left = self.scene.get_pairwise_contact_forces(self.agent.finger1_link, self.handle_link)
        right = self.scene.get_pairwise_contact_forces(self.agent.finger2_link, self.handle_link)
        left_dir = self.agent.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        right_dir = -self.agent.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        return torch.stack([(left * left_dir).sum(-1), (right * right_dir).sum(-1)], dim=-1)

    def evaluate(self):
        return {"opening": self.cabinet.get_qpos()[:, 0]}

    def _get_obs_extra(self, info: dict):
        obs = dict(tcp_pose=self.agent.tcp.pose.raw_pose)
        if self.obs_mode_struct.use_state:
            grasp_point = self.handle_link.pose * Pose.create_from_pq(p=list(GRASP_POINT))
            obs.update(
                drawer_opening=self.cabinet.get_qpos(),
                grasp_point=grasp_point.p,
                pinch_force=self.pinch_forces(),
            )
        return obs

    # The scene carries no task reward of its own: the inverse-skill wrappers
    # build their rewards from predicate scores.
    def compute_dense_reward(self, obs, action, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs, action, info: dict):
        return self.compute_dense_reward(obs, action, info)
