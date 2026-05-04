"""PushCube-v1 inverse skill with symbolic place, then RL table-push correction.

This variant implements the architecture where the symbolic inverse fully
executes pick(cube) -> place(source):

  1. Forward skill: scripted closed-gripper push in +X.
  2. Symbolic inverse: pick cube at post-forward position, carry it back to the
     source region, descend to the table, open the gripper, settle, and retract.
  3. Push handoff: move the open gripper to a non-contact staging pose behind
     the released cube.
  4. RL correction: the cube is free on the table; the policy only controls XY
     pusher motion. The wrapper keeps the gripper open and servo-controls TCP
     height near the cube so the learned phase is table pushing, not grasping.
  5. Measure directly. There is no scripted post-RL release because release
     already happened inside the symbolic place.

Reward:
  V_at_pose(cube, init_pose) + V_near(tcp, cube) - 2.0 * distance(cube, init_pose)

Outputs:
  artifacts/planrob_inverse_rl_pushcube_push_correction_*.{json,png,zip}
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList

matplotlib.use("Agg")

warnings.filterwarnings("ignore")

import mani_skill.envs  # noqa: E402,F401  (registers PushCube)

from inverse_skills.predicates import GripperOpenPredicate  # noqa: E402


_spec = importlib.util.spec_from_file_location("demo", "scripts/planrob_inverse_rl_demo.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

_spec_pc = importlib.util.spec_from_file_location("pc", "scripts/planrob_inverse_rl_pushcube_demo.py")
pc = importlib.util.module_from_spec(_spec_pc)
_spec_pc.loader.exec_module(pc)


# Shared config.
_SAC_SEED = demo._SAC_SEED
_EVAL_SEEDS = demo._EVAL_SEEDS
_CURRICULUM_START_TOL = demo._CURRICULUM_START_TOL
_CURRICULUM_END_TOL = demo._CURRICULUM_END_TOL
_CURRICULUM_SCHEDULE_STEPS = demo._CURRICULUM_SCHEDULE_STEPS
_IDENTITY_QUAT = demo._IDENTITY_QUAT
_TABLE_LOWER = demo._TABLE_LOWER
_TABLE_UPPER = demo._TABLE_UPPER

# PushCube oracle config.
_PUSH_DISPLACEMENT_M = pc._PUSH_DISPLACEMENT_M
_FORWARD_PUSH_SCALE = pc._FORWARD_PUSH_SCALE

# Table-push correction config.
_MAX_STEPS = 50
_SOURCE_PLACE_PERTURBATION_RANGE_M = 0.02
_SOURCE_REGION_HALF_EXTENT_M = 0.05
_PUSH_STANDOFF_M = 0.10
_PUSH_TCP_Z_OFFSET_M = 0.006
_PUSH_STAGING_CLEARANCE_M = 0.12
_PUSH_ACTION_SCALE_XY = 0.2
_PUSH_Z_SERVO_SCALE = 0.2
_GRIPPER_OPEN_MIN_WIDTH = 0.04


def _clip_xy_to_table(xy: np.ndarray, margin: float = 0.02) -> np.ndarray:
    lower = _TABLE_LOWER[:2] + margin
    upper = _TABLE_UPPER[:2] - margin
    return np.clip(xy, lower, upper).astype(np.float32)


def _obs_to_scene_with_tcp(obs, regions: dict[str, demo.Region], timestep: int = 0) -> demo.SceneGraph:
    """Injects the TCP as an object into the scene so predicates can evaluate it."""
    scene = demo._obs_to_scene(obs, regions, timestep)
    tcp_pos = obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy().astype(np.float32)
    scene.objects["tcp"] = demo.ObjectState(
        name="tcp", semantic_class="tcp",
        pose=demo.Pose(position=tcp_pos, quat_xyzw=demo._IDENTITY_QUAT)
    )
    return scene


def _slow_open_and_settle(env, obs):
    """Complete the physical release portion of place(source)."""
    for _ in range(15):
        a = torch.tensor(np.array([0.0, 0.0, 0.0, 0.3], dtype=np.float32))
        obs, *_ = env.step(a)
    for _ in range(3):
        a = torch.tensor(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        obs, *_ = env.step(a)
    for _ in range(3):
        a = torch.tensor(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        obs, *_ = env.step(a)
    return obs


class PushCubeTablePushCorrectionEnv(gym.Env):
    """Symbolic place first; RL only performs free-cube table pushing.

    Observation: cube_pos(3), tcp_pos(3), gripper_width(1), goal_pos(3).
    Action: XY pusher command only. The wrapper injects z servo and gripper-open.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = _MAX_STEPS,
        atpose_tolerance: float = _CURRICULUM_START_TOL,
        atpose_temperature: float = 0.05,
        distance_penalty: float = 2.0,
        action_scale_xy: float = _PUSH_ACTION_SCALE_XY,
        z_servo_scale: float = _PUSH_Z_SERVO_SCALE,
        source_place_perturbation_range_m: float = _SOURCE_PLACE_PERTURBATION_RANGE_M,
        push_displacement_m: float = _PUSH_DISPLACEMENT_M,
        success_distance_m: float = _CURRICULUM_END_TOL,
        tcp_near_weight: float = 1.0,
        tcp_near_threshold: float = 0.05,
    ):
        super().__init__()
        self.distance_penalty = distance_penalty
        self._env = gym.make(
            "PushCube-v1",
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pos",
            render_mode=None,
            max_episode_steps=700,
        )
        self.max_steps = max_steps
        self._current_tolerance = atpose_tolerance
        self.atpose_temperature = atpose_temperature
        self.action_scale_xy = action_scale_xy
        self.z_servo_scale = z_servo_scale
        self.source_place_perturbation_range_m = source_place_perturbation_range_m
        self.push_displacement_m = push_displacement_m
        self.success_distance_m = success_distance_m
        self.tcp_near_weight = tcp_near_weight
        self.tcp_near_threshold = tcp_near_threshold
        self.regions = {"table_surface": demo.Region("table_surface", _TABLE_LOWER, _TABLE_UPPER)}

        self.gripper_open_eval = GripperOpenPredicate(
            min_width=_GRIPPER_OPEN_MIN_WIDTH,
            temperature=0.005,
        )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        self.init_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.at_pose_pred: demo.AtPosePredicate | None = None
        self.at_pose_eval: demo.AtPosePredicate | None = None
        self._last_obs = None
        self._step_count = 0
        self._last_place_perturbation: np.ndarray = np.zeros(2, dtype=np.float32)
        self._last_place_target: np.ndarray = np.zeros(3, dtype=np.float32)
        self._last_symbolic_cube_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self._last_push_stage_pos: np.ndarray = np.zeros(3, dtype=np.float32)

    def set_curriculum_tolerance(self, value: float) -> None:
        self._current_tolerance = float(value)

    # --- forward + symbolic phases ----------------------------------------

    def _run_forward_push(self, obs):
        cube_init = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()
        behind_above = cube_init.copy()
        behind_above[0] -= 0.06
        behind_above[2] += 0.10
        obs = pc._step_toward_scaled(
            self._env, obs, behind_above, 30, 0.012, gripper_cmd=-1.0, scale=1.0
        )
        behind_push = cube_init.copy()
        behind_push[0] -= 0.06
        behind_push[2] = cube_init[2] + 0.005
        obs = pc._step_toward_scaled(
            self._env, obs, behind_push, 25, 0.012, gripper_cmd=-1.0, scale=1.0
        )
        push_tgt = cube_init.copy()
        push_tgt[0] += self.push_displacement_m
        push_tgt[2] += 0.005
        obs = pc._step_toward_scaled(
            self._env,
            obs,
            push_tgt,
            30,
            0.005,
            gripper_cmd=-1.0,
            scale=_FORWARD_PUSH_SCALE,
        )
        return obs

    def _sample_source_place_target(self) -> np.ndarray:
        max_offset = min(
            self.source_place_perturbation_range_m,
            _SOURCE_REGION_HALF_EXTENT_M,
        )
        if max_offset > 0.0:
            dx = float(self.np_random.uniform(-max_offset, max_offset))
            dy = float(self.np_random.uniform(-max_offset, max_offset))
        else:
            dx = dy = 0.0
        self._last_place_perturbation = np.array([dx, dy], dtype=np.float32)
        target = self.init_pos.copy().astype(np.float32)
        target[:2] = _clip_xy_to_table(target[:2] + self._last_place_perturbation)
        return target

    def _move_to_push_staging(self, obs):
        cube = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()
        to_goal = self.init_pos[:2] - cube[:2]
        norm = float(np.linalg.norm(to_goal))
        if norm < 1e-4:
            push_dir = np.array([1.0, 0.0], dtype=np.float32)
        else:
            push_dir = (to_goal / norm).astype(np.float32)

        stage_xy = _clip_xy_to_table(cube[:2] - push_dir * _PUSH_STANDOFF_M)
        push_height_stage = np.array(
            [stage_xy[0], stage_xy[1], cube[2] + _PUSH_TCP_Z_OFFSET_M],
            dtype=np.float32,
        )
        above_stage = push_height_stage.copy()
        above_stage[2] = cube[2] + _PUSH_STAGING_CLEARANCE_M

        obs = pc._step_toward_scaled(
            self._env, obs, above_stage, 25, 0.012, gripper_cmd=1.0, scale=1.0
        )
        self._last_push_stage_pos = above_stage
        return obs

    def _run_symbolic_inverse(self, obs):
        """Execute pick(cube) followed by a full physical place(source).

        The release and gripper-open parts happen here, before the RL episode
        starts. RL receives a free cube on the table plus an open-gripper pusher.
        """
        # Break forward-push contact and open before grasping.
        tcp = obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
        up = tcp.copy()
        up[2] += 0.10
        obs = pc._step_toward_scaled(
            self._env, obs, up, 20, 0.012, gripper_cmd=-1.0, scale=1.0
        )
        obs = pc._step_in_place(self._env, obs, 8, gripper_cmd=1.0)

        # Pick at the post-forward cube pose.
        cube_now = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()
        above_cube = cube_now.copy()
        above_cube[2] += 0.10
        obs = pc._step_toward_scaled(
            self._env, obs, above_cube, 20, 0.012, gripper_cmd=1.0, scale=1.0
        )
        obs = pc._step_toward_scaled(
            self._env, obs, cube_now.copy(), 20, 0.012, gripper_cmd=1.0, scale=1.0
        )
        obs = pc._step_in_place(self._env, obs, 10, gripper_cmd=-1.0)

        lift_h = cube_now.copy()
        lift_h[2] += 0.15
        obs = pc._step_toward_scaled(
            self._env, obs, lift_h, 15, 0.02, gripper_cmd=-1.0, scale=1.0
        )

        # Place in the source region, deliberately leaving a small XY residual
        # for the table-push policy to correct.
        place_target = self._sample_source_place_target()
        self._last_place_target = place_target.copy()
        carry_height = max(float(lift_h[2]), float(self.init_pos[2] + 0.14))
        over_place = np.array(
            [place_target[0], place_target[1], carry_height],
            dtype=np.float32,
        )
        obs = pc._step_toward_scaled(
            self._env, obs, over_place, 25, 0.012, gripper_cmd=-1.0, scale=1.0
        )

        place_tcp = place_target.copy()
        place_tcp[2] = self.init_pos[2] + 0.003
        obs = pc._step_toward_scaled(
            self._env, obs, place_tcp, 25, 0.008, gripper_cmd=-1.0, scale=1.0
        )

        # This is the physical completion of place(source): release and settle.
        obs = _slow_open_and_settle(self._env, obs)
        cube_after_place = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()
        self._last_symbolic_cube_pos = cube_after_place

        # Retract open. The separate push handoff runs after this method returns,
        # so this method's end state is the physical place(source) completion.
        tcp = obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
        retract = tcp.copy()
        retract[2] += 0.08
        obs = pc._step_toward_scaled(
            self._env, obs, retract, 15, 0.012, gripper_cmd=1.0, scale=1.0
        )
        return obs

    # --- gym API -----------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        obs, _ = self._env.reset(seed=seed)
        for _ in range(2):
            obs, *_ = self._env.step(torch.tensor(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)))

        self.init_pos = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()
        init_pose = demo.Pose(position=self.init_pos.astype(np.float32), quat_xyzw=_IDENTITY_QUAT)
        self.at_pose_pred = demo.AtPosePredicate(
            "cube",
            target_pose=init_pose,
            distance_threshold=self._current_tolerance,
            temperature=self.atpose_temperature,
        )
        self.at_pose_eval = demo.AtPosePredicate(
            "cube",
            target_pose=init_pose,
            distance_threshold=_CURRICULUM_END_TOL,
            temperature=0.005,
        )

        obs = self._run_forward_push(obs)
        obs = self._run_symbolic_inverse(obs)

        symbolic_scene = demo._obs_to_scene(obs, self.regions)
        symbolic_cube = symbolic_scene.objects["cube"].pose.position.copy()
        symbolic_distance = float(np.linalg.norm(symbolic_cube - self.init_pos))

        obs = self._move_to_push_staging(obs)
        self._last_obs = obs
        self._step_count = 0

        rl_start_scene = demo._obs_to_scene(obs, self.regions)
        rl_start_cube = rl_start_scene.objects["cube"].pose.position
        rl_start_distance = float(np.linalg.norm(rl_start_cube - self.init_pos))
        return self._encode(obs), {
            "phase": "rl_push_start",
            "seed": seed,
            "init_pose": self.init_pos.tolist(),
            "place_target": self._last_place_target.tolist(),
            "place_perturbation_xy": self._last_place_perturbation.tolist(),
            "symbolic_cube_pos": symbolic_cube.tolist(),
            "symbolic_distance_m": symbolic_distance,
            "rl_start_cube_pos": rl_start_cube.tolist(),
            "rl_start_distance_m": rl_start_distance,
            "push_stage_pos": self._last_push_stage_pos.tolist(),
            "current_tolerance_m": self._current_tolerance,
        }

    def step(self, action):
        action_xy = np.asarray(action, dtype=np.float32).copy()
        action_xy = np.clip(action_xy, -1.0, 1.0)

        tcp = self._last_obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
        cube = self._last_obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy()
        push_z = float(cube[2] + _PUSH_TCP_Z_OFFSET_M)
        z_error = push_z - float(tcp[2])

        ms_action = np.zeros(4, dtype=np.float32)
        ms_action[:2] = action_xy * self.action_scale_xy
        ms_action[2] = np.clip(z_error / 0.02, -1.0, 1.0) * self.z_servo_scale
        ms_action[3] = 1.0

        obs, _r, _term, _trunc, _info = self._env.step(torch.tensor(ms_action))
        self._last_obs = obs
        self._step_count += 1

        scene = _obs_to_scene_with_tcp(obs, self.regions)
        residual_score = float(self.at_pose_pred.evaluate(scene).score)
        cube_pos = scene.objects["cube"].pose.position
        tcp_pos = scene.objects["tcp"].pose.position
        distance = float(np.linalg.norm(cube_pos - self.init_pos))
        gripper_width = float(scene.robot.gripper_width)

        # Evaluate NearPredicate("tcp", "cube") which acts as a restored precondition
        tcp_cube_dist = float(np.linalg.norm(tcp_pos - cube_pos))
        near_margin = self.tcp_near_threshold - tcp_cube_dist
        near_score = 1.0 / (1.0 + np.exp(-near_margin / self.atpose_temperature))

        reward = residual_score + (self.tcp_near_weight * near_score) - self.distance_penalty * distance
        terminated = distance <= self.success_distance_m
        truncated = self._step_count >= self.max_steps
        return self._encode(obs), reward, terminated, truncated, {
            "v_residual": residual_score,
            "v_near": near_score,
            "distance": distance,
            "gripper_width": gripper_width,
            "step": self._step_count,
        }

    def _encode(self, obs) -> np.ndarray:
        cube_pos = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy()
        tcp_pos = obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
        qpos = obs["agent"]["qpos"].squeeze().cpu().numpy()
        gw = float(qpos[-2] + qpos[-1])
        return np.concatenate([cube_pos, tcp_pos, [gw], self.init_pos]).astype(np.float32)

    def close(self):
        self._env.close()


def _measure_current_state(env: PushCubeTablePushCorrectionEnv) -> dict:
    scene = demo._obs_to_scene(env._last_obs, env.regions)
    cube = scene.objects["cube"].pose.position
    distance_mm = float(np.linalg.norm(cube - env.init_pos)) * 1000.0
    v_eval = float(env.at_pose_eval.evaluate(scene).score)
    v_gripper = float(env.gripper_open_eval.evaluate(scene).score)
    return {
        "distance_mm": distance_mm,
        "v_final": v_eval,
        "v_after_release": v_eval,
        "v_gripper_open": v_gripper,
        "cube_final": cube.tolist(),
        "init_pose": env.init_pos.tolist(),
        "gripper_width_mm": float(scene.robot.gripper_width) * 1000.0,
    }


def _eval_episode(env: PushCubeTablePushCorrectionEnv, policy_fn, seed: int, do_rl: bool) -> dict:
    obs, reset_info = env.reset(seed=seed)
    rl_steps = 0
    last_info = None
    if do_rl and policy_fn is not None:
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, _r, terminated, truncated, last_info = env.step(action)
            rl_steps += 1

    measured = _measure_current_state(env)
    return {
        "seed": seed,
        "rl_steps": rl_steps,
        "symbolic_distance_mm": float(reset_info["symbolic_distance_m"]) * 1000.0,
        "place_perturbation_xy": reset_info["place_perturbation_xy"],
        "push_stage_pos": reset_info["push_stage_pos"],
        "terminated_by_distance": bool(last_info is not None and last_info["distance"] <= env.success_distance_m),
        **measured,
    }


def _summarize(label: str, episodes: list[dict]) -> dict:
    distances = np.array([e["distance_mm"] for e in episodes], dtype=np.float64)
    v_finals = np.array([e["v_final"] for e in episodes], dtype=np.float64)
    symbolic = np.array([e["symbolic_distance_mm"] for e in episodes], dtype=np.float64)
    n_success = int(np.sum(distances < 10.0))
    return {
        "label": label,
        "n_episodes": len(episodes),
        "per_episode": episodes,
        "symbolic_distance_mm_mean": float(symbolic.mean()),
        "symbolic_distance_mm_std": float(symbolic.std()),
        "distance_mm_mean": float(distances.mean()),
        "distance_mm_std": float(distances.std()),
        "distance_mm_min": float(distances.min()),
        "distance_mm_max": float(distances.max()),
        "v_final_mean": float(v_finals.mean()),
        "v_final_std": float(v_finals.std()),
        "v_after_release_mean": float(v_finals.mean()),
        "v_after_release_std": float(v_finals.std()),
        "success_rate_at_1cm": n_success / len(episodes),
    }


def evaluate_symbolic_only(env: PushCubeTablePushCorrectionEnv, seeds: list[int]) -> dict:
    eps = [_eval_episode(env, policy_fn=None, seed=s, do_rl=False) for s in seeds]
    return _summarize("symbolic_only_released", eps)


def evaluate_random_policy(env: PushCubeTablePushCorrectionEnv, seeds: list[int]) -> dict:
    rng = np.random.default_rng(0)

    def random_fn(_obs):
        return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)

    eps = [_eval_episode(env, policy_fn=random_fn, seed=s, do_rl=True) for s in seeds]
    return _summarize("random_table_push_rl", eps)


def evaluate_trained_policy(env: PushCubeTablePushCorrectionEnv, model: SAC, seeds: list[int], label: str) -> dict:
    def trained_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    eps = [_eval_episode(env, policy_fn=trained_fn, seed=s, do_rl=True) for s in seeds]
    return _summarize(label, eps)


def train(
    total_timesteps: int,
    checkpoint_path: Path,
    best_checkpoint_path: Path,
) -> tuple[SAC, demo.CheckpointAndLogCallback, PushCubeTablePushCorrectionEnv]:
    env = PushCubeTablePushCorrectionEnv(
        max_steps=_MAX_STEPS,
        atpose_tolerance=_CURRICULUM_START_TOL,
        source_place_perturbation_range_m=_SOURCE_PLACE_PERTURBATION_RANGE_M,
        push_displacement_m=_PUSH_DISPLACEMENT_M,
    )
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1_000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_kwargs={"net_arch": [256, 256]},
        seed=_SAC_SEED,
    )
    log_cb = demo.CheckpointAndLogCallback(
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        checkpoint_every=50_000,
        log_every=5_000,
        best_window=50,
        env=env,
    )
    curr_cb = demo.CurriculumCallback(
        env,
        start_tol=_CURRICULUM_START_TOL,
        end_tol=_CURRICULUM_END_TOL,
        schedule_steps=_CURRICULUM_SCHEDULE_STEPS,
    )
    callbacks = CallbackList([log_cb, curr_cb])
    print("Training SAC for table-push correction after released symbolic place...")
    print(
        f"  action=XY only, gripper forced open, source place perturbation "
        f"+/-{_SOURCE_PLACE_PERTURBATION_RANGE_M * 100:.1f}cm, "
        f"max_steps={_MAX_STEPS}, tol curriculum "
        f"{_CURRICULUM_START_TOL * 1000:.0f}->{_CURRICULUM_END_TOL * 1000:.0f}mm"
    )
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    print(
        f"Training done. {len(log_cb.episode_returns)} episodes completed. "
        f"Best rolling V={log_cb.best_rolling_v:.4f} at step {log_cb.best_at_step}."
    )
    return model, log_cb, env


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "planrob_inverse_rl_pushcube_push_correction_model_final.zip"
    best_path = out_dir / "planrob_inverse_rl_pushcube_push_correction_model_best.zip"
    curve_path = out_dir / "planrob_inverse_rl_pushcube_push_correction_curve.png"
    json_path = out_dir / "planrob_inverse_rl_pushcube_push_correction_demo.json"

    base_env = PushCubeTablePushCorrectionEnv(
        max_steps=_MAX_STEPS,
        atpose_tolerance=_CURRICULUM_END_TOL,
        source_place_perturbation_range_m=_SOURCE_PLACE_PERTURBATION_RANGE_M,
        push_displacement_m=_PUSH_DISPLACEMENT_M,
    )
    base_env.set_curriculum_tolerance(_CURRICULUM_END_TOL)

    print(f"Evaluating symbolic-only released-place baseline on {len(_EVAL_SEEDS)} seeds...")
    sym = evaluate_symbolic_only(base_env, _EVAL_SEEDS)
    print(
        f"  symbolic_only:  dist {sym['distance_mm_mean']:.1f}+/-{sym['distance_mm_std']:.1f}mm  "
        f"V {sym['v_final_mean']:.3f}  success@1cm {sym['success_rate_at_1cm']:.1%}"
    )

    print(f"Evaluating random table-push baseline on {len(_EVAL_SEEDS)} seeds...")
    rnd = evaluate_random_policy(base_env, _EVAL_SEEDS)
    print(
        f"  random_rl:      dist {rnd['distance_mm_mean']:.1f}+/-{rnd['distance_mm_std']:.1f}mm  "
        f"V {rnd['v_final_mean']:.3f}  success@1cm {rnd['success_rate_at_1cm']:.1%}"
    )
    base_env.close()

    final_model, callback, env = train(1_000_000, ckpt_path, best_path)
    final_model.save(ckpt_path)
    demo.plot_curve(callback, curve_path)

    env.set_curriculum_tolerance(_CURRICULUM_END_TOL)

    print(f"Evaluating trained SAC table-push policy on {len(_EVAL_SEEDS)} seeds...")
    final_eval = evaluate_trained_policy(env, final_model, _EVAL_SEEDS, "trained_table_push_final")
    print(
        f"  trained_final:  dist {final_eval['distance_mm_mean']:.1f}+/-{final_eval['distance_mm_std']:.1f}mm  "
        f"V {final_eval['v_final_mean']:.3f}  success@1cm {final_eval['success_rate_at_1cm']:.1%}"
    )

    if best_path.exists():
        best_model = SAC.load(best_path, env=env)
        print(f"Evaluating trained SAC table-push best checkpoint on {len(_EVAL_SEEDS)} seeds...")
        best_eval = evaluate_trained_policy(env, best_model, _EVAL_SEEDS, "trained_table_push_best")
        print(
            f"  trained_best:   dist {best_eval['distance_mm_mean']:.1f}+/-{best_eval['distance_mm_std']:.1f}mm  "
            f"V {best_eval['v_final_mean']:.3f}  success@1cm {best_eval['success_rate_at_1cm']:.1%}"
        )
    else:
        best_eval = None
    env.close()

    payload = {
        "phase": "pushcube_table_push_correction",
        "phase_description": (
            "PushCube-v1 symbolic inverse fully executes pick+place(source) with open gripper; "
            "RL then performs XY table-push correction on the free cube."
        ),
        "sac_seed": _SAC_SEED,
        "eval_seeds": _EVAL_SEEDS,
        "baseline_symbolic_only": sym,
        "baseline_random_rl": rnd,
        "trained_final": final_eval,
        "trained_best": best_eval,
        "training": {
            "n_episodes_logged": len(callback.episode_returns),
            "first50_mean_final_v": float(np.mean(callback.episode_v_finals[:50]))
            if callback.episode_v_finals
            else None,
            "last50_mean_final_v": float(np.mean(callback.episode_v_finals[-50:]))
            if callback.episode_v_finals
            else None,
            "last50_mean_distance_mm": float(np.mean(callback.episode_distances[-50:])) * 1000.0
            if callback.episode_distances
            else None,
            "best_rolling_v": float(callback.best_rolling_v),
            "best_rolling_v_at_step": int(callback.best_at_step),
        },
        "config": {
            "env": "PushCube-v1",
            "algorithm": "SAC",
            "reward": "V_at_pose(cube, init_pose) + V_near(tcp, cube) - 2.0 * distance(cube, init_pose)",
            "termination": "distance(cube, init_pose) <= 1cm OR max_steps",
            "total_timesteps": 1_000_000,
            "max_steps_per_episode": _MAX_STEPS,
            "push_displacement_m": _PUSH_DISPLACEMENT_M,
            "push_action_scale_forward": _FORWARD_PUSH_SCALE,
            "source_place_perturbation_range_m": _SOURCE_PLACE_PERTURBATION_RANGE_M,
            "source_region_half_extent_m": _SOURCE_REGION_HALF_EXTENT_M,
            "rl_action_space": "2D XY pusher command; gripper command forced open",
            "rl_push_standoff_m": _PUSH_STANDOFF_M,
            "rl_push_tcp_z_offset_m": _PUSH_TCP_Z_OFFSET_M,
            "rl_action_scale_xy": _PUSH_ACTION_SCALE_XY,
            "rl_z_servo_scale": _PUSH_Z_SERVO_SCALE,
            "atpose_tolerance_curriculum_start_m": _CURRICULUM_START_TOL,
            "atpose_tolerance_curriculum_end_m": _CURRICULUM_END_TOL,
            "atpose_tolerance_curriculum_schedule_steps": _CURRICULUM_SCHEDULE_STEPS,
            "atpose_temperature_train": 0.05,
            "atpose_temperature_eval": 0.005,
            "distance_penalty_coef": 2.0,
            "success_distance_m": _CURRICULUM_END_TOL,
            "buffer_size": 100_000,
            "batch_size": 256,
            "gamma": 0.99,
            "net_arch": [256, 256],
            "post_rl_release_motion": None,
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
