"""End-to-end inverse skill on PushCube-v1 — RL solves the FULL inverse target.

Differs from `planrob_inverse_rl_pushcube_demo.py` in one key respect: the RL
reward includes both predicates from the operator-extracted inverse target —
`at_pose(cube, init_pose)` AND `gripper_open()` — instead of only at_pose.
The agent must therefore (a) bring the cube to the precise init_pose,
then (b) keep the gripper open without disturbing the cube. In the current
reset sequence, the symbolic inverse already performs the coarse place(source)
release and settle; the RL phase corrects the remaining residual from that
released-on-table handoff.

This keeps the operator-extraction → reward → trained-policy chain explicit:
every term in the inverse target appears in the reward, while the scripted
prefix implements the coarse symbolic inverse.

Reward:    V_at_pose + V_gripper_open - 2.0 * distance(cube, init_pose)
Terminate: V_at_pose >= 0.90 AND V_gripper_open >= 0.90

Outputs:
  artifacts/planrob_inverse_rl_pushcube_full_*.{json,png,zip}
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

# Reuse helpers + RL config from canonical PickCube demo + PushCube oracle.
_spec = importlib.util.spec_from_file_location("demo", "scripts/planrob_inverse_rl_demo.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

_spec_pc = importlib.util.spec_from_file_location("pc", "scripts/planrob_inverse_rl_pushcube_demo.py")
pc = importlib.util.module_from_spec(_spec_pc)
_spec_pc.loader.exec_module(pc)


# ── Config (inherited from Phase 2 + PushCube oracle) ────────────────────────


_SAC_SEED = demo._SAC_SEED
_EVAL_SEEDS = demo._EVAL_SEEDS
_CURRICULUM_START_TOL = demo._CURRICULUM_START_TOL
_CURRICULUM_END_TOL = demo._CURRICULUM_END_TOL
_CURRICULUM_SCHEDULE_STEPS = demo._CURRICULUM_SCHEDULE_STEPS
_ACTION_SCALE_XYZ = demo._ACTION_SCALE_XYZ
_PERTURBATION_RANGE_M = demo._PERTURBATION_RANGE_M
_IDENTITY_QUAT = demo._IDENTITY_QUAT
_TABLE_LOWER = demo._TABLE_LOWER
_TABLE_UPPER = demo._TABLE_UPPER

# PushCube oracle config. The fixed displacement is now only a fallback; by
# default the forward push uses ManiSkill's per-episode `goal_pos` x coordinate.
_PUSH_DISPLACEMENT_M = pc._PUSH_DISPLACEMENT_M
_FORWARD_PUSH_SCALE = pc._FORWARD_PUSH_SCALE
_USE_ENV_GOAL_FOR_PUSH = True

# Phase 5 specifics — full inverse target reward
_GRIPPER_OPEN_TEMP_TRAIN = 0.02   # softer than eval temp for usable training gradient
_GRIPPER_OPEN_TEMP_EVAL = 0.005
_GRIPPER_OPEN_MIN_WIDTH = 0.04    # gripper_width threshold for "open"
_MAX_STEPS = 30                    # room for residual correction after release


# ── Env ──────────────────────────────────────────────────────────────────────


class PushCubeRecoveryFullEnv(gym.Env):
    """PushCube-v1 inverse env where RL's reward covers the FULL inverse target.

    Reward       = V_at_pose + V_gripper_open - 2.0 * distance
    Termination  = V_at_pose >= 0.90 AND V_gripper_open >= 0.90
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = _MAX_STEPS, success_threshold: float = 0.90,
                 atpose_tolerance: float = _CURRICULUM_START_TOL,
                 atpose_temperature: float = 0.05,
                 distance_penalty: float = 2.0,
                 gripper_open_weight: float = 1.0,
                 action_scale_xyz: float = _ACTION_SCALE_XYZ,
                 perturbation_range_m: float = _PERTURBATION_RANGE_M,
                 push_displacement_m: float = _PUSH_DISPLACEMENT_M,
                 use_env_goal_for_push: bool = _USE_ENV_GOAL_FOR_PUSH):
        super().__init__()
        self.distance_penalty = distance_penalty
        self.gripper_open_weight = gripper_open_weight
        self._env = gym.make(
            "PushCube-v1", obs_mode="state_dict",
            control_mode="pd_ee_delta_pos", render_mode=None,
            max_episode_steps=600,
        )
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self._current_tolerance = atpose_tolerance
        self.atpose_temperature = atpose_temperature
        self.action_scale_xyz = action_scale_xyz
        self.perturbation_range_m = perturbation_range_m
        self.push_displacement_m = push_displacement_m
        self.use_env_goal_for_push = use_env_goal_for_push
        self.regions = {"table_surface": demo.Region("table_surface", _TABLE_LOWER, _TABLE_UPPER)}

        # Predicates — the gripper_open ones are non-mutating, set once.
        self.gripper_open_train = GripperOpenPredicate(
            min_width=_GRIPPER_OPEN_MIN_WIDTH, temperature=_GRIPPER_OPEN_TEMP_TRAIN)
        self.gripper_open_eval = GripperOpenPredicate(
            min_width=_GRIPPER_OPEN_MIN_WIDTH, temperature=_GRIPPER_OPEN_TEMP_EVAL)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.init_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.at_pose_pred: demo.AtPosePredicate | None = None
        self.at_pose_eval: demo.AtPosePredicate | None = None
        self._last_obs = None
        self._step_count = 0
        self._last_perturbation: np.ndarray = np.zeros(2, dtype=np.float32)
        self._last_forward_goal_pos: np.ndarray | None = None
        self._last_forward_push_displacement_m = float(push_displacement_m)

    def set_curriculum_tolerance(self, value: float) -> None:
        self._current_tolerance = float(value)

    def _resolve_forward_push_displacement(self, obs, cube_init: np.ndarray) -> float:
        """Use ManiSkill's per-episode goal if available, otherwise fall back
        to the legacy fixed PushCube displacement."""
        goal_pos_tensor = obs.get("extra", {}).get("goal_pos")
        if self.use_env_goal_for_push and goal_pos_tensor is not None:
            goal_pos = goal_pos_tensor.squeeze()[:3].cpu().numpy().astype(np.float32)
            self._last_forward_goal_pos = goal_pos.copy()
            self._last_forward_push_displacement_m = float(goal_pos[0])
        else:
            self._last_forward_goal_pos = None
            self._last_forward_push_displacement_m = float(self.push_displacement_m)
        return self._last_forward_push_displacement_m

    # --- forward + symbolic phases (same as plain PushCube) ---------------

    def _run_forward_push(self, obs):
        cube_init = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()
        push_dx = self._resolve_forward_push_displacement(obs, cube_init)
        behind_push = cube_init.copy(); behind_push[0] -= 0.06; behind_push[2] = cube_init[2] + 0.005
        obs = pc._step_toward_scaled(self._env, obs, behind_push, 30, 0.012, gripper_cmd=-1.0, scale=1.0)
        push_tgt = cube_init.copy(); push_tgt[0] += push_dx; push_tgt[2] += 0.005
        obs = pc._step_toward_scaled(self._env, obs, push_tgt, 30, 0.005, gripper_cmd=-1.0,
                                      scale=_FORWARD_PUSH_SCALE)
        tcp = obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
        up = tcp.copy(); up[2] += 0.10
        obs = pc._step_toward_scaled(self._env, obs, up, 20, 0.012, gripper_cmd=-1.0, scale=1.0)
        obs = pc._step_in_place(self._env, obs, 8, gripper_cmd=1.0)
        return obs

    def _run_symbolic_inverse(self, obs):
        cube_now = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()
        above_cube = cube_now.copy(); above_cube[2] += 0.10
        obs = pc._step_toward_scaled(self._env, obs, above_cube, 20, 0.012, gripper_cmd=1.0, scale=1.0)
        obs = pc._step_toward_scaled(self._env, obs, cube_now.copy(), 20, 0.012, gripper_cmd=1.0, scale=1.0)
        obs = pc._step_in_place(self._env, obs, 10, gripper_cmd=-1.0)
        lift_h = cube_now.copy(); lift_h[2] += 0.15
        obs = pc._step_toward_scaled(self._env, obs, lift_h, 15, 0.02, gripper_cmd=-1.0, scale=1.0)
        over_init = np.array([self.init_pos[0], self.init_pos[1], lift_h[2]], dtype=np.float32)
        obs = pc._step_toward_scaled(self._env, obs, over_init, 25, 0.012, gripper_cmd=-1.0, scale=1.0)
        if self.perturbation_range_m > 0.0:
            dx = float(self.np_random.uniform(-self.perturbation_range_m, self.perturbation_range_m))
            dy = float(self.np_random.uniform(-self.perturbation_range_m, self.perturbation_range_m))
        else:
            dx = dy = 0.0
        self._last_perturbation = np.array([dx, dy], dtype=np.float32)
        handoff = self.init_pos.copy()
        handoff[0] += dx; handoff[1] += dy; handoff[2] += 0.005
        obs = pc._step_toward_scaled(self._env, obs, handoff, 20, 0.012, gripper_cmd=-1.0, scale=1.0)
        obs = pc._step_in_place(self._env, obs, 10, gripper_cmd=1.0)
        obs = pc._step_in_place(self._env, obs, 10, gripper_cmd=1.0)
        return obs

    # --- gym API ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        obs, _ = self._env.reset(seed=seed)
        for _ in range(2):
            obs, *_ = self._env.step(torch.tensor(np.array([0., 0., 0., 1.], dtype=np.float32)))

        self.init_pos = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()
        init_pose = demo.Pose(position=self.init_pos.astype(np.float32), quat_xyzw=_IDENTITY_QUAT)
        self.at_pose_pred = demo.AtPosePredicate(
            "cube", target_pose=init_pose,
            distance_threshold=self._current_tolerance, temperature=self.atpose_temperature)
        self.at_pose_eval = demo.AtPosePredicate(
            "cube", target_pose=init_pose,
            distance_threshold=_CURRICULUM_END_TOL, temperature=0.005)

        obs = self._run_forward_push(obs)
        obs = self._run_symbolic_inverse(obs)
        self._last_obs = obs
        self._step_count = 0
        return self._encode(obs), {
            "phase": "rl_start", "seed": seed,
            "init_pose": self.init_pos.tolist(),
            "perturbation_xy": self._last_perturbation.tolist(),
            "current_tolerance_m": self._current_tolerance,
            "use_env_goal_for_push": self.use_env_goal_for_push,
            "forward_goal_pos": None if self._last_forward_goal_pos is None
                else self._last_forward_goal_pos.tolist(),
            "forward_push_displacement_m": self._last_forward_push_displacement_m,
        }

    def step(self, action):
        action_np = np.asarray(action, dtype=np.float32).copy()
        action_np[:3] *= self.action_scale_xyz
        obs, _r, _term, _trunc, info = self._env.step(torch.tensor(action_np))
        self._last_obs = obs
        self._step_count += 1

        scene = demo._obs_to_scene(obs, self.regions)
        at_pose_score = float(self.at_pose_pred.evaluate(scene).score)
        gripper_score = float(self.gripper_open_train.evaluate(scene).score)
        cube_pos = scene.objects["cube"].pose.position
        distance = float(np.linalg.norm(cube_pos - self.init_pos))

        # Phase 5 reward: full operator-extracted inverse target
        # (at_pose + gripper_open) - distance shaping
        reward = at_pose_score + self.gripper_open_weight * gripper_score \
                 - self.distance_penalty * distance
        # Termination: BOTH predicates satisfied
        terminated = (at_pose_score >= self.success_threshold and
                      gripper_score >= self.success_threshold)
        truncated = self._step_count >= self.max_steps
        return self._encode(obs), reward, terminated, truncated, {
            "v_residual": at_pose_score,
            "v_gripper_open": gripper_score,
            "distance": distance,
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


# ── Eval that reports gripper state too ──────────────────────────────────────


def _eval_episode_full(env: PushCubeRecoveryFullEnv, policy_fn, seed: int, do_rl: bool,
                        do_release: bool = True) -> dict:
    """Same as demo._eval_episode but additionally records gripper state at
    the post-RL and post-release moments."""
    obs, _info_reset = env.reset(seed=seed)
    rl_steps = 0
    if do_rl and policy_fn is not None:
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, _r, terminated, truncated, info = env.step(action)
            rl_steps += 1

    # Pre-release diagnostics
    pre_scene = demo._obs_to_scene(env._last_obs, env.regions)
    v_at_pose_pre = float(env.at_pose_eval.evaluate(pre_scene).score)
    v_gripper_pre = float(env.gripper_open_eval.evaluate(pre_scene).score)
    cube_pre = pre_scene.objects["cube"].pose.position
    pre_dist_mm = float(np.linalg.norm(cube_pre - env.init_pos)) * 1000.0
    pre_gripper_width = float(pre_scene.robot.gripper_width)

    if do_release:
        # Same slow-open release motion as the canonical demo.
        ms_obs = env._last_obs
        for _ in range(15):
            a = torch.tensor(np.array([0.0, 0.0, 0.0, 0.3], dtype=np.float32))
            ms_obs, *_ = env._env.step(a)
        for _ in range(3):
            a = torch.tensor(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            ms_obs, *_ = env._env.step(a)
        tcp = ms_obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
        retract = tcp.copy().astype(np.float32); retract[2] += 0.05
        ms_obs = pc._step_toward_scaled(env._env, ms_obs, retract, 10, 0.01, gripper_cmd=1.0, scale=1.0)
        env._last_obs = ms_obs

    post_scene = demo._obs_to_scene(env._last_obs, env.regions)
    cube_final = post_scene.objects["cube"].pose.position
    post_dist_mm = float(np.linalg.norm(cube_final - env.init_pos)) * 1000.0
    v_at_pose_post = float(env.at_pose_eval.evaluate(post_scene).score)
    v_gripper_post = float(env.gripper_open_eval.evaluate(post_scene).score)
    post_gripper_width = float(post_scene.robot.gripper_width)

    return {
        "seed": seed,
        "rl_steps": rl_steps,
        "pre_release_dist_mm": pre_dist_mm,
        "pre_release_v_at_pose": v_at_pose_pre,
        "pre_release_v_gripper_open": v_gripper_pre,
        "pre_release_gripper_width_mm": pre_gripper_width * 1000.0,
        "post_release_dist_mm": post_dist_mm,
        "v_at_pose_post": v_at_pose_post,
        "v_gripper_open_post": v_gripper_post,
        "post_release_gripper_width_mm": post_gripper_width * 1000.0,
        "distance_mm": post_dist_mm,
        "v_after_release": v_at_pose_post,  # for compatibility with _summarize
        "cube_final": cube_final.tolist(),
        "init_pose": env.init_pos.tolist(),
    }


def evaluate_with_release(env, model_predict_fn, seeds, label, do_rl=True):
    eps = [_eval_episode_full(env, model_predict_fn, s, do_rl=do_rl, do_release=True) for s in seeds]
    return demo._summarize(label, eps)


def evaluate_no_release(env, model_predict_fn, seeds, label, do_rl=True):
    """Eval that skips the scripted release — directly measure cube position
    after RL terminates. The strongest claim: 'RL alone solves the full inverse'."""
    eps = [_eval_episode_full(env, model_predict_fn, s, do_rl=do_rl, do_release=False) for s in seeds]
    return demo._summarize(label, eps)


# ── Training ─────────────────────────────────────────────────────────────────


def train(total_timesteps: int, checkpoint_path: Path, best_checkpoint_path: Path
          ) -> tuple[SAC, demo.CheckpointAndLogCallback, PushCubeRecoveryFullEnv]:
    env = PushCubeRecoveryFullEnv(
        max_steps=_MAX_STEPS,
        atpose_tolerance=_CURRICULUM_START_TOL,
        action_scale_xyz=_ACTION_SCALE_XYZ,
        perturbation_range_m=_PERTURBATION_RANGE_M,
        push_displacement_m=_PUSH_DISPLACEMENT_M,
    )
    model = SAC(
        "MlpPolicy", env, verbose=0,
        learning_rate=3e-4, buffer_size=100_000,
        learning_starts=1_000, batch_size=256,
        tau=0.005, gamma=0.99,
        policy_kwargs={"net_arch": [256, 256]},
        seed=_SAC_SEED,
    )
    log_cb = demo.CheckpointAndLogCallback(
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        checkpoint_every=50_000, log_every=5_000, best_window=50, env=env,
    )
    curr_cb = demo.CurriculumCallback(
        env, start_tol=_CURRICULUM_START_TOL, end_tol=_CURRICULUM_END_TOL,
        schedule_steps=_CURRICULUM_SCHEDULE_STEPS,
    )
    callbacks = CallbackList([log_cb, curr_cb])
    print(f"Training SAC for {total_timesteps} timesteps "
          f"(PushCube-v1, FULL inverse target reward)...")
    print(f"  reward = V_at_pose + V_gripper_open - 2*distance  "
          f"(both terms must satisfy 0.90 to terminate)")
    push_mode = "env_goal_x" if env.use_env_goal_for_push else f"fixed_{_PUSH_DISPLACEMENT_M*100:.1f}cm"
    print(f"  push_mode={push_mode}  fallback_push_disp={_PUSH_DISPLACEMENT_M*100:.1f}cm  "
          f"push_scale={_FORWARD_PUSH_SCALE}  "
          f"action_scale={_ACTION_SCALE_XYZ}  max_steps={_MAX_STEPS}  "
          f"tol curriculum {_CURRICULUM_START_TOL*1000:.0f}->{_CURRICULUM_END_TOL*1000:.0f}mm")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    print(f"Training done. {len(log_cb.episode_returns)} episodes completed.")
    return model, log_cb, env


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "planrob_inverse_rl_pushcube_full_model_final.zip"
    best_path = out_dir / "planrob_inverse_rl_pushcube_full_model_best.zip"
    curve_path = out_dir / "planrob_inverse_rl_pushcube_full_curve.png"
    json_path = out_dir / "planrob_inverse_rl_pushcube_full_demo.json"

    base_env = PushCubeRecoveryFullEnv(
        max_steps=_MAX_STEPS, atpose_tolerance=_CURRICULUM_END_TOL,
        action_scale_xyz=_ACTION_SCALE_XYZ, perturbation_range_m=_PERTURBATION_RANGE_M,
        push_displacement_m=_PUSH_DISPLACEMENT_M,
    )
    base_env.set_curriculum_tolerance(_CURRICULUM_END_TOL)

    print(f"Evaluating symbolic-only baseline on {len(_EVAL_SEEDS)} seeds...")
    sym = evaluate_with_release(base_env, None, _EVAL_SEEDS, "symbolic_only", do_rl=False)
    print(f"  symbolic_only:  dist {sym['distance_mm_mean']:.1f}±{sym['distance_mm_std']:.1f}mm  "
          f"V {sym['v_after_release_mean']:.3f}  success@1cm {sym['success_rate_at_1cm']:.1%}")

    print(f"Evaluating random-RL baseline on {len(_EVAL_SEEDS)} seeds...")
    rng = np.random.default_rng(0)
    rnd = evaluate_with_release(base_env, lambda _o: rng.uniform(-1, 1, size=4).astype(np.float32),
                                 _EVAL_SEEDS, "random_rl", do_rl=True)
    print(f"  random_rl:      dist {rnd['distance_mm_mean']:.1f}±{rnd['distance_mm_std']:.1f}mm  "
          f"V {rnd['v_after_release_mean']:.3f}  success@1cm {rnd['success_rate_at_1cm']:.1%}")
    base_env.close()

    final_model, callback, env = train(1_000_000, ckpt_path, best_path)
    final_model.save(ckpt_path)
    demo.plot_curve(callback, curve_path)

    env.set_curriculum_tolerance(_CURRICULUM_END_TOL)

    def predict_fn(obs):
        action, _ = final_model.predict(obs, deterministic=True)
        return action

    print(f"\nEvaluating trained_final WITH scripted release on {len(_EVAL_SEEDS)} seeds...")
    final_with = evaluate_with_release(env, predict_fn, _EVAL_SEEDS, "trained_final_with_release", do_rl=True)
    print(f"  with_release:   dist {final_with['distance_mm_mean']:.1f}±{final_with['distance_mm_std']:.1f}mm  "
          f"V {final_with['v_after_release_mean']:.3f}  success@1cm {final_with['success_rate_at_1cm']:.1%}")

    print(f"\nEvaluating trained_final WITHOUT scripted release on {len(_EVAL_SEEDS)} seeds...")
    final_without = evaluate_no_release(env, predict_fn, _EVAL_SEEDS, "trained_final_no_release", do_rl=True)
    print(f"  no_release:     dist {final_without['distance_mm_mean']:.1f}±{final_without['distance_mm_std']:.1f}mm  "
          f"V {final_without['v_after_release_mean']:.3f}  success@1cm {final_without['success_rate_at_1cm']:.1%}")

    if best_path.exists():
        best_model = SAC.load(best_path, env=env)
        def best_predict(obs):
            action, _ = best_model.predict(obs, deterministic=True)
            return action
        best_with = evaluate_with_release(env, best_predict, _EVAL_SEEDS, "trained_best_with_release", do_rl=True)
        print(f"\n  best_with_release:  dist {best_with['distance_mm_mean']:.1f}±{best_with['distance_mm_std']:.1f}mm  "
              f"V {best_with['v_after_release_mean']:.3f}  success@1cm {best_with['success_rate_at_1cm']:.1%}")
    else:
        best_with = None

    env.close()

    payload = {
        "phase": "pushcube_full",
        "phase_description": "PushCube-v1 + RL reward over FULL inverse target (at_pose + gripper_open)",
        "sac_seed": _SAC_SEED,
        "eval_seeds": _EVAL_SEEDS,
        "baseline_symbolic_only": sym,
        "baseline_random_rl": rnd,
        "trained_final_with_release": final_with,
        "trained_final_no_release": final_without,
        "trained_best_with_release": best_with,
        "training": {
            "n_episodes_logged": len(callback.episode_returns),
            "first50_mean_final_v": float(np.mean(callback.episode_v_finals[:50])) if callback.episode_v_finals else None,
            "last50_mean_final_v": float(np.mean(callback.episode_v_finals[-50:])) if callback.episode_v_finals else None,
            "last50_mean_distance_mm": float(np.mean(callback.episode_distances[-50:])) * 1000.0
                if callback.episode_distances else None,
            "best_rolling_v": float(callback.best_rolling_v),
            "best_rolling_v_at_step": int(callback.best_at_step),
        },
        "config": {
            "env": "PushCube-v1",
            "algorithm": "SAC",
            "reward": "V_at_pose + V_gripper_open - 2.0 * distance(cube, init_pose)",
            "termination": "V_at_pose >= 0.90 AND V_gripper_open >= 0.90",
            "total_timesteps": 1_000_000,
            "max_steps_per_episode": _MAX_STEPS,
            "forward_push_mode": "env_goal_x" if _USE_ENV_GOAL_FOR_PUSH else "fixed_displacement",
            "use_env_goal_for_push": _USE_ENV_GOAL_FOR_PUSH,
            "fallback_push_displacement_m": _PUSH_DISPLACEMENT_M,
            "push_action_scale": _FORWARD_PUSH_SCALE,
            "atpose_tolerance_curriculum_start_m": _CURRICULUM_START_TOL,
            "atpose_tolerance_curriculum_end_m": _CURRICULUM_END_TOL,
            "atpose_tolerance_curriculum_schedule_steps": _CURRICULUM_SCHEDULE_STEPS,
            "atpose_temperature_train": 0.05,
            "atpose_temperature_eval": 0.005,
            "gripper_open_temperature_train": _GRIPPER_OPEN_TEMP_TRAIN,
            "gripper_open_temperature_eval": _GRIPPER_OPEN_TEMP_EVAL,
            "gripper_open_min_width_m": _GRIPPER_OPEN_MIN_WIDTH,
            "gripper_open_weight": 1.0,
            "buffer_size": 100_000, "batch_size": 256, "gamma": 0.99,
            "net_arch": [256, 256],
            "distance_penalty_coef": 2.0,
            "success_threshold": 0.90,
            "action_scale_xyz": _ACTION_SCALE_XYZ,
            "handoff_perturbation_range_m": _PERTURBATION_RANGE_M,
            "release_motion_for_with_release_eval": "slow_open_v1 (15 steps cmd=0.3 + 3 settle)",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {json_path}")


if __name__ == "__main__":
    main()
