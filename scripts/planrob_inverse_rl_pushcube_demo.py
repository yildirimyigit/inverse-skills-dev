"""End-to-end inverse skill execution on ManiSkill3 PushCube-v1.

Forward skill: scripted closed-gripper push (~6cm displacement in +X).
Inverse via pick-and-place: scripted pick at post-push position, lift, carry
back to above init_pose with handoff perturbation, then RL refines the held
cube precisely. Final release uses the canonical slow-open motion.

This is the canonical PushCube experiment. The architecture mirrors Phase 2 on
PickCube — the only differences are the forward push oracle and the symbolic
inverse needing a pick step (since the cube is on the table after forward).

Pipeline:
  1. Forward push (scripted, action_scale=0.2 in push phase to avoid cube
     inertia overshoot; ends with cube ~6cm in +X from init).
  2. Symbolic inverse (scripted): break gripper-cube contact, open gripper,
     re-read cube position, descend, close, lift, move to above init_pose,
     descend to handoff (5cm above init + ±2cm perturbation, cube held).
  3. RL phase (SAC, 1M steps, Phase 2 hybrid reward) refines the held cube
     to precise init_pose.
  4. Slow-open release motion (15 steps cmd=0.3 + 3 settle), measure final
     cube distance to init_pose.

Outputs:
  artifacts/planrob_inverse_rl_pushcube_*.{json,png,zip}
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

# Reuse helpers + RL config from the canonical PickCube demo.
_spec = importlib.util.spec_from_file_location("demo", "scripts/planrob_inverse_rl_demo.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


# ── Config (inherited from Phase 2) ──────────────────────────────────────────


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

# PushCube-specific
_PUSH_DISPLACEMENT_M = 0.03   # intended forward push distance in +X (cube ends ~6cm)
_FORWARD_PUSH_SCALE = 0.2     # action scaling during push phase to control cube inertia


# ── Helpers (push-specific) ──────────────────────────────────────────────────


def _step_toward_scaled(env, obs, target_xyz: np.ndarray, n_steps: int, tol: float,
                         gripper_cmd: float, scale: float = 1.0):
    """Move EEF toward target with optional action scaling. Used for the push
    phase (scale=0.2) to avoid cube inertia overshoot."""
    for _ in range(n_steps):
        tcp = obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
        delta = (target_xyz - tcp).astype(np.float64)
        norm = float(np.linalg.norm(delta))
        if norm < tol:
            break
        action = np.zeros(4, dtype=np.float32)
        action[:3] = (np.clip(delta / max(norm, 1e-6), -1.0, 1.0) * scale).astype(np.float32)
        action[3] = gripper_cmd
        obs, *_ = env.step(torch.tensor(action))
    return obs


def _step_in_place(env, obs, n_steps: int, gripper_cmd: float):
    """No translation, just send a gripper command for n steps (open/close)."""
    for _ in range(n_steps):
        a = torch.tensor(np.array([0.0, 0.0, 0.0, gripper_cmd], dtype=np.float32))
        obs, *_ = env.step(a)
    return obs


# ── Env ──────────────────────────────────────────────────────────────────────


class PushCubeRecoveryEnv(gym.Env):
    """ManiSkill3 PushCube-v1 wrapped for inverse-skill RL.

    On reset: replays forward push + symbolic inverse (pick + carry above init).
    On step: applies pd_ee_delta_pos action with xyz capping (Phase 2 architecture).
    Reward: Phase 2 hybrid V_residual_sigmoid - 2*distance.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 20, success_threshold: float = 0.90,
                 atpose_tolerance: float = _CURRICULUM_START_TOL,
                 atpose_temperature: float = 0.05,
                 distance_penalty: float = 2.0,
                 action_scale_xyz: float = _ACTION_SCALE_XYZ,
                 perturbation_range_m: float = _PERTURBATION_RANGE_M,
                 push_displacement_m: float = _PUSH_DISPLACEMENT_M):
        super().__init__()
        self.distance_penalty = distance_penalty
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
        self.regions = {"table_surface": demo.Region("table_surface", _TABLE_LOWER, _TABLE_UPPER)}

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.init_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.at_pose_pred: demo.AtPosePredicate | None = None
        self.at_pose_eval: demo.AtPosePredicate | None = None
        self._last_obs = None
        self._step_count = 0
        self._last_perturbation: np.ndarray = np.zeros(2, dtype=np.float32)

    def set_curriculum_tolerance(self, value: float) -> None:
        self._current_tolerance = float(value)

    # --- forward + symbolic phases ----------------------------------------

    def _run_forward_push(self, obs):
        """Closed-gripper push in +X by `push_displacement_m`. Action-scaled
        to 0.2 in the push phase to control cube inertia. Approach-from-behind
        and lower-to-push-height phases run at scale 1.0 (no contact)."""
        cube_init = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()
        # Approach from -X side, gripper closed
        behind_above = cube_init.copy(); behind_above[0] -= 0.06; behind_above[2] += 0.10
        obs = _step_toward_scaled(self._env, obs, behind_above, 30, 0.012, gripper_cmd=-1.0, scale=1.0)
        behind_push = cube_init.copy(); behind_push[0] -= 0.06; behind_push[2] = cube_init[2] + 0.005
        obs = _step_toward_scaled(self._env, obs, behind_push, 25, 0.012, gripper_cmd=-1.0, scale=1.0)
        # Push (scale=0.2 to avoid cube overshoot)
        push_tgt = cube_init.copy(); push_tgt[0] += self.push_displacement_m; push_tgt[2] += 0.005
        obs = _step_toward_scaled(self._env, obs, push_tgt, 30, 0.005, gripper_cmd=-1.0,
                                   scale=_FORWARD_PUSH_SCALE)
        return obs

    def _run_symbolic_inverse(self, obs):
        """Pick at post-push location, lift, carry to above init_pose, descend
        to handoff height (init_z + 5cm + ±2cm xy perturbation, cube held).
        Steps:
          A. Lift EEF straight up with gripper still closed (breaks contact)
          B. Open gripper while up high
          C. Re-read cube position fresh
          D. Move EEF above cube
          E. Descend to grasp height
          F. Close gripper
          G. Lift while held (cube comes up)
          H. Move xy to above init_pose
          I. Descend to handoff (with perturbation)
        """
        # A. Lift EEF straight up
        tcp = obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
        up = tcp.copy(); up[2] += 0.10
        obs = _step_toward_scaled(self._env, obs, up, 20, 0.012, gripper_cmd=-1.0, scale=1.0)
        # B. Open gripper while up high
        obs = _step_in_place(self._env, obs, 8, gripper_cmd=1.0)
        # C. Re-read cube position
        cube_now = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()
        # D. Move above cube
        above_cube = cube_now.copy(); above_cube[2] += 0.10
        obs = _step_toward_scaled(self._env, obs, above_cube, 20, 0.012, gripper_cmd=1.0, scale=1.0)
        # E. Descend to grasp height
        obs = _step_toward_scaled(self._env, obs, cube_now.copy(), 20, 0.012, gripper_cmd=1.0, scale=1.0)
        # F. Close gripper
        obs = _step_in_place(self._env, obs, 10, gripper_cmd=-1.0)
        # G. Lift while held
        lift_h = cube_now.copy(); lift_h[2] += 0.15
        obs = _step_toward_scaled(self._env, obs, lift_h, 15, 0.02, gripper_cmd=-1.0, scale=1.0)
        # H. Move xy to above init_pose at lift height
        over_init = np.array([self.init_pos[0], self.init_pos[1], lift_h[2]], dtype=np.float32)
        obs = _step_toward_scaled(self._env, obs, over_init, 25, 0.012, gripper_cmd=-1.0, scale=1.0)
        # I. Descend to handoff with random xy perturbation (per-episode via np_random)
        if self.perturbation_range_m > 0.0:
            dx = float(self.np_random.uniform(-self.perturbation_range_m, self.perturbation_range_m))
            dy = float(self.np_random.uniform(-self.perturbation_range_m, self.perturbation_range_m))
        else:
            dx = dy = 0.0
        self._last_perturbation = np.array([dx, dy], dtype=np.float32)
        handoff = self.init_pos.copy()
        handoff[0] += dx; handoff[1] += dy; handoff[2] += 0.05
        obs = _step_toward_scaled(self._env, obs, handoff, 20, 0.012, gripper_cmd=-1.0, scale=1.0)
        return obs

    # --- gym API ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        obs, _ = self._env.reset(seed=seed)
        # Two clearing steps (mirrors PickCube demo; mostly defensive)
        for _ in range(2):
            obs, *_ = self._env.step(torch.tensor(np.array([0., 0., 0., 1.], dtype=np.float32)))

        # Record per-episode init_pose (inverse target = where cube starts)
        self.init_pos = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()
        init_pose = demo.Pose(position=self.init_pos.astype(np.float32), quat_xyzw=_IDENTITY_QUAT)
        self.at_pose_pred = demo.AtPosePredicate(
            "cube", target_pose=init_pose,
            distance_threshold=self._current_tolerance, temperature=self.atpose_temperature,
        )
        self.at_pose_eval = demo.AtPosePredicate(
            "cube", target_pose=init_pose,
            distance_threshold=_CURRICULUM_END_TOL, temperature=0.005,
        )

        obs = self._run_forward_push(obs)
        obs = self._run_symbolic_inverse(obs)
        self._last_obs = obs
        self._step_count = 0
        return self._encode(obs), {
            "phase": "rl_start", "seed": seed,
            "init_pose": self.init_pos.tolist(),
            "perturbation_xy": self._last_perturbation.tolist(),
            "current_tolerance_m": self._current_tolerance,
        }

    def step(self, action):
        # Phase 2 action capping (xyz scaled by 0.2 — gripper unscaled)
        action_np = np.asarray(action, dtype=np.float32).copy()
        action_np[:3] *= self.action_scale_xyz
        obs, _r, _term, _trunc, info = self._env.step(torch.tensor(action_np))
        self._last_obs = obs
        self._step_count += 1

        scene = demo._obs_to_scene(obs, self.regions)
        residual_score = float(self.at_pose_pred.evaluate(scene).score)
        cube_pos = scene.objects["cube"].pose.position
        distance = float(np.linalg.norm(cube_pos - self.init_pos))

        # Phase 2 hybrid reward
        reward = residual_score - self.distance_penalty * distance
        terminated = residual_score >= self.success_threshold
        truncated = self._step_count >= self.max_steps
        return self._encode(obs), reward, terminated, truncated, {
            "v_residual": residual_score, "distance": distance, "step": self._step_count
        }

    def _encode(self, obs) -> np.ndarray:
        cube_pos = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy()
        tcp_pos = obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
        qpos = obs["agent"]["qpos"].squeeze().cpu().numpy()
        gw = float(qpos[-2] + qpos[-1])
        return np.concatenate([cube_pos, tcp_pos, [gw], self.init_pos]).astype(np.float32)

    def close(self):
        self._env.close()


# ── Training: same Phase 2 setup but uses PushCubeRecoveryEnv ───────────────


def train(total_timesteps: int, checkpoint_path: Path, best_checkpoint_path: Path
          ) -> tuple[SAC, demo.CheckpointAndLogCallback, PushCubeRecoveryEnv]:
    env = PushCubeRecoveryEnv(
        max_steps=20,
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
    print(f"Training SAC for {total_timesteps} timesteps (PushCube-v1, Phase 2 hybrid reward)...")
    print(f"  push_displacement={_PUSH_DISPLACEMENT_M*100:.1f}cm  push_scale={_FORWARD_PUSH_SCALE}  "
          f"action_scale={_ACTION_SCALE_XYZ}  tol curriculum "
          f"{_CURRICULUM_START_TOL*1000:.0f}->{_CURRICULUM_END_TOL*1000:.0f}mm "
          f"over first {_CURRICULUM_SCHEDULE_STEPS} steps")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    print(f"Training done. {len(log_cb.episode_returns)} episodes completed. "
          f"Best rolling V={log_cb.best_rolling_v:.4f} at step {log_cb.best_at_step}.")
    return model, log_cb, env


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "planrob_inverse_rl_pushcube_model_final.zip"
    best_path = out_dir / "planrob_inverse_rl_pushcube_model_best.zip"
    curve_path = out_dir / "planrob_inverse_rl_pushcube_curve.png"
    json_path = out_dir / "planrob_inverse_rl_pushcube_demo.json"

    # Pre-training baselines on the same 10 perturbation seeds
    base_env = PushCubeRecoveryEnv(
        max_steps=20, atpose_tolerance=_CURRICULUM_END_TOL,
        action_scale_xyz=_ACTION_SCALE_XYZ, perturbation_range_m=_PERTURBATION_RANGE_M,
        push_displacement_m=_PUSH_DISPLACEMENT_M,
    )
    base_env.set_curriculum_tolerance(_CURRICULUM_END_TOL)
    print(f"Evaluating symbolic-only baseline on {len(_EVAL_SEEDS)} seeds...")
    sym = demo.evaluate_symbolic_only(base_env, _EVAL_SEEDS)
    print(f"  symbolic_only:  dist {sym['distance_mm_mean']:.1f}±{sym['distance_mm_std']:.1f}mm  "
          f"V {sym['v_after_release_mean']:.3f}±{sym['v_after_release_std']:.3f}  "
          f"success@1cm {sym['success_rate_at_1cm']:.1%}")
    print(f"Evaluating random-RL baseline on {len(_EVAL_SEEDS)} seeds...")
    rnd = demo.evaluate_random_policy(base_env, _EVAL_SEEDS)
    print(f"  random_rl:      dist {rnd['distance_mm_mean']:.1f}±{rnd['distance_mm_std']:.1f}mm  "
          f"V {rnd['v_after_release_mean']:.3f}±{rnd['v_after_release_std']:.3f}  "
          f"success@1cm {rnd['success_rate_at_1cm']:.1%}")
    base_env.close()

    # Train
    final_model, callback, env = train(1_000_000, ckpt_path, best_path)
    final_model.save(ckpt_path)
    demo.plot_curve(callback, curve_path)

    env.set_curriculum_tolerance(_CURRICULUM_END_TOL)

    print(f"Evaluating trained SAC (final) on {len(_EVAL_SEEDS)} seeds...")
    final_eval = demo.evaluate_trained_policy(env, final_model, _EVAL_SEEDS)
    final_eval["label"] = "trained_sac_pushcube_final"
    print(f"  trained_final:  dist {final_eval['distance_mm_mean']:.1f}±{final_eval['distance_mm_std']:.1f}mm  "
          f"V {final_eval['v_after_release_mean']:.3f}±{final_eval['v_after_release_std']:.3f}  "
          f"success@1cm {final_eval['success_rate_at_1cm']:.1%}")

    if best_path.exists():
        best_model = SAC.load(best_path, env=env)
        print(f"Evaluating trained SAC (best, step {callback.best_at_step}) on {len(_EVAL_SEEDS)} seeds...")
        best_eval = demo.evaluate_trained_policy(env, best_model, _EVAL_SEEDS)
        best_eval["label"] = "trained_sac_pushcube_best"
        print(f"  trained_best:   dist {best_eval['distance_mm_mean']:.1f}±{best_eval['distance_mm_std']:.1f}mm  "
              f"V {best_eval['v_after_release_mean']:.3f}±{best_eval['v_after_release_std']:.3f}  "
              f"success@1cm {best_eval['success_rate_at_1cm']:.1%}")
    else:
        best_eval = None
    env.close()

    payload = {
        "phase": "pushcube",
        "phase_description": "PushCube-v1 forward push + pick-and-place inverse + Phase 2 RL",
        "sac_seed": _SAC_SEED,
        "eval_seeds": _EVAL_SEEDS,
        "baseline_symbolic_only": sym,
        "baseline_random_rl": rnd,
        "trained_final": final_eval,
        "trained_best": best_eval,
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
            "total_timesteps": 1_000_000,
            "max_steps_per_episode": 20,
            "push_displacement_m": _PUSH_DISPLACEMENT_M,
            "push_action_scale": _FORWARD_PUSH_SCALE,
            "atpose_tolerance_curriculum_start_m": _CURRICULUM_START_TOL,
            "atpose_tolerance_curriculum_end_m": _CURRICULUM_END_TOL,
            "atpose_tolerance_curriculum_schedule_steps": _CURRICULUM_SCHEDULE_STEPS,
            "atpose_temperature_train": 0.05,
            "atpose_temperature_eval": 0.005,
            "buffer_size": 100_000, "batch_size": 256, "gamma": 0.99,
            "net_arch": [256, 256],
            "reward": "V_residual_sigmoid - 2.0 * distance(cube, init_pose)",
            "distance_penalty_coef": 2.0,
            "success_threshold": 0.90,
            "action_scale_xyz": _ACTION_SCALE_XYZ,
            "handoff_perturbation_range_m": _PERTURBATION_RANGE_M,
            "release_motion": "slow_open_v1 (15 steps cmd=0.3 + 3 settle)",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
