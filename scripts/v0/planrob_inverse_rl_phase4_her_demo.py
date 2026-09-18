"""Phase 4: HER + sparse reward {0 on success, -1 otherwise}.

Tests whether Hindsight Experience Replay sidesteps the failure modes of:
 - Phase 2 hybrid sigmoid+penalty (works, plateaus at ~3mm pre-release, 60%
   end-to-end success @ 1cm)
 - Phase 3 pure -distance reward (catastrophic divergence — Q-cliff at the
   termination boundary, no positive anchor for SAC's value function)

HER provides positive value anchors automatically by relabeling failed episode
transitions with the *achieved* goal in place of the original *desired* goal,
so every trajectory contributes useful Q-targets even with a sparse {0, -1}
reward. The reward function depends only on (achieved_goal, desired_goal) so
relabeled rewards are well-defined.

Wraps the existing InverseRecoveryEnv mechanics (forward + symbolic + action
cap + handoff perturbation + curriculum) and only changes:
  - Observation: Dict {observation, achieved_goal, desired_goal}
  - Reward: sparse {0 if d<tol, -1 otherwise}
  - Buffer: HerReplayBuffer with `future` strategy, n_sampled_goal=4

Output:
  artifacts/planrob_inverse_rl_phase4_curve.png
  artifacts/planrob_inverse_rl_phase4_demo.json
  artifacts/planrob_inverse_rl_phase4_model_final.zip
  artifacts/planrob_inverse_rl_phase4_model_best.zip
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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

import mani_skill.envs  # noqa: E402,F401  (registers PickCube)

# Pull shared mechanics from the canonical demo script.
_spec = importlib.util.spec_from_file_location("demo", "scripts/planrob_inverse_rl_demo.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


# ── Phase 4 configuration ─────────────────────────────────────────────────────


_SAC_SEED = demo._SAC_SEED
_EVAL_SEEDS = demo._EVAL_SEEDS
_CURRICULUM_START_TOL = demo._CURRICULUM_START_TOL
_CURRICULUM_END_TOL = demo._CURRICULUM_END_TOL
_CURRICULUM_SCHEDULE_STEPS = demo._CURRICULUM_SCHEDULE_STEPS
_ACTION_SCALE_XYZ = demo._ACTION_SCALE_XYZ
_PERTURBATION_RANGE_M = demo._PERTURBATION_RANGE_M


# ── Goal-conditioned env wrapper for HER ─────────────────────────────────────


class InverseRecoveryGoalEnv(gym.Env):
    """GoalEnv-style wrapper around InverseRecoveryEnv for HER training.

    Observation (Dict):
        observation:    7D — cube_pos(3) + tcp_pos(3) + gripper_width(1)
        achieved_goal:  3D — cube_pos
        desired_goal:   3D — per-episode init_pose (the restoration target)

    Reward: sparse {0 if ||achieved - desired|| < tolerance else -1}
    Termination: when distance < tolerance (so terminal Q anchors at 0).

    The inner env keeps doing forward + symbolic phases at reset, action
    capping in step, and the at_pose_eval predicate for honest evaluation.
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 20,
                 atpose_tolerance: float = _CURRICULUM_START_TOL,
                 action_scale_xyz: float = _ACTION_SCALE_XYZ,
                 perturbation_range_m: float = _PERTURBATION_RANGE_M):
        super().__init__()
        self._inner = demo.InverseRecoveryEnv(
            max_steps=max_steps,
            atpose_tolerance=atpose_tolerance,
            action_scale_xyz=action_scale_xyz,
            perturbation_range_m=perturbation_range_m,
        )
        self._tolerance = float(atpose_tolerance)
        self.observation_space = spaces.Dict({
            "observation":   spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
            "desired_goal":  spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
        self.action_space = self._inner.action_space

    # --- delegated attributes (so existing eval code works unchanged) -------

    @property
    def init_pos(self) -> np.ndarray: return self._inner.init_pos
    @property
    def at_pose_eval(self): return self._inner.at_pose_eval
    @property
    def regions(self): return self._inner.regions
    @property
    def _env(self): return self._inner._env
    @property
    def _last_obs(self): return self._inner._last_obs
    @_last_obs.setter
    def _last_obs(self, v): self._inner._last_obs = v

    def set_curriculum_tolerance(self, value: float) -> None:
        """Curriculum hook — updates BOTH the inner env's predicate tolerance
        and the HER reward/termination tolerance. HerReplayBuffer recomputes
        rewards at sampling time using the current tolerance, so changes here
        propagate to relabeled transitions consistently."""
        self._inner.set_curriculum_tolerance(value)
        self._tolerance = float(value)

    # --- HER's reward function: vectorized over goals ------------------------

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray,
                       _info=None) -> np.ndarray:
        """Sparse reward used by HerReplayBuffer to relabel transitions.

        Shapes:
            achieved_goal, desired_goal: (3,) for single, (n, 3) for batch.
        Returns:
            float scalar or shape (n,) array.
        """
        a = np.asarray(achieved_goal, dtype=np.float32)
        d = np.asarray(desired_goal, dtype=np.float32)
        dist = np.linalg.norm(a - d, axis=-1)
        return -(dist >= self._tolerance).astype(np.float32)

    # --- gym API -------------------------------------------------------------

    def _make_obs(self, flat_obs: np.ndarray) -> dict:
        return {
            "observation":   flat_obs[:7].astype(np.float32),
            "achieved_goal": flat_obs[:3].astype(np.float32),
            "desired_goal":  flat_obs[7:10].astype(np.float32),
        }

    def reset(self, seed=None, options=None):
        flat_obs, info = self._inner.reset(seed=seed, options=options)
        return self._make_obs(flat_obs), info

    def step(self, action):
        # Reuse the inner env's action capping + per-step physics, but override
        # reward and termination to reflect the sparse-HER formulation.
        flat_obs, _r_inner, _term_inner, trunc_inner, info = self._inner.step(action)
        achieved = flat_obs[:3].astype(np.float32)
        desired = flat_obs[7:10].astype(np.float32)
        reward = float(self.compute_reward(achieved, desired, info))
        distance = float(np.linalg.norm(achieved - desired))
        terminated = bool(distance < self._tolerance)
        truncated = bool(trunc_inner)
        info["v_residual"] = info.get("v_residual", 0.0)
        info["distance"] = distance
        return self._make_obs(flat_obs), reward, terminated, truncated, info

    def close(self):
        self._inner.close()


# ── Callbacks (mirror the Phase 2/3 ones, adapted for the GoalEnv) ───────────


class CheckpointAndLogCallback(BaseCallback):
    def __init__(self, checkpoint_path: Path, best_checkpoint_path: Path,
                 checkpoint_every: int = 50_000, log_every: int = 5_000,
                 best_window: int = 50, env: InverseRecoveryGoalEnv | None = None):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.best_checkpoint_path = best_checkpoint_path
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every
        self.best_window = best_window
        self._env_ref = env
        self.episode_returns: list[float] = []
        self.episode_v_finals: list[float] = []
        self.episode_distances: list[float] = []
        self.episode_tolerances: list[float] = []
        self.episode_success: list[int] = []  # 1 if terminated (success), 0 otherwise
        self._episode_v: list[float] = []
        self._episode_dist: list[float] = []
        self._episode_terminated_flag: bool = False
        self._last_log_step = 0
        self._last_ckpt_step = 0
        self.best_rolling_v: float = -float("inf")
        self.best_at_step: int = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        # SB3 reports both terminated and truncated via dones[i]; we track
        # success by checking if distance < tolerance at the final step.
        for info, done in zip(infos, dones):
            if "distance" in info:
                self._episode_v.append(float(info.get("v_residual", 0.0)))
                self._episode_dist.append(float(info["distance"]))
            if done:
                if self._episode_v:
                    self.episode_returns.append(float(np.mean(self._episode_v)))
                    self.episode_v_finals.append(float(self._episode_v[-1]))
                    final_dist = float(self._episode_dist[-1])
                    self.episode_distances.append(final_dist)
                    tol = float(self._env_ref._tolerance) if self._env_ref else float("nan")
                    self.episode_tolerances.append(tol)
                    self.episode_success.append(int(final_dist < tol))
                    self._episode_v = []
                    self._episode_dist = []

        if self.num_timesteps - self._last_log_step >= self.log_every and self.episode_v_finals:
            recent_v = self.episode_v_finals[-50:]
            recent_d = self.episode_distances[-50:]
            recent_succ = self.episode_success[-50:]
            tol_now = self._env_ref._tolerance if self._env_ref else float("nan")
            print(f"  step {self.num_timesteps:>7d}  "
                  f"episodes={len(self.episode_v_finals):>5d}  "
                  f"tol={tol_now*1000:.1f}mm  "
                  f"recent V={np.mean(recent_v):.4f}  "
                  f"recent dist={np.mean(recent_d)*1000:.1f}mm  "
                  f"recent success={np.mean(recent_succ)*100:.0f}%  "
                  f"best_rolling_V={self.best_rolling_v:.4f}@step{self.best_at_step}")
            self._last_log_step = self.num_timesteps

        if self.num_timesteps - self._last_ckpt_step >= self.checkpoint_every:
            self.model.save(self.checkpoint_path)
            self._last_ckpt_step = self.num_timesteps

        if len(self.episode_v_finals) >= self.best_window:
            rolling = float(np.mean(self.episode_v_finals[-self.best_window:]))
            if rolling > self.best_rolling_v:
                self.best_rolling_v = rolling
                self.best_at_step = int(self.num_timesteps)
                self.model.save(self.best_checkpoint_path)
        return True


class CurriculumCallback(BaseCallback):
    def __init__(self, env: InverseRecoveryGoalEnv, start_tol: float, end_tol: float,
                 schedule_steps: int):
        super().__init__()
        self._env_ref = env
        self.start_tol = start_tol
        self.end_tol = end_tol
        self.schedule_steps = schedule_steps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / max(1, self.schedule_steps))
        tol = self.start_tol + (self.end_tol - self.start_tol) * progress
        self._env_ref.set_curriculum_tolerance(tol)
        return True


# ── Training ────────────────────────────────────────────────────────────────


def train(total_timesteps: int, checkpoint_path: Path, best_checkpoint_path: Path
          ) -> tuple[SAC, CheckpointAndLogCallback, InverseRecoveryGoalEnv]:
    env = InverseRecoveryGoalEnv(
        max_steps=20,
        atpose_tolerance=_CURRICULUM_START_TOL,
        action_scale_xyz=_ACTION_SCALE_XYZ,
        perturbation_range_m=_PERTURBATION_RANGE_M,
    )
    model = SAC(
        "MultiInputPolicy",  # required for Dict observation
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs={
            "n_sampled_goal": 4,
            "goal_selection_strategy": "future",
        },
        learning_rate=3e-4, buffer_size=200_000,  # larger buffer for HER (4x relabels)
        learning_starts=1_000, batch_size=256,
        tau=0.005, gamma=0.99,
        policy_kwargs={"net_arch": [256, 256]},
        seed=_SAC_SEED,
    )
    log_cb = CheckpointAndLogCallback(
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        checkpoint_every=50_000, log_every=5_000, best_window=50, env=env,
    )
    curr_cb = CurriculumCallback(
        env, start_tol=_CURRICULUM_START_TOL, end_tol=_CURRICULUM_END_TOL,
        schedule_steps=_CURRICULUM_SCHEDULE_STEPS,
    )
    callbacks = CallbackList([log_cb, curr_cb])
    print(f"Training SAC+HER for {total_timesteps} timesteps (Phase 4)...")
    print(f"  sparse reward {{0 / -1}}, terminate-at-tolerance, n_sampled_goal=4, future strategy")
    print(f"  fixed init_pose, ±{_PERTURBATION_RANGE_M*100:.1f}cm handoff perturbation, "
          f"action_scale={_ACTION_SCALE_XYZ}, tol curriculum "
          f"{_CURRICULUM_START_TOL*1000:.0f}->{_CURRICULUM_END_TOL*1000:.0f}mm "
          f"over first {_CURRICULUM_SCHEDULE_STEPS} steps")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    print(f"Training done. {len(log_cb.episode_returns)} episodes completed. "
          f"Best rolling V={log_cb.best_rolling_v:.4f} at step {log_cb.best_at_step}.")
    return model, log_cb, env


# ── Evaluation reuses the demo's eval episode helper ─────────────────────────


def evaluate_with_seeds(env: InverseRecoveryGoalEnv, model: SAC | None,
                         seeds: list[int], do_rl: bool, label: str) -> dict:
    """Evaluate symbolic-only / random-RL / trained-policy on the goal env."""
    eps = []
    for seed in seeds:
        obs, _info = env.reset(seed=seed)
        rl_steps = 0
        v_after_rl = None
        if do_rl:
            terminated = truncated = False
            while not (terminated or truncated):
                if model is None:
                    action = env.action_space.sample()
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, _r, terminated, truncated, info = env.step(action)
                v_after_rl = info.get("v_residual", v_after_rl)
                rl_steps += 1
        # Pre-release diagnostic
        pre_scene = demo._obs_to_scene(env._last_obs, env.regions)
        v_pre_release_eval = float(env.at_pose_eval.evaluate(pre_scene).score)
        # Slow-open release + measure (canonical Phase 2/3 release)
        measured = demo._release_and_measure(env)
        eps.append({
            "seed": seed,
            "rl_steps": rl_steps,
            "v_after_rl": v_after_rl,
            "v_pre_release_eval": v_pre_release_eval,
            **measured,
        })
    return demo._summarize(label, eps)


# ── Plot ────────────────────────────────────────────────────────────────────


def plot_curve(callback: CheckpointAndLogCallback, out_path: Path) -> None:
    if not callback.episode_returns:
        print("No episodes; skipping plot.")
        return
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.8))
    eps = np.arange(1, len(callback.episode_returns) + 1)

    def _smooth(xs, w=50):
        a = np.asarray(xs, dtype=np.float64)
        if len(a) < w:
            return a
        return np.convolve(a, np.ones(w) / w, mode="valid")

    axes[0].plot(eps, callback.episode_v_finals, color="#3060B0", lw=0.4, alpha=0.4)
    smooth = _smooth(callback.episode_v_finals, 50)
    axes[0].plot(np.arange(len(eps) - len(smooth) + 1, len(eps) + 1),
                 smooth, color="#1f3a73", lw=1.4, label="50-ep moving avg")
    axes[0].axhline(0.90, color="#888888", ls="--", lw=0.8, label="success threshold")
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("final V_residual (eval predicate)")
    axes[0].set_title("Final V_residual per episode (Phase 4: HER + sparse)", fontsize=9)
    axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3); axes[0].set_ylim(-0.05, 1.05)

    axes[1].plot(eps, np.asarray(callback.episode_distances) * 1000, color="#B05030", lw=0.4, alpha=0.4)
    smooth_d = _smooth(callback.episode_distances, 50) * 1000
    axes[1].plot(np.arange(len(eps) - len(smooth_d) + 1, len(eps) + 1),
                 smooth_d, color="#722a18", lw=1.4)
    axes[1].axhline(10, color="#888888", ls="--", lw=0.8, label="1cm tolerance")
    axes[1].set_xlabel("episode"); axes[1].set_ylabel("final cube distance (mm)")
    axes[1].set_title("Final cube distance to init_pose", fontsize=9)
    axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    success_smooth = _smooth(callback.episode_success, 50) * 100
    axes[2].plot(eps, np.asarray(callback.episode_success) * 100,
                 color="#3a7a4a", lw=0.4, alpha=0.4)
    axes[2].plot(np.arange(len(eps) - len(success_smooth) + 1, len(eps) + 1),
                 success_smooth, color="#1e4a2e", lw=1.4)
    axes[2].set_xlabel("episode"); axes[2].set_ylabel("success rate (%, 50-ep moving)")
    axes[2].set_title("Episode success rate (terminated within current tolerance)", fontsize=9)
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim(-5, 105)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "planrob_inverse_rl_phase4_model_final.zip"
    best_path = out_dir / "planrob_inverse_rl_phase4_model_best.zip"
    curve_path = out_dir / "planrob_inverse_rl_phase4_curve.png"
    json_path = out_dir / "planrob_inverse_rl_phase4_demo.json"

    # Baselines on the GoalEnv (same physics, same release motion, fair comparison)
    base_env = InverseRecoveryGoalEnv(
        max_steps=20, atpose_tolerance=_CURRICULUM_END_TOL,
        action_scale_xyz=_ACTION_SCALE_XYZ, perturbation_range_m=_PERTURBATION_RANGE_M,
    )
    base_env.set_curriculum_tolerance(_CURRICULUM_END_TOL)
    print(f"Evaluating symbolic-only baseline on {len(_EVAL_SEEDS)} seeds...")
    sym = evaluate_with_seeds(base_env, model=None, seeds=_EVAL_SEEDS, do_rl=False, label="symbolic_only")
    print(f"  symbolic_only:  dist {sym['distance_mm_mean']:.1f}±{sym['distance_mm_std']:.1f}mm  "
          f"V {sym['v_after_release_mean']:.3f}±{sym['v_after_release_std']:.3f}  "
          f"success@1cm {sym['success_rate_at_1cm']:.1%}")
    print(f"Evaluating random-RL baseline on {len(_EVAL_SEEDS)} seeds...")
    rnd = evaluate_with_seeds(base_env, model=None, seeds=_EVAL_SEEDS, do_rl=True, label="random_rl")
    print(f"  random_rl:      dist {rnd['distance_mm_mean']:.1f}±{rnd['distance_mm_std']:.1f}mm  "
          f"V {rnd['v_after_release_mean']:.3f}±{rnd['v_after_release_std']:.3f}  "
          f"success@1cm {rnd['success_rate_at_1cm']:.1%}")
    base_env.close()

    # Train
    final_model, callback, env = train(1_000_000, ckpt_path, best_path)
    final_model.save(ckpt_path)
    plot_curve(callback, curve_path)

    env.set_curriculum_tolerance(_CURRICULUM_END_TOL)

    print(f"Evaluating trained SAC+HER (final) on {len(_EVAL_SEEDS)} seeds...")
    final_eval = evaluate_with_seeds(env, final_model, _EVAL_SEEDS, do_rl=True, label="trained_sac_her_final")
    print(f"  trained_final:  dist {final_eval['distance_mm_mean']:.1f}±{final_eval['distance_mm_std']:.1f}mm  "
          f"V {final_eval['v_after_release_mean']:.3f}±{final_eval['v_after_release_std']:.3f}  "
          f"success@1cm {final_eval['success_rate_at_1cm']:.1%}")

    if best_path.exists():
        best_model = SAC.load(best_path, env=env)
        print(f"Evaluating trained SAC+HER (best, step {callback.best_at_step}) on {len(_EVAL_SEEDS)} seeds...")
        best_eval = evaluate_with_seeds(env, best_model, _EVAL_SEEDS, do_rl=True, label="trained_sac_her_best")
        print(f"  trained_best:   dist {best_eval['distance_mm_mean']:.1f}±{best_eval['distance_mm_std']:.1f}mm  "
              f"V {best_eval['v_after_release_mean']:.3f}±{best_eval['v_after_release_std']:.3f}  "
              f"success@1cm {best_eval['success_rate_at_1cm']:.1%}")
    else:
        best_eval = None
    env.close()

    payload = {
        "phase": 4,
        "phase_description": "HER + sparse reward {0 if d<tol else -1}, terminate-at-tolerance",
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
            "last50_success_rate": float(np.mean(callback.episode_success[-50:]))
                if callback.episode_success else None,
            "best_rolling_v": float(callback.best_rolling_v),
            "best_rolling_v_at_step": int(callback.best_at_step),
        },
        "config": {
            "algorithm": "SAC + HerReplayBuffer",
            "her_n_sampled_goal": 4,
            "her_strategy": "future",
            "policy": "MultiInputPolicy",
            "total_timesteps": 1_000_000,
            "max_steps_per_episode": 20,
            "atpose_tolerance_curriculum_start_m": _CURRICULUM_START_TOL,
            "atpose_tolerance_curriculum_end_m": _CURRICULUM_END_TOL,
            "atpose_tolerance_curriculum_schedule_steps": _CURRICULUM_SCHEDULE_STEPS,
            "buffer_size": 200_000,
            "batch_size": 256,
            "gamma": 0.99,
            "net_arch": [256, 256],
            "reward": "sparse: 0 if dist<tol else -1",
            "termination": "dist < curriculum_tolerance",
            "action_scale_xyz": _ACTION_SCALE_XYZ,
            "handoff_perturbation_range_m": _PERTURBATION_RANGE_M,
            "release_motion": "slow_open_v1 (15 steps cmd=0.3 + 3 settle)",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {json_path}")


if __name__ == "__main__":
    main()
