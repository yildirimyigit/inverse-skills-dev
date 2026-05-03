"""Re-evaluate the existing Phase 2 trained SAC policy with several release motions.

Phase 2 found that the policy brings the cube to ~2mm precision while held,
but the open-loop release sequence then introduces ~19mm of lateral error.
This script tests four release strategies on the same 10 held-out seeds, using
the saved Phase 2 'trained_final' weights. No retraining.

Each release operates on the post-RL state (cube held precisely at goal) and
ends with the gripper fully open and EEF retracted clear of the cube.

Output:
  artifacts/planrob_inverse_rl_release_eval.json
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

import gymnasium as gym  # noqa: E402
import mani_skill.envs  # noqa: E402,F401  (registers PickCube)

from stable_baselines3 import SAC  # noqa: E402

# Load shared bits from the main demo (env, helpers).
_spec = importlib.util.spec_from_file_location("demo", "scripts/planrob_inverse_rl_demo.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


# ── Release variants ─────────────────────────────────────────────────────────


def _release_v0_original(env) -> "ms_obs":
    """Phase 2 baseline: 6 steps of full open, then retract 5cm up."""
    ms_obs = env._last_obs
    ms_obs = demo._open_gripper(env._env, ms_obs, n_steps=6)
    tcp = ms_obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
    retract = tcp.copy().astype(np.float32); retract[2] += 0.05
    ms_obs = demo._step_toward(env._env, ms_obs, retract, n_steps=10, tol=0.01, gripper_cmd=1.0)
    return ms_obs


def _release_v1_slow_open(env) -> "ms_obs":
    """Slow gripper open with reduced command magnitude. Lower per-step gripper
    velocity → less impulsive lateral finger force on a stationary cube."""
    ms_obs = env._last_obs
    for _ in range(15):
        a = torch.tensor(np.array([0.0, 0.0, 0.0, 0.3], dtype=np.float32))
        ms_obs, *_ = env._env.step(a)
    for _ in range(3):  # finish-open + settle
        a = torch.tensor(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        ms_obs, *_ = env._env.step(a)
    tcp = ms_obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
    retract = tcp.copy().astype(np.float32); retract[2] += 0.05
    ms_obs = demo._step_toward(env._env, ms_obs, retract, n_steps=10, tol=0.01, gripper_cmd=1.0)
    return ms_obs


def _release_v2_pre_lift(env) -> "ms_obs":
    """Pre-release lift: while still holding, raise EEF by 1cm so the cube has
    free vertical clearance. Then open gripper (cube drops 1cm onto table from
    air, no table-friction pinning it during open). Then retract."""
    ms_obs = env._last_obs
    tcp = ms_obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
    lift = tcp.copy().astype(np.float32); lift[2] += 0.01
    ms_obs = demo._step_toward(env._env, ms_obs, lift, n_steps=5, tol=0.003, gripper_cmd=-1.0)
    for _ in range(8):
        a = torch.tensor(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        ms_obs, *_ = env._env.step(a)
    for _ in range(3):
        a = torch.tensor(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        ms_obs, *_ = env._env.step(a)
    tcp = ms_obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
    retract = tcp.copy().astype(np.float32); retract[2] += 0.05
    ms_obs = demo._step_toward(env._env, ms_obs, retract, n_steps=10, tol=0.01, gripper_cmd=1.0)
    return ms_obs


def _release_v3_open_while_lifting(env) -> "ms_obs":
    """Combined open + slow upward motion. Each step the gripper opens slightly
    AND the EEF moves up slightly. By the time the gripper has fully opened
    (~6 steps), the EEF is far enough above the cube that residual finger
    motion no longer reaches it."""
    ms_obs = env._last_obs
    for _ in range(10):  # 10 * dz=0.04 = 4cm slow climb + open
        a = torch.tensor(np.array([0.0, 0.0, 0.04, 1.0], dtype=np.float32))
        ms_obs, *_ = env._env.step(a)
    # Final clear-retract
    tcp = ms_obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
    retract = tcp.copy().astype(np.float32); retract[2] += 0.03
    ms_obs = demo._step_toward(env._env, ms_obs, retract, n_steps=8, tol=0.01, gripper_cmd=1.0)
    return ms_obs


def _release_v4_ultra_slow(env) -> "ms_obs":
    """Ultra-slow open: 25 steps at command 0.2 + 8 settle steps + retract.
    Tests whether the v1 success rate (60%) keeps climbing with more conservative
    timing, or whether residual variance is from a different mechanism."""
    ms_obs = env._last_obs
    for _ in range(25):
        a = torch.tensor(np.array([0.0, 0.0, 0.0, 0.2], dtype=np.float32))
        ms_obs, *_ = env._env.step(a)
    for _ in range(8):
        a = torch.tensor(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        ms_obs, *_ = env._env.step(a)
    tcp = ms_obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
    retract = tcp.copy().astype(np.float32); retract[2] += 0.05
    ms_obs = demo._step_toward(env._env, ms_obs, retract, n_steps=10, tol=0.01, gripper_cmd=1.0)
    return ms_obs


VARIANTS = {
    "v0_original":          _release_v0_original,
    "v1_slow_open":         _release_v1_slow_open,
    "v2_pre_lift":          _release_v2_pre_lift,
    "v3_open_while_lifting": _release_v3_open_while_lifting,
    "v4_ultra_slow":        _release_v4_ultra_slow,
}


# ── Evaluation harness ───────────────────────────────────────────────────────


def evaluate_release(env, model, release_fn, seeds: list[int], label: str) -> dict:
    """Run trained policy on each seed, then apply the given release variant."""
    eps = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, _info = env.step(action)

        # Record pre-release cube position
        pre_scene = demo._obs_to_scene(env._last_obs, env.regions)
        pre_cube = pre_scene.objects["cube"].pose.position
        pre_dist_mm = float(np.linalg.norm(pre_cube - env.init_pos)) * 1000.0
        v_pre = float(env.at_pose_eval.evaluate(pre_scene).score)

        # Custom release
        env._last_obs = release_fn(env)

        # Measure
        post_scene = demo._obs_to_scene(env._last_obs, env.regions)
        post_cube = post_scene.objects["cube"].pose.position
        post_dist_mm = float(np.linalg.norm(post_cube - env.init_pos)) * 1000.0
        v_post = float(env.at_pose_eval.evaluate(post_scene).score)
        eps.append({
            "seed": seed,
            "pre_release_dist_mm": pre_dist_mm,
            "post_release_dist_mm": post_dist_mm,
            "release_delta_mm": post_dist_mm - pre_dist_mm,
            "v_pre_release": v_pre,
            "v_after_release": v_post,
        })

    pre = np.array([e["pre_release_dist_mm"] for e in eps])
    post = np.array([e["post_release_dist_mm"] for e in eps])
    delta = np.array([e["release_delta_mm"] for e in eps])
    n_success = int(np.sum(post < 10.0))
    return {
        "label": label,
        "n_episodes": len(seeds),
        "per_episode": eps,
        "pre_release_dist_mm_mean": float(pre.mean()),
        "pre_release_dist_mm_std": float(pre.std()),
        "post_release_dist_mm_mean": float(post.mean()),
        "post_release_dist_mm_std": float(post.std()),
        "release_delta_mm_mean": float(delta.mean()),
        "release_delta_mm_std": float(delta.std()),
        "v_after_release_mean": float(np.mean([e["v_after_release"] for e in eps])),
        "success_rate_at_1cm": n_success / len(seeds),
    }


def main() -> None:
    out_dir = Path("artifacts")
    model_path = out_dir / "planrob_inverse_rl_phase2_model_final.zip"
    assert model_path.exists(), f"Trained model not found at {model_path}"

    env = demo.InverseRecoveryEnv(
        max_steps=20,
        atpose_tolerance=demo._CURRICULUM_END_TOL,
        action_scale_xyz=demo._ACTION_SCALE_XYZ,
        perturbation_range_m=demo._PERTURBATION_RANGE_M,
    )
    env.set_curriculum_tolerance(demo._CURRICULUM_END_TOL)
    model = SAC.load(model_path, env=env)
    print(f"Loaded {model_path}")

    seeds = demo._EVAL_SEEDS
    results = {}
    print()
    print(f"{'variant':<28} {'pre(mm)':>10} {'post(mm)':>12} {'Δ(mm)':>10} {'V_post':>10} {'success@1cm':>14}")
    print("-" * 88)
    for name, fn in VARIANTS.items():
        r = evaluate_release(env, model, fn, seeds, label=name)
        results[name] = r
        print(f"{name:<28} {r['pre_release_dist_mm_mean']:6.1f}±{r['pre_release_dist_mm_std']:>3.0f}  "
              f"{r['post_release_dist_mm_mean']:6.1f}±{r['post_release_dist_mm_std']:>4.0f}  "
              f"{r['release_delta_mm_mean']:6.1f}±{r['release_delta_mm_std']:>2.0f}  "
              f"{r['v_after_release_mean']:>9.3f}  "
              f"{r['success_rate_at_1cm']:>13.0%}")

    env.close()

    payload = {"variants": results, "eval_seeds": list(seeds)}
    out_path = out_dir / "planrob_inverse_rl_release_eval.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
