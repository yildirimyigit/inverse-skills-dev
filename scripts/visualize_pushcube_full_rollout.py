"""Visualize the full PushCube inverse pipeline: forward + symbolic inverse + RL + release.

Mirrors the structure of `visualize_pushcube_forward_symbolic.py` so the
forward push and symbolic inverse phases are byte-identical, then extends
the rollout with two more scripted-or-policy-driven phases:

  4. RL residual correction (trained SAC policy)
  5. release + retract (scripted slow-open then lift-clear)

Run:
    python scripts/visualize_pushcube_full_rollout.py \
        --model artifacts/out/<run_id>/planrob_inverse_rl_pushcube_full_model_best.zip \
        --seed 1000

Outputs (per seed) under the chosen --out-dir:
    planrob_pushcube_full_rollout_seed<seed>.mp4
    planrob_pushcube_full_rollout_seed<seed>.json
    planrob_pushcube_full_rollout_seed<seed>_<still>.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import warnings
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from gymnasium.utils import seeding

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import mani_skill.envs  # noqa: F401  (registers PushCube)

from stable_baselines3 import SAC

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None


_SPEC_FULL = importlib.util.spec_from_file_location(
    "pushcube_full", "scripts/planrob_inverse_rl_pushcube_full_demo_2d_action.py"
)
pushcube_full = importlib.util.module_from_spec(_SPEC_FULL)
_SPEC_FULL.loader.exec_module(pushcube_full)
demo = pushcube_full.demo
pc = pushcube_full.pc

from inverse_skills.predicates import (  # noqa: E402
    AtPosePredicate,
    GripperOpenPredicate,
    TcpNearObjectPredicate,
)


# ── Helpers (mirrors visualize_pushcube_forward_symbolic.py) ────────────────


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _vec3_from_obs(obs: dict, key: str) -> np.ndarray:
    return _to_numpy(obs["extra"][key]).squeeze()[:3].astype(np.float32)


def _gripper_width(obs: dict) -> float:
    qpos = _to_numpy(obs["agent"]["qpos"]).squeeze()
    return float(qpos[-2] + qpos[-1])


def _is_grasped(obs: dict) -> bool | None:
    raw = obs["extra"].get("is_grasped")
    if raw is None:
        return None
    return bool(_to_numpy(raw).squeeze().item())


def _render_frame(env) -> np.ndarray:
    frame = env.render()
    if isinstance(frame, dict):
        frame = frame.get("rgb_array", next(iter(frame.values())))
    frame = _to_numpy(frame)
    while frame.ndim > 3:
        frame = frame[0]
    if frame.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(frame)) <= 1.0 else 1.0
        frame = np.clip(frame * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)


def _annotate(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    if cv2 is None:
        return frame
    out = frame.copy()
    line_height = 24
    pad = 10
    box_h = pad * 2 + line_height * len(lines)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (out.shape[1], box_h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.58, out, 0.42, 0, out)
    for i, line in enumerate(lines):
        y = pad + 18 + i * line_height
        cv2.putText(
            out, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (245, 245, 245), 1, cv2.LINE_AA,
        )
    return out


# ── Visualizer ──────────────────────────────────────────────────────────────


class PushCubeFullRolloutVisualizer:
    def __init__(
        self,
        seed: int,
        out_dir: Path,
        fps: int,
        frame_stride: int,
        overlay: bool,
        push_displacement_m: float,
        use_env_goal_for_push: bool,
        perturbation_range_m: float,
        model: SAC,
    ) -> None:
        self.seed = seed
        self.out_dir = out_dir
        self.fps = fps
        self.frame_stride = max(1, frame_stride)
        self.overlay = overlay
        self.push_displacement_m = push_displacement_m
        self.use_env_goal_for_push = use_env_goal_for_push
        self.perturbation_range_m = perturbation_range_m
        self.model = model

        self.env = gym.make(
            "PushCube-v1",
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pos",
            render_mode="rgb_array",
            max_episode_steps=600,
        )
        self.rng, _ = seeding.np_random(seed)
        self.obs = None

        # Per-rollout state
        self.init_pos = np.zeros(3, dtype=np.float32)
        self.handoff_perturbation = np.zeros(2, dtype=np.float32)
        self.forward_goal_pos: np.ndarray | None = None
        self.forward_push_displacement_m = float(push_displacement_m)
        self.episode_tolerance = float(pushcube_full._CURRICULUM_END_TOL)
        self.action_scale_xyz = float(pushcube_full._ACTION_SCALE_XYZ)

        self.regions = {
            "table_surface": demo.Region(
                "table_surface",
                pushcube_full._TABLE_LOWER,
                pushcube_full._TABLE_UPPER,
            )
        }

        # Non-mutating predicates (set once)
        self.gripper_open_train = GripperOpenPredicate(
            min_width=pushcube_full._GRIPPER_OPEN_MIN_WIDTH,
            temperature=pushcube_full._GRIPPER_OPEN_TEMP_TRAIN,
        )
        self.gripper_open_eval = GripperOpenPredicate(
            min_width=pushcube_full._GRIPPER_OPEN_MIN_WIDTH,
            temperature=pushcube_full._GRIPPER_OPEN_TEMP_EVAL,
        )
        self.tcp_near_pred = TcpNearObjectPredicate(
            object_name="cube",
            distance_threshold=pushcube_full._TCP_NEAR_THRESHOLD_M,
            temperature=pushcube_full._TCP_NEAR_TEMP,
        )

        # Predicates depending on init_pos / forward_goal — built after reset
        self.at_pose_pred: AtPosePredicate | None = None
        self.at_pose_eval: AtPosePredicate | None = None
        self.at_pose_forward_goal_pred: AtPosePredicate | None = None

        self.sim_step = 0
        self.frame_index = 0
        self.frames: list[np.ndarray] = []
        self.trace: list[dict] = []
        self.stills: dict[str, np.ndarray] = {}
        self.phase_counts: dict[str, int] = {}
        self.rl_terminated = False
        self.rl_steps = 0

    def close(self) -> None:
        self.env.close()

    def run(self) -> dict:
        # Mirror the env's validity-gated retry: re-roll the seed if either
        # the forward push or the symbolic inverse fails its tolerance check.
        # Frames from failed attempts are discarded so the saved video shows
        # only the clean, accepted attempt.
        explicit_seed = self.seed
        accepted_seed: int | None = None
        forward_err = float("nan")
        symbolic_err = float("nan")
        attempts = 0

        for attempt in range(pushcube_full._MAX_SCRIPTED_ATTEMPTS):
            attempts = attempt + 1
            try_seed = (explicit_seed if attempt == 0
                        else int(self.rng.integers(0, 2**31 - 1)))
            snapshot = self._snapshot_buffers()

            forward_err, symbolic_err, fwd_ok, sym_ok = self._attempt_scripted_phases(try_seed)
            if fwd_ok and sym_ok:
                accepted_seed = try_seed
                break

            # Reject this attempt: roll back captured frames and try again.
            self._restore_buffers(snapshot)

        if accepted_seed is None:
            raise RuntimeError(
                f"visualize_pushcube_full_rollout: could not find a valid "
                f"scripted setup after {pushcube_full._MAX_SCRIPTED_ATTEMPTS} attempts "
                f"(last forward_err={forward_err*1000:.1f}mm, symbolic_err={symbolic_err*1000:.1f}mm)"
            )

        self._scripted_attempts = attempts
        self._scripted_accepted_seed = accepted_seed
        self._scripted_forward_err = forward_err
        self._scripted_symbolic_err = symbolic_err

        self._run_rl_residual()
        self._capture("after_rl", force=True, still="after_rl")

        self._run_release()
        self._capture("after_release", force=True, still="after_release")

        return self._write_outputs()

    def _attempt_scripted_phases(self, try_seed: int) -> tuple[float, float, bool, bool]:
        """One full scripted-prefix attempt (env reset → forward push →
        symbolic inverse) with frame capture. Returns the per-phase residual
        errors and validity flags. Caller is responsible for restoring buffer
        state on failure."""
        self.obs, _ = self.env.reset(seed=try_seed)
        self._capture("initial", force=True, still="initial")

        for _ in range(2):
            self._step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), "clear_gripper")

        self.init_pos = _vec3_from_obs(self.obs, "obj_pose").copy()
        self._build_init_dependent_predicates()
        self._capture("source_recorded", force=True)

        self._run_forward_push()
        self._capture("after_forward_push", force=True, still="after_forward_push")

        cube_after_forward = _vec3_from_obs(self.obs, "obj_pose")
        forward_target_xy = np.array(
            [self.init_pos[0] + self.forward_push_displacement_m, self.init_pos[1]],
            dtype=np.float32,
        )
        forward_err = float(np.linalg.norm(cube_after_forward[:2] - forward_target_xy))
        forward_tol = max(
            pushcube_full._FORWARD_PUSH_TOLERANCE_MIN_M,
            pushcube_full._FORWARD_PUSH_TOLERANCE_FRAC * abs(self.forward_push_displacement_m),
        )
        forward_ok = forward_err < forward_tol
        if not forward_ok:
            return forward_err, float("nan"), False, False

        # Build the negated-add-effect predicate now that goal is fixed.
        forward_goal_pos = self.init_pos.copy()
        forward_goal_pos[0] += float(self.forward_push_displacement_m)
        self.forward_goal_pos_for_fence = forward_goal_pos
        self.at_pose_forward_goal_pred = AtPosePredicate(
            "cube",
            target_pose=demo.Pose(
                position=forward_goal_pos.astype(np.float32),
                quat_xyzw=pushcube_full._IDENTITY_QUAT,
            ),
            distance_threshold=self.episode_tolerance,
            temperature=0.015,
        )

        self._run_symbolic_inverse()
        self._capture("after_symbolic_inverse", force=True, still="after_symbolic_inverse")

        cube_after_symbolic = _vec3_from_obs(self.obs, "obj_pose")
        handoff_target = self.init_pos + np.array(
            [self.handoff_perturbation[0], self.handoff_perturbation[1], 0.005],
            dtype=np.float32,
        )
        symbolic_err = float(np.linalg.norm(cube_after_symbolic - handoff_target))
        symbolic_ok = symbolic_err < pushcube_full._SYMBOLIC_HANDOFF_TOLERANCE_M
        return forward_err, symbolic_err, True, symbolic_ok

    def _snapshot_buffers(self) -> dict:
        return {
            "frames": list(self.frames),
            "trace": list(self.trace),
            "stills": dict(self.stills),
            "phase_counts": dict(self.phase_counts),
            "sim_step": int(self.sim_step),
            "frame_index": int(self.frame_index),
            "handoff_perturbation": np.array(self.handoff_perturbation, copy=True),
            "forward_goal_pos": (None if self.forward_goal_pos is None
                                  else np.array(self.forward_goal_pos, copy=True)),
            "forward_push_displacement_m": float(self.forward_push_displacement_m),
        }

    def _restore_buffers(self, snap: dict) -> None:
        self.frames = snap["frames"]
        self.trace = snap["trace"]
        self.stills = snap["stills"]
        self.phase_counts = snap["phase_counts"]
        self.sim_step = snap["sim_step"]
        self.frame_index = snap["frame_index"]
        self.handoff_perturbation = snap["handoff_perturbation"]
        self.forward_goal_pos = snap["forward_goal_pos"]
        self.forward_push_displacement_m = snap["forward_push_displacement_m"]

    # --- predicates that depend on init_pos --------------------------------

    def _build_init_dependent_predicates(self) -> None:
        init_pose = demo.Pose(
            position=self.init_pos.astype(np.float32),
            quat_xyzw=pushcube_full._IDENTITY_QUAT,
        )
        self.at_pose_pred = AtPosePredicate(
            "cube", target_pose=init_pose,
            distance_threshold=self.episode_tolerance, temperature=0.015,
        )
        self.at_pose_eval = AtPosePredicate(
            "cube", target_pose=init_pose,
            distance_threshold=pushcube_full._CURRICULUM_END_TOL, temperature=0.005,
        )

    # --- generic step + capture (mirrors symbolic visualizer) --------------

    def _step(self, action: np.ndarray, phase: str) -> None:
        self.obs, *_ = self.env.step(torch.tensor(action, dtype=torch.float32))
        self.sim_step += 1
        self.phase_counts[phase] = self.phase_counts.get(phase, 0) + 1
        self._capture(phase)

    def _capture(self, phase: str, force: bool = False, still: str | None = None) -> None:
        if not force and self.sim_step % self.frame_stride != 0:
            return

        cube = _vec3_from_obs(self.obs, "obj_pose")
        tcp = _vec3_from_obs(self.obs, "tcp_pose")
        dist = float(np.linalg.norm(cube - self.init_pos)) if np.any(self.init_pos) else 0.0
        gripper_width = _gripper_width(self.obs)
        grasped = _is_grasped(self.obs)

        frame = _render_frame(self.env)
        if self.overlay:
            phase_label = phase if len(phase) <= 36 else f"{phase[:33]}..."
            frame = _annotate(frame, [
                f"seed={self.seed}  step={self.sim_step:03d}  {phase_label}",
                f"cube-source dist={dist * 1000.0:5.1f} mm",
                f"gripper={gripper_width * 1000.0:4.1f} mm  grasped={grasped if grasped is not None else 'n/a'}",
            ])
        self.frames.append(frame)
        self.trace.append({
            "frame_index": self.frame_index,
            "sim_step": self.sim_step,
            "phase": phase,
            "cube_pos": cube.tolist(),
            "tcp_pos": tcp.tolist(),
            "distance_to_source_m": dist,
            "gripper_width_m": gripper_width,
            "is_grasped": grasped,
        })
        if still is not None:
            self.stills[still] = frame.copy()
        self.frame_index += 1

    def _step_toward_scaled(
        self,
        target_xyz: np.ndarray,
        n_steps: int,
        tol: float,
        gripper_cmd: float,
        scale: float,
        phase: str,
    ) -> None:
        for _ in range(n_steps):
            tcp = _vec3_from_obs(self.obs, "tcp_pose")
            delta = (target_xyz - tcp).astype(np.float64)
            norm = float(np.linalg.norm(delta))
            if norm < tol:
                break
            action = np.zeros(4, dtype=np.float32)
            action[:3] = (np.clip(delta / max(norm, 1e-6), -1.0, 1.0) * scale).astype(np.float32)
            action[3] = gripper_cmd
            self._step(action, phase)

    def _step_in_place(self, n_steps: int, gripper_cmd: float, phase: str) -> None:
        action = np.array([0.0, 0.0, 0.0, gripper_cmd], dtype=np.float32)
        for _ in range(n_steps):
            self._step(action, phase)

    # --- forward push (identical to visualize_pushcube_forward_symbolic) --

    def _run_forward_push(self) -> None:
        cube_init = _vec3_from_obs(self.obs, "obj_pose").copy()
        self.forward_goal_pos = None
        self.forward_push_displacement_m = float(self.push_displacement_m)
        goal_pos_tensor = self.obs.get("extra", {}).get("goal_pos")
        if self.use_env_goal_for_push and goal_pos_tensor is not None:
            self.forward_goal_pos = _to_numpy(goal_pos_tensor).squeeze()[:3].astype(np.float32).copy()
            self.forward_push_displacement_m = float(self.forward_goal_pos[0] - cube_init[0])

        behind_push = cube_init.copy()
        behind_push[0] -= 0.06
        behind_push[2] = cube_init[2] + 0.005
        self._step_toward_scaled(
            behind_push, 30, 0.012, gripper_cmd=-1.0, scale=1.0,
            phase="forward_lower_to_push_height",
        )

        push_tgt = cube_init.copy()
        push_tgt[0] += self.forward_push_displacement_m
        push_tgt[2] += 0.005
        self._step_toward_scaled(
            push_tgt, 30, 0.005, gripper_cmd=-1.0,
            scale=pushcube_full._FORWARD_PUSH_SCALE,
            phase="forward_push_plus_x",
        )
        tcp = _vec3_from_obs(self.obs, "tcp_pose")
        up = tcp.copy()
        up[2] += 0.10
        self._step_toward_scaled(
            up, 20, 0.012, gripper_cmd=-1.0, scale=1.0,
            phase="symbolic_lift_break_contact",
        )

        self._step_in_place(8, gripper_cmd=1.0, phase="symbolic_open_gripper")

    # --- symbolic inverse (identical to forward_symbolic visualizer) ------

    def _run_symbolic_inverse(self) -> None:
        cube_now = _vec3_from_obs(self.obs, "obj_pose").copy()
        # EEF-only motion (no cube held yet) — full scale for speed.
        above_cube = cube_now.copy()
        above_cube[2] += 0.10
        self._step_toward_scaled(
            above_cube, 20, 0.012, gripper_cmd=1.0, scale=1.0,
            phase="symbolic_move_above_post_push_cube",
        )

        self._step_toward_scaled(
            cube_now.copy(), 20, 0.012, gripper_cmd=1.0, scale=1.0,
            phase="symbolic_descend_to_grasp",
        )

        # Close + dwell so the grasp force settles before any motion.
        self._step_in_place(10, gripper_cmd=-1.0, phase="symbolic_close_gripper")
        self._step_in_place(5, gripper_cmd=-1.0, phase="symbolic_grasp_dwell")

        # Cube-held motion: gentler scale=0.5 with larger step budgets to
        # keep accelerations low (less cube slip during the long carry).
        lift_h = cube_now.copy()
        lift_h[2] += 0.15
        self._step_toward_scaled(
            lift_h, 25, 0.02, gripper_cmd=-1.0, scale=0.5,
            phase="symbolic_lift_held_cube",
        )

        over_init = np.array([self.init_pos[0], self.init_pos[1], lift_h[2]], dtype=np.float32)
        self._step_toward_scaled(
            over_init, 80, 0.012, gripper_cmd=-1.0, scale=0.5,
            phase="symbolic_carry_over_source",
        )

        if self.perturbation_range_m > 0.0:
            dx = float(self.rng.uniform(-self.perturbation_range_m, self.perturbation_range_m))
            dy = float(self.rng.uniform(-self.perturbation_range_m, self.perturbation_range_m))
        else:
            dx = dy = 0.0
        self.handoff_perturbation = np.array([dx, dy], dtype=np.float32)

        handoff = self.init_pos.copy()
        handoff[0] += dx
        handoff[1] += dy
        handoff[2] += 0.005
        self._step_toward_scaled(
            handoff, 30, 0.012, gripper_cmd=-1.0, scale=0.5,
            phase="symbolic_descend_to_handoff",
        )

        self._step_in_place(10, gripper_cmd=1.0, phase="symbolic_release_at_handoff")
        self._step_in_place(10, gripper_cmd=1.0, phase="symbolic_settle_at_handoff")

    # --- RL residual correction --------------------------------------------

    def _encode_predicate_grounded(self) -> np.ndarray:
        """Mirror full.PushCubeRecoveryFullEnv._encode_predicate_grounded so
        the trained policy sees the exact 12-D vector it was trained on."""
        cube_pos = _vec3_from_obs(self.obs, "obj_pose")
        tcp_pos = _vec3_from_obs(self.obs, "tcp_pose")
        gw = _gripper_width(self.obs)
        scene = demo._obs_to_scene(self.obs, self.regions)
        at_pose_result = self.at_pose_pred.evaluate(scene)
        gripper_result = self.gripper_open_train.evaluate(scene)
        scale = max(self.episode_tolerance, 1e-6)
        goal_minus_cube = (self.init_pos - cube_pos) / scale
        cube_minus_tcp = (cube_pos - tcp_pos) / scale
        cube_goal_dist = float(np.linalg.norm(cube_pos - self.init_pos)) / scale
        cube_fwd_goal_dist = (
            float(np.linalg.norm(cube_pos - self.forward_goal_pos_for_fence)) / scale
        )
        normalized_at_pose_margin = float(at_pose_result.margin) / scale
        return np.concatenate([
            goal_minus_cube.astype(np.float32),
            cube_minus_tcp.astype(np.float32),
            [gw / pushcube_full._MAX_GRIPPER_WIDTH_M],
            [float(at_pose_result.score)],
            [normalized_at_pose_margin],
            [cube_goal_dist],
            [float(gripper_result.score)],
            [cube_fwd_goal_dist],
        ]).astype(np.float32)

    def _run_rl_residual(self) -> None:
        max_steps = int(pushcube_full._MAX_STEPS)
        success_threshold = 0.50
        for _ in range(max_steps):
            obs_vec = self._encode_predicate_grounded()
            action_norm, _ = self.model.predict(obs_vec, deterministic=True)
            action_norm = np.asarray(action_norm, dtype=np.float32).reshape(-1)
            maniskill_action = np.array([
                action_norm[0] * self.action_scale_xyz,
                action_norm[1] * self.action_scale_xyz,
                action_norm[2] * self.action_scale_xyz,
                action_norm[3],
            ], dtype=np.float32)
            self._step(maniskill_action, "rl_residual_correction")
            self.rl_steps += 1

            scene = demo._obs_to_scene(self.obs, self.regions)
            v_atpose = float(self.at_pose_pred.evaluate(scene).score)
            v_grip = float(self.gripper_open_train.evaluate(scene).score)
            if v_atpose >= success_threshold and v_grip >= success_threshold:
                self.rl_terminated = True
                break

    # --- scripted release (mirrors the demo's slow_open_v1) ----------------

    def _run_release(self) -> None:
        for _ in range(15):
            self._step(np.array([0.0, 0.0, 0.0, 0.3], dtype=np.float32), "release_slow_open")
        for _ in range(3):
            self._step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), "release_settle")
        tcp = _vec3_from_obs(self.obs, "tcp_pose")
        retract = tcp.copy()
        retract[2] += 0.05
        self._step_toward_scaled(
            retract, 10, 0.01, gripper_cmd=1.0, scale=1.0, phase="release_retract",
        )

    # --- output writer -----------------------------------------------------

    def _write_outputs(self) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        base = self.out_dir / f"planrob_pushcube_full_rollout_seed{self.seed}"
        video_path = base.with_suffix(".mp4")
        json_path = base.with_suffix(".json")

        imageio.mimsave(video_path, self.frames, fps=self.fps, macro_block_size=8)

        still_paths = {}
        for name, frame in self.stills.items():
            path = self.out_dir / f"{base.name}_{name}.png"
            imageio.imwrite(path, frame)
            still_paths[name] = str(path)

        final_cube = _vec3_from_obs(self.obs, "obj_pose")
        final_tcp = _vec3_from_obs(self.obs, "tcp_pose")
        final_distance = float(np.linalg.norm(final_cube - self.init_pos))
        final_scene = demo._obs_to_scene(self.obs, self.regions)
        v_after_release = float(self.at_pose_eval.evaluate(final_scene).score)
        v_gripper_after = float(self.gripper_open_eval.evaluate(final_scene).score)

        payload = {
            "seed": self.seed,
            "scripted_attempts": getattr(self, "_scripted_attempts", 1),
            "scripted_accepted_seed": getattr(self, "_scripted_accepted_seed", self.seed),
            "scripted_forward_err_m": getattr(self, "_scripted_forward_err", float("nan")),
            "scripted_symbolic_err_m": getattr(self, "_scripted_symbolic_err", float("nan")),
            "env": "PushCube-v1",
            "control_mode": "pd_ee_delta_pos",
            "render_mode": "rgb_array",
            "fps": self.fps,
            "frame_stride": self.frame_stride,
            "n_sim_steps": self.sim_step,
            "n_video_frames": len(self.frames),
            "phase_counts": self.phase_counts,
            "rl_steps": self.rl_steps,
            "rl_terminated_early": self.rl_terminated,
            "init_pos": self.init_pos.tolist(),
            "final_cube_pos": final_cube.tolist(),
            "final_tcp_pos": final_tcp.tolist(),
            "final_distance_to_source_m": final_distance,
            "v_at_pose_after_release": v_after_release,
            "v_gripper_open_after_release": v_gripper_after,
            "handoff_perturbation_xy_m": self.handoff_perturbation.tolist(),
            "use_env_goal_for_push": self.use_env_goal_for_push,
            "forward_goal_pos": None if self.forward_goal_pos is None else self.forward_goal_pos.tolist(),
            "forward_push_displacement_m": self.forward_push_displacement_m,
            "fallback_push_displacement_m": self.push_displacement_m,
            "forward_push_scale": pushcube_full._FORWARD_PUSH_SCALE,
            "perturbation_range_m": self.perturbation_range_m,
            "video_path": str(video_path),
            "still_paths": still_paths,
            "trace": self.trace,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload


# ── CLI ─────────────────────────────────────────────────────────────────────


def _seed_list(spec: str) -> list[int]:
    return [int(s.strip()) for s in spec.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Path to SAC .zip checkpoint")
    parser.add_argument("--seeds", type=_seed_list, default=[1000])
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output dir (default: <model_dir>/videos)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument(
        "--no-env-goal-push",
        action="store_true",
        help=(
            "Use the fixed --push-displacement-m for the forward push instead "
            "of ManiSkill's per-episode goal_x. The default uses the env goal "
            "(matches scripts/visualize_pushcube_forward_symbolic.py); flip "
            "this on to keep the forward push at the fixed displacement the "
            "RL policy was trained against."
        ),
    )
    parser.add_argument("--push-displacement-m", type=float,
                        default=pushcube_full._PUSH_DISPLACEMENT_M)
    parser.add_argument("--perturbation-range-m", type=float,
                        default=pushcube_full._PERTURBATION_RANGE_M)
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else args.model.parent / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    # SAC.load wants an env to infer obs/action shapes; use a non-rendering one.
    probe = pushcube_full.PushCubeRecoveryFullEnv(
        atpose_tolerance=pushcube_full._CURRICULUM_END_TOL,
        perturbation_range_m=pushcube_full._PERTURBATION_RANGE_M,
    )
    model = SAC.load(args.model, env=probe)
    probe.close()

    print(f"Rendering {len(args.seeds)} rollout(s) → {out_dir}")
    print(f"{'seed':>6}  {'attempts':>8}  {'rl_steps':>9}  {'rl_done':>7}  "
          f"{'final_mm':>9}  {'V_after':>8}  video")

    for seed in args.seeds:
        viz = PushCubeFullRolloutVisualizer(
            seed=seed,
            out_dir=out_dir,
            fps=args.fps,
            frame_stride=args.frame_stride,
            overlay=not args.no_overlay,
            push_displacement_m=args.push_displacement_m,
            use_env_goal_for_push=not args.no_env_goal_push,
            perturbation_range_m=args.perturbation_range_m,
            model=model,
        )
        try:
            payload = viz.run()
        finally:
            viz.close()
        print(
            f"{payload['seed']:>6d}  {payload['scripted_attempts']:>8d}  "
            f"{payload['rl_steps']:>9d}  {str(payload['rl_terminated_early']):>7}  "
            f"{payload['final_distance_to_source_m']*1000.0:>8.1f}  "
            f"{payload['v_at_pose_after_release']:>8.3f}  {payload['video_path']}"
        )


if __name__ == "__main__":
    main()
