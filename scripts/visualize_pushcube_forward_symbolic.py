"""Visualize the PushCube forward skill and scripted symbolic inverse.

This runner replays only the non-RL part of the Phase 5 PushCube pipeline:

  1. scripted forward push from the initial/source pose to ManiSkill's goal x,
  2. scripted symbolic inverse pick,
  3. scripted carry/descend to the RL handoff above the source pose.

It does not train or evaluate SAC, and it does not modify the Phase 5 demo.

Run:
    python scripts/visualize_pushcube_forward_symbolic.py --seed 1000
    python scripts/visualize_pushcube_forward_symbolic.py --seed 1000 --no-env-goal-push

Outputs:
    artifacts/planrob_pushcube_forward_symbolic_seed1000.mp4
    artifacts/planrob_pushcube_forward_symbolic_seed1000.json
    artifacts/planrob_pushcube_forward_symbolic_seed1000_*.png
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
    import mani_skill.envs  # registers PushCube

try:
    import cv2
except ImportError:  # pragma: no cover - only used if local env lacks opencv
    cv2 = None


_SPEC_FULL = importlib.util.spec_from_file_location(
    "pushcube_full", "scripts/planrob_inverse_rl_pushcube_full_demo.py")
pushcube_full = importlib.util.module_from_spec(_SPEC_FULL)
_SPEC_FULL.loader.exec_module(pushcube_full)


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


class PushCubeSymbolicVisualizer:
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
    ) -> None:
        self.seed = seed
        self.out_dir = out_dir
        self.fps = fps
        self.frame_stride = max(1, frame_stride)
        self.overlay = overlay
        self.push_displacement_m = push_displacement_m
        self.use_env_goal_for_push = use_env_goal_for_push
        self.perturbation_range_m = perturbation_range_m

        self.env = gym.make(
            "PushCube-v1",
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pos",
            render_mode="rgb_array",
            max_episode_steps=600,
        )
        self.rng, _ = seeding.np_random(seed)
        self.obs = None
        self.init_pos = np.zeros(3, dtype=np.float32)
        self.handoff_perturbation = np.zeros(2, dtype=np.float32)
        self.forward_goal_pos: np.ndarray | None = None
        self.forward_push_displacement_m = float(push_displacement_m)
        self.sim_step = 0
        self.frame_index = 0
        self.frames: list[np.ndarray] = []
        self.trace: list[dict] = []
        self.stills: dict[str, np.ndarray] = {}
        self.phase_counts: dict[str, int] = {}

    def close(self) -> None:
        self.env.close()

    def run(self) -> dict:
        self.obs, _ = self.env.reset(seed=self.seed)
        self._capture("initial", force=True, still="initial")

        for _ in range(2):
            self._step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), "clear_gripper")

        self.init_pos = _vec3_from_obs(self.obs, "obj_pose").copy()
        self._capture("source_recorded", force=True)
        self._run_forward_push()
        self._capture("after_forward_push", force=True, still="after_forward_push")
        self._run_symbolic_inverse()
        self._capture("after_symbolic_inverse", force=True, still="after_symbolic_inverse")
        return self._write_outputs()

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

    def _run_forward_push(self) -> None:
        cube_init = _vec3_from_obs(self.obs, "obj_pose").copy()
        self.forward_goal_pos = None
        self.forward_push_displacement_m = float(self.push_displacement_m)
        goal_pos_tensor = self.obs.get("extra", {}).get("goal_pos")
        if self.use_env_goal_for_push and goal_pos_tensor is not None:
            self.forward_goal_pos = _to_numpy(goal_pos_tensor).squeeze()[:3].astype(np.float32).copy()
            self.forward_push_displacement_m = float(self.forward_goal_pos[0])

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

    def _run_symbolic_inverse(self) -> None:
        cube_now = _vec3_from_obs(self.obs, "obj_pose").copy()
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

        self._step_in_place(10, gripper_cmd=-1.0, phase="symbolic_close_gripper")

        lift_h = cube_now.copy()
        lift_h[2] += 0.15
        self._step_toward_scaled(
            lift_h, 15, 0.02, gripper_cmd=-1.0, scale=1.0,
            phase="symbolic_lift_held_cube",
        )

        over_init = np.array([self.init_pos[0], self.init_pos[1], lift_h[2]], dtype=np.float32)
        self._step_toward_scaled(
            over_init, 25, 0.012, gripper_cmd=-1.0, scale=1.0,
            phase="symbolic_carry_over_source",
        )

        # Perturb the handoff target to simulate imperfect symbolic perception. The perturbation is zero if `perturbation_range_m` is zero or negative.
        if self.perturbation_range_m > 0.0:
            dx = float(self.rng.uniform(-self.perturbation_range_m, self.perturbation_range_m))
            dy = float(self.rng.uniform(-self.perturbation_range_m, self.perturbation_range_m))
        else:
            dx = dy = 0.0
        self.handoff_perturbation = np.array([dx, dy], dtype=np.float32)

        handoff = self.init_pos.copy()
        handoff[0] += dx
        handoff[1] += dy
        handoff[2] += 0.005  # small lift to ensure the cube doesn't get dragged on the table
        self._step_toward_scaled(
            handoff, 20, 0.012, gripper_cmd=-1.0, scale=1.0,
            phase="symbolic_descend_to_handoff",
        )

        # Open the gripper to release the cube at the handoff location (`place(source)`)
        self._step_in_place(10, gripper_cmd=1.0, phase="symbolic_release_at_handoff")

        # A few extra steps to settle (for video)
        self._settle_n_steps(10, gripper_cmd=1.0, phase="symbolic_settle_at_handoff")

    def _settle_n_steps(self, n_steps: int, gripper_cmd: float, phase: str) -> None:
        action = np.array([0.0, 0.0, 0.0, gripper_cmd], dtype=np.float32)
        for _ in range(n_steps):
            self._step(action, phase)

    def _write_outputs(self) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        base = self.out_dir / f"planrob_pushcube_forward_symbolic_seed{self.seed}"
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
        payload = {
            "seed": self.seed,
            "env": "PushCube-v1",
            "control_mode": "pd_ee_delta_pos",
            "render_mode": "rgb_array",
            "fps": self.fps,
            "frame_stride": self.frame_stride,
            "n_sim_steps": self.sim_step,
            "n_video_frames": len(self.frames),
            "phase_counts": self.phase_counts,
            "init_pos": self.init_pos.tolist(),
            "final_cube_pos": final_cube.tolist(),
            "final_tcp_pos": final_tcp.tolist(),
            "final_distance_to_source_m": final_distance,
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--no-env-goal-push", action="store_true")
    parser.add_argument("--push-displacement-m", type=float, default=pushcube_full._PUSH_DISPLACEMENT_M)
    parser.add_argument("--perturbation-range-m", type=float, default=pushcube_full._PERTURBATION_RANGE_M)
    args = parser.parse_args()

    visualizer = PushCubeSymbolicVisualizer(
        seed=args.seed,
        out_dir=args.out_dir,
        fps=args.fps,
        frame_stride=args.frame_stride,
        overlay=not args.no_overlay,
        push_displacement_m=args.push_displacement_m,
        use_env_goal_for_push=not args.no_env_goal_push,
        perturbation_range_m=args.perturbation_range_m,
    )
    try:
        payload = visualizer.run()
    finally:
        visualizer.close()

    print(f"Saved video: {payload['video_path']}")
    for name, path in payload["still_paths"].items():
        print(f"Saved still {name}: {path}")
    print(f"Saved trace: {Path(payload['video_path']).with_suffix('.json')}")
    print(
        "Final symbolic handoff: "
        f"dist={payload['final_distance_to_source_m'] * 1000.0:.1f}mm, "
        f"forward_push_dx={payload['forward_push_displacement_m'] * 1000.0:.1f}mm, "
        f"perturb_xy={payload['handoff_perturbation_xy_m']}"
    )


if __name__ == "__main__":
    main()
