"""Visualize the forward push followed by a trajectory-reversal retract.

Motivation visual for the paper's in-class vs inter-class argument: a
non-prehensile push cannot be undone by rewinding the forward trajectory.
This rollout runs the *byte-identical* forward push from
`visualize_pushcube_full_rollout.py`, then performs the *exact reverse* of that
trajectory: close the gripper, lower to just behind the pushed cube, retract -x
back to the start, elevate, and open at the initial pose. Because the closed gripper pushed
from behind (it sits on the -x side of the cube), retracting -x retreats away
from the non-grasped cube instead of carrying it back: the cube stays at the
goal and the world state is *not* restored. That failure is the point.

Style (overlay, fps, resolution, stills) mirrors the full-rollout videos by
subclassing their visualizer; only the post-push phase differs.

Run (from repo root):
    python scripts/visualize_pushcube_forward_retract.py --seeds 1000

Outputs (per seed) under --out-dir:
    planrob_pushcube_forward_retract_seed<seed>.mp4
    planrob_pushcube_forward_retract_seed<seed>.json
    planrob_pushcube_forward_retract_seed<seed>_<still>.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

# This rollout is purely scripted (no RL policy), so it does not need
# stable_baselines3. The reused modules import SAC and subclass BaseCallback at
# import time, however — stub the small surface they touch so this video
# renders without the RL dependency installed.
try:
    import stable_baselines3  # noqa: F401
except ModuleNotFoundError:
    _sb3 = types.ModuleType("stable_baselines3")
    _sb3.__path__ = []  # mark as a package so submodule imports resolve
    _sb3.SAC = object  # name only; never instantiated in this script
    _common = types.ModuleType("stable_baselines3.common")
    _common.__path__ = []
    _callbacks = types.ModuleType("stable_baselines3.common.callbacks")

    class _BaseCallback:  # subclassed at import time by the reused module
        pass

    _callbacks.BaseCallback = _BaseCallback
    _callbacks.CallbackList = object
    _common.callbacks = _callbacks
    _sb3.common = _common
    sys.modules["stable_baselines3"] = _sb3
    sys.modules["stable_baselines3.common"] = _common
    sys.modules["stable_baselines3.common.callbacks"] = _callbacks

# Reuse the full-rollout visualizer (byte-identical forward push + overlay
# style). Loaded by file path to match that module's own cwd assumption.
_SPEC = importlib.util.spec_from_file_location(
    "vizfull", "scripts/visualize_pushcube_full_rollout.py"
)
vizfull = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vizfull)

_vec3_from_obs = vizfull._vec3_from_obs
pushcube_full = vizfull.pushcube_full
demo = vizfull.demo


class PushCubeForwardRetractVisualizer(vizfull.PushCubeFullRolloutVisualizer):
    """Forward push, then retrace the arm to its initial pose (no inverse)."""

    def run(self) -> dict:
        explicit_seed = self.seed
        accepted_seed: int | None = None
        forward_err = float("nan")
        attempts = 0

        for attempt in range(pushcube_full._MAX_SCRIPTED_ATTEMPTS):
            attempts = attempt + 1
            try_seed = (explicit_seed if attempt == 0
                        else int(self.rng.integers(0, 2**31 - 1)))
            snapshot = self._snapshot_buffers()
            forward_err, fwd_ok = self._attempt_forward_only(try_seed)
            if fwd_ok:
                accepted_seed = try_seed
                break
            self._restore_buffers(snapshot)

        if accepted_seed is None:
            raise RuntimeError(
                f"visualize_pushcube_forward_retract: no valid forward push "
                f"after {pushcube_full._MAX_SCRIPTED_ATTEMPTS} attempts "
                f"(last forward_err={forward_err*1000:.1f}mm)"
            )

        self._scripted_attempts = attempts
        self._scripted_accepted_seed = accepted_seed
        self._scripted_forward_err = forward_err

        self._run_retract()
        self._capture("after_retract", force=True, still="after_retract")
        return self._write_outputs()

    def _attempt_forward_only(self, try_seed: int) -> tuple[float, bool]:
        """Env reset → forward push (identical to the full rollout), recording
        the initial cube and EEF pose. Returns (forward_err, forward_ok)."""
        self.obs, _ = self.env.reset(seed=try_seed)
        self._capture("initial", force=True, still="initial")

        for _ in range(2):
            self._step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), "clear_gripper")

        self.init_pos = _vec3_from_obs(self.obs, "obj_pose").copy()
        self.init_tcp = _vec3_from_obs(self.obs, "tcp_pose").copy()
        self._build_init_dependent_predicates()
        self._capture("source_recorded", force=True)

        self._run_forward_push()
        self._capture("after_forward_push", force=True, still="after_forward_push")
        self._cube_after_push = _vec3_from_obs(self.obs, "obj_pose").copy()

        forward_target_xy = np.array(
            [self.init_pos[0] + self.forward_push_displacement_m, self.init_pos[1]],
            dtype=np.float32,
        )
        forward_err = float(np.linalg.norm(self._cube_after_push[:2] - forward_target_xy))
        forward_tol = max(
            pushcube_full._FORWARD_PUSH_TOLERANCE_MIN_M,
            pushcube_full._FORWARD_PUSH_TOLERANCE_FRAC * abs(self.forward_push_displacement_m),
        )
        return forward_err, forward_err < forward_tol

    def _run_retract(self) -> None:
        """Exact reverse of the forward push trajectory: close the (empty)
        gripper, lower to just behind the pushed cube, retract -x back to the
        start, then elevate and open at the initial pose. Because the closed gripper pushed
        from behind (it sits on the -x side of the cube), retracting -x retreats
        away from the non-grasped cube instead of carrying it back: the cube
        stays at the goal and the world state is NOT restored. That failure is
        the point."""
        push_height_z = self.init_pos[2] + 0.005

        # reverse of "open": close the empty gripper.
        self._step_in_place(10, gripper_cmd=-1.0, phase="reverse_close_gripper")

        # reverse of "elevate": lower back down to just behind the pushed cube
        # — its -x side, where the gripper sat while pushing. Lowering here (not
        # onto the cube) lets the -x retract retreat without catching it.
        lower_tgt = np.array(
            [self._cube_after_push[0] - 0.06, self.init_pos[1], push_height_z],
            dtype=np.float32,
        )
        self._step_toward_scaled(
            lower_tgt, 25, 0.01, gripper_cmd=-1.0, scale=1.0,
            phase="reverse_lower_behind_cube",
        )

        # reverse of "push +x": retract -x back behind the start. The gripper
        # retreats from the (non-grasped) cube, which stays at the goal.
        retract_tgt = np.array(
            [self.init_pos[0] - 0.06, self.init_pos[1], push_height_z],
            dtype=np.float32,
        )
        self._step_toward_scaled(
            retract_tgt, 80, 0.005, gripper_cmd=-1.0,
            scale=pushcube_full._FORWARD_PUSH_SCALE,
            phase="reverse_retract_minus_x",
        )

        # reverse of "approach": elevate back to the initial pose, then open.
        self._step_toward_scaled(
            self.init_tcp.copy(), 30, 0.012, gripper_cmd=-1.0, scale=1.0,
            phase="reverse_elevate_to_start",
        )
        self._step_in_place(8, gripper_cmd=1.0, phase="reverse_open_at_start")

    def _write_outputs(self) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        base = self.out_dir / f"planrob_pushcube_forward_retract_seed{self.seed}"
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
        final_scene = demo._obs_to_scene(self.obs, self.regions)

        payload = {
            "seed": self.seed,
            "scripted_attempts": getattr(self, "_scripted_attempts", 1),
            "scripted_accepted_seed": getattr(self, "_scripted_accepted_seed", self.seed),
            "scripted_forward_err_m": getattr(self, "_scripted_forward_err", float("nan")),
            "env": "PushCube-v1",
            "control_mode": "pd_ee_delta_pos",
            "render_mode": "rgb_array",
            "fps": self.fps,
            "frame_stride": self.frame_stride,
            "n_sim_steps": self.sim_step,
            "n_video_frames": len(self.frames),
            "phase_counts": self.phase_counts,
            "init_pos": self.init_pos.tolist(),
            "init_tcp": self.init_tcp.tolist(),
            "forward_goal_pos": None if self.forward_goal_pos is None else self.forward_goal_pos.tolist(),
            "forward_push_displacement_m": self.forward_push_displacement_m,
            "cube_after_push": self._cube_after_push.tolist(),
            "final_cube_pos": final_cube.tolist(),
            "final_tcp_pos": final_tcp.tolist(),
            # The two numbers that make the point:
            "final_cube_to_source_m": float(np.linalg.norm(final_cube - self.init_pos)),
            "final_tcp_to_init_tcp_m": float(np.linalg.norm(final_tcp - self.init_tcp)),
            "v_source_after_retract": float(self.at_pose_eval.evaluate(final_scene).score),
            "use_env_goal_for_push": self.use_env_goal_for_push,
            "video_path": str(video_path),
            "still_paths": still_paths,
            "trace": self.trace,
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload


def _seed_list(spec: str) -> list[int]:
    return [int(s.strip()) for s in spec.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=_seed_list, default=[1000])
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("artifacts/out/1778234240/videos"),
        help="Output dir (default: alongside the existing full-rollout videos)",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--no-env-goal-push", action="store_true",
                        help="Use fixed --push-displacement-m instead of the env goal_x.")
    parser.add_argument("--push-displacement-m", type=float,
                        default=pushcube_full._PUSH_DISPLACEMENT_M)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(args.seeds)} forward+retract rollout(s) → {args.out_dir}")
    print(f"{'seed':>6}  {'attempts':>8}  {'cube→src_mm':>11}  {'tcp→init_mm':>11}  {'V_src':>6}  video")

    for seed in args.seeds:
        viz = PushCubeForwardRetractVisualizer(
            seed=seed,
            out_dir=args.out_dir,
            fps=args.fps,
            frame_stride=args.frame_stride,
            overlay=not args.no_overlay,
            push_displacement_m=args.push_displacement_m,
            use_env_goal_for_push=not args.no_env_goal_push,
            perturbation_range_m=0.0,
            model=None,
        )
        try:
            payload = viz.run()
        finally:
            viz.close()
        print(
            f"{payload['seed']:>6d}  {payload['scripted_attempts']:>8d}  "
            f"{payload['final_cube_to_source_m']*1000.0:>11.1f}  "
            f"{payload['final_tcp_to_init_tcp_m']*1000.0:>11.1f}  "
            f"{payload['v_source_after_retract']:>6.3f}  {payload['video_path']}"
        )


if __name__ == "__main__":
    main()
