"""Stage 0 video: the full scripted inverse on StowCube-v1 for one seed.

Phases: forward stow -> pick B -> place B -> approach A -> drag A (scripted
stand-in for the learned hole) -> pick A -> place A. One frame per control
step with a phase overlay, plus a PNG still at every phase boundary. Style
mirrors scripts/visualize_pushcube_full_rollout.py.

Outputs: artifacts/stow/videos/stow_stage0_seed<seed>.mp4 and *_<phase>.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from inverse_skills.envs import stow_cube as geo
from inverse_skills.envs import stow_primitives as prim

try:
    import cv2
except ImportError:  # overlay is optional
    cv2 = None


def _overlay(frame: np.ndarray, text: str) -> np.ndarray:
    if cv2 is None:
        return frame
    out = frame.copy()
    box = out.copy()
    cv2.rectangle(box, (0, 0), (out.shape[1], 34), (12, 12, 12), -1)
    cv2.addWeighted(box, 0.58, out, 0.42, 0, out)
    cv2.putText(out, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1, cv2.LINE_AA)
    return out


class FrameRecorder(gym.Wrapper):
    """Records one rendered frame per step, labelled with the current phase."""

    def __init__(self, env):
        super().__init__(env)
        self.phase = ""
        self.frames: list[np.ndarray] = []
        self.stills: dict[str, np.ndarray] = {}

    def _grab(self):
        frame = self.env.render()
        frame = np.asarray(frame.squeeze(0).cpu().numpy() if hasattr(frame, "cpu") else frame)
        self.frames.append(_overlay(frame, self.phase))

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._grab()
        return out

    def step(self, action):
        out = self.env.step(action)
        self._grab()
        return out

    def mark(self, name: str):
        """Snapshot the end of the phase just completed, then start `name`."""
        self.stills[f"after {self.phase}"] = self.frames[-1]
        self.phase = name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/stow/videos"))
    args = ap.parse_args()

    rec = FrameRecorder(prim.make_env(render_mode="rgb_array"))
    rec.phase = "initial"
    obs = prim.reset(rec, args.seed)
    rec.mark("forward stow (push A, push B)")
    obs = prim.forward_stow(rec, obs)
    rec.mark("pick B")
    obs = prim.pick(rec, obs, "cube_b")
    rec.mark("place B at src B")
    obs = prim.place(rec, obs, "cube_b", prim.src_pos(obs, "cube_b"))
    rec.mark("approach A")
    obs = prim.approach(rec, obs, "cube_a")
    rec.mark("HOLE: drag A out (scripted stand-in)")
    obs = prim.drag_out(rec, obs, "cube_a", geo.X_CLEAR - 0.01)
    rec.mark("pick A")
    obs = prim.lift(rec, prim.pick(rec, obs, "cube_a"))
    rec.mark("place A at src A")
    obs = prim.place(rec, obs, "cube_a", prim.src_pos(obs, "cube_a"))
    rec.mark("restored")
    rec.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"stow_stage0_seed{args.seed}"
    imageio.mimsave(f"{stem}.mp4", rec.frames, fps=args.fps, macro_block_size=8)
    for i, (name, frame) in enumerate(rec.stills.items()):
        slug = name.split(":")[0].split("(")[0].strip().replace(" ", "_")
        imageio.imwrite(f"{stem}_{i}_{slug}.png", frame)
    a_err = np.linalg.norm(prim.cube_pos(obs, "cube_a")[:2] - prim.src_pos(obs, "cube_a")[:2])
    b_err = np.linalg.norm(prim.cube_pos(obs, "cube_b")[:2] - prim.src_pos(obs, "cube_b")[:2])
    print(f"{len(rec.frames)} frames -> {stem}.mp4 ({len(rec.frames) / args.fps:.1f} s); "
          f"final A err {a_err * 1000:.1f} mm, B err {b_err * 1000:.1f} mm")


if __name__ == "__main__":
    main()
