"""Stage 0 video: the scripted drawer actions on SlidingDrawer-v1 for one seed.

Two runs, one after the other:
  1. push_close (the forward skill) -> approach_handle -> test_grasp (Stage 0
     only, stands in for the skill RL will learn) -> pull_open
  2. the same without the grasp: push_close -> approach_handle -> pull_open,
     which leaves the drawer shut

One frame per control step with a caption, a PNG still at the end of every
phase, and a contact sheet of the stills.

Outputs: artifacts/drawer/videos/drawer_stage0_seed<seed>.mp4, *_<n>_<phase>.png, *_sheet.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from inverse_skills.envs import drawer_primitives as prim


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    box = out.copy()
    cv2.rectangle(box, (0, 0), (out.shape[1], 34), (12, 12, 12), -1)
    cv2.addWeighted(box, 0.58, out, 0.42, 0, out)
    cv2.putText(out, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return out


class FrameRecorder(gym.Wrapper):
    """Records one captioned frame per step; `phase` sets the caption."""

    def __init__(self, env):
        super().__init__(env)
        self.phase = ""
        self.frames: list[np.ndarray] = []
        self.stills: list[tuple[str, np.ndarray]] = []

    def _grab(self):
        frame = np.asarray(self.env.render().squeeze(0).cpu().numpy())
        opening_mm = float(self.env.unwrapped.cabinet.get_qpos()[0, 0]) * 1000
        self.frames.append(caption(frame, f"{self.phase}   (open {opening_mm:.0f} mm)"))

    def reset(self, **kwargs):
        out = self.env.reset(**kwargs)
        self._grab()
        return out

    def step(self, action):
        out = self.env.step(action)
        self._grab()
        return out

    def finish(self, label: str):
        """Keep the last frame of the phase just run as a still."""
        self.stills.append((label, self.frames[-1]))


def contact_sheet(stills: list[tuple[str, np.ndarray]], columns: int = 4) -> np.ndarray:
    tiles = [frame for _, frame in stills]
    while len(tiles) % columns:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.concatenate(tiles[i:i + columns], axis=1) for i in range(0, len(tiles), columns)]
    return np.concatenate(rows, axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/drawer/videos"))
    args = ap.parse_args()

    rec = FrameRecorder(prim.make_env(render_mode="rgb_array"))

    rec.phase = "1. start: drawer open"
    obs = prim.reset(rec, args.seed)
    rec.finish("start")
    rec.phase = "1. push_close (forward skill)"
    obs = prim.push_close(rec, obs)
    rec.finish("pushed shut")
    rec.phase = "1. approach_handle"
    obs = prim.approach_handle(rec, obs)
    rec.finish("at the standoff")
    rec.phase = "1. test_grasp (Stage 0 only)"
    obs = prim.test_grasp(rec, obs)
    rec.finish("handle grasped")
    rec.phase = "1. pull_open"
    obs = prim.pull_open(rec, obs)
    rec.finish("pulled open")

    rec.phase = "2. no grasp: push_close"
    obs = prim.push_close(rec, prim.reset(rec, args.seed))
    rec.phase = "2. approach_handle"
    obs = prim.approach_handle(rec, obs)
    rec.phase = "2. pull_open, nothing held"
    obs = prim.pull_open(rec, obs)
    rec.finish("no grasp: still shut")
    rec.env.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"drawer_stage0_seed{args.seed}"
    imageio.mimsave(f"{stem}.mp4", rec.frames, fps=args.fps)
    for i, (label, frame) in enumerate(rec.stills, 1):
        imageio.imwrite(f"{stem}_{i}_{label.replace(' ', '_').replace(':', '')}.png", frame)
    imageio.imwrite(f"{stem}_sheet.png", contact_sheet(rec.stills, columns=3))
    print(f"wrote {stem}.mp4 ({len(rec.frames)} frames) and {len(rec.stills)} stills")


if __name__ == "__main__":
    main()
