"""Stage 3 video: the learned handle grasp on unseen start states, then the pull.

For each (model, seed): the library steps that lead to the hole (push_close,
approach_handle, plus the episode's small random start offset), the learned
policy's 40 steps, then the scripted pull_open. One captioned frame per control
step, and a contact sheet of the last frame of each phase.

Outputs: <out-dir>/drawer_stage3_<run>.mp4 and drawer_stage3_<run>_sheet.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import SAC

from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.envs.drawer_hole_env import DrawerHoleEnv, load_spec


def caption(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    box = out.copy()
    cv2.rectangle(box, (0, 0), (out.shape[1], 34), (12, 12, 12), -1)
    cv2.addWeighted(box, 0.58, out, 0.42, 0, out)
    cv2.putText(out, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
    return out


class FrameRecorder(gym.Wrapper):
    """Records one captioned frame per step; `phase` sets the caption."""

    def __init__(self, env):
        super().__init__(env)
        self.phase = ""
        self.frames: list[np.ndarray] = []
        self.stills: list[np.ndarray] = []

    def step(self, action):
        out = self.env.step(action)
        frame = np.asarray(self.env.render().squeeze(0).cpu().numpy())
        opening_mm = float(self.env.unwrapped.cabinet.get_qpos()[0, 0]) * 1000
        self.frames.append(caption(frame, f"{self.phase}   (open {opening_mm:.0f} mm)"))
        return out

    def finish(self):
        self.stills.append(self.frames[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[1000, 1001])
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--spec", type=Path, default=Path("artifacts/drawer/hole_spec.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/drawer/videos"))
    args = ap.parse_args()

    hole = DrawerHoleEnv(load_spec(args.spec), render_mode="rgb_array")
    rec = FrameRecorder(hole._env)
    hole._env = rec   # record every step the hole environment and the primitives take
    clips = [("final model", "sac_hole_final", seed) for seed in args.seeds]
    clips.append(("early best checkpoint", "sac_hole_best", args.seeds[0]))
    for label, checkpoint, seed in clips:
        model = SAC.load(args.run_dir / checkpoint, device="cpu")
        rec.phase = f"seed {seed}: push_close, approach_handle (library), random start offset"
        obs, _ = hole.reset(seed=seed)
        rec.finish()
        rec.phase = f"seed {seed}: learned grasp ({label})"
        for _ in range(hole.max_steps):
            obs, _r, _term, _trunc, info = hole.step(model.predict(obs, deterministic=True)[0])
        rec.finish()
        rec.phase = f"seed {seed}: pull_open (library)"
        prim.pull_open(rec, hole._obs)
        rec.finish()
        print(f"{label}, seed {seed}: grasp {'held' if info['postcondition'] else 'NOT held'}, "
              f"drawer pulled to {float(rec.unwrapped.cabinet.get_qpos()[0, 0]) * 1000:.0f} mm")
        hole._handoffs.clear()   # replay push and approach on camera for every clip
    hole.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"drawer_stage3_{args.run_dir.name}"
    imageio.mimsave(f"{stem}.mp4", rec.frames, fps=args.fps)
    rows = [np.concatenate(rec.stills[i:i + 3], axis=1) for i in range(0, len(rec.stills), 3)]
    imageio.imwrite(f"{stem}_sheet.png", np.concatenate(rows, axis=0))
    print(f"wrote {stem}.mp4 ({len(rec.frames)} frames) and {stem}_sheet.png")


if __name__ == "__main__":
    main()
