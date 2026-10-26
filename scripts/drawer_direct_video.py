"""Direct baseline video: learned direct policies opening the drawer on a validation start.

For each run given: the library steps before the hole (the forward push_close
and approach_handle, replayed on camera), the policy's 100-step episode from its
best checkpoint, then the 10-step settling check. The second caption line shows
the drawer's opening, the drawer_open and handle_grasped scores, and the two
finger forces, so it is visible how the drawer is moved.

Output: artifacts/drawer/videos/drawer_direct_seed<start>.mp4 and *_sheet.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import SAC

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.envs.drawer_direct_env import GRASPED, DrawerDirectEnv
from inverse_skills.envs.drawer_hole_env import load_spec

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class FrameRecorder(gym.Wrapper):
    """One captioned frame per simulator step; `title` sets the first caption line."""

    def __init__(self, env, scores):
        super().__init__(env)
        self.scores = scores
        self.title = ""
        self.frames: list[np.ndarray] = []
        self.stills: list[np.ndarray] = []

    def step(self, action):
        out = self.env.step(action)
        obs = out[0]
        frame = np.asarray(self.env.render().squeeze(0).cpu().numpy()).copy()
        scores = self.scores(obs)
        left, right = prim.pinch_force(obs)
        box = frame.copy()
        cv2.rectangle(box, (0, 0), (frame.shape[1], 56), (12, 12, 12), -1)
        cv2.addWeighted(box, 0.62, frame, 0.38, 0, frame)
        cv2.putText(frame, self.title, (10, 22), _FONT, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(frame, f"open {prim.opening(obs) * 1000:.0f} mm | drawer_open {scores['drawer_open(drawer)']:.2f}"
                    f" | handle_grasped {scores[GRASPED]:.2f} | fingers {left:.0f}/{right:.0f} N",
                    (10, 45), _FONT, 0.42, (210, 210, 215), 1, cv2.LINE_AA)
        self.frames.append(frame)
        return out

    def still(self):
        self.stills.append(self.frames[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--start", type=int, default=500, help="a validation start; never an evaluation seed")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--operator", type=Path, default=Path("artifacts/drawer/stage1_operator.json"))
    ap.add_argument("--spec", type=Path, default=Path("artifacts/drawer/direct/hole_spec_direct.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/drawer/videos"))
    args = ap.parse_args()

    env = DrawerDirectEnv(load_spec(args.spec), dd.goal_from_operator(args.operator), render_mode="rgb_array")
    rec = FrameRecorder(env._env, env._scores)
    env._env = rec   # record every step the environment and the library steps take
    for run in args.runs:
        sac_seed = json.loads((run / "config.json").read_text())["seed"]
        model = SAC.load(run / "sac_direct_best", device="cpu")
        env._handoffs.clear()   # replay the push and the approach on camera
        rec.title = f"start {args.start}: forward push_close, then approach_handle (library)"
        obs, _ = env.reset(seed=args.start)
        rec.still()
        rec.title = f"start {args.start}: direct policy, HOLE[drawer_open] (SAC seed {sac_seed}, best checkpoint)"
        truncated = False
        while not truncated:
            obs, _reward, _terminated, truncated, _info = env.step(model.predict(obs, deterministic=True)[0])
        rec.still()
        rec.title = f"start {args.start}: settling check, arm held still"
        settled = env.settle()
        rec.still()
        episode = env.episodes[-1]
        print(f"SAC seed {sac_seed}: goal {'met' if episode['goal_met'] else 'NOT met'} at the end, first open at step "
              f"{episode['first_satisfied_step']}, grasp held at some step: {episode['ever_grasped']}, "
              f"after settling {settled['settled_opening_mm']} mm, goal {'met' if settled['settled_goal_met'] else 'NOT met'}")
    env.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"drawer_direct_start{args.start}"
    imageio.mimsave(f"{stem}.mp4", rec.frames, fps=args.fps)
    rows = [np.concatenate(rec.stills[i:i + 3], axis=1) for i in range(0, len(rec.stills), 3)]
    imageio.imwrite(f"{stem}_sheet.png", np.concatenate(rows, axis=0))
    print(f"wrote {stem}.mp4 ({len(rec.frames)} frames) and {stem}_sheet.png")


if __name__ == "__main__":
    main()
