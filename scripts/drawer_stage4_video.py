"""Stage 4 video: one complete forward-and-inverse sequence through the pipeline.

The steps of drawer_stage4_execute.py, for one start configuration and one
learned policy: the forward push_close, the measured symbolic state, the goal
from the extracted operator, the plan, and its execution by the drawer
executive with the hole filled by the learned grasp. The first caption line
names what is running; the second shows the predicate scores, green once a
predicate holds (>= 0.8). The outcome is compared with the evaluation's record
of the same episode.

Outputs: artifacts/drawer/videos/drawer_stage4_seed<seed>.mp4 and *_sheet.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import SAC

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import drawer_executive as ex
from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.envs.drawer_hole_env import load_spec
from inverse_skills.operators import hole_planner as hp

torch.set_num_threads(1)

LIBRARY = {"approach_handle": prim.approach_handle, "pull_open": prim.pull_open}
HOLD = 40   # frames a title card stays up
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_HOLDS = (120, 220, 130)
_DOES_NOT = (175, 175, 180)


def short(key: str) -> str:
    return key.replace("(drawer)", "")


class FrameRecorder(gym.Wrapper):
    """One captioned frame per control step; `title` sets the first caption line."""

    def __init__(self, env):
        super().__init__(env)
        self.registry = dp.registry()
        self.title = ""
        self.frames: list[np.ndarray] = []
        self.stills: list[np.ndarray] = []
        self.steps = 0
        self._raw = None
        self._obs = None

    def _grab(self, obs):
        self._raw = np.asarray(self.env.render().squeeze(0).cpu().numpy())
        self._obs = obs
        self.frames.append(self.captioned())

    def captioned(self) -> np.ndarray:
        out = self._raw.copy()
        box = out.copy()
        cv2.rectangle(box, (0, 0), (out.shape[1], 76), (12, 12, 12), -1)
        cv2.addWeighted(box, 0.62, out, 0.38, 0, out)
        cv2.putText(out, self.title, (10, 22), _FONT, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
        x = 10
        for key, value in sorted(ex.scores(self._obs, self.registry).items()):
            text = f"{short(key)} {value:.2f}"
            cv2.putText(out, text, (x, 45), _FONT, 0.42, _HOLDS if value >= ex.SATISFIED else _DOES_NOT,
                        1, cv2.LINE_AA)
            x += cv2.getTextSize(text, _FONT, 0.42, 1)[0][0] + 14
        cv2.putText(out, f"drawer opening {prim.opening(self._obs) * 1000:.0f} mm", (10, 66), _FONT, 0.42,
                    (245, 245, 245), 1, cv2.LINE_AA)
        return out

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._grab(obs)
        return obs, info

    def step(self, action):
        self.steps += 1
        out = self.env.step(action)
        self._grab(out[0])
        return out

    def hold(self, title: str):
        """A title card: the current frame, held, with `title` as its caption."""
        self.title = title
        self.frames.extend([self.captioned()] * HOLD)

    def still(self):
        self.stills.append(self.frames[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2000)
    ap.add_argument("--run", type=Path, default=Path("artifacts/drawer/stage3/1792053076"))
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--operator", type=Path, default=Path("artifacts/drawer/stage1_operator.json"))
    ap.add_argument("--spec", type=Path, default=Path("artifacts/drawer/hole_spec.json"))
    ap.add_argument("--evaluation", type=Path, default=Path("artifacts/drawer/stage4_execution.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/drawer/videos"))
    args = ap.parse_args()

    sac_seed = json.loads((args.run / "config.json").read_text())["seed"]
    policy = SAC.load(args.run / "sac_hole_best", device="cpu")
    spec = load_spec(args.spec)
    registry = dp.registry()
    env = FrameRecorder(prim.make_env(render_mode="rgb_array"))

    env.title = f"seed {args.seed}: the start"
    obs = prim.reset(env, args.seed)
    env.still()
    env.title = "FORWARD, the demonstrated skill: push_close"
    obs = prim.push_close(env, obs)
    env.still()

    state = hp.symbolic_state(ex.scores(obs, registry))
    env.hold("measured after the forward skill: " + (", ".join(short(p) for p in sorted(state)) or "nothing"))
    goal = dd.goal_from_operator(args.operator)
    env.hold(f"goal, the inverted operator: {', '.join(map(short, sorted(goal[0])))}"
             f" and not {', '.join(map(short, sorted(goal[1])))}")
    domain = dd.domain()
    result = hp.plan(domain, state, *goal)
    if not result:
        raise SystemExit(f"no plan from {sorted(state)}")
    env.hold("plan: " + " -> ".join(short(a) for a in result.actions))

    count = len(result.steps)
    titles = {step.name: f"INVERSE {i}/{count}: {short(step.name)}"
                         + (f" - learned SAC policy (seed {sac_seed})" if step.hole_for else " - library")
              for i, step in enumerate(result.steps, 1)}

    def show(name: str, attempt: int) -> None:
        if env.title.startswith("INVERSE"):
            env.still()   # the end of the step before
        env.title = titles[name] + (f", attempt {attempt}" if attempt > 1 else "")

    before = env.steps
    obs, report = ex.execute_plan(env, obs, list(result.steps), domain, goal, LIBRARY,
                                  {spec["achieve"]: policy}, on_action=show)
    env.still()
    failed = next((s for s in report.steps if not s.ok), None)
    env.hold("goal met, measured: drawer open, not closed" if report.ok
             else f"NOT restored: {failed.action}: {failed.detail}" if failed else "NOT restored: goal not met")
    env.close()

    inverse_steps = env.steps - before
    print(f"video: {'restored' if report.ok else 'NOT restored'}, final {report.final_opening_mm} mm, "
          f"{inverse_steps} inverse control steps")
    if args.evaluation.exists():
        rows = json.loads(args.evaluation.read_text())["results"][f"learned, SAC seed {sac_seed}"]
        row = next((r for r in rows if r["seed"] == args.seed), None)
        if row is not None:
            print(f"evaluation record: {'restored' if row['ok'] else 'NOT restored'}, final "
                  f"{row['final_opening_mm']} mm, {row['inverse_control_steps']} inverse control steps")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"drawer_stage4_seed{args.seed}"
    imageio.mimsave(f"{stem}.mp4", env.frames, fps=args.fps)
    imageio.imwrite(f"{stem}_sheet.png", np.concatenate(env.stills, axis=1))
    print(f"wrote {stem}.mp4 ({len(env.frames)} frames) and {stem}_sheet.png")


if __name__ == "__main__":
    main()
