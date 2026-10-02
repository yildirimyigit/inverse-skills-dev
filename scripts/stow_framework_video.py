"""The framework running end to end: the forward skill, then its inverse.

No ablations. The forward skill stows A, the planner writes the inverse from
the extracted operator, and the executive runs it — with the learned operator
filling the hole in the middle of the plan. The left panel is the plan, the
right panel is the goal, so both the "what is it doing" and the "is it done"
are legible without narration.

Output: artifacts/stow/videos/stow_framework_seed<seed>.mp4 and stills.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import SAC

from inverse_skills.envs import stow_domain as sd
from inverse_skills.envs import stow_executive as ex
from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs import stow_primitives as prim
from inverse_skills.operators import hole_planner as hp

torch.set_num_threads(1)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_SIZE = 768
_DONE = (120, 220, 130)
_CURRENT = (255, 215, 120)
_PENDING = (160, 160, 168)
_LEARNED = (205, 165, 255)
_MISSING = (235, 120, 110)

GOAL_ROWS = [
    ("A at its source", "in_region(cube_a,src_a)", True),
    ("B at its source", "in_region(cube_b,src_b)", True),
    ("A clear of walls", "clear_of_walls(cube_a)", True),
    ("A out of the pocket", "in_region(cube_a,pocket)", False),
    ("B out of the mouth", "in_region(cube_b,mouth)", False),
    ("gripper open", "gripper_open()", True),
]


def _short(text: str) -> str:
    return (text.replace("cube_a", "A").replace("cube_b", "B").replace("()", "")
            .replace("HOLE[clear_of_walls(A)]", "clear_of_walls(A)"))


def _box(frame, x0, y0, x1, y1, alpha=0.62):
    patch = frame.copy()
    cv2.rectangle(patch, (x0, y0), (x1, y1), (10, 10, 12), -1)
    cv2.addWeighted(patch, alpha, frame, 1 - alpha, 0, frame)


class Recorder(gym.Wrapper):
    def __init__(self, env, registry, plan: list[str]):
        super().__init__(env)
        self.registry = registry
        self.plan = plan
        self.regions = None
        self.obs_cache = None
        self.title = ""
        self.subtitle = ""
        self.current = -1
        self.done: set[int] = set()
        self.frames: list[np.ndarray] = []
        self.last_raw = None

    def annotate(self):
        frame = self.env.render()
        frame = np.asarray(frame.squeeze(0).cpu().numpy() if hasattr(frame, "cpu") else frame)
        frame = cv2.resize(frame, (_SIZE, _SIZE), interpolation=cv2.INTER_CUBIC)
        self.last_raw = frame
        out = frame.copy()

        _box(out, 0, 0, _SIZE, 74, 0.6)
        cv2.putText(out, self.title, (16, 30), _FONT, 0.68, (245, 245, 245), 1, cv2.LINE_AA)
        colour = _LEARNED if "clear_of_walls" in self.subtitle else _CURRENT
        cv2.putText(out, self.subtitle, (16, 58), _FONT, 0.55, colour, 1, cv2.LINE_AA)

        if self.regions is None:
            self.regions = prim.regions(self.obs_cache)
        scores = {k: float(r.score) for k, r in self.registry.evaluate_all(
            prim.obs_to_scene(self.obs_cache, self.regions)).items()}

        row_h, pad = 26, 14
        height = 38 + row_h * max(len(self.plan), len(GOAL_ROWS))
        top = _SIZE - height - pad
        _box(out, pad, top, pad + 350, _SIZE - pad)
        cv2.putText(out, "plan", (pad + 12, top + 26), _FONT, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        for i, step in enumerate(self.plan):
            learned = step.startswith("HOLE")
            if i in self.done:
                mark, colour = "x", _DONE
            elif i == self.current:
                mark, colour = ">", (_LEARNED if learned else _CURRENT)
            else:
                mark, colour = " ", (_LEARNED if learned else _PENDING)
            label = f"{mark} {i + 1}. {_short(step)}" + ("   learned" if learned else "")
            cv2.putText(out, label, (pad + 12, top + 52 + i * row_h), _FONT, 0.44,
                        colour, 1, cv2.LINE_AA)

        left = _SIZE - pad - 300
        _box(out, left, top, _SIZE - pad, _SIZE - pad)
        cv2.putText(out, "goal", (left + 12, top + 26), _FONT, 0.52, (245, 245, 245), 1, cv2.LINE_AA)
        for i, (label, key, want_true) in enumerate(GOAL_ROWS):
            held = (scores[key] >= 0.8) if want_true else (scores[key] < 0.8)
            cv2.putText(out, ("x " if held else "- ") + label, (left + 12, top + 52 + i * row_h),
                        _FONT, 0.44, _DONE if held else _MISSING, 1, cv2.LINE_AA)
        return out

    def step(self, action):
        out = self.env.step(action)
        self.obs_cache = out[0]
        self.frames.append(self.annotate())
        return out

    def hold(self, seconds: float, fps: int):
        self.frames.extend([self.annotate()] * int(seconds * fps))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--operator", type=Path, default=Path("artifacts/stow/stage1_operator.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/stow/videos"))
    args = ap.parse_args()

    model = SAC.load(args.checkpoint, device="cuda")
    goal_pos, goal_neg = sd.goal_from_operator(args.operator)
    planned = hp.plan(sd.domain(), sd.stowed_state(), goal_pos, goal_neg)
    plan = list(planned.actions)
    print("plan:", " -> ".join(plan))

    env = Recorder(prim.make_env(render_mode="rgb_array"), sp.registry(), plan)
    env.title = "forward skill: stow A"
    env.subtitle = "push A into the pocket, then B into its mouth"
    obs = prim.reset(env, args.seed)
    env.regions = prim.regions(obs)
    env.obs_cache = obs
    env.frames = []
    env.hold(1.5, args.fps)
    obs = prim.forward_stow(env, obs)

    env.title = "inverse skill: restore the world"
    env.subtitle = (f"planner: {planned.cost[0]} predicate delegated, "
                    f"hole at step {planned.hole_indices[0] + 1} of {len(plan)}")
    env.hold(2.5, args.fps)

    def announce(action, attempt):
        env.current = plan.index(action) if action in plan else env.current
        env.done = {i for i in range(len(plan)) if i < env.current}
        env.subtitle = _short(action) + (f"   retry {attempt - 1}" if attempt > 1 else "")

    obs, report = ex.execute_plan(env, obs, list(planned.steps), model=model, regions=env.regions,
                                  goal=(goal_pos, goal_neg),
                                  on_action=announce)
    env.current, env.done = -1, set(range(len(plan)))
    env.title = "restored" if report.goal_met else "goal not met"
    env.subtitle = (f"A {report.cube_a_err_mm:.1f} mm from source, "
                    f"B {report.cube_b_err_mm:.1f} mm, "
                    f"{sum(max(s.attempts - 1, 0) for s in report.steps)} retries")
    env.hold(3.5, args.fps)
    env.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"stow_framework_seed{args.seed}"
    imageio.mimsave(f"{stem}.mp4", env.frames, fps=args.fps, macro_block_size=8)
    imageio.imwrite(f"{stem}_final.png", env.frames[-1])
    hole_frames = [i for i, _ in enumerate(env.frames)]
    imageio.imwrite(f"{stem}_mid.png", env.frames[hole_frames[len(hole_frames) // 2]])
    print(f"goal met: {report.goal_met}; A {report.cube_a_err_mm:.1f}mm B {report.cube_b_err_mm:.1f}mm")
    print(f"{len(env.frames)} frames -> {stem}.mp4 ({len(env.frames) / args.fps:.1f} s)")


if __name__ == "__main__":
    main()
