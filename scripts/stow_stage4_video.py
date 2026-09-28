"""Stage 4 video: the three approaches executed on the same scene, in turn.

  full          the planned inverse with the learned operator in the hole
  library_only  the same plan with the hole skipped, because no hole-free plan
                exists — it breaks at pick(cube_a)
  hole_first    the equal-cost ordering the late-hole tie-break rejected

The panel tracks the goal literals throughout, so "the goal is met" is visible
rather than asserted at the end.

Output: artifacts/stow/videos/stow_stage4_seed<seed>.mp4 and stills.
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
_GOOD = (120, 220, 130)
_BAD = (235, 120, 110)
_MID = (190, 190, 195)
_ACCENT = (255, 215, 120)

GOAL_ROWS = [
    ("A at src_a", "in_region(cube_a,src_a)", True),
    ("B at src_b", "in_region(cube_b,src_b)", True),
    ("A clear of walls", "clear_of_walls(cube_a)", True),
    ("A out of pocket", "in_region(cube_a,pocket)", False),
    ("B out of mouth", "in_region(cube_b,mouth)", False),
    ("gripper open", "gripper_open()", True),
]


def _short(text: str) -> str:
    return text.replace("cube_a", "A").replace("cube_b", "B").replace("()", "")


class Recorder(gym.Wrapper):
    def __init__(self, env, registry):
        super().__init__(env)
        self.registry = registry
        self.regions = None
        self.title = ""
        self.action = ""
        self.note = None
        self.frames: list[np.ndarray] = []
        self.last_raw = None

    def annotate(self):
        frame = self.env.render()
        frame = np.asarray(frame.squeeze(0).cpu().numpy() if hasattr(frame, "cpu") else frame)
        self.last_raw = frame
        out = frame.copy()
        box = out.copy()
        cv2.rectangle(box, (0, 0), (out.shape[1], 62), (12, 12, 12), -1)
        cv2.addWeighted(box, 0.6, out, 0.4, 0, out)
        cv2.putText(out, self.title, (10, 23), _FONT, 0.58, (245, 245, 245), 1, cv2.LINE_AA)
        cv2.putText(out, self.action, (10, 48), _FONT, 0.48, _ACCENT, 1, cv2.LINE_AA)

        # reset() steps the env before main() can hand over the slots.
        if self.regions is None:
            self.regions = prim.regions(self._obs_cache)
        scores = {k: float(r.score) for k, r in self.registry.evaluate_all(
            prim.obs_to_scene(self._obs_cache, self.regions)).items()}
        pad, row_h = 10, 18
        top = out.shape[0] - (26 + row_h * len(GOAL_ROWS)) - pad
        box = out.copy()
        cv2.rectangle(box, (pad, top), (300, out.shape[0] - pad), (10, 10, 12), -1)
        cv2.addWeighted(box, 0.62, out, 0.38, 0, out)
        cv2.putText(out, "goal", (pad + 8, top + 18), _FONT, 0.44, (245, 245, 245), 1, cv2.LINE_AA)
        for i, (label, key, want_true) in enumerate(GOAL_ROWS):
            held = (scores[key] >= 0.8) if want_true else (scores[key] < 0.8)
            colour = _GOOD if held else _BAD
            cv2.putText(out, ("x " if held else "- ") + label, (pad + 8, top + 36 + i * row_h),
                        _FONT, 0.4, colour, 1, cv2.LINE_AA)
        if self.note:
            cv2.putText(out, self.note, (12, out.shape[0] - 14), _FONT, 0.46, _ACCENT, 1, cv2.LINE_AA)
        return out

    def step(self, action):
        out = self.env.step(action)
        self._obs_cache = out[0]
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
    conditions = [
        ("full: planner + learned operator", list(planned.actions)),
        ("library only: no hole-free plan exists",
         [a for a in planned.actions if not a.startswith("HOLE")]),
        ("hole first: equal cost, rejected by the tie-break", list(planned.alternatives[0])),
    ]

    registry = sp.registry()
    env = Recorder(prim.make_env(render_mode="rgb_array"), registry)
    env._obs_cache = None
    all_frames: list[np.ndarray] = []
    outcomes = []

    for title, plan in conditions:
        env._obs_cache = None
        env.regions = None
        obs = prim.reset(env, args.seed)
        env.regions = prim.regions(obs)
        env._obs_cache = obs
        env.title, env.action, env.note = title, "forward skill: stow A", None
        env.frames = []
        obs = prim.forward_stow(env, obs)
        env.action = "plan: " + " -> ".join(_short(a) for a in plan[:3]) + " ..."
        env.hold(1.5, args.fps)

        def announce(action, attempt):
            env.action = _short(action) + (f"   retry {attempt - 1}" if attempt > 1 else "")

        obs, report = ex.execute_plan(env, obs, plan, model=model,
                                      regions=env.regions, on_action=announce)
        failed = next((s for s in report.steps if not s.ok), None)
        env.action = "goal met" if report.goal_met else (
            f"FAILED at {_short(failed.action)}" if failed else "all steps passed, goal NOT met")
        env.note = (f"A {report.cube_a_err_mm:.0f}mm from source, "
                    f"B {report.cube_b_err_mm:.0f}mm")
        env.hold(2.5, args.fps)
        all_frames.extend(env.frames)
        outcomes.append((title, report))
        imageio.imwrite(args.out_dir / f"stow_stage4_seed{args.seed}_"
                        f"{title.split(':')[0].replace(' ', '_')}.png", env.frames[-1])

    card = (env.last_raw.astype(np.float32) * 0.25).astype(np.uint8)
    lines = [("three ways to invert the same skill", (245, 245, 245), 0.55), ("", _MID, 0.45)]
    for title, report in outcomes:
        name = title.split(":")[0]
        verdict = "goal met" if report.goal_met else "goal not met"
        colour = _GOOD if report.goal_met else _BAD
        lines.append((f"{name:12s} {verdict}   A {report.cube_a_err_mm:5.1f}mm  "
                      f"B {report.cube_b_err_mm:5.1f}mm", colour, 0.44))
    lines.append(("", _MID, 0.45))
    for title, report in outcomes:
        failed = next((s for s in report.steps if not s.ok), None)
        name = title.split(":")[0]
        if failed is not None:
            lines.append((f"{name}: stops at {_short(failed.action)}", _MID, 0.42))
            lines.append((f"   {failed.detail}", _MID, 0.42))
        elif not report.goal_met:
            lines.append((f"{name}: every step passed, goal missed", _MID, 0.42))
            lines.append(("   a later step disturbed an earlier one", _MID, 0.42))
    y = 56
    for text, colour, scale in lines:
        cv2.putText(card, text, (20, y), _FONT, scale, colour, 1, cv2.LINE_AA)
        y += 28 if scale >= 0.5 else 24
    all_frames.extend([card] * (5 * args.fps))
    env.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"stow_stage4_seed{args.seed}"
    imageio.mimsave(f"{stem}.mp4", all_frames, fps=args.fps, macro_block_size=8)
    imageio.imwrite(f"{stem}_card.png", card)
    for title, report in outcomes:
        print(f"{title:52s} goal_met={report.goal_met} A={report.cube_a_err_mm:.1f}mm")
    print(f"{len(all_frames)} frames -> {stem}.mp4 ({len(all_frames) / args.fps:.1f} s)")


if __name__ == "__main__":
    main()
