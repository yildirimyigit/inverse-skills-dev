"""Stage 2 video: the planned inverse, executed with the hole still empty.

Plans from the measured handoff, then runs the library steps on the robot. At
the hole the arm simply stops — there is no policy yet — and the video shows
what the library alone can do next: pick(cube_a) is attempted and fails,
because that is precisely the precondition the hole exists to establish.

Output: artifacts/stow/videos/stow_stage2_seed<seed>.mp4 and stills.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from inverse_skills.envs import stow_cube as geo
from inverse_skills.envs import stow_domain as sd
from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs import stow_primitives as prim
from inverse_skills.operators import hole_planner as hp

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_DONE = (120, 220, 130)
_CURRENT = (255, 215, 120)
_PENDING = (165, 165, 170)
_HOLE = (205, 165, 255)
_FAIL = (235, 120, 110)


def _short(text: str) -> str:
    return text.replace("cube_a", "A").replace("cube_b", "B").replace("()", "")


def _header(frame: np.ndarray, text: str, color=(245, 245, 245)) -> np.ndarray:
    out = frame.copy()
    box = out.copy()
    cv2.rectangle(box, (0, 0), (out.shape[1], 34), (12, 12, 12), -1)
    cv2.addWeighted(box, 0.58, out, 0.42, 0, out)
    cv2.putText(out, text, (10, 23), _FONT, 0.6, color, 1, cv2.LINE_AA)
    return out


def _plan_panel(frame: np.ndarray, steps: list[str], current: int, done: set[int],
                note: str | None = None) -> np.ndarray:
    out = frame.copy()
    pad, row_h = 10, 18
    height = 32 + row_h * len(steps)
    top = out.shape[0] - height - pad
    box = out.copy()
    cv2.rectangle(box, (pad, top), (300, out.shape[0] - pad), (10, 10, 12), -1)
    cv2.addWeighted(box, 0.62, out, 0.38, 0, out)
    cv2.putText(out, "plan", (pad + 8, top + 20), _FONT, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
    for i, step in enumerate(steps):
        is_hole = step.startswith("HOLE")
        if i in done:
            color, mark = _DONE, "x"
        elif i == current:
            color, mark = (_HOLE if is_hole else _CURRENT), ">"
        else:
            color, mark = (_HOLE if is_hole else _PENDING), " "
        cv2.putText(out, f"{mark} {i + 1}. {_short(step)}", (pad + 8, top + 38 + i * row_h),
                    _FONT, 0.38, color, 1, cv2.LINE_AA)
    if note:
        band = out.copy()
        cv2.rectangle(band, (0, 40), (out.shape[1], 70), (10, 10, 12), -1)
        cv2.addWeighted(band, 0.66, out, 0.34, 0, out)
        cv2.putText(out, note, (12, 60), _FONT, 0.48, _CURRENT, 1, cv2.LINE_AA)
    return out


def _card(frame: np.ndarray, lines: list[tuple[str, tuple, float]]) -> np.ndarray:
    out = (frame.astype(np.float32) * 0.25).astype(np.uint8)
    y = 58
    for text, color, scale in lines:
        cv2.putText(out, text, (24, y), _FONT, scale, color, 1, cv2.LINE_AA)
        y += 28 if scale >= 0.5 else 23
    return out


class Recorder(gym.Wrapper):
    def __init__(self, env, steps: list[str]):
        super().__init__(env)
        self.steps = steps
        self.current = 0
        self.done: set[int] = set()
        self.phase = ""
        self.note: str | None = None
        self.frames: list[np.ndarray] = []
        self.last_raw: np.ndarray | None = None

    def annotate(self) -> np.ndarray:
        frame = self.env.render()
        frame = np.asarray(frame.squeeze(0).cpu().numpy() if hasattr(frame, "cpu") else frame)
        self.last_raw = frame
        return _plan_panel(_header(frame, self.phase), self.steps, self.current,
                           self.done, self.note)

    def step(self, action):
        out = self.env.step(action)
        self.frames.append(self.annotate())
        return out

    def hold(self, seconds: float, fps: int):
        self.frames.extend([self.annotate()] * int(seconds * fps))


def _slot_target(obs, cube: str, slot: str) -> np.ndarray:
    if slot in ("src_a", "src_b"):
        return prim.src_pos(obs, cube)
    xy = geo.POCKET_A_XY if slot == "pocket" else geo.MOUTH_B_XY
    return np.array([xy[0], xy[1], geo.CUBE_HALF], dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/stow/videos"))
    ap.add_argument("--operator", type=Path, default=Path("artifacts/stow/stage1_operator.json"))
    args = ap.parse_args()

    goal_pos, goal_neg = sd.goal_from_operator(args.operator)
    domain = sd.domain()
    registry = sp.registry()

    env = prim.make_env(render_mode="rgb_array")
    obs = prim.reset(env, args.seed)
    regions = prim.regions(obs)
    obs = prim.forward_stow(env, obs)
    scene = prim.obs_to_scene(obs, regions)
    state = hp.symbolic_state({k: float(r.score) for k, r in registry.evaluate_all(scene).items()})
    result = hp.plan(domain, state, goal_pos, goal_neg)
    steps = list(result.actions)
    print("plan:", " -> ".join(steps))

    rec = Recorder(env, steps)
    rec.phase = "stowed: the state the forward skill left"
    rec.note = f"planner: {result.cost[0]} predicate delegated, {result.cost[3]} steps"
    rec.hold(2.0, args.fps)
    rec.note = None

    hole_index = steps.index(next(s for s in steps if s.startswith("HOLE")))
    for i, action in enumerate(steps):
        rec.current = i
        if action.startswith("HOLE"):
            rec.phase = "HOLE: no policy yet"
            rec.note = "Stage 3 learns this step"
            rec.hold(3.0, args.fps)
            rec.note = None
            break
        rec.phase = _short(action)
        cube = "cube_a" if "cube_a" in action else "cube_b"
        if action.startswith("pick"):
            obs = prim.pick(rec, obs, cube)
        elif action.startswith("place"):
            slot = action[action.index(",") + 1:-1]
            obs = prim.place(rec, obs, cube, _slot_target(obs, cube, slot))
        elif action.startswith("approach"):
            obs = prim.approach(rec, obs, cube)
        rec.done.add(i)

    # What the library can do on its own from here: try the next step anyway.
    rec.current = hole_index + 1
    rec.phase = "skipping the hole: pick(A)"
    obs = prim.lift(rec, prim.pick(rec, obs, "cube_a"))
    lifted = float(prim.cube_pos(obs, "cube_a")[2]) > 0.05
    rec.phase = "pick(A) lifted A" if lifted else "pick(A) failed: A is still wedged"
    rec.note = None
    rec.hold(2.0, args.fps)

    raw = rec.last_raw
    rec.close()

    alt = result.alternatives[0] if result.alternatives else ()
    card = _card(raw, [
        ("planning with an incomplete library", (245, 245, 245), 0.55),
        ("", _PENDING, 0.45),
        (f"cost = {result.cost[0]} predicate delegated, {result.cost[1]} literals,", _PENDING, 0.45),
        (f"       {result.cost[2]} hole, {result.cost[3]} steps", _PENDING, 0.45),
        (f"hole at step {result.hole_indices[0] + 1} of {len(steps)}: mid-plan, not last",
         _HOLE, 0.48),
        ("", _PENDING, 0.45),
        ("library only, no holes:  NO PLAN", _FAIL, 0.48),
        ("same cost, rejected by the late-hole rule:", _PENDING, 0.42),
        ("   " + _short(" -> ".join(alt[:2])) + " ...", _PENDING, 0.42),
    ])
    rec.frames.extend([card] * int(4 * args.fps))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"stow_stage2_seed{args.seed}"
    imageio.mimsave(f"{stem}.mp4", rec.frames, fps=args.fps, macro_block_size=8)
    imageio.imwrite(f"{stem}_hole.png", rec.frames[len(rec.frames) // 2])
    imageio.imwrite(f"{stem}_card.png", card)
    print(f"{len(rec.frames)} frames -> {stem}.mp4 ({len(rec.frames) / args.fps:.1f} s); "
          f"pick(A) after skipping the hole: {'LIFTED' if lifted else 'failed as expected'}")


if __name__ == "__main__":
    main()
