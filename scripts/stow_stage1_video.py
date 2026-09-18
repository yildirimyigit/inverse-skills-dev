"""Stage 1 video: one demonstration with its predicate scores, and the operator it yields.

Shows what the extractor actually consumes — the predicate vector at the two
scenes it reads — and what falls out of comparing them. The panel lists every
predicate in the registry with its live score; the two boundary scenes are
held on screen and labelled; the closing card shows the extracted operator.

Output: artifacts/stow/videos/stow_stage1_seed<seed>.mp4 and stills.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np

from inverse_skills.envs import stow_cube as geo
from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs import stow_primitives as prim
from inverse_skills.logging.rollout import ForwardRollout
from inverse_skills.operators.extractor import OperatorExtractor

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GOOD = (120, 220, 130)     # BGR-free: frames are RGB, so these are (R, G, B)
_BAD = (235, 120, 110)
_MID = (200, 200, 200)
_DIM = (150, 150, 150)


def _short(key: str) -> str:
    return key.replace("cube_a", "A").replace("cube_b", "B").replace("()", "")


def _panel(frame: np.ndarray, scores: dict[str, float], title: str,
           note: str | None = None) -> np.ndarray:
    out = frame.copy()
    rows = sorted(scores)
    pad, row_h = 10, 17
    height = 30 + row_h * len(rows)
    top = out.shape[0] - height - pad
    box = out.copy()
    cv2.rectangle(box, (pad, top), (300, out.shape[0] - pad), (10, 10, 12), -1)
    cv2.addWeighted(box, 0.62, out, 0.38, 0, out)

    cv2.putText(out, title, (pad + 8, top + 19), _FONT, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
    for i, key in enumerate(rows):
        v = scores[key]
        y = top + 36 + i * row_h
        color = _GOOD if v >= 0.8 else _BAD if v <= 0.2 else _MID
        cv2.putText(out, _short(key), (pad + 8, y), _FONT, 0.38, color, 1, cv2.LINE_AA)
        bar_x = pad + 195
        cv2.rectangle(out, (bar_x, y - 8), (bar_x + 60, y - 1), (70, 70, 75), -1)
        cv2.rectangle(out, (bar_x, y - 8), (bar_x + max(1, int(60 * v)), y - 1), color, -1)
        cv2.putText(out, f"{v:.2f}", (bar_x + 66, y), _FONT, 0.36, color, 1, cv2.LINE_AA)

    if note:
        band = out.copy()
        cv2.rectangle(band, (0, 40), (out.shape[1], 70), (10, 10, 12), -1)
        cv2.addWeighted(band, 0.66, out, 0.34, 0, out)
        cv2.putText(out, note, (12, 60), _FONT, 0.5, (255, 215, 120), 1, cv2.LINE_AA)
    return out


def _header(frame: np.ndarray, text: str) -> np.ndarray:
    out = frame.copy()
    box = out.copy()
    cv2.rectangle(box, (0, 0), (out.shape[1], 34), (12, 12, 12), -1)
    cv2.addWeighted(box, 0.58, out, 0.42, 0, out)
    cv2.putText(out, text, (10, 23), _FONT, 0.6, (245, 245, 245), 1, cv2.LINE_AA)
    return out


def _wrap(text: str, width: int) -> list[str]:
    lines, line = [], ""
    for word in text.split(" "):
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    return lines + ([line] if line else [])


def _operator_card(frame: np.ndarray, operator: dict) -> np.ndarray:
    out = (frame.astype(np.float32) * 0.25).astype(np.uint8)
    lines = [("operator \"stow\", from 5 demonstrations", (245, 245, 245), 0.55)]
    for label, terms, color in [
        ("Pre", [t["key"] for t in operator["preconditions"]], _GOOD),
        ("Add", [t["key"] for t in operator["add_effects"]], (150, 200, 255)),
        ("Del", [t["key"] for t in operator["delete_effects"]], _BAD),
    ]:
        body = ", ".join(_short(k) for k in sorted(terms))
        for j, line in enumerate(_wrap(body, 42)):
            lines.append((f"{label + ':' if j == 0 else '':5s}{line}", color, 0.45))
    target = ", ".join(
        ("!" if t["polarity"] == "negative" else "") + _short(t["key"])
        for t in operator["inverse_target_terms"])
    lines.append(("", _DIM, 0.45))
    lines.append(("inverse target = restore Pre + Del, negate Add", (245, 245, 245), 0.48))
    for line in _wrap(target, 42):
        lines.append((f"     {line}", (255, 215, 120), 0.45))

    y = 60
    for text, color, scale in lines:
        cv2.putText(out, text, (24, y), _FONT, scale, color, 1, cv2.LINE_AA)
        y += 26 if scale >= 0.5 else 22
    return out


class Recorder(gym.Wrapper):
    """One annotated frame per control step."""

    def __init__(self, env):
        super().__init__(env)
        self.phase = ""
        self.regions = None
        self.registry = sp.registry()
        self.frames: list[np.ndarray] = []
        self.last_raw: np.ndarray | None = None

    def scores(self, obs) -> dict[str, float]:
        # reset() steps the env before main() can hand us the slots, and the
        # sampled sources are already in that first observation.
        if self.regions is None:
            self.regions = prim.regions(obs)
        scene = prim.obs_to_scene(obs, self.regions)
        return {k: float(r.score) for k, r in self.registry.evaluate_all(scene).items()}

    def annotate(self, obs, note: str | None = None) -> np.ndarray:
        frame = self.env.render()
        frame = np.asarray(frame.squeeze(0).cpu().numpy() if hasattr(frame, "cpu") else frame)
        self.last_raw = frame
        return _panel(_header(frame, self.phase), self.scores(obs), "predicate scores", note)

    def step(self, action):
        out = self.env.step(action)
        self.frames.append(self.annotate(out[0]))
        return out

    def hold(self, obs, note: str, n: int):
        self.frames.extend([self.annotate(obs, note)] * n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--operator-json", type=Path,
                    default=Path("artifacts/stow/stage1_operator.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/stow/videos"))
    args = ap.parse_args()

    rec = Recorder(prim.make_env(render_mode="rgb_array"))
    obs = prim.reset(rec, args.seed)
    rec.frames.clear()          # drop the reset's own steps
    start_scene = prim.obs_to_scene(obs, rec.regions)

    # Same sequence as stow_primitives.forward_stow, split so each push is labelled.
    rec.phase = "demonstration start"
    rec.hold(obs, "extractor reads this scene -> preconditions", 2 * args.fps)
    rec.phase = "push A into the pocket"
    obs = prim._push_along_x(rec, obs, "cube_a", geo.POCKET_A_XY[0], n_steps=40)
    rec.phase = "push B into the mouth"
    obs = prim._push_along_x(rec, obs, "cube_b", geo.MOUTH_B_XY[0], n_steps=50)
    rec.phase = "retract, open gripper"
    obs = prim.step_in_place(rec, obs, 8, gripper_cmd=1.0)
    rec.phase = "demonstration end"
    rec.hold(obs, "extractor reads this scene -> effects", 2 * args.fps)
    end_scene = prim.obs_to_scene(obs, rec.regions, timestep=1)

    if args.operator_json.exists():
        operator = json.loads(args.operator_json.read_text())["operator"]
    else:
        rollout = ForwardRollout(skill_name="stow", demo_id=f"demo_{args.seed}",
                                 scenes=[start_scene, end_scene])
        operator = OperatorExtractor(sp.registry()).extract("stow", [rollout]).operator.to_dict()
    card = _operator_card(rec.last_raw, operator)
    rec.frames.extend([card] * (4 * args.fps))
    rec.close()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"stow_stage1_seed{args.seed}"
    imageio.mimsave(f"{stem}.mp4", rec.frames, fps=args.fps, macro_block_size=8)
    imageio.imwrite(f"{stem}_start.png", rec.frames[args.fps])
    imageio.imwrite(f"{stem}_end.png", rec.frames[-4 * args.fps - 1])
    imageio.imwrite(f"{stem}_operator.png", card)
    print(f"{len(rec.frames)} frames -> {stem}.mp4 ({len(rec.frames) / args.fps:.1f} s)")


if __name__ == "__main__":
    main()
