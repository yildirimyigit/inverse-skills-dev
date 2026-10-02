"""Stage 3 video: the learned hole policy, against the scripted drag.

Same seed, same handoff, two controllers. The panel tracks the predicate the
hole is responsible for and the fence it must not break, so the difference is
visible rather than asserted: the scripted drag reaches the goal and shoves B
off its source on the way; the policy stops once the predicate holds.

Output: artifacts/stow/videos/stow_stage3_seed<seed>.mp4 and stills.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from stable_baselines3 import SAC

from inverse_skills.envs import stow_primitives as prim
from inverse_skills.envs.stow_hole_env import StowHoleEnv, load_spec

torch.set_num_threads(1)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GOOD = (120, 220, 130)
_BAD = (235, 120, 110)
_MID = (200, 200, 200)
_ACCENT = (255, 215, 120)
_DRAG = np.array([-0.2 / 0.3, -0.05 / 0.15], dtype=np.float32)


def _short(key: str) -> str:
    return key.replace("cube_a", "A").replace("cube_b", "B")


def _panel(frame, title, rows, note=None):
    out = frame.copy()
    box = out.copy()
    cv2.rectangle(box, (0, 0), (out.shape[1], 34), (12, 12, 12), -1)
    cv2.addWeighted(box, 0.58, out, 0.42, 0, out)
    cv2.putText(out, title, (10, 23), _FONT, 0.58, (245, 245, 245), 1, cv2.LINE_AA)

    pad, row_h = 10, 19
    height = 26 + row_h * len(rows)
    top = out.shape[0] - height - pad
    box = out.copy()
    cv2.rectangle(box, (pad, top), (320, out.shape[0] - pad), (10, 10, 12), -1)
    cv2.addWeighted(box, 0.62, out, 0.38, 0, out)
    for i, (label, value, color) in enumerate(rows):
        y = top + 24 + i * row_h
        cv2.putText(out, label, (pad + 8, y), _FONT, 0.42, _MID, 1, cv2.LINE_AA)
        cv2.putText(out, value, (pad + 200, y), _FONT, 0.42, color, 1, cv2.LINE_AA)
    if note:
        band = out.copy()
        cv2.rectangle(band, (0, 40), (out.shape[1], 70), (10, 10, 12), -1)
        cv2.addWeighted(band, 0.66, out, 0.34, 0, out)
        cv2.putText(out, note, (12, 60), _FONT, 0.48, _ACCENT, 1, cv2.LINE_AA)
    return out


def episode(env: StowHoleEnv, seed: int, model, title: str, fps: int):
    """Run one episode, returning annotated frames and the outcome."""
    frames = []
    obs, _ = env.reset(seed=seed)

    def grab(note=None):
        frame = env._env.render()
        frame = np.asarray(frame.squeeze(0).cpu().numpy() if hasattr(frame, "cpu") else frame)
        scores = env._scores(env._obs)
        active = scores[env.achieve]
        rows = [(_short(env.achieve), f"{active:.2f}" + ("  HELD" if active >= 0.8 else ""),
                 _GOOD if active >= 0.8 else _MID)]
        for key in env.spec["preserve"]:
            held = scores[key] >= 0.8
            rows.append(("fence: " + _short(key), f"{scores[key]:.2f}" + ("" if held else "  BROKEN"),
                         _GOOD if held else _BAD))
        for cube in ("cube_a", "cube_b"):
            rows.append((f"{_short(cube)} x", f"{prim.cube_pos(env._obs, cube)[0] * 1000:+.0f} mm", _MID))
        frames.append(_panel(frame, title, rows, note))

    grab("handoff: seated on A, after pick B / place B / approach A")
    frames.extend([frames[-1]] * (fps - 1))
    done = False
    info = {}
    while not done:
        action = _DRAG if model is None else model.predict(obs, deterministic=True)[0]
        obs, _r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        grab()
    grab("episode end")
    frames.extend([frames[-1]] * (2 * fps))
    return frames, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--spec", type=Path, default=Path("artifacts/stow/hole_spec.json"))
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/stow/videos"))
    args = ap.parse_args()

    model = SAC.load(args.checkpoint, device="cuda")
    env = StowHoleEnv(load_spec(args.spec), seed_pool=(args.seed,), curriculum=1.0,
                      render_mode="rgb_array")

    scripted, scripted_info = episode(env, args.seed, None, "scripted drag (Stage 0)", args.fps)
    learned, learned_info = episode(env, args.seed, model, "learned hole policy (SAC)", args.fps)
    env.close()

    def summarize(info):
        return (f"{'reached' if info['postcondition'] else 'missed'}, "
                f"{info['first_satisfied_step'] or '-'} steps, "
                f"fence {'held' if info['fences_held'] else 'BROKEN'}")

    card = (learned[-1].astype(np.float32) * 0.25).astype(np.uint8)
    lines = [
        ("one predicate, two controllers", (245, 245, 245), 0.55),
        ("", _MID, 0.45),
        ("scripted drag: " + summarize(scripted_info), _BAD, 0.46),
        ("learned policy: " + summarize(learned_info), _GOOD, 0.46),
        ("", _MID, 0.45),
        ("the fence term is the difference: the policy", _MID, 0.44),
        (f"stops once {_short(env.achieve)} holds, instead of", _MID, 0.44),
        ("dragging A onward into B's source", _MID, 0.44),
    ]
    y = 60
    for text, color, scale in lines:
        cv2.putText(card, text, (24, y), _FONT, scale, color, 1, cv2.LINE_AA)
        y += 28 if scale >= 0.5 else 24

    frames = scripted + learned + [card] * (4 * args.fps)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_dir / f"stow_stage3_seed{args.seed}"
    imageio.mimsave(f"{stem}.mp4", frames, fps=args.fps, macro_block_size=8)
    imageio.imwrite(f"{stem}_scripted_end.png", scripted[-1])
    imageio.imwrite(f"{stem}_learned_end.png", learned[-1])
    imageio.imwrite(f"{stem}_card.png", card)
    print(f"scripted: {summarize(scripted_info)}")
    print(f"learned:  {summarize(learned_info)}")
    print(f"{len(frames)} frames -> {stem}.mp4 ({len(frames) / args.fps:.1f} s)")


if __name__ == "__main__":
    main()
