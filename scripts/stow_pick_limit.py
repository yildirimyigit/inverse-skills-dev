"""Where does pick(cube_a) actually stop working?

`clear_of_walls` draws a line at X_CLEAR, and everything downstream — the
planner's precondition, the hole's postcondition, the library-only ablation —
rests on that line being the real grasp limit. This measures it by placing A
at a series of positions and trying the scripted pick, instead of inferring it
from a drag that overshoots its target.

Writes artifacts/stow/pick_limit.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sapien
import torch

from inverse_skills.envs import stow_cube as geo
from inverse_skills.envs import stow_primitives as prim

_LIFTED_Z = 0.05


def try_pick_at(env, seed: int, x_mm: float) -> bool:
    obs = prim.reset(env, seed)
    base = env.unwrapped
    y = float(prim.src_pos(obs, "cube_a")[1])
    base.cube_a.set_pose(sapien.Pose(p=[x_mm / 1000.0, y, geo.CUBE_HALF]))
    # Park B well clear so only the pocket can interfere.
    base.cube_b.set_pose(sapien.Pose(p=[-0.25, -0.2, geo.CUBE_HALF]))
    for _ in range(5):
        obs, *_ = env.step(torch.tensor(np.array([0, 0, 0, 1], dtype=np.float32)))
    obs = prim.lift(env, prim.pick(env, obs, "cube_a"))
    return bool(prim.cube_pos(obs, "cube_a")[2] > _LIFTED_Z)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-mm", type=lambda s: [float(v) for v in s.split(",")],
                    default=[60, 70, 80, 90, 95, 100, 105, 110, 115, 120, 130])
    ap.add_argument("--seeds", type=lambda s: [int(v) for v in s.split(",")],
                    default=[1000, 1001, 1002, 1003, 1004])
    ap.add_argument("--out", type=Path, default=Path("artifacts/stow/pick_limit.json"))
    args = ap.parse_args()

    env = prim.make_env()
    rows = []
    print(f"{'cube x':>7} {'pick lifts':>11}   (arm tips at {geo.POCKET_BACK_X * 1000 - 45:.0f}mm, "
          f"X_CLEAR = {geo.X_CLEAR * 1000:.0f}mm)")
    for x_mm in args.x_mm:
        rate = float(np.mean([try_pick_at(env, s, x_mm) for s in args.seeds]))
        rows.append({"x_mm": x_mm, "lift_rate": rate})
        print(f"{x_mm:>6.0f}mm {rate:>10.0%}")
    env.close()

    reliable = [r["x_mm"] for r in rows if r["lift_rate"] >= 0.99]
    limit = max(reliable) if reliable else float("nan")
    print(f"\nreliable grasp up to x = {limit:.0f}mm")
    print(f"for V>=0.8 exactly at that limit, X_CLEAR should be "
          f"{limit + 0.693 * 20:.0f}mm (T = 20mm)")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows, "reliable_limit_mm": limit,
                                    "seeds": args.seeds}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
