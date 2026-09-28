"""Measure where the learned hole operator actually works.

Sweeps the handoff pose around the nominal one and records success. The region
that comes back is what the operator advertises as its precondition, so the
planner can check it instead of assuming the policy works anywhere.

Writes precondition.json next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from inverse_skills.envs import stow_primitives as prim
from inverse_skills.envs.stow_hole_env import StowHoleEnv

torch.set_num_threads(1)

EVAL_SEEDS = tuple(range(1000, 1010))


def run_from_offset(env: StowHoleEnv, model, seed: int, dx_mm: float, dz_mm: float) -> bool:
    obs, _ = env.reset(seed=seed)
    offset = np.array([dx_mm / 1000.0, 0.0, dz_mm / 1000.0])
    env._obs = prim.step_toward(env._env, env._obs, prim.tcp_pos(env._obs) + offset,
                                20, 0.001, -1.0, 0.25)
    obs = env._observation(env._obs, env._scores(env._obs))
    done, info = False, {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return bool(info.get("postcondition")) and bool(info.get("fences_held"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--dx-mm", type=lambda s: [float(x) for x in s.split(",")],
                    default=[-30, -20, -10, 0, 10, 20, 30])
    ap.add_argument("--dz-mm", type=lambda s: [float(x) for x in s.split(",")],
                    default=[-20, 0, 20, 40, 60, 80])
    args = ap.parse_args()

    model = SAC.load(args.checkpoint, device="cuda")
    env = StowHoleEnv(seed_pool=EVAL_SEEDS, curriculum=1.0)

    print("success rate over handoff offsets (dx across, dz down);")
    print("dx is along the drag, dz is above the seated pose\n")
    print("  dz\\dx " + " ".join(f"{dx:>6.0f}" for dx in args.dx_mm))
    grid = []
    for dz in args.dz_mm:
        row = []
        for dx in args.dx_mm:
            rate = float(np.mean([run_from_offset(env, model, s, dx, dz) for s in EVAL_SEEDS]))
            row.append(rate)
        grid.append({"dz_mm": dz, "rates": row})
        print(f"  {dz:>5.0f} " + " ".join(f"{r:>5.0%} " for r in row))
    env.close()

    # The advertised precondition: the largest box around the nominal handoff
    # where every cell is reliable.
    reliable = 0.9
    dz_ok = [g["dz_mm"] for g in grid if all(r >= reliable for r in g["rates"])]
    dx_ok = [dx for i, dx in enumerate(args.dx_mm)
             if all(g["rates"][i] >= reliable for g in grid)]
    print(f"\nreliable (>= {reliable:.0%}) for every offset tested:"
          f" dz in {dz_ok or 'none'}, dx in {dx_ok or 'none'}")

    out = args.checkpoint.parent / "precondition.json"
    out.write_text(json.dumps({
        "dx_mm": args.dx_mm, "dz_mm": args.dz_mm, "grid": grid,
        "reliable_threshold": reliable, "dz_all_dx_reliable": dz_ok,
        "dx_all_dz_reliable": dx_ok, "seeds": list(EVAL_SEEDS),
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
