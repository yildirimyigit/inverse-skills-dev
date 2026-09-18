"""Stage 0 gate for the wedged-cube PoC: does the scene behave as designed?

Per seed, three fresh episodes:
  1. forward stow        -> A seated in the pocket, B in the mouth (within 1 cm)
  2. pick(A) while stowed -> must NOT lift A (walls block both grasp axes)
  3. scripted inverse with a scripted drag standing in for the learned hole:
     pick B, place B at src B (<= 1.5 cm), approach A, drag A out
     (x_A <= X_CLEAR), pick A (must lift now), place A at src A.

Writes artifacts/stow/stage0_gate.json and prints a per-seed table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inverse_skills.envs import stow_cube as geo
from inverse_skills.envs import stow_primitives as prim

_STOW_TOL = 0.01
_PLACE_TOL = 0.015
_LIFTED_Z = 0.05


def _xy_err(pos, xy) -> float:
    return float(np.linalg.norm(np.asarray(pos[:2]) - np.asarray(xy[:2])))


def run_seed(env, seed: int) -> dict:
    out = {"seed": seed}

    # 1. forward stow
    obs = prim.forward_stow(env, prim.reset(env, seed))
    a, b = prim.cube_pos(obs, "cube_a"), prim.cube_pos(obs, "cube_b")
    out["src_a"] = prim.src_pos(obs, "cube_a")[:2].round(4).tolist()
    out["src_b"] = prim.src_pos(obs, "cube_b")[:2].round(4).tolist()
    out["stow_a_err"] = _xy_err(a, geo.POCKET_A_XY)
    out["stow_b_err"] = _xy_err(b, geo.MOUTH_B_XY)
    out["stow_ok"] = out["stow_a_err"] <= _STOW_TOL and out["stow_b_err"] <= _STOW_TOL

    # 2. pick(A) while stowed must fail
    obs = prim.forward_stow(env, prim.reset(env, seed))
    obs = prim.lift(env, prim.pick(env, obs, "cube_a"))
    out["stowed_pick_a_lifted"] = bool(prim.cube_pos(obs, "cube_a")[2] > _LIFTED_Z)

    # 3. scripted inverse, drag in place of the hole
    obs = prim.forward_stow(env, prim.reset(env, seed))
    obs = prim.lift(env, prim.pick(env, obs, "cube_b"))
    out["pick_b_lifted"] = bool(prim.cube_pos(obs, "cube_b")[2] > _LIFTED_Z)
    obs = prim.place(env, obs, "cube_b", prim.src_pos(obs, "cube_b"))
    out["place_b_err"] = _xy_err(prim.cube_pos(obs, "cube_b"), prim.src_pos(obs, "cube_b"))
    a_before = prim.cube_pos(obs, "cube_a")
    obs = prim.approach(env, obs, "cube_a")
    out["approach_a_moved"] = _xy_err(prim.cube_pos(obs, "cube_a"), a_before)
    obs = prim.drag_out(env, obs, "cube_a", geo.X_CLEAR - 0.01)
    a = prim.cube_pos(obs, "cube_a")
    out["drag_a_x"], out["drag_a_y"] = float(a[0]), float(a[1])
    out["drag_ok"] = bool(a[0] <= geo.X_CLEAR)
    obs = prim.lift(env, prim.pick(env, obs, "cube_a"))
    out["dragged_pick_a_lifted"] = bool(prim.cube_pos(obs, "cube_a")[2] > _LIFTED_Z)
    obs = prim.place(env, obs, "cube_a", prim.src_pos(obs, "cube_a"))
    out["place_a_err"] = _xy_err(prim.cube_pos(obs, "cube_a"), prim.src_pos(obs, "cube_a"))
    out["b_still_at_src"] = _xy_err(prim.cube_pos(obs, "cube_b"), prim.src_pos(obs, "cube_b")) <= _PLACE_TOL
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")],
                    default=list(range(10)))
    ap.add_argument("--out", type=Path, default=Path("artifacts/stow/stage0_gate.json"))
    args = ap.parse_args()

    env = prim.make_env()
    rows = []
    print(f"{'seed':>4} {'stowA':>6} {'stowB':>6} {'pickA@stow':>10} {'pickB':>6} {'placeB':>7} "
          f"{'apprA':>6} {'dragX':>7} {'dragY':>7} {'pickA@drag':>10} {'placeA':>7}")
    for seed in args.seeds:
        r = run_seed(env, seed)
        rows.append(r)
        print(f"{seed:>4} {r['stow_a_err']*1000:>5.1f}mm {r['stow_b_err']*1000:>5.1f}mm "
              f"{'LIFTED' if r['stowed_pick_a_lifted'] else 'blocked':>10} "
              f"{'ok' if r['pick_b_lifted'] else 'FAIL':>6} "
              f"{r['place_b_err']*1000:>5.1f}mm {r['approach_a_moved']*1000:>4.1f}mm "
              f"{r['drag_a_x']*1000:>5.0f}mm {r['drag_a_y']*1000:>+5.0f}mm "
              f"{'lifted' if r['dragged_pick_a_lifted'] else 'FAILED':>10} {r['place_a_err']*1000:>5.1f}mm")
    env.close()

    n = len(rows)
    # The checks after the drag are only meaningful when the drag succeeded:
    # a failed drag leaves A in the pocket and the rest of the sequence acts
    # on nothing, so those rows are scored separately.
    dragged = [r for r in rows if r["drag_ok"]]
    summary = {
        "n": n,
        "stow_ok": sum(r["stow_ok"] for r in rows),
        "stowed_pick_a_blocked": sum(not r["stowed_pick_a_lifted"] for r in rows),
        "pick_b_lifted": sum(r["pick_b_lifted"] for r in rows),
        "place_b_ok": sum(r["place_b_err"] <= _PLACE_TOL for r in rows),
        "approach_ok": sum(r["approach_a_moved"] <= 0.005 for r in rows),
        "drag_ok": len(dragged),
        "after_drag_pick_a_lifted": sum(r["dragged_pick_a_lifted"] for r in dragged),
        "after_drag_place_a_ok": sum(r["place_a_err"] <= _PLACE_TOL for r in dragged),
        "after_drag_b_still_at_src": sum(r["b_still_at_src"] for r in dragged),
    }
    print("\nsummary:", json.dumps(summary))
    gate = (summary["stow_ok"] == n and summary["stowed_pick_a_blocked"] == n
            and summary["pick_b_lifted"] == n and summary["place_b_ok"] == n
            and summary["approach_ok"] == n
            and summary["drag_ok"] >= n - 1
            and summary["after_drag_pick_a_lifted"] == len(dragged)
            and summary["after_drag_b_still_at_src"] == len(dragged))
    print("GATE:", "PASS" if gate else "FAIL")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "gate_pass": gate, "seeds": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
