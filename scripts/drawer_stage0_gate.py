"""Stage 0 gate for the drawer: does the simulated drawer behave as designed?

Per seed, each test in a fresh, deterministic episode:

  1. push_close closes the open drawer and leaves the gripper upright.
  2. approach_handle reaches the standoff without moving the drawer, gripper upright.
     (Upright: within 5 degrees of pointing straight down. The position-only
     controller never corrects a tilt, so one would last the whole episode.)
  3. test_grasp (Stage 0 only) pinches the handle without moving the drawer.
  4. pull_open, after that grasp, opens the drawer.
  5. pull_open without a grasp, straight from the standoff, leaves it shut.
  6. handle_grasped agrees with the physics. From each of these states the
     drawer is pulled, and handle_grasped must say "grasped" exactly when the
     pull opens the drawer:
       near misses   open fingers around the handle, closed in the air at the
                     standoff, closed fist pressing on the handle's top, closed
                     fist pressing on its side;
       real grasps   closing 2.5 cm off-centre (the fingers centre themselves
                     on the fin), and a shallow pinch near the fin's top edge.
  Also measured, not gated:
  7. the inverse from a drawer left 4 cm ajar (does pull_open need the drawer
     to be closed first?).

Writes artifacts/drawer/stage0_gate.json and prints a table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inverse_skills.envs import drawer_env as geo
from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.envs.motion import step_in_place, step_toward

_CLOSED_MM = 5.0         # a pushed drawer must end within 5 mm of shut
_STILL_MM = 1.0          # approaching must move the drawer less than this
_GRASP_STILL_MM = 2.0    # grasping must move the drawer less than this
_OPENED_MM = 130.0       # a pull must reach at least this opening (target 150 mm)
_STANDOFF_MM = 10.0      # the approach must stop within 1 cm of the standoff on every axis
_UPRIGHT_DEG = 5.0       # the gripper must stay within this angle of pointing straight down
_REGISTRY = dp.registry()


def score(obs, key: str) -> float:
    return float(_REGISTRY.get(key).evaluate(prim.obs_to_scene(obs)).score)


def mm(metres: float) -> float:
    return round(metres * 1000.0, 1)


def tilt_deg(obs) -> float:
    """Angle between the fingers' pointing direction and straight down."""
    _, x, y, _ = obs["extra"]["tcp_pose"].squeeze()[3:7].cpu().numpy()   # quaternion w, x, y, z
    cos_tilt = 2 * (x * x + y * y) - 1   # from the TCP's z-axis, which points along the fingers
    return round(float(np.degrees(np.arccos(np.clip(cos_tilt, -1.0, 1.0)))), 1)


def closed_drawer_at_standoff(env, seed: int):
    """A fresh episode: push the drawer shut, then approach the handle."""
    obs = prim.push_close(env, prim.reset(env, seed))
    return prim.approach_handle(env, obs)


def descend_open(env, obs, offset=(0.0, 0.0, 0.0)):
    """From the standoff, bring the open gripper to the grasp point plus `offset`."""
    return step_toward(env, obs, prim.grasp_point(obs) + np.array(offset), 40, 0.003, 1.0, 0.5)


def close_gripper(env, obs):
    return step_in_place(env, obs, 15, gripper_cmd=-1.0)


def fist_on_top(env, obs):
    obs = close_gripper(env, obs)
    return step_toward(env, obs, prim.grasp_point(obs), 40, 0.003, -1.0, 0.5)


def fist_against_side(env, obs):
    obs = close_gripper(env, obs)
    point = prim.grasp_point(obs)
    obs = step_toward(env, obs, point + np.array([0.0, 0.03, 0.0]), 40, 0.003, -1.0, 0.5)
    return step_toward(env, obs, point + np.array([0.0, 0.005, 0.0]), 30, 0.002, -1.0, 0.3)


# name -> (how to get there from the standoff, is it physically a grasp?)
CASES = {
    "open_around": (descend_open, False),
    "closed_in_air": (close_gripper, False),
    "fist_on_top": (fist_on_top, False),
    "fist_against_side": (fist_against_side, False),
    "off_centre_25mm": (lambda env, obs: close_gripper(env, descend_open(env, obs, (0.0, 0.025, 0.0))), True),
    "shallow_20mm": (lambda env, obs: close_gripper(env, descend_open(env, obs, (0.0, 0.0, 0.02))), True),
}


def state(obs) -> dict:
    left, right = prim.pinch_force(obs)
    return {"handle_grasped": round(score(obs, "handle_grasped(drawer)"), 3),
            "pinch_left": round(left, 2), "pinch_right": round(right, 2),
            "width_mm": mm(prim.gripper_width(obs)),
            "tcp_to_grasp_point_mm": mm(float(np.linalg.norm(prim.tcp_pos(obs) - prim.grasp_point(obs))))}


def run_seed(env, seed: int) -> dict:
    out = {"seed": seed}

    # 1-4: push, approach, grasp, pull, in one episode
    obs = prim.reset(env, seed)
    out["start_opening_mm"] = mm(prim.opening(obs))
    obs = prim.push_close(env, obs)
    out["pushed_opening_mm"] = mm(prim.opening(obs))
    out["drawer_closed"] = round(score(obs, "drawer_closed(drawer)"), 3)
    out["tilt_after_push_deg"] = tilt_deg(obs)

    before = prim.opening(obs)
    obs = prim.approach_handle(env, obs)
    standoff = prim.grasp_point(obs) + np.array([0.0, 0.0, geo.STANDOFF_HEIGHT])
    out["standoff_error_mm"] = mm(float(np.max(np.abs(prim.tcp_pos(obs) - standoff))))
    out["approach_moved_mm"] = mm(abs(prim.opening(obs) - before))
    out["at_handle_standoff"] = round(score(obs, "at_handle_standoff(drawer)"), 3)
    out["tilt_at_approach_deg"] = tilt_deg(obs)

    before = prim.opening(obs)
    obs = prim.test_grasp(env, obs)
    out["grasp"] = state(obs)
    out["grasp_moved_mm"] = mm(abs(prim.opening(obs) - before))
    out["tilt_at_grasp_deg"] = tilt_deg(obs)

    obs = prim.pull_open(env, obs)
    out["pulled_opening_mm"] = mm(prim.opening(obs))
    out["drawer_open"] = round(score(obs, "drawer_open(drawer)"), 3)

    # 5: pull straight from the standoff, nothing held
    obs = prim.pull_open(env, closed_drawer_at_standoff(env, seed))
    out["no_grasp_pull_opening_mm"] = mm(prim.opening(obs))

    # 6: does handle_grasped agree with what a pull can do?
    cases = {}
    for name, (setup, really_grasped) in CASES.items():
        obs = setup(env, closed_drawer_at_standoff(env, seed))
        case = state(obs)
        obs = prim.pull_open(env, obs)
        case["pulled_opening_mm"] = mm(prim.opening(obs))
        case["drawer_open_after_pull"] = round(score(obs, "drawer_open(drawer)"), 3)
        case["expected_grasp"] = really_grasped
        cases[name] = case
    out["cases"] = cases

    # 7: the inverse from a drawer left ajar
    obs = prim.go_home(env, prim.reset(env, seed, opening=0.04))
    obs = prim.test_grasp(env, prim.approach_handle(env, obs))
    out["ajar_grasped"] = round(score(obs, "handle_grasped(drawer)"), 3)
    out["ajar_pull_opening_mm"] = mm(prim.opening(prim.pull_open(env, obs)))
    return out


def checks(row: dict) -> dict:
    cases = row["cases"].values()
    return {
        "push_closes": row["pushed_opening_mm"] <= _CLOSED_MM,
        "gripper_upright": max(row["tilt_after_push_deg"], row["tilt_at_approach_deg"]) <= _UPRIGHT_DEG,
        "approach_reaches_standoff": row["standoff_error_mm"] <= _STANDOFF_MM,
        "approach_leaves_drawer": row["approach_moved_mm"] <= _STILL_MM,
        "grasp_holds_handle": row["grasp"]["handle_grasped"] >= 0.8,
        "grasp_leaves_drawer": row["grasp_moved_mm"] <= _GRASP_STILL_MM,
        "pull_opens_after_grasp": row["pulled_opening_mm"] >= _OPENED_MM,
        "pull_fails_without_grasp": row["no_grasp_pull_opening_mm"] <= _CLOSED_MM,
        "near_misses_not_grasped": all(c["handle_grasped"] < 0.8 for c in cases if not c["expected_grasp"]),
        "near_misses_cannot_open": all(c["drawer_open_after_pull"] < 0.8 for c in cases if not c["expected_grasp"]),
        "grasped_iff_pull_opens": all((c["handle_grasped"] >= 0.8) == (c["drawer_open_after_pull"] >= 0.8)
                                      for c in cases),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")],
                    default=list(range(10)))
    ap.add_argument("--out", type=Path, default=Path("artifacts/drawer/stage0_gate.json"))
    args = ap.parse_args()

    env = prim.make_env()
    rows = []
    for seed in args.seeds:
        row = run_seed(env, seed)
        row["checks"] = checks(row)
        rows.append(row)
        misses = [c for c in row["cases"].values() if not c["expected_grasp"]]
        print(f"seed {seed}: start {row['start_opening_mm']:5.1f} -> pushed {row['pushed_opening_mm']:4.1f} mm | "
              f"approach moved {row['approach_moved_mm']:.1f} mm | "
              f"tilt {row['tilt_after_push_deg']:.1f}/{row['tilt_at_approach_deg']:.1f} deg | "
              f"grasp {row['grasp']['pinch_left']:4.1f}/{row['grasp']['pinch_right']:4.1f} N "
              f"score {row['grasp']['handle_grasped']:.2f} | pulled to {row['pulled_opening_mm']:5.1f} mm | "
              f"no-grasp pull {row['no_grasp_pull_opening_mm']:.1f} mm | "
              f"near misses: score <= {max(c['handle_grasped'] for c in misses):.2f}, "
              f"pull <= {max(c['pulled_opening_mm'] for c in misses):4.1f} mm | "
              f"ajar -> {row['ajar_pull_opening_mm']:5.1f} mm")
        failed = [name for name, ok in row["checks"].items() if not ok]
        if failed:
            print("   FAILED:", ", ".join(failed))
    env.close()

    summary = {name: sum(r["checks"][name] for r in rows) for name in rows[0]["checks"]}
    passed = all(count == len(rows) for count in summary.values())
    print("\nchecks passed (out of", len(rows), "seeds):")
    for name, count in summary.items():
        print(f"  {name:32s} {count}")
    print("\nSTAGE 0:", "PASS" if passed else "FAIL")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"passed": passed, "summary": summary, "seeds": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
