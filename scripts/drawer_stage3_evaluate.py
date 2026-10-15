"""Evaluate a Stage 3 run the way the executive will run the learned grasp.

For each test start (seeds 1000-1019, never used for training or checkpoint
selection): run the run's selected checkpoint (sac_hole_best) until
handle_grasped >= 0.8 or 40 steps have passed, and stop there. Record the
drawer's opening and the hand's tilt at that moment, then run the scripted
pull_open and check that the drawer counts as open (drawer_open >= 0.8).

Writes <run_dir>/evaluation.json and prints a summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.envs.drawer_hole_env import SATISFIED, DrawerHoleEnv, load_spec

torch.set_num_threads(1)

TEST_SEEDS = tuple(range(1000, 1020))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--spec", type=Path, default=Path("artifacts/drawer/hole_spec.json"))
    args = ap.parse_args()

    model = SAC.load(args.run_dir / "sac_hole_best", device="cpu")
    env = DrawerHoleEnv(load_spec(args.spec))
    episodes = []
    for seed in TEST_SEEDS:
        obs, _ = env.reset(seed=seed)
        for step in range(1, env.max_steps + 1):
            obs, _reward, _terminated, _truncated, info = env.step(model.predict(obs, deterministic=True)[0])
            if info["postcondition"]:
                break                       # the executive stops the skill here
        episode = {
            "seed": seed,
            "grasped": bool(info["postcondition"]),
            "steps": step,
            "opening_at_stop_mm": round(prim.opening(env._obs) * 1000, 1),
            "tilt_at_stop_deg": prim.tilt_deg(env._obs),
        }
        after = prim.pull_open(env._env, env._obs)
        episode["pulled_opening_mm"] = round(prim.opening(after) * 1000, 1)
        episode["pull_opened"] = bool(env._scores(after)["drawer_open(drawer)"] >= SATISFIED)
        episodes.append(episode)
    env.close()

    grasped = [e for e in episodes if e["grasped"]]
    steps = sorted(e["steps"] for e in grasped)
    summary = {
        "checkpoint": "sac_hole_best",
        "test_seeds": TEST_SEEDS,
        "grasp_rate": len(grasped) / len(episodes),
        "pull_open_rate": float(np.mean([e["pull_opened"] for e in episodes])),
        "steps_to_grasp": [steps[0], steps[len(steps) // 2], steps[-1]] if steps else None,
        "max_opening_at_stop_mm": max(e["opening_at_stop_mm"] for e in episodes),
        "max_tilt_at_stop_deg": max(e["tilt_at_stop_deg"] for e in episodes),
        "episodes": episodes,
    }
    (args.run_dir / "evaluation.json").write_text(json.dumps(summary, indent=2))
    print(f"{args.run_dir.name}: grasp {len(grasped)}/{len(episodes)}, steps to grasp "
          f"(min/median/max) {summary['steps_to_grasp']}, at the stop: drawer <= "
          f"{summary['max_opening_at_stop_mm']} mm open, tilt <= {summary['max_tilt_at_stop_deg']} deg | "
          f"pull_open then opens it {sum(e['pull_opened'] for e in episodes)}/{len(episodes)}")


if __name__ == "__main__":
    main()
