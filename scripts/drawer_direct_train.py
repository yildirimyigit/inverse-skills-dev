"""Direct baseline, step 3: learn HOLE[drawer_open] with SAC, as Stage 3 learned the grasp.

Everything about the learner is Stage 3's: SAC with the default MlpPolicy and
the Stage 3 settings (read from the first Stage 3 run's config.json, so they
cannot drift), the training starts (seeds 0-31, offsets redrawn every episode),
the validation starts (seeds 500-509, used only to choose the checkpoint), the
budget (30,000 environment steps), the evaluation interval, and the framework's
reward with nothing added. Only the problem differs: the planner's unrestricted
hole, 100-step episodes and the 9-number observation (drawer_direct_env.py).

Success: the restoration goal, drawer_open and not drawer_closed, measured at
the end of the episode, as Stage 3 measured the grasp. After every validation
episode the arm is held still for 10 steps and the goal is measured again.
Nothing calls pull_open or any other scripted manipulation.

Never evaluated here: seeds 1000-1019 and 2000-2019 (used in earlier
development) and 3000-3049 (reserved for the final comparison).

Outputs under artifacts/drawer/direct/<run_id>/: config.json, summary.json,
sac_direct_best.zip and sac_direct_final.zip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs.drawer_direct_env import DrawerDirectEnv
from inverse_skills.envs.drawer_hole_env import load_spec

# SAC's small MLPs thrash across this machine's 20 cores: one thread is much faster.
torch.set_num_threads(1)

TRAIN_SEEDS = tuple(range(32))
VALIDATION_SEEDS = tuple(range(500, 510))
STAGE3_CONFIG = Path("artifacts/drawer/stage3/1792053076/config.json")


def rollout(env: DrawerDirectEnv, seeds, model) -> dict:
    """One deterministic episode per seed, each followed by the settling check."""
    episodes = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        while not done:
            obs, _reward, terminated, truncated, _info = env.step(model.predict(obs, deterministic=True)[0])
            done = terminated or truncated
        episode = {"seed": seed, **env.episodes[-1], **env.settle()}
        episodes.append(episode)
    return {"goal_rate": float(np.mean([e["goal_met"] for e in episodes])),
            "settled_goal_rate": float(np.mean([e["settled_goal_met"] for e in episodes])),
            "failures": dict(Counter(e["failure"] for e in episodes if e["failure"])),
            "max_opening_mm": max(e["max_opening_mm"] for e in episodes),
            "episodes": episodes}


class EvalBest(BaseCallback):
    """Every `every` steps: summarise the training episodes since the last report,
    evaluate on the validation starts, and keep the best checkpoint (as Stage 3)."""

    def __init__(self, train_env: DrawerDirectEnv, eval_env: DrawerDirectEnv, every: int, path: Path):
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env
        self.every = every
        self.path = path
        self.history: list[dict] = []
        self.best = -1.0
        self._seen = 0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every:
            return True
        recent = self.train_env.episodes[self._seen:]
        self._seen = len(self.train_env.episodes)
        validation = rollout(self.eval_env, VALIDATION_SEEDS, self.model)
        row = {
            "timesteps": self.num_timesteps,
            "train_episodes": len(recent),
            "train_goal_rate": float(np.mean([e["goal_met"] for e in recent])) if recent else 0.0,
            "train_reached_rate": float(np.mean([e["first_satisfied_step"] is not None for e in recent])) if recent else 0.0,
            "train_grasp_rate": float(np.mean([e["ever_grasped"] for e in recent])) if recent else 0.0,
            "train_max_opening_mm": max((e["max_opening_mm"] for e in recent), default=0.0),
            "train_failures": dict(Counter(e["failure"] for e in recent if e["failure"])),
            "validation_goal_rate": validation["goal_rate"],
            "validation_settled_goal_rate": validation["settled_goal_rate"],
            "validation_failures": validation["failures"],
            "validation_max_opening_mm": validation["max_opening_mm"],
            "validation_episodes": validation["episodes"],
        }
        if row["validation_goal_rate"] > self.best:
            self.best = row["validation_goal_rate"]
            self.model.save(self.path)
            row["saved"] = True
        self.history.append(row)
        print(f"  {self.num_timesteps:>6} steps | training episodes {row['train_episodes']:3d}: grasped "
              f"{row['train_grasp_rate']:.0%}, opened {row['train_reached_rate']:.0%}, held "
              f"{row['train_goal_rate']:.0%}, max opening {row['train_max_opening_mm']:.0f} mm | validation "
              f"goal {row['validation_goal_rate']:.0%} (after settling {row['validation_settled_goal_rate']:.0%}), "
              f"max opening {row['validation_max_opening_mm']:.0f} mm, failures {row['validation_failures']}"
              f"{'  <- best' if row.get('saved') else ''}", flush=True)
        return True


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    stage3 = json.loads(STAGE3_CONFIG.read_text())
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=stage3["timesteps"])
    ap.add_argument("--eval-every", type=int, default=stage3["eval_every"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=stage3["device"])
    ap.add_argument("--operator", type=Path, default=Path("artifacts/drawer/stage1_operator.json"))
    ap.add_argument("--spec", type=Path, default=Path("artifacts/drawer/direct/hole_spec_direct.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/drawer/direct"))
    args = ap.parse_args()
    spec = load_spec(args.spec)
    goal = dd.goal_from_operator(args.operator)
    sac_settings = stage3["sac"]

    train_env = DrawerDirectEnv(spec, goal, seed_pool=TRAIN_SEEDS)
    eval_env = DrawerDirectEnv(spec, goal)
    run_dir = args.out_dir / f"seed{args.seed}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "sac_direct_best"
    (run_dir / "config.json").write_text(json.dumps({
        "command": " ".join(sys.argv), "timesteps": args.timesteps, "eval_every": args.eval_every,
        "seed": args.seed, "device": args.device, "spec": str(args.spec), "sac": sac_settings,
        "sac_from": str(STAGE3_CONFIG), "episode_length": train_env.max_steps,
        "observation_size": int(train_env.observation_space.shape[0]),
        "start_offset_m": train_env.start_offset, "reward": "2 V(drawer_open) - 1, no shaping",
        "seeds": {"train": TRAIN_SEEDS, "validation": VALIDATION_SEEDS},
    }, indent=2))
    print(f"hole {spec['hole']}: start {spec['start']}, preserve {spec['preserve']}, keep false "
          f"{spec['preserve_absent']} | {train_env.max_steps}-step episodes | SAC {sac_settings}, seed {args.seed}")

    model = SAC("MlpPolicy", train_env, verbose=0, seed=args.seed, device=args.device, **sac_settings)
    evaluator = EvalBest(train_env, eval_env, args.eval_every, best_path)
    started = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=evaluator)
    minutes = (time.perf_counter() - started) / 60.0
    model.save(run_dir / "sac_direct_final")

    # When training first grasped, first opened the drawer, and first ended an episode with it open.
    episodes, length = train_env.episodes, train_env.max_steps

    def first(key) -> int | None:
        hit = next((e for e in episodes if key(e)), None)
        return None if hit is None else hit["end_step"] - length + (hit["first_satisfied_step"] or length)

    best_row = max(evaluator.history, key=lambda r: (r["validation_goal_rate"], -r["timesteps"]))
    summary = {
        "timesteps": args.timesteps,
        "minutes": minutes,
        "episode_length": length,
        "spec": spec,
        "training_interactions": train_env.total_steps,
        "training_episodes": len(episodes),
        "validation_interactions": len(evaluator.history) * len(VALIDATION_SEEDS) * length,
        "discovery": {
            "first_grasp_episode_ends_at_step": next((e["end_step"] for e in episodes if e["ever_grasped"]), None),
            "first_opening_at_step": first(lambda e: e["first_satisfied_step"] is not None),
            "first_held_episode_ends_at_step": next((e["end_step"] for e in episodes if e["goal_met"]), None),
            "training_episodes_grasped": sum(e["ever_grasped"] for e in episodes),
            "training_episodes_opened": sum(e["first_satisfied_step"] is not None for e in episodes),
            "training_max_opening_mm": max(e["max_opening_mm"] for e in episodes),
            "training_best_drawer_open_score": max(e["best_score"] for e in episodes),
            "training_failures": dict(Counter(e["failure"] for e in episodes if e["failure"])),
        },
        "best_checkpoint": {"timesteps": best_row["timesteps"], "validation_goal_rate": best_row["validation_goal_rate"],
                            "sha256": sha256(best_path.with_suffix(".zip"))},
        "final": {"timesteps": evaluator.history[-1]["timesteps"],
                  "validation_goal_rate": evaluator.history[-1]["validation_goal_rate"],
                  "validation_failures": evaluator.history[-1]["validation_failures"],
                  "sha256": sha256(run_dir / "sac_direct_final.zip")},
        "history": evaluator.history,
    }
    train_env.close()
    eval_env.close()
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\ntrained in {minutes:.1f} min: {summary['training_interactions']} steps, {len(episodes)} episodes | "
          f"best validation {best_row['validation_goal_rate']:.0%} at {best_row['timesteps']} steps, final "
          f"{summary['final']['validation_goal_rate']:.0%} | first opening at step "
          f"{summary['discovery']['first_opening_at_step']}\nwrote {run_dir}/summary.json")


if __name__ == "__main__":
    main()
