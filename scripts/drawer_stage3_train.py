"""Stage 3 for the drawer: learn the hole's policy (the handle grasp) with SAC.

Trains on DrawerHoleEnv, built from the hole spec Stage 2 wrote, with the
framework's reward and nothing added. The SAC settings are the stow task's.

Three disjoint sets of start states (scene seed, plus a random TCP offset of up
to 1 cm around the standoff):
  training     seeds 0-31, offsets redrawn every episode
  validation   seeds 500-509, used only to pick the best checkpoint
  test         seeds 1000-1019, used only for the final report

Success means handle_grasped >= 0.8 at the end of the 40-step episode, the
physically validated condition from Stage 0. On the test seeds every episode is
then followed by the scripted pull_open, to see whether the learned grasp
really lets the drawer be pulled open.

Two reference points on the test seeds: random actions (what SAC's first
exploration sees) and a hand-coded "descend, then close" policy (the task is
solvable through this action space).

Outputs under artifacts/drawer/stage3/<run_id>/.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.envs.drawer_hole_env import SATISFIED, DrawerHoleEnv, load_spec

# SAC's small MLPs thrash across this machine's 20 cores: one thread is much faster.
torch.set_num_threads(1)

TRAIN_SEEDS = tuple(range(32))
VALIDATION_SEEDS = tuple(range(500, 510))
TEST_SEEDS = tuple(range(1000, 1020))
SAC_SETTINGS = {"learning_starts": 1_000, "batch_size": 256, "train_freq": 1, "gradient_steps": 1}


def random_policy(rng):
    return lambda obs: rng.uniform(-1.0, 1.0, size=4)


def descend_then_close(obs):
    """Hand-coded reference: centre over the grasp point, go down, close once there."""
    height_cm = obs[3] * 5.0
    action = [-2.0 * obs[1], -2.0 * obs[2], -height_cm, 1.0 if height_cm > 0.3 else -1.0]
    return np.clip(action, -1.0, 1.0)


def sac_policy(model):
    return lambda obs: model.predict(obs, deterministic=True)[0]


def rollout(env: DrawerHoleEnv, seeds, policy, pull: bool = False) -> dict:
    """One episode per seed. With `pull`, run the scripted pull_open afterwards."""
    episodes = []
    for seed in seeds:
        obs, start = env.reset(seed=seed)
        done, info = False, {}
        while not done:
            obs, _reward, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
        left, right = prim.pinch_force(env._obs)
        episode = {
            "seed": seed,
            "start_offset_mm": start["start_offset_mm"],
            "success": bool(info["postcondition"]),
            "final_score": round(info["scores"][env.achieve], 3),
            "first_satisfied_step": info["first_satisfied_step"],
            "pinch_n": [round(left, 1), round(right, 1)],
            "tcp_to_grasp_point_mm": round(float(np.linalg.norm(
                prim.tcp_pos(env._obs) - prim.grasp_point(env._obs))) * 1000, 1),
        }
        if pull:
            after = prim.pull_open(env._env, env._obs)
            scores = env._scores(after)
            episode["pulled_opening_mm"] = round(prim.opening(after) * 1000, 1)
            episode["drawer_open_after_pull"] = scores["drawer_open(drawer)"] >= SATISFIED
        episodes.append(episode)
    summary = {"success_rate": float(np.mean([e["success"] for e in episodes])), "episodes": episodes}
    if pull:
        summary["pull_open_rate"] = float(np.mean([e["drawer_open_after_pull"] for e in episodes]))
    return summary


class EvalBest(BaseCallback):
    """Every `every` steps: report the training episodes since the last report,
    evaluate on the validation seeds, and keep the best checkpoint."""

    def __init__(self, train_env: DrawerHoleEnv, eval_env: DrawerHoleEnv, every: int, path: Path):
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
        validation = rollout(self.eval_env, VALIDATION_SEEDS, sac_policy(self.model))
        row = {
            "timesteps": self.num_timesteps,
            "train_episodes": len(recent),
            "train_success_rate": float(np.mean([e["success"] for e in recent])) if recent else 0.0,
            "train_reached_rate": float(np.mean([e["best_score"] >= SATISFIED for e in recent])) if recent else 0.0,
            "train_mean_best_score": float(np.mean([e["best_score"] for e in recent])) if recent else 0.0,
            "validation_success_rate": validation["success_rate"],
        }
        if row["validation_success_rate"] > self.best:
            self.best = row["validation_success_rate"]
            self.model.save(self.path)
            row["saved"] = True
        self.history.append(row)
        print(f"  {self.num_timesteps:>6} steps | training episodes: {row['train_episodes']:3d}, "
              f"grasp reached {row['train_reached_rate']:.0%}, held at the end {row['train_success_rate']:.0%}, "
              f"mean best score {row['train_mean_best_score']:.2f} | validation success "
              f"{row['validation_success_rate']:.0%}{'  <- best' if row.get('saved') else ''}")
        return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=30_000)
    ap.add_argument("--eval-every", type=int, default=2_500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--spec", type=Path, default=Path("artifacts/drawer/hole_spec.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/drawer/stage3"))
    args = ap.parse_args()
    spec = load_spec(args.spec)
    print(f"hole: achieve {spec['achieve']}, start {spec['start']}, preserve {spec['preserve']}, "
          f"keep false {spec['preserve_absent']}")

    run_dir = args.out_dir / str(int(time.time()))
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "sac_hole_best"
    (run_dir / "config.json").write_text(json.dumps({
        "command": " ".join(sys.argv), "timesteps": args.timesteps, "eval_every": args.eval_every,
        "seed": args.seed, "device": args.device, "spec": str(args.spec), "sac": SAC_SETTINGS,
        "seeds": {"train": TRAIN_SEEDS, "validation": VALIDATION_SEEDS, "test": TEST_SEEDS},
    }, indent=2))

    train_env = DrawerHoleEnv(spec, seed_pool=TRAIN_SEEDS)
    eval_env = DrawerHoleEnv(spec)

    print("reference points on the test seeds")
    references = {
        "random_actions": rollout(eval_env, TEST_SEEDS, random_policy(np.random.default_rng(args.seed))),
        "descend_then_close": rollout(eval_env, TEST_SEEDS, descend_then_close, pull=True),
    }
    for name, result in references.items():
        print(f"  {name:20s} success {result['success_rate']:.0%}")

    model = SAC("MlpPolicy", train_env, verbose=0, seed=args.seed, device=args.device, **SAC_SETTINGS)
    evaluator = EvalBest(train_env, eval_env, args.eval_every, best_path)
    print(f"training SAC for {args.timesteps} steps ({args.device})")
    started = time.perf_counter()
    model.learn(total_timesteps=args.timesteps, callback=evaluator)
    minutes = (time.perf_counter() - started) / 60.0
    model.save(run_dir / "sac_hole_final")

    results = {}
    if best_path.with_suffix(".zip").exists():
        best = SAC.load(best_path, device=args.device)
        results["best_checkpoint"] = rollout(eval_env, TEST_SEEDS, sac_policy(best), pull=True)
    results["final_model"] = rollout(eval_env, TEST_SEEDS, sac_policy(model), pull=True)
    print(f"\ntest seeds {TEST_SEEDS[0]}-{TEST_SEEDS[-1]} (never used for training or selection)")
    for name, result in results.items():
        print(f"  {name:16s} grasp success {result['success_rate']:.0%}, "
              f"pull_open then opens the drawer {result['pull_open_rate']:.0%}")
    # When training first produced a grasp, and first ended an episode holding one.
    reached = [e["end_step"] for e in train_env.episodes if e["best_score"] >= SATISFIED]
    held = [e["end_step"] for e in train_env.episodes if e["success"]]
    discovery = {"first_grasp_episode_ends_at_step": reached[0] if reached else None,
                 "first_held_episode_ends_at_step": held[0] if held else None}
    print(f"first training episode with a grasp ended at step {discovery['first_grasp_episode_ends_at_step']}")
    train_env.close()
    eval_env.close()

    print(f"\ntrained in {minutes:.1f} min")
    (run_dir / "summary.json").write_text(json.dumps({
        "timesteps": args.timesteps,
        "minutes": minutes,
        "spec": spec,
        "seeds": {"train": TRAIN_SEEDS, "validation": VALIDATION_SEEDS, "test": TEST_SEEDS},
        "references": references,
        "history": evaluator.history,
        "discovery": discovery,
        "test": results,
    }, indent=2))
    print(f"wrote {run_dir}/summary.json")


if __name__ == "__main__":
    main()
