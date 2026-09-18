"""Stage 3: learn the hole's policy with SAC, then measure its precondition.

Trains on `StowHoleEnv`, evaluates the best checkpoint on held-out seeds, and
sweeps the handoff pose to find where the learned operator actually works —
that success region is what the operator advertises as its precondition, and
what Stage 4's planner may rely on.

Outputs under artifacts/stow/stage3/<run_id>/.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from inverse_skills.envs import stow_cube as geo
from inverse_skills.envs import stow_primitives as prim
from inverse_skills.envs.stow_hole_env import StowHoleEnv

# SAC's small MLPs thrash across this machine's 20 cores: 129 ms/step at the
# default thread count against 10 ms at one thread.
torch.set_num_threads(1)

EVAL_SEEDS = tuple(range(1000, 1010))
_DRAG = np.array([-0.2 / 0.3, -0.05 / 0.15], dtype=np.float32)   # the Stage 0 drag commands
_PRESS = np.array([0.0, -1.0], dtype=np.float32)


def expert_action(env: StowHoleEnv) -> np.ndarray:
    """Reference controller: press until the fingertips bite, then drag.

    A constant action cannot do this from the hover the executive hands over
    at, so the baseline has to be state-dependent like the policy.
    """
    tcp = prim.tcp_pos(env._obs)
    cube = prim.cube_pos(env._obs, "cube_a")
    contact_z = cube[2] + geo.CUBE_HALF + prim._FINGERTIP_OFFSET
    return _PRESS if tcp[2] > contact_z - 0.005 else _DRAG


class Curriculum(BaseCallback):
    """Widen the handoff from fingertips-on-cube to the full hover."""

    def __init__(self, env: StowHoleEnv, schedule_steps: int):
        super().__init__()
        self.env = env
        self.schedule_steps = max(1, schedule_steps)

    def _on_step(self) -> bool:
        self.env.set_curriculum(min(1.0, self.num_timesteps / self.schedule_steps))
        return True


class EvalBest(BaseCallback):
    """Evaluate periodically and keep the best checkpoint by success rate."""

    def __init__(self, env: StowHoleEnv, every: int, path: Path):
        super().__init__()
        self.env = env
        self.every = every
        self.path = path
        self.history: list[dict] = []
        self.best = -1.0

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every:
            return True
        stats = rollout(self.env, EVAL_SEEDS, policy=self.model)
        stats["timesteps"] = self.num_timesteps
        self.history.append(stats)
        if stats["success_rate"] > self.best or (
                stats["success_rate"] == self.best and stats["mean_steps"] < self._best_steps()):
            self.best = stats["success_rate"]
            self.model.save(self.path)
            stats["saved"] = True
        print(f"  {self.num_timesteps:>7} steps: success {stats['success_rate']:.0%}  "
              f"steps {stats['mean_steps']:.1f}  x_A {stats['mean_x_mm']:.0f}mm"
              f"{'  <- best' if stats.get('saved') else ''}")
        return True

    def _best_steps(self) -> float:
        saved = [h["mean_steps"] for h in self.history[:-1] if h.get("saved")]
        return saved[-1] if saved else float("inf")


def rollout(env: StowHoleEnv, seeds, policy=None, curriculum: float = 1.0) -> dict:
    """Run one episode per seed; `policy` None means the scripted expert action."""
    previous = env.curriculum
    env.set_curriculum(curriculum)
    successes, steps, finals, fences = [], [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done = False
        info = {}
        n = 0
        while not done and n < env.max_steps:
            if policy is None:
                action = expert_action(env)
            else:
                action, _ = policy.predict(obs, deterministic=True)
            obs, _r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            n += 1
        successes.append(bool(info.get("postcondition")))
        fences.append(bool(info.get("fences_held")))
        steps.append(n)
        finals.append(info.get("cube_a_x", float("nan")) * 1000.0)
    env.set_curriculum(previous)
    return {
        "success_rate": float(np.mean(successes)),
        "fence_rate": float(np.mean(fences)),
        "mean_steps": float(np.mean(steps)),
        "mean_x_mm": float(np.mean(finals)),
    }


def precondition_sweep(env: StowHoleEnv, model, seeds, offsets_mm) -> list[dict]:
    """Success over handoff TCP offsets: the region the operator can advertise."""
    rows = []
    for dz_mm in offsets_mm:
        successes = []
        for seed in seeds:
            obs, _ = env.reset(seed=seed)
            # Shift the start pose away from the nominal handoff.
            env._obs = prim.step_toward(env._env, env._obs,
                                        prim.tcp_pos(env._obs) + np.array([0, 0, dz_mm / 1000.0]),
                                        10, 0.001, -1.0, 0.25)
            obs = env._observation(env._obs, env._scores(env._obs))
            done, n, info = False, 0, {}
            while not done and n < env.max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, _r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                n += 1
            successes.append(bool(info.get("postcondition")))
        rows.append({"dz_mm": dz_mm, "success_rate": float(np.mean(successes))})
        print(f"  handoff dz {dz_mm:+3d} mm: success {rows[-1]['success_rate']:.0%}")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--eval-every", type=int, default=5_000)
    ap.add_argument("--curriculum-steps", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/stow/stage3"))
    args = ap.parse_args()

    run_dir = args.out_dir / str(int(time.time()))
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "sac_hole_best"

    train_env = StowHoleEnv(seed_pool=tuple(range(32)))
    eval_env = StowHoleEnv(seed_pool=EVAL_SEEDS, curriculum=1.0)

    print("baselines on the eval seeds")
    expert = rollout(eval_env, EVAL_SEEDS, policy=None)
    print(f"  scripted expert: success {expert['success_rate']:.0%}  "
          f"steps {expert['mean_steps']:.1f}  x_A {expert['mean_x_mm']:.0f}mm")

    model = SAC("MlpPolicy", train_env, verbose=0, seed=args.seed, device=args.device,
                learning_starts=1_000, batch_size=256, train_freq=1, gradient_steps=1)
    evaluator = EvalBest(eval_env, args.eval_every, best_path)
    print(f"training SAC for {args.timesteps} steps ({args.device})")
    started = time.perf_counter()
    model.learn(total_timesteps=args.timesteps,
                callback=[Curriculum(train_env, args.curriculum_steps), evaluator])
    minutes = (time.perf_counter() - started) / 60.0
    model.save(run_dir / "sac_hole_final")

    print("\nfinal evaluation")
    best = SAC.load(best_path, device=args.device)
    best_stats = rollout(eval_env, EVAL_SEEDS, policy=best)
    final_stats = rollout(eval_env, EVAL_SEEDS, policy=model)
    for name, stats in (("best checkpoint", best_stats), ("final", final_stats)):
        print(f"  {name:16s} success {stats['success_rate']:.0%}  fences {stats['fence_rate']:.0%}  "
              f"steps {stats['mean_steps']:.1f}  x_A {stats['mean_x_mm']:.0f}mm")

    print("\nprecondition sweep (vertical offset of the handoff)")
    sweep = precondition_sweep(eval_env, best, EVAL_SEEDS, [-10, -5, 0, 5, 10, 15, 20])

    train_env.close()
    eval_env.close()

    passed = best_stats["success_rate"] >= 0.9 and best_stats["fence_rate"] >= 0.9
    print(f"\ntrained in {minutes:.1f} min")
    print("STAGE 3:", "PASS" if passed else "FAIL")
    (run_dir / "summary.json").write_text(json.dumps({
        "timesteps": args.timesteps,
        "minutes": minutes,
        "expert": expert,
        "best": best_stats,
        "final": final_stats,
        "history": evaluator.history,
        "precondition_sweep": sweep,
        "x_clear_mm": geo.X_CLEAR * 1000,
        "passed": passed,
    }, indent=2))
    print(f"wrote {run_dir}/summary.json")


if __name__ == "__main__":
    main()
