"""Stage 3: learn the hole's policy with SAC, select a checkpoint by how the learned operator
executes in Stage 4, and test it on held-out scenes.

Trains on `StowHoleEnv` built from the hole spec Stage 2 wrote (what to achieve,
what to preserve, where to start), with its start-state curriculum, unchanged.

Validation (scenes 4000-4019, every --eval-every interactions) and the test (scenes
5000-5049, after training, once the selection is final) run the same way. Frozen on
2026-11-12, before training (artifacts/round2_20261112/PLAN.md, section 1.4):

  handoff    Stage 4's own: the environment's `exact_handoff`, run afresh in every
             episode (a rebuilt simulator, the forward push, the prefix through the
             executive), with no second seating, so every checkpoint sees the same
             handoffs
  episode    the policy from the handoff for the 40-step training horizon
  stop       the first state, from the handoff on, where the target robustly holds:
             where Stage 4's executive stops the policy (one execution, no retries)
  C_exec     primary: at the stop, the target and every fence robustly hold; a target
             never reached in the 40 steps fails
  margin     auxiliary: at step 40, the minimum over the target and the fences of
             V - 0.8 for p and 0.2 - V for not p; P10 is its 10th percentile over
             the scenes (numpy, linear interpolation)
  selection  the highest C_exec, then the highest P10, then the latest checkpoint;
             mean steps and the contract at step 40 are recorded and never used

Stage 3 passes when the selected checkpoint's C_exec is at least 0.9.

Also recorded: the scripted drag (the Stage 0 command pair) on the validation scenes
as a reference, and the handoff sweep, the selected checkpoint from handoffs raised or
lowered by a few millimetres (a robustness probe, not a criterion).

Outputs under <out-dir>/<run_id>/: config.json, summary.json, sac_hole_best.zip (the
selected checkpoint) and sac_hole_final.zip.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from inverse_skills.envs import stow_cube as geo
from inverse_skills.envs import stow_primitives as prim
from inverse_skills.envs.stow_hole_env import StowHoleEnv, load_spec
from inverse_skills.operators.hole_planner import spec_target
from inverse_skills.operators.literals import ROBUST_FALSE, ROBUST_TRUE, Literal

# SAC's small MLPs thrash across this machine's 20 cores: 129 ms/step at the
# default thread count against 10 ms at one thread.
torch.set_num_threads(1)

TRAIN_SEEDS = tuple(range(32))
VALIDATION_SEEDS = tuple(range(4000, 4020))
TEST_SEEDS = tuple(range(5000, 5050))
_DRAG = np.array([-0.2 / 0.3, -0.05 / 0.15], dtype=np.float32)   # the Stage 0 drag commands


def expert_action(env: StowHoleEnv) -> np.ndarray:
    """Reference controller: the Stage 0 drag, held constant.

    The episode starts seated on the cube, so no approach phase is needed —
    this is exactly the command pair the Stage 0 sweep validated.
    """
    return _DRAG


class Curriculum(BaseCallback):
    """Loosen the handoff from a firm seat on the cube to a light touch."""

    def __init__(self, env: StowHoleEnv, schedule_steps: int):
        super().__init__()
        self.env = env
        self.schedule_steps = max(1, schedule_steps)

    def _on_step(self) -> bool:
        self.env.set_curriculum(min(1.0, self.num_timesteps / self.schedule_steps))
        return True


# -- the frozen measures -------------------------------------------------------------------------


def margin(literal: Literal, score: float) -> float:
    """How far a score lies inside the literal's robust region: V - 0.8 for p, 0.2 - V for not p."""
    return score - ROBUST_TRUE if literal.positive else ROBUST_FALSE - score


def stopping_point(trajectory: list[dict], target: Literal) -> int | None:
    """Where Stage 4's executive stops the policy: the index of the first state (0 is the handoff)
    in which the target robustly holds, or None if it never does."""
    return next((t for t, scores in enumerate(trajectory) if target.robustly_holds(scores[target.atom])), None)


def judge(trajectory: list[dict], target: Literal, fences) -> dict:
    """One episode, from the scores at the handoff and after each step."""
    contract = (target, *sorted(fences))
    stop = stopping_point(trajectory, target)
    at_stop, end = trajectory[stop if stop is not None else -1], trajectory[-1]
    return {
        "stop_step": stop,
        "contract_at_stop": stop is not None and all(f.robustly_holds(at_stop[f.atom]) for f in fences),
        "contract_at_40": all(l.robustly_holds(end[l.atom]) for l in contract),
        "margin_at_40": min(margin(l, end[l.atom]) for l in contract),
        "margin_at_stop": min(margin(l, at_stop[l.atom]) for l in contract),
        "scores_at_stop": {l.atom: at_stop[l.atom] for l in contract},
        "scores_at_40": {l.atom: end[l.atom] for l in contract},
    }


def summarize(episodes: list[dict], horizon: int) -> dict:
    return {
        "contract_rate_at_stop": float(np.mean([e["contract_at_stop"] for e in episodes])),    # C_exec
        "p10_margin_at_40": float(np.percentile([e["margin_at_40"] for e in episodes], 10)),   # P10
        "contract_rate_at_40": float(np.mean([e["contract_at_40"] for e in episodes])),
        "reached_rate": float(np.mean([e["stop_step"] is not None for e in episodes])),
        "mean_steps": float(np.mean([horizon if e["stop_step"] is None else e["stop_step"] for e in episodes])),
        "mean_x_mm": float(np.mean([e["object_x_mm"] for e in episodes])),
        "handoffs_valid": int(sum(e["handoff_valid"] for e in episodes)),
        "episodes": episodes,
    }


def select(history: list[dict]) -> dict:
    """The selected evaluation: the highest C_exec, then the highest P10, then the latest."""
    return max(history, key=lambda h: (h["contract_rate_at_stop"], h["p10_margin_at_40"], h["timesteps"]))


# -- rollouts -------------------------------------------------------------------------------------


def rollout(env: StowHoleEnv, seeds, policy=None) -> dict:
    """One episode per scene, from the exact handoff; `policy` None means the scripted drag."""
    episodes = []
    for seed in seeds:
        obs, start = env.reset(seed=seed)
        trajectory, done, info = [start["scores"]], False, {}
        while not done:
            action = expert_action(env) if policy is None else policy.predict(obs, deterministic=True)[0]
            obs, _r, terminated, truncated, info = env.step(action)
            trajectory.append(info["scores"])
            done = terminated or truncated
        episodes.append({"seed": seed, "scene": seed + (start["prefix_attempts"] - 1) * 10_000,
                         "handoff_valid": bool(start["handoff_valid"]), **judge(trajectory, env.target, env.fences),
                         "object_x_mm": info["object_x"] * 1000.0})
    return summarize(episodes, env.max_steps)


class EvalBest(BaseCallback):
    """Every `every` steps, validate, and keep the checkpoint the frozen rule selects."""

    def __init__(self, env: StowHoleEnv, every: int, path: Path):
        super().__init__()
        self.env = env
        self.every = every
        self.path = path
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.every:
            return True
        stats = rollout(self.env, VALIDATION_SEEDS, policy=self.model)
        stats["timesteps"] = self.num_timesteps
        self.history.append(stats)
        if select(self.history) is stats:
            self.model.save(self.path)
            stats["saved"] = True
        print(f"  {self.num_timesteps:>7} steps: C_exec {stats['contract_rate_at_stop']:.0%}  "
              f"P10 {stats['p10_margin_at_40']:+.3f}  (contract at 40 {stats['contract_rate_at_40']:.0%}, "
              f"steps {stats['mean_steps']:.1f}, x_A {stats['mean_x_mm']:.0f}mm)"
              f"{'  <- selected so far' if stats.get('saved') else ''}", flush=True)
        return True


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
            obs = env._observation(env._obs)
            done, info = False, {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, _r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            successes.append(bool(info.get("postcondition")))
        rows.append({"dz_mm": dz_mm, "success_rate": float(np.mean(successes))})
        print(f"  handoff dz {dz_mm:+3d} mm: success {rows[-1]['success_rate']:.0%}")
    return rows


def same_measures(a: dict, b: dict) -> bool:
    """Do two rollouts of one policy on the same scenes agree episode by episode?"""
    keys = ("seed", "scene", "stop_step", "contract_at_stop", "contract_at_40", "margin_at_40")
    return [[e[k] for k in keys] for e in a["episodes"]] == [[e[k] for k in keys] for e in b["episodes"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--eval-every", type=int, default=5_000)
    ap.add_argument("--curriculum-steps", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--spec", type=Path, default=Path("artifacts/stow/hole_spec.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/stow/stage3"))
    args = ap.parse_args()
    spec = load_spec(args.spec)
    target = spec_target(spec)
    print(f"hole: target {target}, start {spec['start']}, preserve "
          f"{spec['preserve']}, keep false {spec['preserve_absent']}")

    run_dir = args.out_dir / str(int(time.time()))
    run_dir.mkdir(parents=True)              # fails, rather than share a folder with another run
    best_path = run_dir / "sac_hole_best"
    (run_dir / "config.json").write_text(json.dumps({
        "command": " ".join(sys.argv), "timesteps": args.timesteps, "eval_every": args.eval_every,
        "curriculum_steps": args.curriculum_steps, "seed": args.seed, "device": args.device,
        "spec": str(args.spec), "spec_sha256": sha256(args.spec), "target": target.to_dict(),
        "sac": {"learning_starts": 1_000, "batch_size": 256, "train_freq": 1, "gradient_steps": 1},
        "seeds": {"train": list(TRAIN_SEEDS), "validation": list(VALIDATION_SEEDS), "test": list(TEST_SEEDS)},
        "handoff": "validation and test: exact_handoff (Stage 4's, afresh in every episode, no second seating); "
                   "training: the cached handoff and the curriculum's seating, unchanged",
        "selection": "the highest contract_rate_at_stop (C_exec), then the highest p10_margin_at_40, then the "
                     "latest checkpoint; frozen 2026-11-12",
        "pass": "the selected checkpoint's contract_rate_at_stop >= 0.9",
    }, indent=2))

    train_env = StowHoleEnv(spec, seed_pool=TRAIN_SEEDS)
    exact_env = StowHoleEnv(spec, seed_pool=VALIDATION_SEEDS, exact_handoff=True)

    print("reference on the validation scenes: the scripted drag")
    expert = rollout(exact_env, VALIDATION_SEEDS, policy=None)
    print(f"  C_exec {expert['contract_rate_at_stop']:.0%}, reached {expert['reached_rate']:.0%}, "
          f"steps {expert['mean_steps']:.1f}, x_A {expert['mean_x_mm']:.0f}mm")

    model = SAC("MlpPolicy", train_env, verbose=0, seed=args.seed, device=args.device,
                learning_starts=1_000, batch_size=256, train_freq=1, gradient_steps=1)
    evaluator = EvalBest(exact_env, args.eval_every, best_path)
    print(f"training SAC for {args.timesteps} steps ({args.device})")
    started = time.perf_counter()
    model.learn(total_timesteps=args.timesteps,
                callback=[Curriculum(train_env, args.curriculum_steps), evaluator])
    minutes = (time.perf_counter() - started) / 60.0
    model.save(run_dir / "sac_hole_final")

    selected = select(evaluator.history)
    best = SAC.load(best_path, device=args.device)
    print(f"\nselected: step {selected['timesteps']}, C_exec {selected['contract_rate_at_stop']:.0%}, "
          f"P10 {selected['p10_margin_at_40']:+.3f}")
    best_again = rollout(exact_env, VALIDATION_SEEDS, policy=best)
    reproduces = same_measures(selected, best_again)
    print(f"  the saved checkpoint, run again on the validation scenes, reproduces its record: {reproduces}")
    final_stats = rollout(exact_env, VALIDATION_SEEDS, policy=model)

    # The selection is final; the test scenes are used from here on only.
    test = rollout(exact_env, TEST_SEEDS, policy=best)
    print(f"\ntest scenes {TEST_SEEDS[0]}-{TEST_SEEDS[-1]}: C_exec {test['contract_rate_at_stop']:.0%}, "
          f"P10 {test['p10_margin_at_40']:+.3f}, contract at 40 {test['contract_rate_at_40']:.0%}, "
          f"handoffs valid {test['handoffs_valid']}/{len(TEST_SEEDS)}")

    print("\nprecondition sweep (vertical offset of the handoff, validation scenes)")
    sweep = precondition_sweep(exact_env, best, VALIDATION_SEEDS, [-10, -5, 0, 5, 10, 15, 20])

    train_env.close()
    exact_env.close()

    passed = selected["contract_rate_at_stop"] >= 0.9
    print(f"\ntrained in {minutes:.1f} min")
    print("STAGE 3:", "PASS" if passed else "FAIL")
    (run_dir / "summary.json").write_text(json.dumps({
        "timesteps": args.timesteps,
        "minutes": minutes,
        "spec": spec,
        "target": target.to_dict(),
        "checkpoints": {name: {"path": str(run_dir / f"{name}.zip"), "sha256": sha256(run_dir / f"{name}.zip")}
                        for name in ("sac_hole_best", "sac_hole_final")},
        "expert": expert,
        "selected": {key: selected[key] for key in ("timesteps", "contract_rate_at_stop", "p10_margin_at_40",
                                                    "contract_rate_at_40", "mean_steps")},
        "best": best_again,
        "best_reproduces_its_record": reproduces,
        "final": final_stats,
        "test": test,
        "history": evaluator.history,
        "precondition_sweep": sweep,
        "x_clear_mm": geo.X_CLEAR * 1000,
        "passed": passed,
    }, indent=2))
    print(f"wrote {run_dir}/summary.json")


if __name__ == "__main__":
    main()
