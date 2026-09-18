"""Multi-seed study of the residual RL phase.

The published Table 1 is a single training run, and the archived runs show a
large spread (trained_best_with_release ranged 1.4-21.1 mm across seeds). This
script repeats the full train+evaluate loop over several SAC seeds and reports
mean +/- std across them, so claims about the method carry error bars rather
than resting on one run.

Evaluation seeds are held FIXED across all training seeds, so every policy is
scored on exactly the same 10 scenarios; only the SAC seed (policy init +
training episode stream) varies.

Results are written per-seed as they finish, so an interrupted study keeps its
completed seeds and re-running skips them.

Run (from repo root):
    python scripts/planrob_multiseed_study.py --sac-seeds 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from statistics import mean, pstdev

import numpy as np

_SPEC = importlib.util.spec_from_file_location(
    "full", "scripts/planrob_inverse_rl_pushcube_full_demo_2d_action.py"
)
full = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(full)
demo = full.demo

from stable_baselines3 import SAC  # noqa: E402

# Metrics collected per seed; every one is evaluated on _EVAL_SEEDS.
_TRAINED_METRICS = (
    "trained_final_with_release",
    "trained_final_no_release",
    "trained_best_with_release",
)


def _fmt(m: dict) -> str:
    return (f"{m['distance_mm_mean']:6.1f} +/- {m['distance_mm_std']:5.1f} mm"
            f"   succ@1cm {m['success_rate_at_1cm']:5.0%}")


def run_baselines(out_dir: Path) -> dict:
    """Seed-independent baselines: scripted prefix, and prefix + random action.

    Neither depends on the SAC seed (the random policy uses its own fixed RNG),
    so these are run once for the whole study.
    """
    path = out_dir / "baselines.json"
    if path.exists():
        print("  baselines: cached")
        return json.loads(path.read_text())

    env = full.PushCubeRecoveryFullEnv(
        max_steps=full._MAX_STEPS, atpose_tolerance=full._CURRICULUM_END_TOL,
        action_scale_xyz=full._ACTION_SCALE_XYZ,
        perturbation_range_m=full._PERTURBATION_RANGE_M,
        push_displacement_m=full._PUSH_DISPLACEMENT_M,
    )
    env.set_curriculum_tolerance(full._CURRICULUM_END_TOL)
    res = {}
    res["symbolic_only"] = full.evaluate_with_release(
        env, None, full._EVAL_SEEDS, "symbolic_only", do_rl=False)
    rng = np.random.default_rng(0)
    res["random_rl"] = full.evaluate_with_release(
        env, lambda _o: rng.uniform(-1, 1, size=4).astype(np.float32),
        full._EVAL_SEEDS, "random_rl", do_rl=True)
    env.close()
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k, v in res.items():
        print(f"  {k:<28} {_fmt(v)}")
    return res


def run_seed(sac_seed: int, timesteps: int, out_dir: Path) -> dict:
    path = out_dir / f"seed_{sac_seed}.json"
    if path.exists():
        print(f"  seed {sac_seed}: cached")
        return json.loads(path.read_text())

    # train() reads the module-level _SAC_SEED at call time; this varies both
    # the policy initialisation and the training episode stream.
    full._SAC_SEED = sac_seed
    ckpt = out_dir / f"seed_{sac_seed}_final.zip"
    best = out_dir / f"seed_{sac_seed}_best.zip"

    model, callback, env = full.train(timesteps, ckpt, best)
    model.save(ckpt)
    env.set_curriculum_tolerance(full._CURRICULUM_END_TOL)

    def predict(obs):
        a, _ = model.predict(obs, deterministic=True)
        return a

    res = {"sac_seed": sac_seed, "timesteps": timesteps}
    res["trained_final_with_release"] = full.evaluate_with_release(
        env, predict, full._EVAL_SEEDS, "trained_final_with_release", do_rl=True)
    res["trained_final_no_release"] = full.evaluate_no_release(
        env, predict, full._EVAL_SEEDS, "trained_final_no_release", do_rl=True)

    if best.exists():
        best_model = SAC.load(best, env=env)

        def best_predict(obs):
            a, _ = best_model.predict(obs, deterministic=True)
            return a

        res["trained_best_with_release"] = full.evaluate_with_release(
            env, best_predict, full._EVAL_SEEDS, "trained_best_with_release", do_rl=True)
    env.close()

    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    for k in _TRAINED_METRICS:
        if k in res:
            print(f"  seed {sac_seed}  {k:<28} {_fmt(res[k])}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sac-seeds", type=lambda s: [int(x) for x in s.split(",")],
                    default=[0, 1, 2, 3, 4])
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/multiseed"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Multi-seed study: seeds={args.sac_seeds}  timesteps={args.timesteps:,}")
    print(f"Evaluation seeds (fixed): {full._EVAL_SEEDS}")
    print(f"Output: {args.out_dir}\n")

    print("Baselines (seed-independent)")
    baselines = run_baselines(args.out_dir)

    per_seed = []
    for i, s in enumerate(args.sac_seeds, 1):
        print(f"\n[{i}/{len(args.sac_seeds)}] SAC seed {s}")
        per_seed.append(run_seed(s, args.timesteps, args.out_dir))

    # ---- aggregate ----
    summary = {"sac_seeds": args.sac_seeds, "timesteps": args.timesteps,
               "eval_seeds": list(full._EVAL_SEEDS), "baselines": baselines,
               "per_seed": per_seed, "across_seeds": {}}
    print("\n" + "=" * 78)
    print("ACROSS-SEED SUMMARY  (mean +/- std over SAC seeds; each seed already")
    print("                      averages the same 10 fixed evaluation seeds)")
    print("=" * 78)
    print(f"{'metric':<30}{'distance mm':>22}{'success@1cm':>22}")
    for k, v in baselines.items():
        print(f"{k:<30}{v['distance_mm_mean']:>13.1f} +/- {v['distance_mm_std']:<5.1f}"
              f"{v['success_rate_at_1cm']:>21.0%}   (single, seed-free)")
    for k in _TRAINED_METRICS:
        d = [r[k]["distance_mm_mean"] for r in per_seed if k in r]
        s = [r[k]["success_rate_at_1cm"] for r in per_seed if k in r]
        if not d:
            continue
        agg = {"distance_mm_mean_across_seeds": mean(d),
               "distance_mm_std_across_seeds": pstdev(d) if len(d) > 1 else 0.0,
               "success_mean_across_seeds": mean(s),
               "success_std_across_seeds": pstdev(s) if len(s) > 1 else 0.0,
               "per_seed_distance_mm": d, "per_seed_success": s, "n_seeds": len(d)}
        summary["across_seeds"][k] = agg
        print(f"{k:<30}{agg['distance_mm_mean_across_seeds']:>13.1f} +/- "
              f"{agg['distance_mm_std_across_seeds']:<5.1f}"
              f"{agg['success_mean_across_seeds']:>15.0%} +/- "
              f"{agg['success_std_across_seeds']:.0%}")
    print("\nper-seed detail:")
    for k in _TRAINED_METRICS:
        if k in summary["across_seeds"]:
            a = summary["across_seeds"][k]
            print(f"  {k:<30} dist {['%.1f' % x for x in a['per_seed_distance_mm']]}"
                  f"  succ {['%.0f%%' % (100 * x) for x in a['per_seed_success']]}")

    out = args.out_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
