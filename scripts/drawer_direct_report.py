"""Direct baseline, step 4: compare the direct runs and draw their learning curves.

Reads every artifacts/drawer/direct/<run>/summary.json and prints one row per
run. The figure has three panels over the same training steps:
  validation success      the goal at the end of the episode, seeds 500-509; the
                          targeted runs' grasp validation (Stage 3, same seeds and
                          budget, 40-step episodes, before the push fix) in grey
  drawer opening          the furthest any training episode in the interval
                          opened the drawer; drawer_open holds from 94 mm
  grasping                the share of training episodes in the interval in
                          which handle_grasped held at some step

Writes artifacts/drawer/direct/comparison.json and learning_curves.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4, one per SAC seed
REFERENCE = "#8c8b86"                                    # muted ink for the targeted runs
OPEN_MM = 94.0                                           # drawer_open scores 0.8 here


def row(summary: dict, run: str) -> dict:
    d, best, final = summary["discovery"], summary["best_checkpoint"], summary["final"]
    last = summary["history"][-1]
    return {
        "run": run,
        "sac_seed": int(run.split("_")[0].removeprefix("seed")),
        "best_validation": best["validation_goal_rate"], "best_at_step": best["timesteps"],
        "final_validation": final["validation_goal_rate"],
        "final_validation_after_settling": last["validation_settled_goal_rate"],
        "final_validation_failures": final["validation_failures"],
        "final_validation_max_opening_mm": last["validation_max_opening_mm"],
        "first_grasp_episode_ends_at_step": d["first_grasp_episode_ends_at_step"],
        "first_opening_at_step": d["first_opening_at_step"],
        "first_held_episode_ends_at_step": d["first_held_episode_ends_at_step"],
        "training_episodes": summary["training_episodes"],
        "training_episodes_grasped": d["training_episodes_grasped"],
        "training_episodes_opened": d["training_episodes_opened"],
        "training_max_opening_mm": d["training_max_opening_mm"],
        "training_best_drawer_open_score": d["training_best_drawer_open_score"],
        "training_failures": d["training_failures"],
        "training_interactions": summary["training_interactions"],
        "validation_interactions": summary["validation_interactions"],
        "minutes": round(summary["minutes"], 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("artifacts/drawer/direct"))
    ap.add_argument("--stage3", type=Path, default=Path("artifacts/drawer/stage3"))
    args = ap.parse_args()

    rows, histories = [], {}
    for path in args.dir.glob("*/summary.json"):
        summary = json.loads(path.read_text())
        rows.append(row(summary, path.parent.name))
        histories[rows[-1]["sac_seed"]] = summary["history"]
    rows.sort(key=lambda r: r["sac_seed"])
    histories = dict(sorted(histories.items()))

    print(f"{'seed':>4} {'best val':>9} {'at step':>8} {'final val':>9} {'settled':>8} {'1st grasp':>9} "
          f"{'1st open':>8} {'grasped':>9} {'opened':>7} {'max mm':>7} {'best V':>7}  final failures (validation)")
    for r in rows:
        print(f"{r['sac_seed']:>4} {r['best_validation']:>9.0%} {r['best_at_step']:>8} {r['final_validation']:>9.0%} "
              f"{r['final_validation_after_settling']:>8.0%} {r['first_grasp_episode_ends_at_step']!s:>9} "
              f"{r['first_opening_at_step']!s:>8} {r['training_episodes_grasped']:>4}/{r['training_episodes']:<4} "
              f"{r['training_episodes_opened']:>7} {r['training_max_opening_mm']:>7.1f} "
              f"{r['training_best_drawer_open_score']:>7.4f}  {r['final_validation_failures']}")

    targeted = []
    for path in sorted(args.stage3.glob("*/summary.json")):
        history = json.loads(path.read_text())["history"]
        targeted.append(([h["timesteps"] for h in history], [h["validation_success_rate"] for h in history]))

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.6), sharex=True, constrained_layout=True)
    for i, (ts, val) in enumerate(targeted):
        axes[0].plot([t / 1000 for t in ts], [v * 100 for v in val], color=REFERENCE, lw=1.2, ls="--",
                     label="targeted grasp (Stage 3, 4 runs)" if i == 0 else None)
    for seed, history in histories.items():
        steps = [h["timesteps"] / 1000 for h in history]
        style = {"color": SERIES[seed], "lw": 2, "marker": "o", "ms": 4, "label": f"direct, SAC seed {seed}"}
        axes[0].plot(steps, [h["validation_goal_rate"] * 100 for h in history], **style)
        axes[1].plot(steps, [h["train_max_opening_mm"] for h in history], **style)
        axes[2].plot(steps, [h["train_grasp_rate"] * 100 for h in history], **style)
    axes[1].axhline(OPEN_MM, color=REFERENCE, lw=1, ls=":")
    axes[1].text(26.0, OPEN_MM - 12, "drawer_open holds (94 mm)", color="#52514e", fontsize=8)
    axes[0].set_ylabel("validation success (%)")
    axes[1].set_ylabel("furthest opening in\ntraining episodes (mm)")
    axes[2].set_ylabel("training episodes\nwith a grasp (%)")
    axes[2].set_xlabel("environment interactions (thousands)")
    axes[0].set_ylim(-4, 104)
    axes[1].set_ylim(0, max(160.0, max(r["training_max_opening_mm"] for r in rows) + 10))
    axes[2].set_ylim(-4, 104)
    for ax in axes:
        ax.grid(color="#e6e5e0", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b9b8b1")
        ax.tick_params(colors="#52514e", labelsize=8)
        ax.yaxis.label.set_color("#0b0b0b")
    axes[0].legend(fontsize=8, frameon=False, loc="center right")
    axes[0].set_title("Direct learning of HOLE[drawer_open], unshaped reward 2 V(drawer_open) - 1",
                      fontsize=10, loc="left", color="#0b0b0b")
    fig.savefig(args.dir / "learning_curves.png", dpi=160, facecolor="#fcfcfb")
    (args.dir / "comparison.json").write_text(json.dumps({"runs": rows}, indent=2))
    print(f"wrote {args.dir}/comparison.json and learning_curves.png")


if __name__ == "__main__":
    main()
