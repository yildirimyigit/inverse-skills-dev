"""Diagnostic plots for the residual-objective reward shape.

Three PNGs:
  predicate_score_vs_margin.png — soft-predicate score vs signed margin
                                   (multiple temperatures; current highlighted)
  reward_vs_distance.png        — current shaped reward vs cube-target distance,
                                   decomposed into (V_residual, distance penalty)
  reward_alternatives.png       — comparison of alternative reward formulations
                                   over the same distance range, with their
                                   gradient magnitude shown below each.

The scripts use the same numbers as the live training:
  AtPosePredicate margin = threshold - distance
  threshold = 0.01 m  (1 cm)
  temperatures: 0.05 (train), 0.005 (eval)
  shaped reward = V_residual - 2.0 * distance
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_THRESHOLD = 0.01      # at_pose distance threshold (m), our 1 cm target
_TEMP_TRAIN = 0.05     # the softness used during RL training
_TEMP_EVAL = 0.005     # the softness used for honest evaluation
_DIST_PENALTY = 2.0    # coefficient on the linear distance penalty in the shaped reward
_OUT_DIR = Path("artifacts")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ── Plot 1: predicate score vs margin ─────────────────────────────────────────


def plot_predicate_score_vs_margin() -> None:
    margins = np.linspace(-0.05, 0.05, 1500)  # ±5 cm range
    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    palette = {0.005: "#2a8c4d", 0.02: "#4a90c8",
               _TEMP_TRAIN: "#c84040", 0.10: "#a060c8"}
    for temp, color in palette.items():
        scores = sigmoid(margins / temp)
        label = f"T={temp:.3f}"
        if temp == _TEMP_TRAIN:
            label += "  ← training"
        elif temp == _TEMP_EVAL:
            label += "  ← eval"
        ax.plot(margins * 1000, scores, lw=2.2 if temp == _TEMP_TRAIN else 1.4,
                color=color, label=label,
                alpha=1.0 if temp == _TEMP_TRAIN else 0.85)

    ax.axvline(0.0, color="#666666", ls="--", lw=0.8)
    ax.text(1.5, 0.05, "margin=0\n(predicate boundary)", color="#666666",
            fontsize=8, va="bottom", ha="left")
    ax.set_xlabel("signed margin (mm) — positive = inside predicate")
    ax.set_ylabel("predicate score = sigmoid(margin / T)")
    ax.set_title("Soft predicate score vs margin\n"
                 "Sigmoid saturates rapidly outside the transition band; "
                 "useful gradient only in a window of ±a few T",
                 fontsize=10)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out = _OUT_DIR / "predicate_score_vs_margin.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Plot 2: reward vs distance (current shaped reward) ───────────────────────


def plot_reward_vs_distance() -> None:
    distances = np.linspace(0.0, 0.20, 1500)  # 0 to 20 cm
    margin = _THRESHOLD - distances
    v_train = sigmoid(margin / _TEMP_TRAIN)
    v_eval = sigmoid(margin / _TEMP_EVAL)
    pen = -_DIST_PENALTY * distances
    reward_train = v_train + pen
    reward_eval = v_eval + pen  # for reference; not used as RL reward

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.6),
                              gridspec_kw={"height_ratios": [3, 2]})

    ax = axes[0]
    ax.plot(distances * 1000, v_train, color="#c84040", lw=1.6,
            label=f"V_residual (T={_TEMP_TRAIN}, training)")
    ax.plot(distances * 1000, v_eval, color="#2a8c4d", lw=1.4, ls="--",
            label=f"V_residual (T={_TEMP_EVAL}, eval — sharper)")
    ax.plot(distances * 1000, pen, color="#888888", lw=1.4,
            label=f"−{_DIST_PENALTY:.1f}·distance penalty")
    ax.plot(distances * 1000, reward_train, color="#222266", lw=2.2,
            label=f"shaped reward = V_residual + penalty (training)")
    ax.axvline(_THRESHOLD * 1000, color="#888888", ls=":", lw=0.8)
    ax.text(_THRESHOLD * 1000 + 0.6, 0.85, "threshold\n(1cm)",
            color="#666666", fontsize=8, va="top")
    ax.axhline(0.0, color="#cccccc", lw=0.6)
    ax.set_xlabel("cube–target distance (mm)")
    ax.set_ylabel("reward")
    ax.set_title("Current shaped reward vs distance\n"
                 "V_residual is informative only within ~30 mm; "
                 "beyond that the linear penalty dominates",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # Gradient magnitude (informativeness): |d reward / d distance|
    grad_v_train = np.gradient(v_train, distances)
    grad_v_eval = np.gradient(v_eval, distances)
    grad_total = np.gradient(reward_train, distances)

    ax2 = axes[1]
    ax2.plot(distances * 1000, np.abs(grad_v_train), color="#c84040", lw=1.6,
             label="|dV_train/dd|")
    ax2.plot(distances * 1000, np.abs(grad_v_eval), color="#2a8c4d", lw=1.4, ls="--",
             label="|dV_eval/dd|")
    ax2.plot(distances * 1000, np.abs(grad_total), color="#222266", lw=2.0,
             label="|d shaped_reward / dd|")
    ax2.axhline(_DIST_PENALTY, color="#888888", ls=":", lw=0.8)
    ax2.text(180, _DIST_PENALTY + 0.5, "constant gradient\nfrom penalty (=2)",
             color="#666666", fontsize=8, ha="right")
    ax2.set_xlabel("cube–target distance (mm)")
    ax2.set_ylabel("|reward gradient|  (per m)")
    ax2.set_title("Reward gradient magnitude — where the agent gets *informative* signal",
                  fontsize=10)
    ax2.set_yscale("log")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.3, which="both")

    fig.tight_layout()
    out = _OUT_DIR / "reward_vs_distance.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Plot 3: alternative reward formulations ──────────────────────────────────


def plot_reward_alternatives() -> None:
    """Side-by-side comparison of generic, less-engineered reward shapes.

    The point: the 'soft-predicate-as-reward' framing has a structural sparsity
    problem — sigmoid saturates outside the threshold band. Several alternatives
    avoid this without per-task engineering."""
    distances = np.linspace(0.001, 0.20, 1500)

    formulations = {
        "current\n(sigmoid + −2d)": (
            sigmoid((_THRESHOLD - distances) / _TEMP_TRAIN) - _DIST_PENALTY * distances,
            "#c84040",
        ),
        "−distance\n(margin, no sigmoid)": (
            -distances,
            "#1f3a73",
        ),
        "exp(−d/scale)\n(scale=2cm)": (
            np.exp(-distances / 0.02),
            "#2a8c4d",
        ),
        "1/(1+d/scale)\n(scale=2cm)": (
            1.0 / (1.0 + distances / 0.02),
            "#7a4ad0",
        ),
        "−log(d+ε)\n(ε=1mm)": (
            -np.log(distances + 0.001),
            "#d08a30",
        ),
    }

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0),
                              gridspec_kw={"height_ratios": [3, 2]})

    # Top: reward curves (each normalized to [0, 1] over the plotted range for
    # visual comparison — the *shape* matters here, not the absolute scale)
    ax = axes[0]
    for label, (rewards, color) in formulations.items():
        r = rewards.copy()
        r_norm = (r - r.min()) / (r.max() - r.min() + 1e-12)
        ax.plot(distances * 1000, r_norm, color=color, lw=1.6, label=label)
    ax.axvline(_THRESHOLD * 1000, color="#888888", ls=":", lw=0.8)
    ax.set_xlabel("cube–target distance (mm)")
    ax.set_ylabel("reward (normalized to [0, 1] for shape comparison)")
    ax.set_title("Alternative reward formulations — shape comparison\n"
                 "Each normalized so that visual differences reflect curvature, not scale",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=8.5, ncol=1)
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.03, 1.03)

    # Bottom: gradient magnitude — where each formulation provides informative signal
    ax2 = axes[1]
    for label, (rewards, color) in formulations.items():
        grad = np.abs(np.gradient(rewards, distances))
        ax2.plot(distances * 1000, grad, color=color, lw=1.6, label=label.replace("\n", " "))
    ax2.set_xlabel("cube–target distance (mm)")
    ax2.set_ylabel("|reward gradient|  (per m)")
    ax2.set_title("Gradient magnitude — informativeness of the reward signal vs distance",
                  fontsize=10)
    ax2.set_yscale("log")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = _OUT_DIR / "reward_alternatives.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_predicate_score_vs_margin()
    plot_reward_vs_distance()
    plot_reward_alternatives()


if __name__ == "__main__":
    main()
