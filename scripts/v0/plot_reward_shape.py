"""Generate the reward-shape figure for the paper's Method section.

Two panels showing the per-predicate reward contribution as a function of
the normalized signed margin

    x = σ_p · m_p(s) / c_p

  Left  (active residual, bipolar) : tanh(x)              in [-1, +1]
  Right (already-restored, fence)  : min(0, tanh(x))      in [-1,  0]

The figure illustrates two design choices in the predicate-derived reward:
  • tanh saturation keeps each term bounded — value function stays sane.
  • min(0, ·) for fences makes them silent when satisfied, penalizing
    only violations — no perverse incentive to over-saturate a fence at
    the expense of the active residual.

Outputs:
    artifacts/figures/reward_shape.pdf   (vector, for the paper)
    artifacts/figures/reward_shape.png   (raster, for previewing)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> None:
    out_dir = Path("artifacts/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-4.0, 4.0, 1000)
    bipolar = np.tanh(x)
    fence = np.minimum(0.0, np.tanh(x))

    # Paper-friendly styling.
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "mathtext.fontset": "cm",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)

    # ── Left panel: bipolar (active residual) ─────────────────────────
    ax = axes[0]
    ax.plot(x, bipolar, color="#1f3a93", linewidth=2.0)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
    ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.4, linestyle="--")

    # Region shading: predicate violated (left) vs. holds (right).
    ax.axvspan(-4.0, 0.0, color="#d62728", alpha=0.05)
    ax.axvspan(0.0, 4.0, color="#2ca02c", alpha=0.05)
    ax.text(-2.0, 1.05, "violated", fontsize=9, color="#822727", ha="center")
    ax.text(2.0, 1.05, "satisfied", fontsize=9, color="#1f6e1f", ha="center")

    # Annotations on the curve.
    ax.annotate("linear regime\n(slope $1/c_p$)", xy=(0.0, 0.0),
                xytext=(-2.6, 0.55), fontsize=9, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.7))
    ax.annotate("saturation $\\to +1$", xy=(3.0, 0.995), xytext=(0.3, 0.65),
                fontsize=9, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.7))
    ax.annotate("saturation $\\to -1$", xy=(-3.0, -0.995), xytext=(-0.2, -0.7),
                fontsize=9, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.7))

    ax.set_title(r"Active residual: $\tanh(x)$  $\in [-1, +1]$")
    ax.set_xlabel(r"normalized margin $\;x = \sigma_p\, m_p(s) / c_p$")
    ax.set_ylabel("reward contribution")
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-1.18, 1.18)
    ax.grid(alpha=0.18, linewidth=0.5)

    # ── Right panel: fence (one-sided) ────────────────────────────────
    ax = axes[1]
    ax.plot(x, fence, color="#a2231d", linewidth=2.0)
    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
    ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.4, linestyle="--")

    # Silent zone shading.
    ax.axvspan(0.0, 4.0, color="#2ca02c", alpha=0.10)
    ax.axvspan(-4.0, 0.0, color="#d62728", alpha=0.05)
    ax.text(-2.0, 1.05, "violated", fontsize=9, color="#822727", ha="center")
    ax.text(2.0, 1.05, "satisfied", fontsize=9, color="#1f6e1f", ha="center")
    ax.text(2.0, -0.55, "silent zone\n(no gradient)",
            fontsize=9.5, color="#1f6e1f", ha="center", va="center")

    # Annotation on the violated half.
    ax.annotate("penalize violations only", xy=(-2.5, -0.987),
                xytext=(-3.7, -0.30), fontsize=9, color="dimgray",
                arrowprops=dict(arrowstyle="->", color="dimgray", lw=0.7))

    # Mark the kink at the origin.
    ax.plot([0], [0], marker="o", markersize=3.5, color="black",
            zorder=5)

    ax.set_title(r"Fence: $\min(0,\,\tanh(x))$  $\in [-1, 0]$")
    ax.set_xlabel(r"normalized margin $\;x = \sigma_p\, m_p(s) / c_p$")
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-1.18, 1.18)
    ax.grid(alpha=0.18, linewidth=0.5)

    fig.tight_layout()

    pdf_path = out_dir / "reward_shape.pdf"
    png_path = out_dir / "reward_shape.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
