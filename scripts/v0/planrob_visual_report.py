"""
Generate paper figures for PlanRob 2026 submission.

Produces:
  artifacts/planrob_figure_scenario.png  — cross-class push scenario (3 panels)
  artifacts/planrob_figure_predicates.png — per-predicate comparison (3 skills)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from inverse_skills.operators import OperatorExtractor, RestorationObjective
from inverse_skills.operators.toy_planner import ToyInversePlanner
from inverse_skills.toy.domains import (
    build_predicate_registry,
    build_predicate_registry_grasp_hold,
    named_regions,
)
from inverse_skills.toy.generators import (
    make_grasp_hold_rollouts_executable,
    make_pick_place_rollouts_executable,
    make_push_rollouts_executable,
)
from inverse_skills.toy.primitives import PrimitiveLibrary
from inverse_skills.toy.simulator import ToyTabletopSimulator

# ── colour palette ─────────────────────────────────────────────────────────────
_SRC_EDGE = (0.28, 0.47, 0.81)
_TGT_EDGE = (0.94, 0.53, 0.21)
_SRC_FACE = (*_SRC_EDGE, 0.18)
_TGT_FACE = (*_TGT_EDGE, 0.18)
_CUBE_NEUTRAL = "#888888"
_CUBE_FWD     = "#C94040"   # forward final (needs inverse)
_CUBE_INV     = "#4A9E5C"   # inverse final (restored)
_CUBE_R       = 0.040       # display radius


def _draw_scene(ax, cube_xy, cube_color, title, potential, subtitle=None):
    regions = named_regions()
    ax.set_facecolor("#f7f7f7")
    ax.set_xlim(-0.30, 0.80)
    ax.set_ylim(-0.23, 0.23)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    for rname, edge, face in [("source", _SRC_EDGE, _SRC_FACE), ("target", _TGT_EDGE, _TGT_FACE)]:
        r = regions[rname]
        ax.add_patch(mpatches.Rectangle(
            (r.lower[0], r.lower[1]),
            r.upper[0] - r.lower[0], r.upper[1] - r.lower[1],
            linewidth=1.5, edgecolor=edge, facecolor=face, zorder=1,
        ))
        ax.text(r.center[0], r.lower[1] - 0.025, rname,
                ha="center", va="top", fontsize=7, color=edge, fontweight="bold")

    ax.add_patch(mpatches.Circle(cube_xy, _CUBE_R, color=cube_color, zorder=5))

    pot_box = dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#bbbbbb", alpha=0.92)
    ax.text(0.50, 0.88, f"$V\\!=\\!{potential:.3f}$",
            ha="center", va="center", fontsize=8, transform=ax.transAxes, bbox=pot_box)

    ax.set_title(title, fontsize=8, fontweight="bold", pad=4)
    if subtitle:
        ax.text(0.50, -0.04, subtitle, ha="center", va="top", fontsize=7,
                style="italic", color="#555555", transform=ax.transAxes)


def _make_scenario_figure(out_path: Path) -> None:
    """3-panel figure: forward initial → forward final → inverse final (push scenario)."""
    # Build push case
    rollouts = make_push_rollouts_executable()
    registry = build_predicate_registry()
    extraction = OperatorExtractor(registry).extract("push_to_target", rollouts)
    objective = RestorationObjective(extraction.operator, registry)
    planner = ToyInversePlanner(PrimitiveLibrary(), objective)

    fwd_initial = rollouts[0].first()
    fwd_final   = rollouts[0].last()

    result = planner.plan(fwd_final, max_depth=3)
    sim = ToyTabletopSimulator()
    inv_exec = sim.execute("inverse", "inv_000", fwd_final, result.actions)
    inv_final = inv_exec.rollout.last()

    v_initial = objective.potential(fwd_initial)
    v_fwd     = objective.potential(fwd_final)
    v_inv     = objective.potential(inv_final)

    cube_initial = fwd_initial.objects["cube"].pose.position[:2]
    cube_fwd     = fwd_final.objects["cube"].pose.position[:2]
    cube_inv     = inv_final.objects["cube"].pose.position[:2]

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    fig.subplots_adjust(wspace=0.06, left=0.01, right=0.99, top=0.82, bottom=0.12)

    _draw_scene(axes[0], cube_initial, _CUBE_NEUTRAL,
                "(a) Forward initial", v_initial, "cube at src, gripper open")
    _draw_scene(axes[1], cube_fwd, _CUBE_FWD,
                "(b) After push(cube)", v_fwd, "cube at tgt — inverse needed")
    _draw_scene(axes[2], cube_inv, _CUBE_INV,
                "(c) After inverse plan", v_inv,
                f"actions: {' → '.join(result.actions)}")

    # Arrow between (b) and (c) in figure coordinates
    ax2_pos = axes[1].get_position()
    ax3_pos = axes[2].get_position()
    axes[1].annotate(
        "", xy=(1.06, 0.50), xycoords="axes fraction",
        xytext=(1.01, 0.50), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.4,
                        connectionstyle="arc3,rad=0.0"),
    )

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _make_predicate_figure(out_path: Path) -> None:
    """Per-predicate inverse target score comparison for 3 skills at forward final."""
    skill_configs = [
        ("pick\\_place",  make_pick_place_rollouts_executable, build_predicate_registry, "pick_place"),
        ("push",          make_push_rollouts_executable,        build_predicate_registry, "push_to_target"),
        ("grasp\\_hold",  make_grasp_hold_rollouts_executable,  build_predicate_registry_grasp_hold, "grasp_hold"),
    ]

    records = []
    for label, gen_fn, reg_fn, skill_name in skill_configs:
        rollouts  = gen_fn()
        registry  = reg_fn()
        operator  = OperatorExtractor(registry).extract(skill_name, rollouts).operator
        objective = RestorationObjective(operator, registry)
        fwd_final = rollouts[0].last()
        per_term  = {term.key: objective.term_score(term, fwd_final) for term in objective.terms}
        total_pot = objective.potential(fwd_final)
        plan_len  = ToyInversePlanner(PrimitiveLibrary(), objective).plan(fwd_final, max_depth=3)
        records.append((label, per_term, total_pot, plan_len.actions))

    # Collect unique terms in a consistent display order
    term_order = [
        "gripper_open()",
        "in_region(cube,source)",
        "in_region(cube,target)",
        "holding(cube)",
    ]
    all_terms = {t for _, d, _, _ in records for t in d}
    term_order = [t for t in term_order if t in all_terms]

    n_skills = len(records)
    n_terms  = len(term_order)
    matrix   = np.full((n_skills, n_terms), np.nan)
    for i, (_, per_term, _, _) in enumerate(records):
        for j, t in enumerate(term_order):
            if t in per_term:
                matrix[i, j] = per_term[t]

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    fig.subplots_adjust(left=0.22, right=0.98, top=0.88, bottom=0.28)

    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")

    # Annotate cells
    for i in range(n_skills):
        for j in range(n_terms):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9, color="#888888")
            else:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 0.3 or v > 0.7 else "black")

    ax.set_xticks(range(n_terms))
    ax.set_xticklabels(term_order, fontsize=7, rotation=22, ha="right")
    ax.set_yticks(range(n_skills))
    skill_display = ["pick-place", "push", "grasp-hold"]
    ylabels = []
    for (lbl, _, pot, actions), disp in zip(records, skill_display):
        ylabels.append(f"{disp}  V={pot:.3f}  [{' → '.join(actions)}]")
    ax.set_yticklabels(ylabels, fontsize=7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("term contribution", fontsize=7)
    ax.set_title("Inverse target term scores at forward final state", fontsize=8, fontweight="bold")

    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    _make_scenario_figure(out_dir / "planrob_figure_scenario.png")
    _make_predicate_figure(out_dir / "planrob_figure_predicates.png")


if __name__ == "__main__":
    main()
