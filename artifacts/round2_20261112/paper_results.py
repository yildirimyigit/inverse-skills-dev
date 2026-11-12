"""Round 2's figures and tables for the manuscript, built only from saved results (PLAN.md section 7).

Inputs (read, never changed), under artifacts/round2_20261112/:
  R1_training/drawer/<run>/{config,summary}.json   targeted runs: skill-level and complete-goal validation
  R1_training/direct/<run>/{config,summary}.json   direct runs: complete-goal validation after settling
  R1_training/stow/<run>/summary.json              the Stow run: validation history, selection, skill test
  R2_stage4/stow_stage4.json, drawer_stage4.json   Stage 4 (the Stow file exists only if Stage 3 passed)
  R3_final/final_targeted_vs_direct.json            the final comparison on 7000-7049
and, for the supplementary table of previous results, under artifacts/signed_literals_20261111/:
  E4_stage4/{final_targeted_vs_direct,stow_stage4}.json and D_diagnostics/*.json

Outputs in --out (refuses to overwrite):
  drawer_learning_curves.pdf/.png, drawer_learning_curves_data.csv   (a) complete-goal validation, both
      methods; (b) the targeted runs' skill-level grasp validation; every evaluation marked, the selected
      checkpoints ringed, nothing smoothed
  stow_selection.pdf/.png, stow_selection_data.csv   C_exec and C_40, and P10 below, per checkpoint
  drawer_table.tex/.md, drawer_table_data.csv        the final comparison and learning efficiency, per policy
  stow_table.tex/.md, stow_table_data.csv            Stage 3 (validation and test) and Stage 4
  previous_results.md                                the previous protocol's results, labelled
  sources.json                                       SHA-256 of every input
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path("artifacts/round2_20261112")
R1, R2, R3 = HERE / "R1_training", HERE / "R2_stage4", HERE / "R3_final"
SL = Path("artifacts/signed_literals_20261111")
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2, as in the published figure
INK, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#e6e5e0", "#9a9992"
THRESHOLD = 0.9
COMPLETE = {"targeted": "validation_task_success_rate", "direct": "validation_settled_goal_rate"}
SKILL = "validation_success_rate"
SOURCES: list[Path] = []


def read(path: Path) -> dict:
    SOURCES.append(path)
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The 95% Wilson score interval for k successes in n trials."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def interval(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"[{100 * lo:.0f}, {100 * hi:.0f}]%"


def kilo(steps) -> str:
    return "never" if steps is None else f"{steps / 1000:g}k"


def first_90(curve) -> int | None:
    return next((t for t, rate in curve if rate >= THRESHOLD), None)


def stayed_90(curve) -> bool:
    first = next((i for i, (_, rate) in enumerate(curve) if rate >= THRESHOLD), None)
    return first is not None and all(rate >= THRESHOLD for _, rate in curve[first:])


# -- the Drawer --------------------------------------------------------------------------------------


def drawer_runs(method: str) -> list[dict]:
    folder = R1 / ("drawer" if method == "targeted" else "direct")
    found = []
    for summary_path in sorted(folder.glob("*/summary.json")):
        config, summary = read(summary_path.parent / "config.json"), read(summary_path)
        history = summary["history"]
        found.append({"method": method, "seed": config["seed"], "run": summary_path.parent.name,
                      "complete": [(h["timesteps"], h[COMPLETE[method]]) for h in history],
                      "skill": [(h["timesteps"], h[SKILL]) for h in history] if method == "targeted" else None,
                      "selected_at": [h["timesteps"] for h in history if h.get("saved")][-1]})
    return sorted(found, key=lambda r: r["seed"])


def style(ax, ylabel: str, xmax: float) -> None:
    ax.set_xlim(0, xmax)
    ax.set_ylabel(ylabel, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(labelsize=7, colors=MUTED, length=2.5, width=0.6)


def curve_plot(ax, run: dict, curve, colour: str, line: str, marker: str, label: str | None) -> None:
    x = [t / 1000 for t, _ in curve]
    y = [100 * rate for _, rate in curve]
    ax.plot(x, y, color=colour, ls=line, lw=1.1, marker=marker, ms=3.4, mec="white", mew=0.5, alpha=0.8,
            label=label, clip_on=False, zorder=3)
    t_sel, rate_sel = next((t, r) for t, r in curve if t == run["selected_at"])
    ax.plot([t_sel / 1000], [100 * rate_sel], ls="none", marker="o", ms=6.5, mfc="none", mec=colour, mew=0.9,
            clip_on=False, zorder=4)


def drawer_figure(runs: list[dict], out: Path) -> None:
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.5, 4.1), sharex=True)
    for method, colour, line, marker, label in (("targeted", BLUE, "-", "o", "Targeted (learned grasp + pull_open)"),
                                                ("direct", ORANGE, "--", "s", "Direct (one policy)")):
        for i, run in enumerate(r for r in runs if r["method"] == method):
            curve_plot(top, run, run["complete"], colour, line, marker, label if i == 0 else None)
            if method == "targeted":
                curve_plot(bottom, run, run["skill"], colour, line, marker, None)
    for ax, ylabel in ((top, "Complete-goal validation (%)"), (bottom, "Grasp validation (%)")):
        ax.axhline(100 * THRESHOLD, color=MUTED, lw=0.6, ls=":", zorder=1)
        ax.annotate("90%", (30, 100 * THRESHOLD), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=6.5, color=MUTED, annotation_clip=False)
        ax.set_ylim(-5, 105)
        ax.set_yticks(range(0, 101, 20))
        style(ax, ylabel, 30)
    top.set_title("(a) the complete signed goal, after settling", fontsize=7.5, color=INK, loc="left")
    bottom.set_title("(b) targeted only: the learned grasp (skill level)", fontsize=7.5, color=INK, loc="left")
    bottom.set_xticks(range(0, 31, 5))
    bottom.set_xlabel("Training interactions (thousands)", color=INK)
    top.legend(loc="lower center", ncol=1, frameon=False, fontsize=7, handlelength=2.6,
               bbox_to_anchor=(0.5, 1.13), borderaxespad=0)
    fig.subplots_adjust(left=0.15, right=0.88, bottom=0.10, top=0.84, hspace=0.38)
    fig.savefig(out / "drawer_learning_curves.pdf")
    fig.savefig(out / "drawer_learning_curves.png", dpi=300, facecolor="white")
    plt.close(fig)
    with open(out / "drawer_learning_curves_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "sac_seed", "run", "curve", "training_interactions", "validation_rate", "selected"])
        for run in runs:
            for name in ("complete", "skill"):
                for t, rate in run[name] or []:
                    writer.writerow([run["method"], run["seed"], run["run"], COMPLETE[run["method"]] if name == "complete"
                                     else SKILL, t, rate, t == run["selected_at"]])


def drawer_rows(runs: list[dict], final: dict) -> list[dict]:
    rows = []
    for run in runs:
        name = f"{run['method']}, SAC seed {run['seed']}"
        episodes = final["results"][name]
        n, won = len(episodes), sum(e["success"] for e in episodes)
        lo, hi = wilson(won, n)
        valid = [e for e in episodes if e["forward_ok"] and e["start_ok"]]
        rows.append({
            "policy": name, "run": run["run"], "configurations": n, "success": won, "wilson_low": lo, "wilson_high": hi,
            "valid_forward": len(valid), "success_given_valid_forward": sum(e["success"] for e in valid),
            "policy_target": sum(bool(e.get("policy_target_achieved")) for e in episodes),
            "goal_before_settling": sum(bool(e.get("goal_before_settling")) for e in episodes),
            "lost_while_settling": sum(bool(e.get("goal_before_settling")) and not e["success"] for e in episodes),
            "gained_while_settling": sum(not e.get("goal_before_settling") and e["success"] for e in valid),
            "first_90_complete": first_90(run["complete"]), "stayed_90_complete": stayed_90(run["complete"]),
            "best_complete": max(r for _, r in run["complete"]), "final_complete": run["complete"][-1][1],
            "first_90_skill": first_90(run["skill"]) if run["skill"] else None,
            "selected_at": run["selected_at"],
            "failures": {k: v for k, v in final["summary"][name]["failures"].items()},
        })
    return rows


def drawer_tables(rows: list[dict], out: Path) -> None:
    head = ["Policy", "Goal after settling (k/50)", "95% interval", "Goal at the policy's end", "Lost in settling",
            "First ≥90% complete-goal validation", "First ≥90% grasp validation", "Best/final complete-goal validation",
            "Stayed ≥90%", "Selected at"]
    md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    tex = ["\\begin{table*}[t]", "\\centering", "\\footnotesize",
           "\\caption{Targeted versus direct learning on 50 unseen drawer configurations (7000--7049).}",
           "\\label{tab:drawer-round2}", "\\begin{tabular}{@{}lccccccc@{}}", "\\toprule",
           "Policy & Goal after settling & 95\\% interval & Goal at policy end & Lost in settling & "
           "First $\\geq$90\\% (complete goal) & First $\\geq$90\\% (grasp) & Stayed $\\geq$90\\% \\\\", "\\midrule"]
    for r in rows:
        cells = [r["policy"], f"{r['success']}/{r['configurations']}", interval(r["success"], r["configurations"]),
                 f"{r['goal_before_settling']}/{r['configurations']}", str(r["lost_while_settling"]),
                 kilo(r["first_90_complete"]), kilo(r["first_90_skill"]) if r["policy"].startswith("targeted") else "—",
                 f"{100 * r['best_complete']:.0f}/{100 * r['final_complete']:.0f}%", "yes" if r["stayed_90_complete"] else "no",
                 kilo(r["selected_at"])]
        md.append("| " + " | ".join(cells) + " |")
        tex.append(" & ".join(cells[:7] + [cells[8]]).replace("%", "\\%") + " \\\\")
    for method in ("targeted", "direct"):
        mine = [r for r in rows if r["policy"].startswith(method)]
        wins = [r["success"] for r in mine]
        firsts = [r["first_90_complete"] for r in mine]
        median = kilo(statistics.median(firsts)) if None not in firsts else "n/a"
        md.append(f"| {method}, over the 4 seeds | {sum(wins)}/{sum(r['configurations'] for r in mine)}; "
                  f"per seed {min(wins)}–{max(wins)}, median {statistics.median(wins):g} | | | | median {median} | | | | |")
        tex.append(f"\\midrule {method.capitalize()}, 4 seeds & {sum(wins)}/{sum(r['configurations'] for r in mine)} "
                   f"(per seed {min(wins)}--{max(wins)}) & & & & median {median} & & \\\\")
    note = ("Four SAC runs per method, each at its validation-selected checkpoint, on the same 50 configurations with "
            "identical forward initializations. Success is the complete signed restoration goal, every literal judged "
            "robustly, after the established 10 settling steps. Validation (10 configurations every 2,500 interactions, "
            "budget 30k) measures the complete goal for both methods: targeted after its scripted pull_open and "
            "settling, direct after its 100 steps and settling; the targeted checkpoint is selected on its grasp "
            "validation, as established. Wilson intervals are per policy. Four runs per method do not support strong "
            "statistical claims.")
    tex += ["\\bottomrule", "\\end{tabular}", "", f"\\vspace{{2pt}}\\parbox{{\\textwidth}}{{\\scriptsize {note.replace('%', chr(92) + '%')}}}",
            "\\end{table*}"]
    (out / "drawer_table.md").write_text("\n".join(md) + f"\n\n{note}\n")
    (out / "drawer_table.tex").write_text("\n".join(tex) + "\n")
    with open(out / "drawer_table_data.csv", "w", newline="") as f:
        keys = [k for k in rows[0] if k != "failures"] + ["failures"]
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "failures": json.dumps(r["failures"])})


# -- Stow ----------------------------------------------------------------------------------------------


def stow_figure(summary: dict, out: Path) -> None:
    history = summary["history"]
    x = [h["timesteps"] / 1000 for h in history]
    selected = summary["selected"]["timesteps"] / 1000
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(3.5, 3.6), sharex=True)
    top.plot(x, [100 * h["contract_rate_at_stop"] for h in history], color=BLUE, ls="-", lw=1.4, marker="o", ms=4.0,
             mec="white", mew=0.5, label="C_exec: contract where Stage 4 stops (primary)", clip_on=False, zorder=4)
    top.plot(x, [100 * h["contract_rate_at_40"] for h in history], color=ORANGE, ls="--", lw=1.0, marker="s", ms=2.8,
             mec="white", mew=0.4, label="C_40: contract at step 40 (recorded)", clip_on=False, zorder=3)
    bottom.plot(x, [h["p10_margin_at_40"] for h in history], color=BLUE, ls="-", lw=1.1, marker="o", ms=3.4,
                mec="white", mew=0.5, clip_on=False, zorder=3)
    bottom.axhline(0.0, color=MUTED, lw=0.6, ls=":", zorder=1)
    for ax in (top, bottom):
        ax.axvline(selected, color=MUTED, lw=0.6, ls="-", zorder=1)
    top.annotate(f"selected: {selected:g}k", (selected, 104), xytext=(-3, 0), textcoords="offset points", fontsize=6.5,
                 color=INK, va="bottom", ha="right", annotation_clip=False)
    top.set_ylim(-5, 105)
    top.set_yticks(range(0, 101, 20))
    xmax = max(x)
    style(top, "Validation contract (%)", xmax)
    style(bottom, "P10 margin at step 40", xmax)
    bottom.set_xlabel("Training interactions (thousands)", color=INK)
    top.legend(loc="lower center", ncol=1, frameon=False, fontsize=6.5, handlelength=2.6,
               bbox_to_anchor=(0.5, 1.09), borderaxespad=0)
    fig.subplots_adjust(left=0.17, right=0.95, bottom=0.12, top=0.82, hspace=0.18)
    fig.savefig(out / "stow_selection.pdf")
    fig.savefig(out / "stow_selection.png", dpi=300, facecolor="white")
    plt.close(fig)
    with open(out / "stow_selection_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["training_interactions", "contract_rate_at_stop", "p10_margin_at_40", "contract_rate_at_40",
                         "mean_steps", "selected"])
        for h in history:
            writer.writerow([h["timesteps"], h["contract_rate_at_stop"], h["p10_margin_at_40"], h["contract_rate_at_40"],
                             h["mean_steps"], h["timesteps"] == summary["selected"]["timesteps"]])


STOW_CONDITIONS = (("full", "Full plan: planner + learned drag + retreat_to_safe"), ("library_only", "Library only"),
                   ("without_interference", "Without the interference model"),
                   ("relearned", "Relearned: the drag as an extracted library operator"))


def stow_tables(summary: dict, stage4: dict | None, out: Path) -> list[dict]:
    rows = []
    for split in ("best", "test"):
        stats = summary[split]
        n = len(stats["episodes"])
        rows.append({"part": "Stage 3", "item": "validation (4000-4019)" if split == "best" else "test (5000-5049)",
                     "n": n, "contract_at_stop": round(stats["contract_rate_at_stop"] * n),
                     "contract_at_40": round(stats["contract_rate_at_40"] * n), "p10_margin_at_40": stats["p10_margin_at_40"],
                     "mean_steps": stats["mean_steps"], "handoffs_valid": stats["handoffs_valid"]})
    if stage4 is not None:
        for key, label in STOW_CONDITIONS:
            results = stage4["results"][key]
            n = len(results)
            rows.append({"part": "Stage 4", "item": label, "n": n,
                         "goal_met": sum(bool(r["goal_met"]) for r in results),
                         "goal_met_boolean": sum(bool(r["goal_met_boolean"]) for r in results)})
    selected = summary["selected"]
    md = [f"Selected checkpoint: {selected['timesteps']:,} interactions (C_exec {100 * selected['contract_rate_at_stop']:.0f}%, "
          f"P10 {selected['p10_margin_at_40']:+.3f}). Stage 3: {'PASS' if summary['passed'] else 'FAIL'}.", "",
          "| Stage 3 | Contract where Stage 4 stops (C_exec) | 95% interval | Contract at step 40 | P10 margin at step 40 | Mean steps to the target |",
          "|---|---|---|---|---|---|"]
    for r in rows[:2]:
        md.append(f"| {r['item']} | {r['contract_at_stop']}/{r['n']} | {interval(r['contract_at_stop'], r['n'])} | "
                  f"{r['contract_at_40']}/{r['n']} | {r['p10_margin_at_40']:+.3f} | {r['mean_steps']:.1f} |")
    tex = ["\\begin{table}[t]", "\\centering", "\\footnotesize", "\\caption{StowCube: the learned drag and the full inverse.}",
           "\\label{tab:stow-round2}", "\\begin{tabular}{@{}lccc@{}}", "\\toprule",
           "Stage 3 & Contract at the stop & 95\\% interval & P10 margin \\\\", "\\midrule"]
    for r in rows[:2]:
        tex.append(f"{r['item'].replace('-', '--')} & {r['contract_at_stop']}/{r['n']} & "
                   f"{interval(r['contract_at_stop'], r['n']).replace('%', chr(92) + '%')} & {r['p10_margin_at_40']:+.3f} \\\\")
    if stage4 is not None:
        md += ["", "| Stage 4, scenes 5000-5049 | Complete signed goal | 95% interval | Boolean goal |", "|---|---|---|---|"]
        tex += ["\\midrule", "Stage 4 (5000--5049) & Complete signed goal & 95\\% interval & Boolean \\\\", "\\midrule"]
        for r in rows[2:]:
            md.append(f"| {r['item']} | {r['goal_met']}/{r['n']} | {interval(r['goal_met'], r['n'])} | {r['goal_met_boolean']}/{r['n']} |")
            tex.append(f"{r['item'].replace('_', chr(92) + '_')} & {r['goal_met']}/{r['n']} & "
                       f"{interval(r['goal_met'], r['n']).replace('%', chr(92) + '%')} & {r['goal_met_boolean']}/{r['n']} \\\\")
        validity = stage4["handoff_validity"]
        md += ["", f"Target among the learned operator's effects: {'yes' if stage4['learned_effects_contain_the_target'] else 'no'}. "
                   f"Handoff validity: demonstrations {validity['demonstrations']['valid_handoffs']} of "
                   f"{validity['demonstrations']['reached_the_handoff']} scenes that reached the handoff; evaluation scenes "
                   f"{validity['evaluation_seeds']['valid']} of {validity['evaluation_seeds']['of']}. "
                   f"Stage 4: {'PASS' if stage4['passed'] else 'FAIL'}."]
    else:
        md += ["", "Stage 4 was not run: Stage 3 did not pass its gate (PLAN.md section 5)."]
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (out / "stow_table.md").write_text("\n".join(md) + "\n")
    (out / "stow_table.tex").write_text("\n".join(tex) + "\n")
    with open(out / "stow_table_data.csv", "w", newline="") as f:
        keys = sorted({k for r in rows for k in r})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    return rows


# -- previous results ---------------------------------------------------------------------------------


def previous(out: Path) -> None:
    e4_final = read(SL / "E4_stage4/final_targeted_vs_direct.json")["summary"]
    full_horizon = read(SL / "D_diagnostics/direct_full_horizon.json")["summary"]
    e4_stow = read(SL / "E4_stage4/stow_stage4.json")
    fresh = read(SL / "D_diagnostics/stow_stage4_fresh_final_model.json")
    historical = read(SL / "D_diagnostics/stow_stage4_historical_checkpoint.json")
    md = ["# Previous results (signed-literal rerun, 2026-11-11), labelled",
          "", "Recorded under the previous protocol; kept as they were. Not comparable seed for seed with Round 2.", "",
          "## Drawer, configurations 3000-3049", "",
          "Previous protocol: the direct reward had the fence not handle_grasped; the direct policy stopped at the first "
          "step where the goal held, then settled.", "",
          "| Policy | Complete goal after settling (protocol) | Diagnostic: direct run for its full 100 steps |", "|---|---|---|"]
    for name, s in e4_final.items():
        diag = full_horizon.get(name, {}).get("success")
        md.append(f"| {name} | {s['success']}/{s['configurations']} | {'' if diag is None else f'{diag}/50'} |")
    md += ["", "## StowCube, scenes 1000-1009", "",
           "Previous protocol: checkpoint chosen on the contract at step 40, then the fewest mean steps, on the same "
           "scenes as Stage 4; it chose step 4,000.", "",
           "| Checkpoint | Full | Library only | Without interference | Relearned | Target in the learned operator |",
           "|---|---|---|---|---|---|"]
    for label, record in (("protocol-selected, step 4,000 (E4)", e4_stow), ("diagnostic: the run's final model, 120,000", fresh),
                          ("diagnostic: historical checkpoint 1790788534", historical)):
        s = record["summary"]
        md.append(f"| {label} | " + " | ".join(f"{round(10 * s[c]['goal_met_rate'])}/10" for c, _ in STOW_CONDITIONS)
                  + f" | {'yes' if record['learned_effects_contain_the_target'] else 'no'} |")
    (out / "previous_results.md").write_text("\n".join(md) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; outputs are never overwritten")
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                         "font.family": "DejaVu Sans"})
    runs = drawer_runs("targeted") + drawer_runs("direct")
    final = read(R3 / "final_targeted_vs_direct.json")
    stow_runs = sorted((R1 / "stow").glob("*/summary.json"))
    if len(stow_runs) != 1:
        raise SystemExit(f"expected one Stow run, found {len(stow_runs)}")
    stow_summary = read(stow_runs[0])
    stage4_path = R2 / "stow_stage4.json"
    stage4 = read(stage4_path) if stage4_path.exists() else None

    args.out.mkdir(parents=True)
    drawer_figure(runs, args.out)
    drawer_tables(drawer_rows(runs, final), args.out)
    stow_figure(stow_summary, args.out)
    stow_tables(stow_summary, stage4, args.out)
    previous(args.out)
    (args.out / "sources.json").write_text(json.dumps({str(p): sha256(p) for p in sorted(set(SOURCES))}, indent=1))
    print(f"wrote {args.out}: " + ", ".join(sorted(p.name for p in args.out.iterdir())))


if __name__ == "__main__":
    main()
