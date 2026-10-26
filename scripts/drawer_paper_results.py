"""The paper's drawer figure and table, built only from saved results.

Inputs (read, never changed):
  artifacts/drawer/stage3/<run>/{config,summary}.json   targeted runs: validation curves
  artifacts/drawer/direct/<run>/{config,summary}.json   direct runs: validation curves
  artifacts/drawer/final_targeted_vs_direct.json        final evaluation, seeds 3000-3049

Outputs, in artifacts/drawer/paper/ (refuses to overwrite):
  learning_curves.pdf / .png     validation success against training interactions, all
                                 eight runs as recorded: every evaluation is marked and
                                 joined by straight lines, nothing smoothed or averaged
  learning_curves_data.csv       the exact points plotted
  results_table.tex / .md        the results table
  results_table_data.csv         the exact numbers behind it, per policy
  sources.json                   SHA-256 of every input file

Validation (seeds 500-509, every 2,500 interactions) measures each method's learned
part: for targeted, handle_grasped at the end of a 40-step episode; for direct, the
restoration goal at the end of a 100-step episode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGE3 = Path("artifacts/drawer/stage3")
DIRECT = Path("artifacts/drawer/direct")
FINAL = Path("artifacts/drawer/final_targeted_vs_direct.json")
METHODS = {   # name: (folder, validation key, colour, line style, marker, legend label)
    "targeted": (STAGE3, "validation_success_rate", "#2a78d6", "-", "o", "Targeted (learned grasp)"),
    "direct": (DIRECT, "validation_goal_rate", "#eb6834", "--", "s", "Direct (learned opening)"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
THRESHOLD = 0.9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runs(method: str) -> list[dict]:
    """Each run of a method: seed, folder, and its recorded validation curve."""
    folder, key = METHODS[method][:2]
    found = []
    for summary in sorted(folder.glob("*/summary.json")):
        config = summary.parent / "config.json"
        history = json.loads(summary.read_text())["history"]
        found.append({"method": method, "seed": json.loads(config.read_text())["seed"], "run": summary.parent.name,
                      "sources": [config, summary],
                      "curve": [(h["timesteps"], h[key]) for h in history],
                      "selected_at": [h["timesteps"] for h in history if h.get("saved")][-1]})
    return sorted(found, key=lambda r: r["seed"])


def learning(run: dict) -> dict:
    curve = run["curve"]
    first = next((i for i, (_, rate) in enumerate(curve) if rate >= THRESHOLD), None)
    return {"first_90_at": curve[first][0] if first is not None else None,
            "stayed_90": first is not None and all(rate >= THRESHOLD for _, rate in curve[first:]),
            "best": max(rate for _, rate in curve), "final": curve[-1][1], "selected_at": run["selected_at"]}


def figure(all_runs: list[dict], out: Path) -> None:
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.6,
                         "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(3.5, 2.35))
    for method, (_, _, colour, style, marker, label) in METHODS.items():
        for i, run in enumerate(r for r in all_runs if r["method"] == method):
            x = [t / 1000 for t, _ in run["curve"]]
            y = [100 * rate for _, rate in run["curve"]]
            ax.plot(x, y, color=colour, ls=style, lw=1.1, marker=marker, ms=3.4, mec="white", mew=0.5,
                    alpha=0.8, label=label if i == 0 else None, clip_on=False, zorder=3)
    ax.axhline(100 * THRESHOLD, color=MUTED, lw=0.6, ls=":", zorder=1)
    # Label the runs that ended below the threshold, at their last point, in the right margin.
    for run in all_runs:
        end_t, end_rate = run["curve"][-1]
        if end_rate < THRESHOLD:
            ax.annotate(f"{run['method']} seed {run['seed']}", (end_t / 1000, 100 * end_rate), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=6.5, color=INK, annotation_clip=False)
    ax.annotate("90%", (30, 100 * THRESHOLD), xytext=(5, 0), textcoords="offset points", va="center",
                fontsize=6.5, color=MUTED, annotation_clip=False)
    ax.set_xlim(0, 30)
    ax.set_ylim(-5, 105)
    ax.set_xticks(range(0, 31, 5))
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlabel("Training interactions (thousands)", color=INK)
    ax.set_ylabel("Validation success (%)", color=INK)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#9a9992")
    ax.tick_params(labelsize=7, colors=MUTED, length=2.5, width=0.6)
    ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=7, handlelength=2.6, columnspacing=1.2,
              bbox_to_anchor=(0.5, 1.02), borderaxespad=0)
    fig.subplots_adjust(left=0.14, right=0.80, bottom=0.18, top=0.86)
    fig.savefig(out / "learning_curves.pdf")
    fig.savefig(out / "learning_curves.png", dpi=300, facecolor="white")


def per_seed(values) -> str:
    return ", ".join(values)


def k(steps) -> str:
    return "never" if steps is None else f"{steps / 1000:g}k"


def table_rows(all_runs: list[dict], final: dict) -> tuple[list[dict], dict]:
    """Per-policy numbers, and the method-level lines of the table."""
    rows = []
    for run in all_runs:
        name = f"{run['method']}, SAC seed {run['seed']}"
        episodes = final["results"][name]
        valid = [e for e in episodes if e["forward_ok"] and e["start_ok"]]
        rows.append({"policy": name, "run": run["run"], **learning(run),
                     "successes": sum(e["success"] for e in episodes), "configurations": len(episodes),
                     "valid_forward": len(valid), "successes_given_valid_forward": sum(e["success"] for e in valid)})
    lines = {}
    for method in METHODS:
        mine = [r for r in rows if r["policy"].startswith(method)]
        won_valid, valid = [r["successes_given_valid_forward"] for r in mine], [r["valid_forward"] for r in mine]
        won, tried = [r["successes"] for r in mine], [r["configurations"] for r in mine]
        firsts = [r["first_90_at"] for r in mine]
        median = k(statistics.median(firsts)) if None not in firsts else "n/a"
        lines[method] = {
            "conditional": f"{sum(won_valid)}/{sum(valid)} ({per_seed(f'{a}/{b}' for a, b in zip(won_valid, valid))})",
            "full": f"{sum(won)}/{sum(tried)} ({per_seed(f'{a}/{b}' for a, b in zip(won, tried))})",
            "first_90": f"{per_seed(k(t) for t in firsts)} (median {median})",
            "best_final": per_seed(f"{100 * r['best']:.0f}/{100 * r['final']:.0f}" for r in mine),
            "stayed": f"{sum(r['stayed_90'] for r in mine)}/{len(mine)} runs",
        }
    return rows, lines


TABLE_ROWS = [   # key, label
    ("conditional", "Inverse success, valid forward execution"),
    ("full", "Full-pipeline success (incl. shared forward failure)"),
    ("first_90", "Interactions to first reach $\\geq$90\\% validation"),
    ("best_final", "Best / final validation (\\%)"),
    ("stayed", "Stayed $\\geq$90\\% after first reaching it"),
]
NOTE = ("Four SAC runs per method (seeds 0--3, listed in order), each at its validation-selected checkpoint, "
        "on the same 50 unseen configurations. The one shared failure is a forward PushClose that did not leave "
        "the drawer closed, so no inverse was attempted. Validation: 10 configurations every 2,500 interactions "
        "(budget 30k); it scores each method's learned part: the grasp for targeted, the restoration goal for "
        "direct. Four runs per method do not support strong statistical claims.")


def plain(label: str) -> str:
    """A table label without LaTeX."""
    return label.replace("$\\geq$", "≥").replace("\\%", "%")


def write_tables(lines: dict, out: Path) -> None:
    tex = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
           "\\caption{Targeted versus direct learning on the drawer task.}", "\\label{tab:drawer-targeted-direct}",
           "\\begin{tabular}{@{}p{0.34\\columnwidth}p{0.29\\columnwidth}p{0.29\\columnwidth}@{}}", "\\toprule",
           " & Targeted & Direct \\\\", "\\midrule"]
    for key, label in TABLE_ROWS:
        tex.append(f"{label} & {lines['targeted'][key]} & {lines['direct'][key]} \\\\")
    note = NOTE.replace("%", "\\%")
    tex += ["\\bottomrule", "\\end{tabular}", "",
            f"\\vspace{{2pt}}\\parbox{{\\columnwidth}}{{\\scriptsize {note}}}", "\\end{table}"]
    (out / "results_table.tex").write_text("\n".join(tex) + "\n")

    md = ["| | Targeted | Direct |", "|---|---|---|"]
    md += [f"| {plain(label)} | {lines['targeted'][key]} | {lines['direct'][key]} |" for key, label in TABLE_ROWS]
    (out / "results_table.md").write_text("\n".join(md) + f"\n\n{NOTE}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("artifacts/drawer/paper"))
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; outputs are never overwritten")

    all_runs = runs("targeted") + runs("direct")
    final = json.loads(FINAL.read_text())
    rows, lines = table_rows(all_runs, final)
    args.out.mkdir(parents=True)
    figure(all_runs, args.out)
    write_tables(lines, args.out)

    with open(args.out / "learning_curves_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "sac_seed", "run", "training_interactions", "validation_success",
                         "selected_checkpoint"])
        for run in all_runs:
            for steps, rate in run["curve"]:
                writer.writerow([run["method"], run["seed"], run["run"], steps, rate, int(steps == run["selected_at"])])
    with open(args.out / "results_table_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources = [p for run in all_runs for p in run["sources"]] + [FINAL]
    (args.out / "sources.json").write_text(json.dumps({
        "made_by": "python scripts/drawer_paper_results.py",
        "inputs": {str(p): sha256(p) for p in sources}}, indent=2))

    for key, label in TABLE_ROWS:
        print(f"{plain(label):58s} | {lines['targeted'][key]:42s} | {lines['direct'][key]}")
    print(f"\nwrote {args.out}/: learning_curves.pdf/.png, learning_curves_data.csv, results_table.tex/.md, "
          f"results_table_data.csv, sources.json")


if __name__ == "__main__":
    main()
