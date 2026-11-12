"""Round 2 seed audit: were the proposed scene seeds ever used before?

Reads, never writes anything but --out:
  every JSON record under artifacts/   the seeds it names (keys seed, seeds, scene_seed(s),
                                       test_seeds, original_seeds, seed_pool, episode_seed,
                                       and train/validation/test/selection_and_evaluation
                                       under a seeds key; demo ids such as "run_2004")
  every log and report under artifacts/ "seed N", "seeds N-M", "scene N", "configurations N-M"
  the scripts                          their seed constants and --seed(s)/--start defaults

A seed's task is read from its file's path ("stow" or "drawer"); a file whose path names
neither counts against both tasks.

Runs whose outputs never reached artifacts/ (smoke tests, probes) are in OUTSIDE_ARTIFACTS:
the seeds found in this project's shell-command history and in the session scratchpad,
2026-09-17 to 2026-11-12, read by hand on 2026-11-12.

StowCube re-rolls a scene at seed + 10,000 k, k = 1..4 (the forward skill's re-rolls in
Stage 4 and the retreat gate, the hole environment's handoffs, the demonstrations'
replacements). So two Stow seeds can reach the same scene exactly when they are equal
modulo 10,000. The check is that one, for both tasks: it is exact for Stow and stricter
than needed for the Drawer, which never re-rolls.

Usage: PYTHONNOUSERSITE=1 python artifacts/round2_20261112/seed_audit.py --out <file>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

STRIDE = 10_000
ATTEMPTS = 5                       # re-rolls k = 0..4
CANDIDATES = {                     # name: (task, first seed, last seed)
    "stow_validation": ("stow", 4000, 4019),
    "stow_final_test": ("stow", 5000, 5049),
    "drawer_final_test": ("drawer", 7000, 7049),
}
OUTSIDE_ARTIFACTS = {
    # task: {seed: where it was used}
    "drawer": {5000: "drawer_final_evaluate.py --seeds 5000, a smoke run on 2026-11-11 (output in the scratchpad)"},
    "both": {s: "command history or scratchpad outputs (smoke runs, probes)"
             for s in [*range(32), 77, 78, 99, 109, *range(500, 510), *range(1000, 1020), *range(1101, 1105),
                       *range(2000, 2020), 12001, 12004]},
}
SEED_KEYS = {"seed", "seeds", "scene_seed", "scene_seeds", "test_seeds", "original_seeds",
             "seed_pool", "episode_seed", "traced_seeds", "extraction_seeds"}
SPLIT_KEYS = {"train", "validation", "test", "selection_and_evaluation"}
TEXT = re.compile(r"\b(?:seeds?|scenes?|configurations?)\s*[:=]?\s*(\d{1,5})(?:\s*(?:-|–|to)\s*(\d{1,5}))?", re.I)
RANGE = re.compile(r"range\(\s*(\d+)\s*,\s*(\d+)\s*\)")
DEFAULT = re.compile(r"default=\[([\d, ]+)\]|default=(\d+)")


def task_of(path: Path) -> tuple[str, ...]:
    text = str(path).lower()
    tasks = tuple(t for t in ("stow", "drawer") if t in text)
    return tasks or ("stow", "drawer")


def ints(value) -> list[int]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [v for v in value if isinstance(v, int) and not isinstance(v, bool)]
    return []


def from_json(node, parent: str | None = None) -> list[int]:
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            k = key.lower()
            if k in SEED_KEYS or (parent in SEED_KEYS and k in SPLIT_KEYS):
                found += ints(value)
            if k == "demo_id" and isinstance(value, str):
                found += [int(x) for x in re.findall(r"\d+", value)]
            found += from_json(value, k)
    elif isinstance(node, list):
        for item in node:
            found += from_json(item, parent)
    return found


def from_text(text: str) -> list[int]:
    found = []
    for m in TEXT.finditer(text):
        a, b = int(m.group(1)), m.group(2)
        if b is not None and int(b) >= a and int(b) - a <= 200:
            found += list(range(a, int(b) + 1))
        else:
            found.append(a)
    return found


def from_script(text: str) -> list[int]:
    found = []
    for line in text.splitlines():
        if not re.search(r"seed|SEEDS|--start", line):
            continue
        for a, b in RANGE.findall(line):
            found += list(range(int(a), int(b)))
        for listed, single in DEFAULT.findall(line):
            found += [int(x) for x in listed.split(",") if x.strip()] if listed else [int(single)]
    return found


def ranges(values) -> list[str]:
    out, run = [], []
    for v in sorted(set(values)):
        if run and v == run[-1] + 1:
            run.append(v)
        else:
            if run:
                out.append(f"{run[0]}" if len(run) == 1 else f"{run[0]}-{run[-1]}")
            run = [v]
    if run:
        out.append(f"{run[0]}" if len(run) == 1 else f"{run[0]}-{run[-1]}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; results are never overwritten")
    used: dict[str, dict[int, set[str]]] = {"stow": {}, "drawer": {}}

    def note(path: Path, seeds, tasks):
        for task in tasks:
            for s in seeds:
                used[task].setdefault(s, set()).add(str(path))

    files = 0
    for path in sorted(Path("artifacts").rglob("*")):
        if not path.is_file() or "round2_" in str(path):
            continue
        if path.suffix == ".json":
            try:
                note(path, from_json(json.loads(path.read_text())), task_of(path))
                files += 1
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        elif path.suffix in {".log", ".stdout", ".out", ".txt", ".md"}:
            note(path, from_text(path.read_text(errors="replace")), task_of(path))
            files += 1
    for path in sorted(Path("scripts").glob("*.py")):
        note(path, from_script(path.read_text()), task_of(path))
        files += 1
    for task, uses in OUTSIDE_ARTIFACTS.items():
        for seed, where in uses.items():
            for t in (("stow", "drawer") if task == "both" else (task,)):
                used[t].setdefault(seed, set()).add(where)

    verdicts = {}
    for name, (task, first, last) in CANDIDATES.items():
        residues = {s % STRIDE: s for s in range(first, last + 1)}
        clashes = sorted({u for u in used[task] if u < STRIDE * ATTEMPTS and u % STRIDE in residues})
        verdicts[name] = {
            "task": task, "seeds": f"{first}-{last}",
            "scenes_reachable_with_re_rolls": [f"{first + k * STRIDE}-{last + k * STRIDE}" for k in range(ATTEMPTS)],
            "clean": not clashes,
            "clashes": {str(u): sorted(used[task][u]) for u in clashes},
        }
        print(f"{name:18s} {task:6s} {first}-{last}: {'CLEAN' if not clashes else 'USED: ' + ', '.join(map(str, clashes))}")

    record = {
        "rule": "a candidate seed is clean when no seed recorded for its task (or for an unnamed task) is congruent "
                "to it modulo 10,000 below 50,000, i.e. no recorded run could have reached the same scene",
        "files_read": files,
        "used_seeds": {task: ranges(seen) for task, seen in used.items()},
        "candidates": verdicts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))
    for task, seen in record["used_seeds"].items():
        print(f"\nused for {task}: {', '.join(seen)}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
