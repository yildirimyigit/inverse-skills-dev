"""manifest.json: a compact record of this verification, covering the files that stay out of git.

    python artifacts/final_verification_fe374f0/make_manifest.py

Run from the repository root, after the last run. It only reads the files in this folder: nothing
is run and nothing is planned. It writes manifest.json next to this file, never overwriting it.

  planner           the checkout and planner files every run checked (the same in every record)
  environment       the Python and packages every run used (the same in every record)
  runs              each run in the order it started, from its record in provenance/: command,
                    outcome, time, and how many planner calls each implementation answered.
                    The records' per-call lists are left out.
  result_summaries  the summaries the large outputs carry, copied unchanged, with their seeds
  files             every file in this folder except manifest.json: size, sha256, whether
                    .gitignore keeps it in git, and the tracked files elsewhere with the same bytes
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path("artifacts/final_verification_fe374f0")
OUT = HERE / "manifest.json"


def load(name: str):
    return json.loads((HERE / name).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runs() -> tuple[dict, list]:
    """The fields every record shares, checked to be equal, and one row per run."""
    records = {path.stem: json.loads(path.read_text()) for path in sorted((HERE / "provenance").glob("*.json"))}
    shared = {}
    for field in ("checkout", "planner_modules", "environment"):
        if len({json.dumps(record[field], sort_keys=True) for record in records.values()}) != 1:
            raise SystemExit(f"the records differ in {field!r}; nothing written")
        shared[field] = next(iter(records.values()))[field]
    rows = [{"run": name, "record": f"provenance/{name}.json",
             **{field: record[field] for field in ("command", "started", "seconds", "outcome",
                                                     "checks_passed", "planner_calls")}}
            for name, record in sorted(records.items(), key=lambda item: item[1]["started"])]
    return shared, rows


def result_summaries() -> dict:
    """What the large outputs, which stay out of git, say about themselves."""
    stow, drawer = load("stow_stage4.json"), load("drawer_stage4.json")
    final, checks = load("drawer_final_targeted_vs_direct.json"), load("planning_checks.json")
    validity = stow["handoff_validity"]
    return {
        "stow_stage4.json": {
            "seeds": sorted({row["seed"] for rows in stow["results"].values() for row in rows}),
            "summary": stow["summary"],
            "passed": stow["passed"],
            "handoff_validity": {
                "evaluation_seeds": {key: validity["evaluation_seeds"][key] for key in ("conditions", "valid", "of")},
                "demonstrations": validity["demonstrations"]}},
        "drawer_stage4.json": {key: drawer[key] for key in ("seeds", "summary")},
        "drawer_final_targeted_vs_direct.json": {
            key: final[key] for key in ("seeds", "protocol", "checkpoints", "start_states_identical", "summary")},
        "planning_checks.json": {"summary": checks["summary"]},
    }


def kept_in_git(path: Path) -> bool:
    code = subprocess.run(["git", "check-ignore", "-q", str(path)]).returncode
    if code not in (0, 1):
        raise SystemExit(f"git check-ignore failed on {path}")
    return code == 1      # 1: no ignore rule applies to the file


def files() -> dict:
    tracked_elsewhere = {}      # size -> tracked files outside this folder
    listing = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True).stdout
    for name in listing.splitlines():
        path = Path(name)
        if HERE not in path.parents and path.is_file():
            tracked_elsewhere.setdefault(path.stat().st_size, []).append(path)
    rows = {}
    for path in sorted(p for p in HERE.rglob("*") if p.is_file() and p != OUT):
        digest = sha256(path)
        rows[path.relative_to(HERE).as_posix()] = {
            "bytes": path.stat().st_size,
            "sha256": digest,
            "kept_in_git": kept_in_git(path),
            "same_bytes_tracked_at": [str(other) for other in tracked_elsewhere.get(path.stat().st_size, [])
                                      if sha256(other) == digest]}
    return rows


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; it is never overwritten")
    shared, run_rows = runs()
    manifest = {
        "about": "Final verification of the minimal hole planner at fe374f0, run 2026-11-09; see REPORT.md. "
                 "Made by make_manifest.py from the files in this folder, with nothing rerun. Files with "
                 "kept_in_git false stay local; their checksums let a copy or a rerun be checked.",
        "planner": {**shared["checkout"], **shared["planner_modules"]},
        "environment": shared["environment"],
        "runs": run_rows,
        "result_summaries": result_summaries(),
        "files": files(),
    }
    OUT.write_text(json.dumps(manifest, indent=1))
    kept = sum(row["kept_in_git"] for row in manifest["files"].values())
    print(f"{len(run_rows)} runs, {len(manifest['files'])} files ({kept} kept in git); wrote {OUT}")


if __name__ == "__main__":
    main()
