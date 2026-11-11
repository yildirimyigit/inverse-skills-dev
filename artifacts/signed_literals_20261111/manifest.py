"""The experiment manifest: every step of E0-E4 with its inputs, outputs, configuration, seeds,
commit and hashes, read from the runner's records and the files. Nothing is rerun.

    PYTHONNOUSERSITE=1 python artifacts/signed_literals_20261111/manifest.py

Writes manifest.json next to this file, never overwriting it.

  steps    one per runner record: the stage, the command, the commit (and whether the tree was
           clean), the start time, the duration, the exit code, and the command's path arguments
           split into inputs (read) and outputs (written: --out, --spec-out, --out-dir,
           --learned-out); each path with its files' sizes and SHA-256. A record named
           <step>.attempt<k>_<reason>.json is a superseded attempt: its outputs were renamed
           the same way and kept, so its paths are not described (they now hold the rerun);
           the renamed files are listed under files
  seeds    the seeds every stage used, from the outputs that record them
  files    every file in this folder (except this manifest): size and SHA-256
  large    the five large outputs that stay out of git: their summary blocks, so the numbers
           the report cites are tracked with their checksums
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path("artifacts/signed_literals_20261111")
OUT = HERE / "manifest.json"
WRITES = {"--out", "--spec-out", "--out-dir", "--learned-out"}
LARGE = ("E0_stage0/retreat_gate.json", "R_retreat_validation/retreat_gate.json", "E4_stage4/drawer_stage4.json",
         "E4_stage4/final_targeted_vs_direct.json", "D_diagnostics/direct_full_horizon.json")
STAGES = {"E0": ("stage0", "retreat"), "E1": ("stage1",), "E2": ("stage2", "direct_spec", "planner_demo"),
          "E3": ("train", "checkpoints", "precheck"), "E4": ("stage4", "final")}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(path: Path) -> dict:
    if path.is_dir():
        return {str(p): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(path.rglob("*")) if p.is_file()}
    if path.exists():
        return {str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}}
    stem = path.with_suffix(".zip")                       # a checkpoint named without its .zip
    return describe(stem) if stem.exists() else {str(path): "missing"}


def stage_of(step: str) -> str:
    return next((stage for stage, words in STAGES.items() if any(word in step for word in words)), "?")


def paths(command: list[str]) -> tuple[list[Path], list[Path]]:
    reads, writes = [], []
    for i, arg in enumerate(command):
        if arg.startswith(("artifacts/", "scripts/")) and not arg.endswith(".py"):
            (writes if i and command[i - 1] in WRITES else reads).append(Path(arg))
    return reads, writes


def seeds() -> dict:
    def get(path, key):
        return json.loads(path.read_text()).get(key) if path.exists() else None
    e3 = HERE / "E3_stage3"
    return {
        "stage1_demonstrations": {t: get(HERE / "E1_stage1" / f"{t}_stage1_operator.json", "seeds") for t in ("stow", "drawer")},
        "stage2_measured_states": {t: [r["seed"] for r in json.loads((HERE / "E2_stage2" / f"{t}_stage2_plan.json").read_text())["seeds"]]
                                   for t in ("stow", "drawer") if (HERE / "E2_stage2" / f"{t}_stage2_plan.json").exists()},
        "stage3": {"stow": [json.loads(p.read_text())["seeds"] for p in (e3 / "stow").glob("*/config.json")],
                   "drawer": {json.loads(p.read_text())["seed"]: json.loads(p.read_text())["seeds"]
                              for p in (e3 / "drawer").glob("*/config.json")},
                   "direct": {json.loads(p.read_text())["seed"]: json.loads(p.read_text())["seeds"]
                              for p in (e3 / "direct").glob("*/config.json")}},
        "stage4": {"stow_evaluation": "1000-1009 (stow_stage4_execute.py default)",
                   "stow_learned_operator": "2000-2009 (EXTRACTION_SEEDS)",
                   "drawer": get(HERE / "E4_stage4" / "drawer_stage4.json", "seeds"),
                   "final_comparison": get(HERE / "E4_stage4" / "final_targeted_vs_direct.json", "seeds")},
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; it is never overwritten")
    steps = []
    for record_path in sorted((HERE / "records").glob("*.json")):
        record = json.loads(record_path.read_text())
        reads, writes = paths(record["command"])
        superseded = ".attempt" in record_path.name
        steps.append({"record": record_path.name, "stage": stage_of(record["step"]), **record,
                      "tree_clean": not record["checkout"]["tracked_changes"], "superseded": superseded,
                      "inputs": {} if superseded else {str(p): describe(p) for p in reads},
                      "outputs": {} if superseded else {str(p): describe(p) for p in writes}})
    files = {str(p.relative_to(HERE)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(HERE.rglob("*")) if p.is_file() and p != OUT}
    large = {}
    for name in LARGE:
        data = json.loads((HERE / name).read_text())
        large[name] = {**files[name], **{k: data[k] for k in ("summary", "passed", "protocol", "learning", "commit")
                                         if k in data}}
    manifest = {"about": "The signed-literal experiments: every step's command, commit, inputs, outputs and hashes.",
                "steps": sorted(steps, key=lambda s: s["started"]), "seeds": seeds(), "files": files,
                "large_outputs": large}
    OUT.write_text(json.dumps(manifest, indent=1))
    print(f"{len(steps)} steps, {len(files)} files; wrote {OUT}")


if __name__ == "__main__":
    main()
