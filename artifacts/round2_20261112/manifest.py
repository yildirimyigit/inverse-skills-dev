"""Round 2's manifest: every step with its command, commit, inputs, outputs and hashes, read from the
runner's records and the files; the environment; the large outputs' summaries. Nothing is rerun.

    PYTHONNOUSERSITE=1 python artifacts/round2_20261112/manifest.py

Writes manifest.json next to this file, never overwriting it.

  steps        one per runner record: the command, the commit (and whether the tree was clean), the
               start time, the duration, the exit code, and the command's path arguments split into
               inputs (read) and outputs (written: --out, --out-dir, --learned-out), each with its
               files' sizes and SHA-256
  environment  Python and package versions, the GPU, PYTHONNOUSERSITE
  seeds        the scene seeds every training run recorded
  files        every file in this folder (except this manifest and logs/): size and SHA-256
  large        the JSON outputs over 1 MB, which stay out of git: their summary blocks, so the numbers
               the report cites are tracked with their checksums
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from importlib import metadata
from pathlib import Path

import torch

HERE = Path("artifacts/round2_20261112")
OUT = HERE / "manifest.json"
WRITES = {"--out", "--out-dir", "--learned-out"}
LARGE_BYTES = 1_000_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(path: Path) -> dict:
    if path.is_dir():
        return {str(p): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(path.rglob("*")) if p.is_file()}
    if path.exists():
        return {str(path): {"bytes": path.stat().st_size, "sha256": sha256(path)}}
    stem = path.with_suffix(".zip")                       # a checkpoint named without its .zip
    return describe(stem) if stem.exists() else {str(path): "missing"}


def paths(command: list[str]) -> tuple[list[Path], list[Path]]:
    reads, writes = [], []
    for i, arg in enumerate(command):
        if arg.startswith(("artifacts/", "scripts/")) and not arg.endswith(".py"):
            (writes if i and command[i - 1] in WRITES else reads).append(Path(arg))
    return reads, writes


def environment() -> dict:
    packages = ("torch", "stable-baselines3", "gymnasium", "mani-skill", "sapien", "numpy", "matplotlib")
    return {"python": platform.python_version(), "packages": {p: metadata.version(p) for p in packages},
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE")}


def seeds() -> dict:
    runs = {}
    for config in sorted((HERE / "R1_training").glob("*/*/config.json")):
        data = json.loads(config.read_text())
        runs[str(config.parent.relative_to(HERE))] = {"sac_seed": data["seed"], "scenes": data["seeds"]}
    return {"training_runs": runs, "stow_stage4": "5000-5049; learned operator from 2000-2009 (EXTRACTION_SEEDS)",
            "drawer_stage4": "2000-2019", "final_comparison": "7000-7049", "r0_regression": "3000-3049"}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; it is never overwritten")
    steps = []
    for record_path in sorted((HERE / "records").glob("*.json")):
        record = json.loads(record_path.read_text())
        reads, writes = paths(record["command"])
        steps.append({"record": record_path.name, **record, "tree_clean": not record["checkout"]["tracked_changes"],
                      "inputs": {str(p): describe(p) for p in reads}, "outputs": {str(p): describe(p) for p in writes}})
    files = {str(p.relative_to(HERE)): {"bytes": p.stat().st_size, "sha256": sha256(p)}
             for p in sorted(HERE.rglob("*")) if p.is_file() and p != OUT and "logs" not in p.parts}
    large = {}
    for name, entry in files.items():
        if name.endswith(".json") and entry["bytes"] > LARGE_BYTES:
            data = json.loads((HERE / name).read_text())
            large[name] = {**entry, **{k: data[k] for k in ("summary", "passed", "protocol", "learning", "selected",
                                                            "best_reproduces_its_record") if k in data}}
    manifest = {"about": "Round 2: every step's command, commit, inputs, outputs and hashes (PLAN.md section 8).",
                "steps": sorted(steps, key=lambda s: s["started"]), "environment": environment(), "seeds": seeds(),
                "files": files, "large_outputs": large}
    OUT.write_text(json.dumps(manifest, indent=1))
    print(f"{len(steps)} steps, {len(files)} files, {len(large)} large outputs; wrote {OUT}")


if __name__ == "__main__":
    main()
