"""Stage A: the baseline before any signed-literal change. Reads only; runs the test suite once.

    PYTHONNOUSERSITE=1 python artifacts/signed_literals_20261111/00_baseline/capture.py

Run from the repository root. Writes baseline.json and copies of the small reference files into
this folder (refs/), never overwriting anything. Large reference files are recorded by path, size
and sha256 only: they are tracked in git at the recorded commit.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

HERE = Path("artifacts/signed_literals_20261111/00_baseline")
OUT = HERE / "baseline.json"
COPY_LIMIT = 50_000      # bytes: smaller reference files are copied into refs/

REFERENCE_FILES = [
    # StowCube, the published pipeline
    "artifacts/stow/stage0_gate.json",
    "artifacts/stow/stage1_operator.json",
    "artifacts/stow/stage2_plan.json",
    "artifacts/stow/hole_spec.json",
    "artifacts/stow/stage3/1790788534/summary.json",
    "artifacts/stow/stage3/1790788534/precondition.json",
    "artifacts/stow/stage3/1790788534/sac_hole_best.zip",
    "artifacts/stow/learned_operator.json",
    "artifacts/stow/stage4_execution.json",
    "artifacts/stow/planner_demo.json",
    # SlidingDrawer, the published pipeline
    "artifacts/drawer/stage0_gate_after_push_fix.json",
    "artifacts/drawer/stage1_operator.json",
    "artifacts/drawer/stage2_plan.json",
    "artifacts/drawer/hole_spec.json",
    "artifacts/drawer/stage3/checkpoints.json",
    *[f"artifacts/drawer/stage3/{run}/{name}" for run in ("1792053076", "1792056712", "1792056715", "1792056719")
      for name in ("config.json", "summary.json", "evaluation.json")],
    "artifacts/drawer/stage4_execution_after_push_fix.json",
    "artifacts/drawer/final_targeted_vs_direct.json",
    "artifacts/drawer/direct/hole_spec_direct.json",
    "artifacts/drawer/direct/plan_direct.json",
    "artifacts/drawer/direct/checkpoints.json",
    "artifacts/drawer/direct/comparison.json",
    *[f"artifacts/drawer/direct/{run}/{name}" for run in ("seed0_1793033388", "seed1_1793033389",
                                                          "seed2_1793033392", "seed3_1793033394")
      for name in ("config.json", "summary.json")],
    "artifacts/drawer/paper/sources.json",
    # The latest fresh reproduction with the current planner (fe374f0), the closest baseline of the code
    "artifacts/final_verification_fe374f0/stow_stage1_operator.json",
    "artifacts/final_verification_fe374f0/drawer_stage1_operator.json",
    "artifacts/final_verification_fe374f0/stow_stage2_plan.json",
    "artifacts/final_verification_fe374f0/drawer_stage2_plan.json",
    "artifacts/final_verification_fe374f0/stow_hole_spec.json",
    "artifacts/final_verification_fe374f0/drawer_hole_spec.json",
    "artifacts/final_verification_fe374f0/direct/hole_spec_direct.json",
    "artifacts/final_verification_fe374f0/stow_learned_operator.json",
    "artifacts/final_verification_fe374f0/manifest.json",
]

# Where each script fixes its seeds and budgets: the line is recorded as it stands.
PROTOCOL_LINES = {
    "scripts/stow_stage3_train.py": r"EVAL_SEEDS =|--timesteps|--eval-every|--curriculum-steps|seed_pool=",
    "scripts/drawer_stage3_train.py": r"TRAIN_SEEDS =|VALIDATION_SEEDS =|TEST_SEEDS =|--timesteps|--eval-every",
    "scripts/drawer_direct_train.py": r"SEEDS|--timesteps|--eval-every|episode|max_steps",
    "scripts/stow_stage4_execute.py": r"EXTRACTION_SEEDS =|default=list\(range",
    "scripts/drawer_stage4_execute.py": r"SEEDS|default=list\(range|range\(",
    "scripts/drawer_final_evaluate.py": r"SEEDS|default=list\(range|range\(|_STEPS =",
    "scripts/stow_stage1_extract.py": r"SEEDS|range\(",
    "scripts/drawer_stage1_extract.py": r"SEEDS|range\(",
    "scripts/stow_stage2_plan.py": r"SEEDS|range\(",
    "scripts/drawer_stage2_plan.py": r"SEEDS|range\(",
}


def run(*args: str) -> str:
    return subprocess.run(list(args), capture_output=True, text=True).stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def environment() -> dict:
    import torch
    versions = {}
    for name in ("numpy", "torch", "mani-skill", "sapien", "stable-baselines3", "gymnasium", "pytest"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": sys.version.split()[0], "executable": sys.executable, "platform": platform.platform(),
            "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "driver": run("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            "packages": versions}


def tests() -> dict:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                            capture_output=True, text=True)
    lines = result.stdout.strip().splitlines()
    summary = lines[-1] if lines else ""
    counts = {kind: int(number) for number, kind in re.findall(r"(\d+) (passed|failed|skipped|errors?|warnings?)", summary)}
    return {"command": "python -m pytest -q", "exit_code": result.returncode, "summary": summary, "counts": counts}


def references() -> dict:
    rows = {}
    for name in REFERENCE_FILES:
        path = Path(name)
        if not path.exists():
            rows[name] = {"missing": True}
            continue
        row = {"bytes": path.stat().st_size, "sha256": sha256(path),
               "tracked_in_git": subprocess.run(["git", "ls-files", "--error-unmatch", name],
                                                capture_output=True).returncode == 0}
        if path.stat().st_size <= COPY_LIMIT:
            copy = HERE / "refs" / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, copy)
            row["copy"] = str(copy.relative_to(HERE))
        rows[name] = row
    return rows


def protocol() -> dict:
    found = {}
    for script, pattern in PROTOCOL_LINES.items():
        lines = Path(script).read_text().splitlines()
        found[script] = [f"{i + 1}: {line.strip()}" for i, line in enumerate(lines) if re.search(pattern, line)]
    return found


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; it is never overwritten")
    baseline = {
        "about": "Stage A of the signed-literal phase: the state before any change.",
        "git": {"branch": run("git", "branch", "--show-current"), "commit": run("git", "rev-parse", "HEAD"),
                "status_porcelain": run("git", "status", "--porcelain"),
                "last_commit": run("git", "log", "-1", "--format=%H %ad %s", "--date=iso")},
        "environment": environment(),
        "tests": tests(),
        "references": references(),
        "protocol_lines": protocol(),
    }
    OUT.write_text(json.dumps(baseline, indent=1))
    print(json.dumps({k: baseline[k] for k in ("git", "tests")}, indent=1))
    print(f"references: {sum(1 for r in baseline['references'].values() if not r.get('missing'))} recorded, "
          f"{sum(1 for r in baseline['references'].values() if r.get('missing'))} missing; wrote {OUT}")


if __name__ == "__main__":
    main()
