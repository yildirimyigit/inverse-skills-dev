"""Write the checkpoint manifest for a folder of SAC runs: every run's seed and checkpoint hashes.

    python scripts/stage3_checkpoints.py --runs <folder> --selected sac_hole_best.zip --out <folder>/checkpoints.json

Each run folder holds a config.json (with its SAC seed) and the .zip checkpoints the training script
saved. The manifest has the format Stage 4 and the final evaluation read (artifacts/drawer/stage3/
checkpoints.json): for each run its SAC seed, the checkpoint it selects, and every checkpoint's size
and SHA-256. Nothing is copied or changed; the output is never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib import metadata
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--selected", required=True, help="the checkpoint Stage 4 uses, e.g. sac_hole_best.zip")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; it is never overwritten")

    runs = {}
    for config_path in sorted(args.runs.glob("*/config.json")):
        folder = config_path.parent
        if not (folder / args.selected).exists():
            raise SystemExit(f"{folder} has no {args.selected}")
        runs[folder.name] = {
            "sac_seed": json.loads(config_path.read_text())["seed"],
            "selected": args.selected,
            "files": {zip_path.name: {"bytes": zip_path.stat().st_size, "sha256": sha256(zip_path)}
                      for zip_path in sorted(folder.glob("*.zip"))},
        }
    seeds = [run["sac_seed"] for run in runs.values()]
    if len(seeds) != len(set(seeds)):
        raise SystemExit(f"two runs share a SAC seed: {sorted(seeds)}")
    manifest = {
        "note": "Each checkpoint is identified by its SHA-256 (check a copy with `sha256sum`). "
                f"Stage 4 and the final evaluation use each run's {args.selected}.",
        "load": f"stable_baselines3.SAC.load('<run folder>/{args.selected[:-4]}', device='cpu')",
        "versions": {name: metadata.version(name) for name in ("torch", "stable-baselines3", "mani-skill", "sapien")},
        "runs": runs,
    }
    args.out.write_text(json.dumps(manifest, indent=1))
    print(f"{len(runs)} runs, SAC seeds {sorted(seeds)}; wrote {args.out}")


if __name__ == "__main__":
    main()
