"""Diagnostic, not the protocol: the direct policies on the final configurations (3000-3049),
each run for its whole 100-step training horizon, as validation runs it, instead of stopping at
the first step where the goal holds. Everything else is the final evaluation's own code: the
same forward push and approach, the same checked checkpoints, the same 10-step settling with
the gripper command unchanged, and the same robust signed goal.

    PYTHONNOUSERSITE=1 python artifacts/signed_literals_20261111/D_diagnostics/direct_full_horizon.py

Writes direct_full_horizon.json next to this file, never overwriting it.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path("artifacts/signed_literals_20261111")
OUT = HERE / "D_diagnostics" / "direct_full_horizon.json"

spec = importlib.util.spec_from_file_location("final_evaluate", "scripts/drawer_final_evaluate.py")
fe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe)
_run_policy = fe.run_policy


def full_horizon(env, obs, policy, observe, done, max_steps, registry):
    """The direct policy (the only one given DIRECT_STEPS) ignores the early stop."""
    if max_steps == fe.DIRECT_STEPS:
        done = lambda scores: False
    return _run_policy(env, obs, policy, observe, done, max_steps, registry)


def evaluate(name, checkpoint, operator):
    fe.run_policy = full_horizon
    return fe.evaluate(name, "direct", checkpoint, fe.SEEDS, operator)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; it is never overwritten")
    operator = str(HERE / "E1_stage1" / "drawer_stage1_operator.json")
    found = fe.checkpoints(HERE / "E3_stage3" / "drawer", HERE / "E3_stage3" / "direct")
    direct = {name: entry for name, entry in found.items() if entry["method"] == "direct"}
    with ProcessPoolExecutor(max_workers=4, mp_context=fe.multiprocessing.get_context("spawn")) as pool:
        results = dict(pool.map(evaluate, direct, [e["path"] for e in direct.values()], [operator] * len(direct)))
    summary = {name: fe.summarise(rows) for name, rows in results.items()}
    for name, s in summary.items():
        print(f"{name:20s} restored {s['success']}/{s['configurations']}, drawer open {s['policy_target_achieved']}, "
              f"failures {s['failures'] or '-'}")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    OUT.write_text(json.dumps({"about": __doc__, "commit": commit, "seeds": list(fe.SEEDS), "checkpoints": direct,
                               "summary": summary, "results": results}, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
