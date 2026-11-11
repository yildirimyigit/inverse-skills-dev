"""The signed-literal experiments, every stage fresh, each stage reading the previous stage's outputs.

    PYTHONNOUSERSITE=1 python artifacts/signed_literals_20261111/run.py E0    # then E1, E2, E3, E4

Run from the repository root, at a commit with no tracked change (the runner refuses otherwise, so
every result belongs to one commit). Each step runs a repository script unchanged, as a subprocess:
its console output goes to logs/<step>.log, and records/<step>.json holds the command, the commit,
the start time, the duration and the exit code. A step with a record is not run again; every script
also refuses to overwrite its outputs. Nothing is written outside this folder.

  E0  Stage 0 gates for both tasks; the retreat gate, rerun at this commit
  E1  Stage 1: fresh demonstrations, signed extraction
  E2  Stage 2: planning from fresh measured states; the direct baseline's spec; the planner demo
  E3  Stage 3: SAC from scratch. Stow drag skill: 120,000 steps, evaluated every 4,000 (both from
      the reference run 1790788534's record), curriculum 60,000 steps and seed 0 (the script's
      defaults; the reference run did not record them). Drawer: seeds 0-3, 30,000 steps,
      evaluated every 2,500. Direct baseline: its precheck, then seeds 0-3 with the Drawer seed-0
      run's SAC settings. Checkpoint manifests for both.
  E4  Stage 4 for both tasks, and the targeted-versus-direct evaluation on seeds 3000-3049
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path("artifacts/signed_literals_20261111")
LOGS, RECORDS = HERE / "logs", HERE / "records"
E0, E1, E2, E3, E4 = (HERE / name for name in ("E0_stage0", "E1_stage1", "E2_stage2", "E3_stage3", "E4_stage4"))
PY = [sys.executable]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def checkout() -> dict:
    return {"branch": git("branch", "--show-current"), "commit": git("rev-parse", "HEAD"),
            "tracked_changes": git("status", "--porcelain", "--untracked-files=no").splitlines()}


def start(name: str, command: list[str]):
    """Start one step in the background; None if it has run already."""
    if (RECORDS / f"{name}.json").exists():
        print(f"{name}: already run, record kept, not run again")
        return None
    LOGS.mkdir(parents=True, exist_ok=True)
    RECORDS.mkdir(parents=True, exist_ok=True)
    log = open(LOGS / f"{name}.log", "w")
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    print(f"{name}: {' '.join(command)}")
    return name, command, time.time(), time.strftime("%Y-%m-%d %H:%M:%S"), subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env=env), log


def finish(job) -> int:
    name, command, began, began_at, process, log = job
    code = process.wait()
    log.close()
    record = {"step": name, "command": command, "checkout": checkout(), "started": began_at,
              "seconds": round(time.time() - began, 1), "exit_code": code}
    (RECORDS / f"{name}.json").write_text(json.dumps(record, indent=1))
    print(f"{name}: exit {code} in {record['seconds']:.0f} s")
    return code


def run(steps: list[tuple[str, list[str]]], at_once: int = 1) -> None:
    """Run the steps, `at_once` at a time; stop after a group if any step failed."""
    for i in range(0, len(steps), at_once):
        jobs = [job for job in (start(name, command) for name, command in steps[i:i + at_once]) if job]
        codes = [finish(job) for job in jobs]
        if any(codes):
            raise SystemExit("a step failed; see its log")


def only_folder(parent: Path) -> Path:
    folders = [p for p in parent.iterdir() if p.is_dir()]
    if len(folders) != 1:
        raise SystemExit(f"expected one run in {parent}, found {len(folders)}")
    return folders[0]


def seed_run(parent: Path, seed: int) -> Path:
    for config in parent.glob("*/config.json"):
        if json.loads(config.read_text())["seed"] == seed:
            return config.parent
    raise SystemExit(f"no run with seed {seed} in {parent}")


def main() -> None:
    stage = sys.argv[1]
    found = checkout()
    if found["tracked_changes"]:
        raise SystemExit("tracked files differ from HEAD; commit first, so the results belong to one commit")
    print(f"{found['branch']} at {found['commit'][:7]}, stage {stage}")

    if stage == "E0":
        run([("stow_stage0_gate", PY + ["scripts/stow_stage0_gate.py", "--out", str(E0 / "stow_stage0_gate.json")]),
             ("drawer_stage0_gate", PY + ["scripts/drawer_stage0_gate.py", "--out", str(E0 / "drawer_stage0_gate.json")]),
             ("retreat_gate", PY + ["scripts/stow_retreat_gate.py", "--out", str(E0 / "retreat_gate.json")])], at_once=3)
    elif stage == "E1":
        run([("stow_stage1", PY + ["scripts/stow_stage1_extract.py", "--out", str(E1 / "stow_stage1_operator.json")]),
             ("drawer_stage1", PY + ["scripts/drawer_stage1_extract.py", "--out", str(E1 / "drawer_stage1_operator.json")])],
            at_once=2)
    elif stage == "E2":
        run([("stow_stage2", PY + ["scripts/stow_stage2_plan.py", "--operator", str(E1 / "stow_stage1_operator.json"),
                                   "--out", str(E2 / "stow_stage2_plan.json"), "--spec-out", str(E2 / "stow_hole_spec.json")]),
             ("drawer_stage2", PY + ["scripts/drawer_stage2_plan.py", "--operator", str(E1 / "drawer_stage1_operator.json"),
                                     "--out", str(E2 / "drawer_stage2_plan.json"),
                                     "--spec-out", str(E2 / "drawer_hole_spec.json")])], at_once=2)
        run([("direct_spec", PY + ["scripts/drawer_direct_spec.py", "--operator", str(E1 / "drawer_stage1_operator.json"),
                                   "--grasp-spec", str(E2 / "drawer_hole_spec.json"), "--out-dir", str(E2 / "direct")]),
             ("stow_planner_demo", PY + ["scripts/stow_planner_demo.py", "--operator", str(E1 / "stow_stage1_operator.json"),
                                         "--out", str(E2 / "stow_planner_demo.json")])], at_once=2)
    elif stage == "E3":
        drawer = [(f"drawer_train_seed{k}", PY + ["scripts/drawer_stage3_train.py", "--timesteps", "30000",
                                                  "--eval-every", "2500", "--seed", str(k),
                                                  "--spec", str(E2 / "drawer_hole_spec.json"),
                                                  "--out-dir", str(E3 / "drawer")]) for k in range(4)]
        stow = ("stow_train", PY + ["scripts/stow_stage3_train.py", "--timesteps", "120000", "--eval-every", "4000",
                                    "--curriculum-steps", "60000", "--seed", "0",
                                    "--spec", str(E2 / "stow_hole_spec.json"), "--out-dir", str(E3 / "stow")])
        run([stow, *drawer], at_once=5)
        run([("drawer_checkpoints", PY + ["scripts/stage3_checkpoints.py", "--runs", str(E3 / "drawer"),
                                          "--selected", "sac_hole_best.zip", "--out", str(E3 / "drawer" / "checkpoints.json")]),
             ("direct_precheck", PY + ["scripts/drawer_direct_precheck.py", "--operator", str(E1 / "drawer_stage1_operator.json"),
                                       "--grasp-spec", str(E2 / "drawer_hole_spec.json"),
                                       "--spec", str(E2 / "direct" / "hole_spec_direct.json"),
                                       "--out", str(E3 / "direct" / "precheck.json")])], at_once=2)
        precheck = json.loads((E3 / "direct" / "precheck.json").read_text())
        if not precheck["passed"]:
            raise SystemExit("the direct baseline's precheck failed; its training is not started")
        sac_from = seed_run(E3 / "drawer", 0) / "config.json"
        run([(f"direct_train_seed{k}", PY + ["scripts/drawer_direct_train.py", "--seed", str(k), "--stage3-config", str(sac_from),
                                             "--operator", str(E1 / "drawer_stage1_operator.json"),
                                             "--spec", str(E2 / "direct" / "hole_spec_direct.json"),
                                             "--out-dir", str(E3 / "direct")]) for k in range(4)], at_once=4)
        run([("direct_checkpoints", PY + ["scripts/stage3_checkpoints.py", "--runs", str(E3 / "direct"),
                                          "--selected", "sac_direct_best.zip", "--out", str(E3 / "direct" / "checkpoints.json")])])
    elif stage == "E4":
        stow_run = only_folder(E3 / "stow")
        run([("stow_stage4", PY + ["scripts/stow_stage4_execute.py", str(stow_run / "sac_hole_best"),
                                   "--operator", str(E1 / "stow_stage1_operator.json"), "--spec", str(E2 / "stow_hole_spec.json"),
                                   "--plan", str(E2 / "stow_stage2_plan.json"), "--out", str(E4 / "stow_stage4.json"),
                                   "--learned-out", str(E4 / "stow_learned_operator.json")]),
             ("drawer_stage4", PY + ["scripts/drawer_stage4_execute.py", "--operator", str(E1 / "drawer_stage1_operator.json"),
                                     "--spec", str(E2 / "drawer_hole_spec.json"), "--stage3", str(E3 / "drawer"),
                                     "--out", str(E4 / "drawer_stage4.json")])], at_once=2)
        run([("final_targeted_vs_direct", PY + ["scripts/drawer_final_evaluate.py", "--operator",
                                                str(E1 / "drawer_stage1_operator.json"), "--stage3", str(E3 / "drawer"),
                                                "--direct", str(E3 / "direct"), "--workers", "4",
                                                "--out", str(E4 / "final_targeted_vs_direct.json")])])
    else:
        raise SystemExit("usage: run.py E0|E1|E2|E3|E4")


if __name__ == "__main__":
    main()
