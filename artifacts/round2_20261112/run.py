"""Round 2's runs, each step reading its inputs from earlier records (PLAN.md section 5).

    PYTHONNOUSERSITE=1 python artifacts/round2_20261112/run.py R0    # then R1, R2, R3, R4

Run from the repository root, at a commit with no tracked change (the runner refuses otherwise,
so every result belongs to one commit). Each step runs a script as a subprocess: its console
output goes to logs/<step>.log, and records/<step>.json holds the command, the commit, the start
time, the duration and the exit code. A step with a record is not run again; every script also
refuses to overwrite its outputs. Nothing is written outside this folder.

Reused, from artifacts/signed_literals_20261111/: the Stage 1 operators (E1), the Stage 2 plans and
hole specs (E2), and, for the R0 regression only, E3's checkpoints with the E4 and diagnostic
records they produced.

  R0  checks before any training: the test suite; the direct spec (no fences); the direct
      precheck (B'); the Stow handoff equivalence on the validation scenes; the changed final
      evaluation against two earlier records on 3000-3049
  R1  training, all at once: Stow (120,000 steps, validated every 4,000 on 4000-4019, then the
      skill test on 5000-5049), the targeted grasp (SAC seeds 0-3, 30,000 steps, validated every
      2,500 on 500-509) and the direct baseline (the same); then the checkpoint manifests
  R2  Stow Stage 4 on 5000-5049 (only if Stow Stage 3 passed); Drawer Stage 4 on 2000-2019
  R3  the final targeted-versus-direct evaluation on 7000-7049
  R4  figures, tables and the manifest
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path("artifacts/round2_20261112")
LOGS, RECORDS = HERE / "logs", HERE / "records"
R0, R1, R2, R3 = (HERE / name for name in ("R0_checks", "R1_training", "R2_stage4", "R3_final"))
SL = Path("artifacts/signed_literals_20261111")
E1, E2, E3, E4, DIAG = SL / "E1_stage1", SL / "E2_stage2", SL / "E3_stage3", SL / "E4_stage4", SL / "D_diagnostics"
PY = [sys.executable]


def seeds(first: int, last: int) -> str:
    return ",".join(str(s) for s in range(first, last + 1))


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
    print(f"{name}: {' '.join(command)}", flush=True)
    return name, command, time.time(), time.strftime("%Y-%m-%d %H:%M:%S"), subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, env=env), log


def finish(job) -> int:
    name, command, began, began_at, process, log = job
    code = process.wait()
    log.close()
    record = {"step": name, "command": command, "checkout": checkout(), "started": began_at,
              "seconds": round(time.time() - began, 1), "exit_code": code}
    (RECORDS / f"{name}.json").write_text(json.dumps(record, indent=1))
    print(f"{name}: exit {code} in {record['seconds']:.0f} s", flush=True)
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


def gate(ok: bool, what: str) -> None:
    print(f"gate: {what}: {'passed' if ok else 'FAILED'}", flush=True)
    if not ok:
        raise SystemExit(f"gate failed: {what}")


def main() -> None:
    stage = sys.argv[1]
    found = checkout()
    if found["tracked_changes"]:
        raise SystemExit("tracked files differ from HEAD; commit first, so the results belong to one commit")
    print(f"{found['branch']} at {found['commit'][:7]}, stage {stage}", flush=True)

    if stage == "R0":
        run([("tests", PY + ["-m", "pytest", "-q"])])
        R0.mkdir(parents=True, exist_ok=True)
        tests = R0 / f"tests_at_{found['commit'][:7]}.txt"
        if not tests.exists():
            tail = (LOGS / "tests.log").read_text().strip().splitlines()[-1]
            tests.write_text(f"commit {found['commit']}, {time.strftime('%Y-%m-%dT%H:%M:%S')}: "
                             f"PYTHONNOUSERSITE=1 python -m pytest -q\n{tail}\n")
        run([("direct_spec", PY + ["scripts/drawer_direct_spec.py", "--operator", str(E1 / "drawer_stage1_operator.json"),
                                   "--grasp-spec", str(E2 / "drawer_hole_spec.json"), "--out-dir", str(R0 / "direct_spec")])])
        spec = json.loads((R0 / "direct_spec" / "hole_spec_direct.json").read_text())
        plan = json.loads((R0 / "direct_spec" / "plan_direct.json").read_text())
        gate(plan["passed"] and not spec["preserve"] and not spec["preserve_absent"],
             "the direct spec has no fences and the grasp hole's start")
        run([("direct_precheck", PY + ["scripts/drawer_direct_precheck.py", "--operator", str(E1 / "drawer_stage1_operator.json"),
                                       "--grasp-spec", str(E2 / "drawer_hole_spec.json"),
                                       "--spec", str(R0 / "direct_spec" / "hole_spec_direct.json"),
                                       "--out", str(R0 / "direct_precheck.json")]),
             ("handoff_check", PY + [str(HERE / "handoff_check.py"), "--spec", str(E2 / "stow_hole_spec.json"),
                                     "--out", str(R0 / "handoff_check.json")]),
             ("final_eval_regression", PY + ["scripts/drawer_final_evaluate.py", "--seeds", seeds(3000, 3049),
                                             "--operator", str(E1 / "drawer_stage1_operator.json"),
                                             "--stage3", str(E3 / "drawer"), "--direct", str(E3 / "direct"),
                                             "--workers", "4", "--out", str(R0 / "final_eval_3000_regression.json")])],
            at_once=3)
        gate(json.loads((R0 / "direct_precheck.json").read_text())["passed"], "the direct precheck")
        run([("regression_check", PY + [str(HERE / "regression_check.py"),
                                        "--new", str(R0 / "final_eval_3000_regression.json"),
                                        "--targeted-record", str(E4 / "final_targeted_vs_direct.json"),
                                        "--direct-record", str(DIAG / "direct_full_horizon.json"),
                                        "--out", str(R0 / "regression_check.json")])])
    elif stage == "R1":
        stow = ("stow_train", PY + ["scripts/stow_stage3_train.py", "--timesteps", "120000", "--eval-every", "4000",
                                    "--curriculum-steps", "60000", "--seed", "0",
                                    "--spec", str(E2 / "stow_hole_spec.json"), "--out-dir", str(R1 / "stow")])
        targeted = [(f"targeted_train_seed{k}", PY + ["scripts/drawer_stage3_train.py", "--timesteps", "30000",
                                                      "--eval-every", "2500", "--seed", str(k),
                                                      "--spec", str(E2 / "drawer_hole_spec.json"),
                                                      "--operator", str(E1 / "drawer_stage1_operator.json"),
                                                      "--out-dir", str(R1 / "drawer")]) for k in range(4)]
        sac_from = seed_run(E3 / "drawer", 0) / "config.json"      # the Drawer runs' SAC settings, budget, interval
        direct = [(f"direct_train_seed{k}", PY + ["scripts/drawer_direct_train.py", "--seed", str(k),
                                                  "--stage3-config", str(sac_from),
                                                  "--operator", str(E1 / "drawer_stage1_operator.json"),
                                                  "--spec", str(R0 / "direct_spec" / "hole_spec_direct.json"),
                                                  "--out-dir", str(R1 / "direct")]) for k in range(4)]
        run([stow, *targeted, *direct], at_once=9)
        run([(f"{name}_checkpoints", PY + ["scripts/stage3_checkpoints.py", "--runs", str(R1 / name), "--selected", selected,
                                           "--out", str(R1 / name / "checkpoints.json")])
             for name, selected in (("stow", "sac_hole_best.zip"), ("drawer", "sac_hole_best.zip"),
                                    ("direct", "sac_direct_best.zip"))], at_once=3)
        stow_summary = json.loads((only_folder(R1 / "stow") / "summary.json").read_text())
        print(f"Stow Stage 3: {'PASS' if stow_summary['passed'] else 'FAIL'} (selected step "
              f"{stow_summary['selected']['timesteps']}, C_exec {stow_summary['selected']['contract_rate_at_stop']:.0%})")
    elif stage == "R2":
        stow_run = only_folder(R1 / "stow")
        steps = [("drawer_stage4", PY + ["scripts/drawer_stage4_execute.py", "--operator", str(E1 / "drawer_stage1_operator.json"),
                                         "--spec", str(E2 / "drawer_hole_spec.json"), "--stage3", str(R1 / "drawer"),
                                         "--out", str(R2 / "drawer_stage4.json")])]
        if json.loads((stow_run / "summary.json").read_text())["passed"]:
            steps.append(("stow_stage4", PY + ["scripts/stow_stage4_execute.py", str(stow_run / "sac_hole_best"),
                                               "--seeds", seeds(5000, 5049),
                                               "--operator", str(E1 / "stow_stage1_operator.json"),
                                               "--spec", str(E2 / "stow_hole_spec.json"),
                                               "--plan", str(E2 / "stow_stage2_plan.json"),
                                               "--out", str(R2 / "stow_stage4.json"),
                                               "--learned-out", str(R2 / "stow_learned_operator.json")]))
        else:
            print("Stow Stage 3 failed its gate: Stow Stage 4 is not run (PLAN.md section 5)")
        R2.mkdir(parents=True, exist_ok=True)
        run(steps, at_once=2)
    elif stage == "R3":
        R3.mkdir(parents=True, exist_ok=True)
        run([("final_targeted_vs_direct", PY + ["scripts/drawer_final_evaluate.py", "--seeds", seeds(7000, 7049),
                                                "--operator", str(E1 / "drawer_stage1_operator.json"),
                                                "--stage3", str(R1 / "drawer"), "--direct", str(R1 / "direct"),
                                                "--workers", "4", "--out", str(R3 / "final_targeted_vs_direct.json")])])
    elif stage == "R4":
        run([("paper_results", PY + [str(HERE / "paper_results.py"), "--out", str(HERE / "paper")]),
             ("manifest", PY + [str(HERE / "manifest.py")])])
    else:
        raise SystemExit("usage: run.py R0|R1|R2|R3|R4")


if __name__ == "__main__":
    main()
