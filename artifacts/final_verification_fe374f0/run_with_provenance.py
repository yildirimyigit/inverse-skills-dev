"""Run one repository script unchanged, after checking which planner it will use, and record
every planning call it makes.

    python artifacts/final_verification_fe374f0/run_with_provenance.py RECORD.json scripts/NAME.py [ARGS...]

Run from the repository root, like the scripts themselves.

Before the script starts, three things are checked. If any fails, the script is not run:
  1. the checkout is feature/exact-hole-planner at fe374f0, with no tracked file changed;
  2. the planner modules this Python process imports are the files committed at fe374f0;
  3. they are the minimal formulation: `plan` defaults to the exact mode, and `plan_exact`
     takes a learning budget and no length limit.

While the script runs, every call to `hole_planner.plan` and to `exact_hole_planner.plan_exact`
is recorded: the line that called it, the mode it asked for, which implementation answered,
and what it returned. Calls are passed straight through and their results returned unchanged.

Processes the script spawns are fresh interpreters and are not recorded. Of the scripts run
here, only the final targeted-versus-direct evaluation spawns any, and it never plans.

RECORD.json is written when the script ends, also when it fails. It is never overwritten.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import runpy
import subprocess
import sys
import time
import traceback
from collections import Counter
from importlib import metadata
from pathlib import Path

BRANCH = "feature/exact-hole-planner"
COMMIT = "fe374f0037b24071f8381b8a93ad92a21f11564d"
MINIMAL_SIGNATURE = ["domain", "start", "goal_positive", "goal_negative", "max_delegated"]
PACKAGES = ("numpy", "torch", "mani-skill", "sapien", "stable-baselines3", "gymnasium")

MINIMAL = "minimal exact planner (exact_hole_planner.plan_exact, fe374f0)"
RELAXED = "relaxed planner (hole_planner, mode='relaxed')"

calls: list[dict] = []


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def checkout() -> dict:
    return {"branch": git("branch", "--show-current"),
            "commit": git("rev-parse", "HEAD"),
            "tracked_changes": git("status", "--porcelain", "--untracked-files=no").splitlines()}


def planner_modules() -> dict:
    """Where the planner modules this process imports come from, and their signatures."""
    from inverse_skills.operators import exact_hole_planner, hole_planner

    files = {}
    for module in (hole_planner, exact_hole_planner):
        path = Path(module.__file__).resolve()
        relative = path.relative_to(Path.cwd().resolve())   # fails if imported from outside this checkout
        files[module.__name__] = {
            "file": str(relative),
            "git_blob_on_disk": git("hash-object", str(path)),
            "git_blob_at_HEAD": git("rev-parse", f"HEAD:{relative}"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return {"files": files,
            "plan_default_mode": inspect.signature(hole_planner.plan).parameters["mode"].default,
            "plan_exact_parameters": list(inspect.signature(exact_hole_planner.plan_exact).parameters)}


def problems_with(found: dict, modules: dict) -> list[str]:
    problems = []
    if found["branch"] != BRANCH:
        problems.append(f"branch is {found['branch']}, not {BRANCH}")
    if found["commit"] != COMMIT:
        problems.append(f"HEAD is {found['commit']}, not {COMMIT}")
    if found["tracked_changes"]:
        problems.append("tracked files differ from HEAD")
    for name, row in modules["files"].items():
        if row["git_blob_on_disk"] != row["git_blob_at_HEAD"]:
            problems.append(f"{name} differs from the file committed at HEAD")
    if modules["plan_default_mode"] != "exact":
        problems.append("hole_planner.plan does not default to the exact mode")
    if modules["plan_exact_parameters"] != MINIMAL_SIGNATURE:
        problems.append("plan_exact does not have the minimal formulation's signature")
    return problems


def environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": sys.version.split()[0], "executable": sys.executable, "machine": platform.node(),
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"), "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "packages": versions}


def caller() -> str:
    """The first line outside this file and the planner modules: the script line that planned."""
    frame = sys._getframe(2)
    while frame is not None:
        filename = frame.f_code.co_filename
        if not filename.endswith(("run_with_provenance.py", "hole_planner.py")):
            return f"{Path(filename).name}:{frame.f_lineno}"
        frame = frame.f_back
    return "unknown"


def describe(arguments: dict, result) -> dict:
    """What a call asked for and what it returned."""
    domain = arguments["domain"]
    row = {"max_delegated": arguments.get("max_delegated"),
           "library_actions": len(domain.actions),
           "interference_facts": len(domain.disturbances),
           "start": sorted(arguments["start"]),
           "result_type": type(result).__name__,
           "found": bool(result)}
    if result:
        row.update(actions=list(result.actions), delegated=list(result.delegated), cost=list(result.cost))
    return row


def install_recorders() -> None:
    """Replace the two planner entry points by pass-through functions that record each call."""
    from inverse_skills.operators import exact_hole_planner, hole_planner

    plan, plan_exact = hole_planner.plan, exact_hole_planner.plan_exact
    plan_signature, exact_signature = inspect.signature(plan), inspect.signature(plan_exact)
    inside_plan = {"now": False}
    exact_entered = {"count": 0}

    def recorded_plan_exact(*args, **kwargs):
        exact_entered["count"] += 1
        if inside_plan["now"]:          # reached through hole_planner.plan, which records it
            return plan_exact(*args, **kwargs)
        arguments = exact_signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        started = time.perf_counter()
        result = plan_exact(*args, **kwargs)
        calls.append({"called_from": caller(), "entry": "exact_hole_planner.plan_exact",
                      "mode_asked": "exact (plan_exact called directly)", "unrestricted_holes": False,
                      "answered_by": MINIMAL, "milliseconds": round(1000 * (time.perf_counter() - started), 2),
                      **describe(arguments.arguments, result)})
        return result

    def recorded_plan(*args, **kwargs):
        arguments = plan_signature.bind(*args, **kwargs)
        arguments.apply_defaults()
        before = exact_entered["count"]
        inside_plan["now"] = True
        started = time.perf_counter()
        try:
            result = plan(*args, **kwargs)
        finally:
            inside_plan["now"] = False
        calls.append({"called_from": caller(), "entry": "hole_planner.plan",
                      "mode_asked": arguments.arguments["mode"],
                      "unrestricted_holes": arguments.arguments["unrestricted_holes"],
                      "answered_by": MINIMAL if exact_entered["count"] > before else RELAXED,
                      "milliseconds": round(1000 * (time.perf_counter() - started), 2),
                      **describe(arguments.arguments, result)})
        return result

    hole_planner.plan = recorded_plan
    exact_hole_planner.plan_exact = recorded_plan_exact


def tally() -> dict:
    by_site = {}
    for call in calls:
        site = by_site.setdefault(call["called_from"], {"calls": 0, "answered_by": Counter()})
        site["calls"] += 1
        site["answered_by"][call["answered_by"]] += 1
    return {"total": len(calls),
            "answered_by": dict(Counter(call["answered_by"] for call in calls)),
            "by_call_site": {site: {"calls": row["calls"], "answered_by": dict(row["answered_by"])}
                             for site, row in sorted(by_site.items())}}


def main() -> None:
    record_path, script, *script_arguments = sys.argv[1:]
    record_path = Path(record_path)
    if record_path.exists():
        raise SystemExit(f"{record_path} exists; records are never overwritten")
    record_path.parent.mkdir(parents=True, exist_ok=True)

    found, modules = checkout(), planner_modules()
    problems = problems_with(found, modules)
    record = {"command": ["python", script, *script_arguments], "checkout": found, "planner_modules": modules,
              "checks_passed": not problems, "problems": problems, "environment": environment(),
              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if problems:
        record["outcome"] = "not run"
        record_path.write_text(json.dumps(record, indent=1))
        raise SystemExit("not run: " + "; ".join(problems))

    install_recorders()
    sys.argv = [script, *script_arguments]
    sys.path[0] = str(Path(script).resolve().parent)     # as `python scripts/NAME.py` would set it
    started = time.perf_counter()
    outcome = "finished"
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as stop:
        outcome = "finished" if stop.code in (None, 0) else f"stopped: {stop.code}"
        raise
    except BaseException:
        outcome = "failed:\n" + traceback.format_exc()
        raise
    finally:
        record.update(outcome=outcome, seconds=round(time.perf_counter() - started, 1),
                      planner_calls=tally(), calls=calls)
        record_path.write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
