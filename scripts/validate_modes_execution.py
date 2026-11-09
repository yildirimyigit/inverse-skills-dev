"""Default-mode validation, Phase 3: the Stage 4 pipelines with the original policies, in named modes.

`run` executes scripts/stow_stage4_execute.py or scripts/drawer_stage4_execute.py unchanged, on their
original seeds and checkpoints, after replacing two things in memory for this process only:

  hole_planner.plan                every call names the planning mode (a call with
                                   unrestricted_holes keeps relaxed, the only mode that has it)
  extractor.OperatorExtractor      every extractor names the extraction mode

So nothing here depends on a default. Each run also logs every planner call and, at every learned
step, the observation the policy first receives and a hash of the whole simulator state.

  --operator-from       which Phase 1 forward operator gives the goal (mean or min_max)
  --learned-extraction  how Stow Stage 4 models the learned skill (mean or min_max)
  --planner             relaxed or exact

`compare` reads the runs and the saved Stage 4 results and writes every difference, leaf by leaf,
with one row per seed. `stage3-input` checks what Stage 3 would receive from the proposed spec: the
environment built from it, and that its interface fits the checkpoint. Nothing is trained.

Run from the repository root, after Phases 1 and 2, with PYTHONNOUSERSITE=1:
  PYTHONNOUSERSITE=1 python scripts/validate_modes_execution.py run stow --operator-from mean --learned-extraction mean --planner relaxed
  PYTHONNOUSERSITE=1 python scripts/validate_modes_execution.py compare

Writes under artifacts/default_mode_validation/phase3/ (or --out-dir), never overwriting.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

from inverse_skills.operators import extractor as extractor_module
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.extractor import OperatorExtractionConfig

PHASE1 = Path("artifacts/default_mode_validation/phase1")
PHASE2 = Path("artifacts/default_mode_validation/phase2")
OUT_DIR = Path("artifacts/default_mode_validation/phase3")
STOW_CHECKPOINT = "artifacts/stow/stage3/1790788534/sac_hole_best"
SAVED = {"stow": Path("artifacts/stow/stage4_execution.json"),
         "drawer": Path("artifacts/drawer/stage4_execution_after_push_fix.json")}
OPERATOR_PART = {"stow": "stow", "drawer": "drawer_original"}


def run_tag(args) -> str:
    tag = f"{args.operator_from}_{args.planner}"
    if args.domain == "stow" and args.learned_extraction != args.operator_from:
        tag += f"_learned_{args.learned_extraction}"
    return tag


def state_hash(env) -> str:
    hasher = hashlib.sha256()

    def walk(value, path: str = "") -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                walk(value[key], f"{path}/{key}")
        else:
            data = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
            hasher.update(path.encode())
            hasher.update(data.tobytes())

    walk(env.unwrapped.get_state_dict())
    return hasher.hexdigest()


def force_modes(args, log: dict) -> None:
    """Every planner call and every extractor in this process names its mode."""
    original_plan = hp.plan

    def plan(domain, start, goal_positive, goal_negative=None, **kwargs):
        used = "relaxed" if kwargs.get("unrestricted_holes") else args.planner
        result = original_plan(domain, start, goal_positive, goal_negative, mode=used, **kwargs)
        log["plan_calls"].append({"mode": used, "limits": kwargs, "start": sorted(start),
                                  "result": type(result).__name__,
                                  "actions": list(result.actions) if result else None})
        return result

    hp.plan = plan
    original_extractor = extractor_module.OperatorExtractor
    extraction = args.learned_extraction or args.operator_from

    class NamedModeExtractor(original_extractor):
        def __init__(self, registry, config=None):
            super().__init__(registry, OperatorExtractionConfig(mode=extraction))

    extractor_module.OperatorExtractor = NamedModeExtractor


def log_handoffs(domain: str, log: dict) -> None:
    def note(env, observation, label: str) -> None:
        log["handoffs"].append({"label": label, "observation": [float(x) for x in observation],
                                "sim_state_sha256": state_hash(env)})

    if domain == "stow":
        from inverse_skills.envs import stow_executive as executive
        from inverse_skills.envs import stow_predicates as sp
        from inverse_skills.envs import stow_skill_io as skill_io
        original = executive.run_learned

        def run_learned(env, obs, model, achieve, regions, registry=None, *rest, **kwargs):
            predicate = (registry or sp.registry()).get(achieve)
            note(env, skill_io.observe(obs, regions, predicate, hp.predicate_args(achieve)[0]), achieve)
            return original(env, obs, model, achieve, regions, registry, *rest, **kwargs)
    else:
        from inverse_skills.envs import drawer_executive as executive
        from inverse_skills.envs import drawer_skill_io as skill_io
        original = executive.run_learned

        def run_learned(env, obs, policy, achieve, *rest, **kwargs):
            note(env, skill_io.observe(obs, achieve), achieve.key)
            return original(env, obs, policy, achieve, *rest, **kwargs)

    executive.run_learned = run_learned


def run(args) -> None:
    tag = run_tag(args)
    out_dir = args.out_dir
    out = out_dir / f"{args.domain}_{tag}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; outputs are never overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)
    operator = PHASE1 / f"{OPERATOR_PART[args.domain]}_{args.operator_from}_operator.json"
    spec = PHASE2 / f"{args.domain}_{args.operator_from}_{args.planner}_hole_spec.json"
    log = {"domain": args.domain, "tag": tag, "operator_from": args.operator_from,
           "learned_extraction": args.learned_extraction, "planner": args.planner,
           "operator": str(operator), "spec": str(spec), "plan_calls": [], "handoffs": []}
    force_modes(args, log)
    log_handoffs(args.domain, log)
    if args.domain == "stow":
        script = "scripts/stow_stage4_execute.py"
        argv = [script, STOW_CHECKPOINT, "--spec", str(spec), "--operator", str(operator), "--out", str(out),
                "--learned-out", str(out_dir / f"stow_{tag}_learned_operator.json")]
    else:
        script = "scripts/drawer_stage4_execute.py"
        argv = [script, "--spec", str(spec), "--operator", str(operator), "--out", str(out)]
    log["command"] = argv
    sys.argv = argv
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        (out_dir / f"{args.domain}_{tag}_log.json").write_text(json.dumps(log, indent=1))


# ---------------------------------------- compare ----------------------------------------


def flatten(value, prefix: str = "") -> dict:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return out
    if isinstance(value, (list, tuple)):
        out = {f"{prefix}.#len": len(value)}
        for i, item in enumerate(value):
            out.update(flatten(item, f"{prefix}[{i}]"))
        return out
    return {prefix: value}


def differences(a, b) -> dict:
    """Every leaf that differs. Two NaNs count as equal: they are the same missing measurement."""
    left, right = flatten(a), flatten(b)
    found = {}
    for key in sorted(set(left) | set(right)):
        x, y = left.get(key, "<absent>"), right.get(key, "<absent>")
        if x != y and not (isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y)):
            found[key] = [x, y]
    return found


def stow_row(result: dict) -> dict:
    failed = next((s for s in result["steps"] if not s["ok"]), None)
    return {"seed": result["seed"], "goal_met": bool(result["goal_met"]), "a_err_mm": result["cube_a_err_mm"],
            "b_err_mm": result["cube_b_err_mm"], "rerolls": result["forward_rerolls"],
            "retries": sum(max(s["attempts"] - 1, 0) for s in result["steps"]),
            "steps_run_ok": sum(s["ok"] for s in result["steps"]), "steps": len(result["steps"]),
            "failed_at": f"{failed['action']}: {failed['detail']}" if failed else None,
            "diverging_steps": sum(bool(s["divergence"]) for s in result["steps"])}


def drawer_row(result: dict) -> dict:
    failed = next((s for s in result["steps"] if not s["ok"]), None)
    return {"seed": result["seed"], "goal_met": bool(result["goal_met"]), "restored": bool(result["ok"]),
            "final_opening_mm": result["final_opening_mm"],
            "retries": sum(max(s["attempts"] - 1, 0) for s in result["steps"]),
            "steps_run_ok": sum(s["ok"] for s in result["steps"]), "steps": len(result["steps"]),
            "inverse_control_steps": result["inverse_control_steps"],
            "failed_at": f"{failed['action']}: {failed['detail']}" if failed else result.get("stopped"),
            "diverging_steps": sum(bool(s["divergence"]) for s in result["steps"])}


def tables(data: dict, row) -> dict:
    return {condition: [row(result) for result in rows] for condition, rows in data["results"].items()}


def imports_of(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names |= {f"{node.module}.{alias.name}" for alias in node.names}
    return names


def final_evaluation_dependencies() -> dict:
    """Does the reserved 3000-3049 evaluation use the planner or the extractor, and its goal?"""
    names = imports_of(Path("scripts/drawer_final_evaluate.py"))
    from inverse_skills.envs import drawer_domain as dd

    stored = json.loads(Path("artifacts/drawer/final_targeted_vs_direct.json").read_text())
    goal = {mode: [sorted(part) for part in dd.goal_from_operator(PHASE1 / f"drawer_original_{mode}_operator.json")]
            for mode in ("mean", "min_max")}
    return {"imports_the_planner": any("hole_planner" in n for n in names),
            "imports_the_extractor": any("extractor" in n for n in names),
            "reads_the_goal_from": "artifacts/drawer/stage1_operator.json, through dd.goal_from_operator",
            "goal_in_its_saved_results": stored["goal"], "goal_by_mode": goal,
            "goal_unchanged_by_extraction": goal["mean"] == goal["min_max"] == stored["goal"]}


def compare_runs(out_dir: Path) -> None:
    output = {"final_evaluation": final_evaluation_dependencies(), "domains": {}}
    for domain, row in (("stow", stow_row), ("drawer", drawer_row)):
        saved = json.loads(SAVED[domain].read_text())
        runs = {}
        for path in sorted(out_dir.glob(f"{domain}_*.json")):
            name = path.stem[len(domain) + 1:]
            if name == "stage3_input" or name.endswith(("_log", "_learned_operator")):
                continue
            runs[name] = json.loads(path.read_text())
            # Reported on its own: the saved files predate it, so it is not a difference from them.
            runs[name].pop("handoff_validity", None)
        if not runs:
            continue
        baseline = runs["mean_relaxed"]
        record = {"saved_file": str(SAVED[domain]), "runs": {}, "per_seed": {"saved": tables(saved, row)}}
        for name, data in runs.items():
            log = json.loads((out_dir / f"{domain}_{name}_log.json").read_text())
            baseline_log = json.loads((out_dir / f"{domain}_mean_relaxed_log.json").read_text())
            against_saved = differences(saved, data)
            against_baseline = differences(baseline, data)
            record["runs"][name] = {
                "summary": data["summary"],
                "plans": data.get("plans"),
                "passed": data.get("passed"),
                "differing_leaves_from_saved": len(against_saved),
                "differing_leaves_from_mean_relaxed": len(against_baseline),
                "differences_from_saved": against_saved,
                "differences_from_mean_relaxed": against_baseline,
                "planner_modes_used": sorted({call["mode"] for call in log["plan_calls"]}),
                "plan_call_results": sorted({call["result"] for call in log["plan_calls"]}),
                "handoffs": len(log["handoffs"]),
                "handoffs_identical_to_mean_relaxed": log["handoffs"] == baseline_log["handoffs"],
                "shared_handoffs_identical_to_mean_relaxed":
                    log["handoffs"] == baseline_log["handoffs"][:len(log["handoffs"])],
            }
            record["per_seed"][name] = tables(data, row)
        output["domains"][domain] = record

    target = out_dir / "execution_comparison.json"
    if target.exists():
        raise SystemExit(f"{target} exists; outputs are never overwritten")
    target.write_text(json.dumps(output, indent=2))

    print("final evaluation:", {k: v for k, v in output["final_evaluation"].items() if k != "goal_by_mode"})
    for domain, record in output["domains"].items():
        print(f"\n=== {domain} (saved: {record['saved_file']})")
        for name, r in record["runs"].items():
            print(f"\n  {name}: {r['differing_leaves_from_saved']} leaves differ from saved, "
                  f"{r['differing_leaves_from_mean_relaxed']} from mean_relaxed; planner modes {r['planner_modes_used']}, "
                  f"results {r['plan_call_results']}; {r['handoffs']} handoffs, identical to mean_relaxed: "
                  f"{r['handoffs_identical_to_mean_relaxed']}")
            fields = sorted({key.split(".")[-1].split("[")[0] for key in r["differences_from_mean_relaxed"]})
            print(f"    fields differing from mean_relaxed: {fields}")
            for key, values in list(r["differences_from_mean_relaxed"].items())[:12]:
                print(f"      {key[:80]}: {str(values[0])[:60]} -> {str(values[1])[:60]}")
            print(f"    summary: {json.dumps(r['summary'])[:900]}")
            if r["passed"] is not None:
                print(f"    STAGE 4 passed: {r['passed']}; plans: {json.dumps(r['plans'])[:520]}")
    print(f"\nwrote {target}")


# ---------------------------------------- stage 3 input ----------------------------------------


def stage3_input(args) -> None:
    """What Stage 3 would be handed by the proposed spec. No training, no weights touched."""
    from stable_baselines3 import SAC

    if args.domain == "stow":
        from inverse_skills.envs.stow_hole_env import StowHoleEnv as HoleEnv
        from inverse_skills.envs.stow_hole_env import load_spec
        saved_spec, checkpoints = Path("artifacts/stow/hole_spec.json"), [STOW_CHECKPOINT]
        seeds = (1000, 1002)
        options = {"seed_pool": seeds, "curriculum": 1.0}
    else:
        from inverse_skills.envs.drawer_hole_env import DrawerHoleEnv as HoleEnv
        from inverse_skills.envs.drawer_hole_env import load_spec
        saved_spec = Path("artifacts/drawer/hole_spec.json")
        manifest = json.loads(Path("artifacts/drawer/stage3/checkpoints.json").read_text())
        checkpoints = [f"artifacts/drawer/stage3/{run}/sac_hole_best" for run in sorted(manifest["runs"])]
        seeds = (1000, 1001)
        options = {}

    target = args.out_dir / f"{args.domain}_stage3_input.json"
    if target.exists():
        raise SystemExit(f"{target} exists; outputs are never overwritten")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    proposed_path = args.spec or PHASE2 / f"{args.domain}_min_max_exact_hole_spec.json"
    proposed = load_spec(proposed_path)
    env = HoleEnv(proposed, **options)
    fits = []
    for checkpoint in checkpoints:
        model = SAC.load(checkpoint, device="cpu")
        fits.append({"checkpoint": checkpoint,
                     "observation_space_fits": model.observation_space.shape == env.observation_space.shape,
                     "action_space_fits": model.action_space.shape == env.action_space.shape,
                     "observation_shape": list(model.observation_space.shape),
                     "action_shape": list(model.action_space.shape)})
    resets = []
    for seed in seeds:
        observation, info = env.reset(seed=seed)
        resets.append({"seed": seed, "handoff_valid": info["handoff_valid"], "observation_shape": list(observation.shape),
                       "start_literals_scores": {p: info["scores"][p] for p in proposed["start"]}})
    env.close()
    record = {"spec": str(proposed_path), "bytes_equal_saved_spec": proposed_path.read_text() == saved_spec.read_text(),
              "learning_fields": {f: proposed[f] for f in ("achieve", "object", "start", "preserve", "preserve_absent")},
              "prefix": [a["name"] for a in proposed["prefix"]], "suffix": [a["name"] for a in proposed["suffix"]],
              "checkpoints": fits, "resets": resets}
    target.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    commands = ap.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("domain", choices=("stow", "drawer"))
    run_parser.add_argument("--operator-from", choices=("mean", "min_max"), required=True)
    run_parser.add_argument("--learned-extraction", choices=("mean", "min_max"))
    run_parser.add_argument("--planner", choices=("relaxed", "exact"), required=True)
    run_parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    input_parser = commands.add_parser("stage3-input")
    input_parser.add_argument("domain", choices=("stow", "drawer"))
    input_parser.add_argument("--spec", type=Path)
    input_parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    if args.command == "run":
        if args.domain == "stow" and not args.learned_extraction:
            raise SystemExit("stow needs --learned-extraction")
        run(args)
    elif args.command == "compare":
        compare_runs(args.out_dir)
    else:
        stage3_input(args)


if __name__ == "__main__":
    main()
