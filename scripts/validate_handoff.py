"""Handoff validation for learned-operator demonstrations: the evidence, step by step.

  investigate  Why does the known prefix report success at seed 2004, although A has left the
               pocket? Runs the prefix one step at a time and records what the executive checked,
               what it decided, what the plan predicted and what the scene shows.
  collect      Stage 4's own learn_operator(), with the original checkpoint and seeds: ten valid
               demonstrations, then both extractors on exactly those, then replanning with each
               resulting action under both planners, naming every mode.
  drawer       The same conditions and check on the Drawer's plan, with no Drawer-specific code:
               the raw handoff validity on its Stage 4 seeds. No policy runs.
  survey       The raw handoff validity of Stow's training seeds, 0-31, for a larger sample. No
               policy runs. Training itself is untouched: its own check only asks for the start.
  earlier-comparison  scripts/stow_stage1_extractor_comparison.py with its outputs redirected.
  after-migration     Do the plain scripts, with the new defaults and no wrapper, reproduce what
               was validated with the modes named? Reads what they wrote (see the report).

Run from the repository root, with PYTHONNOUSERSITE=1 (stray ~/.local packages break ManiSkill):
  PYTHONNOUSERSITE=1 python scripts/validate_handoff.py investigate

Writes under artifacts/default_mode_validation/, never overwriting.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path

BEFORE_MIGRATION = "c68d830"
OUT_DIR = Path("artifacts/default_mode_validation/handoff")
HANDOFF = OUT_DIR
AFTER = Path("artifacts/default_mode_validation/after_migration")
PHASE2 = Path("artifacts/default_mode_validation/phase2")
PHASE1 = Path("artifacts/default_mode_validation/phase1")
STOW_SPEC = Path("artifacts/stow/hole_spec.json")
DRAWER_SPEC = Path("artifacts/drawer/hole_spec.json")
LEARNED_SAVED = Path("artifacts/stow/learned_operator.json")
STOW_CHECKPOINT = "artifacts/stow/stage3/1790788534/sac_hole_best"
EXPECTED_ACTION = {"pre": ["in_region(cube_a,pocket)", "in_region(cube_b,src_b)", "tcp_near(cube_a)"],
                   "add": ["clear_of_walls(cube_a)"], "delete": ["in_region(cube_a,pocket)"]}
# 2004 and 1002 are the runs where A leaves the pocket; 2003 and 1000 are ordinary controls.
TRACED_SEEDS = (2003, 2004, 1000, 1002)
WATCHED = ("in_region(cube_a,pocket)", "clear_of_walls(cube_a)", "in_region(cube_b,mouth)",
           "in_region(cube_b,src_b)", "tcp_near(cube_a)", "gripper_open()")


def investigate() -> None:
    import stow_stage4_execute as stage4

    from inverse_skills.envs import stow_domain as sd
    from inverse_skills.envs import stow_executive as ex
    from inverse_skills.envs import stow_predicates as sp
    from inverse_skills.envs import stow_primitives as prim
    from inverse_skills.envs.stow_hole_env import load_spec
    from inverse_skills.operators import hole_planner as hp
    from inverse_skills.operators.hole_planner import Action

    target = OUT_DIR / "investigation.json"
    if target.exists():
        raise SystemExit(f"{target} exists; outputs are never overwritten")

    spec = load_spec(STOW_SPEC)
    prefix = [Action.from_dict(a) for a in spec["prefix"]]
    domain = sd.domain()
    registry = sp.registry()
    planned_start = sd.stowed_state()
    env = prim.make_env()

    def measure(obs, regions) -> dict[str, float]:
        return {k: float(r.score) for k, r in registry.evaluate_all(prim.obs_to_scene(obs, regions)).items()}

    traces = []
    for seed in TRACED_SEEDS:
        obs, regions, rerolls = stage4.stowed(env, seed, planned_start)
        scores = measure(obs, regions)
        predicted = planned_start
        rows = [{"step": "after the forward skill", "cube_a_x_mm": float(prim.cube_pos(obs, "cube_a")[0]) * 1000,
                 "scores": {p: scores[p] for p in WATCHED}, "predicted_true": sorted(predicted & domain.scored),
                 "differs_from_prediction": sorted((hp.symbolic_state(scores) ^ predicted) & domain.scored)}]
        for step in prefix:
            # What the executive can compare with the scene before the step: scored literals only.
            checked_before = sorted(p for p in step.pre if p in domain.scored)
            checked_absent = sorted(q for q in step.pre_absent if q in domain.scored)
            obs, report = ex.execute_plan(env, obs, [step], regions=regions)
            scores = measure(obs, regions)
            predicted = domain.apply(predicted, step)
            step_report = report.steps[0]
            rows.append({"step": step.name,
                         "executive_checks_before": {"true": checked_before, "false": checked_absent},
                         "executive_checks_after": step_report.detail,
                         "executive_says_ok": bool(step_report.ok),
                         "attempts": step_report.attempts,
                         "cube_a_x_mm": float(prim.cube_pos(obs, "cube_a")[0]) * 1000,
                         "scores": {p: scores[p] for p in WATCHED},
                         "predicted_true": sorted(predicted & domain.scored),
                         "differs_from_prediction": sorted((hp.symbolic_state(scores) ^ predicted) & domain.scored)})
        traces.append({"seed": seed, "forward_rerolls": rerolls, "steps": rows,
                       "training_env_would_accept": all(scores[p] >= ex.SATISFIED for p in spec["start"])})
    env.close()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"made_by": "PYTHONNOUSERSITE=1 python scripts/validate_handoff.py investigate",
                                  "spec_start": spec["start"], "traces": traces}, indent=2))

    for trace in traces:
        print(f"\nseed {trace['seed']} (forward rerolls {trace['forward_rerolls']}; "
              f"the hole's own start check would accept it: {trace['training_env_would_accept']})")
        print(f"  {'step':26s} {'exec ok':>7} {'A x mm':>7} {'pocket':>7} {'clear':>7} {'tcp_A':>6}  executive measured / differs from the plan")
        for row in trace["steps"]:
            s = row["scores"]
            print(f"  {row['step']:26s} {row.get('executive_says_ok', '-')!s:>7} {row['cube_a_x_mm']:7.1f} "
                  f"{s['in_region(cube_a,pocket)']:7.4f} {s['clear_of_walls(cube_a)']:7.4f} {s['tcp_near(cube_a)']:6.3f}  "
                  f"{row.get('executive_checks_after', '')} / {row['differs_from_prediction'] or 'nothing'}")
    print(f"\nwrote {target}")


def collect() -> None:
    import stow_stage4_execute as stage4
    from stable_baselines3 import SAC
    from validate_modes_extraction import compare_operators, extract_both, per_demonstration

    from inverse_skills.envs import stow_domain as sd
    from inverse_skills.envs import stow_predicates as sp
    from inverse_skills.envs import stow_primitives as prim
    from inverse_skills.envs.stow_hole_env import load_spec
    from inverse_skills.operators import hole_planner as hp
    from inverse_skills.operators.extractor import OperatorExtractionConfig, OperatorExtractor
    from inverse_skills.operators.hole_planner import Action

    target = OUT_DIR / "learned_demonstrations.json"
    if target.exists():
        raise SystemExit(f"{target} exists; outputs are never overwritten")
    captured = []

    class Recording(OperatorExtractor):
        """Stage 4's extractor, with the mode named and the rollouts kept for the other mode."""

        def __init__(self, registry):
            super().__init__(registry, OperatorExtractionConfig(mode="mean"))

        def extract(self, skill_name, rollouts):
            captured.extend(rollouts)
            return super().extract(skill_name, rollouts)

    stage4.OperatorExtractor = Recording
    model = SAC.load(STOW_CHECKPOINT, device="cuda")
    spec = load_spec(STOW_SPEC)
    planned_start = sd.stowed_state()
    env = prim.make_env()
    _, stage4_record = stage4.learn_operator(env, model, spec, stage4.EXTRACTION_SEEDS, planned_start, sd.domain())
    env.close()

    registry = sp.registry()
    name = f"learned:{spec['achieve']}"
    results = extract_both(registry, name, captured)
    scene_seeds = stage4_record["scene_seeds"]
    demonstrations = per_demonstration(registry, captured, scene_seeds)
    actions = {mode: hp.learned_action(spec["achieve"], results[mode].operator).to_dict() for mode in ("mean", "min_max")}
    saved = json.loads(LEARNED_SAVED.read_text())

    # Which runs does the handoff check leave exactly as they were before it? The earlier file labels
    # a run by its original seed, even where the forward skill's re-roll moved it to another scene.
    earlier = json.loads((PHASE1 / "stow_learned.json").read_text())["demonstrations"]
    untouched = {str(now["seed"]): {k: now[k] for k in ("start", "end")} == {k: then[k] for k in ("start", "end")}
                 for now, then in zip(demonstrations, earlier)}

    # Replanning with each action, as Stage 4 does: library only, every mode named.
    goal = sd.goal_from_operator(PHASE1 / "stow_min_max_operator.json")
    replanning = {}
    for mode, action in actions.items():
        for planner in ("relaxed", "exact"):
            domain = sd.domain(extra_actions=[Action.from_dict(action)])
            result = hp.plan(domain, planned_start, *goal, mode=planner, max_delegated=0)
            replanning[f"{mode}+{planner}"] = {"found": bool(result), "actions": list(result.actions) if result else None,
                                               "cost": list(result.cost) if result else None}

    def matches_expected(action: dict) -> dict:
        return {part: action[part] == EXPECTED_ACTION[part] for part in EXPECTED_ACTION}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for mode in ("mean", "min_max"):
        (OUT_DIR / f"learned_{mode}_operator.json").write_text(json.dumps({
            "seeds": scene_seeds, "operator": results[mode].operator.to_dict(), "scores": results[mode].scores,
            "action": actions[mode]}, indent=2))
    record = {
        "made_by": "PYTHONNOUSERSITE=1 python scripts/validate_handoff.py collect",
        "recipe": "learn_operator() from scripts/stow_stage4_execute.py",
        "checkpoint": STOW_CHECKPOINT,
        "original_seeds": list(stage4.EXTRACTION_SEEDS),
        "scene_seeds": scene_seeds,
        "handoff_conditions": stage4_record["handoff_conditions"],
        "handoff_attempts": stage4_record["handoff_attempts"],
        "handoff_summary": stage4_record["handoff_summary"],
        "demonstrations": demonstrations,
        "runs_identical_to_the_runs_before_the_check": untouched,
        "operators": {mode: results[mode].operator.to_dict() for mode in results},
        "scores": {mode: results[mode].scores for mode in results},
        "comparison": compare_operators(results, demonstrations, registry.keys()),
        "actions": actions,
        "actions_match_expected": {mode: matches_expected(actions[mode]) for mode in actions},
        "actions_equal_the_saved_action": {mode: actions[mode] == saved["action"] for mode in actions},
        "replanning": replanning,
    }
    target.write_text(json.dumps(record, indent=2))

    print(f"conditions: {record['handoff_conditions']}")
    print(f"summary: {record['handoff_summary']}")
    for attempt in record["handoff_attempts"]:
        if attempt["stage"] != "valid":
            print(f"  seed {attempt['seed']}, scene {attempt['scene_seed']}: {attempt['stage']}: {attempt['reasons']}")
    print(f"scene seeds: {scene_seeds}")
    print(f"runs identical to those before the check: {untouched}")
    print("per-run scores, start -> end")
    for d in demonstrations:
        print(f"  {d['seed']:>6}  clear {d['start']['clear_of_walls(cube_a)']:.4f} -> {d['end']['clear_of_walls(cube_a)']:.4f}   "
              f"pocket {d['start']['in_region(cube_a,pocket)']:.4f} -> {d['end']['in_region(cube_a,pocket)']:.4f}   "
              f"tcp_A {d['start']['tcp_near(cube_a)']:.4f}   src_b {d['start']['in_region(cube_b,src_b)']:.4f}")
    comparison = record["comparison"]
    print(f"Pre same {comparison['Pre']['same']}, Add same {comparison['Add']['same']}, Del same {comparison['Del']['same']}, "
          f"largest weight difference {comparison['largest_weight_difference']:.4f}")
    for row in comparison["closest_calls"][:3]:
        print(f"  closest: {row['predicate']} {row['line']} {row['margin']:+.4f}")
    for mode, action in actions.items():
        print(f"{mode}: {action}\n  matches expected {record['actions_match_expected'][mode]}, "
              f"equals the saved action {record['actions_equal_the_saved_action'][mode]}")
    for combination, row in replanning.items():
        print(f"replanning {combination}: {' > '.join(row['actions']) if row['found'] else 'NO PLAN'}")
    print(f"wrote {target}")


def drawer() -> None:
    from inverse_skills.envs import drawer_domain as dd
    from inverse_skills.envs import drawer_executive as ex
    from inverse_skills.envs import drawer_predicates as dp
    from inverse_skills.envs import drawer_primitives as prim
    from inverse_skills.operators.handoff import handoff_conditions, handoff_violations

    target = OUT_DIR / "drawer_handoffs.json"
    if target.exists():
        raise SystemExit(f"{target} exists; outputs are never overwritten")
    spec = json.loads(DRAWER_SPEC.read_text())
    conditions = handoff_conditions(dd.domain(), frozenset({dd.CLOSED}), spec)
    registry = dp.registry()
    env = prim.make_env()
    rows = []
    for seed in range(2000, 2020):           # Drawer Stage 4's seeds; the prefix is its one library step
        obs = prim.push_close(env, prim.reset(env, seed))
        obs = prim.approach_handle(env, obs)
        scores = ex.scores(obs, registry)
        violations = handoff_violations(conditions, scores)
        rows.append({"seed": seed, "valid": not violations, "violations": violations,
                     "scores": {p: scores[p] for p in sorted(conditions.hold | conditions.absent)}})
    env.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record = {"made_by": "PYTHONNOUSERSITE=1 python scripts/validate_handoff.py drawer", "conditions": conditions.to_dict(),
              "not_required": sorted(set(dd.domain().scored) - conditions.hold - conditions.absent),
              "seeds": rows, "valid": sum(row["valid"] for row in rows), "of": len(rows)}
    target.write_text(json.dumps(record, indent=2))
    print(f"conditions {record['conditions']}, not required {record['not_required']}")
    print(f"raw handoff validity on the Drawer's Stage 4 seeds: {record['valid']} of {record['of']}")
    for row in rows:
        if not row["valid"]:
            print(f"  seed {row['seed']}: {row['violations']}")
    print(f"wrote {target}")


def survey() -> None:
    import stow_stage4_execute as stage4

    from inverse_skills.envs import stow_domain as sd
    from inverse_skills.envs import stow_primitives as prim
    from inverse_skills.envs.stow_hole_env import load_spec

    target = OUT_DIR / "training_seed_handoffs.json"
    if target.exists():
        raise SystemExit(f"{target} exists; outputs are never overwritten")
    env = prim.make_env()
    record = stage4.survey_handoffs(env, load_spec(STOW_SPEC), range(32), sd.stowed_state(), sd.domain())
    env.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"made_by": "PYTHONNOUSERSITE=1 python scripts/validate_handoff.py survey", **record}, indent=2))
    print(f"raw handoff validity on training seeds 0-31: {record['valid']} of {record['of']}; "
          f"forward re-rolled on {sum(row['forward_rerolls'] > 0 for row in record['seeds'])} seeds")
    for row in record["seeds"]:
        if not row["valid"]:
            print(f"  seed {row['seed']}: {row['violations']}")
    print(f"wrote {target}")


def earlier_comparison() -> None:
    import stow_stage1_extractor_comparison as earlier

    AFTER.mkdir(parents=True, exist_ok=True)
    earlier.SCORES_OUT = AFTER / "stage1_replay_scores.json"
    earlier.COMPARISON_OUT = AFTER / "stage1_extractor_comparison.json"
    earlier.main()


def load(path) -> dict:
    return json.loads(Path(path).read_text())


def without_docstrings(source: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
    return tree


def only_the_default_differs(path: str, set_back) -> bool:
    """Is this file's code, docstrings aside, the code of BEFORE_MIGRATION once `set_back` restores the old default?"""
    before = subprocess.run(["git", "show", f"{BEFORE_MIGRATION}:{path}"], check=True, capture_output=True, text=True).stdout
    now = without_docstrings(Path(path).read_text())
    set_back(now)
    return ast.dump(now) == ast.dump(without_docstrings(before))


def old_extraction_default(tree) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "mode":
            assert node.value.value == "min_max"
            node.value.value = "mean"


def old_planning_default(tree) -> None:
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "plan")
    position = [a.arg for a in function.args.kwonlyargs].index("mode")
    assert function.args.kw_defaults[position].value == "exact"
    function.args.kw_defaults[position].value = "relaxed"


def after_migration() -> None:
    import inspect
    import sys

    from validate_modes_execution import differences

    from inverse_skills.envs import drawer_domain as dd
    from inverse_skills.envs import stow_domain as sd
    from inverse_skills.operators import hole_planner as hp
    from inverse_skills.operators.extractor import OperatorExtractionConfig

    target = AFTER / "verification.json"
    if target.exists():
        raise SystemExit(f"{target} exists; outputs are never overwritten")
    checks = []

    def check(name: str, passed: bool, detail=None) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})
        print(f"{'PASS' if passed else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not passed else ""))

    def text(path) -> str:
        return Path(path).read_text()

    phase2 = {name: load(PHASE2 / f"{name}.json") for name in ("stow", "drawer")}

    # The defaults, read from the code the scripts use.
    check("extraction default is min_max", OperatorExtractionConfig().mode == "min_max")
    check("planning default is exact", inspect.signature(hp.plan).parameters["mode"].default == "exact")
    check("the extractor's code differs from the commit before the migration only in the default",
          only_the_default_differs("src/inverse_skills/operators/extractor.py", old_extraction_default))
    check("the relaxed planner's code differs from that commit only in the default",
          only_the_default_differs("src/inverse_skills/operators/hole_planner.py", old_planning_default))
    check("the exact planner's code is the same as at that commit",
          only_the_default_differs("src/inverse_skills/operators/exact_hole_planner.py", lambda tree: None))

    # Stage 1, by default, against min_max named.
    for name, part, domain_module in (("stow", "stow", sd), ("drawer", "drawer_current", dd)):
        now, named = load(AFTER / f"{name}_stage1_operator.json"), load(PHASE1 / f"{part}_min_max_operator.json")
        check(f"{name} Stage 1 by default equals min_max named: operator, weights, scores",
              now["operator"] == named["operator"] and now["scores"] == named["scores"])
        check(f"{name} Stage 1's own checks pass", all(now["checks"].values()), now["checks"])
        check(f"{name} Stage 1 used min_max", now["operator"]["metadata"]["extractor"] == "first_last_score_min_max_v0")
        saved_operator = f"artifacts/{name}/stage1_operator.json"
        check(f"{name} inverse goal equals the saved one",
              domain_module.goal_from_operator(AFTER / f"{name}_stage1_operator.json") == domain_module.goal_from_operator(saved_operator))

    # Stage 2, by default, against min_max + exact named.
    for name in ("stow", "drawer"):
        now = load(AFTER / f"{name}_stage2_plan.json")
        configs = phase2[name]["configs"]["min_max+exact"]
        check(f"{name} Stage 2's own check passes", now["all_seeds_match"])
        for row in now["seeds"]:
            named = configs[f"frontier, seed {row['seed']}"]
            same = (row["plan"] == named["actions"] and row["cost"] == named["cost"] and row["delegated"] == named["delegated"]
                    and row["hole_indices"] == named["hole_positions"] and row["ties"] == named["tied_plans"]
                    and row["hole_contexts"] == named["diagnostics"]["hole_contexts"]
                    and row["candidates"] == named["diagnostics"]["candidates"] and row["expanded"] == named["diagnostics"]["expanded"])
            check(f"{name} Stage 2 seed {row['seed']} by default equals min_max + exact named, with its hole contexts", same)
        check(f"{name} hole_spec.json by default is byte-identical to the saved one",
              text(AFTER / f"{name}_hole_spec.json") == text(f"artifacts/{name}/hole_spec.json"))
    drawer_now = load(AFTER / "drawer_stage2_plan.json")["ablations"]
    configs = phase2["drawer"]["configs"]["min_max+exact"]
    check("drawer ablation library_only: no plan", drawer_now["library_only"]["plan"] is None and configs["library only"]["actions"] is None)
    check("drawer ablation with_grasp equals the named run", drawer_now["with_grasp"]["plan"] == configs["scripted grasp"]["actions"])
    saved_unrestricted = load("artifacts/drawer/stage2_plan.json")["ablations"]["unrestricted"]
    check("drawer ablation unrestricted keeps its meaning: same plan, cost and gaps as saved",
          all(drawer_now["unrestricted"][k] == saved_unrestricted[k] for k in ("plan", "cost", "capability_gaps", "synthesizable_gaps")))

    # The direct baseline.
    check("direct baseline spec is byte-identical to the saved one",
          text(AFTER / "direct/hole_spec_direct.json") == text("artifacts/drawer/direct/hole_spec_direct.json"))
    check("direct baseline plan file equals the saved one", load(AFTER / "direct/plan_direct.json") == load("artifacts/drawer/direct/plan_direct.json"))

    # What Stage 3 would receive.
    for name in ("stow", "drawer"):
        now = load(AFTER / f"{name}_stage3_input.json")
        check(f"{name} Stage 3 input: spec byte-identical to saved, environment fits every checkpoint, handoffs valid",
              now["bytes_equal_saved_spec"] and all(c["observation_space_fits"] and c["action_space_fits"] for c in now["checkpoints"])
              and all(r["handoff_valid"] for r in now["resets"]))

    # Stage 4, by default, against the named run and the saved file.
    now = load(AFTER / "stow_stage4.json")
    named = load(HANDOFF / "stage4/stow_min_max_exact.json")
    saved = load("artifacts/stow/stage4_execution.json")
    check("stow Stage 4 by default equals min_max + exact named, value for value", not differences(named, now), differences(named, now))
    without_new_section = {k: v for k, v in now.items() if k != "handoff_validity"}
    check("stow Stage 4 by default equals the saved results, value for value", not differences(saved, without_new_section),
          differences(saved, without_new_section))
    check("stow Stage 4's own check passes", now["passed"])
    learned = load(AFTER / "stow_learned_operator.json")
    check("stow learned operator used min_max", learned["operator"]["metadata"]["extractor"] == "first_last_score_min_max_v0")
    check("stow learned action equals the saved action", learned["action"] == load(LEARNED_SAVED)["action"])
    check("stow learned action has the expected Pre, Add and Del", all(learned["action"][k] == v for k, v in EXPECTED_ACTION.items()))

    now = load(AFTER / "drawer_stage4.json")
    named = load("artifacts/default_mode_validation/phase3/drawer_min_max_exact.json")
    saved = load("artifacts/drawer/stage4_execution_after_push_fix.json")
    for label, other in (("min_max + exact named", named), ("the saved results", saved)):
        found = differences(other, now)
        unexplained = {k: v for k, v in found.items()
                       if not (k.startswith("results.library_only[") and (".capability_gaps" in k or ".unmet_needs" in k))}
        check(f"drawer Stage 4 by default equals {label}, apart from the library-only report's field", not unexplained, unexplained)
    check("drawer Stage 4 used the exact mode: its library-only rows report unmet needs",
          all(row["unmet_needs"] == ["drawer_open(drawer)", "handle_grasped(drawer)"] for row in now["results"]["library_only"]))

    # The comparison scripts still compare two modes.
    def without_times(value):
        return {k: v for k, v in differences(value[0], value[1]).items() if not k.endswith("runtime_ms")}

    now, saved = load(AFTER / "planner_comparison_scenarios.json"), load("artifacts/planner_comparison/scenarios.json")
    check("named scenarios: the two modes compare as before", not without_times((saved, now)), without_times((saved, now)))
    now, saved = load(AFTER / "planner_comparison_random.json"), load("artifacts/planner_comparison/random_problems.json")
    check("random problems: every count and saved case as before", now["results"] == saved["results"], differences(saved["results"], now["results"]))
    for name in ("stage1_replay_scores.json", "stage1_extractor_comparison.json"):
        now, saved = load(AFTER / name), load(f"artifacts/stow/{name}")
        now.pop("demonstration_scores", None), saved.pop("demonstration_scores", None)    # a file path
        check(f"earlier extractor comparison reproduces {name}", now == saved)
    now, saved = load(AFTER / "stow_planner_demo.json"), load("artifacts/stow/planner_demo.json")
    check("planner demo: same plans, costs and hole positions",
          all(now["cases"][c][k] == saved["cases"][c][k] for c in saved["cases"] for k in ("plan", "cost", "delegated", "hole_indices", "hole_free_plan")))

    # The failure itself stays reproducible, and the named scripts do not depend on the defaults.
    now, saved = load(AFTER / "phase1_named/stow_learned.json"), load(PHASE1 / "stow_learned.json")
    check("the original failure still reproduces with the check bypassed", now["actions"] == saved["actions"] and now["demonstrations"] == saved["demonstrations"])
    for part in ("stow", "drawer_original", "drawer_current"):
        now, saved = load(AFTER / f"phase1_named/{part}.json"), load(PHASE1 / f"{part}.json")
        check(f"named extraction, {part}: unchanged by the new defaults",
              all(now[k] == saved[k] for k in ("demonstrations", "operators", "scores", "comparison")))
    for name in ("stow", "drawer"):
        now, saved = load(AFTER / f"phase2_named/{name}.json"), load(PHASE2 / f"{name}.json")
        check(f"named planning, {name}: all four combinations unchanged by the new defaults",
              now["configs"] == saved["configs"] and now["comparisons_with_mean_relaxed"] == saved["comparisons_with_mean_relaxed"])

    # The whole test suite.
    run = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"],
                         capture_output=True, text=True, check=False)
    line = next(line for line in reversed(run.stdout.splitlines()) if "passed" in line or "failed" in line)
    check(f"test suite: {line}", run.returncode == 0, line)

    target.write_text(json.dumps({"made_by": "PYTHONNOUSERSITE=1 python scripts/validate_handoff.py after-migration",
                                  "all_passed": all(c["passed"] for c in checks), "checks": checks}, indent=2))
    print(f"\n{sum(c['passed'] for c in checks)} of {len(checks)} checks passed; wrote {target}")


def main() -> None:
    parts = {"investigate": investigate, "collect": collect, "drawer": drawer, "survey": survey,
             "earlier-comparison": earlier_comparison, "after-migration": after_migration}
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=tuple(parts))
    args = ap.parse_args()
    parts[args.part]()


if __name__ == "__main__":
    main()
