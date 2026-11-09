"""Default-mode validation, Phase 1: mean versus min_max extraction on the real demonstrations.

Each part runs in its own process and replays one set of demonstrations with the stage's own
code, then extracts the operator twice from those same rollouts, naming the mode both times:

  stow             Stage 1's demonstrate() on the seeds in artifacts/stow/stage1_operator.json
  drawer-original  Stage 1's demonstrate() with the push_close that wrote the saved operator
                   (ORIGINAL_PUSH_COMMIT, read from Git; the stop rule changed afterwards)
  drawer-current   Stage 1's demonstrate() with today's push_close, which Stages 2-4 now run
  stow-learned     Stage 4's original collection loop, with no handoff check, and the original
                   checkpoint, on its seeds. Kept so that its failure stays reproducible; the
                   collection Stage 4 uses now is in validate_handoff.py.

Every demonstration's start and end score of every predicate is written. The mean operator of a
replay is checked against the saved results, and the mismatch is reported, not hidden.

Run from the repository root, with PYTHONNOUSERSITE=1 (stray ~/.local packages break ManiSkill):
  PYTHONNOUSERSITE=1 python scripts/validate_modes_extraction.py stow

Writes under artifacts/default_mode_validation/phase1/ (or --out-dir), never overwriting:
  <part>.json                    scores, both operators, and every difference
  <part>_<mode>_operator.json    each operator, in the format of Stage 1's output file
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from inverse_skills.operators.extractor import OperatorExtractionConfig, OperatorExtractor

OUT_DIR = Path("artifacts/default_mode_validation/phase1")
STOW_SAVED = Path("artifacts/stow/stage1_operator.json")
DRAWER_SAVED = Path("artifacts/drawer/stage1_operator.json")
LEARNED_SAVED = Path("artifacts/stow/learned_operator.json")
STOW_CHECKPOINT = "artifacts/stow/stage3/1790788534/sac_hole_best"
STOW_SPEC = Path("artifacts/stow/hole_spec.json")
# drawer_primitives.py as it was when artifacts/drawer/stage1_operator.json was written. The stop
# rule of push_close changed in 646797f; the only commit in between (b2eaab0) added tilt_deg.
ORIGINAL_PUSH_COMMIT = "8f428d5"
# As in stow_stage1_extractor_comparison.py: a faithful replay agrees to rounding error.
TOLERANCE = 1e-9
MODES = ("mean", "min_max")
PARTS = (("preconditions", "Pre"), ("add_effects", "Add"), ("delete_effects", "Del"))


def keys(operator, part: str) -> list[str]:
    return sorted(term.key for term in getattr(operator, part))


def goal_literals(operator) -> tuple[list[str], list[str]]:
    """The positive and negative inverse goal, as Stage 2 reads them (`goal_from_operator`)."""
    terms = operator.to_dict()["inverse_target_terms"]
    return (sorted(t["key"] for t in terms if t["polarity"] == "positive"),
            sorted(t["key"] for t in terms if t["polarity"] == "negative"))


def per_demonstration(registry, rollouts, labels) -> list[dict]:
    predicates = registry.keys()
    rows = []
    for label, rollout in zip(labels, rollouts):
        start, end = registry.evaluate_all(rollout.first()), registry.evaluate_all(rollout.last())
        rows.append({"seed": label,
                     "start": {p: start[p].score for p in predicates},
                     "end": {p: end[p].score for p in predicates}})
    return rows


def closest_calls(demonstrations, predicates, config: OperatorExtractionConfig) -> list[dict]:
    """For each predicate, how near its scores come to a line that would change the min_max
    decision, whichever side they are on. Smallest margins first."""
    rows = []
    for p in predicates:
        starts = [d["start"][p] for d in demonstrations]
        ends = [d["end"][p] for d in demonstrations]
        lines = {"start_min vs true line": min(starts) - config.true_threshold,
                 "start_max vs false line": config.false_threshold - max(starts),
                 "end_min vs true line": min(ends) - config.true_threshold,
                 "end_max vs false line": config.false_threshold - max(ends)}
        which = min(lines, key=lambda name: abs(lines[name]))
        rows.append({"predicate": p, "line": which, "margin": lines[which],
                     "start_min": min(starts), "start_max": max(starts),
                     "end_min": min(ends), "end_max": max(ends)})
    return sorted(rows, key=lambda row: abs(row["margin"]))


def compare_operators(results: dict, demonstrations, predicates) -> dict:
    mean_op, min_max_op = results["mean"].operator, results["min_max"].operator
    config = OperatorExtractionConfig(mode="min_max")
    comparison = {}
    all_same = True
    for part, label in PARTS:
        by_mean, by_min_max = keys(mean_op, part), keys(min_max_op, part)
        same = by_mean == by_min_max
        all_same = all_same and same
        differing = []
        for p in sorted(set(by_mean) ^ set(by_min_max)):
            differing.append({
                "predicate": p,
                "extracted_by": "mean only" if p in by_mean else "min_max only",
                "start_scores": [d["start"][p] for d in demonstrations],
                "end_scores": [d["end"][p] for d in demonstrations],
                "mean_scores": results["mean"].scores[p],
                "min_max_scores": results["min_max"].scores[p],
            })
        comparison[label] = {"same": same, "mean": by_mean, "min_max": by_min_max, "differing": differing}

    weights = []
    for part, label in PARTS:
        by_mean = {t.key: t.weight for t in getattr(mean_op, part)}
        by_min_max = {t.key: t.weight for t in getattr(min_max_op, part)}
        for p in sorted(set(by_mean) & set(by_min_max)):
            weights.append({"part": label, "predicate": p, "mean": by_mean[p], "min_max": by_min_max[p],
                            "difference": by_min_max[p] - by_mean[p]})
    goal_mean, goal_min_max = goal_literals(mean_op), goal_literals(min_max_op)
    comparison.update({
        "same_predicate_sets": all_same,
        "inverse_goal": {"same": goal_mean == goal_min_max,
                         "mean": {"positive": goal_mean[0], "negative": goal_mean[1]},
                         "min_max": {"positive": goal_min_max[0], "negative": goal_min_max[1]}},
        "weights": weights,
        "largest_weight_difference": max((abs(w["difference"]) for w in weights), default=0.0),
        "metadata": {"mean": mean_op.metadata, "min_max": min_max_op.metadata},
        "closest_calls": closest_calls(demonstrations, predicates, config),
    })
    return comparison


def extract_both(registry, skill: str, rollouts) -> dict:
    """Both modes named explicitly: neither result depends on the default."""
    return {mode: OperatorExtractor(registry, OperatorExtractionConfig(mode=mode)).extract(skill, rollouts)
            for mode in MODES}


def against_saved(saved_scores: dict, saved_operator: dict, mean_result) -> dict:
    """Does the replay's mean extraction give back the saved numbers and the saved operator?"""
    differences = {f"{p} {which}": mean_result.scores[p][which] - saved_scores[p][which]
                   for p in saved_scores for which in ("start_mean", "end_mean")}
    largest = max(abs(d) for d in differences.values())
    return {
        "largest_score_difference": largest,
        "tolerance": TOLERANCE,
        "scores_match": largest <= TOLERANCE,
        "differences_over_tolerance": {k: v for k, v in differences.items() if abs(v) > TOLERANCE},
        "operator_sets_match": all(keys(mean_result.operator, part) == sorted(t["key"] for t in saved_operator[part])
                                   for part, _ in PARTS),
        "operator_identical_including_weights": mean_result.operator.to_dict() == saved_operator,
    }


def write_part(name: str, record: dict, results: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = record["seeds"]
    for mode in MODES:
        (OUT_DIR / f"{name}_{mode}_operator.json").write_text(json.dumps({
            "seeds": seeds, "operator": results[mode].operator.to_dict(), "scores": results[mode].scores}, indent=2))
    (OUT_DIR / f"{name}.json").write_text(json.dumps(record, indent=2))


def show(name: str, record: dict) -> None:
    comparison = record["comparison"]
    print(f"\n{name}: {len(record['demonstrations'])} demonstrations, seeds {record['seeds']}")
    print(f"  against the saved results: {record['against_saved']}")
    for _, label in PARTS:
        row = comparison[label]
        print(f"  {label} same {row['same']}: {', '.join(row['mean']) or '-'}")
        if not row["same"]:
            print(f"    min_max gives: {', '.join(row['min_max']) or '-'}")
        for d in row["differing"]:
            print(f"    {d['predicate']} ({d['extracted_by']})\n      start {d['start_scores']}\n      end   {d['end_scores']}")
    goal = comparison["inverse_goal"]
    print(f"  inverse goal same {goal['same']}: {goal['mean']}")
    if not goal["same"]:
        print(f"    min_max gives: {goal['min_max']}")
    print(f"  largest weight difference {comparison['largest_weight_difference']:.4f}")
    print("  closest calls:")
    for row in comparison["closest_calls"][:4]:
        print(f"    {row['predicate']:28s} {row['line']:24s} margin {row['margin']:+.4f}  "
              f"start {row['start_min']:.4f}-{row['start_max']:.4f}  end {row['end_min']:.4f}-{row['end_max']:.4f}")


def run_stow() -> None:
    from stow_stage1_extract import demonstrate

    from inverse_skills.envs import stow_predicates as sp
    from inverse_skills.envs import stow_primitives as prim

    saved = json.loads(STOW_SAVED.read_text())
    seeds = saved["seeds"]
    env = prim.make_env()
    rollouts = [demonstrate(env, seed) for seed in seeds]
    env.close()

    registry = sp.registry()
    results = extract_both(registry, "stow", rollouts)
    demonstrations = per_demonstration(registry, rollouts, seeds)
    earlier = json.loads(Path("artifacts/stow/stage1_replay_scores.json").read_text())["demonstrations"]
    earlier_comparison = json.loads(Path("artifacts/stow/stage1_extractor_comparison.json").read_text())
    record = {
        "recipe": "demonstrate() from scripts/stow_stage1_extract.py",
        "seeds": seeds,
        "demonstrations": demonstrations,
        "against_saved": against_saved(saved["scores"], saved["operator"], results["mean"]),
        "against_earlier_replay": {"file": "artifacts/stow/stage1_replay_scores.json",
                                   "per_demonstration_scores_identical": demonstrations == earlier},
        "against_earlier_comparison": {
            "file": "artifacts/stow/stage1_extractor_comparison.json",
            "both_operators_identical": all(results[mode].operator.to_dict() == earlier_comparison["operators"][mode]
                                            for mode in MODES),
            "min_max_scores_identical": results["min_max"].scores == earlier_comparison["min_max_scores"]},
        "operators": {mode: results[mode].operator.to_dict() for mode in MODES},
        "scores": {mode: results[mode].scores for mode in MODES},
        "comparison": compare_operators(results, demonstrations, registry.keys()),
    }
    write_part("stow", record, results)
    show("stow", record)
    print(f"  per-demonstration scores identical to the earlier replay: "
          f"{record['against_earlier_replay']['per_demonstration_scores_identical']}")
    print(f"  against the earlier comparison: {record['against_earlier_comparison']}")


def load_primitives_at(commit: str):
    """drawer_primitives.py as it was at `commit`, read from Git, under its own module name."""
    source = subprocess.run(["git", "show", f"{commit}:src/inverse_skills/envs/drawer_primitives.py"],
                            check=True, capture_output=True, text=True).stdout
    # The module must outlive this function only in memory, so the temporary file can go.
    with tempfile.NamedTemporaryFile("w", suffix=".py") as handle:
        handle.write(source)
        handle.flush()
        spec = importlib.util.spec_from_file_location("drawer_primitives_at_stage1", handle.name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def run_drawer(name: str, primitives_commit: str | None) -> None:
    import drawer_stage1_extract as stage1

    from inverse_skills.envs import drawer_predicates as dp

    if primitives_commit:
        stage1.prim = load_primitives_at(primitives_commit)
    saved = json.loads(DRAWER_SAVED.read_text())
    seeds = saved["seeds"]
    env = stage1.prim.make_env()
    demos = [stage1.demonstrate(env, seed) for seed in seeds]
    env.close()
    rollouts = [rollout for rollout, _ in demos]
    start_openings_mm = [round(opening * 1000, 1) for _, opening in demos]

    registry = dp.registry()
    results = extract_both(registry, "push_close", rollouts)
    demonstrations = per_demonstration(registry, rollouts, seeds)
    record = {
        "recipe": "demonstrate() from scripts/drawer_stage1_extract.py",
        "push_close_from": primitives_commit or "the working tree",
        "seeds": seeds,
        "start_openings_mm": start_openings_mm,
        "start_openings_match_saved": start_openings_mm == saved["start_openings_mm"],
        "demonstrations": demonstrations,
        "against_saved": against_saved(saved["scores"], saved["operator"], results["mean"]),
        "operators": {mode: results[mode].operator.to_dict() for mode in MODES},
        "scores": {mode: results[mode].scores for mode in MODES},
        "comparison": compare_operators(results, demonstrations, registry.keys()),
    }
    write_part(name, record, results)
    show(name, record)
    print(f"  start openings match the saved ones: {record['start_openings_match_saved']}")


def rollouts_without_handoff_check(env, model, spec: dict, seeds, planned_start) -> list:
    """Stage 4's collection loop as it was before the handoff check: the executive's step checks only."""
    import stow_stage4_execute as stage4

    from inverse_skills.envs import stow_executive as ex
    from inverse_skills.envs import stow_predicates as sp
    from inverse_skills.envs import stow_primitives as prim
    from inverse_skills.logging.rollout import ForwardRollout
    from inverse_skills.operators.hole_planner import Action

    prefix = [Action.from_dict(a) for a in spec["prefix"]]
    registry = sp.registry()
    name = f"learned:{spec['achieve']}"
    rollouts = []
    for seed in seeds:
        obs, regions, _ = stage4.stowed(env, seed, planned_start)
        obs, report = ex.execute_plan(env, obs, prefix, regions=regions)
        if len(report.steps) != len(prefix) or not all(s.ok for s in report.steps):
            continue
        before = prim.obs_to_scene(obs, regions)
        obs = ex.run_learned(env, obs, model, spec["achieve"], regions, registry)
        after = prim.obs_to_scene(obs, regions, timestep=1)
        rollouts.append(ForwardRollout(skill_name=name, demo_id=f"run_{seed}", scenes=[before, after]))
    return rollouts


def run_stow_learned() -> None:
    import stow_stage4_execute as stage4
    from stable_baselines3 import SAC

    from inverse_skills.envs import stow_domain as sd
    from inverse_skills.envs import stow_predicates as sp
    from inverse_skills.envs import stow_primitives as prim
    from inverse_skills.envs.stow_hole_env import load_spec
    from inverse_skills.operators import hole_planner as hp

    model = SAC.load(STOW_CHECKPOINT, device="cuda")
    spec = load_spec(STOW_SPEC)
    env = prim.make_env()
    captured = rollouts_without_handoff_check(env, model, spec, stage4.EXTRACTION_SEEDS, sd.stowed_state())
    env.close()

    registry = sp.registry()
    name = f"learned:{spec['achieve']}"
    results = extract_both(registry, name, captured)
    seeds = [int(rollout.demo_id.split("_")[1]) for rollout in captured]
    demonstrations = per_demonstration(registry, captured, seeds)
    saved = json.loads(LEARNED_SAVED.read_text())
    actions = {mode: hp.learned_action(spec["achieve"], results[mode].operator).to_dict() for mode in MODES}
    planned_before_hole = planned_state_before_hole(spec)
    record = {
        "recipe": "Stage 4's collection loop before the handoff check (rollouts_without_handoff_check)",
        "checkpoint": STOW_CHECKPOINT,
        "seeds": seeds,
        "demonstrations": demonstrations,
        "against_saved": against_saved(saved["scores"], saved["operator"], results["mean"]),
        "operators": {mode: results[mode].operator.to_dict() for mode in MODES},
        "scores": {mode: results[mode].scores for mode in MODES},
        "actions": actions,
        "action_same_in_both_modes": actions["mean"] == actions["min_max"],
        "action_mean_matches_saved": actions["mean"] == saved["action"],
        "comparison": compare_operators(results, demonstrations, registry.keys()),
        # Analysis only, from the scores above. Nothing here changes which runs Stage 4 uses.
        "planned_state_before_hole": sorted(planned_before_hole),
        "runs_measured_in_the_planned_state": runs_in_planned_state(registry, demonstrations, planned_before_hole),
        "min_max_leaving_one_run_out": leave_one_run_out(registry, name, demonstrations, actions["mean"], spec["achieve"]),
    }
    write_part("stow_learned", record, results)
    show("stow_learned", record)
    print(f"  action mean:    {actions['mean']}")
    print(f"  action min_max: {actions['min_max']}")
    print(f"  planned state before the hole: {record['planned_state_before_hole']}")
    print("  run -> measured state matches the plan's prediction | min_max without this run gives the mean action")
    left_out = record["min_max_leaving_one_run_out"]
    for row, out in zip(record["runs_measured_in_the_planned_state"], left_out):
        print(f"    {row['seed']}: {row['matches']!s:5} {row['differences'] or ''} | {out['gives_the_mean_action']}")


def planned_state_before_hole(spec: dict) -> frozenset[str]:
    """The symbolic state the Stow domain predicts when the hole takes over: its prefix, replayed."""
    from inverse_skills.envs import stow_domain as sd
    from inverse_skills.operators.hole_planner import Action

    domain = sd.domain()
    state = sd.stowed_state()
    for action in spec["prefix"]:
        state = domain.apply(state, Action.from_dict(action))
    return state


def runs_in_planned_state(registry, demonstrations, planned) -> list[dict]:
    """For each run: the scored literals whose measured truth differs from the prediction."""
    from inverse_skills.operators.hole_planner import symbolic_state

    rows = []
    for demonstration in demonstrations:
        measured = symbolic_state(demonstration["start"])
        differences = sorted((measured ^ planned) & set(registry.keys()))
        rows.append({"seed": demonstration["seed"], "matches": not differences, "differences": differences})
    return rows


def leave_one_run_out(registry, skill: str, demonstrations, mean_action: dict, achieve: str) -> list[dict]:
    """Would min_max give the mean operator's action if this one run were not there?
    Uses the extractor's own min_max rule on the recorded scores of the other runs."""
    from inverse_skills.operators.hole_planner import learned_action

    extractor = OperatorExtractor(registry, OperatorExtractionConfig(mode="min_max"))
    predicates = registry.keys()
    rows = []
    for left_out in demonstrations:
        others = [d for d in demonstrations if d is not left_out]
        start = {p: [d["start"][p] for d in others] for p in predicates}
        end = {p: [d["end"][p] for d in others] for p in predicates}
        result = extractor._extract_min_max(skill, start, end, len(others))
        action = learned_action(achieve, result.operator).to_dict()
        rows.append({"left_out": left_out["seed"], "action": action, "gives_the_mean_action": action == mean_action})
    return rows


def main() -> None:
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("stow", "drawer-original", "drawer-current", "stow-learned"))
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    OUT_DIR = args.out_dir
    target = OUT_DIR / (args.part.replace("-", "_") + ".json")
    if target.exists():
        raise SystemExit(f"{target} exists; outputs are never overwritten")
    if args.part == "stow":
        run_stow()
    elif args.part == "drawer-original":
        run_drawer("drawer_original", ORIGINAL_PUSH_COMMIT)
    elif args.part == "drawer-current":
        run_drawer("drawer_current", None)
    else:
        run_stow_learned()
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
