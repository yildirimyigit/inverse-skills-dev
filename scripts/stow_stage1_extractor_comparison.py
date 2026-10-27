"""Final extractor check: mean versus min/max extraction on the original stow demonstrations.

Stage 1 saved only the averages of its five demonstrations, so the conservative min/max rule
could not be evaluated from the saved file. This replays the same five demonstrations (the
seeds stored in artifacts/stow/stage1_operator.json) with Stage 1's own demonstrate(): the same
reset, the same scripted forward_stow, the same first and last scene, in the same order and the
same environment. It records every demonstration's start and end score for every predicate.

Before anything is compared, the replay's averages must reproduce the saved averages to within
TOLERANCE. If they do not, the replay is not the same demonstrations: the scores are written
with the mismatch and nothing is compared. If they do, the operator is extracted twice from the
replayed demonstrations, by the default mean rule and by the min/max rule, and every difference
is listed with the scores behind it.

Run from the repository root with PYTHONNOUSERSITE=1 (stray ~/.local packages break ManiSkill).

Writes, never overwriting:
  artifacts/stow/stage1_replay_scores.json          per-demonstration scores, averages check
  artifacts/stow/stage1_extractor_comparison.json   both operators, inverse goals, differences
"""

from __future__ import annotations

import json
from pathlib import Path

from stow_stage1_extract import demonstrate  # Stage 1's own recipe (scripts/ is on the path)

from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs import stow_primitives as prim
from inverse_skills.operators.extractor import OperatorExtractionConfig, OperatorExtractor

SAVED = Path("artifacts/stow/stage1_operator.json")
SCORES_OUT = Path("artifacts/stow/stage1_replay_scores.json")
COMPARISON_OUT = Path("artifacts/stow/stage1_extractor_comparison.json")
MADE_BY = "PYTHONNOUSERSITE=1 python scripts/stow_stage1_extractor_comparison.py"
# The saved averages carry full double precision, so a faithful replay agrees with them to
# rounding error (about 1e-16). 1e-9 allows for that while staying far below anything that
# could move a score across a threshold.
TOLERANCE = 1e-9
PARTS = (("preconditions", "Pre"), ("add_effects", "Add"), ("delete_effects", "Del"))


def keys(operator, part: str) -> list[str]:
    return sorted(term.key for term in getattr(operator, part))


def inverse_goal(operator) -> list[str]:
    return [("not " if t.polarity == "negative" else "") + t.key for t in operator.inverse_target_terms()]


def main() -> None:
    for path in (SCORES_OUT, COMPARISON_OUT):
        if path.exists():
            raise SystemExit(f"{path} exists; outputs are never overwritten")
    saved = json.loads(SAVED.read_text())
    seeds = saved["seeds"]

    env = prim.make_env()
    rollouts = [demonstrate(env, seed) for seed in seeds]
    env.close()

    registry = sp.registry()
    predicates = registry.keys()
    demonstrations = []
    for seed, rollout in zip(seeds, rollouts):
        start, end = registry.evaluate_all(rollout.first()), registry.evaluate_all(rollout.last())
        demonstrations.append({"seed": seed,
                               "start": {p: start[p].score for p in predicates},
                               "end": {p: end[p].score for p in predicates}})

    print(f"{len(seeds)} demonstrations replayed, seeds {seeds}; scores start -> end\n")
    print(f"{'predicate':26s}" + "".join(f"{'seed ' + str(s):>17s}" for s in seeds))
    for p in predicates:
        print(f"{p:26s}" + "".join(f"{d['start'][p]:8.3f} ->{d['end'][p]:6.3f}" for d in demonstrations))

    # Check: the default (mean) extractor on the replay must give back the saved averages.
    mean_result = OperatorExtractor(registry).extract("stow", rollouts)
    differences = {f"{p} {which}": mean_result.scores[p][which] - saved["scores"][p][which]
                   for p in predicates for which in ("start_mean", "end_mean")}
    largest = max(abs(d) for d in differences.values())
    matches = largest <= TOLERANCE
    print(f"\naverages vs {SAVED}: largest difference {largest:.3g} (tolerance {TOLERANCE:g}) -> "
          f"{'MATCH' if matches else 'NO MATCH'}")

    SCORES_OUT.write_text(json.dumps({
        "made_by": MADE_BY,
        "recipe": "demonstrate() from scripts/stow_stage1_extract.py: reset(seed), forward_stow, "
                  "the first and last scene",
        "seeds": seeds,
        "demonstrations": demonstrations,
        "averages_check": {"against": str(SAVED), "tolerance": TOLERANCE, "largest_difference": largest,
                           "matches": matches, "differences": differences},
    }, indent=2))
    print(f"wrote {SCORES_OUT}")
    if not matches:
        raise SystemExit("the replay does not reproduce the saved averages; nothing was compared")

    config = OperatorExtractionConfig(mode="min_max")
    min_max_result = OperatorExtractor(registry, config).extract("stow", rollouts)
    mean_op, min_max_op = mean_result.operator, min_max_result.operator
    same_as_saved = all(keys(mean_op, part) == sorted(t["key"] for t in saved["operator"][part])
                        for part, _ in PARTS)
    true_at, false_at = config.true_threshold, config.false_threshold
    rules = {"Pre": f"start_min >= {true_at}",
             "Add": f"start_max <= {false_at} and end_min >= {true_at}",
             "Del": f"start_min >= {true_at} and end_max <= {false_at}"}

    print(f"\noperator 'stow' (mean on the replay = saved Stage 1 operator: {same_as_saved})")
    found = []
    for part, label in PARTS:
        by_mean, by_min_max = keys(mean_op, part), keys(min_max_op, part)
        print(f"  {label}  mean:    {', '.join(by_mean) or '-'}")
        print(f"  {label}  min/max: {', '.join(by_min_max) or '-'}")
        for p in sorted(set(by_mean) ^ set(by_min_max)):
            found.append({"part": label, "predicate": p,
                          "extracted_by": "mean only" if p in by_mean else "min_max only",
                          "min_max_rule": rules[label],
                          **mean_result.scores[p], **min_max_result.scores[p]})
    print("\ndifferences:" if found else "\ndifferences: none")
    for d in found:
        print(f"  {d['part']} {d['predicate']} ({d['extracted_by']}): needs {d['min_max_rule']}; "
              f"start_min {d['start_min']:.3f}, start_max {d['start_max']:.3f}, "
              f"end_min {d['end_min']:.3f}, end_max {d['end_max']:.3f}")
    goals = {"mean": inverse_goal(mean_op), "min_max": inverse_goal(min_max_op)}
    for name, goal in goals.items():
        print(f"  inverse goal ({name}): {', '.join(goal)}")

    COMPARISON_OUT.write_text(json.dumps({
        "made_by": MADE_BY,
        "demonstration_scores": str(SCORES_OUT),
        "mean_operator_matches_saved": same_as_saved,
        "operators": {"mean": mean_op.to_dict(), "min_max": min_max_op.to_dict()},
        "min_max_scores": min_max_result.scores,
        "differences": found,
        "inverse_goals": goals,
    }, indent=2))
    print(f"wrote {COMPARISON_OUT}")


if __name__ == "__main__":
    main()
