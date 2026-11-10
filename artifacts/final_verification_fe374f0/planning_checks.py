"""Planning-only checks with the minimal planner, on recorded states. No simulation.

1. From every recorded start state. The states the saved Stage 2 and Stage 4 records measured
   after the forward skill, and the ones the fresh runs measured, are planned from again; the
   plan must equal the one recorded beside the state.

2. After the learned skill. The executives do not replan during a plan. This asks what the
   planner would do had they replanned right after the learned step, from what they measured
   there. That state is rebuilt from the fresh Stage 4 records: each executive predicts the
   symbolic state step by step from the measured start, and records the scored literals where
   the measurement disagreed, so the measured state after a step is the prediction with those
   literals flipped (planner-only fluents, such as what the gripper holds, are taken from the
   prediction). The replan must be the remaining library steps, with no hole.

Run through run_with_provenance.py, so every planning call is recorded with its implementation.
Writes planning_checks.json next to this file, never overwriting it.
"""

from __future__ import annotations

import json
from pathlib import Path

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import stow_domain as sd
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.hole_planner import Action

HERE = Path("artifacts/final_verification_fe374f0")
OUT = HERE / "planning_checks.json"


def load(path) -> dict:
    return json.loads(Path(path).read_text())


def names(result) -> list[str] | None:
    return list(result.actions) if result else None


# 1. From every recorded start state

def stage2_rows(path, domain, goal) -> list[dict]:
    rows = []
    for row in load(path)["seeds"]:
        replanned = hp.plan(domain, frozenset(row["state"]), *goal)
        rows.append({"record": str(path), "seed": row["seed"], "state": row["state"],
                     "recorded_plan": row["plan"], "replanned": names(replanned),
                     "same": names(replanned) == row["plan"]})
    return rows


def drawer_stage4_rows(path, goal) -> list[dict]:
    rows = []
    for condition, runs in load(path)["results"].items():
        domain = dd.domain(extra_actions=[dd.scripted_grasp()]) if condition == "scripted_grasp" else dd.domain()
        for run in runs:
            replanned = hp.plan(domain, frozenset(run["measured_state"]), *goal)
            rows.append({"record": str(path), "condition": condition, "seed": run["seed"],
                         "state": run["measured_state"], "recorded_plan": run["plan"],
                         "replanned": names(replanned), "same": names(replanned) == run["plan"]})
    return rows


# 2. After the learned skill

def measured_after(start: frozenset[str], steps: list[Action], reports: list[dict], domain, index: int):
    """The executive's prediction after step `index`, with the literals it found different flipped."""
    predicted = start
    for step in steps[:index + 1]:
        predicted = domain.apply(predicted, step)
    flipped = set(reports[index]["divergence"])
    return frozenset(p for p in predicted if p not in flipped) | frozenset(p for p in flipped if p not in predicted)


def learned_index(steps: list[Action]) -> int | None:
    return next((i for i, step in enumerate(steps)
                 if step.hole_for is not None or step.name.startswith("LEARNED[")), None)


def after_learned(condition: str, seed, domain, start, steps, reports, goal, budget=2) -> dict:
    index = learned_index(steps)
    row = {"condition": condition, "seed": seed}
    if index is None or index >= len(reports) or not reports[index]["ok"]:
        row["checked"] = False
        row["why_not"] = "no learned step" if index is None else "the learned step did not run or did not succeed"
        return row
    state = measured_after(start, steps, reports, domain, index)
    replanned = hp.plan(domain, state, *goal, max_delegated=budget)
    remaining = [step.name for step in steps[index + 1:]]
    row.update(checked=True, learned_step=steps[index].name, disagreed_after_it=reports[index]["divergence"],
               state_after=sorted(state), remaining_steps=remaining, replanned=names(replanned),
               replanned_holes=list(replanned.delegated) if replanned else None,
               same_as_remaining=names(replanned) == remaining)
    return row


def stow_after_learned(fresh_stage4: Path, fresh_learned: Path, operator: Path) -> list[dict]:
    goal = sd.goal_from_operator(operator)
    start = sd.stowed_state()
    record = load(fresh_stage4)
    learned = Action.from_dict(load(fresh_learned)["action"])
    domain, blind, learned_domain = sd.domain(), sd.domain(interference=False), sd.domain(extra_actions=[learned])

    # The same calls stow_stage4_execute.py makes, to get the steps behind the recorded names.
    planned = hp.plan(domain, start, *goal)
    blind_plan = hp.plan(blind, start, *goal)
    other = next(alt for alt in (blind_plan.steps, *blind_plan.alternatives)
                 if tuple(a.name for a in alt) != planned.actions)
    relearned = hp.plan(learned_domain, start, *goal, max_delegated=0)
    conditions = {"full": (list(planned.steps), domain, 2),
                  "without_interference": (list(other), blind, 2),
                  "relearned": (list(relearned.steps), learned_domain, 0)}
    rows = []
    for condition, (steps, cond_domain, budget) in conditions.items():
        if [s.name for s in steps] != record["plans"][condition]:
            raise SystemExit(f"stow {condition}: the steps differ from the recorded plan")
        for run in record["results"][condition]:
            rows.append(after_learned(condition, run["seed"], cond_domain, start, steps, run["steps"], goal, budget))
    return rows


def drawer_after_learned(fresh_stage4: Path, operator: Path) -> list[dict]:
    goal = dd.goal_from_operator(operator)
    domain = dd.domain()
    rows = []
    for condition, runs in load(fresh_stage4)["results"].items():
        if not condition.startswith("learned"):
            continue
        for run in runs:
            start = frozenset(run["measured_state"])
            result = hp.plan(domain, start, *goal)
            if list(result.actions) != run["plan"]:
                raise SystemExit(f"drawer {condition} seed {run['seed']}: the steps differ from the recorded plan")
            rows.append(after_learned(condition, run["seed"], domain, start, list(result.steps), run["steps"], goal))
    return rows


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; results are never overwritten")
    stow_goal = sd.goal_from_operator(HERE / "stow_stage1_operator.json")
    drawer_goal = dd.goal_from_operator(HERE / "drawer_stage1_operator.json")
    if stow_goal != sd.goal_from_operator("artifacts/stow/stage1_operator.json") or \
            drawer_goal != dd.goal_from_operator("artifacts/drawer/stage1_operator.json"):
        raise SystemExit("the fresh forward operators give a different goal from the saved ones")

    starts = []
    for path in ("artifacts/stow/stage2_plan.json", "artifacts/default_mode_validation/after_migration/stow_stage2_plan.json",
                 HERE / "stow_stage2_plan.json"):
        starts += stage2_rows(path, sd.domain(), stow_goal)
    for path in ("artifacts/drawer/stage2_plan.json", "artifacts/default_mode_validation/after_migration/drawer_stage2_plan.json",
                 HERE / "drawer_stage2_plan.json"):
        starts += stage2_rows(path, dd.domain(), drawer_goal)
    for path in ("artifacts/drawer/stage4_execution.json", "artifacts/drawer/stage4_execution_after_push_fix.json",
                 "artifacts/default_mode_validation/after_migration/drawer_stage4.json", HERE / "drawer_stage4.json"):
        starts += drawer_stage4_rows(path, drawer_goal)

    after = (stow_after_learned(HERE / "stow_stage4.json", HERE / "stow_learned_operator.json", HERE / "stow_stage1_operator.json")
             + drawer_after_learned(HERE / "drawer_stage4.json", HERE / "drawer_stage1_operator.json"))

    checked = [row for row in after if row["checked"]]
    summary = {
        "from_recorded_starts": {"states": len(starts), "same_plan": sum(r["same"] for r in starts),
                                 "distinct_states": len({json.dumps(r["state"]) for r in starts}),
                                 "differ": [r for r in starts if not r["same"]]},
        "after_the_learned_step": {"runs": len(after), "checked": len(checked),
                                   "replan_is_the_remaining_steps": sum(r["same_as_remaining"] for r in checked),
                                   "replans_with_a_hole": sum(bool(r["replanned_holes"]) for r in checked),
                                   "measured_differs_from_predicted": sum(bool(r["disagreed_after_it"]) for r in checked),
                                   "not_checked": [r for r in after if not r["checked"]],
                                   "differ": [r for r in checked if not r["same_as_remaining"]]},
    }
    OUT.write_text(json.dumps({"summary": summary, "from_recorded_starts": starts, "after_the_learned_step": after}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
