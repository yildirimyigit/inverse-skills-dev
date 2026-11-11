"""E5: the previous system against the updated one, from recorded files only. Nothing is rerun.

    PYTHONNOUSERSITE=1 python artifacts/signed_literals_20261111/compare.py

  previous   the published pipeline (artifacts/stow, artifacts/drawer): positive-only extraction
             and goal, the 0.8 threshold for every check
  updated    this folder's fresh runs, E1 to E4: signed literals, Boolean planning state, robust
             physical checks
  re-judged  the previous runs' recorded final scores read against the updated signed goal,
             robustly. A re-reading of recorded numbers, not a rerun: it shows what the changed
             goal alone does to the old results

Writes E5_comparison/comparison.json and comparison.md, never overwriting either.
"""

from __future__ import annotations

import json
from pathlib import Path

from inverse_skills.operators.literals import Literal, join, robustly_unmet

HERE = Path("artifacts/signed_literals_20261111")
OUT = HERE / "E5_comparison"
E1, E2, E3, E4 = (HERE / n for n in ("E1_stage1", "E2_stage2", "E3_stage3", "E4_stage4"))


def load(path) -> dict:
    return json.loads(Path(path).read_text())


def signed(terms) -> list[str]:
    """Operator terms (with a polarity field) as "p" / "not p"."""
    return [str(Literal(t["key"], t["polarity"] == "positive")) for t in terms]


def effects(operator: dict) -> list[str]:
    return sorted([t["key"] for t in operator["add_effects"]] + [f"not {t['key']}" for t in operator["delete_effects"]])


def literal_names(dicts) -> list[str]:
    return [str(Literal.from_dict(d)) for d in dicts]


def judged(rows, goal) -> dict:
    """How many recorded episodes end with every literal of `goal` robustly met."""
    literals = join(*goal)
    finals = [row.get("final_scores") or (row.get("report") or {}).get("final_scores") for row in rows]
    met = [bool(f) and not robustly_unmet(literals, f) for f in finals]
    unmet = sorted({str(l) for f in finals if f for l in robustly_unmet(literals, f)})
    return {"met": sum(met), "of": len(met), "literals_ever_unmet": unmet}


def stow() -> dict:
    old_op, new_op = load("artifacts/stow/stage1_operator.json")["operator"], load(E1 / "stow_stage1_operator.json")["operator"]
    old_plan, new_plan = load("artifacts/stow/stage2_plan.json"), load(E2 / "stow_stage2_plan.json")
    old_spec, new_spec = load("artifacts/stow/hole_spec.json"), load(E2 / "stow_hole_spec.json")
    old_sac = load("artifacts/stow/stage3/1790788534/summary.json")
    new_sac = load(next((E3 / "stow").glob("*/summary.json")))
    old_s4, new_s4 = load("artifacts/stow/stage4_execution.json"), load(E4 / "stow_stage4.json")
    new_goal = (set(new_plan["goal_positive"]), set(new_plan["goal_negative"]))
    return {
        "signed_preconditions": {"previous": signed(old_op["preconditions"]), "updated": signed(new_op["preconditions"])},
        "signed_effects": {"previous": effects(old_op), "updated": effects(new_op)},
        "restoration_goal": {"previous": signed(old_op["inverse_target_terms"]), "updated": literal_names(new_plan["goal"])},
        "missing_capabilities": {"previous": old_plan["seeds"][0]["delegated"],
                                 "updated": literal_names(new_plan["seeds"][0]["delegated"])},
        "symbolic_plan": {"previous": old_plan["seeds"][0]["plan"], "updated": new_plan["seeds"][0]["plan"]},
        "initiation_context": {"previous": old_spec["start"], "updated": new_spec["start"] + [f"not {a}" for a in new_spec["start_absent"]]},
        "preservation": {"previous": old_spec["preserve"] + [f"not {a}" for a in old_spec["preserve_absent"]],
                         "updated": literal_names(new_spec["fences"])},
        "sac_held_out": {
            "previous": {"seeds": "1000-1009", "success": old_sac["best"]["success_rate"], "fences": old_sac["best"]["fence_rate"],
                         "contract": old_sac["best"]["contract_rate"], "timesteps": old_sac["timesteps"]},
            "updated": {"seeds": "1000-1009", "success": new_sac["best"]["success_rate"], "fences": new_sac["best"]["fence_rate"],
                        "contract": new_sac["best"]["contract_rate"], "timesteps": new_sac["timesteps"]}},
        "full_execution": {
            "previous": {name: s["goal_met_rate"] for name, s in old_s4["summary"].items()},
            "updated": {name: s["goal_met_rate"] for name, s in new_s4["summary"].items()},
            "updated_boolean": {name: s["goal_met_rate_boolean"] for name, s in new_s4["summary"].items()},
            "previous_rejudged_by_the_signed_goal": {name: judged(rows, new_goal) for name, rows in old_s4["results"].items()}},
        "model_world_discrepancies": {"previous": {n: s["diverging_steps"] for n, s in old_s4["summary"].items()},
                                      "updated": {n: s["diverging_steps"] for n, s in new_s4["summary"].items()}},
        "learned_operator": {"previous": old_s4["learned_action"], "updated": new_s4["learned_action"],
                             "updated_effects_contain_the_target": new_s4["learned_effects_contain_the_target"]},
    }


def drawer() -> dict:
    old_op, new_op = load("artifacts/drawer/stage1_operator.json")["operator"], load(E1 / "drawer_stage1_operator.json")["operator"]
    old_plan, new_plan = load("artifacts/drawer/stage2_plan.json"), load(E2 / "drawer_stage2_plan.json")
    old_spec, new_spec = load("artifacts/drawer/hole_spec.json"), load(E2 / "drawer_hole_spec.json")
    old_runs = sorted(Path("artifacts/drawer/stage3").glob("*/summary.json"))
    new_runs = sorted((E3 / "drawer").glob("*/summary.json"))

    def held_out(paths):
        rows = {}
        for path in paths:
            seed = load(path.parent / "config.json")["seed"]
            test = load(path)["test"]["best_checkpoint"]
            rows[f"SAC seed {seed}"] = {"grasp": test["success_rate"], "pull_open_opens": test["pull_open_rate"]}
        return rows

    old_s4, new_s4 = load("artifacts/drawer/stage4_execution_after_push_fix.json"), load(E4 / "drawer_stage4.json")
    old_final, new_final = load("artifacts/drawer/final_targeted_vs_direct.json"), load(E4 / "final_targeted_vs_direct.json")
    new_goal = (set(new_plan["goal_positive"]), set(new_plan["goal_negative"]))
    return {
        "signed_preconditions": {"previous": signed(old_op["preconditions"]), "updated": signed(new_op["preconditions"])},
        "signed_effects": {"previous": effects(old_op), "updated": effects(new_op)},
        "restoration_goal": {"previous": signed(old_op["inverse_target_terms"]), "updated": literal_names(new_plan["goal"])},
        "missing_capabilities": {"previous": old_plan["seeds"][0]["delegated"],
                                 "updated": literal_names(new_plan["seeds"][0]["delegated"])},
        "symbolic_plan": {"previous": old_plan["seeds"][0]["plan"], "updated": new_plan["seeds"][0]["plan"]},
        "initiation_context": {"previous": old_spec["start"], "updated": new_spec["start"] + [f"not {a}" for a in new_spec["start_absent"]]},
        "preservation": {"previous": old_spec["preserve"] + [f"not {a}" for a in old_spec["preserve_absent"]],
                         "updated": literal_names(new_spec["fences"])},
        "sac_held_out": {"previous": held_out(old_runs), "updated": held_out(new_runs), "seeds": "1000-1019"},
        "full_execution": {
            "previous": {name: f"{s['goal_met']}/{len(old_s4['seeds'])}" for name, s in old_s4["summary"].items()},
            "updated": {name: f"{s['goal_met']}/{len(new_s4['seeds'])}" for name, s in new_s4["summary"].items()},
            "updated_boolean": {name: f"{s['goal_met_boolean']}/{len(new_s4['seeds'])}" for name, s in new_s4["summary"].items()},
            "previous_rejudged_by_the_signed_goal": {name: judged(rows, new_goal) for name, rows in old_s4["results"].items()}},
        "model_world_discrepancies": {"previous": {n: s["model_vs_world_differences"] for n, s in old_s4["summary"].items()},
                                      "updated": {n: s["model_vs_world_differences"] for n, s in new_s4["summary"].items()}},
        "targeted_vs_direct": {
            "previous": {n: f"{s['success']}/{s['configurations']}" for n, s in old_final["summary"].items()},
            "updated": {n: f"{s['success']}/{s['configurations']}" for n, s in new_final["summary"].items()},
            "updated_policy_target": {n: f"{s['policy_target_achieved']}/{s['configurations']}" for n, s in new_final["summary"].items()},
            "previous_rejudged_by_the_signed_goal": {n: judged(rows, new_goal) for n, rows in old_final["results"].items()},
            "updated_failures": {n: s["failures"] for n, s in new_final["summary"].items()}},
    }


def table(results: dict) -> str:
    def cell(value) -> str:
        if isinstance(value, list):
            return ", ".join(value) if value else "none"
        if isinstance(value, dict):
            return "; ".join(f"{k}: {cell(v)}" for k, v in value.items())
        return str(value)
    rows = ["| Property | StowCube previous | StowCube updated | SlidingDrawer previous | SlidingDrawer updated |",
            "|---|---|---|---|---|"]
    names = [("Signed preconditions", "signed_preconditions"), ("Signed effects", "signed_effects"),
             ("Restoration goal", "restoration_goal"), ("Missing capabilities", "missing_capabilities"),
             ("Symbolic plan", "symbolic_plan"), ("Initiation context", "initiation_context"),
             ("Preservation conditions", "preservation"), ("SAC held-out success", "sac_held_out"),
             ("Full physical execution", "full_execution"), ("Model/world discrepancies", "model_world_discrepancies")]
    for label, key in names:
        s, d = results["stow"][key], results["drawer"][key]
        rows.append(f"| {label} | {cell(s.get('previous'))} | {cell(s.get('updated'))} | "
                    f"{cell(d.get('previous'))} | {cell(d.get('updated'))} |")
    return "\n".join(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ("comparison.json", "comparison.md"):
        if (OUT / name).exists():
            raise SystemExit(f"{OUT / name} exists; it is never overwritten")
    results = {"stow": stow(), "drawer": drawer()}
    (OUT / "comparison.json").write_text(json.dumps(results, indent=1))
    (OUT / "comparison.md").write_text("# Previous against updated\n\n" + table(results) + "\n")
    print(table(results))


if __name__ == "__main__":
    main()
