"""Round 1 symbolic regression: replan StowCube and SlidingDrawer from the recorded Stage 2 states
with the corrected planner, and compare everything with the recorded values. No simulation runs,
and nothing outside this folder is written.

    PYTHONNOUSERSITE=1 python artifacts/round1_correctness_20261111/regression.py

  recorded   artifacts/signed_literals_20261111/E2_stage2 (plans, specs, ablations, the direct
             spec) and E4_stage4 (the measured state of every replanned Stow episode)
  corrected  this checkout: transitions must give valid states; contexts come only from
             library states where the hole's literal does not hold
  previous   the planner at ca2d029, run in its own process from an extracted copy, for the
             reachable states and the contexts it gave

Writes regression.json next to this file, never overwriting it, and prints a summary.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path("artifacts/round1_correctness_20261111")
OUT = HERE / "regression.json"
RECORDED = Path("artifacts/signed_literals_20261111")
E2 = RECORDED / "E2_stage2"
PREVIOUS = "ca2d029"

# The part both checkouts run: reachable states and contexts, from the recorded start states.
PROBE = r'''
import json, sys
from inverse_skills.envs import drawer_domain as dd, stow_domain as sd
from inverse_skills.operators.exact_hole_planner import hole_starts, library_states, reachable_states, _HoleSets
from inverse_skills.operators.literals import join, pos, neg

problems = json.loads(sys.argv[1])
out = {}
for name, problem in problems.items():
    domain = sd.domain() if name == "stow" else dd.domain()
    start = frozenset(problem["start"])
    goal = join(problem["goal_pos"], problem["goal_neg"])
    library = library_states(domain, start)
    candidates = sorted(_HoleSets(domain, start, goal, 2).candidates)
    contexts = {str(l): [sorted(h.pre) for h in hole_starts(domain, l, library)] for l in candidates}
    every = {str(l): [sorted(h.pre) for h in hole_starts(domain, l, library)]
             for key in sorted(domain.scored) for l in (pos(key), neg(key))}
    holes = [h for l in candidates for h in hole_starts(domain, l, library)]
    out[name] = {"library": sorted(sorted(s) for s in library),
                 "with_candidate_holes": sorted(sorted(s) for s in reachable_states(domain, start, holes)),
                 "candidate_contexts": contexts, "every_context": every}
print(json.dumps(out))
'''


def load(path) -> dict:
    return json.loads(Path(path).read_text())


def probe(problems: dict, src: Path | None) -> dict:
    env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    if src is not None:
        env["PYTHONPATH"] = str(src)
    code = PROBE if src is None else f"import inverse_skills; assert inverse_skills.__file__.startswith({str(src)!r})\n" + PROBE
    done = subprocess.run([sys.executable, "-c", code, json.dumps(problems)], capture_output=True, text=True, env=env)
    if done.returncode:
        raise SystemExit(done.stderr[-2000:])
    return json.loads(done.stdout.strip().splitlines()[-1])


def previous_checkout() -> dict:
    with tempfile.TemporaryDirectory() as folder:
        archive = subprocess.run(["git", "archive", PREVIOUS, "src"], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", "-C", folder], input=archive, check=True)
        return probe(problems(), Path(folder) / "src")


def problems() -> dict:
    out = {}
    for name in ("stow", "drawer"):
        record = load(E2 / f"{name}_stage2_plan.json")
        out[name] = {"start": record["seeds"][0]["state"], "goal_pos": record["goal_positive"],
                     "goal_neg": record["goal_negative"]}
    return out


def literal_dicts(literals) -> list[dict]:
    return [l.to_dict() for l in literals]


def replanned(name: str) -> dict:
    """Stage 2 again from every recorded seed state, with the corrected planner: the fields the
    Stage 2 record holds, compared field by field."""
    from inverse_skills.envs import drawer_domain as dd
    from inverse_skills.envs import stow_domain as sd
    from inverse_skills.operators import hole_planner as hp

    record = load(E2 / f"{name}_stage2_plan.json")
    goal_pos, goal_neg = set(record["goal_positive"]), set(record["goal_negative"])
    domain = sd.domain() if name == "stow" else dd.domain()
    fields = ("plan", "candidates", "delegated", "hole_indices", "cost", "planning_cost", "ties", "hole_contexts",
              "replay_problems")
    seeds = []
    for row in record["seeds"]:
        state = frozenset(row["state"])
        result = hp.plan(domain, state, goal_pos, goal_neg)
        now = {"plan": list(result.actions), "candidates": literal_dicts(result.candidates),
               "delegated": literal_dicts(result.delegated), "hole_indices": list(result.hole_indices),
               "cost": list(result.cost), "planning_cost": list(result.planning_cost),
               "ties": [list(a) for a in result.alternative_names],
               "hole_contexts": [c.to_dict() for c in result.hole_contexts],
               "replay_problems": hp.replay_problems(domain, state, result.steps, goal_pos, goal_neg)}
        seeds.append({"seed": row["seed"], "violations": domain.violations(state),
                      "differs": [f for f in fields if now[f] != row[f]],
                      "no_context": [str(l) for l in result.no_context]})

    start = frozenset(record["seeds"][0]["state"])
    result = hp.plan(domain, start, goal_pos, goal_neg)
    spec, recorded_spec = hp.hole_spec(domain, start, result, goal_pos, goal_neg), load(E2 / f"{name}_hole_spec.json")
    spec_differs = [key for key in recorded_spec if spec.get(key) != recorded_spec[key]]

    ablations = {}
    if name == "stow":
        blind = hp.plan(sd.domain(interference=False), start, goal_pos, goal_neg)
        library_only = hp.plan(domain, start, goal_pos, goal_neg, max_delegated=0)
        ablations["without_interference_model"] = (
            [list(blind.actions), *map(list, blind.alternative_names)] == record["without_interference_model"])
        ablations["library_only"] = (literal_dicts(library_only.unmet_needs) == record["library_only"]["unmet_needs"]
                                     and record["library_only"]["plan"] is None and not library_only)
    else:
        def summary(result):
            return {"plan": list(result.actions) if result else None,
                    "cost": list(result.cost) if result else None,
                    "delegated": literal_dicts(result.delegated) if result else []}
        now = {"library_only": hp.plan(domain, start, goal_pos, goal_neg, max_delegated=0),
               "frontier": hp.plan(domain, start, goal_pos, goal_neg),
               "unrestricted": hp.plan(domain, start, goal_pos, goal_neg, mode="relaxed", unrestricted_holes=True),
               "with_grasp": hp.plan(dd.domain(extra_actions=[dd.scripted_grasp()]), start, goal_pos, goal_neg)}
        for key, ablation in now.items():
            was = record["ablations"][key]
            ablations[key] = all(summary(ablation)[f] == was.get(f, summary(ablation)[f]) for f in ("plan", "cost", "delegated"))
        ablations["library_only_unmet_needs"] = (literal_dicts(now["library_only"].unmet_needs)
                                                 == record["ablations"]["library_only"]["unmet_needs"])
    return {"seeds": seeds, "spec_fields_that_differ": spec_differs, "ablations_unchanged": ablations,
            "no_context": [str(l) for l in result.no_context]}


def direct_protocol() -> dict:
    """The direct baseline's protocol plan: approach_handle, then HOLE[drawer_open] from the
    context the corrected rule gives, and the fence the plan derives."""
    from inverse_skills.envs import drawer_domain as dd
    from inverse_skills.operators import hole_planner as hp
    from inverse_skills.operators.exact_hole_planner import hole_starts, library_states
    from inverse_skills.operators.literals import pos

    record = load(E2 / "drawer_stage2_plan.json")
    spec = load(E2 / "direct" / "hole_spec_direct.json")
    domain, start = dd.domain(), frozenset(record["seeds"][0]["state"])
    holes = hole_starts(domain, pos(dd.OPEN), library_states(domain, start))
    steps = [next(a for a in domain.actions if a.name == "approach_handle(drawer)"), holes[0]]
    kept_true, kept_false = hp.derive_fences(domain, start, steps, 1, set(record["goal_positive"]),
                                             set(record["goal_negative"]))
    return {"contexts": [sorted(h.pre) for h in holes], "recorded_start": spec["start"],
            "context_unchanged": [sorted(h.pre) for h in holes] == [sorted(spec["start"])],
            "fences": {"kept_true": kept_true, "kept_false": kept_false},
            "fences_unchanged": (kept_true == spec["preserve"] and kept_false == spec["preserve_absent"])}


def stage4_states() -> dict:
    """Every state the recorded Stow Stage 4 replanned from: would the corrected planner refuse it?"""
    from inverse_skills.envs import stow_domain as sd

    record = load(RECORDED / "E4_stage4" / "stow_stage4.json")
    domain = sd.domain()
    states = [frozenset(row["measured_state"]) for rows in record["results"].values() for row in rows
              if "measured_state" in row]
    return {"replanned_episodes": len(states), "refused": sum(bool(domain.violations(s)) for s in states)}


def invalid_now(name: str, previous: dict) -> list[str]:
    """Transitions the previous model admitted from its reachable states that the corrected one
    rejects, with candidate holes at their previous contexts."""
    from inverse_skills.envs import drawer_domain as dd
    from inverse_skills.envs import stow_domain as sd
    from inverse_skills.operators.literals import Literal

    domain = sd.domain() if name == "stow" else dd.domain()
    holes = [domain.hole(Literal(text.removeprefix("not "), not text.startswith("not ")), context)
             for text, contexts in previous["candidate_contexts"].items() for context in contexts]
    found = []
    for state in map(frozenset, previous["with_candidate_holes"]):
        for action in [*domain.actions, *holes]:
            if domain.can_run(state, action) and domain.transition_problems(state, action):
                found.append(f"{action.name} from {sorted(state)}: {'; '.join(domain.transition_problems(state, action))}")
    return found


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; it is never overwritten")
    before, after = previous_checkout(), probe(problems(), None)
    results = {"previous_commit": PREVIOUS,
               "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
               "tracked_changes": subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                                 capture_output=True, text=True).stdout.splitlines()}
    for name in ("stow", "drawer"):
        was, now = before[name]["every_context"], after[name]["every_context"]
        results[name] = {
            "library_states": {"previous": len(before[name]["library"]), "corrected": len(after[name]["library"]),
                               "same": before[name]["library"] == after[name]["library"]},
            "states_with_candidate_holes": {"previous": len(before[name]["with_candidate_holes"]),
                                            "corrected": len(after[name]["with_candidate_holes"]),
                                            "same": before[name]["with_candidate_holes"] == after[name]["with_candidate_holes"]},
            "transitions_now_invalid": invalid_now(name, before[name]),
            "candidate_contexts": {"previous": before[name]["candidate_contexts"],
                                   "corrected": after[name]["candidate_contexts"]},
            "contexts_changed": {key: {"previous": was[key], "corrected": now[key]}
                                 for key in now if was[key] != now[key]},
            **replanned(name)}
    results["direct_protocol"] = direct_protocol()
    results["stow_stage4_states"] = stage4_states()
    OUT.write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k not in ("stow", "drawer")}, indent=1))
    for name in ("stow", "drawer"):
        r = results[name]
        print(f"\n{name}: library states {r['library_states']}; with candidate holes {r['states_with_candidate_holes']}")
        print(f"  transitions now invalid: {len(r['transitions_now_invalid'])}")
        print(f"  seeds differing: {[s['differs'] for s in r['seeds']]}; violations: {[s['violations'] for s in r['seeds']]}")
        print(f"  spec fields that differ: {r['spec_fields_that_differ']}; ablations unchanged: {r['ablations_unchanged']}")
        print(f"  no context: {r['no_context']}")
        print(f"  contexts changed (any scored literal): {json.dumps(r['contexts_changed'])}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
