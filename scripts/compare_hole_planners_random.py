"""Both planning modes on many random problems, with every disagreement and its reason.

`compare_hole_planners.py` covers hand-picked scenarios. This covers the differences nobody
picked, including those where the relaxed mode has a plan and the exact mode does not. Planning
only: no simulation, no RL.

It uses the problem generators and the `Oracle` of tests/test_exact_hole_planner.py. The oracle
shares no code with the planner, and here it says why the modes disagree. A second part times both
modes on larger random domains, with a cap on each run. Only the times differ between runs.

Run from the repository root, with PYTHONNOUSERSITE=1 (stray ~/.local packages break ManiSkill):
  PYTHONNOUSERSITE=1 python scripts/compare_hole_planners_random.py [--out results.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import signal
import statistics
import sys
import time
from pathlib import Path

from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.hole_planner import Action, Domain

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from test_exact_hole_planner import Oracle, layered_problem, random_problem

GENERATORS = (("few literals", random_problem, 77), ("chains", layered_problem, 78))
PROBLEMS = 20000
MAX_LENGTH = 5
LARGE_DOMAINS = ((10, 10, 300), (16, 16, 150), (24, 24, 60))    # literals, actions, problems
LARGE_MAX_DELEGATED, LARGE_MAX_LENGTH = 2, 8
TIME_CAP_S = 5
KEPT_IN_FULL = ("plan only in relaxed mode", "different holes", "same holes, other plan")


def category(relaxed, exact) -> str:
    if relaxed and exact:
        if relaxed.actions == exact.actions:
            return "same plan"
        return "same holes, other plan" if set(relaxed.delegated) == set(exact.delegated) else "different holes"
    if relaxed:
        return "plan only in relaxed mode"
    return "plan only in exact mode" if exact else "no plan in either"


def reason(kind: str, oracle: Oracle, relaxed) -> str:
    """Why the exact mode did not return the relaxed plan."""
    holes = set(relaxed.delegated)
    if not oracle.is_minimal(holes):
        return "the goal can be reached without some of the relaxed plan's holes"
    if not oracle.admissible(holes):
        return "the causal preference refuses the relaxed plan's holes"
    if kind == "plan only in relaxed mode":
        return "these holes pass, but the plan fails Used or a hole start"
    return "these holes pass too, and the cost chose between them"


def describe_problem(domain, start, goal_positive, goal_negative, max_delegated) -> dict:
    return {"start": sorted(start), "goal_positive": sorted(goal_positive), "goal_negative": sorted(goal_negative),
            "max_delegated": max_delegated, "max_length": MAX_LENGTH, "actions": [a.to_dict() for a in domain.actions],
            "scored": sorted(domain.scored), "robot_relative": sorted(domain.robot_relative),
            "mutex_groups": [sorted(group) for group in domain.mutex_groups],
            "implications": {key: sorted(value) for key, value in domain.implications.items()},
            "disturbances": [{"trigger": d.trigger, "clobbers": sorted(d.clobbers)} for d in domain.disturbances]}


def describe_case(kind, problem, relaxed, exact) -> dict:
    domain, start, goal_positive, goal_negative, max_delegated = problem
    oracle = Oracle(domain, start, goal_positive, goal_negative)
    case = {"category": kind, "reason": reason(kind, oracle, relaxed),
            "actions_needing_their_own_effect": [a.name for a in domain.actions
                                                 if set(a.pre) & oracle.closure(a.add)],
            "relaxed": {"actions": list(relaxed.actions), "cost": list(relaxed.cost),
                        "capability_gaps": list(relaxed.capability_gaps)},
            "problem": describe_problem(domain, start, goal_positive, goal_negative, max_delegated)}
    if exact:
        case["exact"] = {"actions": list(exact.actions), "cost": list(exact.cost),
                         "tied_with_the_relaxed_plan": relaxed.actions in (exact.actions, *exact.alternative_names)}
    else:
        case["exact"] = {"no_plan": True, "limits_reached": list(exact.limits_reached),
                         "unnecessary_holes": list(exact.unnecessary_holes),
                         "replaceable_holes": list(exact.replaceable_holes),
                         "feasible_with_all_holes": exact.feasible_with_all_holes}
    return case


def compare_on_random_problems(name, make_problem, seed) -> dict:
    rng = random.Random(seed)
    counts = collections.Counter()
    cases = []
    for _ in range(PROBLEMS):
        domain, start, goal_positive, goal_negative = make_problem(rng)
        max_delegated = rng.choice([1, 2])
        limits = {"max_delegated": max_delegated, "max_length": MAX_LENGTH}
        relaxed = hp.plan(domain, frozenset(start), set(goal_positive), set(goal_negative), mode="relaxed", **limits)
        exact = hp.plan(domain, frozenset(start), set(goal_positive), set(goal_negative), mode="exact", **limits)
        kind = category(relaxed, exact)
        counts[kind] += 1
        if kind in KEPT_IN_FULL:
            problem = (domain, start, goal_positive, goal_negative, max_delegated)
            cases.append(describe_case(kind, problem, relaxed, exact))
    return {"generator": name, "seed": seed, "problems": PROBLEMS, "counts": dict(sorted(counts.items())),
            "reasons": dict(sorted(collections.Counter((c["category"], c["reason"]) for c in cases).items())),
            "cases": cases}


def large_problem(rng, literal_count, action_count):
    literals = [f"l{i}" for i in range(literal_count)]
    actions = []
    for i in range(action_count):
        add = frozenset(rng.sample(literals, rng.randint(1, 2)))
        actions.append(Action(f"a{i}", pre=frozenset(rng.sample(literals, rng.randint(0, 3))), add=add,
                              delete=frozenset(rng.sample(literals, rng.randint(0, 2))) - add))
    domain = Domain(actions=actions, scored=frozenset(rng.sample(literals, literal_count // 2)))
    return domain, frozenset(rng.sample(literals, 2)), set(rng.sample(literals, 2))


class TimeUp(Exception):
    pass


def time_up(*_):
    raise TimeUp


def time_mode(problems, mode) -> dict:
    """Medians and maxima are of the runs that finished within the cap."""
    finished = []
    over_cap = 0
    signal.signal(signal.SIGALRM, time_up)
    for domain, start, goal in problems:
        began = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, TIME_CAP_S)
        try:
            hp.plan(domain, start, goal, set(), max_delegated=LARGE_MAX_DELEGATED, max_length=LARGE_MAX_LENGTH, mode=mode)
            finished.append((time.perf_counter() - began) * 1000)
        except TimeUp:
            over_cap += 1
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    return {"median_ms": statistics.median(finished), "max_ms": max(finished), "over_cap": over_cap}


def time_on_large_domains() -> list[dict]:
    rows = []
    for literal_count, action_count, count in LARGE_DOMAINS:
        rng = random.Random(literal_count)
        problems = [large_problem(rng, literal_count, action_count) for _ in range(count)]
        rows.append({"literals": literal_count, "actions": action_count, "problems": count, "time_cap_s": TIME_CAP_S,
                     "relaxed": time_mode(problems, "relaxed"), "exact": time_mode(problems, "exact")})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, help="also write the results as JSON here; never overwrites")
    args = ap.parse_args()
    if args.out and args.out.exists():
        raise SystemExit(f"{args.out} exists; outputs are never overwritten")

    results = [compare_on_random_problems(*generator) for generator in GENERATORS]
    for result in results:
        print(f"\n{result['generator']}: {result['problems']} problems, up to {MAX_LENGTH} steps, seed {result['seed']}")
        for kind, count in result["counts"].items():
            print(f"  {count:6d}  {kind}")
        for (kind, why), count in result["reasons"].items():
            print(f"  {count:6d}  {kind}: {why}")

    large = time_on_large_domains()
    print(f"\nLarger random domains, each run capped at {TIME_CAP_S} s. Medians are of the runs that finished.")
    for row in large:
        for mode in ("relaxed", "exact"):
            print(f"  {row['literals']} literals, {row['actions']} actions, {mode:8s}: median {row[mode]['median_ms']:9.2f} ms, "
                  f"max {row[mode]['max_ms']:8.1f} ms, over the cap {row[mode]['over_cap']}/{row['problems']}")

    if args.out:
        for result in results:
            result["reasons"] = [{"category": kind, "reason": why, "count": count}
                                 for (kind, why), count in result["reasons"].items()]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"made_by": "python scripts/compare_hole_planners_random.py",
                                        "results": results, "large_domains": large}, indent=1))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
