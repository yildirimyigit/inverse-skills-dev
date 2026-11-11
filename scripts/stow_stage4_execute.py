"""Stage 4: run the whole inverse on the robot, and the ablations beside it.

Before the conditions run, the learned skill joins the library the same way the
forward skill entered the framework: its own executions are handed to the
operator extractor, and what comes back — preconditions, add and delete
effects — becomes a library action. Nothing about it is written by hand.

An execution counts as a demonstration of that operator only if it starts at the
handoff the plan predicts, which is checked before the policy runs (see
`operators/handoff.py`). A scene that fails the check is replaced by the next one
and recorded. What the policy then does never decides whether a run is kept.

Conditions, on the same seeds:

  full                  the planned inverse, the hole filled by the learned skill;
                        planned in every episode from that episode's measured state
  library_only          no plan exists; the hole is skipped to show where it breaks
  without_interference  the ordering a planner without the interference model
                        cannot tell apart from the chosen one — the two tie
  relearned             re-planned, in every episode from its measured state, once
                        the learned skill is a library action (no holes allowed)

The goal is the signed restoration goal of the Stage 1 operator, every literal
judged robustly (at least 0.8 for p, at most 0.2 for not p); the Boolean verdict
is recorded beside it. The library includes retreat_to_safe, and the planner
decides where it goes; nothing appends it.

Every run is monitored: the executive reports each scored literal where the
measured scene disagrees with what that condition's domain predicts.

An inverse is only defined when the forward skill actually happened, so each
episode must start where Stage 2 planned from: the forward operator's signed
effects robustly hold (a physical check, like every other), and the measured
Boolean state is the one in the Stage 2 record. When the forward push falls short
(it occasionally leaves A outside the pocket's seat, or B half in the mouth), the
scene is re-rolled, as the hole environment does for its handoffs, and the re-roll
is reported rather than hidden.

How often a scene really gives a valid handoff is reported apart from the
conditions' success, both for the demonstrations and on the evaluation seeds. The
evaluation does not re-roll for it: those seeds run as they always have.

Writes artifacts/stow/learned_operator.json and artifacts/stow/stage4_execution.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from inverse_skills.envs import stow_domain as sd
from inverse_skills.envs import stow_executive as ex
from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs import stow_primitives as prim
from inverse_skills.envs.stow_hole_env import load_spec
from inverse_skills.logging.rollout import ForwardRollout
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.extractor import OperatorExtractor
from inverse_skills.operators.handoff import (
    Prepared,
    collect_demonstrations,
    handoff_conditions,
    handoff_violations,
    summarize_attempts,
)
from inverse_skills.operators.hole_planner import Action, spec_target
from inverse_skills.operators.literals import join, robustly_unmet

torch.set_num_threads(1)

EXTRACTION_SEEDS = tuple(range(2000, 2010))   # disjoint from the evaluation seeds
_MAX_FORWARD_ATTEMPTS = 5


def measured_scores(obs, regions) -> dict[str, float]:
    return {k: float(r.score) for k, r in sp.registry().evaluate_all(prim.obs_to_scene(obs, regions)).items()}


def measured(obs, regions) -> frozenset[str]:
    return hp.symbolic_state(measured_scores(obs, regions))


def forward_once(env, scene_seed: int):
    """The forward skill in one scene. Returns the observation and regions."""
    obs = prim.reset(env, scene_seed)
    regions = prim.regions(obs)
    return prim.forward_stow(env, obs), regions


def forward_effects(operator_path: Path):
    """The extracted forward operator's signed effects."""
    data = json.loads(Path(operator_path).read_text())["operator"]
    return join((t["key"] for t in data["add_effects"]), (t["key"] for t in data["delete_effects"]))


def not_stowed(obs, regions, expected: frozenset[str], forward) -> list[str]:
    """Why this scene is not the one the plan starts from. The forward skill happened when its
    signed effects robustly hold, a physical check like every other; and the plan applies when
    the Boolean state is the one Stage 2 planned from. An empty list: the scene is stowed."""
    scores = measured_scores(obs, regions)
    reasons = [f"forward effect {l} not robustly measured ({scores[l.atom]:.2f})"
               for l in robustly_unmet(forward, scores)]
    state = hp.symbolic_state(scores)
    if state != expected:
        reasons.append("differs from the planned start: " + ", ".join(sorted(state ^ expected)))
    return reasons


def stowed(env, seed: int, expected: frozenset[str], forward):
    """The scene the forward skill leaves for this seed, re-rolled until the forward skill
    happened and left the state the plan starts from. Returns the observation, regions and
    re-rolls."""
    for attempt in range(_MAX_FORWARD_ATTEMPTS):
        obs, regions = forward_once(env, seed + attempt * 10_000)
        if not not_stowed(obs, regions, expected, forward):
            return obs, regions, attempt
    raise RuntimeError(f"seed {seed}: the forward skill never produced the planned start")


def run_prefix(env, obs, prefix: list[Action], regions):
    """The known prefix, checked by the executive. Returns the observation and why it failed, if it did."""
    obs, report = ex.execute_plan(env, obs, prefix, regions=regions)
    failed = next((s for s in report.steps if not s.ok), None)
    if failed is not None:
        return obs, f"{failed.action}: {failed.detail}"
    return obs, None if len(report.steps) == len(prefix) else f"stopped after {len(report.steps)} steps"


def learn_operator(env, model, spec: dict, seeds, planned_start, domain, forward) -> tuple[Action, dict]:
    """Model the learned skill from its own executions, as a demonstration would be.

    Each execution starts at a valid handoff, checked before the policy runs. A scene that
    fails, in the forward skill, the prefix or the handoff, is replaced by the next and recorded.
    """
    prefix = [Action.from_dict(a) for a in spec["prefix"]]
    registry = sp.registry()
    target = spec_target(spec)
    name = f"learned:{target}"
    conditions = handoff_conditions(domain, planned_start, spec)

    def prepare(scene_seed: int) -> Prepared:
        obs, regions = forward_once(env, scene_seed)
        reasons = not_stowed(obs, regions, planned_start, forward)
        if reasons:
            return Prepared(failed_stage="forward", reason="; ".join(reasons))
        obs, failure = run_prefix(env, obs, prefix, regions)
        if failure:
            return Prepared(failed_stage="prefix", reason=failure)
        return Prepared(scores=measured_scores(obs, regions), context=(obs, regions))

    def demonstrate(scene_seed: int, context) -> ForwardRollout:
        obs, regions = context
        before = prim.obs_to_scene(obs, regions)
        obs = ex.run_learned(env, obs, model, target, regions, registry)
        after = prim.obs_to_scene(obs, regions, timestep=1)
        return ForwardRollout(skill_name=name, demo_id=f"run_{scene_seed}", scenes=[before, after])

    rollouts, attempts = collect_demonstrations(seeds, conditions, prepare, demonstrate)
    extracted = OperatorExtractor(registry).extract(name, rollouts)
    action = hp.learned_action(target, extracted.operator)
    return action, {"rollouts": len(rollouts), "seeds": list(seeds),
                    "scene_seeds": [a.scene_seed for a in attempts if a.stage == "valid"],
                    "handoff_conditions": conditions.to_dict(),
                    "handoff_attempts": [a.to_dict() for a in attempts],
                    "handoff_summary": summarize_attempts(attempts),
                    "operator": extracted.operator.to_dict(), "scores": extracted.scores,
                    "action": action.to_dict()}


def survey_handoffs(env, spec: dict, seeds, planned_start, domain, forward) -> dict:
    """Would each seed pass the handoff check? Measured only: no policy runs and nothing is
    re-rolled for it, so this is the raw rate behind the conditions' results."""
    prefix = [Action.from_dict(a) for a in spec["prefix"]]
    conditions = handoff_conditions(domain, planned_start, spec)
    rows = []
    for seed in seeds:
        obs, regions, rerolls = stowed(env, seed, planned_start, forward)
        obs, failure = run_prefix(env, obs, prefix, regions)
        scores = measured_scores(obs, regions)
        violations = [failure] if failure else handoff_violations(conditions, scores)
        rows.append({"seed": seed, "forward_rerolls": rerolls, "valid": not violations, "violations": violations,
                     "scores": {p: scores[p] for p in sorted(conditions.hold | conditions.absent)}})
    return {"conditions": conditions.to_dict(), "seeds": rows, "valid": sum(row["valid"] for row in rows), "of": len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")],
                    default=list(range(1000, 1010)))
    ap.add_argument("--spec", type=Path, default=Path("artifacts/stow/hole_spec.json"))
    ap.add_argument("--operator", type=Path, default=Path("artifacts/stow/stage1_operator.json"))
    ap.add_argument("--plan", type=Path, default=Path("artifacts/stow/stage2_plan.json"),
                    help="the Stage 2 record: its measured state is the one every episode must start in")
    ap.add_argument("--out", type=Path, default=Path("artifacts/stow/stage4_execution.json"))
    ap.add_argument("--learned-out", type=Path, default=Path("artifacts/stow/learned_operator.json"))
    args = ap.parse_args()
    for path in (args.out, args.learned_out):
        if path.exists():
            raise SystemExit(f"{path} exists; saved results are never overwritten. Choose another --out and --learned-out.")

    model = SAC.load(args.checkpoint, device="cuda")
    spec = load_spec(args.spec)
    target = spec_target(spec)
    goal = sd.goal_from_operator(args.operator)
    forward = forward_effects(args.operator)
    stage2 = json.loads(args.plan.read_text())
    starts = {frozenset(row["state"]) for row in stage2["seeds"]}
    if len(starts) != 1:
        raise SystemExit("the Stage 2 seeds measured different states; there is no single state to start from")
    start = next(iter(starts))
    env = prim.make_env()

    domain = sd.domain()
    planned = hp.plan(domain, start, *goal)
    if list(planned.actions) != spec["plan"]:
        raise SystemExit("the plan differs from the one the policy was trained for; rerun stage 2")
    blind_domain = sd.domain(interference=False)
    blind = hp.plan(blind_domain, start, *goal)
    other = next(alt for alt in (blind.steps, *blind.alternatives)
                 if tuple(a.name for a in alt) != planned.actions)

    print(f"learning the model of LEARNED[{target}] from its executions "
          f"on seeds {EXTRACTION_SEEDS[0]}-{EXTRACTION_SEEDS[-1]}")
    learned, learned_record = learn_operator(env, model, spec, EXTRACTION_SEEDS, start, domain, forward)
    print(f"  handoff must have true: {', '.join(learned_record['handoff_conditions']['hold'])}; "
          f"false: {', '.join(learned_record['handoff_conditions']['absent'])}")
    for attempt in learned_record["handoff_attempts"]:
        if attempt["stage"] != "valid":
            print(f"  seed {attempt['seed']}, scene {attempt['scene_seed']}: {attempt['stage']} failed, "
                  f"replaced: {', '.join(attempt['reasons'])}")
    print(f"  raw handoff validity: {learned_record['handoff_summary']['valid_handoffs']} of "
          f"{learned_record['handoff_summary']['reached_the_handoff']} scenes that reached the handoff")
    print(f"  from {learned_record['rollouts']} runs:")
    print(f"    pre   {', '.join(str(l) for l in sorted(learned.precondition_literals))}")
    print(f"    eff   {', '.join(str(l) for l in sorted(learned.effect_literals))}")
    print(f"    the extracted effects {'contain' if target in learned.effect_literals else 'do NOT contain'} "
          f"the target {target}")
    args.learned_out.write_text(json.dumps(learned_record, indent=2))
    learned_domain = sd.domain(extra_actions=[learned])
    relearned = hp.plan(learned_domain, start, *goal, max_delegated=0)
    replanned_in = {"full": domain, "relearned": learned_domain}   # planned again in every episode

    conditions = {
        "full": (list(planned.steps), domain),
        "library_only": ([s for s in planned.steps if s.hole_for is None], domain),
        "without_interference": (list(other), blind_domain),
        "relearned": (list(relearned.steps) if relearned else [], learned_domain),
    }
    print("\nplans")
    for name, (steps, _) in conditions.items():
        print(f"  {name:21s}", " -> ".join(s.name for s in steps) or "no plan")

    results = {name: [] for name in conditions}
    for name, (steps, cond_domain) in conditions.items():
        print(f"\n{name}")
        for seed in args.seeds:
            obs, regions, rerolls = stowed(env, seed, start, forward)
            row = {"seed": seed, "forward_rerolls": rerolls}
            if name in replanned_in:
                # Forward skill -> measured state -> exact planning, in this episode.
                state = measured(obs, regions)
                episode_plan = hp.plan(replanned_in[name], state, *goal,
                                       max_delegated=0 if name == "relearned" else 2)
                steps = list(episode_plan.steps) if episode_plan else []
                row.update(measured_state=sorted(state), plan=[s.name for s in steps],
                           plan_as_planned_at_the_start=[s.name for s in steps] == [s.name for s in conditions[name][0]])
            obs, report = ex.execute_plan(env, obs, steps, regions=regions, goal=goal, domain=cond_domain,
                                          learned={target: model})
            results[name].append({**row, **report.to_dict()})
            failed = next((s for s in report.steps if not s.ok), None)
            status = "goal met" if report.goal_met else (
                f"failed at {failed.action} ({failed.detail})" if failed else "goal not met")
            diverged = [f"{s.action}: {', '.join(s.divergence)}" for s in report.steps if s.divergence]
            retries = sum(max(s.attempts - 1, 0) for s in report.steps)
            print(f"  seed {seed}: {status}; A {report.cube_a_err_mm:5.1f}mm "
                  f"B {report.cube_b_err_mm:5.1f}mm; retries {retries}"
                  + (f"; forward skill re-rolled {rerolls}x" if rerolls else ""))
            for line in diverged:
                print(f"      model != world after {line}")

    # Apart from the results above: how often the evaluation seeds really give a valid handoff.
    evaluation_handoffs = survey_handoffs(env, spec, args.seeds, start, domain, forward)
    env.close()

    print(f"\n{'condition':22s} {'goal met':>9} {'(Boolean)':>10} {'mean A err':>11} {'mean B err':>11} {'diverging steps':>16}")
    summary = {}
    for name, rows in results.items():
        met = float(np.mean([bool(r["goal_met"]) for r in rows]))
        met_boolean = float(np.mean([bool(r["goal_met_boolean"]) for r in rows]))
        a_err = float(np.mean([r["cube_a_err_mm"] for r in rows]))
        b_err = float(np.mean([r["cube_b_err_mm"] for r in rows]))
        diverging = sum(bool(s["divergence"]) for r in rows for s in r["steps"])
        unmet = sorted({literal for r in rows for literal in r["goal_unmet"]})
        summary[name] = {"goal_met_rate": met, "goal_met_rate_boolean": met_boolean, "mean_a_err_mm": a_err,
                         "mean_b_err_mm": b_err, "diverging_steps": diverging, "goal_literals_ever_unmet": unmet}
        print(f"{name:22s} {met:>8.0%} {met_boolean:>9.0%} {a_err:>10.1f}mm {b_err:>10.1f}mm {diverging:>16}")

    passed = (summary["full"]["goal_met_rate"] >= 0.8
              and summary["library_only"]["goal_met_rate"] == 0.0
              and summary["relearned"]["goal_met_rate"] >= 0.8
              and target in learned.effect_literals)
    print(f"\nraw handoff validity, evaluation seeds: {evaluation_handoffs['valid']} of {evaluation_handoffs['of']}")
    for row in evaluation_handoffs["seeds"]:
        if not row["valid"]:
            print(f"  seed {row['seed']}: {', '.join(row['violations'])}")
    print("\nSTAGE 4:", "PASS" if passed else "FAIL")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "inputs": {"checkpoint": str(args.checkpoint), "spec": str(args.spec), "operator": str(args.operator),
                   "plan": str(args.plan)},
        "start": sorted(start), "goal": [l.to_dict() for l in sorted(join(*goal))], "target": target.to_dict(),
        "learned_effects_contain_the_target": target in learned.effect_literals,
        "plans": {name: [s.name for s in steps] for name, (steps, _) in conditions.items()},
        "learned_action": learned.to_dict(),
        "handoff_validity": {"demonstrations": learned_record["handoff_summary"], "evaluation_seeds": evaluation_handoffs},
        "summary": summary, "results": results, "passed": passed}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
