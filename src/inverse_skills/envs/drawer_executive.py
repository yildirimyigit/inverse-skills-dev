"""Run a drawer plan on the robot, checking every step against its own model.

The contract of the stow executive (stow_executive.py), bound to the drawer's
predicates, primitives and learned-skill interface. That executive calls stow's
primitives and predicates directly, so the drawer needs its own binding; the
rules are the same, with two checks made stricter:

  before each attempt  the step's preconditions are checked against the
                       measured scene (stow checked once per step); a step
                       refused before it ever runs reports 0 attempts
  the attempt          a scripted controller, or the learned policy for a hole
  after each attempt   the step's expected effects are measured: what the
                       planner's model says it makes true must hold, and what
                       it makes false (deletes, and the mutex partners of its
                       adds) must not (stow checked one physical measure per
                       action); otherwise the step is retried, up to
                       MAX_RETRIES attempts in all
  monitor              after every step that ran, the planner's predicted
                       symbolic state is compared with the measured one, and
                       differences are reported

Execution stops at the first step that cannot be made to work. A learned skill
runs until the predicate it owes holds, or for at most LEARNED_STEPS steps.
Each step also records the drawer's opening and the hand's tilt after it, the
physical side of what the step did.

Nothing here names a particular skill. A step is learned if it is a hole or a
learned operator; the predicate it must achieve is read from its name, and the
policy for it comes from the caller (built from the hole spec). Scripted steps
run only the controllers the caller passes, so a condition can never use a
controller it was not given.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.envs import drawer_skill_io as skill_io
from inverse_skills.operators.hole_planner import Action, delegated_predicate, symbolic_state

MAX_RETRIES = 3
SATISFIED = 0.8
LEARNED_STEPS = 40


@dataclass
class StepReport:
    action: str
    attempts: int
    ok: bool
    detail: str = ""
    divergence: list[str] = field(default_factory=list)
    opening_mm: float = float("nan")   # the drawer's opening after the step
    tilt_deg: float = float("nan")     # the hand's tilt from straight down after the step


@dataclass
class ExecutionReport:
    steps: list[StepReport] = field(default_factory=list)
    goal_met: bool | None = None
    final_scores: dict[str, float] = field(default_factory=dict)
    final_opening_mm: float = float("nan")

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps) and bool(self.goal_met)

    def to_dict(self) -> dict:
        return {"steps": [vars(s) for s in self.steps], "goal_met": self.goal_met,
                "final_scores": self.final_scores, "final_opening_mm": self.final_opening_mm,
                "ok": self.ok}


def scores(obs, registry) -> dict[str, float]:
    return {k: float(r.score) for k, r in registry.evaluate_all(prim.obs_to_scene(obs)).items()}


def run_learned(env, obs, policy, achieve, steps: int = LEARNED_STEPS):
    """Drive a learned skill until the predicate it owes holds or the budget runs out."""
    for _ in range(steps):
        if achieve.evaluate(prim.obs_to_scene(obs)).score >= SATISFIED:
            break
        action, _ = policy.predict(skill_io.observe(obs, achieve), deterministic=True)
        obs, *_ = env.step(torch.tensor(skill_io.command(action)))
    return obs


def execute_plan(env, obs, steps: list[Action], domain, goal: tuple[set[str], set[str]],
                 controllers: dict[str, Callable], learned: dict[str, object] | None = None,
                 on_action: Callable[[str, int], None] | None = None
                 ) -> tuple[object, ExecutionReport]:
    """Run each planned step, verifying and retrying it.

    `controllers` maps a library action's name (without arguments) to its
    scripted controller. `learned` maps a predicate to the policy that
    establishes it.
    """
    registry = dp.registry()
    learned = learned or {}
    report = ExecutionReport()
    predicted = symbolic_state(scores(obs, registry))
    for step in steps:
        achieve = step.hole_for or delegated_predicate(step.name)
        name = step.name.split("(")[0]
        attempts, ok, detail = 0, False, ""
        while attempts < MAX_RETRIES:
            # Only scored literals can be checked against the scene.
            measured = scores(obs, registry)
            unmet = [p for p in sorted(step.pre) if p in measured and measured[p] < SATISFIED]
            unmet += [f"not {q}" for q in sorted(step.pre_absent)
                      if q in measured and measured[q] >= SATISFIED]
            if unmet:
                refused = "precondition not met: " + ", ".join(unmet)
                ok, detail = False, f"{detail}; retry refused, {refused}" if attempts else refused
                break
            if achieve is not None and achieve not in learned:
                ok, detail = False, f"no learned skill for {achieve}"
                break
            if achieve is None and name not in controllers:
                ok, detail = False, f"no controller for {step.name}"
                break

            attempts += 1
            if on_action is not None:
                on_action(step.name, attempts)
            if achieve is not None:
                obs = run_learned(env, obs, learned[achieve], registry.get(achieve))
            else:
                obs = controllers[name](env, obs)

            measured = scores(obs, registry)
            missing = [p for p in sorted(domain.closure(step.add))
                       if p in measured and measured[p] < SATISFIED]
            missing += [f"not {q}" for q in sorted(domain.removes(step))
                        if q in measured and measured[q] >= SATISFIED]
            ok = not missing
            detail = "expected effects measured" if ok else "effect not measured: " + ", ".join(missing)
            if ok:
                break

        divergence: list[str] = []
        if attempts:   # a step refused before it ran changed nothing to compare
            predicted = domain.apply(predicted, step)
            after = symbolic_state(scores(obs, registry))
            divergence = sorted(p for p in domain.scored if (p in predicted) != (p in after))
        report.steps.append(StepReport(step.name, attempts, ok, detail, divergence,
                                       round(prim.opening(obs) * 1000, 1), prim.tilt_deg(obs)))
        if not ok:
            break

    final = scores(obs, registry)
    report.final_scores = final
    report.final_opening_mm = round(prim.opening(obs) * 1000, 1)
    positive, negative = goal
    report.goal_met = (all(final.get(p, 0.0) >= SATISFIED for p in positive)
                       and all(final.get(q, 0.0) < SATISFIED for q in negative))
    return obs, report
