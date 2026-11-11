"""Run a drawer plan on the robot, checking every step against its own model, in signed literals.

The contract of the stow executive (stow_executive.py), bound to the drawer's
predicates, primitives and learned-skill interface. That executive calls stow's
primitives and predicates directly, so the drawer needs its own binding; the
rules are the same, with two checks made stricter:

  before each attempt  the step's preconditions, of both signs, are checked
                       against the measured scene (stow checked once per step); a
                       step refused before it ever runs reports 0 attempts
  the attempt          a scripted controller, or the learned policy for the signed
                       literal the step owes (`hole_for` or `learned_for`)
  after each attempt   the step's expected effects are measured: what the
                       planner's model says it makes true must hold, and what
                       it makes false (deletes, and the mutex partners of its
                       adds) must not (stow checked one physical measure per
                       action); otherwise the step is retried, up to
                       MAX_RETRIES attempts in all
  monitor              after every step that ran, the planner's predicted Boolean
                       symbolic state is compared with the measured one, and the
                       scored atoms left between 0.2 and 0.8 are listed; for a
                       learned step, the fences it was planned to keep are checked

Every check on the robot is robust (`literals.py`): `p` is met by a score of at
least 0.8 and `not p` by a score of at most 0.2. The goal is reported robustly
(`goal_met`) and in the Boolean sense (`goal_met_boolean`).

Execution stops at the first step that cannot be made to work. A learned skill
runs until the literal it owes robustly holds, or for at most LEARNED_STEPS steps.
Each step also records the drawer's opening and the hand's tilt after it, the
physical side of what the step did.

Nothing here names a particular skill. A step is learned if it carries the literal
it owes, and the policy for it comes from the caller, keyed by that literal.
Scripted steps run only the controllers the caller passes, so a condition can
never use a controller it was not given.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.envs import drawer_skill_io as skill_io
from inverse_skills.operators.hole_planner import Action, derive_fences, symbolic_state
from inverse_skills.operators.literals import Literal, join, robust_value, robustly_unmet

MAX_RETRIES = 3
LEARNED_STEPS = 40


@dataclass
class StepReport:
    action: str
    attempts: int
    ok: bool
    detail: str = ""
    divergence: list[str] = field(default_factory=list)       # Boolean: predicted and measured differ
    opening_mm: float = float("nan")   # the drawer's opening after the step
    tilt_deg: float = float("nan")     # the hand's tilt from straight down after the step
    uncertain: list[str] = field(default_factory=list)        # scored atoms left between 0.2 and 0.8
    fences_broken: list[str] = field(default_factory=list)    # a learned step's fences not robustly kept


@dataclass
class ExecutionReport:
    steps: list[StepReport] = field(default_factory=list)
    goal_met: bool | None = None                  # every goal literal robustly holds
    goal_met_boolean: bool | None = None          # every goal literal holds in the Boolean sense
    goal_unmet: list[str] = field(default_factory=list)   # the goal literals not robustly met
    final_scores: dict[str, float] = field(default_factory=dict)
    final_opening_mm: float = float("nan")

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps) and bool(self.goal_met)

    def to_dict(self) -> dict:
        return {"steps": [vars(s) for s in self.steps], "goal_met": self.goal_met,
                "goal_met_boolean": self.goal_met_boolean, "goal_unmet": self.goal_unmet,
                "final_scores": self.final_scores, "final_opening_mm": self.final_opening_mm,
                "ok": self.ok}


def scores(obs, registry) -> dict[str, float]:
    return {k: float(r.score) for k, r in registry.evaluate_all(prim.obs_to_scene(obs)).items()}


def run_learned(env, obs, policy, achieve, steps: int = LEARNED_STEPS, positive: bool = True):
    """Drive a learned skill until the literal it owes robustly holds or the budget runs out.
    `achieve` is that literal's predicate and `positive` its sign."""
    target = Literal(achieve.key, positive)
    for _ in range(steps):
        if target.robustly_holds(achieve.evaluate(prim.obs_to_scene(obs)).score):
            break
        action, _ = policy.predict(skill_io.observe(obs, achieve, positive), deterministic=True)
        obs, *_ = env.step(torch.tensor(skill_io.command(action)))
    return obs


def modelled_effects(domain, step: Action) -> frozenset[Literal]:
    """What the model says the step makes true and false: its additions with what they imply,
    and its deletions with the mutex partners of what it adds."""
    return join(domain.closure(step.add), domain.removes(step) - domain.closure(step.add))


def execute_plan(env, obs, steps: list[Action], domain, goal: tuple[set[str], set[str]],
                 controllers: dict[str, Callable], learned: dict[Literal, object] | None = None,
                 on_action: Callable[[str, int], None] | None = None
                 ) -> tuple[object, ExecutionReport]:
    """Run each planned step, verifying and retrying it.

    `controllers` maps a library action's name (without arguments) to its
    scripted controller. `learned` maps the signed literal a learned step owes to
    the policy that establishes it. `goal` is the goal's two projections, the
    atoms that must be true and those that must be false.
    """
    registry = dp.registry()
    learned = learned or {}
    report = ExecutionReport()
    start = symbolic_state(scores(obs, registry))
    predicted = start
    for index, step in enumerate(steps):
        target = step.hole_for or step.learned_for     # the signed literal a learned step owes
        name = step.name.split("(")[0]
        attempts, ok, detail = 0, False, ""
        while attempts < MAX_RETRIES:
            measured = scores(obs, registry)
            unmet = robustly_unmet([l for l in step.precondition_literals if l.atom in measured], measured)
            if unmet:
                refused = "precondition not met: " + ", ".join(f"{l} ({measured[l.atom]:.2f})" for l in unmet)
                ok, detail = False, f"{detail}; retry refused, {refused}" if attempts else refused
                break
            if target is not None and target not in learned:
                ok, detail = False, f"no learned skill for {target}"
                break
            if target is None and name not in controllers:
                ok, detail = False, f"no controller for {step.name}"
                break

            attempts += 1
            if on_action is not None:
                on_action(step.name, attempts)
            if target is not None:
                obs = run_learned(env, obs, learned[target], registry.get(target.atom), positive=target.positive)
            else:
                obs = controllers[name](env, obs)

            measured = scores(obs, registry)
            missing = robustly_unmet([l for l in modelled_effects(domain, step) if l.atom in measured], measured)
            ok = not missing
            detail = "expected effects measured" if ok else "effect not measured: " + ", ".join(
                f"{l} ({measured[l.atom]:.2f})" for l in missing)
            if ok:
                break

        divergence: list[str] = []
        fences_broken: list[str] = []
        after = scores(obs, registry)
        if attempts:   # a step refused before it ran changed nothing to compare
            if target is not None:
                kept_true, kept_false = derive_fences(domain, start, steps, index, *goal)
                fences_broken = [str(l) for l in robustly_unmet(join(kept_true, kept_false), after)]
            predicted = domain.apply(predicted, step)
            measured_state = symbolic_state(after)
            divergence = sorted(p for p in domain.scored if (p in predicted) != (p in measured_state))
        uncertain = sorted(k for k, v in after.items() if robust_value(v) == "uncertain")
        report.steps.append(StepReport(step.name, attempts, ok, detail, divergence,
                                       round(prim.opening(obs) * 1000, 1), prim.tilt_deg(obs),
                                       uncertain, fences_broken))
        if not ok:
            break

    final = scores(obs, registry)
    report.final_scores = final
    report.final_opening_mm = round(prim.opening(obs) * 1000, 1)
    literals = join(*goal)
    report.goal_unmet = [str(l) for l in robustly_unmet(literals, final)]
    report.goal_met = not report.goal_unmet
    report.goal_met_boolean = all(l.boolean_holds(final[l.atom]) for l in literals)
    return obs, report
