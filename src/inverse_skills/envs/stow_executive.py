"""Run a plan on the robot, checking every step against its own model, in signed literals.

Every step is treated the same way, scripted or learned: its preconditions are
checked against the measured scene, it runs, its effects are measured, and it is
retried if they do not hold. That contract is what lets a stochastic policy sit in
a plan beside scripted primitives — the executive never assumes a step worked.

Every check on the robot is robust (`literals.py`): `p` is met by a score of at
least 0.8 and `not p` by a score of at most 0.2, so a score in between meets
neither. Only scored literals can be measured; planner fluents such as what the
gripper holds are the model's bookkeeping.

  before a step   its scored preconditions, of both signs
  the step        a scripted controller, or the learned policy for the signed
                  literal the step owes (`hole_for` or `learned_for`), which runs
                  until that literal robustly holds, or for LEARNED_STEPS steps
  after it        each scripted action's own physical check (a cube lifted, a cube
                  placed, the hand seated, the hand clear of both cubes) or, for a
                  learned step, its target literal; and, given the planning domain,
                  every effect the model says the step has: what it makes true must
                  robustly hold, what it makes false must robustly not
  fences          for a learned step, given the planning domain and the goal: the
                  literals the plan relies on across it, reported if not kept
  monitor         given the domain, the predicted Boolean symbolic state after each
                  step against the measured one; and the scored atoms whose score is
                  left between 0.2 and 0.8, where the Boolean and robust readings differ
  the goal        every goal literal robustly (`goal_met`), and in the Boolean sense
                  (`goal_met_boolean`)

Nothing here names a particular skill: a step is learned if it carries the literal
it owes, and its policy comes from the caller, keyed by that literal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs import stow_primitives as prim
from inverse_skills.envs import stow_skill_io as skill_io
from inverse_skills.operators.hole_planner import (
    Action,
    derive_fences,
    predicate_args,
    symbolic_state,
)
from inverse_skills.operators.literals import (
    ROBUST_FALSE,
    ROBUST_TRUE,
    TRUE_AT,
    Literal,
    join,
    robust_value,
    robustly_unmet,
)

MAX_RETRIES = 3
LEARNED_STEPS = 40
_LIFTED_Z = 0.05


@dataclass
class StepReport:
    action: str
    attempts: int
    ok: bool
    detail: str = ""
    divergence: list[str] = field(default_factory=list)       # Boolean: predicted and measured differ
    uncertain: list[str] = field(default_factory=list)        # scored atoms left between 0.2 and 0.8
    fences_broken: list[str] = field(default_factory=list)    # a learned step's fences not robustly kept


@dataclass
class ExecutionReport:
    steps: list[StepReport] = field(default_factory=list)
    goal_met: bool | None = None                  # every goal literal robustly holds
    goal_met_boolean: bool | None = None          # every goal literal holds in the Boolean sense
    goal_unmet: list[str] = field(default_factory=list)   # the goal literals not robustly met
    final_scores: dict[str, float] = field(default_factory=dict)
    cube_a_err_mm: float = float("nan")
    cube_b_err_mm: float = float("nan")

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps) and bool(self.goal_met)

    def to_dict(self) -> dict:
        return {
            "steps": [vars(s) for s in self.steps],
            "goal_met": self.goal_met,
            "goal_met_boolean": self.goal_met_boolean,
            "goal_unmet": self.goal_unmet,
            "cube_a_err_mm": self.cube_a_err_mm,
            "cube_b_err_mm": self.cube_b_err_mm,
            "final_scores": self.final_scores,
            "ok": self.ok,
        }


def scored(literals, scores) -> list[Literal]:
    """The literals that can be measured: those on scored atoms."""
    return sorted(l for l in literals if l.atom in scores)


def modelled_effects(domain, step: Action) -> frozenset[Literal]:
    """What the model says the step makes true and false: its additions with what they imply,
    and its deletions with the mutex partners of what it adds."""
    return join(domain.closure(step.add), domain.removes(step) - domain.closure(step.add))


def execute_action(env, obs, name: str, regions, score) -> tuple[object, bool, str]:
    """Run one scripted library action and measure its postcondition."""
    args = predicate_args(name)
    cube = args[0] if args else None
    if name.startswith("pick("):
        obs = prim.lift(env, prim.pick(env, obs, cube))
        z = float(prim.cube_pos(obs, cube)[2])
        return obs, z > _LIFTED_Z, f"cube z {z * 1000:.0f}mm"
    if name.startswith("place("):
        slot = args[1]
        obs = prim.place(env, obs, cube, prim.slot_target(obs, cube, slot))
        value = score(obs)[f"in_region({cube},{slot})"]
        return obs, value >= ROBUST_TRUE, f"in_region {value:.2f}"
    if name.startswith("approach("):
        obs = prim.approach(env, obs, cube)
        seated = prim.seated_on(obs, cube)
        return obs, seated, "seated" if seated else "not seated on the cube"
    if name == "retreat_to_safe":
        obs = retreat_to_safe(env, obs, regions)
        values = score(obs)
        near = {key: values[key] for key in ("tcp_near(cube_a)", "tcp_near(cube_b)")}
        return (obs, all(value <= ROBUST_FALSE for value in near.values()),
                ", ".join(f"{key} {value:.2f}" for key, value in near.items()))
    return obs, False, f"no executor bound for {name}"


def run_learned(env, obs, model, target: Literal, regions, registry=None,
                steps: int = LEARNED_STEPS, on_step: Callable | None = None):
    """Drive a learned skill until the literal it owes robustly holds or the budget runs out.
    For `p` that is a score of at least 0.8; for `not p`, at most 0.2."""
    registry = registry or sp.registry()
    predicate = registry.get(target.atom)
    obj = predicate_args(target.atom)[0]
    for _ in range(steps):
        if target.robustly_holds(predicate.evaluate(prim.obs_to_scene(obs, regions)).score):
            break
        action, _ = model.predict(skill_io.observe(obs, regions, predicate, obj, target.positive),
                                  deterministic=True)
        obs, *_ = env.step(torch.tensor(skill_io.command(action)))
        if on_step is not None:
            on_step(obs)
    return obs


def retreat_to_safe(env, obs, regions, registry=None):
    """The library action retreat_to_safe: the existing `retreat()` rise to the travel height,
    with the gripper kept as it is.

    The gripper is commanded open if gripper_open() holds when the action starts (its score at or
    above 0.5, the predicate's own truth boundary), and closed otherwise, so the action changes
    nothing about the gripper. What the rise really does to both cubes, the gripper and the
    TCP-near predicates is measured, not assumed: scripts/stow_retreat_gate.py.
    """
    registry = registry or sp.registry()
    gripper_open = registry.get("gripper_open()").evaluate(prim.obs_to_scene(obs, regions)).score >= TRUE_AT
    return prim.retreat(env, obs, gripper_cmd=1.0 if gripper_open else -1.0)


def execute_plan(env, obs, steps: list[Action], model=None, regions=None,
                 goal: tuple[set[str], set[str]] | None = None, domain=None,
                 on_step: Callable | None = None,
                 on_action: Callable[[str, int], None] | None = None,
                 learned: dict[Literal, object] | None = None,
                 ) -> tuple[object, ExecutionReport]:
    """Run each planned step, verifying and retrying it. Stops at the first step that cannot
    be made to work, as an executive must.

    `learned` maps the signed literal a learned step owes to its policy; `model`, if given, is
    the policy for any learned step `learned` does not name. `goal` is the goal's two projections,
    the atoms that must be true and those that must be false.
    """
    registry = sp.registry()
    regions = regions or prim.regions(obs)
    learned = dict(learned or {})
    report = ExecutionReport()

    def scores(o):
        return {k: float(r.score) for k, r in
                registry.evaluate_all(prim.obs_to_scene(o, regions)).items()}

    start = symbolic_state(scores(obs)) if domain is not None else None
    predicted = start
    for index, step in enumerate(steps):
        measured = scores(obs)
        unmet = robustly_unmet(scored(step.precondition_literals, measured), measured)
        if unmet:
            report.steps.append(StepReport(step.name, 0, False, "precondition not met: " + ", ".join(
                f"{l} ({measured[l.atom]:.2f})" for l in unmet)))
            break

        target = step.hole_for or step.learned_for     # the signed literal a learned step owes
        policy = learned.get(target, model) if target is not None else None
        attempts, ok, detail = 0, False, ""
        for attempts in range(1, MAX_RETRIES + 1):
            if on_action is not None:
                on_action(step.name, attempts)
            if target is None:
                obs, ok, detail = execute_action(env, obs, step.name, regions, scores)
            elif policy is None:
                ok, detail = False, f"no policy for {target}"
                break
            else:
                obs = run_learned(env, obs, policy, target, regions, registry, on_step=on_step)
                value = scores(obs)[target.atom]
                ok, detail = target.robustly_holds(value), f"{target} {value:.2f}"
            if ok and domain is not None:
                after = scores(obs)
                missing = robustly_unmet(scored(modelled_effects(domain, step), after), after)
                if missing:
                    ok = False
                    detail += "; effect not measured: " + ", ".join(f"{l} ({after[l.atom]:.2f})" for l in missing)
            if ok:
                break

        after = scores(obs)
        divergence: list[str] = []
        fences_broken: list[str] = []
        if predicted is not None:
            if target is not None and goal is not None:
                kept_true, kept_false = derive_fences(domain, start, steps, index, *goal)
                fences_broken = [str(l) for l in robustly_unmet(join(kept_true, kept_false), after)]
            predicted = domain.apply(predicted, step)
            measured_state = symbolic_state(after)
            divergence = sorted(p for p in domain.scored if (p in predicted) != (p in measured_state))
        uncertain = sorted(k for k, v in after.items() if robust_value(v) == "uncertain")
        report.steps.append(StepReport(step.name, attempts, bool(ok), detail, divergence, uncertain, fences_broken))
        if not ok:
            break

    final = scores(obs)
    report.final_scores = final
    report.cube_a_err_mm = float(np.linalg.norm(
        prim.cube_pos(obs, "cube_a")[:2] - prim.src_pos(obs, "cube_a")[:2]) * 1000)
    report.cube_b_err_mm = float(np.linalg.norm(
        prim.cube_pos(obs, "cube_b")[:2] - prim.src_pos(obs, "cube_b")[:2]) * 1000)
    if goal is not None:
        literals = join(*goal)
        report.goal_unmet = [str(l) for l in robustly_unmet(literals, final)]
        report.goal_met = not report.goal_unmet
        report.goal_met_boolean = all(l.boolean_holds(final[l.atom]) for l in literals)
    return obs, report
