"""Run a plan on the robot, checking every step against its own model.

Every step is treated the same way, scripted or learned: its preconditions are
checked against the measured scene, it runs, its postcondition is measured, and
it is retried if the postcondition does not hold. That contract is what lets a
stochastic policy sit in a plan beside scripted primitives — the executive never
assumes a step worked.

Given the planning domain, the executive also predicts the symbolic state after
each step and reports every scored literal where the measurement disagrees. An
unmodelled side effect shows up there, at the step that caused it.

Nothing here names a particular skill: a step is learned if it is a hole or a
learned operator, and the predicate it must achieve is read from its name.
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
    delegated_predicate,
    predicate_args,
    symbolic_state,
)

MAX_RETRIES = 3
SATISFIED = 0.8
LEARNED_STEPS = 40
_LIFTED_Z = 0.05


@dataclass
class StepReport:
    action: str
    attempts: int
    ok: bool
    detail: str = ""
    divergence: list[str] = field(default_factory=list)


@dataclass
class ExecutionReport:
    steps: list[StepReport] = field(default_factory=list)
    goal_met: bool | None = None
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
            "cube_a_err_mm": self.cube_a_err_mm,
            "cube_b_err_mm": self.cube_b_err_mm,
            "final_scores": self.final_scores,
            "ok": self.ok,
        }


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
        return obs, value >= SATISFIED, f"in_region {value:.2f}"
    if name.startswith("approach("):
        obs = prim.approach(env, obs, cube)
        seated = prim.seated_on(obs, cube)
        return obs, seated, "seated" if seated else "not seated on the cube"
    return obs, False, f"no executor bound for {name}"


def run_learned(env, obs, model, achieve: str, regions, registry=None,
                steps: int = LEARNED_STEPS, on_step: Callable | None = None):
    """Drive a learned skill until the predicate it owes holds or the budget runs out."""
    registry = registry or sp.registry()
    predicate = registry.get(achieve)
    obj = predicate_args(achieve)[0]
    for _ in range(steps):
        if predicate.evaluate(prim.obs_to_scene(obs, regions)).score >= SATISFIED:
            break
        action, _ = model.predict(skill_io.observe(obs, regions, predicate, obj), deterministic=True)
        obs, *_ = env.step(torch.tensor(skill_io.command(action)))
        if on_step is not None:
            on_step(obs)
    return obs


def execute_plan(env, obs, steps: list[Action], model=None, regions=None,
                 goal: tuple[set[str], set[str]] | None = None, domain=None,
                 on_step: Callable | None = None,
                 on_action: Callable[[str, int], None] | None = None
                 ) -> tuple[object, ExecutionReport]:
    """Run each planned step, verifying and retrying it. Stops at the first step
    that cannot be made to work, as an executive must."""
    registry = sp.registry()
    regions = regions or prim.regions(obs)
    report = ExecutionReport()

    def scores(o):
        return {k: float(r.score) for k, r in
                registry.evaluate_all(prim.obs_to_scene(o, regions)).items()}

    predicted = symbolic_state(scores(obs)) if domain is not None else None
    for step in steps:
        measured = scores(obs)
        # Only scored literals can be checked; planner fluents such as what the
        # gripper holds are the model's bookkeeping, not measurements.
        unmet = [p for p in sorted(step.pre) if p in measured and measured[p] < SATISFIED]
        unmet += [f"not {q}" for q in sorted(step.pre_absent)
                  if q in measured and measured[q] >= SATISFIED]
        if unmet:
            report.steps.append(StepReport(step.name, 0, False,
                                           "precondition not met: " + ", ".join(unmet)))
            break

        achieve = step.hole_for or delegated_predicate(step.name)
        attempts, ok, detail = 0, False, ""
        for attempts in range(1, MAX_RETRIES + 1):
            if on_action is not None:
                on_action(step.name, attempts)
            if achieve is None:
                obs, ok, detail = execute_action(env, obs, step.name, regions, scores)
            elif model is None:
                ok, detail = False, "no policy for this operator"
                break
            else:
                obs = run_learned(env, obs, model, achieve, regions, registry, on_step=on_step)
                value = scores(obs)[achieve]
                ok, detail = value >= SATISFIED, f"{achieve} {value:.2f}"
            if ok:
                break

        divergence: list[str] = []
        if predicted is not None:
            predicted = domain.apply(predicted, step)
            after = symbolic_state(scores(obs))
            divergence = sorted(p for p in domain.scored if (p in predicted) != (p in after))
        report.steps.append(StepReport(step.name, attempts, bool(ok), detail, divergence))
        if not ok:
            break

    final = scores(obs)
    report.final_scores = final
    report.cube_a_err_mm = float(np.linalg.norm(
        prim.cube_pos(obs, "cube_a")[:2] - prim.src_pos(obs, "cube_a")[:2]) * 1000)
    report.cube_b_err_mm = float(np.linalg.norm(
        prim.cube_pos(obs, "cube_b")[:2] - prim.src_pos(obs, "cube_b")[:2]) * 1000)
    if goal is not None:
        positive, negative = goal
        report.goal_met = (all(final.get(p, 0.0) >= SATISFIED for p in positive)
                           and all(final.get(q, 0.0) < SATISFIED for q in negative))
    return obs, report
