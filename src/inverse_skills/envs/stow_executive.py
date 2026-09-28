"""Run a plan on the robot, checking each step's postcondition.

Every operator is treated the same way, scripted or learned: it runs, its
postcondition is measured, and it is retried if the postcondition does not
hold. That is the contract that lets a stochastic policy sit in a plan beside
scripted primitives — the executive never assumes a step worked.

The learned operator additionally checks its own precondition before running,
because the policy is only trusted inside the region Stage 3 measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

from inverse_skills.envs import stow_cube as geo
from inverse_skills.envs import stow_domain as sd
from inverse_skills.envs import stow_hole_env as hole
from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs import stow_primitives as prim

MAX_RETRIES = 3
SATISFIED = 0.8
_LIFTED_Z = 0.05
_HOLE_STEPS = 40


@dataclass
class StepReport:
    action: str
    attempts: int
    ok: bool
    detail: str = ""


@dataclass
class ExecutionReport:
    steps: list[StepReport] = field(default_factory=list)
    goal_met: bool = False
    final_scores: dict[str, float] = field(default_factory=dict)
    cube_a_err_mm: float = float("nan")
    cube_b_err_mm: float = float("nan")

    @property
    def ok(self) -> bool:
        return all(step.ok for step in self.steps) and self.goal_met

    def to_dict(self) -> dict:
        return {
            "steps": [vars(s) for s in self.steps],
            "goal_met": self.goal_met,
            "cube_a_err_mm": self.cube_a_err_mm,
            "cube_b_err_mm": self.cube_b_err_mm,
            "final_scores": self.final_scores,
            "ok": self.ok,
        }


def _slot_target(obs, cube: str, slot: str) -> np.ndarray:
    if slot in ("src_a", "src_b"):
        return prim.src_pos(obs, cube)
    xy = geo.POCKET_A_XY if slot == "pocket" else geo.MOUTH_B_XY
    return np.array([xy[0], xy[1], geo.CUBE_HALF], dtype=np.float64)


def run_hole_policy(env, obs, model, steps: int = _HOLE_STEPS, on_step: Callable | None = None):
    """Drive the learned operator until its postcondition holds or the budget runs out."""
    for _ in range(steps):
        scene_score = sp.ClearOfWallsPredicate("cube_a").evaluate(
            prim.obs_to_scene(obs, prim.regions(obs))).score
        if scene_score >= SATISFIED:
            break
        action, _ = model.predict(hole.policy_observation(obs), deterministic=True)
        obs, *_ = env.step(torch.tensor(hole.policy_command(action)))
        if on_step is not None:
            on_step(obs)
    return obs


def _seated_on(obs, cube: str) -> bool:
    c = prim.cube_pos(obs, cube)
    tcp = prim.tcp_pos(obs)
    contact_z = c[2] + geo.CUBE_HALF + prim._FINGERTIP_OFFSET
    return bool(abs(tcp[0] - c[0]) <= hole._SEATED_XY
                and abs(tcp[1] - c[1]) <= hole._SEATED_XY
                and tcp[2] <= contact_z + hole._SEATED_Z)


def execute_plan(env, obs, plan: list[str], model=None, regions=None,
                 on_step: Callable | None = None,
                 on_action: Callable[[str, int], None] | None = None
                 ) -> tuple[object, ExecutionReport]:
    """Run each planned action, verifying and retrying it. Stops at the first step
    that cannot be made to work, as an executive must."""
    registry = sp.registry()
    regions = regions or prim.regions(obs)
    report = ExecutionReport()

    def scores(o):
        return {k: float(r.score) for k, r in
                registry.evaluate_all(prim.obs_to_scene(o, regions)).items()}

    library = {a.name: a for a in sd.library()}
    for action in plan:
        cube = "cube_a" if "cube_a" in action else "cube_b"
        attempts, ok, detail = 0, False, ""

        # Check the action's own preconditions before running it. Without this
        # the executive flails: a pick whose precondition is missing still
        # nudges the cube on each retry, which can free it by accident and make
        # an impossible plan look like it sometimes works.
        modelled = library.get(action)
        if modelled is not None:
            measured = scores(obs)
            unmet = [p for p in modelled.pre
                     if p in measured and measured[p] < SATISFIED]
            if unmet:
                report.steps.append(StepReport(action, 0, False,
                                               "precondition not met: " + ", ".join(unmet)))
                break

        for attempts in range(1, MAX_RETRIES + 1):
            if on_action is not None:
                on_action(action, attempts)
            if action.startswith("pick"):
                obs = prim.lift(env, prim.pick(env, obs, cube))
                ok = float(prim.cube_pos(obs, cube)[2]) > _LIFTED_Z
                detail = f"cube z {prim.cube_pos(obs, cube)[2] * 1000:.0f}mm"
            elif action.startswith("place"):
                slot = action[action.index(",") + 1:-1]
                obs = prim.place(env, obs, cube, _slot_target(obs, cube, slot))
                score = scores(obs)[f"in_region({cube},{slot})"]
                ok = score >= SATISFIED
                detail = f"in_region {score:.2f}"
            elif action.startswith("approach"):
                obs = prim.approach(env, obs, cube)
                ok = _seated_on(obs, cube)
                detail = "seated" if ok else "not seated on the cube"
            elif action.startswith(("HOLE", "drag")):
                if model is None:
                    ok, detail = False, "no policy for this operator"
                    break
                if not _seated_on(obs, "cube_a"):      # the operator's precondition
                    obs = prim.seat_on(env, obs, "cube_a")
                if not _seated_on(obs, "cube_a"):
                    ok, detail = False, "precondition not met: not seated on cube_a"
                    continue
                obs = run_hole_policy(env, obs, model, on_step=on_step)
                score = scores(obs)["clear_of_walls(cube_a)"]
                ok = score >= SATISFIED
                detail = f"clear_of_walls {score:.2f}"
            else:
                ok, detail = False, f"no executor bound for {action}"
                break
            if ok:
                break
        report.steps.append(StepReport(action, attempts, bool(ok), detail))
        if not ok:
            break

    final = scores(obs)
    report.final_scores = final
    report.cube_a_err_mm = float(np.linalg.norm(
        prim.cube_pos(obs, "cube_a")[:2] - prim.src_pos(obs, "cube_a")[:2]) * 1000)
    report.cube_b_err_mm = float(np.linalg.norm(
        prim.cube_pos(obs, "cube_b")[:2] - prim.src_pos(obs, "cube_b")[:2]) * 1000)
    report.goal_met = (
        final["in_region(cube_a,src_a)"] >= SATISFIED
        and final["in_region(cube_b,src_b)"] >= SATISFIED
        and final["clear_of_walls(cube_a)"] >= SATISFIED
        and final["gripper_open()"] >= SATISFIED
        and final["in_region(cube_a,pocket)"] < SATISFIED
        and final["in_region(cube_b,mouth)"] < SATISFIED
    )
    return obs, report
