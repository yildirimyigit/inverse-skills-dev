from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inverse_skills.core import SceneGraph
from inverse_skills.operators.restoration import (
    ResidualInverseObjective,
    RestorationObjective,
)
from inverse_skills.operators.toy_planner import PlanResult, ToyInversePlanner


@dataclass
class TwoPhaseInverseResult:
    """Symbolic prefix from BFS plus a residual reward for downstream RL.

    Phase 1 (symbolic): the BFS plan that closes as much of the inverse target
    as the primitive library can within the search budget.

    Phase 2 (residual): an objective restricted to the inverse target terms
    that BFS provably could not satisfy across any visited state.  This is the
    shaped reward function for an RL agent that should bridge the remaining
    gap, with the BFS handoff scene as its starting state.
    """

    symbolic_prefix: list[str]
    handoff_scene: SceneGraph
    residual_objective: ResidualInverseObjective
    v_initial: float
    v_handoff: float
    v_target: float
    term_max_scores: dict[str, float]
    fully_solved: bool
    expanded_nodes: int

    @property
    def gap_total(self) -> float:
        return self.v_target - self.v_initial

    @property
    def gap_closed_by_symbolic(self) -> float:
        return self.v_handoff - self.v_initial

    @property
    def gap_remaining_for_rl(self) -> float:
        return self.v_target - self.v_handoff

    @property
    def residual_term_keys(self) -> list[str]:
        return [t.key for t in self.residual_objective.terms]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbolic_prefix": list(self.symbolic_prefix),
            "v_initial": float(self.v_initial),
            "v_handoff": float(self.v_handoff),
            "v_target": float(self.v_target),
            "gap_total": float(self.gap_total),
            "gap_closed_by_symbolic": float(self.gap_closed_by_symbolic),
            "gap_remaining_for_rl": float(self.gap_remaining_for_rl),
            "term_max_scores": {k: float(v) for k, v in self.term_max_scores.items()},
            "residual_term_keys": list(self.residual_term_keys),
            "fully_solved": bool(self.fully_solved),
            "expanded_nodes": int(self.expanded_nodes),
        }


def two_phase_inverse(
    planner: ToyInversePlanner,
    start_scene: SceneGraph,
    *,
    max_depth: int = 3,
    term_reachable_threshold: float = 0.90,
    v_target: float = 1.0,
) -> TwoPhaseInverseResult:
    """Run BFS, then expose the residual reward and warm-start state for RL.

    A term is considered "achieved by the symbolic phase" iff its max score
    across every BFS-visited scene reaches `term_reachable_threshold`.  Terms
    that never reach that threshold form the residual objective.
    """
    plan = planner.plan(start_scene, max_depth=max_depth)
    base = planner.objective

    residual_keys: set[str] = set()
    for term in base.terms:
        max_score = plan.term_max_scores.get(term.key, 0.0)
        if max_score < term_reachable_threshold:
            residual_keys.add(term.key)

    residual_objective = ResidualInverseObjective(base, residual_keys)
    handoff_scene = plan.handoff_scene if plan.handoff_scene is not None else start_scene

    return TwoPhaseInverseResult(
        symbolic_prefix=list(plan.actions),
        handoff_scene=handoff_scene,
        residual_objective=residual_objective,
        v_initial=float(plan.initial_potential),
        v_handoff=float(plan.final_potential),
        v_target=float(v_target),
        term_max_scores=dict(plan.term_max_scores),
        fully_solved=bool(plan.success),
        expanded_nodes=int(plan.expanded_nodes),
    )
