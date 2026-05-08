from __future__ import annotations

import math
from typing import Iterable

from inverse_skills.core.scene import SceneGraph
from inverse_skills.operators.schema import LearnedOperator, PredicateTerm
from inverse_skills.predicates.base import Predicate, PredicateRegistry


def signed_margin_reward(
    scene: SceneGraph,
    terms: Iterable[tuple],
    saturate: bool = True,
) -> float:
    """Dense reward derived directly from the predicate framework.

    Each term is `(predicate, sign, scale)` or `(predicate, sign, scale, mode)`.
    For each, evaluate the predicate on `scene` and add `sign * f(margin/scale)`
    to the total — clipped according to `mode`.

      • `sign=+1` for predicates that should hold in the inverse target;
        `sign=-1` for predicates that should be violated (negated add-effects).
      • `scale` puts every term in unit-free units so heterogeneous predicates
        can be summed (tolerance for at_pose, min_width for gripper_open, etc.).
      • `mode="bipolar"` (default for 3-tuples) → contribution in [-1, +1].
        Use this for the *active residual* predicate the symbolic prefix did
        not fully restore. The agent gets gradient toward satisfaction AND
        bonus for over-satisfaction.
      • `mode="fence"` → contribution in [-1, 0]. Use this for predicates the
        symbolic prefix already restored: violations cost reward, non-violation
        is silent. This is the literal content of "preconditions and
        delete-effects should hold" — no perverse incentive to over-saturate
        a fence and ignore the residual.

    With `saturate=True` (default), `f = tanh` — the signed, bounded analog of
    the sigmoid score the predicate framework already uses. Linear near
    `margin = 0` with slope `1/scale`, smoothly saturates to ±1 outside
    ~3·scale. Bounded reward keeps the SAC value function from blowing up
    when state drifts far during exploration.

    With `saturate=False`, `f` is the identity — useful for diagnostics or for
    tasks where the margin is already bounded by construction.
    """
    total = 0.0
    for term in terms:
        if len(term) == 3:
            predicate, sign, scale = term
            mode = "bipolar"
        elif len(term) == 4:
            predicate, sign, scale, mode = term
        else:
            raise ValueError(
                f"Each term must be (predicate, sign, scale) or "
                f"(predicate, sign, scale, mode); got {term!r}"
            )
        margin = predicate.evaluate(scene).margin
        x = float(sign) * float(margin) / max(float(scale), 1e-6)
        if saturate:
            x = math.tanh(x)
        if mode == "fence":
            x = min(x, 0.0)
        elif mode != "bipolar":
            raise ValueError(f"Unknown term mode {mode!r}; expected 'bipolar' or 'fence'")
        total += x
    return float(total)


class RestorationObjective:
    def __init__(self, operator: LearnedOperator, predicate_registry: PredicateRegistry):
        self.operator = operator
        self.predicates = predicate_registry
        self.terms = operator.inverse_target_terms()

    def term_score(self, term: PredicateTerm, scene: SceneGraph) -> float:
        result = self.predicates.get(term.key).evaluate(scene)
        score = result.score
        if term.polarity == "negative":
            score = 1.0 - score
        return float(term.weight * score)

    def term_scores(self, scene: SceneGraph) -> dict[str, float]:
        return {term.key: self.term_score(term, scene) for term in self.terms}

    def potential(self, scene: SceneGraph) -> float:
        if not self.terms:
            return 0.0
        total_weight = sum(max(term.weight, 1e-6) for term in self.terms)
        return float(sum(self.term_score(term, scene) for term in self.terms) / total_weight)

    def reward(self, previous_scene: SceneGraph, next_scene: SceneGraph) -> float:
        return self.potential(next_scene) - self.potential(previous_scene)


class ResidualInverseObjective(RestorationObjective):
    """RestorationObjective filtered to terms BFS provably cannot satisfy.

    Drop-in reward function for an RL agent: same potential/reward API, but
    only the residual subset of inverse target terms contributes.
    """

    def __init__(self, base: RestorationObjective, residual_keys: set[str]):
        self.operator = base.operator
        self.predicates = base.predicates
        self.terms = [t for t in base.terms if t.key in residual_keys]
        self.residual_keys = set(residual_keys)
