from __future__ import annotations

import math
from typing import Iterable

from inverse_skills.core.scene import SceneGraph
from inverse_skills.operators.schema import LearnedOperator, PredicateTerm
from inverse_skills.predicates.base import Predicate, PredicateRegistry


def signed_score_reward(
    scene: SceneGraph,
    terms: Iterable[tuple],
    saturate: bool = True,
) -> float:
    """Dense reward derived directly from the predicate framework.

    Terminology, so that no word names two different objects:

        margin  m_p(s)   in R, metres    <- geometry
        score   V_p(s)   in [0, 1]       <- squash by temperature T_p
        signed  V^_p(s)  in [-1, +1]     <- sign_p * (2*V_p - 1)
        score

    Each term is `(predicate, sign, mode)`. For each, evaluate the predicate on
    `scene` and add its *signed score* to the total, clipped according to `mode`.

      • `sign=+1` for predicates that should hold in the inverse target;
        `sign=-1` for predicates that should be violated (negated add-effects).
      • `mode="bipolar"` → contribution in [-1, +1]. Use this for the *active
        residual* predicate the symbolic prefix did not fully restore. The agent
        gets gradient toward satisfaction AND bonus for over-satisfaction.
      • `mode="fence"` → contribution in [-1, 0]. Use this for predicates the
        symbolic prefix already restored: violations cost reward, non-violation
        is silent. This is the literal content of "preconditions and
        delete-effects should hold" — no perverse incentive to over-saturate
        a fence and ignore the residual.

    There is NO reward scale hyperparameter. The reward is the predicate's own
    score V_p, mapped onto the signed axis:

        contribution = sign * (2 * V_p(s) - 1) = sign * tanh(margin / T_p)

    the two being identical because V_p = (1 + tanh(margin/T_p)) / 2. The single
    temperature T_p therefore fixes the reward shape as well as the score, and
    "the reward is the score" holds by construction — they cannot drift apart.
    The map x -> 2x-1 is the unique affine, order-preserving bijection
    [0,1] -> [-1,+1] sending the decision boundary V_p = 1/2 to 0, so there is no
    freedom in it.

    Trade-off, stated plainly: T_p controls both how sharp the predicate is for
    symbolic thresholding AND how far the RL gradient reaches (slope 1/T_p at the
    boundary, saturating past ~3*T_p). Decoupling them would need a second
    parameter, at which point the reward would no longer BE the score.

    With `saturate=False`, the contribution is the raw normalized margin
    `sign * margin / T_p` — unbounded, and a margin rather than a score. Used
    only for the "no tanh" ablation.
    """
    total = 0.0
    for term in terms:
        if len(term) != 3:
            raise ValueError(
                f"Each term must be (predicate, sign, mode); got {term!r}. "
                f"The per-term reward scale was removed — the predicate's own "
                f"temperature now sets the reward shape."
            )
        predicate, sign, mode = term
        result = predicate.evaluate(scene)
        if saturate:
            # == sign * tanh(margin / T_p); computed from the score so the
            # identity is manifest in the code, not just in the comments.
            x = float(sign) * (2.0 * float(result.score) - 1.0)
        else:
            temp = max(float(result.temperature), 1e-6)
            x = float(sign) * float(result.margin) / temp
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

    def term_value(self, term: PredicateTerm, scene: SceneGraph) -> float:
        """V_p(s): the predicate's own score with polarity applied, in [0, 1].

        This is the quantity thresholds are defined over — the fence/active
        partition (V_p(s_h) >= theta) and BFS term reachability. Keep it
        unweighted: extraction weights are data-dependent (mean start score or
        effect delta), so a satisfied predicate carrying weight 0.78 must not
        be read as "only 78% restored".
        """
        score = self.predicates.get(term.key).evaluate(scene).score
        if term.polarity == "negative":
            score = 1.0 - score
        return float(score)

    def term_values(self, scene: SceneGraph) -> dict[str, float]:
        return {term.key: self.term_value(term, scene) for term in self.terms}

    def term_score(self, term: PredicateTerm, scene: SceneGraph) -> float:
        """Weighted contribution of a term to `potential`."""
        return float(term.weight * self.term_value(term, scene))

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
