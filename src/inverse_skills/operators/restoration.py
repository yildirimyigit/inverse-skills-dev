from __future__ import annotations

from inverse_skills.core.scene import SceneGraph
from inverse_skills.operators.schema import LearnedOperator, PredicateTerm
from inverse_skills.predicates.base import PredicateRegistry


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
