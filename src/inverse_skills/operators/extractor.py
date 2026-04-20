from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from inverse_skills.logging.rollout import ForwardRollout
from inverse_skills.operators.schema import LearnedOperator, PredicateTerm
from inverse_skills.predicates.base import PredicateRegistry


@dataclass(frozen=True)
class OperatorExtractionConfig:
    precondition_threshold: float = 0.80
    effect_delta_threshold: float = 0.35
    min_abs_effect_delta: float = 0.15


@dataclass
class OperatorExtractionResult:
    operator: LearnedOperator
    scores: dict[str, dict[str, float]]


class OperatorExtractor:
    """Extracts a first simple operator from successful forward rollouts.

    This intentionally uses only first and final scenes. It gives us a robust baseline before
    we add temporal segmentation, causal filtering, or learned role binding.
    """

    def __init__(self, predicate_registry: PredicateRegistry, config: OperatorExtractionConfig | None = None):
        self.predicates = predicate_registry
        self.config = config or OperatorExtractionConfig()

    def extract(self, skill_name: str, rollouts: list[ForwardRollout]) -> OperatorExtractionResult:
        if not rollouts:
            raise ValueError("At least one rollout is required")

        start_scores: dict[str, list[float]] = {key: [] for key in self.predicates.keys()}
        end_scores: dict[str, list[float]] = {key: [] for key in self.predicates.keys()}

        for rollout in rollouts:
            if rollout.skill_name != skill_name:
                raise ValueError(f"Expected skill '{skill_name}', got rollout skill '{rollout.skill_name}'")
            first_eval = self.predicates.evaluate_all(rollout.first())
            last_eval = self.predicates.evaluate_all(rollout.last())
            for key in self.predicates.keys():
                start_scores[key].append(first_eval[key].score)
                end_scores[key].append(last_eval[key].score)

        scores: dict[str, dict[str, float]] = {}
        preconditions: list[PredicateTerm] = []
        add_effects: list[PredicateTerm] = []
        delete_effects: list[PredicateTerm] = []

        for key in self.predicates.keys():
            start_mean = float(mean(start_scores[key]))
            end_mean = float(mean(end_scores[key]))
            delta = end_mean - start_mean
            abs_delta = abs(delta)
            scores[key] = {
                "start_mean": start_mean,
                "end_mean": end_mean,
                "delta": float(delta),
                "abs_delta": float(abs_delta),
            }

            if start_mean >= self.config.precondition_threshold:
                preconditions.append(PredicateTerm(key=key, weight=start_mean, polarity="positive"))
            if delta >= self.config.effect_delta_threshold and abs_delta >= self.config.min_abs_effect_delta:
                add_effects.append(PredicateTerm(key=key, weight=abs_delta, polarity="positive"))
            if -delta >= self.config.effect_delta_threshold and abs_delta >= self.config.min_abs_effect_delta:
                delete_effects.append(PredicateTerm(key=key, weight=abs_delta, polarity="positive"))

        operator = LearnedOperator(
            skill_name=skill_name,
            preconditions=preconditions,
            add_effects=add_effects,
            delete_effects=delete_effects,
            metadata={"num_rollouts": len(rollouts), "extractor": "first_last_score_delta_v0"},
        )
        return OperatorExtractionResult(operator=operator, scores=scores)
