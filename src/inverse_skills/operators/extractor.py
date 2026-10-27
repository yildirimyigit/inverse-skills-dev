from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Literal

from inverse_skills.logging.rollout import ForwardRollout
from inverse_skills.operators.schema import LearnedOperator, PredicateTerm
from inverse_skills.predicates.base import PredicateRegistry


@dataclass(frozen=True)
class OperatorExtractionConfig:
    precondition_threshold: float = 0.80
    effect_delta_threshold: float = 0.35
    min_abs_effect_delta: float = 0.15
    # "mean" thresholds the scores averaged over the demonstrations. "min_max" asks
    # every demonstration to show the condition or the transition (_extract_min_max).
    mode: Literal["mean", "min_max"] = "mean"
    true_threshold: float = 0.80   # min_max: a score at or above this counts as true
    false_threshold: float = 0.20  # min_max: a score at or below this counts as false


@dataclass
class OperatorExtractionResult:
    operator: LearnedOperator
    scores: dict[str, dict[str, float]]


class OperatorExtractor:
    def __init__(self, predicate_registry: PredicateRegistry, config: OperatorExtractionConfig | None = None):
        self.predicates = predicate_registry
        self.config = config or OperatorExtractionConfig()
        if self.config.mode not in ("mean", "min_max"):
            raise ValueError(f"Unknown extraction mode '{self.config.mode}'")

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

        if self.config.mode == "min_max":
            return self._extract_min_max(skill_name, start_scores, end_scores, len(rollouts))

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

    def _extract_min_max(
        self,
        skill_name: str,
        start_scores: dict[str, list[float]],
        end_scores: dict[str, list[float]],
        num_rollouts: int,
    ) -> OperatorExtractionResult:
        """The conservative rule: every demonstration must show the condition or the transition.

          Pre: p is true at the start of every demonstration.
          Add: p is false at the start and true at the end of every demonstration.
          Del: p is true at the start and false at the end of every demonstration.

        True means a score >= true_threshold, false a score <= false_threshold.
        The rule assumes every demonstration exhibits each effect. STRIPS allows an
        add effect that already held before some executions, or a delete effect that
        was already false; this rule does not extract such effects.
        """
        true_at, false_at = self.config.true_threshold, self.config.false_threshold
        scores: dict[str, dict[str, float]] = {}
        preconditions: list[PredicateTerm] = []
        add_effects: list[PredicateTerm] = []
        delete_effects: list[PredicateTerm] = []

        for key in self.predicates.keys():
            start_min, start_max = min(start_scores[key]), max(start_scores[key])
            end_min, end_max = min(end_scores[key]), max(end_scores[key])
            scores[key] = {"start_min": start_min, "start_max": start_max, "end_min": end_min, "end_max": end_max}

            true_at_every_start = start_min >= true_at
            false_at_every_start = start_max <= false_at
            true_at_every_end = end_min >= true_at
            false_at_every_end = end_max <= false_at

            # Each weight is the weakest evidence any demonstration gave for the term.
            if true_at_every_start:
                preconditions.append(PredicateTerm(key=key, weight=start_min, polarity="positive"))
            if false_at_every_start and true_at_every_end:
                add_effects.append(PredicateTerm(key=key, weight=end_min - start_max, polarity="positive"))
            if true_at_every_start and false_at_every_end:
                delete_effects.append(PredicateTerm(key=key, weight=start_min - end_max, polarity="positive"))

        operator = LearnedOperator(
            skill_name=skill_name,
            preconditions=preconditions,
            add_effects=add_effects,
            delete_effects=delete_effects,
            metadata={"num_rollouts": num_rollouts, "extractor": "first_last_score_min_max_v0"},
        )
        return OperatorExtractionResult(operator=operator, scores=scores)
