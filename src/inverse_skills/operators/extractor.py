from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Literal

from inverse_skills.logging.rollout import ForwardRollout
from inverse_skills.operators.literals import ROBUST_FALSE, ROBUST_TRUE, neg, pos, validate_scores
from inverse_skills.operators.schema import LearnedOperator, PredicateTerm
from inverse_skills.predicates.base import PredicateRegistry


@dataclass(frozen=True)
class OperatorExtractionConfig:
    precondition_threshold: float = 0.80
    effect_delta_threshold: float = 0.35
    min_abs_effect_delta: float = 0.15
    # "min_max" asks every demonstration to show the condition or the transition, for either
    # sign (_extract_min_max). "mean" thresholds the scores averaged over the demonstrations and
    # extracts positive preconditions only: the earlier rule, a historical comparison mode kept
    # for the workshop paper's experiments. New experiments use min_max.
    mode: Literal["mean", "min_max"] = "min_max"
    true_threshold: float = ROBUST_TRUE    # min_max: a score at or above this is robustly true
    false_threshold: float = ROBUST_FALSE  # min_max: a score at or below this is robustly false


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
            for scene_eval in (first_eval, last_eval):
                validate_scores({key: result.score for key, result in scene_eval.items()}, self.predicates.keys())
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
        """The strict rule, the same for both signs: every demonstration must show the condition
        or the transition. True means a score >= true_threshold, false a score <= false_threshold.

          Pre:  p      p is true at the start of every demonstration
                not p  p is false at the start of every demonstration
          Eff:  p      p is false at the start and true at the end of every demonstration
                not p  p is true at the start and false at the end of every demonstration

        A predicate whose start is neither true in every demonstration nor false in every one
        gives no precondition: a score between the thresholds is evidence of neither. So every
        effect's complement is a precondition, and the restoration goal is Pre itself.

        The rule assumes every demonstration exhibits each effect. STRIPS allows an add effect
        that already held before some executions, or a delete effect that was already false;
        this rule does not extract such effects.
        """
        true_at, false_at = self.config.true_threshold, self.config.false_threshold
        if not false_at < true_at:
            raise ValueError(f"false_threshold {false_at} must lie below true_threshold {true_at}")
        scores: dict[str, dict] = {}
        preconditions: list[PredicateTerm] = []
        add_effects: list[PredicateTerm] = []
        delete_effects: list[PredicateTerm] = []

        for key in self.predicates.keys():
            starts, ends = start_scores[key], end_scores[key]
            start_min, start_max = min(starts), max(starts)
            end_min, end_max = min(ends), max(ends)

            true_at_every_start = start_min >= true_at
            false_at_every_start = start_max <= false_at
            true_at_every_end = end_min >= true_at
            false_at_every_end = end_max <= false_at

            # Each weight is the weakest evidence any demonstration gave for the term.
            precondition = effect = None
            if true_at_every_start:
                precondition = pos(key)
                preconditions.append(PredicateTerm(key=key, weight=start_min, polarity="positive"))
            elif false_at_every_start:
                precondition = neg(key)
                preconditions.append(PredicateTerm(key=key, weight=1.0 - start_max, polarity="negative"))
            if false_at_every_start and true_at_every_end:
                effect = pos(key)
                add_effects.append(PredicateTerm(key=key, weight=end_min - start_max, polarity="positive"))
            if true_at_every_start and false_at_every_end:
                effect = neg(key)
                delete_effects.append(PredicateTerm(key=key, weight=start_min - end_max, polarity="positive"))

            scores[key] = {
                "start_min": start_min, "start_max": start_max, "end_min": end_min, "end_max": end_max,
                "support": {"demonstrations": num_rollouts,
                            "true_at_start": sum(v >= true_at for v in starts),
                            "false_at_start": sum(v <= false_at for v in starts),
                            "true_at_end": sum(v >= true_at for v in ends),
                            "false_at_end": sum(v <= false_at for v in ends)},
                "precondition": None if precondition is None else precondition.to_dict(),
                "effect": None if effect is None else effect.to_dict(),
                "changed_by_the_skill": effect is not None,
            }

        operator = LearnedOperator(
            skill_name=skill_name,
            preconditions=preconditions,
            add_effects=add_effects,
            delete_effects=delete_effects,
            metadata={"num_rollouts": num_rollouts, "extractor": "first_last_score_min_max_signed_v1",
                      "true_threshold": true_at, "false_threshold": false_at},
        )
        if not operator.effects_complement_in_preconditions():
            raise RuntimeError("an extracted effect's complement is not a precondition; the strict rule guarantees it")
        return OperatorExtractionResult(operator=operator, scores=scores)
