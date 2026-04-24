from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Iterable

Polarity = Literal["positive", "negative"]


def _term_sort_key(term) -> tuple:
    polarity_rank = 0 if term.polarity == "positive" else 1
    return (term.key, polarity_rank, round(float(term.weight), 12))


def _canonicalize_terms(terms: Iterable["PredicateTerm"]) -> list["PredicateTerm"]:
    dedup: dict[tuple[str, Polarity], PredicateTerm] = {}
    for term in terms:
        key = (term.key, term.polarity)
        if key not in dedup or term.weight > dedup[key].weight:
            dedup[key] = term
    return sorted(dedup.values(), key=_term_sort_key)


@dataclass(frozen=True)
class PredicateTerm:
    key: str
    weight: float = 1.0
    polarity: Polarity = "positive"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "weight": float(self.weight),
            "polarity": self.polarity,
        }


@dataclass
class LearnedOperator:
    skill_name: str
    preconditions: list[PredicateTerm] = field(default_factory=list)
    add_effects: list[PredicateTerm] = field(default_factory=list)
    delete_effects: list[PredicateTerm] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.preconditions = _canonicalize_terms(self.preconditions)
        self.add_effects = _canonicalize_terms(self.add_effects)
        self.delete_effects = _canonicalize_terms(self.delete_effects)

    def inverse_target_terms(self) -> list[PredicateTerm]:
        terms: list[PredicateTerm] = []
        terms.extend(self.preconditions)
        terms.extend(self.delete_effects)
        for term in self.add_effects:
            terms.append(
                PredicateTerm(
                    key=term.key,
                    weight=term.weight,
                    polarity="negative",
                )
            )
        return _canonicalize_terms(terms)

    def to_dict(self) -> dict:
        return {
            "skill_name": self.skill_name,
            "preconditions": [term.to_dict() for term in _canonicalize_terms(self.preconditions)],
            "add_effects": [term.to_dict() for term in _canonicalize_terms(self.add_effects)],
            "delete_effects": [term.to_dict() for term in _canonicalize_terms(self.delete_effects)],
            "inverse_target_terms": [term.to_dict() for term in self.inverse_target_terms()],
            "metadata": self.metadata,
        }
