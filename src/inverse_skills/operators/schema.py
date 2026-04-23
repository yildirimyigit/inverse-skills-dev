from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Polarity = Literal["positive", "negative"]


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

    def inverse_target_terms(self) -> list[PredicateTerm]:
        terms: list[PredicateTerm] = []
        terms.extend(self.preconditions)
        terms.extend(self.delete_effects)
        for term in self.add_effects:
            terms.append(PredicateTerm(key=term.key, weight=term.weight, polarity="negative"))
        return terms

    def to_dict(self) -> dict:
        return {
            "skill_name": self.skill_name,
            "preconditions": [term.to_dict() for term in self.preconditions],
            "add_effects": [term.to_dict() for term in self.add_effects],
            "delete_effects": [term.to_dict() for term in self.delete_effects],
            "inverse_target_terms": [term.to_dict() for term in self.inverse_target_terms()],
            "metadata": self.metadata,
        }
