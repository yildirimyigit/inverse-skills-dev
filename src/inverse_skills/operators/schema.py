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
        return {"key": self.key, "weight": float(self.weight), "polarity": self.polarity}

    @classmethod
    def from_dict(cls, data: dict) -> "PredicateTerm":
        return cls(key=data["key"], weight=float(data.get("weight", 1.0)), polarity=data.get("polarity", "positive"))


@dataclass
class LearnedOperator:
    skill_name: str
    preconditions: list[PredicateTerm] = field(default_factory=list)
    add_effects: list[PredicateTerm] = field(default_factory=list)
    delete_effects: list[PredicateTerm] = field(default_factory=list)
    roles: dict[str, str] = field(default_factory=dict)
    continuous_slots: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def inverse_target_terms(self) -> list[PredicateTerm]:
        """Terms that define the inverse restoration objective.

        Preconditions and delete effects must become true again.
        Add effects should become false.
        Duplicate keys are merged by keeping the strongest positive request before adding negatives.
        """
        merged: dict[tuple[str, str], PredicateTerm] = {}
        for term in [*self.preconditions, *self.delete_effects]:
            merged[(term.key, "positive")] = PredicateTerm(term.key, max(term.weight, merged.get((term.key, "positive"), term).weight), "positive")
        for term in self.add_effects:
            merged[(term.key, "negative")] = PredicateTerm(term.key, term.weight, "negative")
        return list(merged.values())

    def to_dict(self) -> dict:
        return {
            "skill_name": self.skill_name,
            "preconditions": [term.to_dict() for term in self.preconditions],
            "add_effects": [term.to_dict() for term in self.add_effects],
            "delete_effects": [term.to_dict() for term in self.delete_effects],
            "roles": self.roles,
            "continuous_slots": self.continuous_slots,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearnedOperator":
        return cls(
            skill_name=data["skill_name"],
            preconditions=[PredicateTerm.from_dict(item) for item in data.get("preconditions", [])],
            add_effects=[PredicateTerm.from_dict(item) for item in data.get("add_effects", [])],
            delete_effects=[PredicateTerm.from_dict(item) for item in data.get("delete_effects", [])],
            roles=data.get("roles", {}),
            continuous_slots=data.get("continuous_slots", {}),
            metadata=data.get("metadata", {}),
        )
