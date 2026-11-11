"""A symbolic operator learned from demonstrations, in signed literals.

An operator has one precondition set Pre and one effect set Eff, both sets of signed literals:
`p` (the predicate is true, or made true) and `not p` (false, or made false). Its restoration goal
is G = Pre ∪ complement(Eff): what held before the operator ran, including what it changed. The
strict extractor only extracts an effect whose complement held before every demonstration, so for
its operators complement(Eff) ⊆ Pre and G = Pre (`effects_complement_in_preconditions`).

Serialized operators keep the fields positive-only readers know: `add_effects` holds the atoms Eff
makes true and `delete_effects` the atoms it makes false, and `inverse_target_terms` is the goal.
Beside them, `effects` lists Eff as signed literals and `restoration_goal` repeats G, and
`schema_version` says which format wrote them. An operator written before literals had signs
(positive preconditions only) reads back unchanged: `LearnedOperator.from_dict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable
from typing import Literal as OneOf

from inverse_skills.operators.literals import Literal as SignedLiteral

Polarity = OneOf["positive", "negative"]
SCHEMA_VERSION = "signed_literals_v1"


def _term_sort_key(term) -> tuple:
    polarity_rank = 0 if term.polarity == "positive" else 1
    return (term.key, polarity_rank, round(float(term.weight), 12))


def _canonicalize_terms(terms: Iterable["PredicateTerm"]) -> list["PredicateTerm"]:
    """The terms sorted, duplicates merged (the larger weight is kept). Both signs of one
    predicate in one list is a contradiction, and is rejected."""
    dedup: dict[tuple[str, Polarity], PredicateTerm] = {}
    for term in terms:
        key = (term.key, term.polarity)
        if key not in dedup or term.weight > dedup[key].weight:
            dedup[key] = term
    contradictions = sorted(k for k, polarity in dedup if polarity == "positive" and (k, "negative") in dedup)
    if contradictions:
        raise ValueError("contradictory terms, both p and not p: " + ", ".join(contradictions))
    return sorted(dedup.values(), key=_term_sort_key)


@dataclass(frozen=True)
class PredicateTerm:
    """A signed literal with a weight: the weakest evidence the demonstrations gave for it."""

    key: str
    weight: float = 1.0
    polarity: Polarity = "positive"

    def __post_init__(self) -> None:
        if self.polarity not in ("positive", "negative"):
            raise ValueError(f"unknown polarity {self.polarity!r} for {self.key!r}")
        SignedLiteral(self.key)          # the name must be a plain predicate, with no sign in it

    @property
    def literal(self) -> SignedLiteral:
        return SignedLiteral(self.key, self.polarity == "positive")

    @classmethod
    def of(cls, literal: SignedLiteral, weight: float = 1.0) -> "PredicateTerm":
        return cls(key=literal.atom, weight=weight, polarity=literal.polarity)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "weight": float(self.weight),
            "polarity": self.polarity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PredicateTerm":
        return cls(key=data["key"], weight=float(data.get("weight", 1.0)),
                   polarity=data.get("polarity", "positive"))


@dataclass
class LearnedOperator:
    """Pre and Eff in signed literals. `add_effects` and `delete_effects` hold Eff's atoms by
    sign, as positive terms; `effects` is the signed view of the two."""

    skill_name: str
    preconditions: list[PredicateTerm] = field(default_factory=list)
    add_effects: list[PredicateTerm] = field(default_factory=list)
    delete_effects: list[PredicateTerm] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.preconditions = _canonicalize_terms(self.preconditions)
        self.add_effects = _canonicalize_terms(self.add_effects)
        self.delete_effects = _canonicalize_terms(self.delete_effects)
        for term in self.add_effects + self.delete_effects:
            if term.polarity != "positive":
                raise ValueError(f"add_effects and delete_effects name atoms, so their terms are positive; "
                                 f"got {term.polarity} {term.key!r}")
        both = sorted({t.key for t in self.add_effects} & {t.key for t in self.delete_effects})
        if both:
            raise ValueError("contradictory effects, made both true and false: " + ", ".join(both))

    @property
    def effects(self) -> list[PredicateTerm]:
        """Eff as signed literals: `p` for an atom made true, `not p` for one made false."""
        signed = [PredicateTerm(t.key, t.weight, "positive") for t in self.add_effects]
        signed += [PredicateTerm(t.key, t.weight, "negative") for t in self.delete_effects]
        return _canonicalize_terms(signed)

    def precondition_literals(self) -> frozenset[SignedLiteral]:
        return frozenset(t.literal for t in self.preconditions)

    def effect_literals(self) -> frozenset[SignedLiteral]:
        return frozenset(t.literal for t in self.effects)

    def effects_complement_in_preconditions(self) -> bool:
        """Did the complement of every effect hold before the operator, as Pre says? Always true
        for the strict extractor, which is why its goal is Pre itself."""
        pre = self.precondition_literals()
        return all(effect.complement() in pre for effect in self.effect_literals())

    def restoration_goal(self) -> list[PredicateTerm]:
        """G = Pre ∪ complement(Eff): every condition that held before the operator ran."""
        terms: list[PredicateTerm] = list(self.preconditions)
        terms.extend(PredicateTerm.of(t.literal.complement(), t.weight) for t in self.effects)
        try:
            return _canonicalize_terms(terms)
        except ValueError as error:
            raise ValueError(f"the restoration goal of {self.skill_name!r} contradicts itself: {error}") from error

    def inverse_target_terms(self) -> list[PredicateTerm]:
        """The restoration goal under the name positive-only readers know."""
        return self.restoration_goal()

    def to_dict(self) -> dict:
        goal = [term.to_dict() for term in self.restoration_goal()]
        return {
            "schema_version": SCHEMA_VERSION,
            "skill_name": self.skill_name,
            "preconditions": [term.to_dict() for term in self.preconditions],
            "effects": [term.to_dict() for term in self.effects],
            "add_effects": [term.to_dict() for term in self.add_effects],
            "delete_effects": [term.to_dict() for term in self.delete_effects],
            "restoration_goal": goal,
            "inverse_target_terms": goal,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearnedOperator":
        """Read an operator written by `to_dict`, in this format or the earlier positive-only one."""
        return cls(skill_name=data["skill_name"],
                   preconditions=[PredicateTerm.from_dict(t) for t in data.get("preconditions", ())],
                   add_effects=[PredicateTerm.from_dict(t) for t in data.get("add_effects", ())],
                   delete_effects=[PredicateTerm.from_dict(t) for t in data.get("delete_effects", ())],
                   metadata=dict(data.get("metadata", {})))
