"""Signed literals, and the one place where a score becomes truth.

A literal is an atomic predicate with a sign: `p` says the predicate must be true, `not p` says
it must be false. Preconditions, effects and goals are sets of literals, and neither sign is
special: each is a condition on the truth value of one predicate.

A symbolic state is the set of atomic predicates that are true; every other predicate is false
(the closed world). So `p` holds in a state when p is in it, and `not p` when p is not.

Code that works on states may store a set of literals as two sets of atoms, the positive and the
negative ones (`Action.pre` and `Action.pre_absent`, `Action.add` and `Action.delete`). Those are
projections of one set of literals, made for speed; `split` and `join` convert between the two.

Truth from a score, in two senses that are kept apart:

  Boolean    an atom is true when its score is at least TRUE_AT (0.5), the predicate's own
             boundary, where its margin is zero. The symbolic state the planner works on uses it.
  robust     `p` is robustly satisfied when the score is at least ROBUST_TRUE (0.8), and `not p`
             when it is at most ROBUST_FALSE (0.2). A score in between is evidence of neither.
             Operator extraction and every check on the robot use it: preconditions, effects,
             learned targets, fences, handoffs and the final goal.

A missing or non-finite score is never read as false: `validate_scores` raises instead.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

TRUE_AT = 0.5          # Boolean truth: the score at which a predicate's margin is zero
ROBUST_TRUE = 0.8      # robust evidence that a predicate is true
ROBUST_FALSE = 0.2     # robust evidence that a predicate is false


class MeasurementError(ValueError):
    """A predicate that should have been measured has no valid score."""


@dataclass(frozen=True, order=True)
class Literal:
    """An atomic predicate `atom` with a sign: true if `positive`, false otherwise."""

    atom: str
    positive: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.atom, str) or not self.atom:
            raise ValueError(f"a literal needs a predicate name, got {self.atom!r}")
        if self.atom.startswith(("not ", "¬", "!")):
            raise ValueError(f"the sign of a literal is its `positive` field, not part of its name: {self.atom!r}")
        if not isinstance(self.positive, bool):
            raise ValueError(f"`positive` must be True or False, got {self.positive!r}")

    @property
    def polarity(self) -> str:
        return "positive" if self.positive else "negative"

    def complement(self) -> "Literal":
        """`not p` for `p`, and `p` for `not p`."""
        return Literal(self.atom, not self.positive)

    def holds(self, state: Iterable[str]) -> bool:
        """Does the literal hold in a symbolic state, the set of atoms that are true?"""
        return (self.atom in state) == self.positive

    def signed_score(self, score: float) -> float:
        """The literal's satisfaction score in [-1, 1]: 2V - 1 for `p`, 1 - 2V for `not p`."""
        return 2.0 * score - 1.0 if self.positive else 1.0 - 2.0 * score

    def boolean_holds(self, score: float) -> bool:
        return (score >= TRUE_AT) == self.positive

    def robustly_holds(self, score: float) -> bool:
        return score >= ROBUST_TRUE if self.positive else score <= ROBUST_FALSE

    def __str__(self) -> str:
        return self.atom if self.positive else f"not {self.atom}"

    def to_dict(self) -> dict:
        return {"atom": self.atom, "polarity": self.polarity}

    @classmethod
    def from_dict(cls, data) -> "Literal":
        """Read a literal written by `to_dict`. A bare string is a positive literal: that is how
        positive-only artifacts, written before literals had a sign, name one."""
        if isinstance(data, str):
            return cls(data, True)
        polarity = data["polarity"]
        if polarity not in ("positive", "negative"):
            raise ValueError(f"unknown polarity {polarity!r}")
        return cls(data["atom"], polarity == "positive")


def pos(atom: str) -> Literal:
    return Literal(atom, True)


def neg(atom: str) -> Literal:
    return Literal(atom, False)


def literal_set(literals: Iterable[Literal]) -> frozenset[Literal]:
    """The literals as one set, duplicates merged. A set holding both `p` and `not p` asks for
    the impossible, so it is rejected."""
    result = frozenset(literals)
    contradictions = sorted(l.atom for l in result if l.positive and l.complement() in result)
    if contradictions:
        raise ValueError("contradictory literals, both p and not p: " + ", ".join(contradictions))
    return result


def join(positive: Iterable[str] = (), negative: Iterable[str] = ()) -> frozenset[Literal]:
    """The literals whose positive atoms are `positive` and negative atoms are `negative`."""
    return literal_set([pos(a) for a in positive] + [neg(a) for a in negative])


def split(literals: Iterable[Literal]) -> tuple[frozenset[str], frozenset[str]]:
    """The positive atoms and the negative atoms of a set of literals."""
    literals = literal_set(literals)
    return (frozenset(l.atom for l in literals if l.positive),
            frozenset(l.atom for l in literals if not l.positive))


def satisfied(literals: Iterable[Literal], state: Iterable[str]) -> bool:
    """Does every literal hold in the symbolic state?"""
    state = frozenset(state)
    return all(l.holds(state) for l in literals)


def validate_scores(scores: Mapping[str, float], keys: Iterable[str] = ()) -> None:
    """Every score is a finite number, and every key in `keys` has one."""
    missing = sorted(set(keys) - set(scores))
    if missing:
        raise MeasurementError("no score for " + ", ".join(missing))
    invalid = sorted(k for k, v in scores.items()
                     if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v))
    if invalid:
        raise MeasurementError("invalid score for " + ", ".join(invalid))


def symbolic_state(scores: Mapping[str, float]) -> frozenset[str]:
    """The Boolean symbolic state a measurement supports: the atoms scoring at least TRUE_AT."""
    validate_scores(scores)
    return frozenset(key for key, score in scores.items() if score >= TRUE_AT)


def robust_value(score: float) -> str:
    """'true', 'false' or 'uncertain': what the score is robust evidence of."""
    return "true" if score >= ROBUST_TRUE else "false" if score <= ROBUST_FALSE else "uncertain"


def robustly_unmet(literals: Iterable[Literal], scores: Mapping[str, float]) -> list[Literal]:
    """The literals the measured scores do not robustly satisfy. Every literal must be measured."""
    literals = sorted(literal_set(literals))
    validate_scores(scores, (l.atom for l in literals))
    return [l for l in literals if not l.robustly_holds(scores[l.atom])]


def boolean_unmet(literals: Iterable[Literal], scores: Mapping[str, float]) -> list[Literal]:
    """The literals the measured scores do not satisfy in the Boolean sense."""
    literals = sorted(literal_set(literals))
    validate_scores(scores, (l.atom for l in literals))
    return [l for l in literals if not l.boolean_holds(scores[l.atom])]


def describe(literal: Literal, score: float) -> str:
    """A report line: the literal, its score, and what the score is evidence of."""
    return f"{literal} (score {score:.3f}, robustly {robust_value(score)})"
