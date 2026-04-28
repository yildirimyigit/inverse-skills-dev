from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Any, Literal

from inverse_skills.core import SceneGraph
from inverse_skills.logging import ForwardRollout
from inverse_skills.operators.schema import (
    LearnedOperator,
    PredicateTerm,
    _canonicalize_terms,
)


Polarity = Literal["positive", "negative"]
_TERM_RE = re.compile(r"^(?P<name>[^()]+)\((?P<args>.*)\)$")


@dataclass(frozen=True)
class TemplatePredicateTerm:
    key: str
    weight: float = 1.0
    polarity: Polarity = "positive"

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "weight": float(self.weight),
            "polarity": self.polarity,
        }


@dataclass
class ParameterizedOperatorTemplate:
    skill_name: str
    parameters: list[dict[str, str]] = field(default_factory=list)
    preconditions: list[TemplatePredicateTerm] = field(default_factory=list)
    add_effects: list[TemplatePredicateTerm] = field(default_factory=list)
    delete_effects: list[TemplatePredicateTerm] = field(default_factory=list)
    inverse_target_terms: list[TemplatePredicateTerm] = field(default_factory=list)
    bindings: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.preconditions = _canonicalize_terms(self.preconditions)
        self.add_effects = _canonicalize_terms(self.add_effects)
        self.delete_effects = _canonicalize_terms(self.delete_effects)
        self.inverse_target_terms = _canonicalize_terms(self.inverse_target_terms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "parameters": self.parameters,
            "preconditions": [t.to_dict() for t in _canonicalize_terms(self.preconditions)],
            "add_effects": [t.to_dict() for t in _canonicalize_terms(self.add_effects)],
            "delete_effects": [t.to_dict() for t in _canonicalize_terms(self.delete_effects)],
            "inverse_target_terms": [t.to_dict() for t in _canonicalize_terms(self.inverse_target_terms)],
            "bindings": self.bindings,
            "metadata": self.metadata,
        }


def parse_predicate_key(key: str) -> tuple[str, list[str]]:
    match = _TERM_RE.match(key)
    if match is None:
        raise ValueError(f"Could not parse predicate key: {key}")
    name = match.group("name")
    args_raw = match.group("args").strip()
    args = [] if not args_raw else [a.strip() for a in args_raw.split(",")]
    return name, args


class RoleBindingInferer:
    def infer(self, rollout: ForwardRollout) -> dict[str, str]:
        first = rollout.first()
        last = rollout.last()

        object_name = self._infer_changed_object(first, last)
        src_region = self._infer_region_for_position(first.get_object(object_name).pose.position, first)
        dst_region = self._infer_region_for_position(last.get_object(object_name).pose.position, last)

        bindings = {
            object_name: "?obj",
            src_region: "?src",
            dst_region: "?dst",
        }
        return bindings

    @staticmethod
    def _infer_changed_object(first: SceneGraph, last: SceneGraph) -> str:
        best_name: str | None = None
        best_dist = -math.inf
        for name in sorted(set(first.objects.keys()) & set(last.objects.keys())):
            a = first.get_object(name).pose.position
            b = last.get_object(name).pose.position
            dist = float(((a - b) ** 2).sum() ** 0.5)
            if dist > best_dist:
                best_name = name
                best_dist = dist
        if best_name is None:
            raise ValueError("Could not infer manipulated object")
        return best_name

    @staticmethod
    def _infer_region_for_position(position, scene: SceneGraph) -> str:
        best_region: str | None = None
        best_margin = -math.inf
        for name, region in scene.regions.items():
            margin = float(region.signed_margin(position))
            if margin > best_margin:
                best_region = name
                best_margin = margin
        if best_region is None:
            raise ValueError("Could not infer region")
        return best_region


class OperatorParameterizer:
    def __init__(self, binding_inferer: RoleBindingInferer | None = None):
        self.binding_inferer = binding_inferer or RoleBindingInferer()

    def parameterize(self, operator: LearnedOperator, exemplar_rollout: ForwardRollout) -> ParameterizedOperatorTemplate:
        bindings = self.binding_inferer.infer(exemplar_rollout)
        parameters = [
            {"slot": "?obj", "type": "object"},
            {"slot": "?src", "type": "region"},
            {"slot": "?dst", "type": "region"},
        ]
        return ParameterizedOperatorTemplate(
            skill_name=operator.skill_name,
            parameters=parameters,
            preconditions=[self._map_term(t, bindings) for t in operator.preconditions],
            add_effects=[self._map_term(t, bindings) for t in operator.add_effects],
            delete_effects=[self._map_term(t, bindings) for t in operator.delete_effects],
            inverse_target_terms=[self._map_term(t, bindings) for t in operator.inverse_target_terms()],
            bindings=bindings,
            metadata={**operator.metadata, "parameterizer": "role_binding_v0"},
        )

    @staticmethod
    def _map_term(term: PredicateTerm, bindings: dict[str, str]) -> TemplatePredicateTerm:
        name, args = parse_predicate_key(term.key)
        mapped_args = [bindings.get(arg, arg) for arg in args]
        template_key = f"{name}({','.join(mapped_args)})"
        return TemplatePredicateTerm(key=template_key, weight=term.weight, polarity=term.polarity)
