"""Is a learned skill about to start where the plan says it starts?

A learned operator is modelled from its own executions, so every execution has to begin at the
handoff the plan predicts. The executive cannot tell: it checks what each step promises (a cube
lifted, a cube placed), not what a step should leave alone. So a step can disturb an object nothing
mentions, report success, and hand the skill a state the plan never predicted. An execution from
there is not a demonstration of the operator being modelled.

This module checks the handoff before the skill runs. What it checks is read off the plan and the
domain's facts about its world, before anything about the skill's execution is known:

  the start        literals the hole is told it starts from
  the transition   literals the hole's modelled transition changes. Those it removes must hold and
                   those it establishes must not: a transition begins in its own pre-state.
  the fences       literals the plan relies on across the hole, true or false

Other literals of the predicted state mean nothing to this skill, and nothing is asked of them.
Only scored literals can be measured. True and false are the symbolic state's (`symbolic_state`),
as everywhere else in the planner and the executive.

The check never looks at what the skill then does. A scene that fails it is replaced by the next
scene and recorded, in the same way the forward skill's re-rolls are; a scene that passes is kept,
whatever comes of it. Nothing in this module names a predicate of any particular domain.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from inverse_skills.operators.hole_planner import Action, Domain, symbolic_state

# As the forward skill's re-rolls and the hole environment's prefix attempts.
MAX_ATTEMPTS = 5
RE_ROLL_STRIDE = 10_000


@dataclass(frozen=True)
class HandoffConditions:
    """What must be measured when the skill takes over: `hold` true, `absent` false."""

    hold: frozenset[str]
    absent: frozenset[str]

    def to_dict(self) -> dict:
        return {"hold": sorted(self.hold), "absent": sorted(self.absent)}


def handoff_conditions(domain: Domain, start: frozenset[str], spec: dict) -> HandoffConditions:
    """The conditions on the handoff of the hole in `spec` (a `hole_spec`), from the plan's start."""
    before = frozenset(start)
    for step in spec["prefix"]:
        before = domain.apply(before, Action.from_dict(step))
    hole = Action(name=spec["hole"], pre=frozenset(spec["start"]), add=frozenset({spec["achieve"]}),
                  hole_for=spec["achieve"])
    after = domain.apply(before, hole)
    hold = set(spec["start"]) | set(spec["preserve"]) | (before - after)
    absent = set(spec["preserve_absent"]) | (after - before)
    return HandoffConditions(frozenset(hold & domain.scored), frozenset(absent & domain.scored))


def handoff_violations(conditions: HandoffConditions, scores: dict[str, float]) -> list[str]:
    """Each condition the measured scores break, with the score that breaks it."""
    measured = symbolic_state(scores)
    reasons = [f"{p} should hold, score {scores[p]:.4f}" for p in sorted(conditions.hold - measured)]
    reasons += [f"{q} should not hold, score {scores[q]:.4f}" for q in sorted(conditions.absent & measured)]
    return reasons


@dataclass
class Prepared:
    """What the forward skill and the known prefix left behind in one scene."""

    scores: dict[str, float] = field(default_factory=dict)   # measured at the handoff
    context: object = None                                   # what the demonstration continues from
    failed_stage: str | None = None                          # "forward" or "prefix", if it got no further
    reason: str = ""


@dataclass(frozen=True)
class Attempt:
    """One scene tried for one demonstration. `stage` says how far it got: "forward" or "prefix"
    if that failed, "handoff" if the handoff was invalid, "valid" if the skill was then run."""

    seed: int
    number: int
    scene_seed: int
    stage: str
    reasons: tuple[str, ...] = ()
    scores: tuple[tuple[str, float], ...] = ()               # the conditions' literals, as measured

    def to_dict(self) -> dict:
        return {"seed": self.seed, "number": self.number, "scene_seed": self.scene_seed, "stage": self.stage,
                "reasons": list(self.reasons), "scores": dict(self.scores)}


class HandoffsExhausted(RuntimeError):
    """No scene gave a valid handoff within the limit. The demonstration is not silently dropped."""

    def __init__(self, seed: int, attempts: list[Attempt]):
        self.seed = seed
        self.attempts = attempts
        tried = [a for a in attempts if a.seed == seed]
        details = "; ".join(f"scene {a.scene_seed}: {a.stage}: {', '.join(a.reasons)}" for a in tried)
        super().__init__(f"seed {seed}: no valid handoff in {len(tried)} attempts ({details})")


def collect_demonstrations(seeds: Iterable[int], conditions: HandoffConditions,
                           prepare: Callable[[int], Prepared], demonstrate: Callable[[int, object], object],
                           max_attempts: int = MAX_ATTEMPTS) -> tuple[list, list[Attempt]]:
    """One demonstration per seed, each from a valid handoff, and a record of every scene tried.

    `prepare(scene_seed)` runs the forward skill and the known prefix in that scene.
    `demonstrate(scene_seed, context)` runs the skill from there; it is called only for a valid
    handoff, and what it returns is kept whatever it shows. A seed that gets no valid handoff within
    `max_attempts` scenes raises `HandoffsExhausted`.
    """
    demonstrations, attempts = [], []
    for seed in seeds:
        for number in range(max_attempts):
            scene_seed = seed + number * RE_ROLL_STRIDE
            prepared = prepare(scene_seed)
            attempt = _judge(seed, number, scene_seed, prepared, conditions)
            attempts.append(attempt)
            if attempt.stage == "valid":
                demonstrations.append(demonstrate(scene_seed, prepared.context))
                break
        else:
            raise HandoffsExhausted(seed, attempts)
    return demonstrations, attempts


def _judge(seed: int, number: int, scene_seed: int, prepared: Prepared, conditions: HandoffConditions) -> Attempt:
    if prepared.failed_stage is not None:
        return Attempt(seed, number, scene_seed, prepared.failed_stage, (prepared.reason,))
    literals = sorted(conditions.hold | conditions.absent)
    scores = tuple((p, prepared.scores[p]) for p in literals)
    violations = handoff_violations(conditions, prepared.scores)
    return Attempt(seed, number, scene_seed, "handoff" if violations else "valid", tuple(violations), scores)


def summarize_attempts(attempts: list[Attempt]) -> dict:
    """The raw rates, before any retry: how often a scene really gave a valid handoff."""
    stages = [a.stage for a in attempts]
    reached = stages.count("handoff") + stages.count("valid")
    return {"attempts": len(attempts), "failed_forward": stages.count("forward"), "failed_prefix": stages.count("prefix"),
            "reached_the_handoff": reached, "invalid_handoffs": stages.count("handoff"), "valid_handoffs": stages.count("valid"),
            "raw_handoff_validity": stages.count("valid") / reached if reached else None}
