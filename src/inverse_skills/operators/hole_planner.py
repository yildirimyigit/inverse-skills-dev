"""A STRIPS planner whose library may be incomplete, with holes for the rest.

A hole is an abstract action standing for a policy nobody has learned yet: it
establishes one predicate, and it carries the default schema agreed for this
work — preconditions {tcp_near(object)}, add effects {the predicate}. The
planner may place holes anywhere in the plan, so the learned operator is not
condemned to run last.

Plans are ordered lexicographically by

    (predicates delegated to holes, literals holes change, holes, length)

The first component is the rule "use library actions as much as possible".
The second is what stops a single hole from swallowing the task: a hole that
puts a cube straight back on its source changes more of the world than one
that merely frees it, so the planner prefers the smaller delegation even when
the resulting plan is longer. Ties are broken by placing holes as late as the
plan allows, which is decided among optimal plans rather than during search.

The domain carries two facts about the world that no action owns: `implications`
(one literal entails another, e.g. a cube on a reachable slot is graspable) and
`mutex_groups` (at most one literal per group holds, e.g. a cube sits on one
slot). Both apply to every action, holes included, so a hole cannot cheat by
adding a literal whose consequences it ignores.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Action:
    """A ground action. `hole_for` marks the predicate a hole is asked to establish."""

    name: str
    pre: frozenset[str] = frozenset()
    pre_absent: frozenset[str] = frozenset()
    add: frozenset[str] = frozenset()
    delete: frozenset[str] = frozenset()
    hole_for: str | None = None

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class PlanResult:
    actions: tuple[str, ...]
    delegated: tuple[str, ...]
    hole_literals: int
    hole_indices: tuple[int, ...]
    expanded: int
    alternatives: tuple[tuple[str, ...], ...] = ()

    @property
    def cost(self) -> tuple[int, int, int, int]:
        return (len(self.delegated), self.hole_literals,
                len(self.hole_indices), len(self.actions))


@dataclass
class Domain:
    actions: list[Action]
    hole_candidates: list[str] = field(default_factory=list)
    mutex_groups: list[frozenset[str]] = field(default_factory=list)
    implications: dict[str, frozenset[str]] = field(default_factory=dict)

    def applicable(self, state: frozenset[str], action: Action) -> bool:
        return action.pre <= state and not (action.pre_absent & state)

    def apply(self, state: frozenset[str], action: Action) -> frozenset[str]:
        result = set(state) - action.delete
        added: set[str] = set()
        pending = list(action.add)
        while pending:
            literal = pending.pop()
            if literal in added:
                continue
            added.add(literal)
            result.add(literal)
            pending.extend(self.implications.get(literal, ()))
        for literal in added:
            for group in self.mutex_groups:
                if literal in group:
                    result -= group - {literal}
        return frozenset(result)

    def hole(self, key: str) -> Action:
        """The default hole schema: reach the object, then establish the predicate."""
        pre: frozenset[str] = frozenset()
        if "(" in key:
            args = key[key.index("(") + 1:key.rindex(")")].split(",")
            if args and args[0]:
                pre = frozenset({f"tcp_near({args[0]})"})
        return Action(name=f"HOLE[{key}]", pre=pre, add=frozenset({key}), hole_for=key)


def _lateness(hole_indices: tuple[int, ...]) -> tuple[int, int]:
    """Bigger is later. Compared only among plans of equal cost."""
    if not hole_indices:
        return (0, 0)
    return (min(hole_indices), sum(hole_indices))


def plan(domain: Domain, start: frozenset[str], goal_positive: set[str],
         goal_negative: set[str] | None = None, *, max_delegated: int = 2,
         max_length: int = 10, latest_holes: bool = True) -> PlanResult | None:
    """Cheapest plan under the lexicographic cost, or None within the limits."""
    goal_negative = set(goal_negative or ())
    actions = list(domain.actions) + [domain.hole(key) for key in domain.hole_candidates]

    def satisfied(state: frozenset[str]) -> bool:
        return goal_positive <= state and not (goal_negative & state)

    if satisfied(start):
        return PlanResult((), (), 0, (), 0)

    counter = 0
    settled: dict[frozenset[str], tuple] = {}
    goals: list[tuple[tuple, tuple[Action, ...]]] = []
    best_key: tuple | None = None
    expanded = 0
    queue = [((0, 0, 0, 0), counter, start, (), frozenset())]

    while queue:
        key, _, state, taken, delegated = heapq.heappop(queue)
        if best_key is not None and key > best_key:
            break
        if state in settled and settled[state] <= key:
            continue
        settled[state] = key
        expanded += 1
        if key[3] >= max_length:
            continue
        for action in actions:
            if not domain.applicable(state, action):
                continue
            nxt = domain.apply(state, action)
            if nxt == state:
                continue
            next_delegated = delegated
            changed = 0
            if action.hole_for is not None:
                next_delegated = delegated | {action.hole_for}
                changed = len(nxt ^ state)
                if len(next_delegated) > max_delegated:
                    continue
            next_key = (len(next_delegated), key[1] + changed,
                        key[2] + (1 if action.hole_for else 0), key[3] + 1)
            next_taken = taken + (action,)
            if satisfied(nxt):
                if best_key is None or next_key < best_key:
                    best_key = next_key
                goals.append((next_key, next_taken))
                continue
            counter += 1
            heapq.heappush(queue, (next_key, counter, nxt, next_taken, next_delegated))

    if best_key is None:
        return None

    tied = [taken for key, taken in goals if key == best_key]
    ranked = sorted(
        tied,
        key=lambda taken: _lateness(tuple(i for i, a in enumerate(taken) if a.hole_for)),
        reverse=latest_holes,
    )
    chosen = ranked[0]
    hole_indices = tuple(i for i, a in enumerate(chosen) if a.hole_for)
    return PlanResult(
        actions=tuple(a.name for a in chosen),
        delegated=tuple(sorted({a.hole_for for a in chosen if a.hole_for})),
        hole_literals=best_key[1],
        hole_indices=hole_indices,
        expanded=expanded,
        alternatives=tuple(tuple(a.name for a in t) for t in ranked[1:]),
    )


def symbolic_state(scores: dict[str, float], threshold: float = 0.8) -> frozenset[str]:
    """The literals a measured scene supports: {p : V_p >= threshold}."""
    return frozenset(key for key, score in scores.items() if score >= threshold)
