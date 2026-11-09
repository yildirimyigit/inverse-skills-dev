"""The exact-transition planner: the default mode of `hole_planner.plan`.

This module implements the "Minimal Hole Planner" formulation. It judges holes by real
transitions, with the same `Domain` and the same hole action as the relaxed mode, and it never
calls `relaxed_reachable` or `capability_gaps`.

The model

  Library   The library operators are assumed complete and correct under the modelled
            transition semantics, `Domain.transition`. That transition applies an action's own
            add and delete effects together with the domain's world constraints: implications,
            mutex groups and disturbances. An effect a world constraint gives a library action is
            that action's effect, and this module reads library effects only through the
            transition and `Domain.closure`.
  Holes     A hole for a scored literal l adds l and deletes nothing itself. It runs only while
            l is false, from a largest set of robot-relative facts about l's object that the
            library really holds together (`hole_starts`). The world constraints give it its
            other effects, as they do for library actions.

Definitions, with k = `max_delegated`, the learning budget

  C           The candidates: scored literals that are goal literals or preconditions of some
              library action.
  F(H)        H is feasible: the library and holes for H reach the goal, in any order and at any
              length. Decided by exhaustive search (`reachable_states`).
  Minimal     F(H) holds, and F(H - {l}) fails for every l in H.
  U(l)        The candidates strictly upstream of l. The producer graph has an edge from l to u
              when a library action adds l, counting what its additions imply, does not need l,
              and needs u. u is upstream of l when a path leads from l to u and none leads back.
  Admissible  |H| <= k, H is minimal, and no l in H has a replacement: a set R within
              (H - {l}) | U(l) with |R| <= k and F(R).
  Cost        (holes, displaced literals, length), compared lexicographically.

The algorithm. For i = 0, 1, ..., min(k, |C|), collect the admissible sets of size i, and stop at
the first size that has any. For each of those sets, run a uniform-cost search over the library
and its holes, ordered by (displaced, length), with no length limit. Return the cheapest plan with
the equally cheap ones. If no size up to k has an admissible set, return `ExactNoPlan`.

What is guaranteed

  - A returned plan runs under the transition model and reaches the goal.
  - Its holes are subset-minimal: no proper subset of them reaches the goal at any length, so
    every plan for them uses all of them, and the library alone cannot reach the goal. This is
    not minimum cardinality. The preference can choose more holes than the fewest that reach the
    goal, though never more than k.
  - The preference never blocks a plan: if some set of at most k candidates is feasible, an
    admissible set exists. Upstream is a strict partial order, and every replacement moves to
    literals strictly lower in it, so refusals cannot go round in a circle.
  - `ExactNoPlan` is exact relative to the candidate set C, the budget k and the symbolic
    transition model: it is returned if and only if no set of at most k candidates is feasible.
    `feasible_with_all_holes` then says whether holes for every candidate would be.
  - Among the admissible sets of the smallest size, the plan has the least (displaced, length).

What is a design choice, not a fact

  - The upstream preference. It reads positive preconditions only: what a library producer
    needs, not what it deletes, needs absent or interferes with. It is not a complete
    causal-dependency analysis, and it does not show that the preferred skill is easier to learn.
    It decides which subset-minimal set is learned, and its answer can depend on k, because a
    replacement must fit the budget.
  - Ties. Plans of equal cost are all reported, except orderings that pass through the same state
    at the same cost, which are merged into one.

`ExactPlanResult.cost` keeps the five numbers every planning result reports: holes, displaced
literals, literals the holes change, hole steps, length. This mode minimizes the first, second and
fifth, which `ExactPlanResult.planning_cost` gives. The search is exhaustive over finite state
spaces, so it is exponential in the worst case.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass

from inverse_skills.operators.hole_planner import Action, Domain, NoPlan, PlanResult


@dataclass(frozen=True)
class HoleContext:
    """One hole of a plan: what it must do, where it sits, and what needs it."""

    literal: str                      # the ground predicate it must establish
    index: int                        # its position in the plan
    state_before: frozenset[str]      # the symbolic state immediately before it
    used_by: str                      # the first later step that needs what it establishes, "goal"
                                      # if only the goal does, or "none" if neither does: the hole
                                      # is then needed only for its forced effects
    used_at: int | None               # that step's position in the plan; None otherwise
    waiting: tuple[str, ...]          # what needs this literal: the library actions that have it as
                                      # a precondition, and "goal" if the goal needs it
    prefix: tuple[str, ...]           # the plan up to the hole
    suffix: tuple[str, ...]           # the plan after the hole
    # Each library action that establishes the literal, and what happens if it takes the hole's
    # place in this plan. The test that it cannot do the job in any order is minimality.
    library_alternatives: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict:
        return {"literal": self.literal, "index": self.index,
                "state_before": sorted(self.state_before), "used_by": self.used_by,
                "used_at": self.used_at, "waiting": list(self.waiting),
                "prefix": list(self.prefix), "suffix": list(self.suffix),
                "library_alternatives": [list(pair) for pair in self.library_alternatives]}


@dataclass(frozen=True)
class ExactPlanResult(PlanResult):
    """A `PlanResult` that also says why each hole is in the plan.

    `candidates` is the candidate set C. The inherited gap fields stay empty: this mode computes
    no flat set of gaps. `limits_reached` is always empty: there is no length limit, and a plan
    is returned only within the budget.
    """

    hole_contexts: tuple[HoleContext, ...] = ()
    limits_reached: tuple[str, ...] = ()

    @property
    def planning_cost(self) -> tuple[int, int, int]:
        """The cost this mode minimizes: (holes, displaced literals, length)."""
        return (len(self.delegated), self.collateral, len(self.actions))


@dataclass(frozen=True)
class ExactNoPlan(NoPlan):
    """No set of at most `max_delegated` candidates reaches the goal, at any length.

    This is exact relative to the candidate set (`candidates`), the budget and the symbolic
    transition model. `feasible_with_all_holes` says whether holes for every candidate would
    reach the goal: False means no number of candidate holes can; True means more than
    `max_delegated` are needed, and `limits_reached` then names "max_delegated".

    `unmet_needs` are goal literals and library preconditions the library alone never makes
    true; `unsynthesizable_needs` are those without a soft score, which no hole can be offered
    for. `max_length` is 0, because there is no length limit. `unnecessary_holes` and
    `replaceable_holes` are always empty: when some set within the budget is feasible, an
    admissible one exists, so no plan was refused. They and `expanded` stay for callers that
    read every field. The inherited gap fields stay empty, as in `ExactPlanResult`.
    """

    limits_reached: tuple[str, ...] = ()
    max_delegated: int = 0
    max_length: int = 0
    unmet_needs: tuple[str, ...] = ()
    unsynthesizable_needs: tuple[str, ...] = ()
    unnecessary_holes: tuple[str, ...] = ()
    replaceable_holes: tuple[str, ...] = ()
    feasible_with_all_holes: bool = False
    expanded: int = 0


def reachable_states(domain: Domain, start: frozenset[str], holes=()) -> set[frozenset[str]]:
    """Every state reachable from `start` with the library and `holes`, by real transitions,
    in any order. No step limit applies. A hole is usable only while its literal is missing."""
    seen = {start}
    frontier = [start]
    while frontier:
        state = frontier.pop()
        for action in [*domain.actions, *holes]:
            if action.hole_for not in state and domain.applicable(state, action):
                reached = domain.apply(state, action)
                if reached not in seen:
                    seen.add(reached)
                    frontier.append(reached)
    return seen


def library_states(domain: Domain, start: frozenset[str]) -> set[frozenset[str]]:
    """Every state the library alone reaches from `start`."""
    return reachable_states(domain, start)


def hole_starts(domain: Domain, key: str, states) -> list[Action]:
    """The holes for `key`: one for each largest set of robot-relative facts about its
    object that hold together in some state the library really reaches.

    So the library still does all it can before the learned skill takes over, as in
    `Domain.hole`, and a start is never an impossible conjunction: every start was
    really held by the library. Usually there is exactly one.
    """
    about_object = domain.hole(key, domain.robot_relative).pre
    held = {state & about_object for state in states}
    largest = [facts for facts in held if not any(facts < other for other in held)]
    return [domain.hole(key, facts) for facts in sorted(largest, key=sorted)]


def _establishers(domain: Domain, literal: str) -> list[Action]:
    """The library actions that add `literal`, counting what their additions imply. One that
    already needs it does not establish it."""
    return [action for action in domain.actions
            if literal in domain.closure(action.add) and literal not in action.pre]


def _producer_edges(domain: Domain) -> dict[str, set[str]]:
    """The producer graph: an edge from l to u when a library action establishes l and needs u.
    Only positive preconditions are read."""
    edges: dict[str, set[str]] = {}
    for action in domain.actions:
        for literal in domain.closure(action.add) - action.pre:
            edges.setdefault(literal, set()).update(action.pre)
    return edges


def _below(edges: dict[str, set[str]], literal: str) -> set[str]:
    """Every literal that a path of one or more edges reaches from `literal`."""
    found: set[str] = set()
    pending = [literal]
    while pending:
        for following in edges.get(pending.pop(), ()):
            if following not in found:
                found.add(following)
                pending.append(following)
    return found


def upstream(domain: Domain, literal: str) -> set[str]:
    """The literals strictly upstream of `literal`: reached from it in the producer graph, with
    no path back. Members of a cycle are not upstream of each other."""
    edges = _producer_edges(domain)
    return {other for other in _below(edges, literal)
            if other != literal and literal not in _below(edges, other)}


class _HoleSets:
    """What real transitions say about sets of holes, and which sets the definitions admit.
    Each answer comes from an exhaustive search with no step limit."""

    def __init__(self, domain: Domain, start: frozenset[str], goal_positive: set[str],
                 goal_negative: set[str], budget: int):
        self.domain = domain
        self.start = start
        self.goal_positive = goal_positive
        self.goal_negative = goal_negative
        self.budget = budget
        self.library = library_states(domain, start)
        self.needs = goal_positive | {literal for action in domain.actions for literal in action.pre}
        self.candidates = frozenset(self.needs & domain.scored)

        edges = _producer_edges(domain)
        below: dict[str, set[str]] = {}

        def reached_from(literal: str) -> set[str]:
            if literal not in below:
                below[literal] = _below(edges, literal)
            return below[literal]

        self.upstream = {literal: {other for other in reached_from(literal) & self.candidates
                                   if other != literal and literal not in reached_from(other)}
                         for literal in self.candidates}
        self._starts: dict[str, list[Action]] = {}
        self._reached = {frozenset(): self.library}

    def holes(self, literals) -> list[Action]:
        for literal in literals:
            if literal not in self._starts:
                self._starts[literal] = hole_starts(self.domain, literal, self.library)
        return [hole for literal in sorted(literals) for hole in self._starts[literal]]

    def states(self, literals) -> set[frozenset[str]]:
        literals = frozenset(literals)
        if literals not in self._reached:
            self._reached[literals] = reachable_states(self.domain, self.start, self.holes(literals))
        return self._reached[literals]

    def feasible(self, literals) -> bool:
        """F: the goal is reachable with the library and holes for `literals`."""
        return any(self.goal_positive <= state and not (self.goal_negative & state)
                   for state in self.states(literals))

    def minimal(self, hole_set: frozenset[str]) -> bool:
        return self.feasible(hole_set) and not any(self.feasible(hole_set - {literal})
                                                   for literal in hole_set)

    def replaceable(self, hole_set: frozenset[str], literal: str) -> bool:
        """Some set of at most k literals, drawn from the other holes and the candidates upstream
        of `literal`, is feasible. Adding holes never makes the goal unreachable, so only the
        sets of size min(k, |pool|) need a test."""
        pool = sorted((hole_set - {literal}) | self.upstream[literal])
        size = min(self.budget, len(pool))
        return any(self.feasible(chosen) for chosen in itertools.combinations(pool, size))

    def admissible(self, hole_set: frozenset[str]) -> bool:
        return (len(hole_set) <= self.budget and self.minimal(hole_set)
                and not any(self.replaceable(hole_set, literal) for literal in hole_set))

    def smallest_admissible_sets(self) -> list[frozenset[str]]:
        """The admissible sets of the smallest size up to k, or none."""
        for size in range(min(self.budget, len(self.candidates)) + 1):
            found = [frozenset(chosen) for chosen in itertools.combinations(sorted(self.candidates), size)
                     if self.admissible(frozenset(chosen))]
            if found:
                return found
        return []


def _cheapest_plans(domain: Domain, start: frozenset[str], satisfied, holes: list[Action]):
    """Uniform-cost search over the library and `holes`, ordered by (displaced, length), with no
    length limit. A node is a state. Returns the least cost, the plans found at that cost, and how
    many states were expanded. Equally cheap orderings through one state are merged."""
    queue = [((0, 0), 0, start, ())]
    settled: set[frozenset[str]] = set()
    best: tuple[int, int] | None = None
    plans: list[tuple[Action, ...]] = []
    expanded = 0
    counter = 0
    while queue:
        key, _, state, taken = heapq.heappop(queue)
        if best is not None and key > best:
            break
        if state in settled:
            continue
        settled.add(state)
        expanded += 1
        for action in domain.actions + [hole for hole in holes if hole.hole_for not in state]:
            if not domain.applicable(state, action):
                continue
            nxt, displaced = domain.transition(state, action)
            if nxt == state:
                continue
            next_key = (key[0] + len(displaced), key[1] + 1)
            if satisfied(nxt):
                if best is None or next_key < best:
                    best, plans = next_key, []
                if next_key == best:
                    plans.append(taken + (action,))
                continue
            counter += 1
            heapq.heappush(queue, (next_key, counter, nxt, taken + (action,)))
    return best, plans, expanded


def _library_alternatives(domain, before, key, suffix) -> tuple[tuple[str, str], ...]:
    """For the final plan: what happens if each library action establishing `key` runs where
    the hole is, and the rest of the plan follows."""
    reasons = []
    for action in _establishers(domain, key):
        if not domain.applicable(before, action):
            reasons.append((action.name, "cannot run here"))
            continue
        state = domain.apply(before, action)
        for step in suffix:
            if step.hole_for == key:
                continue
            if not domain.applicable(state, step):
                reasons.append((action.name, f"the plan then fails at {step.name}"))
                break
            state = domain.apply(state, step)
        else:
            reasons.append((action.name, "the rest of the plan runs, but the goal is not reached"))
    return tuple(reasons)


def _hole_context(domain, start, steps, index, goal_positive) -> HoleContext:
    before = start
    for step in steps[:index]:
        before = domain.apply(before, step)
    hole = steps[index]
    key = hole.hole_for

    # Follow what the hole established to the first later step that needs it.
    state = domain.apply(before, hole)
    established = (domain.closure(hole.add) - before) & state
    used_by, used_at = None, None
    for later in range(index + 1, len(steps)):
        step = steps[later]
        if established & step.pre:
            used_by, used_at = step.name, later
            break
        state = domain.apply(state, step)
        established &= state
    if used_by is None:
        used_by = "goal" if established & goal_positive else "none"

    waiting = [action.name for action in domain.actions if key in action.pre]
    if key in goal_positive:
        waiting.append("goal")
    return HoleContext(
        literal=key,
        index=index,
        state_before=before,
        used_by=used_by,
        used_at=used_at,
        waiting=tuple(waiting),
        prefix=tuple(step.name for step in steps[:index]),
        suffix=tuple(step.name for step in steps[index + 1:]),
        library_alternatives=_library_alternatives(domain, before, key, steps[index + 1:]),
    )


def _literals_the_holes_change(domain: Domain, start: frozenset[str], steps) -> int:
    """Reported in `cost`, not minimized: the literals each hole step changes, apart from those
    a disturbance displaces."""
    state, changed = start, 0
    for step in steps:
        nxt, displaced = domain.transition(state, step)
        if step.hole_for is not None:
            changed += len((nxt ^ state) - displaced)
        state = nxt
    return changed


def plan_exact(domain: Domain, start, goal_positive, goal_negative=None, *,
               max_delegated: int = 2) -> ExactPlanResult | ExactNoPlan:
    """The cheapest plan whose holes form an admissible set, or an `ExactNoPlan` report, as the
    module docstring defines them. The same as `hole_planner.plan(..., mode="exact")`.
    """
    start = frozenset(start)
    goal_positive = set(goal_positive)
    goal_negative = set(goal_negative or ())

    def satisfied(state: frozenset[str]) -> bool:
        return goal_positive <= state and not (goal_negative & state)

    if satisfied(start):
        return ExactPlanResult((), (), (), 0, 0, (), 0)

    hole_sets = _HoleSets(domain, start, goal_positive, goal_negative, max_delegated)
    admissible = hole_sets.smallest_admissible_sets()
    if not admissible:
        feasible = hole_sets.feasible(hole_sets.candidates)
        unmet = hole_sets.needs - set().union(*hole_sets.library)
        return ExactNoPlan(candidates=tuple(sorted(hole_sets.candidates)),
                           limits_reached=("max_delegated",) if feasible else (),
                           max_delegated=max_delegated,
                           unmet_needs=tuple(sorted(unmet)),
                           unsynthesizable_needs=tuple(sorted(unmet - domain.scored)),
                           feasible_with_all_holes=feasible)

    best: tuple[int, int] | None = None
    tied: list[tuple[Action, ...]] = []
    expanded = 0
    for hole_set in admissible:
        cost, plans, explored = _cheapest_plans(domain, start, satisfied, hole_sets.holes(hole_set))
        expanded += explored
        if best is None or cost < best:
            best, tied = cost, []
        if cost == best:
            tied.extend(steps for steps in plans if steps not in tied)

    chosen = tied[0]
    hole_indices = tuple(i for i, action in enumerate(chosen) if action.hole_for)
    return ExactPlanResult(
        actions=tuple(action.name for action in chosen),
        steps=chosen,
        delegated=tuple(sorted({action.hole_for for action in chosen if action.hole_for})),
        collateral=best[0],
        hole_literals=_literals_the_holes_change(domain, start, chosen),
        hole_indices=hole_indices,
        expanded=expanded,
        candidates=tuple(sorted(hole_sets.candidates)),
        alternatives=tuple(tied[1:]),
        hole_contexts=tuple(_hole_context(domain, start, chosen, i, goal_positive)
                            for i in hole_indices),
    )
