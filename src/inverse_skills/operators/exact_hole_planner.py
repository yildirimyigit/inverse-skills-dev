"""The exact-transition planner: the default mode of `hole_planner.plan`.

This module implements the "Minimal Hole Planner" formulation. It judges holes by real
transitions, with the same `Domain` and the same hole action as the relaxed mode, and it never
calls `relaxed_reachable` or `capability_gaps`.

The model

  Library   The library operators are assumed complete and correct under the modelled
            transition semantics, `Domain.transition`. That transition applies an action's own
            add and delete effects, what its additions imply, the mutex partners its additions
            displace, and the disturbances its changes trigger. An effect a world constraint gives
            a library action is that action's effect, and this module reads library effects only
            through the transition and `Domain.closure`.
  Validity  The domain's implications and mutex groups are constraints on states. A transition
            is valid when its result keeps all of them, holds every atom the action adds and
            none it deletes; otherwise the action, or hole, is unavailable in that state
            (`Domain.successor`). Nothing is deleted to repair a constraint. The start must keep
            them too, or the planner refuses it (`InconsistentState`). Every state reached is
            therefore consistent.
  Literals  Every literal is signed: `p` (the atom p is true) or `not p` (it is false), and the
            two signs are treated alike throughout (`literals.py`). A state is the set of true
            atoms; a goal and a precondition set are sets of signed literals.
  Holes     A hole for a scored literal l establishes l and nothing else itself: a hole for `p`
            adds p, a hole for `not p` deletes p. It runs only while l does not hold, and only
            where its transition is valid, from an initiation context: a largest set of
            robot-relative facts about l's object that hold together in some library-reachable
            state where l does not hold (`hole_starts`), whatever l's sign. If l holds in every
            library-reachable state there is no context and no hole for l, and the result says
            so (`no_context`). The world constraints give a hole its other effects, as they do
            for library actions.

Definitions, with k = `max_delegated`, the learning budget

  C           The candidates: signed literals on scored atoms that are goal literals or
              preconditions, positive or negative, of some library action.
  F(H)        H is feasible: the library and holes for H reach the goal, in any order and at any
              length. Decided by exhaustive search (`reachable_states`).
  Minimal     F(H) holds, and F(H - {l}) fails for every l in H.
  U(l)        The candidates strictly upstream of l. The producer graph has an edge from l to u
              when a library action can establish l and does not need l, and needs u. An action
              establishes `p` by adding p, counting what its additions imply; it establishes
              `not p` by deleting p, by adding a mutex partner of p, or by changing a disturbance
              trigger that displaces p. It needs its preconditions of both signs. u is upstream
              of l when a path leads from l to u and none leads back.
  Admissible  |H| <= k, H is minimal, and no l in H has a replacement: a set R within
              (H - {l}) | U(l) with |R| <= k and F(R).
  Cost        (holes, displaced literals, length), compared lexicographically.

The algorithm. For i = 0, 1, ..., min(k, |C|), collect the admissible sets of size i, and stop at
the first size that has any. For each of those sets, run a uniform-cost search over the library
and its holes, ordered by (displaced, length), with no length limit. Return the cheapest plan with
the equally cheap ones. If no size up to k has an admissible set, return `ExactNoPlan`.

What is guaranteed

  - A returned plan runs under the transition model, every state along it is consistent with
    the declared constraints, and it reaches the goal. This is a symbolic guarantee: that a
    hole's literal can be established physically, or that a library action really has its
    modelled effects, is not shown by planning.
  - Its holes are subset-minimal: no proper subset of them reaches the goal at any length, so
    every plan for them uses all of them, and the library alone cannot reach the goal. This is
    not minimum cardinality. The preference can choose more holes than the fewest that reach the
    goal, though never more than k.
  - The preference never blocks a plan: if some set of at most k candidates is feasible, an
    admissible set exists. Upstream is a strict partial order, and every replacement moves to
    literals strictly lower in it, so refusals cannot go round in a circle.
  - `ExactNoPlan` is exact relative to the candidate set C, the budget k, the initiation
    contexts and the symbolic transition model: it is returned if and only if no set of at most
    k candidates is feasible.
    `feasible_with_all_holes` then says whether holes for every candidate would be.
  - Among the admissible sets of the smallest size, the plan has the least (displaced, length).

What is a design choice, not a fact

  - The upstream preference. It reads what a library producer needs, of both signs, and what
    it can establish, of both signs. It is not a complete causal-dependency analysis, and it does
    not show that the preferred skill is easier to learn.
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
from inverse_skills.operators.literals import Literal, join, neg, pos, satisfied


@dataclass(frozen=True)
class HoleContext:
    """One hole of a plan: what it must do, where it sits, and what needs it."""

    literal: Literal                  # the signed literal it must establish
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
        return {"literal": self.literal.to_dict(), "index": self.index,
                "state_before": sorted(self.state_before), "used_by": self.used_by,
                "used_at": self.used_at, "waiting": list(self.waiting),
                "prefix": list(self.prefix), "suffix": list(self.suffix),
                "library_alternatives": [list(pair) for pair in self.library_alternatives]}


@dataclass(frozen=True)
class ExactPlanResult(PlanResult):
    """A `PlanResult` that also says why each hole is in the plan.

    `candidates` is the candidate set C, and `no_context` the candidates that hold in every
    library-reachable state, so no initiation context and no hole exists for them. The inherited
    gap fields stay empty: this mode computes no flat set of gaps. `limits_reached` is always
    empty: there is no length limit, and a plan is returned only within the budget.
    """

    hole_contexts: tuple[HoleContext, ...] = ()
    limits_reached: tuple[str, ...] = ()
    no_context: tuple[Literal, ...] = ()

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

    `unmet_needs` are goal literals and library preconditions, of either sign, that hold in no
    state the library alone reaches; `unsynthesizable_needs` are those on atoms without a soft
    score, which no hole can be offered for. `no_context`, as in `ExactPlanResult`, lists the
    candidates with no initiation context. `max_length` is 0, because there is no length limit. `unnecessary_holes` and
    `replaceable_holes` are always empty: when some set within the budget is feasible, an
    admissible one exists, so no plan was refused. They and `expanded` stay for callers that
    read every field. The inherited gap fields stay empty, as in `ExactPlanResult`.
    """

    limits_reached: tuple[str, ...] = ()
    max_delegated: int = 0
    max_length: int = 0
    unmet_needs: tuple[Literal, ...] = ()
    unsynthesizable_needs: tuple[Literal, ...] = ()
    unnecessary_holes: tuple[str, ...] = ()
    replaceable_holes: tuple[str, ...] = ()
    feasible_with_all_holes: bool = False
    expanded: int = 0
    no_context: tuple[Literal, ...] = ()


def reachable_states(domain: Domain, start: frozenset[str], holes=()) -> set[frozenset[str]]:
    """Every state reachable from `start` with the library and `holes`, by valid transitions,
    in any order. No step limit applies. A hole is usable only while its literal does not hold.
    A start that breaks a declared constraint is refused (`InconsistentState`)."""
    domain.check_state(start, "the start state")
    seen = {start}
    frontier = [start]
    while frontier:
        state = frontier.pop()
        for action in [*domain.actions, *holes]:
            outcome = domain.successor(state, action)
            if outcome is not None:
                reached = outcome[0]
                if reached not in seen:
                    seen.add(reached)
                    frontier.append(reached)
    return seen


def library_states(domain: Domain, start: frozenset[str]) -> set[frozenset[str]]:
    """Every state the library alone reaches from `start`."""
    return reachable_states(domain, start)


def hole_starts(domain: Domain, literal: Literal, states) -> list[Action]:
    """The holes for `literal`: one for each initiation context, a largest set of robot-relative
    facts about its object that hold together in some state of `states` (the library-reachable
    states) where `literal` does not hold yet.

    So the library still does all it can before the learned skill takes over, as in
    `Domain.hole`, and a context is never an impossible conjunction, nor one held only where the
    literal is already established: every context was really held by the library in a state the
    hole could start from. The rule is the same for both signs. If the literal holds in every
    state there is no context, and no hole: none is invented. A hole still runs only where its
    context holds, its literal does not, and its transition is valid (`Domain.enabled`).
    Usually there is exactly one context.
    """
    about_object = domain.about_object(literal)
    held = {state & about_object for state in states if not literal.holds(state)}
    largest = [facts for facts in held if not any(facts < other for other in held)]
    return [domain.hole(literal, facts) for facts in sorted(largest, key=sorted)]


def established_by(domain: Domain, action: Action) -> frozenset[Literal]:
    """The signed literals the action can establish by its modelled transition, other than those
    it already needs: `p` for each atom it adds, counting implications; `not p` for each atom it
    deletes, each mutex partner of what it adds, and each atom a disturbance it can trigger
    displaces."""
    added = domain.closure(action.add)
    removed = domain.removes(action)
    displaced = set()
    for disturbance in domain.disturbances:
        if disturbance.trigger in added or disturbance.trigger in removed:
            displaced |= disturbance.clobbers
    found = {pos(atom) for atom in added}
    found |= {neg(atom) for atom in (removed | displaced) - added}
    return frozenset(literal for literal in found if literal not in action.precondition_literals)


def _establishers(domain: Domain, literal: Literal) -> list[Action]:
    """The library actions that can establish `literal` (`established_by`)."""
    return [action for action in domain.actions if literal in established_by(domain, action)]


def _producer_edges(domain: Domain) -> dict[Literal, set[Literal]]:
    """The producer graph: an edge from l to u when a library action can establish l and needs u.
    Preconditions of both signs are read."""
    edges: dict[Literal, set[Literal]] = {}
    for action in domain.actions:
        for literal in established_by(domain, action):
            edges.setdefault(literal, set()).update(action.precondition_literals)
    return edges


def _below(edges: dict[Literal, set[Literal]], literal: Literal) -> set[Literal]:
    """Every literal that a path of one or more edges reaches from `literal`."""
    found: set[str] = set()
    pending = [literal]
    while pending:
        for following in edges.get(pending.pop(), ()):
            if following not in found:
                found.add(following)
                pending.append(following)
    return found


def upstream(domain: Domain, literal: Literal) -> set[Literal]:
    """The literals strictly upstream of `literal`: reached from it in the producer graph, with
    no path back. Members of a cycle are not upstream of each other."""
    edges = _producer_edges(domain)
    return {other for other in _below(edges, literal)
            if other != literal and literal not in _below(edges, other)}


class _HoleSets:
    """What real transitions say about sets of holes, and which sets the definitions admit.
    Each answer comes from an exhaustive search with no step limit."""

    def __init__(self, domain: Domain, start: frozenset[str], goal: frozenset[Literal], budget: int):
        self.domain = domain
        self.start = start
        self.goal = goal
        self.budget = budget
        self.library = library_states(domain, start)
        self.needs = set(goal) | {literal for action in domain.actions for literal in action.precondition_literals}
        self.candidates = frozenset(literal for literal in self.needs if literal.atom in domain.scored)

        edges = _producer_edges(domain)
        below: dict[str, set[str]] = {}

        def reached_from(literal: str) -> set[str]:
            if literal not in below:
                below[literal] = _below(edges, literal)
            return below[literal]

        self.upstream = {literal: {other for other in reached_from(literal) & self.candidates
                                   if other != literal and literal not in reached_from(other)}
                         for literal in self.candidates}
        self._starts: dict[Literal, list[Action]] = {}
        self._reached = {frozenset(): self.library}

    def holes(self, literals) -> list[Action]:
        for literal in literals:
            if literal not in self._starts:
                self._starts[literal] = hole_starts(self.domain, literal, self.library)
        return [hole for literal in sorted(literals) for hole in self._starts[literal]]

    def no_context(self) -> tuple[Literal, ...]:
        """The candidates with no initiation context: they hold in every library-reachable state."""
        return tuple(sorted(literal for literal in self.candidates if not self.holes([literal])))

    def states(self, literals) -> set[frozenset[str]]:
        literals = frozenset(literals)
        if literals not in self._reached:
            self._reached[literals] = reachable_states(self.domain, self.start, self.holes(literals))
        return self._reached[literals]

    def feasible(self, literals) -> bool:
        """F: the goal is reachable with the library and holes for `literals`."""
        return any(satisfied(self.goal, state) for state in self.states(literals))

    def minimal(self, hole_set: frozenset[Literal]) -> bool:
        return self.feasible(hole_set) and not any(self.feasible(hole_set - {literal})
                                                   for literal in hole_set)

    def replaceable(self, hole_set: frozenset[Literal], literal: Literal) -> bool:
        """Some set of at most k literals, drawn from the other holes and the candidates upstream
        of `literal`, is feasible. Adding holes never makes the goal unreachable, so only the
        sets of size min(k, |pool|) need a test."""
        pool = sorted((hole_set - {literal}) | self.upstream[literal])
        size = min(self.budget, len(pool))
        return any(self.feasible(chosen) for chosen in itertools.combinations(pool, size))

    def admissible(self, hole_set: frozenset[Literal]) -> bool:
        return (len(hole_set) <= self.budget and self.minimal(hole_set)
                and not any(self.replaceable(hole_set, literal) for literal in hole_set))

    def smallest_admissible_sets(self) -> list[frozenset[Literal]]:
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
        for action in domain.actions + list(holes):
            outcome = domain.successor(state, action)
            if outcome is None:
                continue
            nxt, displaced = outcome
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


def _library_alternatives(domain, before, literal, suffix) -> tuple[tuple[str, str], ...]:
    """For the final plan: what happens if each library action establishing `literal` runs where
    the hole is, and the rest of the plan follows."""
    reasons = []
    for action in _establishers(domain, literal):
        outcome = domain.successor(before, action)
        if outcome is None:
            reasons.append((action.name, "cannot run here"))
            continue
        state = outcome[0]
        for step in suffix:
            if step.hole_for == literal:
                continue
            outcome = domain.successor(state, step)
            if outcome is None:
                reasons.append((action.name, f"the plan then fails at {step.name}"))
                break
            state = outcome[0]
        else:
            reasons.append((action.name, "the rest of the plan runs, but the goal is not reached"))
    return tuple(reasons)


def _hole_context(domain, start, steps, index, goal) -> HoleContext:
    before = start
    for step in steps[:index]:
        before = domain.apply(before, step)
    hole = steps[index]
    literal = hole.hole_for

    # Follow what the hole established to the first later step that needs it: what it adds,
    # counting implications, or the atom it deletes. What a mutex or a disturbance removes on
    # the way is a forced effect, not what the hole was placed for.
    state = domain.apply(before, hole)
    established = {pos(atom) for atom in (domain.closure(hole.add) - before) & state}
    established |= {neg(atom) for atom in (hole.delete & before) - state}
    used_by, used_at = None, None
    for later in range(index + 1, len(steps)):
        step = steps[later]
        if established & step.precondition_literals:
            used_by, used_at = step.name, later
            break
        state = domain.apply(state, step)
        established = {l for l in established if l.holds(state)}
    if used_by is None:
        used_by = "goal" if established & goal else "none"

    waiting = [action.name for action in domain.actions if literal in action.precondition_literals]
    if literal in goal:
        waiting.append("goal")
    return HoleContext(
        literal=literal,
        index=index,
        state_before=before,
        used_by=used_by,
        used_at=used_at,
        waiting=tuple(waiting),
        prefix=tuple(step.name for step in steps[:index]),
        suffix=tuple(step.name for step in steps[index + 1:]),
        library_alternatives=_library_alternatives(domain, before, literal, steps[index + 1:]),
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

    The goal is one set of signed literals, passed as its two projections: the atoms that must
    be true and the atoms that must be false. A goal asking for both p and not p is rejected.
    """
    start = frozenset(start)
    domain.check_state(start, "the start state")
    goal = join(goal_positive, goal_negative or ())

    def reaches_goal(state: frozenset[str]) -> bool:
        return satisfied(goal, state)

    if reaches_goal(start):
        return ExactPlanResult((), (), (), 0, 0, (), 0)

    hole_sets = _HoleSets(domain, start, goal, max_delegated)
    admissible = hole_sets.smallest_admissible_sets()
    if not admissible:
        feasible = hole_sets.feasible(hole_sets.candidates)
        unmet = {need for need in hole_sets.needs
                 if not any(need.holds(state) for state in hole_sets.library)}
        return ExactNoPlan(candidates=tuple(sorted(hole_sets.candidates)),
                           limits_reached=("max_delegated",) if feasible else (),
                           max_delegated=max_delegated,
                           unmet_needs=tuple(sorted(unmet)),
                           unsynthesizable_needs=tuple(sorted(l for l in unmet if l.atom not in domain.scored)),
                           feasible_with_all_holes=feasible,
                           no_context=hole_sets.no_context())

    best: tuple[int, int] | None = None
    tied: list[tuple[Action, ...]] = []
    expanded = 0
    for hole_set in admissible:
        cost, plans, explored = _cheapest_plans(domain, start, reaches_goal, hole_sets.holes(hole_set))
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
        hole_contexts=tuple(_hole_context(domain, start, chosen, i, goal) for i in hole_indices),
        no_context=hole_sets.no_context(),
    )
