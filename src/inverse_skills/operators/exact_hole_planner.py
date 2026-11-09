"""The exact-transition planner: the default mode of `hole_planner.plan`.

The relaxed mode first decides where holes may go (`capability_gaps`, computed
with delete effects ignored) and then searches. A delete interaction can hide a gap
from that first step, and the planner then reports no plan. This module decides with
real transitions instead, with the same `Domain`, the same lexicographic cost and the
same hole action. It never calls `relaxed_reachable` or `capability_gaps`.

A hole stands for a skill somebody must learn, so the holes of a plan must pass two
separate tests. The first is a fact about the problem. The second is a design choice
about what to learn, and it is made before the cost is looked at. "What is exact, and
what is not" below keeps the two apart.

Exact feasibility and necessity. Run the library and holes for a set of literals R, in
every order, with every delete effect, and no step limit:

  Feasible(R)      The goal can be reached that way.
  Necessary(l, R)  The goal cannot be reached that way once l is taken out of R.

If every literal of R is necessary, no subset of R solves the problem, so R is
subset-minimal. It says nothing about other sets: in a chain r -> p -> goal, {r}, {p} and
{goal} all pass, and they all have one literal.

Causal preference. Literal u is upstream of l if a library way of establishing l depends
on u and u does not depend on l. A way of establishing l is an action that adds it, or one
that clears something such an action needs absent. It depends on what it needs, and on
what it uses up, since restoring that is also a repair. See `upstream`. Inside a cycle
nothing is upstream of anything, because a cycle has no unique root.

  Upstream repair  A hole for l is refused if the goal can be reached without it, given
                   holes for the plan's other literals and for every learnable literal
                   upstream of l. That alternative is judged by real transitions like any
                   other, so a repair that cannot really work refuses nothing.
  Last resort      A hole for a goal literal the library can establish replaces an action
                   that achieves the goal, so it comes last. It is refused if the goal can be
                   reached by holes for literals that are not such goal literals and do not
                   depend on it. Such a goal literal also never counts as a repair for a literal
                   that is not one.
  Waiting          A hole for a goal literal is also refused if an action establishing it
                   cannot run as things are, but would run if its prerequisites could be
                   learned, including ones that cannot. The goal is not what is missing, so
                   the planner reports it rather than work around the library. This is the
                   one test that uses holes nobody could learn, and they never enter a plan.

So a goal hole stays possible if nothing establishes the literal, if the actions that do
cannot run even then, or if they run but destroy other goal literals and no repair
elsewhere solves that. An upstream literal that cannot be learned is no repair, so a hole
can go at a learnable literal downstream of it.

The other rules are about plans:

  Offered          In a state of the search, a hole is offered for a scored literal that
                   the goal, or a library action that can contribute to the goal, is still
                   missing. "Can contribute" is a backward walk over actions' preconditions
                   and effects (`relevant_actions`). It says nothing about what can be reached.
  Used             A later relevant step, or the goal, must use what the hole established
                   before it is lost. A hole nothing uses is not accepted.

Among the plans that pass, the search returns the cheapest under the lexicographic cost.

Hole starts use `Domain.hole` unchanged, but its `reachable` argument comes from the
states the library really reaches from the start, not from the relaxation (see
`hole_starts`). The two modes therefore give a hole the same start unless the
relaxation claims a fact the library cannot really establish, or two facts the
library cannot hold together. In that second case the relaxed start can never be
satisfied, and here the hole gets one start for each set of facts that do hold together.

What is exact, and what is not.

Exact reachability. These are facts about the problem, from exhaustive search of the real
transitions, and they do not depend on `max_length` or `max_delegated`:
  - A returned plan runs and reaches the goal.
  - Its literals are subset-minimal: no smaller set of them solves the problem at any length,
    and if it has holes, the library alone cannot solve the problem at any length.
  - `feasible_with_all_holes` False proves the offered holes cannot solve the problem.

Causal preference. These are design choices. They decide which holes are chosen, and they
prove nothing about what can be solved or learned:
  - Upstream repair and Last resort refuse a hole only if real holes that can be learned reach
    the goal by real transitions. No chain of such refusals leads back to where it began, so
    if some set of holes can reach the goal, one that passes these two tests can too.
  - Keeping the library's operators is why an upstream repair is preferred. It does not mean
    the chosen skill is easier to learn, or that it can be learned at all. With `use` needing
    p and q, and `make_q` consuming p, make_q -> HOLE[p] -> use is chosen although
    HOLE[q] -> use is cheaper.
  - Waiting is a counterfactual, not a reachability fact. It asks what would run if holes
    nobody can learn existed, and it can refuse a goal hole the goal cannot be reached
    without. So `ExactNoPlan` with `replaceable_holes` and `feasible_with_all_holes` True does
    not mean the problem cannot be solved. It means this rule, or Used, refused the plans.
  - The walk behind `upstream` reads preconditions and effects, and ignores what can be reached.
    Replacing a literal is tested one literal at a time.

Search limits. The returned plan is the cheapest plan passing every rule within the limits,
and the cheapest at any length if `limits_reached` does not name `max_length`. An
`ExactNoPlan` that names no limit means no plan passes these rules at any length; one that
names a limit proves nothing. All of this rests on exhaustive search of finite state spaces,
which can grow quickly on large domains.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

from inverse_skills.operators.hole_planner import Action, Domain, NoPlan, PlanResult


@dataclass(frozen=True)
class HoleContext:
    """One hole of a plan: what it must do, where it sits, and why it is there."""

    literal: str                      # the ground predicate it must establish
    index: int                        # its position in the plan
    state_before: frozenset[str]      # the symbolic state immediately before it
    used_by: str                      # the later step, or "goal", that uses what it establishes
    used_at: int | None               # that step's position in the plan; None for the goal
    waiting: tuple[str, ...]          # what was missing this literal in that state: relevant
                                      # library actions, and "goal" if the goal needs it
    prefix: tuple[str, ...]           # the plan up to the hole
    suffix: tuple[str, ...]           # the plan after the hole
    # Each library action that establishes the literal, and what happens if it takes the hole's
    # place in this plan. The test that it cannot do the job in any order is `Necessary`.
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

    `candidates` lists the literals holes were offered for during the search. The
    inherited gap fields stay empty: this mode computes no flat set of gaps. If
    `limits_reached` does not name `max_length`, the step limit changed nothing and this
    is the cheapest passing plan at any length.
    """

    hole_contexts: tuple[HoleContext, ...] = ()
    limits_reached: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExactNoPlan(NoPlan):
    """No plan passes the rules within these limits. `limits_reached` names any limit that
    cut the search short; with one, this proves nothing about the problem.

    `feasible_with_all_holes` is exact and ignores the limits: can the goal be reached at all
    with the library and holes for every offered literal? False is a proof that these holes
    cannot solve the problem. True means holes can, and the limits or the rules refused them,
    so this report is then about the rules, not about whether the problem can be solved.

    `unmet_needs` are literals the goal or a relevant action needs that the library alone
    never established; `unsynthesizable_needs` are those without a soft score, which no
    hole can be offered for. The literals of plans that reached the goal but were refused are
    `unnecessary_holes`, if the goal can be reached without them, or `replaceable_holes`, if
    they are needed but the causal preference refused them. The inherited gap fields stay
    empty, as in `ExactPlanResult`.
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


def relevant_actions(domain: Domain, goal_positive, goal_negative=()) -> list[Action]:
    """The library actions that can contribute to the goal, found by working backward.

    An action is relevant if it establishes a literal that is wanted, or removes one that
    is unwanted; its preconditions then become wanted, and what it needs absent becomes
    unwanted. This looks only at preconditions and effects, never at what can be reached.
    """
    wanted = set(goal_positive)
    unwanted = set(goal_negative)
    relevant: list[Action] = []
    found = True
    while found:
        found = False
        for action in domain.actions:
            if action in relevant:
                continue
            if domain.closure(action.add) & wanted or domain.removes(action) & unwanted:
                relevant.append(action)
                wanted |= action.pre
                unwanted |= action.pre_absent
                found = True
    return [action for action in domain.actions if action in relevant]


def reachable_states(domain: Domain, start: frozenset[str], holes=()) -> set[frozenset[str]]:
    """Every state reachable from `start` with the library and `holes`, by real transitions,
    in any order. No step limit applies, so the answer does not depend on the search's limits.
    A hole is usable only while its literal is missing, as in the search."""
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
    """The library actions that can establish `literal`. One that already needs it cannot."""
    return [action for action in domain.actions
            if literal in domain.closure(action.add) and literal not in action.pre]


def _ways_of_establishing(domain: Domain, literal: str) -> list[Action]:
    """Its establishers, and the actions that clear their way by removing what they need absent."""
    ways = _establishers(domain, literal)
    pending = list(ways)
    while pending:
        blocked = pending.pop()
        for other in domain.actions:
            if other not in ways and blocked.pre_absent & domain.removes(other):
                ways.append(other)
                pending.append(other)
    return ways


def _dependencies(domain: Domain, literal: str, used_up: bool) -> set[str]:
    """Every literal a way of establishing `literal` needs, directly or not. With `used_up`,
    also every literal such a way deletes. This walks over preconditions and effects only."""
    found: set[str] = set()
    pending = [literal]
    while pending:
        for action in _ways_of_establishing(domain, pending.pop()):
            for fact in (action.pre | action.delete if used_up else action.pre):
                if fact not in found:
                    found.add(fact)
                    pending.append(fact)
    return found


def _strict_dependencies(domain: Domain, literal: str, used_up: bool) -> set[str]:
    """The dependencies of `literal` that do not depend on it in turn. Members of a cycle with
    it are left out: a cycle has no unique root, so the cost chooses among them."""
    return {other for other in _dependencies(domain, literal, used_up)
            if other != literal and literal not in _dependencies(domain, other, used_up)}


def prerequisites(domain: Domain, literal: str) -> set[str]:
    """What establishing `literal` needs, strictly."""
    return _strict_dependencies(domain, literal, used_up=False)


def upstream(domain: Domain, literal: str) -> set[str]:
    """Where a repair may go instead of a hole for `literal`: what establishing it needs, and
    what it uses up. Actions that only destroy each other's effects are a cycle in the second
    sense, so neither is upstream of the other, although neither needs the other. Asked
    separately, a literal that is needed stays upstream when establishing it destroys its user.
    A chain of these always ends, because no literal can lead back to itself."""
    return prerequisites(domain, literal) | _strict_dependencies(domain, literal, used_up=True)


class _HoleSets:
    """What real transitions say about sets of holes. Each answer comes from an exhaustive
    search with no step limit, so none depends on a plan's ordering or on the search's limits."""

    def __init__(self, domain: Domain, start: frozenset[str], goal_positive: set[str],
                 goal_negative: set[str], learnable: set[str]):
        self.domain = domain
        self.start = start
        self.goal_positive = goal_positive
        self.goal_negative = goal_negative
        self.learnable = learnable
        self.last_resort = {literal for literal in goal_positive if _establishers(domain, literal)}
        self.library = library_states(domain, start)
        self._reached = {frozenset(): self.library}
        self._verdicts: dict[frozenset[str], tuple[set[str], set[str]]] = {}

    def states(self, literals) -> set[frozenset[str]]:
        literals = frozenset(literals)
        if literals not in self._reached:
            holes = [hole for literal in sorted(literals)
                     for hole in hole_starts(self.domain, literal, self.library)]
            self._reached[literals] = reachable_states(self.domain, self.start, holes)
        return self._reached[literals]

    def feasible(self, literals) -> bool:
        return any(self.goal_positive <= state and not (self.goal_negative & state)
                   for state in self.states(literals))

    def verdict(self, hole_set: frozenset[str]) -> tuple[set[str], set[str]]:
        """The literals of `hole_set` that fail exact necessity, and the rest that the causal
        preference refuses. This depends only on the set, so each set is judged once."""
        if hole_set not in self._verdicts:
            unnecessary = {literal for literal in hole_set if self.feasible(hole_set - {literal})}
            refused = {literal for literal in hole_set - unnecessary
                       if self.feasible((hole_set - {literal}) | self._repairs_instead_of(literal))
                       or (literal in self.goal_positive and self._waits(literal, hole_set))}
            self._verdicts[hole_set] = (unnecessary, refused)
        return self._verdicts[hole_set]

    def _repairs_instead_of(self, literal: str) -> set[str]:
        """The learnable literals that should be tried before a hole for `literal`."""
        above = upstream(self.domain, literal) & self.learnable
        if literal not in self.last_resort:
            return above - self.last_resort
        elsewhere = {other for other in self.learnable - self.last_resort
                     if literal not in _dependencies(self.domain, other, used_up=True)}
        return above | elsewhere

    def _waits(self, literal: str, hole_set: frozenset[str]) -> bool:
        """Would an action establishing this goal literal run if its prerequisites could be
        learned, although it cannot run as things are? This is the one counterfactual here: it
        uses holes nobody may be able to learn, so it can refuse a hole that is exactly necessary."""
        others = hole_set - {literal}
        needed = prerequisites(self.domain, literal)
        as_things_are = self.states(others | (needed & self.learnable))
        if_learned = self.states(others | needed)

        def runs(states, action):
            return any(self.domain.applicable(state, action) for state in states)

        return any(runs(if_learned, action) and not runs(as_things_are, action)
                   for action in _establishers(self.domain, literal))


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


def _hole_context(domain, start, steps, index, relevant, goal_positive) -> HoleContext:
    before = start
    for step in steps[:index]:
        before = domain.apply(before, step)
    hole = steps[index]
    key = hole.hole_for

    # Follow what the hole established to its first user, by the rule the search applied.
    state = domain.apply(before, hole)
    established = (domain.closure(hole.add) - before) & state
    used_by, used_at = "goal", None
    for later in range(index + 1, len(steps)):
        step = steps[later]
        if established & step.pre and (step in relevant or step.hole_for):
            used_by, used_at = step.name, later
            break
        state = domain.apply(state, step)
        established &= state

    waiting = [action.name for action in relevant if key in action.pre]
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


def plan_exact(domain: Domain, start, goal_positive, goal_negative=None, *,
               max_delegated: int = 2, max_length: int = 10) -> ExactPlanResult | ExactNoPlan:
    """The cheapest plan under the lexicographic cost whose holes pass the rules in this
    module's docstring, or an `ExactNoPlan` report. The same as
    `hole_planner.plan(..., mode="exact")`.
    """
    start = frozenset(start)
    goal_positive = set(goal_positive)
    goal_negative = set(goal_negative or ())

    def satisfied(state: frozenset[str]) -> bool:
        return goal_positive <= state and not (goal_negative & state)

    if satisfied(start):
        return ExactPlanResult((), (), (), 0, 0, (), 0)

    relevant = relevant_actions(domain, goal_positive, goal_negative)
    needs = goal_positive | {literal for action in relevant for literal in action.pre}
    learnable = needs & domain.scored
    hole_sets = _HoleSets(domain, start, goal_positive, goal_negative, learnable)
    holes = {key: hole_starts(domain, key, hole_sets.library) for key in sorted(learnable)}

    counter = 0
    shortest: dict[tuple, int] = {}   # node -> fewest steps it was explored with
    goals: list[tuple[tuple, tuple[Action, ...]]] = []
    best_key: tuple | None = None
    expanded = 0
    offered: set[str] = set()
    limits_reached: set[str] = set()
    unnecessary_holes: set[str] = set()
    replaceable_holes: set[str] = set()
    queue = [((0, 0, 0, 0, 0), counter, start, (), frozenset(), frozenset())]

    while queue:
        key, _, state, taken, delegated, unused = heapq.heappop(queue)
        if best_key is not None and key > best_key:
            break
        # As in `hole_planner._search`, but whether a plan is accepted also depends on what
        # holes established that nothing has used yet (`unused`), so that is part of the node.
        # Necessity and the causal preference depend only on `delegated`, which is too.
        # So two paths that share a node can finish in exactly the same ways.
        node = (state, delegated, unused)
        length = key[4]
        if node in shortest and shortest[node] <= length:
            continue
        shortest[node] = length
        expanded += 1
        if length >= max_length:
            limits_reached.add("max_length")
            continue

        missing = [literal for literal in holes if literal not in state]
        offered.update(missing)
        for action in domain.actions + [hole for literal in missing for hole in holes[literal]]:
            if not domain.applicable(state, action):
                continue
            nxt, collateral = domain.transition(state, action)
            if nxt == state:
                continue
            next_delegated = delegated
            changed = 0
            if action.hole_for is not None:
                next_delegated = delegated | {action.hole_for}
                if len(next_delegated) > max_delegated:
                    limits_reached.add("max_delegated")
                    continue
                changed = len((nxt ^ state) - collateral)

            # Used: this step uses what it needs. What holes established and nothing has used
            # yet must survive to the goal, and the goal must need it.
            is_user = action.hole_for is not None or action in relevant
            next_unused = {facts & nxt for facts in unused if not (is_user and facts & action.pre)}
            if action.hole_for is not None:
                next_unused.add(frozenset(domain.closure(action.add) - state) & nxt)

            next_key = (len(next_delegated), key[1] + len(collateral), key[2] + changed,
                        key[3] + (action.hole_for is not None), key[4] + 1)
            next_taken = taken + (action,)
            if satisfied(nxt):
                unnecessary, refused = hole_sets.verdict(next_delegated)
                unnecessary_holes |= unnecessary
                replaceable_holes |= refused
                goal_uses_the_rest = all(facts & goal_positive for facts in next_unused)
                if not unnecessary and not refused and goal_uses_the_rest:
                    if best_key is None or next_key < best_key:
                        best_key = next_key
                    goals.append((next_key, next_taken))
                continue
            counter += 1
            heapq.heappush(queue, (next_key, counter, nxt, next_taken, next_delegated,
                                   frozenset(next_unused)))

    if best_key is None:
        library_reached = set().union(*hole_sets.library)
        unmet = needs - library_reached
        return ExactNoPlan(candidates=tuple(sorted(offered)),
                           limits_reached=tuple(sorted(limits_reached)),
                           max_delegated=max_delegated,
                           max_length=max_length,
                           unmet_needs=tuple(sorted(unmet)),
                           unsynthesizable_needs=tuple(sorted(unmet - domain.scored)),
                           unnecessary_holes=tuple(sorted(unnecessary_holes)),
                           replaceable_holes=tuple(sorted(replaceable_holes)),
                           feasible_with_all_holes=hole_sets.feasible(learnable),
                           expanded=expanded)

    tied: list[tuple[Action, ...]] = []
    for goal_key, taken in goals:
        if goal_key == best_key and taken not in tied:
            tied.append(taken)
    chosen = tied[0]
    hole_indices = tuple(i for i, action in enumerate(chosen) if action.hole_for)
    return ExactPlanResult(
        actions=tuple(action.name for action in chosen),
        steps=chosen,
        delegated=tuple(sorted({action.hole_for for action in chosen if action.hole_for})),
        collateral=best_key[1],
        hole_literals=best_key[2],
        hole_indices=hole_indices,
        expanded=expanded,
        candidates=tuple(sorted(offered)),
        alternatives=tuple(tied[1:]),
        hole_contexts=tuple(_hole_context(domain, start, chosen, i, relevant, goal_positive)
                            for i in hole_indices),
        limits_reached=tuple(sorted(limits_reached)),
    )
