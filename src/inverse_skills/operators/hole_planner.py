"""A STRIPS planner whose library may be incomplete, with holes for the rest.

A hole is an abstract action standing for a policy nobody has learned yet. It
establishes one predicate the library cannot, and the planner may place it
anywhere in the plan, so the learned operator is not condemned to run last.

Nothing in this module names a predicate of any particular domain. Everything
the planner needs to reason about holes is derived from facts the domain
declares about its own vocabulary and world:

  scored          predicates with a soft score. Only these can be delegated,
                  because a learned skill is trained against the score.
  robot_relative  predicates that relate the robot to the object in their first
                  argument. A hole starts from every such predicate about its
                  object that the library can already establish: the library
                  does all it can before the learned skill takes over.
  implications, mutex_groups
                  facts every action obeys: one literal entails another; at most
                  one literal of a group holds.
  disturbances    side effects of the world rather than of any action: changing
                  one literal sweeps through where others hold, displacing those
                  objects to an unknown place (leaving a pocket pushes whatever
                  sits in its mouth). They apply to library actions and holes alike.

Which predicates may become holes is derived too. A hole belongs at an
irreducible missing capability, never at a literal the library could produce
once its own missing prerequisites were supplied: a hole there would learn a
way around an action the library already has. `capability_gaps` finds the
irreducible ones — literals no action establishes, and cycles whose members can
each be established only through another member. A capability gap becomes a
hole only if it is synthesizable: it has the soft score a learned skill's
objective is built from. If the library and the synthesizable gaps admit no
plan, the planner returns `NoPlan`, which reports the capability gaps and which
of them lack a soft score; it never widens the search to other literals.
(`unrestricted_holes=True` restores the old, unrestricted candidates, for
ablations and debugging only.)

Plans are ordered lexicographically by

    (predicates delegated, collateral disturbances, literals holes change,
     holes, length)

in words: use the library as much as possible; do not displace objects the plan
is not acting on, because where they end up is unknown; delegate the smallest
change; then the fewest learned steps; then the shortest plan. Equally cheap
plans are returned beside the chosen one, so an ordering the model does not
determine is visible instead of being settled by an unstated rule.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field, replace


def predicate_args(key: str) -> tuple[str, ...]:
    """The arguments of a ground literal such as "in_region(cube_a,pocket)"."""
    if "(" not in key:
        return ()
    inner = key[key.index("(") + 1:key.rindex(")")]
    return tuple(arg for arg in inner.split(",") if arg)


def delegated_predicate(action_name: str) -> str | None:
    """The predicate a hole or a learned operator establishes: HOLE[p] / LEARNED[p] -> p."""
    for prefix in ("HOLE[", "LEARNED["):
        if action_name.startswith(prefix) and action_name.endswith("]"):
            return action_name[len(prefix):-1]
    return None


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

    def to_dict(self) -> dict:
        return {"name": self.name, "pre": sorted(self.pre),
                "pre_absent": sorted(self.pre_absent), "add": sorted(self.add),
                "delete": sorted(self.delete), "hole_for": self.hole_for}

    @classmethod
    def from_dict(cls, data: dict) -> "Action":
        return cls(name=data["name"], pre=frozenset(data.get("pre", ())),
                   pre_absent=frozenset(data.get("pre_absent", ())),
                   add=frozenset(data.get("add", ())),
                   delete=frozenset(data.get("delete", ())), hole_for=data.get("hole_for"))


@dataclass(frozen=True)
class Disturbance:
    """Changing `trigger` sweeps through where the `clobbers` literals hold.

    Any of them that hold when the trigger changes are deleted: the object is
    displaced, and where it ends up is not known.
    """

    trigger: str
    clobbers: frozenset[str]


@dataclass(frozen=True)
class PlanResult:
    actions: tuple[str, ...]
    steps: tuple[Action, ...]
    delegated: tuple[str, ...]
    collateral: int
    hole_literals: int
    hole_indices: tuple[int, ...]
    expanded: int
    candidates: tuple[str, ...] = ()
    alternatives: tuple[tuple[Action, ...], ...] = ()
    capability_gaps: tuple[str, ...] = ()
    synthesizable_gaps: tuple[str, ...] = ()

    @property
    def cost(self) -> tuple[int, int, int, int, int]:
        return (len(self.delegated), self.collateral, self.hole_literals,
                len(self.hole_indices), len(self.actions))

    @property
    def unsynthesizable_gaps(self) -> tuple[str, ...]:
        return tuple(g for g in self.capability_gaps if g not in self.synthesizable_gaps)

    @property
    def alternative_names(self) -> tuple[tuple[str, ...], ...]:
        return tuple(tuple(a.name for a in alt) for alt in self.alternatives)


@dataclass(frozen=True)
class NoPlan:
    """The planner's answer when no plan exists within its limits: the
    capability gaps that remain, and which of them could be offered as holes.
    It is falsy, so `if result:` tells a plan from this report."""

    capability_gaps: tuple[str, ...] = ()
    synthesizable_gaps: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return False

    @property
    def unsynthesizable_gaps(self) -> tuple[str, ...]:
        return tuple(g for g in self.capability_gaps if g not in self.synthesizable_gaps)


@dataclass
class Domain:
    actions: list[Action]
    scored: frozenset[str] = frozenset()
    robot_relative: frozenset[str] = frozenset()
    mutex_groups: list[frozenset[str]] = field(default_factory=list)
    implications: dict[str, frozenset[str]] = field(default_factory=dict)
    disturbances: list[Disturbance] = field(default_factory=list)

    def applicable(self, state: frozenset[str], action: Action) -> bool:
        return action.pre <= state and not (action.pre_absent & state)

    def closure(self, literals) -> set[str]:
        """The literals together with everything they imply."""
        closed: set[str] = set()
        pending = list(literals)
        while pending:
            literal = pending.pop()
            if literal in closed:
                continue
            closed.add(literal)
            pending.extend(self.implications.get(literal, ()))
        return closed

    def removes(self, action: Action) -> set[str]:
        """Literals the action falsifies: its deletes and the mutex partners of what it adds."""
        removed = set(action.delete)
        for literal in self.closure(action.add):
            for group in self.mutex_groups:
                if literal in group:
                    removed |= group - {literal}
        return removed

    def transition(self, state: frozenset[str],
                   action: Action) -> tuple[frozenset[str], frozenset[str]]:
        """The next state, and the literals displaced as collateral on the way."""
        added = self.closure(action.add)
        result = (set(state) - action.delete) | added
        for literal in added:
            for group in self.mutex_groups:
                if literal in group:
                    result -= group - {literal}
        changed = set(state) ^ result
        collateral: set[str] = set()
        for disturbance in self.disturbances:
            if disturbance.trigger in changed:
                collateral |= (disturbance.clobbers & result) - added
        return frozenset(result - collateral), frozenset(collateral)

    def apply(self, state: frozenset[str], action: Action) -> frozenset[str]:
        return self.transition(state, action)[0]

    def hole(self, key: str, reachable) -> Action:
        """A hole for `key`, starting from every robot-relative predicate about
        its object that the library can already establish."""
        args = predicate_args(key)
        start = frozenset(
            p for p in self.robot_relative
            if args and p != key and p in reachable and predicate_args(p)[:1] == args[:1])
        return Action(name=f"HOLE[{key}]", pre=start, add=frozenset({key}), hole_for=key)


def relaxed_reachable(domain: Domain, start) -> frozenset[str]:
    """Every literal the library can reach with delete effects ignored.

    Dropping deletes only makes more things possible, so a literal outside this
    set is unreachable by the library outright: a proof, not a failed search.
    """
    reached = set(start)
    changed = True
    while changed:
        changed = False
        for action in domain.actions:
            if action.pre <= reached:
                new = domain.closure(action.add) - reached
                if new:
                    reached |= new
                    changed = True
    return frozenset(reached)


def open_conditions(domain: Domain, goal_positive) -> frozenset[str]:
    """Literals something needs: the goal, and every library action's preconditions."""
    needed = set(goal_positive)
    for action in domain.actions:
        needed |= action.pre
    return frozenset(needed)


def capability_gaps(domain: Domain, start, goal_positive,
                    goal_negative=()) -> frozenset[str]:
    """The irreducible missing capabilities behind a goal: where holes belong.

    Begin with what the goal needs and the library cannot reach, and follow
    every action that could establish such a literal, or remove a negated goal
    literal, to its own unreachable prerequisites. A literal is reducible if
    some action establishes it from prerequisites that can be supplied without
    it: the gap lies in those prerequisites, not in the literal. It is
    irreducible if no action establishes it, or if it lies on a cycle where
    every action establishing a member needs another member and nothing
    outside the cycle supplies one. Any member of such a cycle would break it,
    so all of them are returned, and the plan's cost chooses.

    Computed in the delete relaxation: supply the literals nothing establishes,
    see what becomes reachable, and treat what is still stuck. Every way to
    establish a stuck literal needs another stuck literal, so the stuck literals
    form cycles, and the cycles nothing leads out of are irreducible. Supply
    those too, and repeat until nothing is stuck.

    Because delete effects are ignored, this is a filter on where holes may go,
    not a feasibility test: the state-space search decides whether a plan
    exists. A delete interaction can hide a gap from it (a literal that looks
    reachable but is not), and that limitation is deliberately left alone.
    """
    reachable = relaxed_reachable(domain, start)
    establishers: dict[str, list[frozenset[str]]] = {}
    for action in domain.actions:
        for literal in domain.closure(action.add):
            establishers.setdefault(literal, []).append(action.pre - reachable)

    pending = [p for p in goal_positive if p not in reachable]
    for action in domain.actions:
        if domain.removes(action) & set(goal_negative):
            pending.extend(action.pre - reachable)
    needed: set[str] = set()
    while pending:
        literal = pending.pop()
        if literal not in needed:
            needed.add(literal)
            for missing in establishers.get(literal, ()):
                pending.extend(missing)

    gaps = {p for p in needed if p not in establishers}
    while True:
        supplied = relaxed_reachable(domain, frozenset(start) | domain.closure(gaps))
        stuck = needed - supplied
        if not stuck:
            return frozenset(gaps)
        needs = {p: {q for missing in establishers[p] for q in missing if q in stuck}
                 for p in stuck}
        gaps |= _closed_cycles(needs)


def _closed_cycles(needs: dict[str, set[str]]) -> set[str]:
    """Members of the strongly connected components of `needs` that no edge leaves."""
    def onward(p: str) -> set[str]:
        seen: set[str] = set()
        stack = list(needs[p])
        while stack:
            q = stack.pop()
            if q not in seen:
                seen.add(q)
                stack.extend(needs[q])
        return seen

    reach = {p: onward(p) for p in needs}
    return {p for p in needs if all(p in reach[q] for q in reach[p])}


def plan(domain: Domain, start: frozenset[str], goal_positive,
         goal_negative=None, *, max_delegated: int = 2, max_length: int = 10,
         unrestricted_holes: bool = False) -> PlanResult | NoPlan:
    """Cheapest plan under the lexicographic cost, or a `NoPlan` report.

    Holes are offered only at the synthesizable capability gaps: the capability
    gaps with a soft score. Both sets are reported either way.

    `unrestricted_holes=True` is for ablations and debugging only: it offers a
    hole for every unreachable scored open condition and, failing that, for
    every scored open condition — the planner's behaviour before the gap rule.
    """
    goal_positive = set(goal_positive)
    goal_negative = set(goal_negative or ())
    reachable = relaxed_reachable(domain, start)
    gaps = capability_gaps(domain, start, goal_positive, goal_negative)
    report = {"capability_gaps": tuple(sorted(gaps)),
              "synthesizable_gaps": tuple(sorted(gaps & domain.scored))}
    needed = open_conditions(domain, goal_positive)
    if not max_delegated:
        candidates = []
    elif unrestricted_holes:
        candidates = sorted((needed - reachable) & domain.scored)
    else:
        candidates = list(report["synthesizable_gaps"])
    result = _search(domain, start, goal_positive, goal_negative, candidates, reachable,
                     max_delegated, max_length)
    if result is None and unrestricted_holes and max_delegated:
        wider = sorted(needed & domain.scored)
        if wider != candidates:
            candidates = wider
            result = _search(domain, start, goal_positive, goal_negative, candidates,
                             reachable, max_delegated, max_length)
    if result is None:
        return NoPlan(candidates=tuple(candidates), **report)
    return replace(result, **report)


def _search(domain, start, goal_positive, goal_negative, candidates, reachable,
            max_delegated, max_length) -> PlanResult | None:
    holes = [domain.hole(key, reachable) for key in candidates]
    actions = list(domain.actions) + holes

    def satisfied(state: frozenset[str]) -> bool:
        return goal_positive <= state and not (goal_negative & state)

    if satisfied(start):
        return PlanResult((), (), (), 0, 0, (), 0, tuple(candidates))

    counter = 0
    shortest: dict[tuple, int] = {}   # node -> fewest steps it was explored with
    goals: list[tuple[tuple, tuple[Action, ...]]] = []
    best_key: tuple | None = None
    expanded = 0
    queue = [((0, 0, 0, 0, 0), counter, start, (), frozenset())]

    while queue:
        key, _, state, taken, delegated = heapq.heappop(queue)
        if best_key is not None and key > best_key:
            break
        # A node is the state plus the facts already handed to holes: the only
        # history the cost of any continuation depends on. Entries leave the
        # queue in cost order, so an earlier exploration of this node cost no
        # more than this one. That makes this one redundant, unless it used
        # fewer steps and so has more of the step limit left.
        node = (state, delegated)
        length = key[4]
        if node in shortest and shortest[node] <= length:
            continue
        shortest[node] = length
        expanded += 1
        if length >= max_length:
            continue
        for action in actions:
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
                    continue
                changed = len((nxt ^ state) - collateral)
            next_key = (len(next_delegated), key[1] + len(collateral), key[2] + changed,
                        key[3] + (action.hole_for is not None), key[4] + 1)
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

    tied: list[tuple[Action, ...]] = []
    for key, taken in goals:
        if key == best_key and taken not in tied:
            tied.append(taken)
    chosen = tied[0]
    return PlanResult(
        actions=tuple(a.name for a in chosen),
        steps=chosen,
        delegated=tuple(sorted({a.hole_for for a in chosen if a.hole_for})),
        collateral=best_key[1],
        hole_literals=best_key[2],
        hole_indices=tuple(i for i, a in enumerate(chosen) if a.hole_for),
        expanded=expanded,
        candidates=tuple(candidates),
        alternatives=tuple(tied[1:]),
    )


def derive_fences(domain: Domain, start: frozenset[str], steps, hole_index: int,
                  goal_positive, goal_negative=()) -> tuple[list[str], list[str]]:
    """What the step at `hole_index` must preserve, read off the plan's causal structure.

    Fences are causal links spanning the step: literals true before it that a
    later step or the goal consumes, with no step in between re-establishing
    them. Negative fences are the mirror image: literals false before it that a
    later step or the goal needs absent, with nothing in between removing them.
    Only scored literals are returned, since only those can be measured while
    the skill is learned.
    """
    steps = list(steps)
    before = start
    for action in steps[:hole_index]:
        before = domain.apply(before, action)
    step = steps[hole_index]
    end = len(steps)
    later = range(hole_index + 1, end)

    consumers = [(j, p) for j in later for p in steps[j].pre]
    consumers += [(end, p) for p in goal_positive]
    fences = {p for j, p in consumers
              if p in before and p not in step.add
              and not any(p in domain.closure(steps[k].add) for k in range(hole_index + 1, j))}

    absences = [(j, q) for j in later for q in steps[j].pre_absent]
    absences += [(end, q) for q in goal_negative]
    negative = {q for j, q in absences
                if q not in before
                and not any(q in domain.removes(steps[k]) for k in range(hole_index + 1, j))}
    return sorted(fences & domain.scored), sorted(negative & domain.scored)


def hole_spec(domain: Domain, start: frozenset[str], result: PlanResult, goal_positive,
              goal_negative=(), index: int = 0) -> dict:
    """The learning problem the planner hands to RL for one hole of a plan."""
    i = result.hole_indices[index]
    hole = result.steps[i]
    preserve, preserve_absent = derive_fences(domain, start, result.steps, i,
                                              goal_positive, goal_negative)
    return {
        "hole": hole.name,
        "achieve": hole.hole_for,
        "object": (predicate_args(hole.hole_for) or (None,))[0],
        "start": sorted(hole.pre),
        "preserve": preserve,
        "preserve_absent": preserve_absent,
        "prefix": [a.to_dict() for a in result.steps[:i]],
        "suffix": [a.to_dict() for a in result.steps[i + 1:]],
        "plan": list(result.actions),
        "hole_index": i,
        "cost": list(result.cost),
    }


def learned_action(achieve: str, operator) -> Action:
    """A learned skill as a library action, modelled from its own executions.

    `operator` is what the operator extractor made of the skill's (start, end)
    scenes, exactly as it does for demonstrations: its preconditions are what
    held whenever the skill was run, so the planner uses it only where it was
    verified.
    """
    return Action(name=f"LEARNED[{achieve}]",
                  pre=frozenset(t.key for t in operator.preconditions),
                  add=frozenset(t.key for t in operator.add_effects),
                  delete=frozenset(t.key for t in operator.delete_effects))


def symbolic_state(scores: dict[str, float], threshold: float = 0.8) -> frozenset[str]:
    """The literals a measured scene supports: {p : V_p >= threshold}."""
    return frozenset(key for key, score in scores.items() if score >= threshold)
