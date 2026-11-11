"""A STRIPS planner whose library may be incomplete, with holes for the rest.

A hole is an abstract action standing for a policy nobody has learned yet. It
establishes one signed literal the library cannot, and the planner may place it
anywhere in the plan, so the learned operator is not condemned to run last. A hole
for `p` makes the predicate p true; a hole for `not p` makes it false, by deleting
p (there is no artificial predicate `not_p`). Literals, their signs and their truth
in a measured scene are defined once, in `literals.py`.

Nothing in this module names a predicate of any particular domain. Everything
the planner needs to reason about holes is derived from facts the domain
declares about its own vocabulary and world:

  scored          predicates with a soft score. Only these can be delegated,
                  because a learned skill is trained against the score.
  robot_relative  predicates that relate the robot to the object in their first
                  argument. A hole starts from the largest set of such predicates
                  about its object that the library establishes in a state where
                  the hole's literal does not hold yet: the library does all it can
                  before the learned skill takes over.
  implications, mutex_groups
                  constraints every valid state satisfies: one atom entails another;
                  at most one atom of a group is true. What an action adds brings
                  what it implies and displaces its mutex partners. A transition
                  whose result still breaks a constraint, or undoes the action's own
                  effects, is invalid: the action, or hole, cannot run in that state.
                  Nothing is ever deleted to repair a broken constraint.
  disturbances    side effects of the world rather than of any action: changing
                  one literal sweeps through where others hold, displacing those
                  objects to an unknown place (leaving a pocket pushes whatever
                  sits in its mouth). They apply to library actions and holes alike.

`plan` has two modes. `mode="exact"`, the default, is in `exact_hole_planner`: it
chooses the holes by real transitions, as the "Minimal Hole Planner" formulation
defines, with signed candidates, and then plans. `mode="relaxed"` is this module's
own gap analysis, below, kept as a historical comparison mode and for the
`unrestricted_holes` ablation; it offers holes for positive literals only, as it
always did. Both use this module's `Domain` and report the same five cost numbers.

In the relaxed mode, which predicates may become holes is derived in advance. A
hole belongs at an irreducible missing capability, never at a literal the library
could produce once its own missing prerequisites were supplied: a hole there would
learn a way around an action the library already has. `capability_gaps` finds the
irreducible ones — literals no action establishes, and cycles whose members can
each be established only through another member. A capability gap becomes a
hole only if it is synthesizable: it has the soft score a learned skill's
objective is built from. If the library and the synthesizable gaps admit no
plan, the planner returns `NoPlan`, which reports the capability gaps and which
of them lack a soft score; it never widens the search to other literals.
(`unrestricted_holes=True` restores the old, unrestricted candidates, for
ablations and debugging only.)

The relaxed mode orders plans lexicographically by

    (predicates delegated, collateral disturbances, literals holes change,
     holes, length)

and every result reports these five numbers as `cost`. The exact mode minimizes
the first, second and fifth only. In words, the relaxed order is: use the library as much as possible; do not displace objects the plan
is not acting on, because where they end up is unknown; delegate the smallest
change; then the fewest learned steps; then the shortest plan. Equally cheap
plans are returned beside the chosen one, so an ordering the model does not
determine is visible instead of being settled by an unstated rule.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field, replace

from inverse_skills.operators import literals as lit
from inverse_skills.operators.literals import Literal, join, pos


def predicate_args(key: str) -> tuple[str, ...]:
    """The arguments of a ground literal such as "in_region(cube_a,pocket)"."""
    if "(" not in key:
        return ()
    inner = key[key.index("(") + 1:key.rindex(")")]
    return tuple(arg for arg in inner.split(",") if arg)


@dataclass(frozen=True)
class Action:
    """A ground action, its signed preconditions and effects stored by sign.

    `pre` and `pre_absent` are the atoms Pre needs true and false; `add` and `delete` the atoms
    Eff makes true and false. `hole_for` marks a hole and the signed literal it is asked to
    establish; `learned_for` marks a library action run by a learned policy, and the literal that
    policy was trained to establish. The name is for people: nothing reads a sign from it.
    """

    name: str
    pre: frozenset[str] = frozenset()
    pre_absent: frozenset[str] = frozenset()
    add: frozenset[str] = frozenset()
    delete: frozenset[str] = frozenset()
    hole_for: Literal | None = None
    learned_for: Literal | None = None

    def __post_init__(self) -> None:
        for name in ("hole_for", "learned_for"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Literal):
                raise TypeError(f"{self.name}: {name} must be a Literal, got {value!r}")
        if self.pre & self.pre_absent:
            raise ValueError(f"{self.name}: needs both p and not p for " + ", ".join(sorted(self.pre & self.pre_absent)))
        if self.add & self.delete:
            raise ValueError(f"{self.name}: makes both p and not p for " + ", ".join(sorted(self.add & self.delete)))

    @property
    def precondition_literals(self) -> frozenset[Literal]:
        return join(self.pre, self.pre_absent)

    @property
    def effect_literals(self) -> frozenset[Literal]:
        return join(self.add, self.delete)

    def __str__(self) -> str:
        return self.name

    def to_dict(self) -> dict:
        data = {"name": self.name, "pre": sorted(self.pre),
                "pre_absent": sorted(self.pre_absent), "add": sorted(self.add),
                "delete": sorted(self.delete),
                "hole_for": None if self.hole_for is None else self.hole_for.to_dict()}
        if self.learned_for is not None:          # library actions keep the fields they always had
            data["learned_for"] = self.learned_for.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Action":
        """Read an action written by `to_dict`. In artifacts written before literals had signs,
        `hole_for` is the bare name of a positive literal."""
        def literal(value):
            return None if value is None else Literal.from_dict(value)
        return cls(name=data["name"], pre=frozenset(data.get("pre", ())),
                   pre_absent=frozenset(data.get("pre_absent", ())),
                   add=frozenset(data.get("add", ())),
                   delete=frozenset(data.get("delete", ())), hole_for=literal(data.get("hole_for")),
                   learned_for=literal(data.get("learned_for")))


class InconsistentState(ValueError):
    """A symbolic state that breaks one of the domain's declared implications or mutual
    exclusions. A planner refuses to start from one."""


class InvalidTransition(ValueError):
    """A transition whose result is not a valid state (`Domain.transition_problems`)."""


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
    """A plan. `delegated` and `candidates` are signed literals."""

    actions: tuple[str, ...]
    steps: tuple[Action, ...]
    delegated: tuple[Literal, ...]
    collateral: int
    hole_literals: int
    hole_indices: tuple[int, ...]
    expanded: int
    candidates: tuple[Literal, ...] = ()
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
    candidates: tuple[Literal, ...] = ()

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
        """Every signed precondition holds: the `pre` atoms are true, the `pre_absent` false."""
        return action.pre <= state and not (action.pre_absent & state)

    def can_run(self, state: frozenset[str], action: Action) -> bool:
        """Do the action's own conditions hold here? Its preconditions; and a hole runs only while
        its literal does not hold: a hole for `p` while p is false, for `not p` while p is true.
        Whether the result is a valid state is a separate question (`transition_problems`)."""
        if action.hole_for is not None and action.hole_for.holds(state):
            return False
        return self.applicable(state, action)

    def enabled(self, state: frozenset[str], action: Action) -> bool:
        """Can the action run here: its conditions hold and its result is a valid state."""
        return self.successor(state, action) is not None

    def violations(self, state) -> list[str]:
        """The declared constraints `state` breaks, in words; empty when it is consistent. An
        implication `q => p` is broken when q is true and p false, a mutex group when more than
        one of its atoms is true."""
        state = frozenset(state)
        broken = [f"{q} implies {p}, but {p} is false"
                  for q in sorted(state & self.implications.keys())
                  for p in sorted(self.implications[q]) if p not in state]
        broken += [f"{' and '.join(sorted(group & state))} exclude each other"
                   for group in self.mutex_groups if len(group & state) > 1]
        return broken

    def check_state(self, state, what: str = "the state") -> None:
        """Raise `InconsistentState` if `state` breaks a declared constraint."""
        broken = self.violations(state)
        if broken:
            raise InconsistentState(f"{what} breaks the domain's constraints: {'; '.join(broken)}")

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
        """The next state, and the literals displaced as collateral on the way. Raises
        `InvalidTransition` when the result is not a valid state. Preconditions are not
        checked here: `successor` checks both."""
        after, collateral = self._modelled(state, action)
        problems = self._problems(action, after)
        if problems:
            raise InvalidTransition(f"{action.name} cannot lead to a valid state: {'; '.join(problems)}")
        return after, collateral

    def transition_problems(self, state: frozenset[str], action: Action) -> list[str]:
        """Why the action's modelled result is not a valid state; empty when it is."""
        return self._problems(action, self._modelled(state, action)[0])

    def successor(self, state: frozenset[str],
                  action: Action) -> tuple[frozenset[str], frozenset[str]] | None:
        """The transition, if the action can run here and leads to a valid state; else None."""
        if not self.can_run(state, action):
            return None
        after, collateral = self._modelled(state, action)
        return None if self._problems(action, after) else (after, collateral)

    def _problems(self, action: Action, after: frozenset[str]) -> list[str]:
        """A valid result keeps every declared implication and mutual exclusion, and holds every
        atom the action adds and none it deletes. It is judged after the action's effects, what
        its additions imply and displace, and the disturbances have all been applied."""
        problems = self.violations(after)
        problems += [f"{atom} is added but false after" for atom in sorted(action.add - after)]
        problems += [f"{atom} is deleted but true after" for atom in sorted(action.delete & after)]
        return problems

    def _modelled(self, state: frozenset[str],
                  action: Action) -> tuple[frozenset[str], frozenset[str]]:
        """The modelled result before it is checked: the action's deletes and adds, what the adds
        imply, their mutex partners displaced, then the disturbances its changes trigger."""
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

    def about_object(self, literal: Literal) -> frozenset[str]:
        """The robot-relative atoms about the object of `literal` (its first argument), other
        than the literal's own atom: what a hole for it may start from."""
        args = predicate_args(literal.atom)
        return frozenset(p for p in self.robot_relative
                         if args and p != literal.atom and predicate_args(p)[:1] == args[:1])

    def hole(self, literal: Literal, available) -> Action:
        """h(l, c): a hole for the signed literal `literal`, starting from every robot-relative
        atom about its object that is in `available`. For `p` it adds p; for `not p` it deletes p.
        The domain's world constraints give it its other effects, as for any action."""
        start = self.about_object(literal) & frozenset(available)
        return Action(name=f"HOLE[{literal}]", pre=start,
                      add=frozenset({literal.atom}) if literal.positive else frozenset(),
                      delete=frozenset() if literal.positive else frozenset({literal.atom}),
                      hole_for=literal)


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
         unrestricted_holes: bool = False, mode: str = "exact") -> PlanResult | NoPlan:
    """Cheapest plan under the mode's lexicographic cost, or a `NoPlan` report.

    `mode="exact"`, the default, is in `exact_hole_planner`: it chooses the holes by
    real transitions first, then plans, and computes no gaps. It has no length limit,
    so `max_length` does not apply to it; `max_delegated` is its learning budget.

    `mode="relaxed"` offers holes only at the synthesizable capability gaps: the
    capability gaps with a soft score. Both sets are reported either way. Its search
    stops at `max_length` steps.

    `unrestricted_holes=True` needs `mode="relaxed"` named, and is for ablations and
    debugging only: it offers a hole for every unreachable scored open condition and,
    failing that, for every scored open condition — the behaviour before the gap rule.
    """
    if mode not in ("relaxed", "exact"):
        raise ValueError(f"Unknown planning mode '{mode}'")
    domain.check_state(start, "the start state")
    if mode == "exact":
        if unrestricted_holes:
            raise ValueError("unrestricted_holes is an option of the relaxed mode only")
        # Imported here because the exact planner builds on this module.
        from inverse_skills.operators.exact_hole_planner import plan_exact
        return plan_exact(domain, start, goal_positive, goal_negative, max_delegated=max_delegated)
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
        return NoPlan(candidates=tuple(pos(key) for key in candidates), **report)
    return replace(result, **report)


def _search(domain, start, goal_positive, goal_negative, candidates, reachable,
            max_delegated, max_length) -> PlanResult | None:
    holes = [domain.hole(pos(key), reachable) for key in candidates]   # positive holes only
    actions = list(domain.actions) + holes

    def satisfied(state: frozenset[str]) -> bool:
        return goal_positive <= state and not (goal_negative & state)

    if satisfied(start):
        return PlanResult((), (), (), 0, 0, (), 0, tuple(pos(key) for key in candidates))

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
            try:
                nxt, collateral = domain.transition(state, action)
            except InvalidTransition:
                continue
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
        candidates=tuple(pos(key) for key in candidates),
        alternatives=tuple(tied[1:]),
    )


def domain_atoms(domain: Domain) -> frozenset[str]:
    """Every atom the domain's actions and world facts mention."""
    atoms: set[str] = set()
    for action in domain.actions:
        atoms |= action.pre | action.pre_absent | action.add | action.delete
    for group in domain.mutex_groups:
        atoms |= group
    for literal, implied in domain.implications.items():
        atoms |= {literal} | set(implied)
    for disturbance in domain.disturbances:
        atoms |= {disturbance.trigger} | disturbance.clobbers
    return frozenset(atoms)


def replay_problems(domain: Domain, start: frozenset[str], steps, goal_positive, goal_negative=()) -> list[str]:
    """Replay a plan with the transition model from `start`: every step that could not run there,
    and the goal literals the final state misses. An empty list means the plan is valid. A step
    whose result is not a valid state ends the replay, since no state follows it."""
    problems = [f"the start: {broken}" for broken in domain.violations(start)]
    state = frozenset(start)
    for index, step in enumerate(steps):
        if not domain.can_run(state, step):
            problems.append(f"step {index + 1}, {step.name}, cannot run")
        invalid = domain.transition_problems(state, step)
        if invalid:
            problems.append(f"step {index + 1}, {step.name}, leads to an invalid state: {'; '.join(invalid)}")
            return problems
        state = domain.apply(state, step)
    for literal in sorted(join(goal_positive, goal_negative or ())):
        if not literal.holds(state):
            problems.append(f"the goal literal {literal} does not hold at the end")
    return problems


def state_before(domain: Domain, start: frozenset[str], steps, index: int) -> frozenset[str]:
    """The predicted symbolic state just before step `index` of a plan."""
    state = frozenset(start)
    for action in list(steps)[:index]:
        state = domain.apply(state, action)
    return state


def derive_fences(domain: Domain, start: frozenset[str], steps, hole_index: int,
                  goal_positive, goal_negative=()) -> tuple[list[str], list[str]]:
    """What the step at `hole_index` must preserve, read off the plan's causal structure.

    Fences are causal links spanning the step, of either sign. A positive fence is an atom true
    before the step that a later step or the goal needs true, with no step in between making it
    true again. A negative fence is the mirror image: an atom false before the step that a later
    step or the goal needs false, with no step in between making it false again. The atom the
    step itself must change is never a fence. Only scored atoms are returned, since only those
    can be measured while the skill is learned. Returns (atoms kept true, atoms kept false).
    """
    steps = list(steps)
    before = state_before(domain, start, steps, hole_index)
    step = steps[hole_index]
    target = step.hole_for or step.learned_for
    own = {target.atom} if target is not None else set(step.add) | set(step.delete)
    end = len(steps)
    later = range(hole_index + 1, end)

    consumers = [(j, p) for j in later for p in steps[j].pre]
    consumers += [(end, p) for p in goal_positive]
    fences = {p for j, p in consumers
              if p in before and p not in own
              and not any(p in domain.closure(steps[k].add) for k in range(hole_index + 1, j))}

    absences = [(j, q) for j in later for q in steps[j].pre_absent]
    absences += [(end, q) for q in goal_negative]
    negative = {q for j, q in absences
                if q not in before and q not in own
                and not any(q in domain.removes(steps[k]) for k in range(hole_index + 1, j))}
    return sorted(fences & domain.scored), sorted(negative & domain.scored)


def handoff_conditions_of(domain: Domain, before: frozenset[str], hole: Action,
                          kept_true, kept_false) -> tuple[list[str], list[str]]:
    """What must be measured when a learned skill takes over, as (atoms true, atoms false).

    Its start, its fences, and its transition's pre-state: what the modelled transition removes
    must hold, and what it establishes must not, since a transition begins in its own pre-state.
    Only scored atoms can be measured.
    """
    after = domain.apply(before, hole)
    hold = set(hole.pre) | set(kept_true) | (before - after)
    absent = set(hole.pre_absent) | set(kept_false) | (after - before)
    return sorted(hold & domain.scored), sorted(absent & domain.scored)


def hole_spec(domain: Domain, start: frozenset[str], result: PlanResult, goal_positive,
              goal_negative=(), index: int = 0) -> dict:
    """The learning problem the planner hands to RL for one hole of a plan.

    `target` is the signed literal the skill must establish; `achieve` repeats its atom for
    readers written before literals had signs. `start` and `start_absent` are the signed start,
    `preserve` and `preserve_absent` the signed fences, `fences` the same fences as one signed
    set, `handoff_state` the predicted symbolic state when the skill takes over, and
    `handoff_conditions` what the robot must measure there.
    """
    i = result.hole_indices[index]
    hole = result.steps[i]
    preserve, preserve_absent = derive_fences(domain, start, result.steps, i,
                                              goal_positive, goal_negative)
    before = state_before(domain, start, result.steps, i)
    hold, absent = handoff_conditions_of(domain, before, hole, preserve, preserve_absent)
    return {
        "hole": hole.name,
        "target": hole.hole_for.to_dict(),
        "achieve": hole.hole_for.atom,
        "object": (predicate_args(hole.hole_for.atom) or (None,))[0],
        "start": sorted(hole.pre),
        "start_absent": sorted(hole.pre_absent),
        "preserve": preserve,
        "preserve_absent": preserve_absent,
        "fences": [l.to_dict() for l in sorted(join(preserve, preserve_absent))],
        "handoff_state": sorted(before),
        "handoff_conditions": {"hold": hold, "absent": absent},
        "prefix": [a.to_dict() for a in result.steps[:i]],
        "suffix": [a.to_dict() for a in result.steps[i + 1:]],
        "plan": list(result.actions),
        "hole_index": i,
        "cost": list(result.cost),
    }


def spec_target(spec: dict) -> Literal:
    """The signed literal a hole spec asks for. Specs written before literals had signs name a
    positive target by its atom only."""
    return Literal.from_dict(spec["target"]) if "target" in spec else pos(spec["achieve"])


def spec_fences(spec: dict) -> frozenset[Literal]:
    """The signed fences of a hole spec: kept true (`preserve`) and kept false (`preserve_absent`)."""
    return join(spec["preserve"], spec["preserve_absent"])


def spec_start(spec: dict) -> frozenset[Literal]:
    """The signed start of a hole spec. Specs written before literals had signs have no
    negative part."""
    return join(spec["start"], spec.get("start_absent", ()))


def learned_action(target: Literal, operator) -> Action:
    """A learned skill as a library action, modelled from its own executions.

    `operator` is what the operator extractor made of the skill's (start, end)
    scenes, exactly as it does for demonstrations: its preconditions are what
    held whenever the skill was run, so the planner uses it only where it was
    verified. A positive precondition becomes `pre`, a negative one `pre_absent`;
    an effect that makes an atom true becomes `add`, one that makes it false `delete`.
    `target` is the literal the policy was trained to establish; whether the extracted
    effects contain it is for the caller to check, never assumed.
    """
    positive_pre = frozenset(t.key for t in operator.preconditions if t.polarity == "positive")
    negative_pre = frozenset(t.key for t in operator.preconditions if t.polarity == "negative")
    return Action(name=f"LEARNED[{target}]",
                  pre=positive_pre,
                  pre_absent=negative_pre,
                  add=frozenset(t.key for t in operator.add_effects),
                  delete=frozenset(t.key for t in operator.delete_effects),
                  learned_for=target)


def symbolic_state(scores: dict[str, float]) -> frozenset[str]:
    """The Boolean symbolic state a measured scene supports: the atoms scoring at least 0.5
    (`literals.TRUE_AT`). A missing or non-finite score raises; it is never read as false."""
    return lit.symbolic_state(scores)
