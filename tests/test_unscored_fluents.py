"""Round 1 (2026-11-11): what the system knows about fluents no predicate measures.

StowCube's `holding(cube)` has no soft score. Three things are kept apart:

  closed-world start    the measured start state holds only scored atoms, so an unscored atom is
                        false at the start;
  symbolic bookkeeping  after that it changes only through modelled transitions: pick adds it,
                        place removes it, and requirements on it, of both signs, are checked by
                        the symbolic model;
  physical observation  the executive measures scored atoms only. It never claims to have
                        measured an unscored one, and never checks one during execution: what
                        holds of it is guaranteed by the plan's symbolic model alone.
"""

from __future__ import annotations

from inverse_skills.envs import stow_domain as sd
from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs.stow_executive import modelled_effects, scored
from inverse_skills.operators.hole_planner import symbolic_state

HOLD_A, HOLD_B = "holding(cube_a)", "holding(cube_b)"


def action(name: str):
    return next(a for a in sd.library() if a.name == name)


def test_holding_has_no_soft_score_and_is_false_in_the_measured_start():
    assert {HOLD_A, HOLD_B}.isdisjoint(sd.domain().scored)
    every_score_high = {key: 1.0 for key in sp.registry().keys()}
    assert {HOLD_A, HOLD_B}.isdisjoint(symbolic_state(every_score_high))


def test_pick_adds_holding_and_place_removes_it():
    domain = sd.domain()
    picked = domain.apply(sd.stowed_state(), action("pick(cube_b)"))
    assert HOLD_B in picked and "in_region(cube_b,mouth)" not in picked
    placed = domain.apply(picked, action("place(cube_b,src_b)"))
    assert HOLD_B not in placed and "in_region(cube_b,src_b)" in placed


def test_requirements_on_holding_of_both_signs_are_checked_by_the_symbolic_model():
    domain = sd.domain()
    start = sd.stowed_state()
    assert not domain.enabled(start, action("place(cube_b,src_b)"))     # needs holding(cube_b)
    picked = domain.apply(start, action("pick(cube_b)"))
    for name in ("approach(cube_a)", "push(cube_b,src_b->mouth)", "retreat_to_safe"):
        assert HOLD_B in action(name).pre_absent and not domain.enabled(picked, action(name))
    assert domain.enabled(picked, action("place(cube_b,src_b)"))


def test_the_executive_neither_measures_nor_claims_to_measure_holding():
    """`scored` is the executive's filter for every robust check: preconditions, modelled effects,
    fences and the goal. It keeps only literals with a measured score."""
    measured = {key: 0.5 for key in sp.registry().keys()}
    pick = action("pick(cube_b)")
    checked = scored(pick.precondition_literals, measured)
    assert {str(l) for l in pick.precondition_literals} - {str(l) for l in checked} == {f"not {HOLD_A}", f"not {HOLD_B}"}
    effects = modelled_effects(sd.domain(), pick)
    assert HOLD_B in {l.atom for l in effects}
    assert HOLD_B not in {l.atom for l in scored(effects, measured)}
