"""Stage 1's checks of an extracted forward operator, in signed literals.

What a forward skill is designed to do is stated by its task as signed effects, and compared with
what the demonstrations gave; it is never an input to the extraction. The preconditions, and with
them the restoration goal G = Pre, are whatever the demonstrations reliably showed, of either
sign. They are reported in full and never compared with an expected set.

  effects_match_the_designed_skill     the extracted Eff is the skill's designed effects
  no_robot_relative_effects            no robot-relative predicate is an effect: the
                                       demonstrations span the whole skill, so the hand ends
                                       where it is not mid-contact (a cut inside the skill
                                       would turn tcp_near into a spurious effect)
  effects_complement_in_preconditions  every effect's complement held before every
                                       demonstration, as the strict rule guarantees
  goal_is_pre                          so the restoration goal is Pre itself
"""

from __future__ import annotations

from inverse_skills.operators.literals import Literal
from inverse_skills.operators.schema import LearnedOperator


def stage1_checks(operator: LearnedOperator, designed_effects: set[Literal], robot_relative) -> dict[str, bool]:
    effects = operator.effect_literals()
    return {
        "effects_match_the_designed_skill": effects == frozenset(designed_effects),
        "no_robot_relative_effects": not any(l.atom in robot_relative for l in effects),
        "effects_complement_in_preconditions": operator.effects_complement_in_preconditions(),
        "goal_is_pre": {t.literal for t in operator.restoration_goal()} == operator.precondition_literals(),
    }


def uncertain_at_start(scores: dict[str, dict]) -> list[str]:
    """The predicates no precondition was extracted for: neither true nor false before every
    demonstration. They are not part of the restoration goal."""
    return sorted(key for key, row in scores.items() if row["precondition"] is None)


def stage1_table(scores: dict[str, dict]) -> list[str]:
    """One line per predicate: its score ranges, and the signed precondition and effect extracted."""
    def signed(entry):
        return "-" if entry is None else str(Literal.from_dict(entry))
    lines = [f"{'predicate':28s} {'start min':>9} {'start max':>9} {'end min':>9} {'end max':>9}   "
             f"{'precondition':30s} {'effect':30s} changed"]
    for key, row in sorted(scores.items()):
        lines.append(f"{key:28s} {row['start_min']:9.3f} {row['start_max']:9.3f} {row['end_min']:9.3f} "
                     f"{row['end_max']:9.3f}   {signed(row['precondition']):30s} {signed(row['effect']):30s} "
                     f"{'yes' if row['changed_by_the_skill'] else 'no'}")
    return lines


def signed_summary(operator: LearnedOperator) -> dict:
    """Pre, Eff and G as lists of signed literals, for the Stage 1 record."""
    return {"preconditions": [str(l) for l in sorted(operator.precondition_literals())],
            "effects": [str(l) for l in sorted(operator.effect_literals())],
            "restoration_goal": [str(t.literal) for t in operator.restoration_goal()]}
