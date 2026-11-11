# From the Method section to the code

This file pairs every definition in `method_signed_literals.tex` with the code on `feature/signed-literals` that implements it. All paths are under `src/inverse_skills/` unless they start with `scripts/` or `tests/`.

| Method section | Definition | Code | Tests |
|---|---|---|---|
| §Literals | signed literal `p` / `not p`, complement, contradiction | `operators/literals.py`: `Literal`, `complement`, `literal_set` | `tests/test_signed_literals.py` |
| | closed-world satisfaction S ⊨ l, S ⊨ L | `Literal.holds`, `satisfied` | same |
| | Boolean state σ(s) = {p : V_p ≥ 0.5} | `literals.symbolic_state` (`TRUE_AT`); `hole_planner.symbolic_state` | same; `tests/test_hole_planner.py` |
| | robust reading ⊨_ε (ε = 0.8) | `Literal.robustly_holds`, `robustly_unmet`, `ROBUST_TRUE`/`ROBUST_FALSE` | `tests/test_signed_literals.py` |
| | a missing or non-finite score is a measurement failure | `literals.validate_scores`, `MeasurementError` | same |
| §Operators | operator ⟨Pre, Eff⟩, its add/delete projection, contradictions rejected | `operators/schema.py`: `LearnedOperator`, `PredicateTerm`; `operators/hole_planner.py`: `Action` (`pre`, `pre_absent`, `add`, `delete`) | `tests/test_signed_literals.py`, `tests/test_hole_planner.py` |
| | applicability, the modelled result (implications, mutex displacement, disturbances), displaced κ | `hole_planner.Domain`: `applicable`, `_modelled`, `closure`, `removes` | `tests/test_exact_hole_planner.py` |
| | consistent states; valid transitions (constraints kept, own effects held), δ undefined otherwise; nothing deleted to repair | `Domain.violations`, `check_state`, `transition_problems`, `successor`, `transition`; `InconsistentState`, `InvalidTransition` | `tests/test_transition_validity.py` |
| §Extraction | Eqs. (pre), (eff), the robust thresholds, per-predicate evidence | `operators/extractor.py`: `OperatorExtractor._extract_min_max` | `tests/test_signed_literals.py`, `tests/test_operator_extraction.py` |
| | Stage 1 checks: designed effects, no robot-relative effects, complement ⊆ Pre, G = Pre | `operators/stage1_checks.py`; `scripts/stow_stage1_extract.py`, `scripts/drawer_stage1_extract.py` | |
| §Goal | G = Pre ∪ complement(Eff), and G = Pre for the strict rule | `LearnedOperator.restoration_goal`, `effects_complement_in_preconditions` | `tests/test_signed_literals.py` (200 random tables) |
| §Problem | P_inv = ⟨O, S, S0, G, δ⟩; S0 measured and consistent; unmeasured fluents false in S0, then changed only by δ | `scripts/stow_stage2_plan.py`, `scripts/drawer_stage2_plan.py` (`hole_planner.domain_atoms`); `plan_exact` refuses an inconsistent start | `tests/test_unscored_fluents.py`, `tests/test_transition_validity.py` |
| §Holes | h(l, c): adds p for `p`, deletes p for `not p`; enabled while S ⊨ c, S ⊭ l and δ(S, h) is valid | `Domain.hole`, `Domain.can_run`, `Domain.enabled` | `tests/test_signed_planning.py`, `tests/test_transition_validity.py` |
| | the context c: largest robot-relative facts held together in a state of R_l (library-reachable, l unsatisfied); none when R_l is empty | `Domain.about_object`; `operators/exact_hole_planner.py`: `hole_starts`; `no_context` on both results | `tests/test_hole_contexts.py`, `tests/test_exact_hole_planner.py` |
| | candidates C = L_scored ∩ (G ∪ ⋃ Pre(a)) | `exact_hole_planner._HoleSets.__init__` | oracle tests |
| §Choice | F(R), by exhaustive reachability, no length bound | `_HoleSets.feasible`, `reachable_states` | the oracle: `tests/test_exact_hole_planner.py::Oracle` |
| | Min(R) | `_HoleSets.minimal` | same |
| | producer graph; establishing `not p` by delete, mutex partner or disturbance; U(l) | `established_by`, `_producer_edges`, `_below`, `upstream` | `tests/test_signed_planning.py`, `tests/test_minimal_formulation.py` |
| | admissible sets, budget-aware replacement | `_HoleSets.replaceable`, `admissible`, `smallest_admissible_sets` | oracle tests (3,400 random problems) |
| | cost C(π) = (holes, displaced, length) | `_cheapest_plans`, `ExactPlanResult.planning_cost` | oracle `best_plans` |
| | NoPlan | `ExactNoPlan`, in `plan_exact` | oracle tests |
| §Learning | preservation set F_i, both signs, scored atoms only | `hole_planner.derive_fences` | `tests/test_signed_planning.py` |
| | handoff conditions, checked robustly | `hole_planner.handoff_conditions_of`; `operators/handoff.py`: `handoff_conditions`, `handoff_violations` | `tests/test_handoff_validation.py` |
| | the learning specification | `hole_planner.hole_spec`, `spec_target`, `spec_fences`, `spec_start` | `tests/test_minimal_formulation.py` |
| | signed score V̂ and reward R_i | `Literal.signed_score`; `operators/restoration.py`: `hole_reward`; `envs/stow_hole_env.py`, `envs/drawer_hole_env.py` (`_reward`) | `tests/test_signed_execution.py` |
| | episode success s ⊨_ε l_i; fixed horizon; checkpoint selection | the hole environments' `step`; `scripts/stow_stage3_train.py`, `scripts/drawer_stage3_train.py` | `tests/test_signed_execution.py` |
| | learned operator: valid handoffs, same extraction, signs kept, target checked | `scripts/stow_stage4_execute.py` (`learn_operator`); `handoff.collect_demonstrations`; `hole_planner.learned_action` | `tests/test_signed_planning.py`, `tests/test_hole_planner.py` |
| §Execution | robust preconditions and modelled effects on scored atoms, learned stop rule, fences, Boolean monitor (stops on an invalid state), robust goal; unscored literals only symbolic | `envs/stow_executive.py`, `envs/drawer_executive.py`: `execute_plan`, `run_learned`, `modelled_effects`, `scored`, `model_stopped` | `tests/test_signed_execution.py`, `tests/test_unscored_fluents.py` |
| | the predefined StowCube retreat | `envs/stow_domain.py`: `retreat_to_safe`; `envs/stow_executive.py`: `retreat_to_safe`; measured by `scripts/stow_retreat_gate.py` | `tests/test_signed_execution.py` |
