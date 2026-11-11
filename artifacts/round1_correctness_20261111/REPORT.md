# Round 1: correctness audit and targeted repairs

- **Date:** 2026-11-11
- **Branch:** `feature/signed-literals`, on top of `ca2d029`:
  - `c6d3050`: the corrections, with their tests;
  - `05cdc33`: the documentation;
  - `82a49e0`: this report, sections 1–10;
  - the commit that adds section 11, your decisions, with the documentation decision 2 asks for.
- **Pushing:** on 2026-11-11 you approved pushing `c6d3050`, `05cdc33` and `82a49e0`, without merging. The commit that adds section 11 waits for your approval.
- **Not done in this round, as instructed:** no SAC training, no physics evaluation, no checkpoint replaced, no recorded result changed.
- **Files in this folder:**
  - `regression.py` and `regression.json`: the symbolic regression of section 7.
  - `tests_at_05cdc33.txt`: the test run.
  - `regression.attempt1_shadowed.json`: a first run with a bug in my script. It is kept locally, not committed, and explained in section 7.

**Your decisions on the questions of section 10 are in section 11.** Only the direct reward's equation and weights remain open.

## 1. Confirmed correctness defects

**D1. Transitions were never checked against the declared constraints.**
- A transition applied an action's effects, the implication closure of its additions, the mutex displacement of its additions, and the disturbances. It never checked the result.
- **Your example was accepted.** With q ⇒ p, deleting p from {q, p} gave {q}, and `test_deleting_an_implied_atom_leaves_what_implies_it` asserted it.
- **The same gap had four more forms:**
  - a mutex displacement or a disturbance removing an atom that a true atom implies;
  - an action adding two atoms that exclude each other, so each displaced the other, and the action's own effects silently disappeared;
  - an action whose additions imply an atom it deletes, so its delete was silently undone;
  - a start state that breaks a constraint, accepted without comment.

**D2. Initiation contexts could come from states where the target already holds.**
- `hole_starts` took the largest robot-relative facts held in any library-reachable state.
- A state where the target already holds cannot start the hole, yet its facts could become the context. In your example the context was {near(o)}, held only after ¬p(o) was already true and z lost. So the feasible plan HOLE[¬p(o)] was missed.

**D3. Documentation that stated something false.**
- **The Method section:**
  - it said the constraints "hold for every action";
  - it noted that implications are not applied in reverse to deletes, as if that were acceptable;
  - it described the preservation set, the handoff and the execution checks without their real restriction to scored atoms.
- **The direct environment's docstring** said its spec had "no fences", and gave its reward as 2V(drawer_open) − 1. The implemented reward includes the ¬handle_grasped fence term.
- **`stow_domain.py`** said "the executive knows what the gripper holds". Nothing measures `holding`.

**D4. Found while fixing D1.** Once a transition can be invalid, two consumers would have crashed on an invalid state instead of reporting it:
- the executives' model monitor;
- Stow Stage 4's per-episode replanning, and Drawer Stage 4's per-seed planning.

Neither case occurs in any recorded run (section 7).

## 2. Changes implemented, and why

**Transition validity** (`operators/hole_planner.py`, `Domain`). One centralized check, used everywhere:
- **`violations(state)`** lists the broken implications (q true, p false) and mutex groups (more than one atom true). **`check_state`** raises `InconsistentState`.
- **The modelled result keeps its existing order:** the action's deletes and adds; what the adds imply; the mutex partners of the adds displaced; the disturbances whose triggers changed.
- **A transition is valid when** its result keeps every implication and mutual exclusion, holds every atom the action adds, and holds none it deletes.
  - `transition` raises `InvalidTransition` otherwise.
  - `successor` returns None when the action cannot run or its result is invalid.
  - `enabled` means `successor` is not None. `can_run` keeps the old meaning: preconditions, plus a hole's literal not holding.
- **Nothing is deleted to repair a constraint.** Deleting p while q ⇒ p holds is invalid; q is not deleted.
- **Mutex displacement by an addition is kept, as a modelled consequence and not a repair.** It is the declared model: the Drawer domain says "the planner applies them instead of each schema repeating them", and `pull_open` relies on it to make drawer_closed false. So it is applied forwards to additions, like implication closure, and the result is then checked. Question 4 asks you to confirm this.
- **Refusing an inconsistent start:** `plan` (both modes), `plan_exact` and `reachable_states` refuse one with `InconsistentState`.
- **Holes of both signs** go through exactly the same check as library actions.

**Planners.**
- The exact planner's reachability, plan search and library-alternative analysis all use `successor`, so invalid transitions are unavailable, never admitted, and never a crash.
- The finite-state reachability, the cost (holes, displaced, length), minimality, admissibility and the upstream preference are unchanged.
- The relaxed (historical) search skips invalid transitions too.
- `replay_problems` reports an invalid step and stops there.

**Initiation contexts** (`exact_hole_planner.hole_starts`).
- Contexts are drawn only from R_l = {S ∈ Reach(S0, O, δ) : S ⊭ l}, by the same rule for both signs. They keep the positive-fact representation and the largest-context preference.
- If R_l is empty there is no context and no hole.
- A new field, `no_context`, on both `ExactPlanResult` and `ExactNoPlan`, lists such candidates explicitly.

**Consumers.**
- Both executives stop their model monitor when the measured start, or a predicted step, is invalid, and say why in `model_stopped`.
- Stow Stage 4 records a refused per-episode replanning as `plan_refused` on that episode; Drawer Stage 4 records it as `stopped`.
- Nothing about execution itself changed.

**The independent oracle** (`tests/test_exact_hole_planner.py::Oracle`) implements both rules again from the definitions, without calling the planner's code:
- a step whose result breaks a constraint, misses an added atom or keeps a deleted one, is no transition;
- a hole starts only from what a library state holds where its literal does not.

## 3. Limitations that are not correctness defects

1. **Unscored fluents are never measured.** `holding(·)` is false in the measured start (closed world). It then changes only through δ: pick adds it, place removes it, and preconditions of both signs on it are checked by the planner. The executive neither measures it nor re-checks it during execution, so what holds of it rests on the plan's symbolic model alone. It does not claim otherwise: `test_unscored_fluents.py` shows the executive's robust checks exclude it.
2. **Fences and handoff conditions are restricted to scored atoms.** A literal on an unscored atom that a skill should preserve is not enforced during learning.
3. **The producer graph behind the upstream preference is static.** It reads what an action can establish from its definition, not whether that transition is valid in some state. It is a stated design choice, and the feasibility, minimality and NoPlan guarantees do not depend on it.
4. **R_l is taken over library-reachable states, as your definition says.** So a literal true in every library state has no hole, even in states that other holes reach.
   - In the Drawer this applies to ¬handle_grasped: no hole for it exists even after HOLE[handle_grasped].
   - No current plan needs one, since pull_open releases the handle. Question 5 asks about it.
5. **The historical validation and comparison scripts** (`validate_*.py`, `compare_hole_planners*.py`) were not updated. They replay plans with `apply`, which now raises on an invalid transition, so they would stop with an explicit error rather than continue.

## 4. Tests added or modified

**New: 20 tests.**
- `tests/test_transition_validity.py` (11):
  - the state check;
  - a library delete of an implied atom;
  - two exclusive additions;
  - additions implying a delete;
  - forward implication and displacement kept and valid;
  - a disturbance breaking an implication;
  - a positive hole whose displacement breaks an implication;
  - a negative hole and a library delete judged alike;
  - inconsistent starts refused by every entry point;
  - every state the Stow and Drawer models reach is consistent;
  - oracle agreement on 1,000 constraint-heavy random problems.
- `tests/test_hole_contexts.py` (5):
  - your {p(o), z} problem: HOLE[¬p(o)] with an empty context, where the earlier rule gave {near(o)} and no plan;
  - both signs use the same rule;
  - a literal kept true everywhere has no context and is reported in `no_context`;
  - every context is supported by a library state where its literal does not hold, checked on 450 random problems (those with a valid start);
  - the contexts of the Stow and Drawer holes are unchanged.
- `tests/test_unscored_fluents.py` (4):
  - `holding` has no score and is false in the measured start;
  - pick adds it and place removes it;
  - requirements on it, of both signs, are checked symbolically;
  - the executive neither measures it nor claims to.

**Replaced.**
- `test_signed_planning.py::test_deleting_an_implied_atom_leaves_what_implies_it` became `test_a_hole_for_not_p_cannot_run_while_an_atom_implying_p_holds`. From {q, p} with q ⇒ p there is no plan. Once q can be dropped, the plan is drop_q → HOLE[¬p]. The oracle agrees with both.

**Superseded, each marked "Round 1" in the test with its reason.**
- `test_exact_hole_planner.py::test_hole_starts_are_the_relaxed_modes_on_stow_and_the_drawer`: two literals the library keeps true everywhere, in_region(A,pocket) and drawer_closed, now have no context. Every other scored atom still matches the relaxed mode.
- `test_exact_hole_planner.py::test_a_hole_is_usable_only_while_its_literal_is_missing`: its start, p without q while p ⇒ q, is now refused.
- `test_exact_hole_planner.py::test_a_the_exact_result_types_keep_their_fields`: the new `no_context` field.

**Adapted, with no expectation changed.**
- Three random-problem loops now first check that a refused start really is refused, then skip it, drawing the same random numbers as before:
  - `test_the_order_of_the_action_list_does_not_change_the_cost`;
  - two in `test_minimal_formulation.py`.
- The oracle comparison helper does the same.

## 5. Test results

**Full suite: 321 passed** at `05cdc33` (`tests_at_05cdc33.txt`), against 301 at `ca2d029`:
- 6 of them run in simulation;
- the 176 in the planning and model files all pass.

## 6. Independent oracle results

The oracle agrees with the planner on every problem of every suite. It checks all of these:
- NoPlan is exact;
- the holes are admissible, so subset-minimal, and of the smallest size;
- the plan is valid;
- the cost is the least;
- every best hole set is reported.

| Suite | Problems | Starts refused (now checked) | Plans with a hole | With a negative hole |
|---|---|---|---|---|
| few literals | 1,500 | 122 | 422 | 18 |
| chains | 700 | 12 | 337 | 8 |
| signed | 1,200 | 61 | 393 | 147 |
| constraint-heavy (new) | 1,000 | 364 | 153 | — |

- **The new suite exercises invalid transitions directly:** 227 of its consistent problems have a library action that can run in some library state but leads to an invalid one.
- **The named signed problems also agree,** together with the new named cases of section 4.

## 7. StowCube and SlidingDrawer symbolic regression

**How it was run.** `regression.py` replanned both tasks with the corrected planner (commit `c6d3050`, clean tree):
- from the recorded Stage 2 measured states of all five seeds;
- with the recorded signed goals;
- without any simulation.

It compared everything with the recorded `E2_stage2` values. It also ran the planner at `ca2d029`, in its own process from an extracted copy, to compare reachable states.

| | StowCube | SlidingDrawer |
|---|---|---|
| Library-reachable states, previous / corrected | 18 / 18, identical | 2 / 2, identical |
| States reachable with every candidate's holes | 320 / 320, identical | 9 / 9, identical |
| Transitions the previous model admitted that are now invalid | 0 | 0 |
| Measured start states that break a constraint | 0 of 5 | 0 of 5 |
| Plan, candidates, capability set, hole positions, cost, ties, hole contexts, replay (all 5 seeds) | identical | identical |
| Hole spec (target, start, fences, handoff, prefix, suffix, cost) | identical | identical |
| Ablations | library only and without interference identical | library only, frontier, unrestricted and with grasp identical |
| Candidates with no context | none | ¬handle_grasped |

- **The corrected rule changes only the contexts of literals the library keeps true in every reachable state.** Each now has none, where before it had one taken from target-satisfied states:
  - StowCube: ¬clear_of_walls(A), in_region(A,pocket), ¬in_region(A,src_a), none of them a candidate;
  - SlidingDrawer: drawer_closed, ¬drawer_open, ¬handle_grasped.
- **The only candidate among them, the Drawer's ¬handle_grasped, is used by no plan.**
- **The intended contexts are unchanged:** the drag from tcp_near(A), the grasp from the standoff, and the direct baseline's HOLE[drawer_open] from the standoff, with the same derived fence ¬handle_grasped.
- **Recorded Stow Stage 4 states:** none of the 20 states replanned from would be refused.
- **About the kept failed run.** The first run of the script reported an empty no-context list for the Drawer. In its Drawer branch, the ablation loop reused the variable holding the main result, so it reported the scripted-grasp ablation's list. The per-seed lists were right all along. That run is kept, unused, as `regression.attempt1_shadowed.json`.

## 8. The direct baseline's reward: audit, no change made

**The reward implemented.** At every step t = 1, …, 100 of an episode, with no early termination and SAC's default discount:

  R(s_t) = V̂_{drawer_open}(s_t) + min(0, V̂_{¬handle_grasped}(s_t)) = (2 V_open(s_t) − 1) + min(0, 1 − 2 V_grasped(s_t))

This is `restoration.hole_reward` applied to the direct spec: target drawer_open, and the single fence ¬handle_grasped the protocol plan derives.

**Success and selection** use the complete signed goal G, judged robustly at the end of the episode:
- drawer_open ≥ 0.8;
- drawer_closed ≤ 0.2;
- at_handle_standoff ≤ 0.2;
- handle_grasped ≤ 0.2.

The checkpoint kept is the first with a strictly higher validation goal rate.

| Goal literal | In the reward | Changes during a legitimate direct execution |
|---|---|---|
| drawer_open | rewarded, dense, the only active term | false to true |
| ¬drawer_closed | not in the reward; it follows the opening only indirectly | true to false: the drawer starts shut |
| ¬at_handle_standoff | not in the reward | true at the start, must become false: the hand must leave the standoff |
| ¬handle_grasped | a preservation fence: penalised at every step while handle_grasped > 0.5, never rewarded | a grasp-and-pull strategy makes it true while pulling, then releases |

**How this differs from optimizing the complete terminal goal.**
1. **One of four literals is rewarded, at every step.** The task is a conjunction that must hold at the end.
2. **¬at_handle_standoff is not optimized at all.** It counts only through the end-of-episode success used for selection.
3. **¬handle_grasped is enforced as an invariant, although the task needs it only at the end.** The penalty, up to −1 per step, makes the legitimate grasp, pull and release strategy costly while it pulls. The recorded policies mostly open the drawer with a one-finger press instead: in 20 of 20, 24 of 34 and 30 of 39 of their successes, for SAC seeds 0, 2 and 3. Seed 1 had no successes.
4. **The fence costs nothing once handle_grasped ≤ 0.5, but the goal needs ≤ 0.2.** No term pushes it across that band. ¬at_handle_standoff has no term at all, so nothing pushes it below 0.2 either.

These are your direction's points. For the targeted holes, preservation fences stay as they are.

**What a fair Round 2 direct baseline would need changed.** None of this was done.
- `operators/restoration.py`: a goal-level reward function beside `hole_reward`. `hole_reward` stays as it is, for targeted holes.
- `scripts/drawer_direct_spec.py`: the direct spec would state the whole signed goal as the objective and carry no preservation fences. It would still record what the protocol plan derives, for reference.
- `envs/drawer_direct_env.py`: `_reward` is currently inherited from `DrawerHoleEnv`; it would be overridden. Termination, if one is chosen, also goes here, and the docstring would change.
- `scripts/drawer_direct_precheck.py`: its expected-reward check would use the new formula.
- `scripts/drawer_direct_train.py`: the reward description. Selection on the end-of-episode goal rate needs no change.
- `scripts/drawer_final_evaluate.py`: the direct policy's stop rule. This is the earlier open question B, and it must agree with the reward and termination chosen.
- Tests for the new reward and the precheck.

**Options for the reward, for your choice.** G is the four goal literals, T the last step of the episode. Section 11 explains why, under the fixed 100-step horizon you chose, B, C and D must pay their last-step term at every step instead, and gives them in that form.

| Option | Reward | What it implies |
|---|---|---|
| A. Dense goal sum | R_t = Σ_{l∈G} V̂_l(s_t) | Every literal counts at every step, so leaving the standoff is rewarded. Being ungrasped is now rewarded rather than only not penalised, but a temporary grasp still costs reward while it lasts, so the bias against grasping is weaker but remains. |
| B. Target progress, goal at the end | R_t = V̂_open(s_t) for t < T; R_T = V̂_open(s_T) + Σ_{l∈G∖{open}} V̂_l(s_T) | No intermediate penalty for grasping. The release and leaving the standoff are paid once, at the end. That end signal is sparse, so learning them may be slow. |
| C. Sparse terminal success | R_T = 1[s_T ⊨_ε G], 0 otherwise | Exactly the task. Exploration is probably too hard at 30,000 steps: random actions met the goal in 0 of 32 precheck episodes. |
| D. Potential-based shaping of the terminal goal | R_t = γΦ(s_{t+1}) − Φ(s_t), with Φ = Σ_{l∈G} V̂_l, plus Σ_{l∈G} V̂_l(s_T) at T | The shaping terms cancel over an episode, so the optimal policies are those of the terminal goal (Ng et al., 1999), with a dense signal. A temporary grasp costs nothing in the optimum. |
| E. Worst literal | R_t = min_{l∈G} V̂_l(s_t) | Dense, and pressure always falls on the literal furthest from holding. It is non-smooth, and penalises a temporary grasp. |

**Two choices go with any option.**
- **Termination:** keep the fixed 100-step horizon, or end an episode once s ⊨_ε G. Under a dense positive reward, early termination rewards stalling just short of the goal, which is why the hole environments never terminate early. Under B, C or D it is coherent.
- **Evaluation:** the final evaluation's stop rule must match the choice. Either run the full horizon, as validation does, or stop once the goal holds.

**Reruns Round 2 would need:**
- E2 direct spec;
- the E3 direct precheck and direct training, seeds 0–3 at 30k;
- the E4 final evaluation, in which the targeted policies are unchanged.

## 9. Does any of this invalidate existing evidence?

**No recorded result changes.**
- Reachable states, plans, capability sets, costs, contexts, fences, specs and ablations of both tasks are identical (section 7).
- No recorded start or Stage 4 state is refused.
- The changes to the executives and Stage 4 only add reporting, for situations that did not occur in any recorded run.
- **No stage needs rerunning because of these corrections.**

**Some written statements in the earlier records are now superseded.** The records themselves are left unchanged:
- In `artifacts/signed_literals_20261111/PLAN.md`:
  - item 7 calls "Deleting p does not delete the literals that imply p" a limitation; such a transition is now invalid;
  - item 6 says unmeasured fluents "stay false". They are false at the start and then change through δ.
- In that folder's `REPORT.md`, section A's planner description does not mention validity or the context rule.

**What still awaits your decisions, unchanged by this round:**
- Round 2's direct baseline (section 8);
- the two open questions from the rerun report: the Stow checkpoint selection, and the direct stop rule.

## 10. Questions for you

Answered on 2026-11-11 in section 11, except the reward's equation and weights.

1. **The direct baseline's Round 2 reward:** one of options A–E in section 8, or another. Also its termination rule.
2. **The final evaluation's stop rule for the direct policy:** question B from the rerun report. It now depends on question 1.
3. **The Stow checkpoint selection:** question A from the rerun report, still open.
4. **Mutex displacement.** I kept the declared model: an addition displaces its mutex partners, and the result is then checked. Treating mutexes as pure constraints would make the Drawer's `pull_open` invalid unless its definition deleted drawer_closed itself. That would change an action definition, which you asked me not to do. Please confirm.
5. **Contexts over library-reachable states only.** With R_l over library states, the Drawer's ¬handle_grasped has no hole even after a grasp hole. Please confirm this is intended. The alternative would take R_l over states reachable with the other holes.
6. **The historical scripts** `validate_*.py` and `compare_hole_planners*.py`: update them to the corrected model, or leave them as records of the earlier one.
7. **Pushing** `c6d3050` and `05cdc33`, and the commit that adds this report. Nothing is pushed until you say so.

## 11. Your decisions, 2026-11-11

They answer the questions of section 10. Sections 1–10 are unchanged, apart from pointers to this section.

**Decision 1, mutex displacement (question 4): kept.**
- Adding an atom displaces its mutex partners. This is a defined effect of the transition model, not a repair of an inconsistent state.
- A transition that breaks any other declared constraint stays invalid.
- `c6d3050` already implements this, and the Method section already states it. Nothing changed.

**Decision 2, initiation contexts (question 5): kept, with the limitation now stated.**
- Contexts still come from R_l, the library-reachable states where the candidate literal does not hold.
- The limitation is now written in the Method section, the README and the planner's docstring:
  - a context is always a situation the library itself produces;
  - a literal that is unsatisfied only after another hole has run gets no hole, so a plan that needs a learned skill for it after another learned skill is not found;
  - this limits the generality of planning with several learned capabilities in sequence. Extending R_l to the states reached with other holes is later work.
- In the Drawer, ¬handle_grasped is such a literal. No plan needs a hole for it, since pull_open lets go of the handle.

**Decision 3, historical scripts (question 6): kept as records.**
- The six scripts `validate_*.py` and `compare_hole_planners*.py`, and their outputs, are unchanged. Each output describes the model at the commit that produced it.
- Run again at the current commit, they would differ in two ways, both visible:
  - `compare_hole_planners.py` compares every scored literal's contexts in the two planner modes, so it would list the six literals of section 7 as new differences;
  - a start that breaks a constraint, or an invalid step in a replay, now raises an explicit `InconsistentState` or `InvalidTransition`. Section 7 found neither in any recorded state.
- `validate_modes_planning.py` catches one expected `ValueError` and records its message. Its Drawer start is consistent, so the message stays the same.
- No version-specific script is needed now. The one comparison the corrections needed is `regression.py`, which runs `ca2d029` and `c6d3050` side by side. A later validation will get its own script, named for its version.
- Nothing is rerun because of Round 1, since the recorded plans and execution semantics are identical (section 7).

**Decision 4, the direct policy's termination (question 2; question B of the rerun report): the full horizon.**
- Training keeps the fixed 100-step horizon, with no early termination.
- The primary final evaluation runs each direct policy for all 100 steps. It then applies the established 10 settling steps, with the arm held still and the gripper as last commanded, and judges the complete signed goal robustly.
- The E4 results under the old rule, which stopped at the first goal step, stay as recorded, as a historical diagnostic: direct 20, 0, 34 and 39 of 50.
- The same full-horizon rule was already run as a diagnostic on the current direct policies: 42, 49, 49 and 49 of 50 (`signed_literals_20261111/D_diagnostics/direct_full_horizon.json`). Round 2 replaces those policies.
- Round 2 changes the direct stop rule in `scripts/drawer_final_evaluate.py`.

**Decision 5, StowCube checkpoint selection (question 3; question A of the rerun report): a new rule, then retraining.**
- The 120k checkpoint is not chosen in hindsight.
- Round 2's plan will propose the rule, to be frozen before any training:
  - separate validation and final-test seeds;
  - robust target and fence satisfaction, and robustness margins, ranked above execution speed.

**Decision 6, the direct reward (question 1): open.**
- Your direction: a complete-task objective, dense progress feedback, and an explicit reward for satisfying the complete signed goal.
- You choose the exact equation and weights after reviewing section 8. Until then, no reward code changes and nothing is retrained.
- Decision 4 has one consequence for those options that section 8 does not state. It follows.

### What the fixed horizon means for the reward options

**A reward paid only at the last step cannot be learned from here.**
- The direct policy observes no clock. Its 9 numbers are the grasp policy's 7, the target's margin and the opening (`drawer_direct_env.observe`).
- SAC treats the 100-step limit as a time-out, not an end: it keeps estimating the value beyond it. This is the default of SB3 2.8.0 (`handle_timeout_termination=True`).
- The recorded runs keep that default. Their SAC settings are only learning_starts, batch_size, train_freq and gradient_steps, so the discount is the default γ = 0.99.
- So a term paid only at T, as options B, C and D are written in section 8, depends on nothing the policy observes. The same observation can come at step 40 or at step 100, and learning sees the term as noise.
- Options A and E pay at every step, and are unaffected.

**Two consistent ways to keep an explicit goal reward.**
- **Pay it at every step where the complete goal holds:** 1_G(s_t) = 1 if s_t ⊨_ε G, else 0, with the same thresholds as success. The best policy then reaches the goal and keeps it until step 100, which is what the full-horizon evaluation checks. I recommend this way.
- **Give the policy the time remaining, and make step T a real end.** A reward at T is then well defined. But the direct policy would get an input the targeted grasp policy lacks, and SB3's time-out handling would have to be switched off for it.

**The options with the goal paid at every step.**
- s_t is the state after step t, as in section 8.
- Φ(s) = Σ_{l∈G} V̂_l(s), which lies between −4 and 4.
- Any option can add w_G·1_G(s_t).

| Option | Reward at step t | Weights | What it implies |
|---|---|---|---|
| A. Dense goal sum | Φ(s_t) | — | As in section 8, now exact. A grasp costs 2 per step on its literal (V̂ goes from +1 to −1), against at most 1 under the current fence. Relative to the reward's range, that is 2 of 8 against 1 of 3. |
| B′. Target progress, goal paid per step | V̂_open(s_t) + w_G·1_G(s_t) | w_G | Dense on the target only: the term the targeted policies are paid for their own target, without fences. A temporary grasp costs nothing. Releasing and leaving the standoff are learned from the goal term alone. For w_G > 0.4, every state in the goal pays more per step than any state outside it, since V̂_open ≥ 0.6 wherever the goal holds. |
| C′. Sparse success | 1_G(s_t) | — | Exactly the task, but sparse: random actions met the goal in 0 of 32 precheck episodes. |
| D′. Success with potential-based shaping | w_G·1_G(s_t) + w_Φ·(γΦ(s_t) − Φ(s_{t−1})) | w_G, w_Φ | The goal term alone decides which policies are best, for any w_Φ (Ng et al., 1999). That result applies because SAC bootstraps through the time-out, and so treats the task as continuing. Feedback is dense on every literal, but it pays changes in Φ, not its level. A temporary grasp costs nothing in the optimum. For tabular learners, shaping is equivalent to starting the value estimates at Φ (Wiewiora, 2003). So it guides early learning, and does not change what is learned in the end. |
| E. Worst literal | min_{l∈G} V̂_l(s_t) | — | As in section 8. While the grasp holds robustly, the minimum is at most −0.6, whatever the drawer does. So pulling further while grasping earns nothing until the release. |

**My recommendation: B′ with w_G = 1.**
- Its dense term is the kind the targeted policies are paid for their own target, so neither method gets a richer dense signal.
- A temporary grasp costs nothing, and the complete goal is optimal.
- Per step it pays between −1 and 2, close to the current direct reward's −2 to 1.
- The current direct policies reach the complete goal in 42–49 of 50 full-horizon runs, although nothing in their reward asks them to leave the standoff: leaving it evidently comes with opening the drawer. They mostly open by pressing, though, so they show little about learning to let go, which B′ leaves to the goal term.

**D′ is the choice if the dense feedback must provably leave the objective unchanged.** It would take w_G = 1 and w_Φ = 1. Its feedback is the small change in Φ from step to step, which could slow a baseline that already needs about twice the interactions.

**Round 2's plan will also have to settle these:**
- **The direct checkpoint selection.** It uses the goal rate at the end of a validation episode, before settling, while the primary evaluation judges after settling.
- **The Stow rule of decision 5,** and its seeds.
- **The reruns:**
  - the E2 direct spec;
  - in E3: the direct precheck, the direct training (seeds 0–3, 30k) and the Stow drag training;
  - in E4: the final Drawer evaluation, and the Stow Stage 4 run with the new checkpoint.

The plan will come to you before anything runs.
