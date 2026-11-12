# Round 2: execution plan

- **Date:** 2026-11-12
- **Branch:** `feature/signed-literals` at `63c6e9b`. The four Round 1 commits are on origin (checked with `git ls-remote`).
- **Decisions:** your two messages of 2026-11-12.
  - The first set the protocol.
  - The second approved this plan: the change in §1.4 (no second seating), B′, the seed sets, the direct protocol and the reuse of Stage 0–2. It added two adjustments, now built in:
    - the Stow contract is judged where Stage 4 stops the policy (§1.4);
    - the targeted Drawer policies are retrained, with a complete-goal validation measure for comparing learning efficiency (§1.6).
- **Status:** approved for execution. This plan is committed before any code changes.
  - I stop only for a correctness issue, or a failed gate that needs a new scientific decision.
  - Nothing is retuned after a result.

## 1. Frozen rules

### 1.1 The direct baseline's reward, B′

At every step t = 1, …, 100 of an episode, with s_t the state after step t:

> R_t = 2 V_open(s_t) − 1 + 𝟙[s_t ⊨_ε G]

- V_open is drawer_open's score, between 0 and 1.
- G is the complete signed restoration goal of the Stage 1 operator: drawer_open, ¬drawer_closed, ¬at_handle_standoff and ¬handle_grasped.
- s ⊨_ε G means every positive literal scores 0.8 or more, and every negative one 0.2 or less. These are the existing robust thresholds.
- **Exactly as approved:**
  - the goal weight is exactly 1;
  - there are no fences: ¬handle_grasped is dropped, so a temporary grasp costs nothing;
  - there are no other terms, and no weight search.
- **Not claimed:** that B′ makes a goal-reaching policy optimal.
  - B′ only orders single steps: a state in G earns at least 1.6, any other state at most 1.
  - Round 1's §11 sentence "the complete goal is optimal" gets a dated note withdrawing it.

### 1.2 The targeted rewards: unchanged

Both come from `restoration.hole_reward` and the E2 hole specs. V̂ = 2V − 1 for p, and 1 − 2V for ¬p.
- **Stow drag:** V̂_clear_of_walls(A) + min(0, V̂_in_region(B,src_b)) + min(0, V̂_¬in_region(B,mouth)).
- **Drawer grasp:** V̂_handle_grasped, with no fences.

### 1.3 The direct baseline: training, validation, selection and final evaluation

- **Training is unchanged:**
  - 100-step episodes, never ended early;
  - the Drawer runs' SAC settings;
  - 30,000 interactions;
  - SAC seeds 0–3.
- **Validation:** every 2,500 interactions, giving 12 checkpoints.
  - It uses scenes 500–509, the same validation set as the targeted runs.
  - Each episode runs all 100 steps with the policy acting deterministically.
  - Then come the established 10 settling steps: the arm held still, the last gripper command held.
  - Success is s ⊨_ε G after settling.
- **Selection:** the highest settled validation success. Among equal rates, the earliest checkpoint wins. The rate before settling is still recorded, but never used.
- **Final evaluation** on scenes 7000–7049: the same 100 steps with no stop at the first goal step, the same settling and the same success.
- **E4's early-stop results** stay as recorded, labelled as a historical diagnostic: direct 20, 0, 34 and 39 of 50 on 3000–3049.

### 1.4 The Stow drag skill: validation and selection

**Scenes:** validation uses 4000–4019 (§2).

**Handoff:** in every validation episode it is run fresh, exactly as Stage 4 runs it.
- The simulator is rebuilt for the scene (`prim.reset` reconfigures it).
- Then the forward push runs, and the executive runs pick(B), place(B) and approach(A). approach(A) ends with the hand seated on A.
- The policy starts right there. There is no second seating motion, and nothing is restored from a saved state.
- Why nothing is restored: a saved state holds poses, velocities and joint positions. It does not hold the contact solver's internal state or the controllers' targets, so a restored start need not play out as Stage 4's does.
- Because the rebuild makes every episode a pure function of its scene, every checkpoint sees the same 20 handoffs.
- If the prefix fails, or the hole's start literals are not robustly measured, the scene is re-rolled at seed + 10,000 k, at most 5 tries, as now.
- A scene with no valid handoff after 5 tries stays in, flagged, and is the same for every checkpoint.

**Handoff equivalence, verified.** In the tests on training scenes, and before training (R0) on all 20 validation scenes, the validation start must equal Stage 4's, which is produced by Stage 4's own functions. It is compared bitwise in:
- the whole saved simulator state: both cubes' poses and velocities, and the robot's joint positions and velocities, both gripper fingers included;
- the gripper opening;
- the contact forces between each finger and cube A;
- the trajectory under the same actions, up to the stopping point, with the same stopping step and the same scores there.

In R0 the actions are the scripted reference drag, so no learned policy touches the validation scenes before training.

**Episode:** one execution of the learned operator, as Stage 4 runs it, then continued to 40 steps for the auxiliary margin.
- **Stopping point:** the first state, from the handoff on, where the target robustly holds. That is where Stage 4's executive stops the policy (`run_learned`).
  - If the target never holds within the 40-step budget, the stopping point is after step 40, and the contract fails.
  - It is one attempt. Stage 4's executive may retry a failed step up to 3 times; validation measures a single execution, as each demonstration of the learned operator is.
- **Primary contract:** at the stopping point, the target and both fences robustly hold.
- **Auxiliary margin:** the rollout continues to step 40, the training horizon. Continuing changes nothing before the stop, because the trajectory up to there is the same.
  - The margin is m = the minimum of μ_l(s_40) over the target and both fences.
  - μ_l = V_l − 0.8 for a positive literal, and 0.2 − V_l for a negative one.

**Per checkpoint,** every 4,000 interactions, 30 checkpoints in all:
- **C_exec,** primary: the share of the 20 episodes with the contract at the stopping point;
- **P10,** auxiliary: `numpy.percentile(m, 10)` with linear interpolation. With 20 episodes it lies between the 2nd and 3rd lowest margins;
- **also recorded:**
  - the contract at step 40 (C_40);
  - each episode's stopping step;
  - the scores at the stop and at step 40;
  - mean steps.

**Selection is lexicographic:** the highest C_exec, then the highest P10, then the latest checkpoint.
- Mean steps are never used.
- Test scenes are never used.

**Stage 3 passes** if the selected checkpoint's C_exec is 0.9 or more.

**Unchanged:**
- 120,000 interactions;
- the SAC settings and SAC seed 0;
- the training curriculum, including its seating;
- the 2-D action and 4-D observation;
- the reward.

**Why retrain:** E3 saved only two checkpoints, at 4,000 and 120,000 steps. Retraining makes all 30 available to the frozen rule. The rule chooses on scenes no run has seen.

### 1.5 Stow: the final test

- **Skill test:** after training, once the selection is final, the selected checkpoint runs the validation episode on scenes 5000–5049. It records the same measures, and selects nothing.
- **Stage 4,** with unchanged code, on scenes 5000–5049. Four conditions:
  - the full plan, planned in each episode from its measured state, ending with retreat_to_safe;
  - library only;
  - without the interference model;
  - relearned.
- **Relearned:** the learned operator is extracted from the policy's own executions on the established demonstration scenes 2000–2009.
- **Pass criteria, unchanged:**
  - full ≥ 80%;
  - library only 0%;
  - relearned ≥ 80%;
  - the target among the learned operator's effects.

### 1.6 The targeted Drawer policies: retrained, with a complete-goal validation measure

**The established protocol, unchanged:**
- the E2 grasp spec and its reward;
- the SAC settings;
- 30,000 interactions;
- training scenes 0–31, with a random start offset;
- validation every 2,500 interactions on scenes 500–509;
- SAC seeds 0–3;
- the Stage 3 test on scenes 1000–1019, followed by pull_open.

**Selection, unchanged:** the highest skill-level validation success; among equal rates, the earliest. It is never switched to the complete-goal measure.

**At every validation checkpoint, two measures from the same starts:**
- **skill-level, unchanged:** handle_grasped robustly true at the end of a 40-step episode;
- **complete-goal, new:** the final evaluation's own targeted procedure:
  - the policy runs until handle_grasped robustly holds, at most 40 steps;
  - if the grasp holds, pull_open runs;
  - then the 10 settling steps;
  - success is s ⊨_ε G after settling.
  - The validation code calls the final evaluation's function, so the two procedures cannot differ.

**Learning efficiency:** compared on complete-goal validation: targeted against the direct baseline's settled validation. The skill-level curves stay as an additional analysis.

**Downstream reruns:** Drawer Stage 4 (scenes 2000–2019) depends on these policies, so it is rerun with them.

## 2. Seeds

| Task | Role | Scenes | Status |
|---|---|---|---|
| Stow | training | 0–31 | unchanged |
| Stow | validation (selection) | 4000–4019 | new, never used |
| Stow | final test: skill test and Stage 4 | 5000–5049 | new, never used |
| Stow | demonstrations for the learned operator | 2000–2009 | established, reused |
| Drawer | training (both methods) | 0–31 | unchanged |
| Drawer | validation (selection, both methods) | 500–509 | unchanged |
| Drawer | targeted Stage 3 test | 1000–1019 | established |
| Drawer | Stage 4 | 2000–2019 | established |
| Drawer | final comparison | 7000–7049 | new, never used |
| Stow | E3 selection and E4 Stage 4 | 1000–1009 | previous results only |
| Drawer | E4 final comparison | 3000–3049 | previous results, and the R0 regression |

**The audit** (`seed_audit.py`, output `seed_audit.json`):
- **What it read:** 506 files.
  - Every JSON record, log and report under `artifacts/`.
  - Every script's seed constants and defaults.
  - Separately, the project's shell-command history and scratchpad since 2026-09-17, read by hand, for runs whose outputs never reached `artifacts/`.
- **Re-rolls:** Stow re-rolls a scene at seed + 10,000 k, for k up to 4.
  - So two Stow seeds can reach the same scene only if they are equal modulo 10,000. That is the check.
  - It also clears the candidates' re-roll scenes, 14000–14019 up to 45000–45049.
  - The Drawer never re-rolls, so for it the check is stricter than needed.
- **Result: all three ranges are clean.**
  - The only near miss is Drawer seed 5000, in one smoke run on 2026-11-11. That is a Drawer scene, not Stow scene 5000.
  - Every other recorded number in these ranges is a timestep count, checked key by key.

## 3. Code changes

**Drawer:**
1. `src/inverse_skills/envs/drawer_direct_env.py`:
   - B′ as one small function, which the environment's reward calls at every step;
   - a spec with any fence is refused;
   - the docstring.
2. `scripts/drawer_direct_spec.py`: writes the direct spec without fences. The fence the plan derives, ¬handle_grasped, is recorded beside it as not used.
3. `scripts/drawer_direct_precheck.py`: recomputes B′ from its definition at every step.
4. `scripts/drawer_direct_train.py`:
   - selects on the settled validation rate, earliest among equals, in one small function;
   - its config records the reward and the rule.
5. `scripts/drawer_final_evaluate.py`:
   - the part of an episode after the handoff becomes one function, which the targeted validation also calls;
   - the direct policy runs all 100 steps;
   - the default scenes become 7000–7049;
   - the learning summary reports every validation curve a run recorded, each under its own name, so historical runs keep their meaning.
6. `scripts/drawer_stage3_train.py`: the complete-goal measure at every validation checkpoint. Selection, settings and seeds are unchanged.

**Stow:**

7. `src/inverse_skills/envs/stow_hole_env.py`: an `exact_handoff` option, off by default, so training is unchanged. When on, every reset runs the handoff fresh, as Stage 4 does, with no second seating.
8. `scripts/stow_stage3_train.py`:
   - validation scenes 4000–4019 and test scenes 5000–5049, with `exact_handoff`;
   - the stopping-point contract, the step-40 margins and the lexicographic rule, in small functions;
   - the skill test after training;
   - the handoff sweep on the validation scenes.

**Unchanged:**
- `stow_stage4_execute.py` and `drawer_stage4_execute.py`: the runner passes their inputs;
- the executives, the planner, the extractor and the predicates;
- the targeted rewards.

**New, in this folder:**
- `run.py`: the runner, built like the rerun's;
- `handoff_check.py`: the R0 equivalence check;
- `manifest.py`: hashes of the files kept local;
- `paper_results.py`: the figures and tables;
- `REPORT.md`: written after the runs.

**Documents:**
- Round 1 `REPORT.md` §11 gets one dated line withdrawing the optimality sentence.
- `.gitignore` gets an exception for this folder; checkpoints and console logs stay excluded. Both go in the plan commit.
- `README.md`'s status paragraph is updated in the evidence commit.

## 4. Reused, not rerun

| What | Where (under `artifacts/signed_literals_20261111/`) | Why it stays valid |
|---|---|---|
| Stage 0 gates and the retreat gate | `E0_stage0/` | The scenes and controllers are unchanged. |
| Stage 1 operators, both tasks | `E1_stage1/` | The demonstrations and the extractor are unchanged. |
| Stage 2 plans and hole specs, both tasks | `E2_stage2/` | Round 1 left every plan and spec identical (`round1_correctness_20261111/regression.json`). |

- E3's targeted Drawer policies and E4's Drawer Stage 4 become previous results.
- The E3 checkpoints are used once more, in the R0 regression below.

## 5. Runs

- Every step runs through `run.py`, at one commit with no tracked change.
- Each step writes a log and a record. No step overwrites anything.
- The times are scaled from E3 and E4.

| Step | What | Gate | Time |
|---|---|---|---|
| R0 | (a) the test suite at the code commit; (b) the direct spec; (c) the direct precheck; (d) the Stow handoff check on 4000–4019; (e) the final-evaluation regression: the new script, with E3's checkpoints on 3000–3049, must reproduce E4's targeted rows and the earlier full-horizon diagnostic's direct rows (`D_diagnostics/direct_full_horizon.json`), row for row | all must pass | 3 min |
| R1 | in parallel: Stow training, then its skill test; targeted SAC seeds 0–3; direct SAC seeds 0–3. Then the checkpoint manifests | Stow Stage 3 passes | 30 min |
| R2 | Stow Stage 4 on 5000–5049; Drawer Stage 4 on 2000–2019 with the retrained grasp policies | reported | 4 min |
| R3 | the Drawer final evaluation: 4 targeted and 4 direct policies on 7000–7049 | reported | 1 min |
| R4 | figures, tables, manifest and report | — | — |

**If a gate fails:**
- R0: I stop and report.
- Stow Stage 3: I stop before Stow Stage 4 and report. The Drawer steps are independent, so they continue.

## 6. Tests

All new tests go in `tests/test_round2_protocol.py`.

**Without simulation:**
- **B′:**
  - with drawer_open at 0.9 and G met, the reward is 1.8;
  - a score of exactly 0.8, or exactly 0.2 for a negative literal, meets G;
  - a grasp costs nothing unless it breaks G;
  - there is no bonus at 0.79, or with a negative literal at 0.21.
- **The direct spec:** one with a fence is refused.
- **Direct selection:** the settled rate decides, and the earliest wins among equals. The test history is one where the rate before settling would choose differently.
- **The Stow stopping point:**
  - it is the first robust target, step 0 included;
  - the contract there checks both fences;
  - a target never reached fails the contract.
- **Stow margins:**
  - μ is right for both signs;
  - P10 equals numpy's linear percentile.
- **Stow selection:**
  - C_exec decides first, then P10, then the latest checkpoint;
  - mean steps and C_40 never matter. The test history is one where the old rule picks a different checkpoint.
- **Seeds:** the sets are disjoint modulo 10,000, and the audit's verdict is clean.

**With simulation, on training scenes only:**
- **The direct environment:**
  - the reward equals B′ on each step's scores;
  - nothing ends an episode before step 100;
  - in the state the scripted inverse leaves, which meets G, the reward includes the bonus. No precheck episode meets G, so this is where the bonus meets a measured state.
- **The final evaluation:**
  - the direct episode runs exactly 100 steps, then 10 settling steps, even with the goal test forced true from the first step;
  - a targeted stand-in that never grasps gets no pull_open, then exactly 10 settling steps.
- **Stow handoff equivalence:** the full comparison of §1.4, against Stage 4's own functions (`forward_once`, `run_prefix`, `run_learned`), with the scripted drag as the actions.

**The full suite,** 321 tests today, must pass with these added.

## 7. Figures and tables for the manuscript

`paper_results.py` builds all of them from saved results only. Each comes with its data as CSV and the SHA-256 of its inputs.
1. **Drawer figure, two panels,** each with its own axis:
   - (a) complete-goal validation against training interactions for all 8 runs: targeted after pull_open and settling, direct after settling. This is the learning-efficiency comparison.
   - (b) the targeted runs' skill-level grasp validation.
   - Every evaluation and every selected checkpoint is marked, with the 90% line.
2. **Stow figure, two panels:**
   - the contract rates C_exec (primary) and C_40;
   - P10 below them.
   - The selected checkpoint is marked.
3. **Drawer table,** scenes 7000–7049, per SAC seed:
   - complete-goal success after settling, k/50 with a 95% Wilson interval;
   - the policy-level target;
   - the goal at the end of the policy's run and after settling, which shows stability;
   - interactions to first reach 90% on complete-goal validation, for both methods; for targeted runs, also on skill-level validation;
   - the best and final validation, and whether validation stayed at 90% or above;
   - then the median and range over the four seeds.
4. **Stow table:**
   - Stage 3: the selected step; C_exec, C_40 and P10 on validation and on the test scenes;
   - Stage 4 on 5000–5049: the four conditions, each k/50 with a Wilson interval and the Boolean result beside it;
   - whether the target is among the learned operator's effects;
   - handoff validity.
5. **Supplementary table:** the previous protocol's results, labelled, from the existing records:
   - Drawer 3000–3049, with the early stop and the fence;
   - Stow E4 on 1000–1009, with the old rule, and its diagnostics.

**Protocol checks in the report:**
- Every direct failure is classified as one of:
  - never opened;
  - opened, but a literal unmet at step 100 (which one);
  - goal held at step 100 and lost while settling, with the last gripper command.
- Validation and test results are compared per seed.
- A weak direct result is not read as an advantage of the targeted method where the classification points at the protocol.

## 8. Reproducibility and artifact management

- **Commits, in order:**
  1. this plan, the seed audit, the Round 1 note and the `.gitignore` exception;
  2. the code and the tests;
  3. the evidence: records, small outputs, figures, tables, the report, the manifest and the README status.
- **Runs** happen only at commit 2, through `run.py`, which:
  - refuses a dirty tracked tree;
  - records each step's command, commit, start time, duration and exit code;
  - skips steps already run.
- **New outputs** go only under `artifacts/round2_20261112/`. Every script refuses to overwrite, and nothing historical is touched.
- **Kept local, as in the rerun:** checkpoints, console logs, and JSON outputs over 1 MB, which `.gitignore` names one by one. `manifest.json` records their SHA-256 and summaries.
- **Recorded with the results:**
  - PYTHONNOUSERSITE=1;
  - the package versions;
  - the GPU;
  - the test output at commit 2.
- **Pushing:** you push from the host, by hash: `git push origin <sha>:feature/signed-literals`. Nothing is merged.
