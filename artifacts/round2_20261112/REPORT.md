# Round 2: report

- **Date:** 2026-11-12
- **Branch:** `feature/signed-literals`, on top of `63c6e9b`:
  - `f4a0e8b`: the plan, frozen before any code changed (`PLAN.md`), with the seed audit;
  - `f4f753f`: the code and tests. R0–R3 ran at this commit, every step with a clean tree;
  - `2db49e1`: the figure, table, manifest and weight-check scripts. R4 ran at this commit;
  - the commit that adds this report and the evidence.
- **Nothing was retuned** after any result. Every number below comes from a file in this folder. §8 lists them.

## Summary

**SlidingDrawer, the final comparison on 50 fresh configurations (7000–7049):**
- **Final success is equal.** All eight policies restore the complete signed goal after settling in **50 of 50** configurations: targeted 4 × 50/50, direct 4 × 50/50. The 95% Wilson interval for each is [93, 100]%.
- **Learning efficiency differs.** On complete-goal validation, the targeted method first reaches 90% at **10k, 10k, 12.5k and 10k** interactions (SAC seeds 0–3). The direct baseline needs **25k, 22.5k, 22.5k and 25k**: medians 10k against 23.75k, about 2.4 times as many.
- **Stability differs.** Every targeted run stayed at 90% or above after first reaching it. Two of the four direct runs did not: seed 0 ends at 70%, seed 2 at 0%.

**StowCube:**
- **Selection:** the frozen rule selected the checkpoint at **116,000** steps.
- **Stage 3 passes:**
  - validation: the contract held where Stage 4 stops the skill in 20 of 20 episodes;
  - test scenes 5000–5049: 50 of 50, with every handoff valid.
- **Stage 4 passes** on 5000–5049:
  - full plan, ending with retreat_to_safe: **48/50**;
  - relearned, with the learned skill extracted as a library operator: **48/50**. The target is among that operator's effects;
  - library only: **0/50**;
  - without the interference model: **1/50**.
- **The only failures,** seeds 5006 and 5037, are a scripted library action, pick(cube_b), at the plan's first step. Those are the two scenes the handoff survey marks invalid. On the 48 valid scenes, full and relearned succeed every time.

**Reproducibility:**
- The retraining repeated E3 bit for bit:
  - the four targeted policies' weights are identical to E3's;
  - the Stow run's final weights are identical to E3's.
- So Round 2 changes no training outcome, only the checkpoint Stow's rule selects (116,000 here; 4,000 under the old rule) and how the policies are evaluated.

**Checks before training all passed (§1).** In particular:
- the Stow validation start is bitwise Stage 4's handoff on all 20 validation scenes;
- the changed final evaluation reproduces 400 of 400 earlier rows exactly.

**One incident, handled without loss of results (§2.3).** Nine training runs at once used up the laptop's memory. I stopped the four direct runs and reran them afterwards with the same settings; the stopped attempt is kept, labelled.

## 1. Checks before training (R0)

| Check | Result |
|---|---|
| Test suite at `f4f753f` | 333 passed (`R0_checks/tests_at_f4f753f.txt`) |
| Direct spec | no fences; the grasp hole's start and prefix. The fence the plan derives, ¬handle_grasped, is recorded as not used |
| Direct precheck | PASS. B′ recomputed from its definition at all 4,200 measured steps; the same starts as the grasp hole; the reference controller opens the drawer in 42 of 42 starts; nothing scripted runs inside an episode |
| Random actions, 32 training starts | 1 grasp, at most 17 mm of opening, the goal met in 0 |
| Stow handoff equivalence, scenes 4000–4019 | bitwise equal on all 20, against Stage 4's own functions. Compared: the whole saved simulator state, both fingers, the finger–cube contact forces (0–31 N), and every state up to Stage 4's stopping point (steps 6–8 under the scripted drag) |
| Final-evaluation regression, 3000–3049, E3 checkpoints | the targeted rows equal E4's and the direct rows equal the earlier full-horizon diagnostic: 400 of 400 rows, every shared field |

**The B′ goal term on a measured state** is checked in the tests: in the state the scripted inverse leaves, which meets the goal, the reward is 2V − 1 + 1. No precheck episode meets the goal.

## 2. StowCube

### 2.1 Selection and the skill test (R1, `R1_training/stow/1794476593/`)

**The rule (PLAN §1.4):**
- the validation episode starts exactly where Stage 4 starts the skill;
- **C_exec**, the contract (target and both fences robustly) where Stage 4 stops the policy, comes first;
- then **P10**, the 10th percentile of the minimum margin at step 40;
- then the latest checkpoint.

| | C_exec | C_40 | P10 at step 40 | Mean steps to the target |
|---|---|---|---|---|
| Selected, step 116,000, validation 4000–4019 | 20/20 | 20/20 | +0.1764 | 4.3 |
| The same checkpoint, test 5000–5049 | 50/50 | 50/50 | +0.1741 | 4.1 |
| Reference: the scripted drag, validation | 20/20 | 0/20 | −0.799 | 6.3 |

- **C_exec saturates.** It is 100% at 29 of the 30 checkpoints, all but the first, 4,000 (90%).
  - Even the scripted drag meets the contract where Stage 4 would stop it.
  - The drag breaks a fence only when it keeps dragging: C_40 is 0/20.
  - So P10 decided, as the rule allows.
- **The P10 tie, precisely:**
  - seven checkpoints share exactly the maximum P10, +0.17639033: steps 8,000, 56,000, 68,000, 72,000, 92,000, 104,000 and 116,000;
  - the rule took the latest, 116,000;
  - the final model, at 120,000, has +0.17638883, lower by 0.0000015.
- **What sets P10 at the top:** in every episode the binding literal is the fence in_region(cube_b, src_b), the score of cube B resting where place(B) left it, not the target. Differences of that size come from micrometre-level shifts of B.
  - So, in effect, the rule selected the latest checkpoint that never disturbs B. Treating near-equal values as ties would have been a change of method after seeing results, so I report the exact comparison as frozen.
- **The saved checkpoint,** run again on the validation scenes, reproduces its record episode for episode.
- **Handoff sweep** (a robustness probe, not a criterion): success 100% with the handoff 10 mm lower to 5 mm higher, 95% from 10 to 20 mm higher. E3's selected checkpoint gave 50% at +5 mm, but from the old validation start, with its second seating, so the two are not directly comparable.
- **The training itself repeated E3's:** the 120,000-step weights are identical (`R1_training/weights_vs_e3.json`). E3 kept only its 4,000- and 120,000-step checkpoints, which is why retraining was needed to apply the rule.
- **Training time:** 48 min, against 19 min in E3, because of the memory incident (§2.3).

### 2.2 Stage 4 on 5000–5049 (R2, `R2_stage4/stow_stage4.json`)

**The learned operator,** extracted from the policy's own executions on the demonstration scenes 2000–2009:
- **Demonstrations:** 10 runs; raw handoff validity 10 of 11. Scene 2001's forward push failed and scene 2004's handoff failed; both were replaced.
- **Preconditions:** ¬clear_of_walls(A), ¬gripper_open, in_region(A, pocket), ¬in_region(A, src_a), ¬in_region(B, mouth), in_region(B, src_b), tcp_near(A), ¬tcp_near(B).
- **Effects:** clear_of_walls(A) and ¬in_region(A, pocket). **The target is among them.**

| Condition | Complete signed goal | 95% interval | Boolean |
|---|---|---|---|
| Full: the planner, the learned drag, retreat_to_safe | **48/50** | [87, 99]% | 48/50 |
| Relearned: replanned with the extracted operator, no holes | **48/50** | [87, 99]% | 48/50 |
| Library only | 0/50 | [0, 7]% | 0/50 |
| Without the interference model | 1/50 | [0, 10]% | 1/50 |

- **Verdict: Stage 4 passes.** Full ≥ 80%, library only 0%, relearned ≥ 80%, and the target is among the effects.
- **The two failures,** seeds 5006 and 5037, in both full and relearned: pick(cube_b), the plan's first step, a scripted library action, left B on the table after 2 retries.
  - These are exactly the two scenes the separate handoff survey marks invalid (48 of 50 valid). The learned skill never ran on them.
- **Forward pushes:** they fell short and were re-rolled once on 5 seeds (5007, 5008, 5034, 5044 and 5048), as the protocol does.
- **Model against world:** 2 divergences, both at the failed pick(cube_b).
- **Without the interference model:** 49 seeds are refused at approach(cube_a), because ¬tcp_near(B) is not robustly measured (0.25–0.29), as in E4. Seed 5049 succeeded.

### 2.3 The memory incident

- **What happened:** R1 started all nine training processes at once. Each took about 1.2 GB, which filled the laptop's 14 GB and all 4 GB of swap and froze it.
- **What I did:** at about 15,000 of their 30,000 steps, I stopped the four direct runs (exit −15). The other five finished normally.
- **The rerun:** the direct runs were rerun from scratch with the same settings, four at once.
- **What is kept, labelled:**
  - the stopped attempt's folders: `R1_training/direct.attempt1_stopped/` (with `NOTE.md`);
  - its records: `records/direct_train_seed<k>.attempt1_stopped.json`;
  - its console output: `logs/run_R1.attempt1.out`. Logs stay local.
- **Effect on results:** none. Training does not depend on wall time, and the reruns started from scratch.
- **Recorded durations:** the runner collects jobs in order, so the first group's records all show 3,038 s, when Stow finished.

## 3. SlidingDrawer

### 3.1 Training (R1)

**Targeted grasp, SAC seeds 0–3, the established protocol** (`R1_training/drawer/`):
- **Selection** is on the skill-level grasp validation, unchanged: 10k, 10k, 12.5k and 12.5k, the same as E3.
- **Weights:** identical to E3's, for all four.
- **Stage 3 test** on 1000–1019: grasp 20/20 and pull_open 20/20, for every seed.
- **The new complete-goal validation** runs the final evaluation's own procedure: the grasp until robust, then pull_open, then settling.

| SAC seed | Grasp validation (selection), first ≥ 90% | Complete-goal validation, first ≥ 90% | Complete goal, whole curve |
|---|---|---|---|
| 0 | 10k | 10k | 0, 0, 0, then 100% from 10k |
| 1 | 10k | 10k | 0, 0, 80% at 7.5k, then 100% from 10k |
| 2 | 12.5k | 12.5k | 0 until 12.5k, then 100% |
| 3 | 12.5k (70% at 10k) | 10k | 0, 0, 0, then 100% from 10k |

The complete-goal measure can run ahead of the grasp measure. It stops at the first robust grasp, as the executive does, while the grasp measure needs the grasp still held at step 40.

**Direct baseline, B′, SAC seeds 0–3** (`R1_training/direct/`):

| SAC seed | Settled complete-goal validation, first ≥ 90% | Selected | Final | Stayed ≥ 90% | First drawer opening in training |
|---|---|---|---|---|---|
| 0 | 25k | 25k (100%) | 70% | no | 18.1k |
| 1 | 22.5k | 22.5k (100%) | 100% | yes | 14.5k |
| 2 | 22.5k | 22.5k (100%) | 0% | no | 16.6k |
| 3 | 25k | 25k (100%) | 100% | yes | 18.9k |

Before and after settling, validation agreed at every checkpoint except one: seed 0 at 30k, 90% before settling against 70% after.

### 3.2 Stage 4 on 2000–2019 (R2, `R2_stage4/drawer_stage4.json`)

- **Results:**
  - each learned grasp, then pull_open: 20/20 on the complete signed goal, 0 retries, 0 model-against-world differences;
  - library only: 0/20;
  - scripted grasp: 20/20.
- **Comparison with E4:** the summary equals E4's exactly, as the identical weights predict.

### 3.3 The final comparison on 7000–7049 (R3, `R3_final/final_targeted_vs_direct.json`)

**The protocol:**
- the same forward push and approach for all eight policies; their start states were checked identical;
- targeted: the grasp until robust (at most 40 steps), then pull_open;
- direct: all 100 steps, with no early stop;
- then the same 10 settling steps;
- success is the complete signed goal, robustly.

| Policy | Goal after settling | Goal at the policy's end | Lost in settling | First ≥ 90% complete-goal validation | Stayed ≥ 90% | How the drawer was moved (at 50 mm) | Final opening, median |
|---|---|---|---|---|---|---|---|
| targeted 0 | 50/50 | 50/50 | 0 | 10k | yes | pinch 50 | 147 mm |
| targeted 1 | 50/50 | 50/50 | 0 | 10k | yes | pinch 50 | 147 mm |
| targeted 2 | 50/50 | 50/50 | 0 | 12.5k | yes | pinch 50 | 148 mm |
| targeted 3 | 50/50 | 50/50 | 0 | 10k | yes | pinch 50 | 147 mm |
| direct 0 | 50/50 | 50/50 | 0 | 25k | no | press 48, pinch 2 | 176 mm |
| direct 1 | 50/50 | 50/50 | 0 | 22.5k | yes | press 23, pinch 27 | 200 mm |
| direct 2 | 50/50 | 50/50 | 0 | 22.5k | no | pinch 38, press 12 | 154 mm |
| direct 3 | 50/50 | 50/50 | 0 | 25k | yes | press 50 | 199 mm |

- **The protocol checks of PLAN §7 found nothing to classify:**
  - no failures;
  - no goal lost or gained while settling;
  - every forward push reached the handoff.
- **Unlike 3000–3049,** none of these configurations had a failed forward push.
- **Control steps:** a targeted inverse takes 31–35 steps (the grasp, then the scripted pull); a direct one, 100.

## 4. Against the previous results

These come from different seeds and earlier protocols, so they are not paired comparisons. `paper/previous_results.md` has them with their sources.

| | Previous (E4, 2026-11-11) | Round 2 |
|---|---|---|
| Direct reward | the target, plus the ¬handle_grasped fence | B′: the target, plus 1 where the complete goal holds |
| Direct evaluation | stop at the first goal step, then settle | all 100 steps, then settle |
| Direct, complete goal | 20, 0, 34, 39 of 50 (3000–3049); 42, 49, 49, 49 run for the full horizon (diagnostic) | 50, 50, 50, 50 of 50 (7000–7049) |
| Direct, first ≥ 90% validation | 30k, 17.5k, 22.5k, 30k (validation before settling) | 25k, 22.5k, 22.5k, 25k (after settling) |
| Targeted, complete goal | 4 × 49/50, the 50th a failed forward push | 4 × 50/50 |
| Stow checkpoint | step 4,000: the contract at step 40, then the fewest mean steps, on the Stage 4 scenes | step 116,000: C_exec, then P10, then the latest, on separate validation scenes |
| Stow Stage 4 | full 8/10, relearned 0/10, the target not learned; FAIL (1000–1009) | full 48/50, relearned 48/50, the target learned; PASS (5000–5049) |

## 5. For the manuscript

**What the evidence supports:**
- **Identification and composition:** the planner identifies the missing capability in both tasks. RL acquires it from the planner's spec, and the executive composes it with the library to restore the forward skill's initial conditions:
  - StowCube: 48/50 full plan, 48/50 after reintegration as an extracted operator;
  - SlidingDrawer: 4 × 50/50.
- **Targeted against direct:**
  - Both reach the same final success on fresh configurations: 4 × 50/50 each.
  - The targeted method needs about 2.4 times fewer interactions to reach 90% on the same complete-goal objective.
  - It stays there in 4 of 4 runs, against 2 of 4.
  - The claim is efficiency and stability, not final success.

**Limits to state:**
- **Seeds:** four SAC seeds per Drawer method, one Stow training run. Four runs per method do not support strong statistical claims, and the final-success intervals overlap completely.
- **Validation scenes:** the Drawer validation scenes (500–509) are shared by both methods and were used in earlier rounds, as established.
- **The Stow selection:** C_exec saturates (§2.1), so the selection was made by P10. In practice P10 was a tie between the robust checkpoints, settled by "the latest".
- **Stow's two failures** are a scripted library action (pick(cube_b)) failing on scenes whose handoff is invalid. They are not failures of the learned skill.
- **B′** only orders single steps. No optimality of the learned policy is claimed (PLAN §1.1).
- **The direct policies** often open the drawer by a one-finger press, and further than the targeted pull (up to its 200 mm travel). This is allowed by the goal, and recorded.
- **Decision 2's context limitation still holds:** a literal violated only after another learned skill gets no hole.

## 6. Figures and tables (`paper/`)

Built by `paper_results.py` from saved results only. Each comes with its data as CSV, and `sources.json` gives the SHA-256 of every input.
- **`drawer_learning_curves.pdf/.png`:**
  - (a) complete-goal validation for all eight runs;
  - (b) the targeted runs' grasp validation;
  - selected checkpoints ringed, with the 90% line.
- **`stow_selection.pdf/.png`:** C_exec and C_40 per checkpoint, with P10 below, and the selected checkpoint marked.
- **`drawer_table.tex/.md`:** the final comparison and learning efficiency, per policy, with Wilson intervals.
- **`stow_table.tex/.md`:** Stage 3 on validation and test, and Stage 4 with intervals.
- **`previous_results.md`:** the earlier protocol's results, labelled.

## 7. What changed in the code

The changes are listed in PLAN §3 and are in `f4f753f`:
- B′ and the no-fence check, in `drawer_direct_env.py`;
- the direct spec without fences;
- the precheck's B′;
- settled selection, in `drawer_direct_train.py`;
- in `drawer_final_evaluate.py`: the extracted `inverse`, the full direct horizon, and 7000–7049;
- the complete-goal validation measure, in `drawer_stage3_train.py`;
- `exact_handoff`, in `stow_hole_env.py`;
- the stop-point contract, P10 and the lexicographic rule, in `stow_stage3_train.py`;
- 12 new tests.

The planner, the extractor, the predicates, the executives and the targeted rewards are unchanged.

## 8. Reproducibility

- **Runner:** `run.py` R0–R4. It refuses a dirty tracked tree. `records/<step>.json` holds each step's command, commit, start time, duration and exit code; logs stay local.
- **Manifest:** `manifest.json` records every step's inputs and outputs with SHA-256, the environment, the seeds, and every file.
- **Environment:**
  - Python 3.10.21;
  - torch 2.7.1+cu128, stable-baselines3 2.8.0, gymnasium 1.2.3, ManiSkill 3.0.1, SAPIEN 3.0.3;
  - NVIDIA GeForce RTX 5070 Ti Laptop GPU;
  - PYTHONNOUSERSITE=1.
- **Checkpoints** stay local. Their SHA-256 are in `R1_training/*/checkpoints.json` and `manifest.json`. Every JSON output is under 1 MB, so all of them are committed.
- **Seeds:** the audit is `seed_audit.json` (PLAN §2).
- **Pushing:** from the host, by hash: `git push origin <sha>:feature/signed-literals`. Nothing is merged.
