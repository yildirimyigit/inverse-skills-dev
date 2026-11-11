# Signed literals: final report

- **Date:** 2026-11-11
- **Branch:** `feature/signed-literals`, from `63dc282`. Local commits only; nothing pushed.
- **Instructions followed:** "Signed-Literal Operator Extraction, Polarity-Symmetric Planning and Full End-to-End Revalidation" and "Final Design Decisions and Instructions" (the decisions).
- **Plan and interpretations:** `PLAN.md` in this folder. Every interpretation is listed there with its status.
- **Every number below** comes from a file in this folder. Section H lists the files.

## Decisions needed from you

The experiments raised two protocol questions. Both are on the stop-and-ask list (§17), so I changed nothing and report the protocol's results as they are, with labelled diagnostics beside them.

**A. Which Stow checkpoint should the selection rule pick?** (section E; PLAN item 16)
- With the checkpoint the unchanged rule picked (step 4,000), Stow Stage 4 fails:
  - the full plan meets the signed goal on 8 of 10 seeds;
  - the relearned plan does not exist, because the learned operator lacks its target.
- With the same run's final model, or the historical checkpoint, the same code passes: full 10/10, relearned 9/10.
- The rule froze on step 4,000 because nothing later could beat its 3.7 mean steps.

**B. When should the final evaluation stop the direct baseline?** (section E; PLAN item 17)
- The historical rule stops a direct policy at the first step where the goal holds, then settles with the gripper command unchanged.
- Under the signed goal, that stop often falls mid-motion. The direct policies restore 20, 0, 34 and 39 of 50.
- Run for their full 100-step horizon, as in validation, they restore 42, 49, 49 and 49.
- The targeted method restores 49 of 50 for every seed either way.

## Items that need your attention

These are the places where the implementation or the evidence goes beyond the text of the decisions. Each one is explained in the section named.

1. **The retreat has a measured precondition, gripper_open()** (section C). The gate found that the retreat fails whenever the gripper starts closed and pressing on cube A: 0 of 40 runs. It passed every run with the gripper open. The plan places it where the gripper is open.
2. **The direct baseline has one more fence, ¬handle_grasped** (sections B and E). It comes from the protocol plan under the signed goal, by your §10 (fences derived from the plan) and §6 (retrain if the reward changes). It changes the direct reward.
3. **Hole start conditions are an open question** (PLAN item 1). I kept the established rule. The alternative restricts starts to states where the hole can run. It changes the definition in edge cases, though neither task.
4. **The Stow forward check is now robust** (PLAN item 13). Before, a half-finished forward push on seed 2001 passed as stowed.
5. **The training protocol for Stow is partly reconstructed** (PLAN item 14). The reference run did not record its curriculum length or SAC seed, so the script defaults are used.
6. **Parameterized templates now differ by distractor** (PLAN item 15). The published "distractor template invariance" holds only for the positive-only extraction.
7. **The Stow drag environment cannot train a negative skill** (decision 4). The framework can, and a test shows a negative target learned in the Drawer environment.
8. **Stage 3 attempt 1 collided** (section D). The four Drawer seeds started in the same second and wrote into one run folder. The outputs are kept, labelled; the script now names folders by seed and never reuses one; all four seeds were retrained.
9. **Stage 4 attempt 1 crashed while writing its results** (section E). Both scripts wrote into a folder that did not exist yet. Fixed, rerun, and the failed records kept.
10. **The interference ablation fails earlier than before** (section E). The robust check refuses approach(A) at once, where the Boolean check let it run on and fail later: 0/10 now, against 3/10 historically.

## A. Code changes

The implementation commits (7279aad to b621e00) changed 48 files: 3,039 lines added, 811 removed. Two fixes made during the experiments, adfa69d and 6acc02e, touch six scripts, and are described in sections D and E. By area:

**Literals and thresholds** (`operators/literals.py`, new)
- A literal is an atom with a sign. Its complement flips the sign. A set holding both p and ¬p is rejected.
- Boolean truth: V ≥ 0.5 (`TRUE_AT`). It is used only for the planner's symbolic state.
- Robust truth: p holds when V ≥ 0.8, and ¬p holds when V ≤ 0.2 (`ROBUST_TRUE`, `ROBUST_FALSE`). It is used for extraction, physical checks, RL success, fences, handoffs and the final goal.
- A missing or non-finite score is a measurement error. It is never read as false.

**Operators** (`operators/schema.py`, `operators/extractor.py`, `operators/stage1_checks.py`)
- The operator stores signed preconditions and signed effects. Files carry a schema version, `signed_literals_v1`. Positive-only files still load, with their meaning unchanged.
- The extraction rule is signed:
  - p becomes a precondition when every demonstration starts with V ≥ 0.8. ¬p becomes one when every demonstration starts with V ≤ 0.2.
  - p becomes an effect when every demonstration goes from ≤ 0.2 to ≥ 0.8. ¬p becomes an effect for the opposite transition.
  - Each predicate's evidence is recorded.
- The restoration goal is G = Pre ∪ complement(Eff). The extractor checks that every effect's complement is a precondition, so G = Pre.

**Planner** (`operators/hole_planner.py`, `operators/exact_hole_planner.py`)
- A hole for p adds p. A hole for ¬p deletes p. A hole can run only while its literal does not hold.
- The candidate holes are the scored signed literals that appear in the goal or in a library precondition, of either sign.
- An action establishes ¬p in one of three ways: it deletes p, it adds a literal that excludes p, or it triggers a disturbance that displaces p. The upstream preference reads preconditions of both signs.
- The exact planner keeps its objective (holes, displaced, length), its admissibility rule and its exact NoPlan.
- Fences are derived from the plan, in both signs, excluding the hole's own atom. The learning spec stores the target and the fences as signed literals.
- `learned_action` had a bug: it read every precondition as positive. It now keeps the signs.

**Learning** (`envs/stow_hole_env.py`, `envs/drawer_hole_env.py`, `envs/drawer_direct_env.py`, `operators/restoration.py`)
- The reward is R = V̂(target) + Σ min(0, V̂(fence)), with V̂(p) = 2V − 1 and V̂(¬p) = 1 − 2V.
- An episode succeeds when the target holds robustly. Fences and handoffs are judged robustly.
- The first observation entry is the target's signed margin. This is identical for positive targets.
- The Stow drag environment is otherwise unchanged: initialization, gripper, press start and curriculum (decision 4).

**Execution** (`envs/stow_executive.py`, `envs/drawer_executive.py`)
- Every precondition and every modelled effect is checked robustly. Scores between 0.2 and 0.8 are reported as uncertain.
- A learned step stops once its target holds robustly. Its policy is looked up by its target literal.
- The Boolean model-versus-world comparison is kept as a monitor.
- The goal is judged robustly, and the Boolean verdict is reported beside it.
- `retreat_to_safe` is bound to the existing `retreat()` motion, with the gripper kept as it is.

**StowCube library** (`envs/stow_domain.py`)
- `retreat_to_safe` is added:
  - pre: gripper_open(), ¬holding(cube_a), ¬holding(cube_b)
  - effects: ¬tcp_near(cube_a), ¬tcp_near(cube_b)
- The planner places it. No plan is hard-coded.

**Scripts**
- Stages 0–4 for both tasks, the final evaluation, and the direct spec, precheck and training all read and write signed literals.
- New scripts: `stow_retreat_gate.py` (the retreat's physical gate) and `stage3_checkpoints.py` (a checkpoint manifest with SHA-256 hashes).
- `validate_handoff.py`, `validate_modes_extraction.py`, `drawer_stage3_evaluate.py`, `drawer_stage4_video.py` and `stow_planner_demo.py` got only the minimal fixes needed to run.

## B. Symbolic results (E1, E2)

**Stage 1 operators**, from 5 fresh demonstrations per task (seeds 0–4). Every Stage 1 check passed for both tasks: effects match the designed skill, no robot-relative effects, every effect's complement is in Pre, and G = Pre. No predicate was uncertain at the start.

| | StowCube | SlidingDrawer |
|---|---|---|
| Positive preconditions | clear_of_walls(A), gripper_open(), in_region(A,src_a), in_region(B,src_b) | drawer_open |
| Negative preconditions | in_region(A,pocket), in_region(B,mouth), **tcp_near(A), tcp_near(B)** | drawer_closed, **at_handle_standoff, handle_grasped** |
| Effects | in_region(A,pocket), in_region(B,mouth), ¬clear_of_walls(A), ¬in_region(A,src_a), ¬in_region(B,src_b) | drawer_closed, ¬drawer_open |
| Restoration goal G = Pre | the 8 literals above | the 4 literals above |

The literals in bold are new in the goal. The historical goals had 6 literals for Stow and 2 for the Drawer.

**Stage 2 plans**, from fresh measured states (seeds 0–4). Every seed gave the same plan, within budget 2, and every replay reached the goal.

| | StowCube | SlidingDrawer |
|---|---|---|
| Plan | pick(B) → place(B,src_b) → approach(A) → HOLE[clear_of_walls(A)] → pick(A) → place(A,src_a) → **retreat_to_safe** | approach_handle → HOLE[handle_grasped] → pull_open |
| Cost (holes, displaced, length) | (1, 0, 7) | (1, 0, 3) |
| Missing capability | clear_of_walls(A), positive | handle_grasped, positive |
| Hole start | tcp_near(A) | at_handle_standoff |
| Fences | in_region(B,src_b), ¬in_region(B,mouth) | none |
| Library only | no plan; unmet needs listed | no plan; unmet needs listed |

- Both drag specs are identical to the historical ones in target, start and fences. Only the plan around them changed.
- **StowCube without the interference model:** two plans tie, and both put the drag before B is moved. Every tie now uses the one trained skill, so the old "untrained second skill" problem is gone.
- **Drawer ablations:**
  - With a grasp primitive in the library, the plan needs no hole.
  - The relaxed "unrestricted" planner now returns the grasp plan too. Historically it chose HOLE[drawer_open].
  - So the direct baseline's plan, approach_handle → HOLE[drawer_open], is now fixed by protocol rather than produced by a planner (PLAN item 10).
- **The direct baseline's spec** (`E2_stage2/direct/`):
  - target drawer_open, start at_handle_standoff;
  - one derived fence, ¬handle_grasped;
  - in the model, its plan leaves ¬at_handle_standoff unmet. Both facts are recorded in the spec's folder.
- **Fluents nothing measures:** holding(·), and the in_region slots outside the registry. Every Stage 2 record lists them; they are false at the start by the closed world.

## C. Retreat verification (R, E0)

The gate (`scripts/stow_retreat_gate.py`) ran the unchanged `retreat()` rise from 5 states that the historical inverse reaches, on 20 seeds each (1000–1009 and 2000–2009). It checked:
- both tcp_near ≤ 0.2 afterwards;
- every other predicate keeps its robust value;
- no cube moves more than 1 mm;
- no contact after the first step.

Results of the rerun at the final code (E0, commit c03f4eb):

| Start state | Gripper | Passed | Largest tcp_near after | Largest cube displacement |
|---|---|---|---|---|
| after the forward skill | open | 20/20 | 0.0007 | 0 mm |
| after place(B,src_b) | open | 19/19 | 0.005 | 0.0005 mm |
| after approach(A) | closed, pressing on A | 0/19 | 0.004 | A: 79 mm |
| after the learned drag | closed, on A | 0/19 | 0.020 | A: 44 mm |
| after place(A,src_a), the plan's end | open | 19/19 | 0.006 | 0 mm |

- **The motion itself is collision-free in every state.** Contact force after the first step was zero in all 96 runs that reached a state, and both tcp_near scores ended at 0.02 or below.
- **What fails with a closed gripper** is the release of the press: cube A, squeezed against the pocket wall, springs back as the fingers lift.
- **So the precondition gripper_open() comes from measurement,** not from design (attention item 1). With it, the action passed 58 of 58 runs at the final code, and 60 of 60 in the first gate at 7279aad. The plan uses it after place(A), where the gripper is open.
- **Seed 2001 is missing from four states:** its forward push stopped short, the Boolean forward check of that time accepted it, and pick(B) failed (attention item 4).
- The full analysis is in `R_retreat_validation/ANALYSIS.md`, and the rerun in `E0_stage0/retreat_gate.json`.
- **Stage 0 gates:** both tasks passed at c03f4eb (`E0_stage0/`). The Drawer gate checks both negative goal literals at ≤ 0.2.

## D. RL results (E3)

### StowCube drag skill, HOLE[clear_of_walls(A)]

Run `E3_stage3/stow/1794407215`, from scratch:
- **Budget and schedule:** 120,000 steps, SAC seed 0, curriculum 60,000 steps.
- **Evaluation:** every 4,000 steps, on seeds 1000–1009.
- **Learning problem:** the E2 spec, which is identical to the historical one in target, start and fences.
- **Unchanged:** the SAC settings and the environment's initialization, gripper and press start (decision 4).

| | Fresh run (signed) | Historical run 1790788534 |
|---|---|---|
| Best checkpoint, chosen on the contract | step 4,000 | step 120,000 (also saved at 4,000 and 16,000) |
| Held-out success (clear_of_walls(A) ≥ 0.8) | 10/10 | 10/10 |
| Fences intact (in_region(B,src_b) ≥ 0.8, in_region(B,mouth) ≤ 0.2) | 10/10 | 10/10 |
| Contract (both) | 10/10 | 10/10 |
| Mean steps; mean x of A | 3.7; 56 mm | 3.7; 53 mm |
| Final model's contract | 10/10 | 10/10 |
| Evaluations at 100% contract | 27 of 30 (dips at 16k: 0%, 20k: 90%, 64k: 50%) | 8 of 30 |
| Training time | 19.0 min, sharing the GPU with four Drawer runs | 16.7 min |
| Stage 3 verdict | PASS | PASS |

- **Reference point:** the scripted drag frees A in 10/10 scenes, but breaks a fence in all 10.
- **Handoff sweep:** raising the handoff by 5–20 mm gives 50%, 100%, 40% and 20%. Historically it gave 100%, 90%, 70% and 80%. It is a robustness probe, not a pass criterion.
- **One run per condition:** the two learning curves cannot be compared for stability.

### SlidingDrawer grasp, HOLE[handle_grasped]

- **Runs:** `E3_stage3/drawer/seed<k>_1794408487`, SAC seeds 0–3, 30,000 steps each.
- **Seeds:** validation every 2,500 steps on scenes 500–509; test on scenes 1000–1019; training scenes 0–31.
- **Test procedure:** each test episode is followed by the scripted pull_open.

| SAC seed | First validation ≥ 90% | Selected checkpoint | Test: grasp held | Test: pull_open then opens | Historical: first ≥ 90%, test grasp |
|---|---|---|---|---|---|
| 0 | 10,000 | 10,000 | 20/20 | 20/20 | 12,500; 20/20 |
| 1 | 10,000 | 10,000 | 20/20 | 20/20 | 7,500; 20/20 |
| 2 | 12,500 | 12,500 | 20/20 | 20/20 | 10,000; 20/20 |
| 3 | 12,500 | 12,500 | 20/20 | 20/20 | 10,000; 20/20 |

- **Final models:** all four also score 20/20 on both measures.
- **Checkpoints:** their SHA-256 hashes are in `E3_stage3/drawer/checkpoints.json`.
- **First attempt:** it collided (attention item 8). Its logs, kept, show exactly the same validation curves and first-grasp steps for every seed. So the retraining reproduced it step for step.
- **Recorded durations:** the runner collects jobs in order. So the attempt-1 records show 1,162 s for every Drawer seed, which is when the Stow run finished. The rerun records show each process's own wall time, 408–410 s, of which training took 6.4 min.

### Direct baseline, HOLE[drawer_open]

**Protocol: the historical one, unchanged** (decision 6):
- one policy, from the handle standoff;
- 100-step episodes;
- the Drawer seed-0 run's SAC settings;
- 30,000 steps, SAC seeds 0–3;
- checkpoints chosen on validation scenes 500–509.

**What changed is the goal and the reward:**
- Each episode is judged on the complete signed goal: drawer_open ≥ 0.8, drawer_closed ≤ 0.2, handle_grasped ≤ 0.2 and at_handle_standoff ≤ 0.2.
- The reward has the derived fence ¬handle_grasped (attention item 2).
- The precheck passed (`E3_stage3/direct/precheck.json`). The episodes start exactly as the grasp hole's do, the reward and goal are measured at every step, and nothing is scripted.
- The precheck's scripted reference controller opens the drawer in 42 of 42 starts, but meets the complete signed goal in none, because it never lets go of the handle.

| SAC seed | First validation ≥ 90% (complete signed goal) | Stays ≥ 90% afterwards | Selected | Historical (positive-only goal): first ≥ 90%, stays |
|---|---|---|---|---|
| 0 | 30,000 (the last evaluation) | — | 30,000 | 25,000; no (40% at 30,000) |
| 1 | 17,500 | yes | 17,500 | 17,500; yes |
| 2 | 22,500 | yes | 22,500 | 22,500; no (0% at 30,000) |
| 3 | 30,000 (the last evaluation) | — | 30,000 | 25,000; yes |

- **Outcome:** all four direct policies reach the complete signed goal on validation. They take 17,500–30,000 steps, against 10,000–12,500 for the targeted grasp, about twice as long, as historically.
- **How they open the drawer:** the training statistics count more episodes that opened the drawer than episodes that grasped the handle. Seed 1, at 30,000 steps: grasped 0%, opened 100%. Under the fence, the policies appear to open the drawer without what the predicate counts as a grasp. Section E checks this on the test scenes.

## E. Full physical execution (E4)

**How every condition is judged:**
- each plan step's preconditions and modelled effects are checked robustly;
- success means the complete signed goal, every literal judged robustly at the end;
- the Boolean verdict is reported beside it. It agreed with the robust verdict in every Stage 4 condition of both tasks.

**Implementation fix during E4.** The first E4 attempt (commit adfa69d) crashed in both Stage 4 scripts when writing results into an output folder that did not exist. Commit 6acc02e creates the folder; E4 was rerun from the start. The failed attempt's records and logs are kept as `*.attempt1_failed.*`.

### StowCube (seeds 1000–1009; the learned operator from scenes 2000–2009)

**The plan**, made in every episode from the measured post-forward state: the E2 plan, on all 10 seeds. Two forward pushes (seeds 1001 and 1007) fell short of the stowed state and were re-rolled.

**With the checkpoint the protocol selected (4,000 steps):**

| Condition | Complete signed goal | Boolean goal | Where it stops |
|---|---|---|---|
| full: planner + learned drag + retreat | **8/10** | 8/10 | seeds 1001 and 1009: the drag leaves clear_of_walls(A) at 0.38 and 0.74 after 2 retries; A ends 118 and 100 mm from its seat |
| library only | 0/10 | 0/10 | pick(A) refused, clear_of_walls(A) not robustly true (0.12; 0.78 on seed 1002) |
| without the interference model | 0/10 | 0/10 | approach(A) refused on 9 seeds: tcp_near(B) reads 0.25–0.31 next to A, so ¬tcp_near(B) is not robustly measured. On seed 1002 the hand is not seated on A. |
| relearned: replanned with the learned operator, no holes allowed | 0/10 | 0/10 | **no plan:** the learned operator has no effect clear_of_walls(A) |

- **Stage 4 verdict: FAIL.** Its criteria need relearned ≥ 80% and the target among the learned operator's effects.
- **Why the learned operator lacks its target:**
  - The operator was extracted, by the approved rule, from 10 executions that each started at a valid handoff (raw validity 10 of 11).
  - clear_of_walls(A) ended at 0.8 or above in 9 of them, and at 0.77 in one.
  - The rule asks every demonstration to show the transition robustly, so the effect is dropped. Nothing was added by hand (decision §11).
- **The handoff on the evaluation seeds** was valid on 9 of 10. On seed 1002, placing B pulled A partly out of the pocket. The full plan still met the goal there.
- **Model versus world:** 4 diverging steps in the full condition. Placing B disturbed A on seed 1002, and the drag left the measured state short of the model on the two failed seeds.

**Diagnostic (not the protocol): is it the code or the checkpoint?** The same Stage 4 code and inputs were run with two other checkpoints (`D_diagnostics/`):

| Checkpoint | full | library only | without interference | relearned | Target in the learned operator | Stage 4 |
|---|---|---|---|---|---|---|
| Protocol-selected, step 4,000 (E4 above) | 8/10 | 0/10 | 0/10 | 0/10 | no | FAIL |
| Fresh run's final model, step 120,000 | 10/10 | 0/10 | 0/10 | 9/10 | yes | PASS |
| Historical checkpoint 1790788534 | 10/10 | 0/10 | 0/10 | 9/10 | yes | PASS |

- **The code is sound.** The failures come from the checkpoint the selection rule chose.
- **Why the rule chose step 4,000:**
  - That checkpoint already reached contract 10/10 at the fewest mean steps, 3.7.
  - The rule replaces the best checkpoint only when a later one is strictly better, so no later one could.
  - The historical run's first perfect checkpoint took 4.5 steps, so later checkpoints replaced it.
- **The checkpoint is also weaker away from the selection scenes:** 50% at a +5 mm handoff offset (section D).
- **Changing the selection is your decision** (decision 7 keeps checkpoint selection; §17). I have not changed it.
- **The interference ablation now fails earlier than before:** 0/10 against the historical 3/10. The robust check refuses approach(A) at once, where the Boolean check let it continue and fail later.

### SlidingDrawer Stage 4 (seeds 2000–2019)

| Condition | Plan | Complete signed goal | Boolean | Retries | Model ≠ world |
|---|---|---|---|---|---|
| learned grasp, SAC seed 0 | approach_handle → HOLE[handle_grasped] → pull_open | 20/20 | 20/20 | 0 | 0 |
| learned grasp, SAC seed 1 | same | 20/20 | 20/20 | 0 | 0 |
| learned grasp, SAC seed 2 | same | 20/20 | 20/20 | 0 | 0 |
| learned grasp, SAC seed 3 | same | 20/20 | 20/20 | 0 | 0 |
| library only | same, no policy for the hole | 0/20 (pull_open refused: handle_grasped 0.12) | 0/20 | 0 | 0 |
| scripted grasp in the library | approach_handle → grasp_handle → pull_open | 20/20 | 20/20 | 0 | 0 |

- Both negative goal literals, ¬at_handle_standoff and ¬handle_grasped, were verified at ≤ 0.2 in every successful episode (decision 5).
- The drawer moved at most 4.6 mm before pull_open, and ended open at 146.5–147.9 mm.

### Targeted against direct (configurations 3000–3049, never used before)

**Protocol: the historical one** (decision 6):
- the same forward push and approach for every policy;
- the targeted policy runs until handle_grasped ≥ 0.8 (at most 40 steps), then pull_open;
- the direct policy runs until the complete signed goal holds (at most 100 steps);
- then 10 steps of settling, the arm held still and the gripper command unchanged;
- success is the complete signed goal after settling.

The starts were identical for all eight policies.

| Policy | Complete signed goal (the protocol's success) | Policy-level target (grasp / drawer open) | Goal held when the policy stopped | Validation: first ≥ 90% |
|---|---|---|---|---|
| targeted, SAC seed 0 | 49/50 | 49/50 | 49 | 10,000 |
| targeted, SAC seed 1 | 49/50 | 49/50 | 49 | 10,000 |
| targeted, SAC seed 2 | 49/50 | 49/50 | 49 | 12,500 |
| targeted, SAC seed 3 | 49/50 | 49/50 | 49 | 12,500 |
| direct, SAC seed 0 | **20/50** | 42/50 | 42 | 30,000 |
| direct, SAC seed 1 | **0/50** | 49/50 | 49 | 17,500 |
| direct, SAC seed 2 | **34/50** | 49/50 | 49 | 22,500 |
| direct, SAC seed 3 | **39/50** | 49/50 | 49 | 30,000 |

- **Every policy's 50th configuration** is the one the forward push left short of the handoff, as in the historical evaluation.
- **The direct policies reach the complete goal and then lose it while settling.** In 22, 49, 15 and 10 episodes, by seed, the goal held when the policy stopped and was gone 10 steps later:
  - Seeds 0 and 1, all 71 losses: the fingers closed on the handle while the gripper command was held, so the last command was a close (handle_grasped up to 0.98).
  - Seeds 2 and 3: at_handle_standoff rose above 0.2 during the still steps, to 0.20–0.47, in 24 episodes; one more on seed 2 ended grasped.
  - The evaluation script files these as "drawer open, but goal literals unmet", because the drawer is still open. The rows record goal_before_settling, which shows what happened.
- **Why validation did not see it:**
  - Validation runs each direct episode for its full 100 steps, so the policy finishes its motion before the goal is judged.
  - The final protocol stops it at the first step where the goal holds, often mid-motion.
  - With the positive-only goal this did not matter: an open drawer stays open whatever the gripper does.
- **Diagnostic (not the protocol):** the same configurations, each direct policy run for its full 100 steps as in validation. Complete signed goal: **42, 49, 49 and 49 of 50** (`D_diagnostics/direct_full_horizon.json`). Seed 0's 7 further misses never opened the drawer past 88 mm.
- **How the drawer was moved** (recorded when it first passes 50 mm open):
  - targeted: always a two-finger pinch;
  - direct, mostly a one-finger press: seed 0, 20 of its 20 successes; seed 2, 24 of 34 (10 pinch); seed 3, 30 of 39 (9 pinch).
  - The ¬handle_grasped fence appears to push the direct policies away from a full grasp.
- **So under the historical protocol, the targeted method restores more often.** But most of the direct baseline's shortfall comes from how the protocol stops and settles a policy, not from what the policy learned. The direct baseline still needs about twice the interactions to learn: first ≥ 90% at 17,500–30,000 steps, against 10,000–12,500. Which evaluation is right for the paper is your decision (§17); I have not changed the protocol.

## Previous against updated (E5)

- **Previous:** the published pipeline in `artifacts/stow/` and `artifacts/drawer/`: positive-only goal, one 0.8 threshold.
- **Updated:** this folder's fresh runs.
- **Re-judged:** the previous runs' recorded final scores read against the new signed goal. These are recorded numbers, not new runs.
- The full table, with every literal, is in `E5_comparison/comparison.md` and `comparison.json`.

| Property | StowCube previous | StowCube updated | SlidingDrawer previous | SlidingDrawer updated |
|---|---|---|---|---|
| Preconditions | 4 positive | 4 positive, 4 negative (¬in_region(A,pocket), ¬in_region(B,mouth), ¬tcp_near(A), ¬tcp_near(B)) | drawer_open | drawer_open, ¬drawer_closed, ¬at_handle_standoff, ¬handle_grasped |
| Effects | 2 add, 3 delete | the same 5, as signed literals | drawer_closed, ¬drawer_open | the same |
| Restoration goal | 6 literals | 8 (adds ¬tcp_near(A), ¬tcp_near(B)) | 2 literals | 4 (adds ¬at_handle_standoff, ¬handle_grasped) |
| Missing capability | clear_of_walls(A) | clear_of_walls(A) | handle_grasped | handle_grasped |
| Symbolic plan | 6 steps, ends at place(A) | the same, then retreat_to_safe | approach → HOLE → pull_open | the same |
| Initiation context | tcp_near(A) | tcp_near(A) | at_handle_standoff | at_handle_standoff |
| Preservation conditions | in_region(B,src_b), ¬in_region(B,mouth) | the same | none | none |
| SAC held-out success | contract 10/10 (120k steps) | contract 10/10 (selected at 4k) | 4 seeds: 20/20 grasp, 20/20 pull | 4 seeds: 20/20 grasp, 20/20 pull |
| Full execution, full plan | 10/10 (positive goal); **0/10** re-judged on the signed goal (the hand stays at A) | **8/10** on the signed goal; 10/10 with the final or historical checkpoint (diagnostic) | 4 × 20/20; 20/20 re-judged | 4 × 20/20 |
| Library only | 0/10 | 0/10 | 0/20 | 0/20 |
| Without interference model | 3/10 | 0/10 (refused at approach(A)) | — | — |
| Relearned operator | 9/10 | 0/10, no plan (9/10 with the final or historical checkpoint) | — | — |
| Targeted vs direct (3000–3049) | — | — | targeted 4 × 49/50, direct 4 × 49/50 (positive goal); re-judged on the signed goal: targeted 4 × 49/50, direct 4 × **0/50** (never lets go) | targeted 4 × 49/50; direct 20, 0, 34, 39 /50; run for the full horizon (diagnostic): 42, 49, 49, 49 /50 |
| Model/world differences (steps) | full 3; library 3; without 30; relearned 3 | full 4; library 2; without 4; relearned 0 | 0 in every condition | 0 in every condition |

## F. Tests and the oracle

**301 tests pass** at 6acc02e, the final code (`tests_at_6acc02e.txt`), against 239 at the baseline `63dc282`. That is 62 more tests, 6 of which run in simulation. Ten expectations were changed on purpose, listed below.

**The decisions' 22 required tests:**

| # | Required test | Test |
|---|---|---|
| 1 | Positive preconditions | `test_signed_literals.py::test_a_reliably_true_start_gives_p_and_a_reliably_false_start_gives_not_p`, `::test_an_unchanged_condition_is_a_precondition_and_no_effect` |
| 2 | Negative preconditions | `::test_a_reliably_true_start_gives_p_and_a_reliably_false_start_gives_not_p` |
| 3 | Positive effects | `::test_a_false_to_true_transition_is_an_effect_p_with_not_p_before` |
| 4 | Negative effects | `::test_a_true_to_false_transition_is_an_effect_not_p_with_p_before` |
| 5 | Inconsistent initial scores | `::test_an_inconsistent_start_gives_no_precondition`, `::test_every_demonstration_must_agree`, `::test_a_weak_or_inconsistent_transition_is_no_effect` |
| 6 | Robust threshold boundaries | `::test_the_robust_thresholds_are_inclusive`, `::test_robust_truth_needs_0_8_and_robust_falsity_0_2`, `::test_boolean_truth_puts_0_5_on_the_positive_side` |
| 7 | Contradictory literals | `::test_contradictory_literal_sets_are_rejected`, `::test_contradictory_operators_are_rejected`, `test_signed_planning.py::test_a_goal_asking_for_both_p_and_not_p_is_rejected` |
| 8 | Signed goal satisfaction | `test_signed_literals.py::test_a_literal_and_its_complement_hold_in_opposite_states`, `::test_the_restoration_goal_is_pre_with_the_complement_of_every_effect`, `test_signed_planning.py::test_a_negative_goal_is_met_by_the_library_without_a_hole` |
| 9 | Positive hypothetical operators | `test_signed_planning.py::test_a_positive_prerequisite_is_learned_to_enable_an_operator_that_makes_a_goal_false` |
| 10 | Negative hypothetical operators | `::test_a_negative_prerequisite_is_learned_to_enable_a_positive_operator`, `::test_a_negative_hole_obeys_the_disturbances_like_any_action`, `::test_a_negative_hole_ends_a_long_plan_with_no_length_bound` |
| 11 | Mixed-polarity exact planning | `::test_a_mixed_goal_takes_one_positive_and_one_negative_hole`, `::test_the_planner_agrees_with_the_oracle_on_signed_problems` |
| 12 | Signed upstream dependencies | `::test_a_negative_hole_is_replaced_by_its_upstream_prerequisites_only_within_the_budget`, and the both-signs upstream test in `test_exact_hole_planner.py` |
| 13 | Subset-minimality | the oracle comparisons: every returned hole set is admissible, so subset-minimal, and of the smallest admissible size |
| 14 | Planner costs | the oracle comparisons: the plan's (holes, displaced, length) equals the oracle's best |
| 15 | Negative fence penalties | `test_signed_execution.py::test_the_reward_rises_toward_either_target_and_charges_only_broken_fences` |
| 16 | Positive fence penalties | the same test |
| 17 | Negative RL target achievement | `::test_the_drawer_environment_learns_toward_a_negative_target` |
| 18 | Negative physical effect verification | `::test_the_executive_runs_a_learned_step_that_owes_a_negative_literal`, `::test_retreat_to_safe_runs_through_the_executive_and_leaves_both_cubes` |
| 19 | Polarity-preserving learned operators | `test_signed_planning.py::test_a_learned_operator_with_a_negative_precondition_runs_only_while_its_atom_is_false`, `test_signed_literals.py::test_serialization_keeps_every_sign` |
| 20 | Complete symbolic and physical goal | `test_signed_execution.py::test_the_executive_judges_a_mixed_signed_goal_robustly` |
| 21 | The StowCube retreat | `::test_retreat_to_safe_is_modelled_as_measured_and_placed_by_the_planner`, `::test_retreat_to_safe_runs_through_the_executive_and_leaves_both_cubes` |
| 22 | Positive-only artifacts | `test_signed_literals.py::test_a_positive_only_artifact_reads_back_with_its_goal_unchanged` (the three committed positive-only operators) |

**The independent oracle** (`tests/test_exact_hole_planner.py::Oracle`) enumerates hole sets and plans by brute force and shares no code with the planner. It now uses signed pairs. It checks every guarantee:
- NoPlan is exact;
- the holes are admissible, so subset-minimal, and of the smallest admissible size;
- the plan is valid;
- the cost is the least;
- every best hole set is reported.

It agrees with the planner on 3,400 random problems:
- 1,500 with few literals and 700 with chains, from before;
- 1,200 new problems weighted to deletes, negative preconditions and negative goals. Over 100 of these need a negative hole.
- It also agrees on 6 named signed problems.

**Superseded tests.** Ten expectations changed on purpose. Each is marked "Superseded 2026-11-11" where it changed, with the reason:
- `test_operator_extraction.py`, 2: the signed rule now extracts ¬p preconditions (the extractor's recorded name changed with it);
- `test_parameterized_operator.py`, 1: the distractor-invariance test, split into the historical mean-mode check and the signed difference;
- `test_handoff_validation.py`, 2: the robust handoff threshold, and the seed 2004 reasons that follow from it;
- `test_hole_planner.py`, 2: the Boolean 0.5 state threshold, and a hole's literal read from a signed field instead of its name;
- `test_exact_hole_planner.py`, 2: two tests that pinned positive-only limitations (a negative-only goal got no plan; upstream read positive preconditions only), now pinning the signed behaviour;
- `test_minimal_formulation.py`, 1: upstream reads both signs.

Elsewhere, only the tests' representation changed (names(), strs(), pos()); their expectations did not.

## G. Manuscript

- `docs/manuscript/method_signed_literals.tex` is a standalone Method section in LaTeX. Its subsections:
  - predicates, literals and states;
  - operators;
  - extraction;
  - the restoration objective;
  - the inverse problem;
  - hypothetical operators;
  - choosing what to learn;
  - learning a capability;
  - execution.
- `docs/manuscript/definition_to_code.md` maps every definition to its code and tests.
- It describes `retreat_to_safe` as a predefined library primitive, verified physically, not a learned skill.
- It does not claim causal necessity for the chosen holes, nor that every signed effect is physically realizable.
- The repository holds no manuscript source, so this section is written to be merged into the paper by hand.

## H. Reproducibility

**Code.** Every result belongs to one commit, recorded with it:

| Commit | What | Stages run there |
|---|---|---|
| 7279aad, 4f40cd8 | retreat binding and gate; retreat_to_safe registered | R |
| 068fc66, 50e35cf, 266c40a | signed core, planner, learning and execution | |
| 01809a9, c03f4eb | checkpoint manifest tool; the gate reads a signed target | E0, E1, E2 |
| b621e00 | robust Stow forward check | E3: Stow, the direct precheck |
| adfa69d | training scripts never share a run folder | E3: Drawer, direct |
| 6acc02e | Stage 4 scripts create their output folder | E4; the diagnostics; tests |
| d43c67d | README, the Method section, the older documents labelled | |

**The runner, `run.py`.**
- It runs each stage's repository scripts unchanged.
- It refuses to start with tracked changes.
- It writes `records/<step>.json` (command, commit, tracked changes, start time, duration, exit code) and `logs/<step>.log`.
- It never reruns a step that has a record.
- Superseded attempts keep their records and logs, renamed `*.attempt1_*`, with notes.

**The manifest, `manifest.json`** (written by `manifest.py`) lists:
- every step, with its command, commit and clean tree;
- each input's and output's size and SHA-256;
- the seeds of every stage;
- the size and SHA-256 of every file in this folder.

The checkpoint manifests, `E3_stage3/{drawer,direct}/checkpoints.json`, give each selected checkpoint's SHA-256. Stage 4 and the final evaluation check them before loading.

**Seeds:**
- Stage 1 demonstrations: 0–4.
- Stage 2 measured states: 0–4.
- Stow SAC: seed 0; training scenes 0–31; selection and evaluation on 1000–1009.
- Drawer SAC: seeds 0–3; training scenes 0–31, validation 500–509, test 1000–1019.
- Direct SAC: seeds 0–3, on the same scene sets.
- Stow Stage 4: evaluation 1000–1009; learned operator from 2000–2009, with failed scenes replaced at seed + 10,000 (12001, 12004).
- Drawer Stage 4: 2000–2019.
- Final evaluation: 3000–3049.
- Retreat gate: 1000–1009 and 2000–2009.

**Software:** Python 3.10.21, torch 2.7.1+cu128, stable-baselines3 2.8.0, mani-skill 3.0.1, sapien 3.0.3, gymnasium 1.2.3, numpy 2.2.6. Hardware: an RTX 5070 Ti Laptop GPU, driver 595.99.02. Everything ran with PYTHONNOUSERSITE=1.

**What is committed and what stays local.**
- Committed: this report, the plan, the runner and the comparison and manifest scripts, the records, the stages' JSON outputs, the comparison, the diagnostics' scripts and JSON outputs, the notes, and the test output.
- Local only: the checkpoints (.zip), the console logs, and the baseline's copies of committed files. The manifest identifies each by SHA-256.

**To rerun:** `PYTHONNOUSERSITE=1 python artifacts/signed_literals_20261111/run.py E0`, then E1 to E4, in a copy of this folder without the outputs. Then run `compare.py` and `manifest.py`.
