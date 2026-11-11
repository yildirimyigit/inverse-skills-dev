# Signed literals: execution plan

- **Version:** revised 2026-11-11 after the user's decisions, given in "Final Design Decisions and Instructions".
- **Branch:** `feature/signed-literals`, from `63dc282`. Local commits are made at each finished stage and never pushed.
- **Status:** experiments complete, two protocol decisions open (items 16 and 17).
  - Stages A–D: commits 7279aad to b621e00.
  - E0–E2: run at c03f4eb.
  - E3: at b621e00 (Stow), then adfa69d (Drawer and direct, after the run-folder fix).
  - E4: at 6acc02e, after the output-folder fix.
  - E5: the comparison, from recorded files.
  - Tests: 301 pass at 6acc02e.

## 1. Decisions in force

| # | Decision | Source |
|---|---|---|
| 1 | Signed-literal formulation. G = Pre(o). P_inv = ⟨O, S, S0, G, δ⟩. No relevance filtering. | §1 |
| 2 | **Thresholds.** Boolean truth is V ≥ 0.5, used for the planner's state. Robust truth is p ≥ 0.8 and ¬p ≤ 0.2. It applies to extraction, physical preconditions and effects, RL target success, fences, handoffs and the final goal. Mismatches are reported. | §2 |
| 3 | **Stow retreat.** No second learned skill. A known action, `retreat_to_safe`, built on the existing `retreat()` primitive, with effects {¬tcp_near(A), ¬tcp_near(B)}. It is registered only after it is verified in simulation. If it fails, stop and report. | §3 |
| 4 | **Stow drag skill.** Its initialization, gripper behaviour, press/contact start, curriculum and control interface stay as they are. The generic framework supports negative targets. Stow's drag-specific training code cannot train a negative skill, and that is stated as a limitation. | §4 |
| 5 | **Drawer.** Rerun everything fresh, without forcing the old plan. Verify both negative goal literals at ≤ 0.2. | §5 |
| 6 | **Direct baseline.** Keep the historical one-policy protocol. It is retrained and judged on the same full signed goal. Report results at policy level and at full-restoration level. | §6 |
| 7 | **Budgets and seeds.** Stow drag skill: 120,000 steps. Drawer: 4 seeds × 30,000 steps. Training scenes 0–31, validation 500–509, test 1000–1019. Hyperparameters and checkpoint selection are unchanged. Everything is retrained from scratch. | §7 |
| 8 | **Git.** Work on `feature/signed-literals`. Commit locally by stage and never push. Never reset or discard work, and never overwrite artifacts. | §8 |

## 2. Superseded requirements

| Earlier requirement | Superseded by |
|---|---|
| Original spec §31: train every hole of the plan, which would mean a second Stow policy for ¬tcp_near(A) | Decision 3: the `retreat_to_safe` library action |
| My plan, Decision 1(a): every Stow skill inherits its gripper state from the plan | Decision 4: withdrawn; the drag skill is unchanged |
| Original spec §31: Stow default of 150,000 steps | Decision 7: 120,000, the recorded reference budget |
| My plan, Decision 3(a); original §34 | Decision 6, which adds reporting at both policy level and full-restoration level |
| Planner freeze of 2026-11-10 | Lifted by this request |
| `test_i_a_negative_only_goal_with_no_library_gets_no_hole` | Replaced by a test of the signed behaviour, as the original spec requires |
| Stage 1 checks `no_tcp_near_terms` and `no_robot_terms` | Replaced by signed-literal checks |

## 3. Review of my earlier interpretations (old §3), and new items

Status key:
- **Required:** the spec's own text demands it.
- **Engineering:** no change of scientific meaning.
- **Existing:** keeps current behaviour.
- **Confirm:** it changes something, so I am asking you.

1. **Hole start conditions.** **Not implemented; open question.**
   - I first restricted starts to library states where the hole can run. Then I went back to the established rule, for both signs: the largest set of robot-relative facts the library holds together in any reachable state.
   - The established rule already meets the spec's wording: a hole only ever runs where its start and its own enabling condition hold together.
   - The restriction would change the definition in edge cases, as `test_hole_starts_are_the_relaxed_modes_on_stow_and_the_drawer` showed, so it needs your approval.
   - Neither task is affected.
2. **Signed upstream graph.**
   - An action establishes ¬p by deleting p, by adding a mutex partner of p, or by triggering a disturbance that displaces p.
   - Both positive and negative preconditions count.
   - Status: **Required.** Original §13 says "deletion, mutex implications, or other modeled transitions", and your §9 says "signed effects and signed preconditions".
   - Effect: none on either task's plan.
3. **Observation of a learned skill.** Its first number becomes the target literal's signed margin.
   - Status: **Engineering.** Original §16 says "update ... observations wherever they assume a positive target".
   - Effect: identical for positive targets, and every target trained here is positive.
4. **The relaxed planner** stays as the historical comparison mode, with positive holes only and the new types. The Drawer's "unrestricted" ablation keeps running, labelled historical.
   - Status: **Existing** (an established ablation is kept).
5. **Stow Stage 4 plans each episode from its measured post-forward state.**
   - A forward skill that does not leave the state Stage 2 planned from is re-rolled and reported, as today.
   - The reference state comes from the Stage 2 artifact, not from the hard-coded `stowed_state()`.
   - Status: **Required** ("forward → measured state → exact planning").
6. **Domain fluents nothing measures** are `holding(·)` and the `in_region` slots outside the registry. They stay false under the closed world, and every Stage 2 and Stage 4 record lists them explicitly.
   - Status: **Existing**, made explicit.
7. **Implications and deletes.** Deleting p does not delete the literals that imply p, as with library deletes today. This does not arise in either task.
   - Status: **Existing**, stated as a limitation.
8. **Stow without the interference model**, where the tie needed an untrained skill.
   - Status: **Moot.** With the retreat action, every tie uses the one trained drag skill.
9. **Stow checkpoint selection** uses seeds 1000–1009, the same as Stage 4 evaluation.
   - Status: **Existing**, stated as a limitation.

New items arising from the decisions:

10. **The direct baseline's learning problem.**
    - The historical direct plan, `approach_handle > HOLE[drawer_open]`, is fixed by protocol, because the planner no longer produces it under the signed goal. Its fences are derived from that plan under the new goal, as the protocol always did.
    - This adds one fence: don't grasp the handle (handle_grasped ≤ 0.2). The direct reward becomes 2V(open) − 1 + min(0, 1 − 2V(grasped)).
    - Status: **Follows decisions §6 and §10; implemented; flagged in the final report.** I first listed it for confirmation. Your decisions then settled both parts: §10 makes plan-derived signed fences non-negotiable, and §6 says to retrain the baseline when its goal or reward changes. So I did not ask again.
    - Recorded in `E2_stage2/direct/`: the fence the protocol plan derives (¬handle_grasped), and the literal its model plan leaves unmet (¬at_handle_standoff). That literal is judged physically in the final evaluation, like every goal literal.
11. **`retreat_to_safe` binding.**
    - It is the existing `retreat()` motion, with the gripper commanded to stay as it is: open stays open, closed stays closed. So the action has no gripper effect.
    - Its precondition is "not holding a cube", because rising while holding would carry the cube along. Any other precondition will come only from the simulation evidence, and will be reported.
    - Status: **Engineering plus evidence.** Everything is reported before registration.
    - **Outcome (gate in `R_retreat_validation/`, commit 4f40cd8):**
      - Open-gripper states: 60 of 60 runs reached both tcp_near ≤ 0.2, with no cube displaced and no contact.
      - Closed-gripper states: 0 of 40 succeeded. These are the states where the gripper presses on A: after approach(A), and after the drag. The rise itself touches nothing, but releasing the press lets A spring back, by 1.4 to 79 mm.
      - So the registered precondition is gripper_open() together with "not holding". It comes from measurement, not from design, and the final report flags it.
      - The plan places the retreat after place(A), where the gripper is open. The E0 rerun at c03f4eb passed 58 of the 58 runs that met the registered preconditions, and none of the 38 closed-gripper runs. Seed 2001 reached only the first state, after the forward skill, where the retreat passed: its push stopped short of the stowed state, the Boolean forward check of that time accepted it, and pick(cube_b) then failed. Item 13 is the fix.
12. **The Stow executive verifies every modeled effect of each step robustly**, as the Drawer executive already does. Today it checks one measure per action.
    - Status: **Required**, by original §18 and your approval of robust effect verification.
    - Effect: it can change Stow retries and failures compared with the historical protocol, and I will report any such change.

13. **The Stow Stage 4 forward check is robust and signed** (commit b621e00).
    - The forward skill counts as done when every one of its signed effects holds robustly, and the measured Boolean state equals the state Stage 2 planned from.
    - Before this, the check read the Boolean state alone. The retreat gate then accepted seed 2001's half-finished push as stowed.
    - Status: **Required**, by decision 2 (robust physical effect verification).
14. **The Stow training protocol is reconstructed** from the reference run 1790788534.
    - Recorded there: 120,000 steps and 30 evaluations, so one every 4,000 steps.
    - Not recorded there: the curriculum length and the SAC seed. I use the script's defaults, 60,000 steps and seed 0.
    - Status: **Existing**, stated in the report.
15. **Parameterized templates now differ by distractor.**
    - The signed rule keeps every reliably false initial fact, including facts about a static distractor, and decision 1 forbids relevance filtering. So templates built with different distractors now differ, in those facts only.
    - The published "distractor template invariance" holds for the historical positive-only extraction. The test that checked it was split: it keeps that check under the `mean` mode, and adds one for the signed difference.
    - Status: **Required** consequence; reported.

16. **Stow checkpoint selection. Open: your decision.**
    - The unchanged rule picks the first checkpoint with the best contract and replaces it only with a strictly better one (fewer mean steps). In the fresh run that froze the choice at step 4,000, contract 10/10 at 3.7 steps.
    - With that checkpoint, E4's full plan meets the goal on 8 of 10 seeds. The learned operator lacks its target (one of 10 executions ends at 0.77), so the relearned plan does not exist, and Stow Stage 4 FAILs.
    - With the same run's final model, or the historical checkpoint, the same code gives full 10/10, relearned 9/10 and PASS (`D_diagnostics/`).
    - Options include: keep the result as is; prefer the latest of the tied best checkpoints; or select on more scenes. Each changes the protocol (§17), so nothing was changed.
17. **The final evaluation's stop rule for the direct baseline. Open: your decision.**
    - The historical protocol stops a direct policy at the first step where the goal holds, then settles 10 steps with the gripper command unchanged.
    - Under the signed goal, that stop often falls mid-motion. The goal held at the stop in 42–49 of 50 episodes per seed, and was lost while settling in 10–49 of them.
    - Run for its full 100-step horizon, as in validation, each direct policy restores 42, 49, 49 and 49 of 50 (`D_diagnostics/direct_full_horizon.json`). Under the protocol it restores 20, 0, 34 and 39.
    - Whether the paper should report the protocol as is, the full-horizon variant, or both, is an evaluation-protocol choice (§17). Nothing was changed.
18. **Implementation fixes made during the experiments** (no change of meaning; affected stages rerun; failed attempts kept):
    - adfa69d: the four Drawer seeds had shared one run folder. The folder is now named by seed, and no training script reuses a folder.
    - 6acc02e: Stage 4 and the final evaluation now create their output folder. The first E4 attempt crashed when writing.

## 4. Stages

| Stage | Content | Check |
|---|---|---|
| A | **Baseline** into `00_baseline/`: commit, tree, versions, the test run, and copies with hashes of the reference operators, specs, plans, SAC summaries, budgets, seeds and results | Test counts recorded |
| R | **Retreat validation, before registration.** The existing `retreat()` runs from reachable states of the historical inverse, using the historical drag policy as you asked: after place B, after approach A, after the drag, and after place A, on 20 seeds. I measure tcp_near(A/B), gripper_open and region scores before and after, cube displacement, and contact forces against both cubes and the pocket during the rise. Output goes to `R_retreat_validation/`. | Both tcp_near ≤ 0.2; gripper and regions unchanged; no displacement or contact. If any fails, stop Stow and report. |
| B | **Signed core.** `operators/literals.py` holds the literal type and the thresholds. `schema.py` gets validated polarity, G, and a schema version. `extractor.py` gets the signed rules, diagnostics, and the effect-complement check. | Spec §13 tests 1–8 and 22 |
| C | **Planner.** Signed holes that delete. The exact planner gets signed candidates, enabling, starts, the producer graph, contexts and diagnostics. The `learned_action` fix. Signed fences and specs. The oracle is extended. | Tests 9–14 and 19; oracle and exhaustive checks; the old 239 tests pass, or each intentional change is listed |
| D | **Learning and execution.** Central thresholds. Hole environments with signed targets and fences. Executives with signed checks, one policy per target, and `retreat_to_safe` bound. `handoff.py`. The Stage 1–4, final-evaluation and direct-spec scripts. | Tests 15–18, 20 and 21 |
| E0 | Stage 0 gates for both tasks, and the retreat validation rerun at the final code | Gates pass |
| E1 | Stage 1, fresh, for both tasks | Signed checks |
| E2 | Stage 2, fresh: Stow with the retreat in its library, the Drawer, and the ablations | A plan within budget 2; otherwise stop |
| E3 | **SAC.** Stow drag skill, 120k steps. Drawer, 4 × 30k. Direct baseline, 4 × 30k, with the fence of item 10. | Curves, selection, held-out results, hashes |
| E4 | **Stage 4 for both tasks**, with reintegration and ablations, and the final evaluation on 3000–3049 (targeted and direct) | Full signed goal at ≤ 0.2 / ≥ 0.8 |
| E5 | Comparison with the historical results, labelled | Table |
| F | **Documentation.** README, `pipeline.md`, docstrings and planner docs, plus a standalone LaTeX Method section with a map from definitions to code. There is no manuscript source in the repository. | |
| G | **Final report**, sections A–H of the decisions | |

All outputs go under `artifacts/signed_literals_20261111/`, with a manifest of inputs, outputs, configurations, seeds, hashes and commit.
