# Validation of `min_max` extraction and `exact` planning

> **Follow-up, added later:** this report records the first validation, and its results are kept as they were. [REPORT_handoff_and_migration.md](REPORT_handoff_and_migration.md) fixes the failure it found and completes the migration. It also corrects one estimate below: the jam is not "about one run in ten" but 2 of 53 Stow scenes, once the training seeds are counted.

**Result: `exact` planning and `min_max` forward extraction pass on Stow and Drawer. `min_max` extraction of the learned Stow operator fails: it loses the operator's only add effect, so replanning with the learned skill finds no plan and Stage 4's own check fails. The defaults were not changed.**

| | Stow, `mean + relaxed` (original) | Stow, `min_max + exact` (proposed) | Drawer, `mean + relaxed` (original) | Drawer, `min_max + exact` (proposed) |
|---|---|---|---|---|
| Forward operator (Pre, Add, Del) and inverse goal | as saved | same | as saved | same |
| Hole | `clear_of_walls(cube_a)`, step 4 of 6 | same | `handle_grasped(drawer)`, step 2 of 3 | same |
| Learning specification | as saved | byte-identical | as saved | byte-identical |
| Full inverse, original policies | 10/10 | same, value for value | 20/20 for each of 4 policies | same, value for value |
| Library-only baseline | 0/10 | same | 0/20 | same |
| Other baseline | no interference model: 3/10 | same | scripted grasp: 20/20 | same |
| Learned operator's add effect | `clear_of_walls(cube_a)` | **none** | not used | not used |
| Replanning with the learned operator | plan found | **no plan** | not used | not used |
| Execution of that replan | 9/10 | **0/10** | not used | not used |
| Stage 4's own check | PASS | **FAIL** | no such check | no such check |

## A. Environment and reproducibility

- **Commit:** `c68d83009f7f2ccf06167d2ece380323006957d5` on `feature/exact-hole-planner`, plus the uncommitted files in section F.
- **Runtime:** Python 3.10.21, torch 2.7.1+cu128, stable-baselines3 2.8.0, gymnasium 1.2.3, mani_skill 3.0.1, sapien 3.0.3, numpy 2.2.6, NVIDIA RTX 5070 Ti Laptop GPU, driver 595.99.02, Linux 7.0.0-42. Simulator only; nothing here ran on a real robot.
- **Defaults during the validation, unchanged:** `OperatorExtractionConfig.mode = "mean"` ([extractor.py:19](../../src/inverse_skills/operators/extractor.py#L19)) and `plan(mode="relaxed")` ([hole_planner.py:318](../../src/inverse_skills/operators/hole_planner.py#L318)). Thresholds as committed: mean rule 0.80 and 0.35; `min_max` true at 0.80, false at 0.20.
- **Modes are always named.** Phases 1 and 2 pass the mode in every call. Phase 3 runs the Stage 4 scripts unchanged, with `hole_planner.plan` and `OperatorExtractor` replaced in memory by versions that name the mode, so no result depends on a default.
- **Seeds:** Stage 1 and Stage 2 seeds 0-4 on both tasks; Stow Stage 4 seeds 1000-1009, with learned-operator extraction on 2000-2009; Drawer Stage 4 seeds 2000-2019. Seeds 3000-3049 were not touched.
- **Checkpoints, weights unmodified, SHA-256 matching the record where one exists:**

| Policy | Checkpoint | SHA-256 |
|---|---|---|
| Stow | `artifacts/stow/stage3/1790788534/sac_hole_best.zip` (tracked by Git) | `1e4ce9cc4ad8511f...` |
| Drawer, SAC seed 0 | `artifacts/drawer/stage3/1792053076/sac_hole_best.zip` | `4a35a07f8059ab38...` |
| Drawer, SAC seed 1 | `artifacts/drawer/stage3/1792056712/sac_hole_best.zip` | `0f3d9d0a74608b05...` |
| Drawer, SAC seed 2 | `artifacts/drawer/stage3/1792056715/sac_hole_best.zip` | `c2648db7a8e2c78f...` |
| Drawer, SAC seed 3 | `artifacts/drawer/stage3/1792056719/sac_hole_best.zip` | `a5aed26f033891ab...` |

**Where the modes are chosen today:**

|  | Implicit (uses the default) | Explicit |
|---|---|---|
| Extraction, Stow and Drawer | `stow_stage1_extract.py:53`, `drawer_stage1_extract.py:62`, `stow_stage4_execute.py:87`, `stow_stage1_video.py:181`, `stow_stage1_extractor_comparison.py:78` | `stow_stage1_extractor_comparison.py:99` (`min_max`) |
| Extraction, legacy PlanRob | 5 calls in `src/inverse_skills/planrob_bundle.py`, 9 in `scripts/v0/` | none |
| Planning | 24 `hp.plan` calls in 10 scripts: Stage 2 and 4 for both tasks, their videos, `stow_framework_video.py`, `stow_planner_demo.py`, `drawer_direct_spec.py`; also `compare_hole_planners_random.py:101` | `compare_hole_planners.py:127`; `compare_hole_planners_random.py:102,141` |

**Commands, from the repository root, in this order:**

```bash
export PYTHONNOUSERSITE=1
python scripts/validate_modes_extraction.py stow
python scripts/validate_modes_extraction.py drawer-original
python scripts/validate_modes_extraction.py drawer-current
python scripts/validate_modes_extraction.py stow-learned
python scripts/validate_modes_planning.py stow
python scripts/validate_modes_planning.py drawer
python scripts/validate_modes_execution.py run drawer --operator-from mean --planner relaxed
python scripts/validate_modes_execution.py run drawer --operator-from min_max --planner exact
python scripts/validate_modes_execution.py run stow --operator-from mean --learned-extraction mean --planner relaxed
python scripts/validate_modes_execution.py run stow --operator-from min_max --learned-extraction min_max --planner exact
python scripts/validate_modes_execution.py run stow --operator-from min_max --learned-extraction mean --planner exact
python scripts/validate_modes_execution.py run drawer --operator-from mean --planner exact
python scripts/validate_modes_execution.py run stow --operator-from mean --learned-extraction mean --planner exact
python scripts/validate_modes_execution.py stage3-input stow
python scripts/validate_modes_execution.py stage3-input drawer
python scripts/validate_modes_execution.py compare
```

**Determinism:** the phases were run twice from nothing (the `mean + exact` runs and the Stage 3 check once). The two runs agree exactly on 39 compared results: every score, operator, plan, spec, Stage 4 result and handoff. The `mean + relaxed` reruns also match the saved Stage 4 files value for value. So on this machine the differences reported below are caused by the modes, not by replay variability. No claim is made about other machines.

## B. Operator extraction

Each row extracts twice from the same rollouts. "Tightest margin" is how far the scores stay from the nearest threshold that would change a `min_max` decision. Every demonstration's start and end score of every predicate is in `phase1/<part>.json`.

| Demonstrations | Mean replay vs saved | Pre | Add | Del | Inverse goal | Largest weight difference | Tightest `min_max` margin |
|---|---|---|---|---|---|---|---|
| Stow forward, seeds 0-4 | exact (difference 0) | same | same | same | same | 0.0070 | +0.0801 |
| Drawer forward, original push | exact (difference 0) | same | same | same | same | 0.0054 | +0.1778 |
| Drawer forward, today's push | differs by 0.0100 (expected: the stop rule changed in `646797f`) | same | same | same | same | 0.0452 | +0.1272 |
| **Learned Stow, seeds 2000-2009** | exact (difference 0) | **differs** | **differs** | **differs** | **differs** | 0.0206 | **crossed at seed 2004** |

**Forward operators: both modes agree.**

- **Stow:** Pre `clear_of_walls(cube_a)`, `gripper_open()`, `in_region(cube_a,src_a)`, `in_region(cube_b,src_b)`. Add `in_region(cube_a,pocket)`, `in_region(cube_b,mouth)`. Del `clear_of_walls(cube_a)`, `in_region(cube_a,src_a)`, `in_region(cube_b,src_b)`. The replay also matches the earlier `stage1_replay_scores.json` and `stage1_extractor_comparison.json` exactly.
- **Drawer:** Pre `drawer_open(drawer)`. Add `drawer_closed(drawer)`. Del `drawer_open(drawer)`. The saved operator was written with the `push_close` of `8f428d5`, so that version was read from Git and replayed: it reproduces the saved averages exactly. Today's push (`646797f`) ends less firmly on seed 0 but extracts the same operator.
- **Weights differ slightly, as expected.** No Stow or Drawer stage reads them: `goal_from_operator` and `forward_effects` read only predicate names and polarity. `metadata.extractor` differs by design.

| Term | Predicate | `mean` weight | `min_max` weight | Difference |
|---|---|---|---|---|
| Pre | `clear_of_walls(cube_a)` | 1.0000 | 1.0000 | -0.0000 |
| Pre | `gripper_open()` | 0.9820 | 0.9820 | +0.0000 |
| Pre | `in_region(cube_a,src_a)` | 0.9820 | 0.9820 | -0.0000 |
| Pre | `in_region(cube_b,src_b)` | 0.9820 | 0.9820 | -0.0000 |
| Add | `in_region(cube_a,pocket)` | 0.9662 | 0.9592 | -0.0070 |
| Add | `in_region(cube_b,mouth)` | 0.9554 | 0.9535 | -0.0018 |
| Del | `clear_of_walls(cube_a)` | 0.8805 | 0.8801 | -0.0004 |
| Del | `in_region(cube_a,src_a)` | 0.9820 | 0.9820 | -0.0000 |
| Del | `in_region(cube_b,src_b)` | 0.9820 | 0.9820 | -0.0000 |
| Pre | `drawer_open(drawer)` | 0.9947 | 0.9893 | -0.0054 |
| Add | `drawer_closed(drawer)` | 0.9824 | 0.9778 | -0.0046 |
| Del | `drawer_open(drawer)` | 0.9943 | 0.9889 | -0.0054 |

Per-predicate statistics, Stow forward:

| Predicate | Start mean | Start min - max | End mean | End min - max |
|---|---|---|---|---|
| `clear_of_walls(cube_a)` | 1.0000 | 1.0000 - 1.0000 | 0.1195 | 0.1192 - 0.1199 |
| `gripper_open()` | 0.9820 | 0.9820 - 0.9820 | 0.9760 | 0.9760 - 0.9760 |
| `in_region(cube_a,pocket)` | 0.0000 | 0.0000 - 0.0000 | 0.9662 | 0.9592 - 0.9737 |
| `in_region(cube_a,src_a)` | 0.9820 | 0.9820 - 0.9820 | 0.0000 | 0.0000 - 0.0000 |
| `in_region(cube_b,mouth)` | 0.0000 | 0.0000 - 0.0000 | 0.9554 | 0.9535 - 0.9580 |
| `in_region(cube_b,src_b)` | 0.9820 | 0.9820 - 0.9820 | 0.0000 | 0.0000 - 0.0000 |
| `tcp_near(cube_a)` | 0.0001 | 0.0000 - 0.0003 | 0.0002 | 0.0002 - 0.0002 |
| `tcp_near(cube_b)` | 0.0000 | 0.0000 - 0.0000 | 0.0006 | 0.0006 - 0.0006 |

Per-predicate statistics, Drawer forward with the original push:

| Predicate | Start mean | Start min - max | End mean | End min - max |
|---|---|---|---|---|
| `at_handle_standoff(drawer)` | 0.0000 | 0.0000 - 0.0000 | 0.0000 | 0.0000 - 0.0000 |
| `drawer_closed(drawer)` | 0.0000 | 0.0000 - 0.0000 | 0.9824 | 0.9778 - 0.9864 |
| `drawer_open(drawer)` | 0.9947 | 0.9893 - 0.9996 | 0.0004 | 0.0004 - 0.0004 |
| `handle_grasped(drawer)` | 0.0000 | 0.0000 - 0.0000 | 0.0000 | 0.0000 - 0.0000 |

### The learned Stow operator: the two modes disagree

The mean replay reproduces the saved operator exactly, so the checkpoint and seeds are right. `min_max` gives a different operator:

|  | `mean` (equals the saved action) | `min_max` |
|---|---|---|
| Pre | `in_region(cube_a,pocket)`, `in_region(cube_b,src_b)`, `tcp_near(cube_a)` | `in_region(cube_b,src_b)`, `tcp_near(cube_a)` |
| Add | `clear_of_walls(cube_a)` | **none** |
| Del | `in_region(cube_a,pocket)` | **none** |

**One run is responsible.** At seed 2004, the state the policy receives is not the state the plan predicts: `in_region(cube_a,pocket)` is measured false.

| Seed | `clear_of_walls` start | end | `pocket` start | end | `tcp_near(A)` start | `src_b` start | Measured state = plan's prediction | `min_max` without this run = saved action |
|---|---|---|---|---|---|---|---|---|
| 2000 | 0.1201 | 0.8808 | 0.9549 | 0.0180 | 0.8752 | 0.9789 | yes | no |
| 2001 | 0.1198 | 0.8695 | 0.9579 | 0.0220 | 0.8608 | 0.9811 | yes | no |
| 2002 | 0.1199 | 0.8728 | 0.9719 | 0.0208 | 0.8731 | 0.9762 | yes | no |
| 2003 | 0.1196 | 0.8796 | 0.9668 | 0.0184 | 0.8735 | 0.9814 | yes | no |
| **2004** | **0.4148** | 0.8364 | **0.6656** | 0.0368 | 0.8702 | 0.9563 | **no** | yes |
| 2005 | 0.1200 | 0.8750 | 0.9672 | 0.0200 | 0.8742 | 0.9752 | yes | no |
| 2006 | 0.1196 | 0.8311 | 0.9738 | 0.0397 | 0.8628 | 0.9809 | yes | no |
| 2007 | 0.1199 | 0.8772 | 0.9639 | 0.0192 | 0.8737 | 0.9765 | yes | no |
| 2008 | 0.1194 | 0.8793 | 0.9643 | 0.0185 | 0.8678 | 0.9813 | yes | no |
| 2009 | 0.1196 | 0.8738 | 0.9756 | 0.0204 | 0.8653 | 0.9812 | yes | no |

Which rule and threshold decide each lost term:

- **Add `clear_of_walls(cube_a)`:** needs `start_max <= 0.20`. Seed 2004 starts at 0.4148.
- **Pre and Del `in_region(cube_a,pocket)`:** need `start_min >= 0.80`. Seed 2004 starts at 0.6656.

Thresholds were not changed. Removing seed 2004 alone gives the saved action exactly; removing any other run leaves the empty one.

**It is not a fluke of this seed.** It is the interaction the README already describes: lifting B can drag A partly out of the pocket. The same thing happens once in the evaluation seeds, at seed 1002, where the executive reports the divergence and refuses the relearned step. That is about one run in ten in each seed set, so any larger extraction set would be likely to contain one.

Per-predicate statistics, learned operator:

| Predicate | Start mean | Start min - max | End mean | End min - max |
|---|---|---|---|---|
| `clear_of_walls(cube_a)` | 0.1493 | 0.1194 - 0.4148 | 0.8676 | 0.8311 - 0.8808 |
| `gripper_open()` | 0.0000 | 0.0000 - 0.0000 | 0.0000 | 0.0000 - 0.0000 |
| `in_region(cube_a,pocket)` | 0.9362 | 0.6656 - 0.9756 | 0.0234 | 0.0180 - 0.0397 |
| `in_region(cube_a,src_a)` | 0.0000 | 0.0000 - 0.0000 | 0.0000 | 0.0000 - 0.0000 |
| `in_region(cube_b,mouth)` | 0.0000 | 0.0000 - 0.0000 | 0.0000 | 0.0000 - 0.0000 |
| `in_region(cube_b,src_b)` | 0.9769 | 0.9563 - 0.9814 | 0.9769 | 0.9563 - 0.9814 |
| `tcp_near(cube_a)` | 0.8697 | 0.8608 - 0.8752 | 0.8691 | 0.8557 - 0.8784 |
| `tcp_near(cube_b)` | 0.0000 | 0.0000 - 0.0000 | 0.0000 | 0.0000 - 0.0000 |

## C. Planning

Every combination was run from states measured with Stage 2's own `measured_state()`. Measured symbolic states equal the saved ones on all seeds: `gripper_open(), in_region(cube_a,pocket), in_region(cube_b,mouth)` for Stow, `drawer_closed(drawer)` for Drawer. Saved Stow scores match exactly. Saved Drawer scores differ on seeds 0 and 3 (largest 0.0506), in every mode, because the saved Stage 2 file predates the push change. Inverse goals are the same from all operators: Stow `clear_of_walls(cube_a)`, `gripper_open()`, `in_region(cube_a,src_a)`, `in_region(cube_b,src_b)`, not `in_region(cube_a,pocket)`, `in_region(cube_b,mouth)`; Drawer `drawer_open(drawer)`, not `drawer_closed(drawer)`.

"Same" means the same plan, hole positions, delegated literals, hole starts, cost, tied plans and complete `hole_spec()`, compared with the baseline. All found plans replay to the goal with real transitions.

| Scenario | `mean + relaxed` (baseline) | `min_max + relaxed` | `mean + exact` | `min_max + exact` |
|---|---|---|---|---|
| Stow: Frontier, each of seeds 0-4 | `pick(B) > place(B,src_b) > approach(A) > HOLE[clear_of_walls(A)] > pick(A) > place(A,src_a)`, cost (1, 0, 2, 1, 6) | same | same | same |
| Stow: No interference model | `pick(B) > place(B,src_b) > approach(A) > HOLE[clear_of_walls(A)] > pick(A) > place(A,src_a)`, cost (1, 0, 2, 1, 6) | same | same | same |
| Stow: Library only | no plan | same | same | same |
| Stow: Relearned | `pick(B) > place(B,src_b) > approach(A) > LEARNED[clear_of_walls(A)] > pick(A) > place(A,src_a)`, cost (0, 0, 0, 0, 6) | **no plan** | same | **no plan** |
| Drawer: Frontier, each of seeds 0-4 | `approach_handle > HOLE[handle_grasped] > pull_open`, cost (1, 0, 2, 1, 3) | same | same | same |
| Drawer: Library only | no plan | same | same | same |
| Drawer: Scripted grasp | `approach_handle > grasp_handle > pull_open`, cost (0, 0, 0, 0, 3) | same | same | same |
| Drawer: unrestricted (direct baseline) | `approach_handle > HOLE[drawer_open]`, cost (1, 0, 2, 1, 2) | same | relaxed only: exact refuses this option | relaxed only: exact refuses this option |

**The one difference comes from extraction, not planning.** `relearned` fails with `min_max` under both planners and succeeds with `mean` under both. The ordering Stage 4 takes from the tied plans for its `without_interference` condition is the same in all four combinations.

**The learning specification is the same in all four combinations, on all seeds, and each Stage 2 spec file is byte-identical to the saved one:**

| `hole_spec()` field | Stow | Drawer |
|---|---|---|
| `achieve` | `clear_of_walls(cube_a)` | `handle_grasped(drawer)` |
| `object` | `cube_a` | `drawer` |
| `start` | `tcp_near(cube_a)` | `at_handle_standoff(drawer)` |
| `preserve` | `in_region(cube_b,src_b)` | none |
| `preserve_absent` | `in_region(cube_b,mouth)` | none |
| `prefix` (library) | `pick(cube_b)`, `place(cube_b,src_b)`, `approach(cube_a)` | `approach_handle(drawer)` |
| `suffix` (library) | `pick(cube_a)`, `place(cube_a,src_a)` | `pull_open(drawer)` |
| Hole position | step 4 of 6 | step 2 of 3 |
| `cost` | (1, 0, 2, 1, 6) | (1, 0, 2, 1, 3) |

**The exact planner keeps its intended behaviour:** existing skills run before and after the hole on both tasks, and neither plan hit a search limit. The direct baseline's spec stays byte-identical to the saved one under both extractors, through the relaxed mode's `unrestricted_holes`. Exact mode refuses that option, so that baseline must keep naming `mode="relaxed"`.

`HoleContext`, which only the exact planner produces:

| `HoleContext` field | Stow | Drawer |
|---|---|---|
| `state_before` | `in_region(cube_a,pocket)`, `in_region(cube_b,src_b)`, `tcp_near(cube_a)` | `at_handle_standoff(drawer)`, `drawer_closed(drawer)` |
| `used_by` | `pick(cube_a)`, step 5 | `pull_open(drawer)`, step 3 |
| `waiting` | `pick(cube_a)`, `goal` | `pull_open(drawer)` |
| `library_alternatives` | `place(cube_a,src_a)`: cannot run here; `place(cube_a,src_b)`: cannot run here; `place(cube_a,mouth)`: cannot run here | none: no library action establishes the grasp |

**Differences that are diagnostics only.** They never reach RL and do not change the learned capability, but they would change what some scripts print and save (section F):

| Diagnostic (never reaches RL) | Stow, relaxed | Stow, exact | Drawer, relaxed | Drawer, exact |
|---|---|---|---|---|
| `capability_gaps` | `clear_of_walls(cube_a)`, `holding(cube_a)` | empty: exact computes no such set | `handle_grasped(drawer)` | empty |
| `candidates` | `clear_of_walls(cube_a)` | `clear_of_walls(cube_a)`, `gripper_open()`, `in_region(cube_a,src_a)`, `in_region(cube_b,src_b)` | `handle_grasped(drawer)` | `drawer_open(drawer)`, `handle_grasped(drawer)` |
| `expanded` | 44 | 95 | 4 | 4 |
| `limits_reached` | not reported | none | not reported | none |

## D. Execution with the original policies

No policy was retrained, and all weights are unmodified.

**Stow, seeds 1000-1009.** Means over the seeds. A and B are the cubes' final position errors. "Base" is `mean + relaxed`.

| Condition | Saved | `mean + relaxed` | `min_max + exact` | `min_max + exact`, learned operator by `mean` | Retries (base / proposed) | Rerolls (base / proposed) | Diverging steps (base / proposed) |
|---|---|---|---|---|---|---|---|
| Full inverse | 10/10, A 1.5 mm, B 0.8 mm | 10/10, A 1.5 mm, B 0.8 mm | 10/10, A 1.5 mm, B 0.8 mm | 10/10, A 1.5 mm, B 0.8 mm | 0 / 0 | 2 / 2 | 3 / 3 |
| Library only | 0/10, A 128.1 mm, B 0.8 mm | 0/10, A 128.1 mm, B 0.8 mm | 0/10, A 128.1 mm, B 0.8 mm | 0/10, A 128.1 mm, B 0.8 mm | 0 / 0 | 2 / 2 | 3 / 3 |
| Without interference model | 3/10, A 51.3 mm, B 89.4 mm | 3/10, A 51.3 mm, B 89.4 mm | 3/10, A 51.3 mm, B 89.4 mm | 3/10, A 51.3 mm, B 89.4 mm | 13 / 13 | 2 / 2 | 30 / 30 |
| Relearned | 9/10, A 11.6 mm, B 0.8 mm | 9/10, A 11.6 mm, B 0.8 mm | 0/10, A 131.3 mm, B 188.2 mm | 9/10, A 11.6 mm, B 0.8 mm | 0 / 0 | 2 / 2 | 3 / 0 |
| Stage 4's own check | PASS | PASS | **FAIL** | PASS |  |  |  |

The fourth result column, the "hybrid" run, uses `min_max` for the forward operator, `exact` planning and `mean` for the learned operator only. Its file has **0** values differing from the saved Stage 4 file.

`mean + relaxed` and `mean + exact` also have 0 values differing from the saved file. `min_max + exact` has 442, all in the learned operator, `passed` and the `relearned` condition. Its full, library-only and no-interference rows are identical value for value, including every retry, reroll and step check.

Per seed. The first ten result columns are the same in the saved file, `mean + relaxed` and `min_max + exact`. Library-only runs 3 of its 5 steps and stops at `pick(cube_a)`: its precondition is unmet.

| Seed | Rerolls | Full: goal | A mm | B mm | Retries | Library: goal | Steps ok | No interference: goal | A mm | B mm | Relearned, `mean`: goal | A mm | B mm | Relearned, `min_max`: goal | A mm | B mm |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 0 | yes | 1.1 | 0.3 | 0 | no | 3/5 | no | 124.3 | 148.3 | yes | 1.1 | 0.3 | no plan | 131.8 | 186.6 |
| 1001 | 1 | yes | 1.4 | 0.3 | 0 | no | 3/5 | yes | 8.1 | 3.9 | yes | 1.4 | 0.3 | no plan | 132.8 | 182.0 |
| 1002 | 0 | yes | 1.7 | 4.2 | 0 | no | 3/5 | no | 135.1 | 194.1 | no (3/6 steps) | 102.4 | 4.2 | no plan | 135.1 | 194.1 |
| 1003 | 0 | yes | 2.3 | 0.1 | 0 | no | 3/5 | no | 34.8 | 6.1 | yes | 2.3 | 0.1 | no plan | 134.6 | 185.5 |
| 1004 | 0 | yes | 1.6 | 0.9 | 0 | no | 3/5 | no | 72.5 | 141.9 | yes | 1.6 | 0.9 | no plan | 124.9 | 189.5 |
| 1005 | 0 | yes | 1.8 | 0.1 | 0 | no | 3/5 | no | 20.8 | 150.1 | yes | 1.8 | 0.1 | no plan | 127.2 | 190.7 |
| 1006 | 0 | yes | 0.6 | 0.9 | 0 | no | 3/5 | no | 45.2 | 181.2 | yes | 0.6 | 0.9 | no plan | 132.6 | 181.3 |
| 1007 | 1 | yes | 1.6 | 0.9 | 0 | no | 3/5 | no | 59.3 | 66.2 | yes | 1.6 | 0.9 | no plan | 130.6 | 189.8 |
| 1008 | 0 | yes | 1.5 | 0.2 | 0 | no | 3/5 | yes | 8.1 | 1.2 | yes | 1.5 | 0.2 | no plan | 133.1 | 192.0 |
| 1009 | 0 | yes | 1.8 | 0.3 | 0 | no | 3/5 | yes | 5.2 | 1.2 | yes | 1.8 | 0.3 | no plan | 130.2 | 190.4 |

With `min_max`, the relearned condition has no plan, so nothing runs: its cubes stay where the forward skill left them. Under `mean`, seed 1002 runs 3 of 6 steps and is refused at the learned step, because `in_region(cube_a,pocket)` is measured false.

**Drawer, seeds 2000-2019.** 40 values differ between `mean + relaxed` and `min_max + exact`: `capability_gaps` in the 20 library-only rows, which is empty under exact. The same 40 differ for `mean + exact`. Everything else, including `mean + relaxed` against the saved file, is identical value for value.

| Condition | Saved: goal, final opening | `mean + relaxed` | `min_max + exact` | Retries | Model vs world differences |
|---|---|---|---|---|---|
| learned, SAC seed 0 | 20/20, 146.5-147.2 mm | 20/20, 146.5-147.2 mm | 20/20, 146.5-147.2 mm | 0 | 0 |
| learned, SAC seed 1 | 20/20, 146.6-147.2 mm | 20/20, 146.6-147.2 mm | 20/20, 146.6-147.2 mm | 0 | 0 |
| learned, SAC seed 2 | 20/20, 147.1-147.6 mm | 20/20, 147.1-147.6 mm | 20/20, 147.1-147.6 mm | 0 | 0 |
| learned, SAC seed 3 | 20/20, 146.7-147.2 mm | 20/20, 146.7-147.2 mm | 20/20, 146.7-147.2 mm | 0 | 0 |
| library_only | 0/20, 0.7-4.6 mm | 0/20, 0.7-4.6 mm | 0/20, 0.7-4.6 mm | 0 | 0 |
| scripted_grasp | 20/20, 147.2-147.4 mm | 20/20, 147.2-147.4 mm | 20/20, 147.2-147.4 mm | 0 | 0 |

Per seed, the same in the saved file, `mean + relaxed` and `min_max + exact`. Final opening, with inverse control steps in brackets. Library-only runs `approach_handle`, then stops at `pull_open` because `handle_grasped(drawer)` is unmet. There were no retries, no failed checks and no model-versus-world differences in the policy runs.

| Seed | SAC seed 0: mm (steps) | SAC seed 1: mm (steps) | SAC seed 2: mm (steps) | SAC seed 3: mm (steps) | Library only: mm | Steps ok | Scripted grasp: mm |
|---|---|---|---|---|---|---|---|
| 2000 | 147.1 (43) | 147.1 (42) | 147.2 (46) | 147.2 (43) | 2.3 | 1/2 | 147.3 |
| 2001 | 147.1 (43) | 146.8 (42) | 147.3 (46) | 146.7 (43) | 3.0 | 1/2 | 147.4 |
| 2002 | 146.7 (42) | 147.2 (42) | 147.1 (46) | 146.8 (43) | 1.0 | 1/2 | 147.2 |
| 2003 | 146.6 (42) | 147.2 (42) | 147.2 (46) | 146.9 (43) | 1.8 | 1/2 | 147.2 |
| 2004 | 147.1 (43) | 147.1 (42) | 147.1 (46) | 146.8 (43) | 2.2 | 1/2 | 147.3 |
| 2005 | 147.2 (43) | 147.2 (42) | 147.1 (46) | 146.8 (43) | 0.7 | 1/2 | 147.2 |
| 2006 | 147.2 (43) | 147.2 (42) | 147.1 (46) | 146.8 (43) | 1.4 | 1/2 | 147.2 |
| 2007 | 147.1 (43) | 147.2 (42) | 147.2 (46) | 146.8 (43) | 1.7 | 1/2 | 147.3 |
| 2008 | 147.2 (43) | 147.2 (42) | 147.2 (46) | 146.8 (43) | 1.5 | 1/2 | 147.2 |
| 2009 | 147.2 (43) | 147.2 (42) | 147.2 (46) | 146.8 (43) | 1.4 | 1/2 | 147.2 |
| 2010 | 146.6 (42) | 147.2 (42) | 147.2 (46) | 146.8 (43) | 1.6 | 1/2 | 147.2 |
| 2011 | 146.9 (42) | 146.8 (41) | 147.6 (46) | 146.8 (43) | 4.6 | 1/2 | 147.4 |
| 2012 | 146.5 (42) | 147.2 (42) | 147.2 (46) | 146.8 (43) | 1.9 | 1/2 | 147.3 |
| 2013 | 147.1 (43) | 147.1 (42) | 147.2 (46) | 146.8 (43) | 1.8 | 1/2 | 147.2 |
| 2014 | 147.1 (43) | 147.2 (42) | 147.2 (46) | 146.9 (43) | 1.6 | 1/2 | 147.2 |
| 2015 | 146.7 (42) | 146.6 (41) | 147.4 (46) | 146.9 (43) | 3.3 | 1/2 | 147.3 |
| 2016 | 146.8 (42) | 146.7 (41) | 147.4 (46) | 146.9 (43) | 3.5 | 1/2 | 147.4 |
| 2017 | 146.5 (43) | 147.2 (42) | 147.2 (46) | 146.8 (43) | 1.7 | 1/2 | 147.3 |
| 2018 | 146.8 (42) | 146.8 (41) | 147.5 (46) | 146.8 (43) | 4.4 | 1/2 | 147.4 |
| 2019 | 147.1 (43) | 147.2 (42) | 147.2 (46) | 146.8 (43) | 1.6 | 1/2 | 147.3 |

**Handoffs.** At every learned step, the policy's first observation and a hash of the whole simulator state were logged and compared with `mean + relaxed`. Drawer: 80 of 80 identical. Stow: 38 of 38 identical in the hybrid run, and all 29 that `min_max + exact` reaches are identical (the 9 relearned ones never run).

**Stage 3 input.** Stage 3 only reads the spec file, which is byte-identical. Hole environments built from the `min_max + exact` spec fit every checkpoint: 4 observations and 2 actions on Stow, 7 and 4 on Drawer. Their handoffs are valid on the seeds tried (Stow 1000 and 1002, Drawer 1000 and 1001). Details are in `phase3/*_stage3_input.json`. Nothing was trained.

**Final evaluation, seeds 3000-3049: not rerun, because it is not affected.** `scripts/drawer_final_evaluate.py` imports neither the planner nor the extractor. It reads only the goal from `artifacts/drawer/stage1_operator.json`, and that goal is the same under both extractors and equals the one in its saved results. Its direct policies' spec is unchanged too (section C).

## E. Regression tests

| Run | Command | Result |
|---|---|---|
| Unmodified suite, defaults unchanged | `PYTHONNOUSERSITE=1 python -m pytest tests -q` | **202 passed**, 0 failed |
| Extraction default flipped in memory (not in the repository) | same, with a scratch plugin | 1 failed, 201 passed |
| Both defaults flipped in memory (not in the repository) | same, with scratch plugins | 30 failed, 172 passed |

The in-memory runs only measure what a migration would touch. The 30 failures:

- 1 asserts that `mean` is the default: `test_the_mean_mode_is_the_default_and_unchanged`.
- 13 pin relaxed behaviour by relying on the default.
- 6 hit the `ValueError` for `unrestricted_holes`.
- 5 read `capability_gaps`, which exact leaves empty.
- 4 compare with a cost-only enumeration, which exact's causal preference is not.
- 1 reads `candidates`.

None of them shows an invalid plan. **The suite does not catch the failures that matter here:** it passes the learned-operator failure, and both Stage 1 scripts crash under a flipped extraction default (`KeyError: 'start_mean'` at `stow_stage1_extract.py:60` and `drawer_stage1_extract.py:70`), confirmed by running them.

The legacy PlanRob bundle is safe under a flipped extraction default: only `metadata.extractor` changes, with 0 numeric differences in 494 values.

Phase 5's reruns with new defaults were not done, because the defaults were not changed.

## F. Migration status

**Defaults were not changed.** No algorithm, threshold, reward, hyperparameter, checkpoint or saved result was modified.

Files added or changed, all uncommitted:

- `scripts/validate_modes_extraction.py`, `scripts/validate_modes_planning.py`, `scripts/validate_modes_execution.py`: new.
- `artifacts/default_mode_validation/`: new, all results and this report.
- `.gitignore`: an allow-line and its comment, so these results can be committed.

**What is needed to finish.** The learned operator needs a decision only you can make, because each way out changes either the method or the paper:

| Option | What changes | Evidence so far | Cost |
|---|---|---|---|
| **1. `min_max` everywhere, and a learned-skill run counts only if its measured state at the handoff matches the plan's prediction** | A few lines in `learn_operator()`, which already skips runs whose prefix failed. 9 of the 10 runs count. | `min_max` on those 9 runs gives the saved action exactly, so the plan and execution are those of the hybrid run. The gated run itself still needs executing. | A protocol rule added after seeing this result, which must be disclosed in the paper. |
| **2. `min_max` for forward demonstrations, `mean` for the learned operator** | One explicit `mode="mean"` at `stow_stage4_execute.py:87`. | Fully run: the hybrid column above, 0 values differing from the saved file. | The paper must state and justify two rules. This departs from your request to apply `min_max` in Stage 4. |
| **3. `mean` stays the method, `exact` becomes the default** | Planner migration only. | Fully run on both tasks: `mean + exact` gives the same plans and specs (section C) and the same Stage 4 results (section D). | The Method section goes back to `mean`. `min_max` agreement on forward operators can be reported as a robustness check. |

**Recommendation: option 1, if the Method section is to keep saying `min_max` throughout.** A run that did not begin in the operator's modelled state is not evidence about that operator, and Stage 4 already applies the same idea when it re-rolls a forward skill that fell short. It changes no number in the results, because the action comes out the same. It is still a rule added after seeing the data, so it must be stated in the paper. If you would rather add no rule, option 3 is the cleanest.

Whichever you choose, a migration would also need these changes, none of which is applied:

- **Stage 1 scripts:** the score table reads `start_mean`, which `min_max` does not produce (`stow_stage1_extract.py:60`, `drawer_stage1_extract.py:70`, and `scripts/v0/planrob_pushcube_symbolic_pipeline.py:304`).
- **Relaxed ablations:** add `mode="relaxed"` at `drawer_stage2_plan.py:103` and `drawer_direct_spec.py:51,64`.
- **Comparison scripts:** `compare_hole_planners_random.py:101` relies on the default for its relaxed column, so after a flip it would silently compare exact with itself. `stow_stage1_extractor_comparison.py:78` relies on it for its mean column, and would stop with an error when it reads `start_mean`.
- **Legacy PlanRob scripts:** the 9 extraction calls in `scripts/v0/` should name `mode="mean"`, to keep those published experiments reproducible.
- **Gap reports:** `stow_stage2_plan.py:101-103` would print empty gaps and "lacking a soft score: none", although `holding(cube_a)` has no score. `drawer_stage2_plan.py:92-93,110` and `drawer_stage4_execute.py:128` would print or save empty gaps.
- **Tests and docs:** the 30 tests above, the planner and extractor docstrings, and the README.
- **Output paths:** Stage 1, 2 and 4 scripts write to the saved artifact paths by default, so running them after a migration without `--out` would overwrite the historical results.

**Remaining limitations of this validation:**

- Every Stage 4 run reuses the stored seeds. This shows the behaviour is unchanged, not that it generalizes further.
- The stored data contains one symbolic start per task. The earlier audit swept all 969 consistent hypothetical starts for the planner modes only.
- `min_max` is sensitive to any single bad demonstration. Forward agreement holds on seeds 0-4, but Stow's forward push falls short and is re-rolled on 2 of the 10 Stage 4 seeds, and Stage 1 has no such re-roll. A different demonstration set could break forward agreement the same way.

## Verdict

**NOT VERIFIED — DEFAULTS UNCHANGED**
