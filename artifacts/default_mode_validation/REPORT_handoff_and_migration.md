# Handoff validation and migration to `min_max` and `exact`

**Result: the learned Stow operator now extracts correctly under `min_max`, because each demonstration is checked before the policy runs. Stage 4 passes with `min_max + exact`, and every Stage 4 value matches the saved results. `min_max` and `exact` are now the defaults; `mean` and `relaxed` stay available by naming them.**

This follows [REPORT.md](REPORT.md), the first validation, which stopped because of seed 2004. That report and its results are kept as they were.

|  | Stow, `mean + relaxed` (saved) | Stow, `min_max + exact` (now the default) | Drawer, `mean + relaxed` (saved) | Drawer, `min_max + exact` (now the default) |
|---|---|---|---|---|
| Forward operator and inverse goal | as saved | same | as saved | same |
| Learning specification | as saved | byte-identical | as saved | byte-identical |
| Learned operator: Pre, Add, Del | as saved | same | not used | not used |
| Replanning with it | plan found | same plan | not used | not used |
| Full inverse, original policies | 10/10 | same, value for value | 20/20 for each of 4 policies | same, value for value |
| Baselines | library 0/10, no interference 3/10 | same | library 0/20, scripted grasp 20/20 | same |
| Relearned | 9/10 | same, value for value | not used | not used |
| Stage 4's own check | PASS | PASS | no such check | no such check |
| Raw handoff validity on the evaluation seeds | not measured then | 9 of 10 | not measured then | 20 of 20 |

## 1. Why the prefix reported success at seed 2004

**Physically, the forward skill was fine; the prefix moved cube A.** Lifting B out of the mouth drags A with it. No step of the plan is about A at that point, so nothing noticed.

Seed 2004, one step at a time:

| Step | Executive: ok | Executive measured | A's x, mm | `pocket` | `clear_of_walls` | Differs from the plan's prediction |
|---|---|---|---|---|---|---|
| `after the forward skill` | not a step |  | 130.0 | 0.9710 | 0.1194 | none |
| `pick(cube_b)` | yes | cube z 107mm | 115.1 | 0.7340 | 0.3758 | `in_region(cube_a,pocket)` |
| `place(cube_b,src_b)` | yes | in_region 0.96 | 102.6 | 0.1841 | 0.6779 | `in_region(cube_a,pocket)` |
| `approach(cube_a)` | yes | seated | 113.4 | 0.6656 | 0.4148 | `in_region(cube_a,pocket)` |

An ordinary seed, 2003, for comparison:

| Step | Executive: ok | Executive measured | A's x, mm | `pocket` | `clear_of_walls` | Differs from the plan's prediction |
|---|---|---|---|---|---|---|
| `after the forward skill` | not a step |  | 127.9 | 0.9730 | 0.1428 | none |
| `pick(cube_b)` | yes | cube z 114mm | 127.9 | 0.9730 | 0.1428 | none |
| `place(cube_b,src_b)` | yes | in_region 0.98 | 127.9 | 0.9730 | 0.1428 | none |
| `approach(cube_a)` | yes | seated | 130.0 | 0.9668 | 0.1196 | none |

**What the executive checks** ([stow_executive.py](../../src/inverse_skills/envs/stow_executive.py)):

- **Before each step:** the step's own preconditions, on scored literals only. No prefix step has one: `pick` and `approach` need only the unscored `holding` absent, and `place` needs the unscored `holding`.
- **After each step:** one physical promise of that step: B's height after `pick(cube_b)`, B's region after `place(cube_b,src_b)`, the hand seated on A after `approach(cube_a)`.

**What it does not verify:**

- **Anything a step should leave alone.** A's position is no step's promise, so the drag is not a failure of any step.
- **The plan's prediction,** unless it is given the domain. It then reports differences but still does not fail, and `learn_operator` did not give it the domain. The last column above shows it would have flagged the problem from the first step.
- **The hole's real pre-state.** The hole's own start check, also the one training uses, asks only for `tcp_near(cube_a)`, which stays true.

Evaluation seed 1002 shows the same pattern, and seed 1000 shows none.

**So this is a physical-execution effect the symbolic model does not contain.** `min_max` did not cause it; it only refused to average it away.

## 2. The handoff check

[operators/handoff.py](../../src/inverse_skills/operators/handoff.py) names no predicate of any task. It reads what must hold off the plan and the domain's world facts, before anything about the skill's execution is known, using the symbolic state's existing threshold (true at a score of 0.8 or more).

| Condition | Why | Stow | Drawer |
|---|---|---|---|
| The hole's start | What the skill is told it starts from | `tcp_near(cube_a)` true | `at_handle_standoff(drawer)` true |
| The hole's modelled transition | A transition begins in its own pre-state: what it removes must hold, what it establishes must not | `in_region(cube_a,pocket)` true, `clear_of_walls(cube_a)` false | `at_handle_standoff(drawer)` true, `handle_grasped(drawer)` false |
| The fences | What the plan relies on across the hole | `in_region(cube_b,src_b)` true, `in_region(cube_b,mouth)` false | none |
| **Not asked for** | The plan does not rely on them | `gripper_open()`, `tcp_near(cube_b)`, `in_region(cube_a,src_a)` | `drawer_closed(drawer)`, `drawer_open(drawer)` |

`in_region(cube_a,pocket)` is required only because the domain's mutex says the hole removes it. Nothing refers to seed 2004.

**The check never looks at what the skill does.** It runs before the policy, and a run that passes is kept whatever comes of it. A test pins this: a valid handoff from which the policy fails stays in the set, and `min_max` then reports the lost effect.

Requiring the whole predicted state instead would flag the same single run among those recorded (`phase1/stow_learned.json`), so the result does not depend on where this line was drawn.

## 3. Bounded retries, recorded

A scene that fails is replaced by `seed + k * 10000`, for at most 5 scenes per demonstration. Those are the existing conventions of `stowed()` and `StowHoleEnv`. Running out raises `HandoffsExhausted`, which names every scene and reason. The old loop's silent `continue` is gone.

The two scenes replaced in this collection:

| Seed | Scene | Failed at | Recorded reason |
|---|---|---|---|
| 2001 | 2001 | forward | differs from the planned start: in_region(cube_a,pocket), in_region(cube_b,mouth) |
| 2004 | 2004 | handoff | in_region(cube_a,pocket) should hold, score 0.6656 |

The first of these already happened before, inside `stowed()`, without a record. Now it is recorded.

## 4. Ten valid demonstrations

Nine of the ten are the original runs, score for score. Only 2004 is replaced. The policy never ran from scene 2004.

| Scene | `clear_of_walls` start | end | `pocket` start | end | `tcp_near(A)` start | `src_b` start | `mouth` start | Against the run before the check |
|---|---|---|---|---|---|---|---|---|
| 2000 | 0.1201 | 0.8808 | 0.9549 | 0.0180 | 0.8752 | 0.9789 | 0.0000 | same as before |
| 12001 | 0.1198 | 0.8695 | 0.9579 | 0.0220 | 0.8608 | 0.9811 | 0.0000 | same as before |
| 2002 | 0.1199 | 0.8728 | 0.9719 | 0.0208 | 0.8731 | 0.9762 | 0.0000 | same as before |
| 2003 | 0.1196 | 0.8796 | 0.9668 | 0.0184 | 0.8735 | 0.9814 | 0.0000 | same as before |
| 12004 | 0.1201 | 0.8695 | 0.9708 | 0.0220 | 0.8790 | 0.9813 | 0.0000 | **new**, replaces 2004 |
| 2005 | 0.1200 | 0.8750 | 0.9672 | 0.0200 | 0.8742 | 0.9752 | 0.0000 | same as before |
| 2006 | 0.1196 | 0.8311 | 0.9738 | 0.0397 | 0.8628 | 0.9809 | 0.0000 | same as before |
| 2007 | 0.1199 | 0.8772 | 0.9639 | 0.0192 | 0.8737 | 0.9765 | 0.0000 | same as before |
| 2008 | 0.1194 | 0.8793 | 0.9643 | 0.0185 | 0.8678 | 0.9813 | 0.0000 | same as before |
| 2009 | 0.1196 | 0.8738 | 0.9756 | 0.0204 | 0.8653 | 0.9812 | 0.0000 | same as before |

## 5. Both extractors on exactly those demonstrations

**`min_max` recovers the expected operator, including `clear_of_walls(cube_a)`.** Both modes give the saved action exactly, with no threshold changed.

|  | Before the check, `min_max` | With the check, `min_max` | With the check, `mean` | Saved |
|---|---|---|---|---|
| Pre | `in_region(cube_b,src_b)`, `tcp_near(cube_a)` | `in_region(cube_a,pocket)`, `in_region(cube_b,src_b)`, `tcp_near(cube_a)` | `in_region(cube_a,pocket)`, `in_region(cube_b,src_b)`, `tcp_near(cube_a)` | `in_region(cube_a,pocket)`, `in_region(cube_b,src_b)`, `tcp_near(cube_a)` |
| Add | **none** | `clear_of_walls(cube_a)` | `clear_of_walls(cube_a)` | `clear_of_walls(cube_a)` |
| Del | **none** | `in_region(cube_a,pocket)` | `in_region(cube_a,pocket)` | `in_region(cube_a,pocket)` |

Weights differ slightly, and no stage reads them:

| Term | Predicate | `mean` weight | `min_max` weight | Difference |
|---|---|---|---|---|
| Pre | `in_region(cube_a,pocket)` | 0.9667 | 0.9549 | -0.0118 |
| Pre | `in_region(cube_b,src_b)` | 0.9794 | 0.9752 | -0.0042 |
| Pre | `tcp_near(cube_a)` | 0.8706 | 0.8608 | -0.0097 |
| Add | `clear_of_walls(cube_a)` | 0.7511 | 0.7109 | -0.0401 |
| Del | `in_region(cube_a,pocket)` | 0.9448 | 0.9153 | -0.0296 |

The tightest margin is `clear_of_walls` at the end of a run: 0.8311 against the 0.80 line, margin +0.0311. That is close by construction, because the executive stops the policy as soon as the score reaches 0.8.

**Replanning works.** Library only, in all four combinations of extractor and planner, gives the same plan: `pick(cube_b) > place(cube_b,src_b) > approach(cube_a) > LEARNED[clear_of_walls(cube_a)] > pick(cube_a) > place(cube_a,src_a)`.

## 6. Stage 4, with the raw rate reported apart

Stage 4 was rerun on its original seeds and checkpoint, writing to new files. **Results are identical to the saved file: 0 values differ, for both `mean + relaxed` and `min_max + exact`,** apart from the new section that reports handoff validity. The first validation's 442 differences are gone. Between the two modes, the policy's first observation and the full simulator state match at all 38 learned steps.

| Condition | Saved | `mean + relaxed`, with the check | `min_max + exact`, with the check | Retries | Rerolls | Diverging steps |
|---|---|---|---|---|---|---|
| Full inverse | 10/10, A 1.5 mm, B 0.8 mm | 10/10, A 1.5 mm, B 0.8 mm | 10/10, A 1.5 mm, B 0.8 mm | 0 | 2 | 3 |
| Library only | 0/10, A 128.1 mm, B 0.8 mm | 0/10, A 128.1 mm, B 0.8 mm | 0/10, A 128.1 mm, B 0.8 mm | 0 | 2 | 3 |
| Without interference model | 3/10, A 51.3 mm, B 89.4 mm | 3/10, A 51.3 mm, B 89.4 mm | 3/10, A 51.3 mm, B 89.4 mm | 13 | 2 | 30 |
| Relearned | 9/10, A 11.6 mm, B 0.8 mm | 9/10, A 11.6 mm, B 0.8 mm | 9/10, A 11.6 mm, B 0.8 mm | 0 | 2 | 3 |
| Stage 4's own check | PASS | PASS | PASS |  |  |  |

**Raw handoff validity is a separate number.** It says how often the physical scene matches the plan at the handoff, before any retry and whatever the policy then does.

| Scenes | Valid handoffs | Invalid: which, and why | Forward fell short first |
|---|---|---|---|
| Stow, learned-operator demonstrations, seeds 2000-2009 | 10 of 11 (91%) | scene 2004: `pocket` 0.6656 | 1 of 12 scenes (2001) |
| Stow, evaluation seeds 1000-1009 | 9 of 10 (90%) | seed 1002: in_region(cube_a,pocket) should hold, score 0.0710 | 2 of 10 seeds |
| Stow, training seeds 0-31 (measured only) | 32 of 32 (100%) | none | 0 of 32 seeds |
| Drawer, Stage 4 seeds 2000-2019 (measured only) | 20 of 20 (100%) | none | not applicable |

Across the Stow scenes that reached a handoff, that is 51 of 53 (96%). It is a property of the physical prefix, not of the extractor or the policy.

Per evaluation seed. The evaluation is **not** re-rolled for the check, so seed 1002 keeps its place: its handoff is invalid, the full plan still succeeds, and the relearned step is refused because its precondition is measured false.

| Seed | Valid handoff | Rerolls | Full: goal | A mm | B mm | Relearned: goal | A mm | B mm |
|---|---|---|---|---|---|---|---|---|
| 1000 | yes | 0 | yes | 1.1 | 0.3 | yes | 1.1 | 0.3 |
| 1001 | yes | 1 | yes | 1.4 | 0.3 | yes | 1.4 | 0.3 |
| 1002 | **no** | 0 | yes | 1.7 | 4.2 | no, refused at the learned step | 102.4 | 4.2 |
| 1003 | yes | 0 | yes | 2.3 | 0.1 | yes | 2.3 | 0.1 |
| 1004 | yes | 0 | yes | 1.6 | 0.9 | yes | 1.6 | 0.9 |
| 1005 | yes | 0 | yes | 1.8 | 0.1 | yes | 1.8 | 0.1 |
| 1006 | yes | 0 | yes | 0.6 | 0.9 | yes | 0.6 | 0.9 |
| 1007 | yes | 1 | yes | 1.6 | 0.9 | yes | 1.6 | 0.9 |
| 1008 | yes | 0 | yes | 1.5 | 0.2 | yes | 1.5 | 0.2 |
| 1009 | yes | 0 | yes | 1.8 | 0.3 | yes | 1.8 | 0.3 |

## 7. Regression tests

`tests/test_handoff_validation.py` adds 16 tests, with no simulator, using the scores recorded on the real pipeline:

- **Invalid handoffs:** seed 2004 is invalid for exactly `in_region(cube_a,pocket)`, and every other recorded handoff is valid. Unrelated predicates are not asked for, and the threshold is the symbolic one.
- **Bounded retries:** the next scene is used, the skill never runs from an invalid handoff, running out is an error naming every scene, forward and prefix failures are recorded, and the raw rate is computed apart.
- **The original failure:** without the check, `min_max` loses the add effect and replanning finds no plan under either planner. With it, `min_max` recovers the operator and exact replanning works. A run is never selected by its outcome.
- **Generality:** Stow and Drawer conditions come from the plan, and the module names no domain vocabulary.

Breaking the handoff code in 15 different ways was caught by these tests 15 times out of 15.

## 8. The migration

**Defaults:** `OperatorExtractionConfig.mode` is `"min_max"` and `plan(mode=...)` is `"exact"`. Neither the extraction rules nor the planner's search changed, and `mean` and `relaxed` are selectable by naming them.

| Change | Files |
|---|---|
| The two defaults, and the docstrings that described them | `operators/extractor.py`, `operators/hole_planner.py`, `operators/exact_hole_planner.py` |
| The handoff check (new) | `operators/handoff.py`, `scripts/stow_stage4_execute.py` |
| Stage 1 crash fixed: the score table reads what `min_max` produces | `scripts/stow_stage1_extract.py`, `scripts/drawer_stage1_extract.py` |
| Reports use exact's own fields, not the gap fields it leaves empty | `scripts/stow_stage2_plan.py`, `scripts/drawer_stage2_plan.py`, `scripts/drawer_stage4_execute.py` |
| The direct baseline names `mode="relaxed"`, so its meaning is unchanged | `scripts/drawer_stage2_plan.py`, `scripts/drawer_direct_spec.py` |
| Comparisons name both modes | `scripts/compare_hole_planners_random.py`, `scripts/stow_stage1_extractor_comparison.py` |
| The workshop paper's experiments stay on `mean` | `src/inverse_skills/planrob_bundle.py`, 8 scripts in `scripts/v0/` |
| Saved results cannot be overwritten by rerunning with the new defaults: a stage stops if its output exists | Stage 1, 2 and 4 scripts of both tasks, `scripts/drawer_direct_spec.py`, `scripts/stow_planner_demo.py` |
| Tests: legacy behaviour names its mode, new tests pin the defaults | `tests/test_hole_planner.py`, `tests/test_hole_planner_verification.py`, `tests/test_exact_hole_planner.py`, `tests/test_operator_extraction.py` |
| Docs | `README.md` |

**Verified after the migration: 55 of 55 checks pass** (`after_migration/verification.json`). The plain scripts ran with no wrapper, using only the new defaults:

| What | Result |
|---|---|
| Stage 1, Stow and Drawer | Reproduce `min_max` named: operator, weights and scores. Own checks pass. Goals equal the saved ones. |
| Stage 2, Stow and Drawer, each seed | Reproduce `min_max + exact` named, including hole contexts. Spec files byte-identical to the saved ones. |
| Direct baseline | Spec and plan file identical to the saved ones. The unrestricted ablation keeps its plan, cost and gaps. |
| Stage 3 input | Spec byte-identical. Environments built from it fit every checkpoint, and handoffs are valid. |
| Stage 4, Stow | Equals `min_max + exact` named and the saved results, value for value. Own check passes. |
| Stage 4, Drawer | Equals them too, apart from one report field: `unmet_needs` replaces the `capability_gaps` exact leaves empty. |
| Comparison scripts | Still compare two modes: all 26 scenarios and 40,000 random problems match the saved results, apart from run times. |
| The code itself | Apart from docstrings, the extractor and the relaxed planner differ from `c68d830` only in the default value, and the exact planner not at all. |
| Named-mode scripts | Unchanged by the new defaults, and the original failure still reproduces with the check bypassed. |
| Video scripts (Stage 2, Stage 4, framework, Drawer Stage 4) | Run to completion under the new defaults, writing elsewhere. Not part of the 55 checks. |
| Test suite | 221 passed, 0 failed (was 202 before this work). |

The final evaluation on seeds 3000-3049 was not rerun. It uses neither the planner nor the extractor, and its goal is unchanged.

## One saved file was overwritten

While smoke-testing legacy scripts under the new defaults, I ran `scripts/v0/smoke_operator_extraction.py` without first checking where it writes, and **it overwrote `artifacts/smoke_operator.json`**. That file was git-ignored and has no other copy, so the original is lost. I cannot show the new one is byte-identical to it. Nothing reads this file: it is a toy smoke test, its only writer is that script, and no experiment or result here depends on it. The same run of `scripts/v0/planrob_toy_demo.py` created `artifacts/planrob_toy_demo.json`, which did not exist before, and I removed it. No other file under `artifacts/` outside this folder changed, and every tracked artifact is unchanged.

## Limitations

- **The check was added after seeing this failure, so the paper must state it.** A suggested sentence: *"An execution of a learned skill counted as a demonstration only if the scene measured at the handoff agreed with the literals its plan relies on; one of eleven scenes failed this check and was replaced."* The rule is derived from the plan and looks only at the state before the skill, but it is still a procedural choice.
- **The evaluation seeds include one invalid handoff.** The 10/10 for the full inverse includes seed 1002. Report the 9-of-10 raw rate beside it.
- **Training is unchanged.** The hole environment's own check asks only for the start, and the policy was not retrained. Measured today, all 32 training seeds pass the new check, so its training scenes are valid by it.
- **A correction to the first report.** It said the jam happens in "about one run in ten". That came from two small samples. With the 32 training seeds, it is 2 of the 53 Stow scenes that reached a handoff (4%): scenes 2004 and 1002 are the only two seen. The chance of meeting one in a set of ten is then roughly 30%, on very few events.
- **The check and `min_max` draw "false" differently.** The check treats a literal as false below 0.8, as the planner and executive do; `min_max` needs 0.2 or less. A drag of about 6 to 13 mm would leave `pocket` true and `clear_of_walls` between 0.2 and about 0.33. Such a run would pass the check, and `min_max` would then lose the add effect. No recorded scene falls in that window (across the 51 valid Stow handoffs, `clear_of_walls` never exceeds 0.1207). If one does, Stage 4's own check fails loudly rather than silently. Tightening the check to 0.2 would tie it to the extractor, so I did not.
- **`min_max` stays sensitive to any single bad demonstration.** Stage 1 has no re-roll, and Stow's forward push fell short on 2 of the 10 evaluation seeds and 0 of the 32 training seeds. Forward agreement holds on the recorded seeds 0-4.
- **The default input files are still the saved `mean` operators.** Stages 2 and 4 read `artifacts/*/stage1_operator.json` unless told otherwise. Their predicate sets and goals equal the `min_max` ones, so nothing changes. The `min_max` operators are in `after_migration/`. I did not replace the saved files.
- **Determinism is on this machine.** All of this was generated twice from nothing with identical results (52 files, apart from run times), and the saved results reproduce exactly here. Every run reuses stored seeds, in simulation only.

## Reproducing

```bash
export PYTHONNOUSERSITE=1
H=artifacts/default_mode_validation/handoff
A=artifacts/default_mode_validation/after_migration
python scripts/validate_handoff.py investigate
python scripts/validate_handoff.py collect
python scripts/validate_handoff.py drawer
python scripts/validate_handoff.py survey
python scripts/validate_modes_execution.py run stow --operator-from mean --learned-extraction mean --planner relaxed --out-dir $H/stage4
python scripts/validate_modes_execution.py run stow --operator-from min_max --learned-extraction min_max --planner exact --out-dir $H/stage4
python scripts/validate_modes_execution.py compare --out-dir $H/stage4
# The plain scripts, with the new defaults and no wrapper:
python scripts/stow_stage1_extract.py --out $A/stow_stage1_operator.json
python scripts/drawer_stage1_extract.py --out $A/drawer_stage1_operator.json
python scripts/stow_stage2_plan.py --operator $A/stow_stage1_operator.json --out $A/stow_stage2_plan.json --spec-out $A/stow_hole_spec.json
python scripts/drawer_stage2_plan.py --operator $A/drawer_stage1_operator.json --out $A/drawer_stage2_plan.json --spec-out $A/drawer_hole_spec.json
python scripts/drawer_direct_spec.py --operator $A/drawer_stage1_operator.json --grasp-spec $A/drawer_hole_spec.json --out-dir $A/direct
python scripts/validate_modes_execution.py stage3-input stow --spec $A/stow_hole_spec.json --out-dir $A
python scripts/validate_modes_execution.py stage3-input drawer --spec $A/drawer_hole_spec.json --out-dir $A
python scripts/stow_stage4_execute.py artifacts/stow/stage3/1790788534/sac_hole_best --operator $A/stow_stage1_operator.json --spec $A/stow_hole_spec.json --out $A/stow_stage4.json --learned-out $A/stow_learned_operator.json
python scripts/drawer_stage4_execute.py --operator $A/drawer_stage1_operator.json --spec $A/drawer_hole_spec.json --out $A/drawer_stage4.json
python scripts/stow_planner_demo.py --operator $A/stow_stage1_operator.json --out $A/stow_planner_demo.json
python scripts/compare_hole_planners.py --out $A/planner_comparison_scenarios.json
python scripts/compare_hole_planners_random.py --out $A/planner_comparison_random.json
python scripts/validate_handoff.py earlier-comparison
python scripts/validate_modes_extraction.py stow --out-dir $A/phase1_named
python scripts/validate_modes_extraction.py drawer-original --out-dir $A/phase1_named
python scripts/validate_modes_extraction.py drawer-current --out-dir $A/phase1_named
python scripts/validate_modes_extraction.py stow-learned --out-dir $A/phase1_named
python scripts/validate_modes_planning.py stow --out-dir $A/phase2_named
python scripts/validate_modes_planning.py drawer --out-dir $A/phase2_named
python scripts/validate_handoff.py after-migration
```

Commit `c68d830` plus the uncommitted changes above. Python 3.10.21, torch 2.7.1+cu128, stable-baselines3 2.8.0, gymnasium 1.2.3, mani_skill 3.0.1, sapien 3.0.3, numpy 2.2.6, NVIDIA RTX 5070 Ti, driver 595.99.02. Nothing was retrained, and no reward, threshold or hyperparameter changed.

## Verdict

**VERIFIED — DEFAULTS UPDATED**
