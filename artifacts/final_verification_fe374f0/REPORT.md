# Final verification of the minimal hole planner, `fe374f0`

2026-11-09. **Result: the implementation matches the agreed Minimal Hole Planner v2 formulation, all 239 tests pass, and fresh simulation runs reproduce every saved Stow and Drawer result.** The policies were loaded unchanged and nothing was retrained. The reproduced results are the holes, the learning specifications, the handoff conditions and the success counts. No implementation problem was found. The differences found are reporting fields, one tie listing and the forward operators' extraction weights; none changes a plan, a hole, a specification or an outcome.

Every run used the minimal planner at `fe374f0`. The saved results served only as comparison references. Each stage's inputs were the fresh outputs of the stage before it.

## 1. Commit and environment

| | |
|---|---|
| Branch and commit | `feature/exact-hole-planner`, `fe374f0037b24071f8381b8a93ad92a21f11564d`. No tracked file changed, before or after. |
| Remote | The local tracking ref `origin/feature/exact-hole-planner` is `fe374f0`. GitHub cannot be reached from the container (no ssh), so the push and the tag `before-minimal-planner` were not checked on origin from here. |
| Planner files imported | `src/inverse_skills/operators/hole_planner.py` (blob `5c119da`) and `exact_hole_planner.py` (blob `c94a795`). Every run checked that these are the files committed at `fe374f0`, imported from this checkout. |
| Machine | RTX 5070 Ti Laptop GPU, driver 595.99.02, Linux 7.0.0-42 |
| Software | Python 3.10.21, numpy 2.2.6, torch 2.7.1+cu128, mani-skill 3.0.1, sapien 3.0.3, stable-baselines3 2.8.0, gymnasium 1.2.3, run with `PYTHONNOUSERSITE=1` |

## 2. Tests

`python -m pytest -q`: **239 passed** in 5.5 s. The 2 warnings are SAPIEN's harmless Vulkan messages. The suite includes:

- **118 tests** in `test_exact_hole_planner.py`, checked against an oracle written from the formulation that shares no code with the planner.
- **16 tests** in `test_minimal_formulation.py`, covering:
  - the learning-budget counterexample;
  - NoPlan for budgets 0 to 3;
  - subset-minimality;
  - the upstream preference and its limits;
  - the trained specifications;
  - the direct baseline;
  - library effects written out.

## 3. The implementation against the formulation

Each definition maps onto one piece of `exact_hole_planner.py`. I checked them by reading the code:

| Formulation | Code | Matches |
|---|---|---|
| Candidates C = scored ∩ (goal ∪ library preconditions) | `_HoleSets.__init__` | yes |
| F(H) by exhaustive reachability, any length | `reachable_states`, `_HoleSets.feasible` | yes |
| Subset-minimal | `_HoleSets.minimal` | yes |
| Producer graph over positive preconditions; U(l) strict and within C | `_producer_edges`, `_below`, `_HoleSets.upstream` | yes |
| Budget-aware replacement R ⊆ (H∖{l}) ∪ U(l), \|R\| ≤ k | `_HoleSets.replaceable` tests only sets of size min(k, \|pool\|) | yes: adding holes never removes a reachable state, so a larger set within the pool is feasible whenever a smaller one is |
| Admissible sets of the smallest size up to k | `smallest_admissible_sets` | yes |
| Uniform-cost search by (displaced, length), no length limit | `_cheapest_plans` | yes: every step costs strictly more than the last, so stopping once the best plan is passed is safe |
| NoPlan | `plan_exact` | yes: returned only when no admissible set exists, which by T3 means no feasible set of at most k candidates. It names `max_delegated` exactly when holes for every candidate would work. |

The argument for T3 (the preference never blocks a plan) holds for the code as written. U is a strict partial order, and each refusal moves to a set whose highest-ranked differing literal is lower. **No correctness bug was found.**

## 4. Fresh simulation runs

All runs went through `run_with_provenance.py`, which:

- checks the branch, the commit and the planner files before the script starts;
- records every planner call and which implementation answered it.

The order was Stage 1, then Stage 2, then Stage 4, each reading the fresh outputs of the one before. **A smoke test ran first:** Drawer Stage 2 on seed 0, Stage 4 on seed 2000 in all six conditions, and the final evaluation on seed 3000 for all 8 policies. Every row was identical to the saved one.

| Run | Configuration and seeds | Planner that answered | Fresh result | Against the saved paper record | Against the 2026-11-05 run (legacy exact planner) |
|---|---|---|---|---|---|
| Stage 1, forward operators | 5 demonstrations per task, seeds 0–4, `min_max` | none | PASS for both tasks | same predicate sets and goals; weights differ, since the saved files used `mean` | byte-identical |
| Stage 2, Stow | measured states, seeds 0–4 | minimal, 8 calls | `pick B > place B > approach A > HOLE[clear_of_walls(A)] > pick A > place A` on every seed, no tie; library only: no plan | same plans; **spec byte-identical** to the one the policy was trained on | same; only `expanded` (44 vs 95 states) and the no-interference tie list differ |
| Stage 2, Drawer | measured states, seeds 0–4 | minimal, 8 calls; relaxed, 1 call, for the script's "unrestricted" ablation, which is defined that way | `approach_handle > HOLE[handle_grasped] > pull_open` on every seed; library only: no plan; with the scripted grasp: no hole | same plans; **spec byte-identical** | same; only `expanded` (3 vs 4) differs |
| Direct-baseline spec | measured states, seeds 0–4 | relaxed `unrestricted_holes`, 6 calls, the script's definition of the baseline; minimal, 5 calls, for comparison | `approach_handle > HOLE[drawer_open]` | plan and spec byte-identical | byte-identical |
| Stage 3 input | the fresh specs | none | specs load; all 5 checkpoints fit the observation and action spaces; the environments report valid handoffs at reset (Stow 1000 and 1002, Drawer 1000 and 1001). The environment's own check asks only that the prefix succeeded, the start literal holds and the hole's literal is false. Stage 4's handoff check also asks for the fences and the hole's pre-state, which is why Stow seed 1002 passes here and fails there. | — | identical apart from the spec's file path |
| Stow Stage 4 | checkpoint `1790788534/sac_hole_best`; evaluation seeds 1000–1009; learned operator from seeds 2000–2009 | minimal, 3 calls: full, without interference, relearned | full **10/10** (A 1.5 mm, B 0.8 mm, 0 retries); library only **0/10**; without interference **3/10**; relearned **9/10**; learned operator Pre {A in pocket, B at src_b, tcp near A}, Add {clear_of_walls(A)}, Del {A in pocket}; raw handoff validity 9 of 10 evaluation seeds, 10 of 11 demonstrations; Stage 4 PASS | every value identical; the saved file predates the handoff-validity section | byte-identical |
| Drawer Stage 4 | 4 SAC policies (Stage 3 seeds 0–3), library only, scripted grasp; seeds 2000–2019 | minimal, 140 calls: each episode planned from its own measured state, plus 20 library-only | each policy **20/20** restored, 0 retries, 0 model-versus-world differences, final opening 146.5–147.6 mm; library only **0/20**, refused at `pull_open` because `handle_grasped` was not met; scripted grasp **20/20** | identical, except one report field in the library-only rows: the relaxed planner's `capability_gaps` against `unmet_needs` now | byte-identical |
| Targeted vs direct | 8 frozen best checkpoints; seeds 3000–3049 | none: the script runs a fixed pipeline, and its workers call no planner | every policy **49/50**; the failure is seed 3021 for all 8 (the forward push rebounds, so there is no handoff) | **file byte-identical** | — |
| Planner demo | complete vs incomplete library | minimal, 4 calls | as saved | byte-identical | only `expanded` differs |

## 5. Planning-only checks, minimal planner

- **From the recorded start states:** all 510 states recorded in the saved, legacy and fresh records replan to the plan recorded beside them. They are only **2 distinct symbolic states, one per task**.
- **After the learned skill:**
  - **Method:** I rebuilt the state each executive measured right after the learned step: its prediction, with the literals it flagged as different flipped. Then I planned from that state.
  - **Result:** in **108 of 108 runs**, the plan is exactly the remaining library steps, with no new hole. That includes 9 no-interference runs where B had been pushed out of the mouth.
  - **Not checked:** two runs on Stow seed 1002 where the learned step never ran, because of the B–A jam, as in the saved results.
  - **Scope:** the executives never replan during a plan. This is an analysis of the fresh records, not an experiment.
- **Handoff conditions** computed from the fresh specs equal the saved ones:
  - Stow must have A in the pocket, B at src_b and tcp near A, and must not have clear_of_walls(A) or B in the mouth.
  - Drawer must have `at_handle_standoff`, and must not have `handle_grasped`.
- **The direct baseline from the minimal planner itself:** with `handle_grasped` unscored, or with no `pull_open` in the library, the spec equals the direct-baseline spec byte for byte, from all 5 fresh measured states.
- **The preference decides the hole in both tasks.** Two single holes are subset-minimal in each:

  | Task | Learned hole | Refused hole | Cost (holes, displaced, length) |
  |---|---|---|---|
  | Drawer | `{handle_grasped}` | `{drawer_open}` | (1, 0, 3) against a cheaper (1, 0, 2) |
  | Stow | `{clear_of_walls(A)}` | `{in_region(A,src_a)}` | both (1, 0, 6), a tie |

- **The 26 named scenarios** of the relaxed-versus-exact comparison, rerun:

  | | Same plan | Only exact | Different plans | Neither |
  |---|---|---|---|---|
  | Saved, legacy planner | 10 | 12 | 0 | 4 |
  | Fresh, minimal planner | 9 | 14 | 1 | 2 |

  The relaxed plans are unchanged.

## 6. Differences, classified

**Implementation problems: none.**

**Expected from this change:**
- The search explores fewer states (`expanded`).
- The library-only report now lists both unmet needs, `drawer_open` and `handle_grasped`, where the relaxed record listed the gap `handle_grasped`.
- The Stow no-interference ablation lists different orderings:
  - three orderings tie;
  - the legacy version listed B-first and A-first, and the fresh one lists A-first and approach-A-then-B;
  - neither lists all three, because orderings through one state are merged;
  - Stage 4 executes the A-first ordering either way.

**From earlier changes, not this one:**
- extraction weights, `min_max` against `mean`;
- Drawer Stage 2 scores on 2 seeds, from the 2026-10-15 push fix;
- the handoff-validity section, added 2026-11-05.

**Environment:** no blocker. Notes:
- GitHub cannot be reached from the container.
- `PYTHONNOUSERSITE=1` is still needed.
- The README's stage times are far above this machine's: Stow Stage 4 took 46 s, against "~15 min".

## 7. Limitations relevant to publication

1. **NoPlan is relative to C, k and the transition model.** A scored literal outside C is never offered, even when a hole for it would work. A test pins this.
2. **Subset-minimal, not minimum cardinality,** and the choice can depend on k. For Stow and the Drawer, k = 1 to 4 all give the same plan.
3. **The preference reads positive preconditions only.**
   - It is not a complete causal-dependency analysis. Clearing a blocker, or restoring what a producer deletes, does not make a literal upstream.
   - It is not evidence that the chosen skill is easier to learn. The targeted-versus-direct experiment is that evidence, on one task.
4. **The preference decides the learned hole in both tasks.** For Stow, without it the choice would come down to a tie-break.
5. **Library completeness is an assumption under the transition model,** including declared world facts. Without Stow's interference fact, the executed ordering succeeds 3/10.
6. **Tie lists are incomplete:** equally cheap orderings through one state are merged.
7. **Exhaustive search,** exponential in the worst case. Here a plan takes 2.6 ms for Stow and 0.2 ms for the Drawer.
8. **One symbolic start per task.** The physical start varies continuously, but each task's experiments solve one planning problem plus ablations. Random-problem tests cover the planner more widely.
9. **The Stow model has no "one cube per slot" constraint.** In the ablation where `clear_of_walls` is unscored, the plan briefly puts both cubes on `src_a`. This doesn't arise in the paper's experiments.
10. **Earlier physical caveats, unchanged:**
    - Stow handoff validity is 51 of 53 overall, and the check was added after seed 2004.
    - Drawer seed 3021 fails because the forward push rebounds.
    - The targeted Drawer policies were trained before the push fix.
    - Contact is rigid: shallow grasps count, and the direct policies drag with one finger.
    - Seeds 3000–3049 were evaluated a second time here, as a reproduction only; nothing was selected or tuned on them.

## 8. Manuscript corrections, by priority

**The manuscript is not on this machine, so I could not read it.** This list says what must change wherever the text follows the 2026-11-03 page "Exact Planning Mode" (Definitions 1 to 10, which the Method section was written from) or quotes the legacy planner's numbers.

**Priority 1: wrong if still present**

1. **Algorithm.** Remove all of these:
   - the Waiting rule (Def. 7);
   - Last resort (in Def. 6);
   - "a hole must be used" (Def. 9);
   - the "offered" holes;
   - the search node (state, delegated, unused).

   Replace them with:
   - collect the admissible sets of the smallest size up to k;
   - for each set, run a uniform-cost search by (displaced, length), with no length limit;
   - return the cheapest plan with its ties, or NoPlan.
2. **Admissible sets (Def. 8).** \|H\| ≤ k, subset-minimal, and no l has a replacement R ⊆ (H∖{l}) ∪ U(l) with \|R\| ≤ k and F(R). Remove "replacement is tested one literal at a time".
3. **Upstream (Def. 5).** Replace the dependency walk with the producer graph:
   - an edge l→u when a library action adds l (counting implications), does not need l, and needs u;
   - u is upstream of l when a path leads from l to u and none leads back;
   - only candidates count.

   Remove every use of deletes, "uses up", blockers or "needs absent".
4. **Objective (Def. 10).** The planner minimizes (holes, displaced, length). The five-number cost may still be shown, but "literals the holes change" and "hole steps" are not minimized.
5. **Limits and NoPlan.**
   - There is no length limit. The only limit is k.
   - Remove:
     - "cheapest within the limits";
     - "a no-plan answer naming a limit proves nothing";
     - "Waiting can refuse an exactly necessary hole";
     - "feasible_with_all_holes = True means the rules refused them".
   - NoPlan now means no set of at most k candidates is feasible. `max_delegated` is named exactly when holes for every candidate would work.
6. **Guarantees.** Replace them with T1 to T5:
   - T1, soundness;
   - T2, subset-minimality, not minimum cardinality;
   - T3, the preference never blocks a plan, with the strict-order argument;
   - T4, NoPlan exact relative to C, k and T;
   - T5, optimality among the admissible sets of the smallest size.
7. **Worked example, if used.** The row "only g is learnable" now returns `HOLE[g]`, not no plan. The other five rows are unchanged.
8. **Stow explanation.** Replace the cycle and Last-resort argument:
   - Two single holes are subset-minimal.
   - The preference refuses `in_region(A,src_a)`: `clear_of_walls(A)` is upstream of it, because placing A needs holding A and picking A needs A clear, and it suffices.
   - Nothing within C is upstream of `clear_of_walls(A)`: its producers need holding A, which lies on a cycle with it.
9. **Numbers from the legacy planner.** These describe `c68d830`, not the current planner:
   - the verification counts (109,000 random problems, 28 mutants, 31,217 solvable problems, 300,000 problems);
   - the relaxed-versus-exact comparison: 26 scenarios at 10/12/0/0/4, and on 40,000 random problems 27,049 same / 11,425 neither / 1,458 only exact / 22 only relaxed / 45 different holes / 1 other plan.

   The scenarios are now 9/14/1/2. The 40,000-problem comparison was not rerun. If the paper cites it, regenerate it with `python scripts/compare_hole_planners_random.py --out <new path>`.

**Priority 2: the four distinctions, stated precisely**

10. **The library** is complete under the modelled transition, which includes the world constraints: implications, mutex groups and disturbances. A hole adds only its literal, and the constraints give its other effects.
11. **The upstream preference** reads positive preconditions only. It is not a complete causal-dependency analysis. It is a design choice, it decides both tasks, and its answer can depend on k.
12. **Subset-minimality is not minimum cardinality.** In the counterexample, k = 2 learns {p, q} although {g} alone suffices.
13. **NoPlan is exact relative to the candidate set, the budget and the symbolic transition model.** State the candidate-set limitation.

**Priority 3: experiments**

14. **Direct baseline.** Describe it as the minimal planner's answer when `handle_grasped` cannot be learned, which is the same as a library without `pull_open`; the spec is byte-identical. Drop, or demote, the relaxed `unrestricted_holes` description.
15. **Stow without the interference model.** Say "several equally cheap orderings": there are three, and the planner lists two. The executed A-first ordering succeeds 3/10.
16. **Say which planner produced the results.** The specs the policies were trained on are byte-identical to `fe374f0`'s output, and Stage 4 and the targeted-versus-direct numbers were reproduced with `fe374f0`. Nothing was retrained.

## 9. Reproduce

From the repository root:

```bash
bash artifacts/final_verification_fe374f0/run.sh stage1    # then: smoke, stage2, stow4, drawer4, final
F=artifacts/final_verification_fe374f0
for s in compare_with_saved planning_checks direct_from_minimal_planner worked_example_check; do
  PYTHONNOUSERSITE=1 python $F/run_with_provenance.py $F/provenance/$s.json $F/$s.py; done
PYTHONNOUSERSITE=1 python $F/run_with_provenance.py $F/provenance/planner_comparison_scenarios.json \
  scripts/compare_hole_planners.py --out $F/planner_comparison_scenarios.json
python $F/make_manifest.py    # manifest.json, from the files above; runs nothing
```

Nothing overwrites anything: each script refuses an existing output, so a rerun needs the outputs, `provenance/` and `logs/` moved aside first. `run_with_provenance.py` also refuses any checkout but `fe374f0` on `feature/exact-hole-planner`, and it is kept exactly as it ran. To rerun from a later commit, first check that `git diff fe374f0 -- src/inverse_skills/operators/` is empty, then set `COMMIT` in it to that commit.

**What's in this folder.** In git:
- this report and `manifest.json`;
- the scripts: `run.sh`, `run_with_provenance.py`, `make_manifest.py`, and the analysis scripts `compare_with_saved.py`, `planning_checks.py`, `direct_from_minimal_planner.py` and `worked_example_check.py`;
- the small outputs: the forward operators, plans and hole specs (`direct/` for the baseline), the Stage 3 input checks, the learned operator, the planner demo, `comparison.json`, `planner_comparison_scenarios.json`, `direct_from_minimal_planner.json` and `worked_example_check.json`.

Local only:
- the large outputs `stow_stage4.json`, `drawer_stage4.json`, `drawer_final_targeted_vs_direct.json` and `planning_checks.json`;
- the planner-call records in `provenance/`, the smoke test in `smoke/` and the console logs in `logs/`.

`manifest.json` keeps, compactly, what the local files show:
- the planner files and environment every run checked;
- each run's command, outcome and planner calls by implementation;
- the summaries the large outputs carry, with their seeds;
- every file's size and sha256, and the tracked files elsewhere with the same bytes.

The first three large outputs are byte-identical to tracked files: `artifacts/default_mode_validation/after_migration/stow_stage4.json` and `drawer_stage4.json`, and `artifacts/drawer/final_targeted_vs_direct.json`.
