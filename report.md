# Inverse Skill Learning as World-State Restoration

*A technical report for readers with an undergraduate CS background.*

---

## 1. The Problem

A robot arm learns to do things — pick up a cup, push a box to a shelf, grasp and hold an object. These are **forward skills**: they change the world from some starting state to some ending state.

But real deployment often requires the opposite. A museum guide robot demonstrates a skill (picking up an exhibit), and then needs to undo it before the next visitor arrives. A household robot accidentally pushes a mug off its coaster and needs to put it back.

**The question this project answers:** Given only a record of what a forward skill *did*, can we automatically synthesize a plan that *undoes* it — without hand-coding the inverse?

The key insight is:

> **Undoing a skill = restoring the world to the skill's preconditions, minus anything the skill permanently added.**

This turns the inverse problem into a **planning problem with a learned objective function**, rather than a trajectory-reversal problem.

---

## 2. Background Concepts

### 2.1 STRIPS Operators (Classical AI Planning)

Classical robot planning describes skills as **operators**:

```
operator pick_place:
  Preconditions:  gripper_open()  AND  in_region(cube, source)
  Add effects:    in_region(cube, target)
  Delete effects: in_region(cube, source)
```

- **Preconditions** — what must be true *before* the skill runs
- **Add effects** — what becomes true *after* the skill runs
- **Delete effects** — what stops being true *after* the skill runs

Classical planning requires hand-coding these operators. This project *learns* them from data.

### 2.2 Soft Predicates (Bridging Continuous to Symbolic)

A predicate like `in_region(cube, source)` is classically Boolean: either true or false. But robot sensor data is continuous — a cube might be *almost* inside a region but not quite.

This project uses a **soft sigmoid scoring** approach:

```
score = sigmoid(margin / temperature)
       = 1 / (1 + exp(-margin / temperature))
```

where `margin` is a signed distance (positive = inside/satisfied, negative = outside/unsatisfied), and `temperature` controls sharpness.

```
margin:   -0.05   -0.02    0.0    0.02   0.05
          ──────────────────────────────────────
score:     0.01    0.12    0.50   0.88   0.99

(for temperature = 0.01, as used in InRegionPredicate)
```

This means:
- A cube well inside a region → score ≈ 1.0 (definitely true)
- A cube well outside a region → score ≈ 0.0 (definitely false)
- A cube on the edge → score ≈ 0.5 (uncertain)

The same formula bridges the continuous robot world to symbolic planning concepts.

---

## 3. The Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     OFFLINE LEARNING PHASE                          │
│                                                                     │
│  Forward skill demonstrations                                       │
│  (e.g., robot picks cube from source and places it at target)       │
│                                                                     │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐                           │
│  │Rollout 1│   │Rollout 2│   │Rollout 3│  ...                       │
│  │start/end│   │start/end│   │start/end│                           │
│  └────┬────┘   └────┬────┘   └────┬────┘                           │
│       └─────────────┴─────────────┘                                 │
│                      │                                              │
│                      ▼                                              │
│            ┌─────────────────┐                                      │
│            │  Predicate      │  Evaluate every predicate at         │
│            │  Evaluator      │  start and end of each rollout       │
│            └────────┬────────┘                                      │
│                     │  start_scores[], end_scores[]                 │
│                     ▼                                               │
│            ┌─────────────────┐                                      │
│            │ OperatorExtract │  For each predicate key p:           │
│            │      -or        │   start_mean = mean(start_scores[p]) │
│            └────────┬────────┘   end_mean = mean(end_scores[p])     │
│                     │            delta = end_mean - start_mean      │
│                     │                                               │
│                     │   if start_mean >= 0.80 → PRECONDITION        │
│                     │   if delta >= 0.35      → ADD EFFECT          │
│                     │   if -delta >= 0.35     → DELETE EFFECT       │
│                     │                                               │
│                     ▼                                               │
│            ┌─────────────────┐                                      │
│            │ LearnedOperator │  e.g., for pick_place:               │
│            │                 │   pre:  {gripper_open, in_region_src}│
│            │                 │   add:  {in_region_tgt}              │
│            │                 │   del:  {in_region_src}              │
│            └────────┬────────┘                                      │
│                     │                                               │
│                     ▼                                               │
│        inverse_target = preconditions                               │
│                       + delete_effects                              │
│                       + negate(add_effects)                         │
│                                                                     │
│            ┌─────────────────────────────────────────┐             │
│            │ RestorationObjective (potential func.)   │             │
│            │  V(s) = weighted avg of inverse targets  │             │
│            │         evaluated in scene s             │             │
│            └─────────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘

                      │  (V, inverse_targets learned)
                      ▼

┌─────────────────────────────────────────────────────────────────────┐
│                     ONLINE PLANNING PHASE                           │
│                                                                     │
│  After forward skill completes → current scene s_fwd               │
│                                                                     │
│            ┌─────────────────┐                                      │
│            │ ToyInversePlanner│  BFS over primitive action sequences │
│            │  (BFS search)   │  scored by V(s)                      │
│            └────────┬────────┘                                      │
│                     │                                               │
│  Expand: pick(cube), place(source), place(target), push, noop       │
│                                                                     │
│  Score each reachable state with V(s)                               │
│  Return first sequence reaching V(s) >= 0.98                        │
│                                                                     │
│            ┌─────────────────┐                                      │
│            │  Inverse Plan   │  e.g., ["pick(cube)", "place(source)"]│
│            └─────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Operator Extraction in Detail

The extractor evaluates all registered predicates at the **first** and **last** scenes of every demonstration rollout, then averages across rollouts.

### Example: learning the `push_to_target` operator

Suppose we run 3 push rollouts. The cube starts near the source region and ends at the target region. The robot gripper stays open throughout.

```
Predicate                  start_mean   end_mean   delta   Role?
─────────────────────────────────────────────────────────────────
gripper_open()               0.9997      0.9997    +0.000  Precondition (start≥0.80)
in_region(cube, source)      0.9900      0.0100    -0.980  Precondition + DELETE effect
in_region(cube, target)      0.0100      0.9900    +0.980  ADD effect
holding(cube)                0.0180      0.0180    +0.000  (neither)
```

Thresholds applied:
- `start_mean >= 0.80` → precondition
- `delta >= 0.35` → add effect
- `-delta >= 0.35` → delete effect

Result:
```
Operator push_to_target:
  Pre:  gripper_open(), in_region(cube, source)
  Add:  in_region(cube, target)
  Del:  in_region(cube, source)
```

### The inverse target is derived automatically

```python
def inverse_target_terms(self) -> list[PredicateTerm]:
    terms = []
    terms.extend(self.preconditions)        # restore preconditions
    terms.extend(self.delete_effects)       # restore deleted facts
    for term in self.add_effects:
        terms.append(negate(term))          # undo added facts
    return terms
```

For `push_to_target`:
```
inverse target:
  [+] gripper_open()           (precondition — must remain true)
  [+] in_region(cube, source)  (deleted — must be restored)
  [-] in_region(cube, target)  (added — must be negated)
```

The `[-]` negation flips the score: `term_score = weight * (1 - raw_score)` for negative polarity.

---

## 5. The Restoration Potential Function

The `RestorationObjective` converts the inverse target terms into a scalar score for any world state:

```
V(s) = Σ  weight_i * term_score_i(s)
       ─────────────────────────────
           Σ weight_i
```

This is a weighted average in [0, 1]:
- V = 1.0 means the world is fully restored (all inverse targets satisfied)
- V = 0.0 means nothing has been restored
- V ≥ 0.98 triggers "success" in the planner

### Worked example

After `push(cube)` completes, the cube sits at the target. Evaluate V:

```
Term                          raw_score   polarity   term_score
───────────────────────────────────────────────────────────────
gripper_open()                  0.9997      [+]        0.9997
in_region(cube, source)         0.0000      [+]        0.0000
in_region(cube, target)         0.9900      [-]        0.0100

V = (0.9997 + 0.0000 + 0.0100) / 3 ≈ 0.333
```

The planner then searches for actions that increase V:

```
s₀ (after push):    V = 0.333   cube at target, gripper open
  → pick(cube):     V = ?
  → place(source):  V = ?   (can't place if not holding)
  → noop:           V = 0.333

After pick(cube):
  in_region(cube,source): 0.000 [+]
  in_region(cube,target): 0.018 [-] → 1-0.018 = 0.982
  gripper_open(): 0.018 [+]
  V ≈ (0.000 + 0.982 + 0.018) / 3 ≈ 0.333   (same, different reason)

After pick(cube) → place(source):
  in_region(cube,source): 0.9997 [+]
  in_region(cube,target): 0.000  [-] → 1.000
  gripper_open(): 0.9997 [+]
  V ≈ (0.9997 + 1.000 + 0.9997) / 3 ≈ 1.000  ✓  SUCCESS
```

Inverse plan: `["pick(cube)", "place(source)"]`

---

## 6. Three Skills, Three Operators

The project validates on three skills, each with a different operator structure.

```
┌────────────────┬──────────────────────────────────────┬─────────────────────────────────────────┐
│ Skill          │ Operator                             │ Inverse Plan                            │
├────────────────┼──────────────────────────────────────┼─────────────────────────────────────────┤
│ pick_place     │ Pre: gripper_open, in_region_src     │ pick(cube) → place(source)              │
│ (pick + place) │ Add: in_region_tgt                   │ 2 steps                                 │
│                │ Del: in_region_src                   │                                         │
├────────────────┼──────────────────────────────────────┼─────────────────────────────────────────┤
│ push_to_target │ Pre: gripper_open, in_region_src     │ pick(cube) → place(source)              │
│ (push only)    │ Add: in_region_tgt                   │ 2 steps — SAME inverse as pick_place    │
│                │ Del: in_region_src                   │ (same abstract effect, different motion)│
├────────────────┼──────────────────────────────────────┼─────────────────────────────────────────┤
│ grasp_hold     │ Pre: gripper_open, in_region_src     │ place(source)                           │
│ (pick only,    │ Add: holding(cube)                   │ 1 step — DIFFERENT inverse              │
│  no place)     │ Del: gripper_open                    │ (cube already at source; just release)  │
└────────────────┴──────────────────────────────────────┴─────────────────────────────────────────┘
```

The first two skills have **different motions** (pick+place vs. push) but **identical abstract operators** (same preconditions, add/delete effects). The system correctly infers the same inverse plan for both.

The `grasp_hold` skill has a **genuinely different operator** (it adds `holding` and deletes `gripper_open` instead of moving the cube). The system automatically infers a different 1-step inverse.

This cross-class generalization is a key claim: the inverse is derived from *what changed abstractly*, not from *how the motion was executed*.

---

## 7. Parameterized Templates

The learned operators are grounded in specific names (`cube`, `source`, `target`). The `OperatorParameterizer` generalizes them into reusable templates with variable slots.

### Role binding inference

The parameterizer reads the first rollout and infers:
- **?obj** = the object that moved most (Euclidean distance)
- **?src** = the region the object was in at the start
- **?dst** = the region the object was in at the end

```
cube  →  ?obj
source  →  ?src
target  →  ?dst
```

### Grounded vs. parameterized

```
GROUNDED operator (cube, source, target):
  Pre:  gripper_open(), in_region(cube, source)
  Add:  in_region(cube, target)
  Del:  in_region(cube, source)

PARAMETERIZED template:
  Parameters: ?obj:object, ?src:region, ?dst:region
  Pre:  gripper_open(), in_region(?obj, ?src)
  Add:  in_region(?obj, ?dst)
  Del:  in_region(?obj, ?src)
```

The template is **invariant** across renamings. Running the same experiment with `mug/home/goal` instead of `cube/source/target` produces an identical template — verified by `parameterized["templates_match"] == True`.

### Distractor robustness

With a second irrelevant object (`can`) present during demonstrations, the extracted operator still focuses only on `cube`. The can appears in some predicates (`in_region(can, source)`) but those predicates have low start/end delta, so they are not included in the operator. Template invariance holds with distractors too.

---

## 8. Architecture Overview

```
inverse_skills/
│
├── core/                       Scene representation layer
│   ├── scene.py                SceneGraph, RobotState, ObjectState
│   └── geometry.py             Pose, Region, signed_margin()
│
├── predicates/                 Soft predicate evaluation
│   ├── base.py                 Predicate ABC, PredicateResult, sigmoid scoring
│   └── geometric.py            InRegion, GripperOpen, Holding, Near, AtPose
│
├── logging/
│   └── rollout.py              ForwardRollout: skill_name, scenes[], first(), last()
│
├── operators/                  Core learning and planning
│   ├── schema.py               LearnedOperator, PredicateTerm, inverse_target_terms()
│   ├── extractor.py            OperatorExtractor: first/last score delta → operator
│   ├── restoration.py          RestorationObjective: potential(), term_score(), reward()
│   ├── parameterized.py        OperatorParameterizer, RoleBindingInferer
│   └── toy_planner.py          ToyInversePlanner: BFS over PrimitiveLibrary
│
└── toy/                        Toy tabletop domain
    ├── domains.py              Scene factories, predicate registry builders
    ├── generators.py           Rollout generators (pick_place, push, grasp_hold)
    ├── primitives.py           PrimitiveLibrary: pick, place, push, noop
    └── simulator.py            ToyTabletopSimulator
```

Data flows through the system as:

```
ForwardRollout[]
       │
       ▼
OperatorExtractor  ──►  LearnedOperator  ──►  RestorationObjective
                                                      │
                              OperatorParameterizer ◄─┤
                              (optional)               │
                                                      ▼
                                              ToyInversePlanner
                                                      │
                                                      ▼
                                                  PlanResult
                                            (actions[], success)
```

---

## 9. Validation on Physics Simulation (ManiSkill3)

The pipeline is also validated on `PickCube-v1` in ManiSkill3, a real physics simulator using a Panda arm.

The oracle:
1. Moves the gripper above the cube
2. Descends to grasp height
3. Closes the gripper
4. Lifts to 15 cm above start

```
5/5 seeds succeeded  (cube lifted from z=0.020 to z≈0.180)
```

The extracted operator from physics data:
```
Pre:  gripper_open(), in_region(cube, table_surface)
Add:  holding(cube)
Del:  gripper_open(), in_region(cube, table_surface)
```

This matches the toy `grasp_hold` structure exactly. One extra delete effect appears — `in_region(cube, table_surface)` — because in physics the cube physically leaves the table surface when lifted. In the toy domain, regions are abstract and the cube "stays" in its region symbolically.

**Engineering note:** ManiSkill has a state-leak bug: after a successful grasp, `env.reset()` returns an observation with `is_grasped=True` even though the gripper is open. Two "open-gripper, no-translation" steps after each reset clear the flag reliably.

---

## 10. Key Results Summary

| Claim | Result |
|---|---|
| Same-class inverse success (pick_place) | True — plan: `pick → place(source)` |
| Cross-class inverse success (push) | True — same 2-step plan |
| Grasp-hold inverse (different operator) | True — 1-step plan: `place(source)` |
| Template invariance across object/region renamings | True |
| Template invariance with irrelevant distractor present | True |
| Physics simulator generalisation (ManiSkill3) | True — correct operator recovered |

---

## 11. Why This Is Not Trajectory Reversal

A common naive approach to "undoing" a skill is to reverse the trajectory: play the motion backwards. This fails in most real scenarios because:

1. Grasping is not the reverse of releasing (different contact physics)
2. The path taken during the forward skill may not be collision-free in reverse
3. Pushing cannot be reversed by the same gripper motion (the gripper was behind the object, not in front)

This project instead asks: *what world state should we restore to?* and then *searches for a plan that reaches it*, using whatever primitive actions are available. The inverse planner is allowed to use actions completely different from the forward skill.

For push: the forward skill uses `push(target)`. The inverse plan uses `pick(cube) → place(source)` — a pick-and-place, which is physically easier and entirely different from the push motion.

---

## 12. Limitations and Honest Scope

- **Toy domain planner is BFS** — scales poorly to large action spaces or long horizons. A learned policy or model-predictive controller would be needed for real deployment.
- **Physics demo uses a scripted oracle**, not a learned policy. The key claim is about operator extraction from rollouts, not about learning the forward skill.
- **Predicate registry is hand-specified** — an engineer must decide which predicates to track. Automatic predicate invention is out of scope.
- **Single object, simple table domain** — multi-object manipulation with dependencies (e.g., stacked objects) is not addressed.
