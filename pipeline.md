# The Inverse-Skill Pipeline

*Complete specification of the framework: soft predicate grounding, operator extraction,
inverse-target derivation, symbolic planning, the fence/active partition, and the residual
RL reward — with every hyperparameter and its justification.*

Written 2026-06-28, after the grounding/reward unification and the planner fixes.

---

## 0. Overview

The framework undoes a manipulation skill by **restoring the world state**, not by rewinding
the motor trajectory. Six steps:

| # | Step | Output |
|---|------|--------|
| 1 | Extract a STRIPS operator from demonstrations | $o=\langle Pre, Add, Del\rangle$ |
| 2 | Derive the inverse target | $\mathcal I(o) = Pre \cup Del \cup \neg Add$ |
| 3 | BFS-plan over known primitives | symbolic prefix $\pi_{\text{sym}}$ |
| 4 | Partition on the executed handoff scene | fences $\mathcal F$, active residual $\mathcal A$ |
| 5 | Residual RL from the handoff | policy $\pi_{\text{res}}$ |
| 6 | Concatenate | inverse skill $\pi_{\text{sym}} \circ \pi_{\text{res}}$ |

Everything downstream of step 1 is derived automatically. The only human inputs are the
predicate repertoire and its geometric parameters.

---

## 1. Terminology

Three distinct objects, three distinct names. No word is reused:

| symbol | name | range | units | produced by |
|---|---|---|---|---|
| $m_p(s)$ | **margin** | $\mathbb R$ | metres | geometry |
| $V_p(s)$ | **score** | $[0,1]$ | — | squash by temperature $T_p$ |
| $\hat V_p(s)$ | **signed score** | $[-1,+1]$ | — | polarity + centring, $\sigma_p(2V_p-1)$ |

$$m_p(s) \;\xrightarrow{\;\text{temperature } T_p\;}\; V_p(s) \;\xrightarrow{\;\text{polarity } \sigma_p,\ \text{centring}\;}\; \hat V_p(s)$$

**Margin vs. temperature.** Both are in metres, but they are different kinds of thing. The
margin is a *variable*: it depends on the state and answers "how far is the world from the
decision boundary?". The temperature is a *constant* of the predicate: it answers "how far
counts as far?". They only ever appear as the dimensionless ratio $m_p/T_p$ — temperature is
the unit in which margin is measured. A 5 mm margin is "practically satisfied" when
$T_p=1$ mm ($V_p=0.993$) and "undecided" when $T_p=50$ mm ($V_p=0.525$).

$T_p$ controls **only** the steepness of the transition,

$$\frac{dV_p}{dm_p} = \frac{V_p(1-V_p)}{T_p}, \qquad \left.\frac{dV_p}{dm_p}\right|_{m_p=0} = \frac{1}{4T_p},$$

and has **no** effect on where the boundary sits, since $V_p \ge \tfrac12 \iff m_p \ge 0$ for
every $T_p$. Small $T_p$ gives a sharp, near-Boolean predicate whose gradient dies within a few
$T_p$ of the boundary; large $T_p$ gives a smooth, far-reaching signal with a blurrier truth
value. Beyond $|m_p|\approx 3T_p$ the score is saturated to within ~5% of 0 or 1.

> **Naming note.** The earlier term *signed normalized margin* ($\tilde m_p$) has been retired.
> It named the object after the wrong parent: it is not a margin (dimensionless, bounded), and
> "normalized" understates a nonlinear squash. It is exactly $\sigma_p(2V_p-1)$
> — a transform of the **score**. Hence *signed score*, written $\hat V_p$. In code:
> `signed_margin_reward` → `signed_score_reward`.
>
> Still correctly called margins, and unchanged: `Region.signed_margin()`, and the observation
> feature `at_pose_margin_by_tol` ($m_p/\tau$ — linear, unbounded, genuinely a normalized margin).

---

## 2. Scene graph and margins

A scene $s$ records continuous state: object poses $x_o$, end-effector pose $x_{tcp}$,
gripper width $w$, grasp state, and named regions.

Every predicate $p$ defines a **signed margin** $m_p(s) \in \mathbb R$, in metres, positive
when satisfied and equal to the distance from the decision boundary:

$
\begin{aligned}
\text{at\_pose}(o, q) &: & m &= \tau - \lVert x_o - q\rVert \\
\text{gripper\_open}() &: & m &= w - w_{\min} \\
\text{tcp\_near}(o) &: & m &= \delta - \lVert x_{tcp} - x_o\rVert \\
\text{in\_region}(o, R) &: & m &= \begin{cases}
\min_i \min(x_{o,i}-R^{\text{lo}}_i,\; R^{\text{hi}}_i-x_{o,i}) & x_o \in R\\
-\lVert \max(R^{\text{lo}}-x_o,0) + \max(x_o-R^{\text{hi}},0)\rVert & \text{otherwise}
\end{cases}
\end{aligned}
$

The Boolean reading is simply $\operatorname{truth}(p,s) \equiv m_p(s) \ge 0$.

---

## 3. Soft grounding — one function for scoring **and** reward

### 3.1 The score

The grounding maps the margin to a soft truth value in $[0,1]$:

$$
\boxed{\;V_p(s) \;=\; \tfrac12\left(1 + \tanh\frac{m_p(s)}{T_p}\right)\;}
\tag{1}
$$

$T_p$ is the predicate's **temperature**: the width, in metres, of the soft band around the
decision boundary. Concretely, $T_p$ is the margin at which the score reaches
$\tfrac12(1+\tanh 1) = 0.881$, and the distance over which the signed score ramps
(slope $1/T_p$ at the boundary, saturated past $\approx 3T_p$).

> **Recent change — two steps.** (i) `PredicateResult.score` previously computed
> `sigmoid(margin/temperature)`; it now computes the $\tanh$ form. These are the same function,
> since $\sigma(x)=\tfrac12(1+\tanh\tfrac x2)$. (ii) $T_p$ was then **redefined as the $\tanh$
> scale** rather than the logistic temperature, which removes a factor of 2 from every formula
> — and, because the reward scale was already $2T_p$, removes the separate reward scale symbol
> $c_p$ from the framework altogether. Every stored temperature was doubled to compensate, so
> the change is exactly behaviour-preserving: score curves verified **bit-identical** over
> $m\in[-200,200]$ mm for all six predicate classes and all four configured temperatures; tests
> pass; toy bundle byte-identical.
>
> **The one caveat.** "Temperature" now denotes the $\tanh$ scale, which is *twice* the logistic
> temperature it used to mean. Anyone assuming the $\sigma(m/T)$ convention will be off by 2×,
> and old logs recording e.g. `temperature: 0.015` describe the same predicate that new logs
> record as `0.03`. The equivalent sigmoid form is now $V_p = \sigma(2m_p/T_p)$.

Useful properties, all independent of $T_p$:

$$V_p(s) \ge \tfrac12 \iff m_p(s) \ge 0 \iff \operatorname{truth}(p,s)$$

so the half-point is exactly the Boolean boundary, and thresholds at $\tfrac12$ are
hyperparameter-free.

### 3.2 The signed score is the same quantity, re-centred

RL needs a **signed** score: a fence must be able to contribute *negative* reward when
violated, which $[0,1]$ cannot express. The signed score is

$$
\boxed{\;\hat V_p(s) \;=\; \sigma_p\bigl(2V_p(s) - 1\bigr) \;=\; \sigma_p \tanh\frac{m_p(s)}{T_p} \;\in[-1,+1]\;}
\qquad \sigma_p\in\{+1,-1\}
\tag{2}
$$

where $\sigma_p=-1$ for negated add-effects. The two expressions are identical — substitute (1)
— which is the whole point: **the reward is the score**, on a signed axis, using the same $T_p$.

**Why $x \mapsto 2x-1$ and nothing else.** We need a map $[0,1]\to[-1,+1]$ that is (i) affine,
(ii) order-preserving, and (iii) sends the decision boundary $V_p=\tfrac12$ to $0$, so that
"satisfied" is exactly "non-negative". Those three conditions determine the map uniquely:
$f(x)=2x-1$. There is no free parameter and no alternative.

Consequently **the reward introduces no hyperparameter of its own**. There is one temperature
per predicate and nothing else; the earlier separate reward scale $c_p$ no longer exists.

### 3.3 One temperature per predicate, and nothing else

**There is no reward scale.** `signed_score_reward` computes the contribution straight from the
score:

```python
x = sign * (2 * predicate.evaluate(scene).score - 1)      # == sign * tanh(margin / T_p)
```

A term is `(predicate, sign, mode)` — passing a 4-tuple with an explicit scale raises an error.
This makes "the reward is the score" true *by construction* rather than by convention: the two
cannot drift apart, and the framework has exactly **one hyperparameter per predicate**, $T_p$.

Historically the reward had its own scale $c_p$, set per term to that predicate's geometric
threshold while $T_p$ was chosen independently. Eliminating $c_p$ is equivalent to imposing
$c_p = 2T_p^{\text{(old)}} = T_p^{\text{(new)}}$. Two of the four terms already satisfied that
and are unchanged; two did not:

| Term | role | $T_p$ | effective scale, before | changed? |
|---|---|---|---|---|
| `at_pose(cube,src)` | **active** | 0.030 | $3\tau = 0.030$ | no — **bit-identical** |
| `gripper_open()` | fence | 0.040 | $w_{\min} = 0.040$ | no — **bit-identical** |
| `at_pose(cube,goal)` | fence | 0.030 | $\tau = 0.010$ | yes |
| `tcp_near(cube)` | fence | 0.040 | $\delta = 0.050$ | yes |

Both changed terms are fences, so they differ only where they are *violated*:

- `at_pose(cube,goal)` (max $\Delta$ 0.454) fires only if the cube comes within $\tau=10$ mm of
  the forward goal. The cube sits ~200 mm away and RL nudges it by millimetres, so that region
  is never visited.
- `tcp_near(cube)` (max $\Delta$ 0.100) fires only when the end-effector is more than
  $\delta=5$ cm from the cube — reachable during early exploration, so this one can genuinely
  differ.

**A retrain is therefore warranted** to report numbers under the unified reward, even though the
term RL actually optimises is unchanged.

**The trade-off, stated plainly.** $T_p$ controls *both* the symbolic sharpness of the predicate
and the reach of the RL gradient. These were previously independent knobs. Coupling them is what
makes "reward = score" true; decoupling them again would need a second parameter, and the reward
would no longer *be* the score. For `at_pose` the coupling is comfortable: $T = 3\tau$ gives
usable gradient out to $\approx 3T = 9$ cm, covering the whole residual range. A much sharper
$T$ would kill the gradient past $\approx\tau$ and strand the policy — the failure the old
independent scale factor existed to prevent.

So the requirement "an active predicate's reward is its score" holds exactly:

$$R_{\text{active}}(s) = 2V_{\text{at\_pose}(cube,src)}(s) - 1 = \tanh\frac{m(s)}{T}$$

The reward crosses zero exactly at the tolerance boundary:

| placement error | margin $m$ | score $V_p$ | reward $2V_p-1$ |
|---|---|---|---|
| 0 mm | $+10.0$ mm | 0.661 | $+0.322$ |
| 5 mm | $+5.0$ mm | 0.583 | $+0.165$ |
| **10 mm ( = $\tau$)** | $0.0$ mm | **0.500** | **0.000** |
| 16.6 mm (symbolic prefix) | $-6.6$ mm | 0.392 | $-0.217$ |
| 30 mm | $-20.0$ mm | 0.209 | $-0.583$ |

and the whole reward collapses to a function of the soft scores alone:

$$
R(s) \;=\; \underbrace{\sum_{p\in\mathcal A}\sigma_p\bigl(2V_p(s)-1\bigr)}_{\text{active}}
\;+\; \underbrace{\sum_{p\in\mathcal F}\min\Bigl(0,\;\sigma_p\bigl(2V_p(s)-1\bigr)\Bigr)}_{\text{fence}}
\tag{4}
$$

No quantity appears in (4) that is not a predicate score.

### 3.4 Why bounded, and why fences are one-sided

$\tanh$ saturates, so a state far outside tolerance contributes at most $\pm1$; this keeps the
SAC value function from diverging during exploration. The ablation confirms both choices are
load-bearing:

| Reward | Success @1 cm | Failure mode |
|---|---|---|
| fences with plain $\tanh$ | 10% | saturated fences pay the agent to stand still |
| unbounded margin (no $\tanh$) | 0% | value function explodes, cube ends >100 mm off |
| active + one-sided fences (ours) | **90%** | — |

A fence is silent while satisfied ($\min(0,\cdot)=0$, no gradient) and only bites when
violated — the literal reading of "preconditions and delete-effects must continue to hold",
with no incentive to over-saturate a fence at the residual's expense.

---

## 4. Step 1 — Operator extraction

For each predicate, average $V_p$ over the first and last scene of $N$ demonstrations:

$$\bar V_p^{\text{start}},\quad \bar V_p^{\text{end}},\quad \Delta_p = \bar V_p^{\text{end}} - \bar V_p^{\text{start}}$$

Classification, with $\theta_{\text{pre}}=0.80$, $\theta_{\Delta}=0.35$, $\Delta_{\min}=0.15$:

$$
p \in Pre \iff \bar V_p^{\text{start}} \ge 0.80,\qquad
p \in Add \iff \Delta_p \ge 0.35,\qquad
p \in Del \iff -\Delta_p \ge 0.35
$$

(both effect rules additionally require $|\Delta_p| \ge \Delta_{\min}$). Term weights are
$w_p = \bar V_p^{\text{start}}$ for preconditions and $w_p = |\Delta_p|$ for effects.

**Demonstration boundaries matter more than you would expect.** The rollout must span the
*whole* forward skill — the world before `push(cube)` runs, to the world it leaves behind
(after the stroke, the lift that breaks contact, and the gripper opening). Cutting at the push
*stroke* instead samples the gripper mid-close and the TCP mid-contact, which yields
$Pre=\emptyset$ (the Panda pushes with a **closed** gripper: $V_{\text{gripper\_open}}=0.119$)
and turns `tcp_near` into a spurious **add effect** ($0.269 \to 0.702$). With the correct
boundaries, extraction on PushCube gives:

```
Pre: gripper_open(), in_region(cube,src)
Add: in_region(cube,goal)
Del: in_region(cube,src)
```

> **Caveat vs. the paper.** The submitted operator lists
> `Pre: AT_POSE ∧ TCP_NEAR ∧ GRIPPER_OPEN`. `TCP_NEAR` and `GRIPPER_OPEN` cannot both hold at
> any single boundary of this scripted skill: at reset the TCP is at home (not near), and
> during the stroke the gripper is closed. The operator above is what real demonstrations
> actually yield.

---

## 5. Step 2 — Inverse target

$$\mathcal I(o) \;=\; Pre \;\cup\; Del \;\cup\; \neg Add \tag{5}$$

Preserve the preconditions, restore what was deleted, negate what was added. Negation flips
polarity, so its soft value is $1 - V_p$ (equivalently $\sigma_p=-1$ in (2)).

The **restoration potential** aggregates the target into a scalar:

$$
\mathcal V(s) \;=\; \frac{\sum_{p\in\mathcal I(o)} w_p\, V^{\pm}_p(s)}{\sum_{p\in\mathcal I(o)} w_p},
\qquad V^{\pm}_p = \begin{cases}V_p & \sigma_p=+1\\ 1-V_p & \sigma_p=-1\end{cases}
\tag{6}
$$

### Weighted vs. unweighted — a distinction that was previously conflated

Two different quantities are needed, and mixing them was a real bug:

- $V^{\pm}_p(s)$ — **unweighted**, in $[0,1]$. This is what *thresholds* are defined over:
  the fence test $V_p(s_h)\ge\theta$, and BFS term-reachability.
- $w_p V^{\pm}_p(s)$ — **weighted**. Only for the potential (6).

`RestorationObjective` previously exposed only the weighted form, and the fence/reachability
tests thresholded it. Since extraction weights run 0.78–0.88, a fully satisfied predicate
scored $0.88 \times 1.0 = 0.88 < 0.90$ and was reported as *unrestored*. In the PushCube run
this produced an **empty fence set with every margin satisfied**. The API now separates them:

```python
term_value(term, scene)  ->  V_p^±        # unweighted — for thresholds
term_score(term, scene)  ->  w_p · V_p^±  # weighted   — for the potential
```

The bug was latent in the toy domain, where predicates saturate and $w_p\approx1$; the toy
bundle regenerates byte-identical after the fix.

---

## 6. Step 3 — BFS inverse planning

Breadth-first search over the primitive library $\{\text{pick}(o), \text{place}(R),
\text{push}(R), \text{noop}\}$, scoring states by $\mathcal V$, to depth $d_{\max}=3$.
It returns the best-scoring reachable state, the action sequence to it, and

$$\text{term\_max}_p = \max_{s \text{ visited}} V^{\pm}_p(s)$$

On PushCube the planner returns `pick(cube) → place(src)` on every seed tested — matching the
prefix the paper states.

### The abstract model must move the end-effector

`PrimitiveLibrary.apply()` is the planner's **forward model**: it predicts each primitive's
effect on the scene. It updated the *object* pose but never the *end-effector* pose.

**What that caused.** During search, the TCP stayed frozen wherever the forward push left it —
about 10 cm above the goal — no matter which actions were simulated. So after simulating
`pick(cube) → place(src)`, the model believed the cube was at `src` while the gripper was still
20 cm away above `goal`. Any TCP-relative predicate, e.g. $\text{tcp\_near}(cube)$, then scored
$\approx0$ in **every** visited state, so $\text{term\_max}_{\text{tcp\_near}}$ never crossed
the reachability threshold and the planner declared it **provably unsatisfiable by any
primitive** — dumping it into the active residual and instructing RL to "fix" it.

That conclusion is false: on the real robot the gripper is obviously beside the cube right
after placing it there. The planner would have handed RL a **phantom residual** — a predicate
that is already satisfied in reality but that the search model says is unreachable.

**Why it never showed up in the paper:** `tcp_near` drops out of the extracted operator, so it
never entered $\mathcal I(o)$. The defect was latent, and would have fired for any operator
whose inverse target contains a TCP-relative term.

**Fix.** Each primitive now places the modelled TCP at the object it acted on:

$$\text{pick}(o),\ \text{place}(R),\ \text{push}(R) \;\Longrightarrow\; x_{tcp} \leftarrow x_o'$$

where $x_o'$ is the object's post-action position. Toy results are unchanged (byte-identical
bundle).

---

## 7. Step 4 — Handoff, validity gate, and the partition

### 7.1 Validity gate

$\pi_{\text{sym}}$ is executed on the real robot, producing the **handoff scene** $s_h$. It is
accepted only if both gates pass, otherwise the scenario is re-rolled with a fresh seed (up to
50 attempts):

| Gate | Test | Value |
|---|---|---|
| forward push | $\lVert x^{xy}_{cube} - x^{xy}_{\text{goal}}\rVert < \max(0.02,\ 0.30\,|dx|)$ | 2 cm floor / 30% |
| prefix execution | $\lVert x_{cube} - x_{\text{commanded drop}}\rVert < 0.015$ | 15 mm |

The second gate measures against *where the place primitive was commanded to release*, which
is what "did the prefix do its job" means. Without it a failed grasp (cube never leaving the
goal, ~200 mm error) pollutes the partition with phantom residuals.

### 7.2 The partition

$$
\mathcal F = \{\,p \in \mathcal I(o) \;:\; V^{\pm}_p(s_h) \ge \theta\,\},
\qquad \mathcal A = \mathcal I(o)\setminus\mathcal F,
\qquad \theta = 0.80
\tag{7}
$$

$\theta$ is set equal to the extractor's precondition threshold $\theta_{\text{pre}}$, so
"counts as satisfied" means the same thing in the symbolic and residual phases. This is a
choice worth stating explicitly rather than tuning.

> **Recent change.** These modes were previously **hardcoded** — the term list literally
> spelled out `"bipolar"` / `"fence"` per predicate, decided before the handoff existed. They
> are now derived from $s_h$ by `_partition_inverse_target()`. Verified to reproduce the
> hardcoded partition exactly on every seed tested, so the trained policy remains valid:
>
> ```
> at_pose(cube,src)   V_p=0.56  -> bipolar   (active residual A)
> gripper_open()      V_p=0.88  -> fence
> NOT at_pose(goal)   V_p=1.00  -> fence
> tcp_near(cube)      V_p=0.89  -> fence
> ```
>
> The difference is that a handoff which *fails* to restore a predicate now promotes it into
> the residual automatically, instead of being silently fenced as if already satisfied.

---

## 8. Step 5 — Residual RL

SAC (Stable-Baselines3) optimises (4) from the handoff state.

| Setting | Value |
|---|---|
| observation | 12-D predicate-grounded feature vector |
| action | 4-D: end-effector $\Delta xyz$ + gripper width, scale 0.2 |
| episode length | 60 steps |
| handoff perturbation | $\pm2$ cm (curriculum) |
| tolerance $\tau$ | 1 cm (curriculum start = end) |
| success | $V_p \ge 0.5$ on the active term, i.e. $m_p \ge 0$ |

The observation is predicate-grounded and tolerance-normalised, so the policy transfers across
the curriculum. Its 12 dimensions are the cube→src offset (3), cube→TCP offset (3), gripper
width (1), and four scalar predicate features: `at_pose` score, its normalised margin, the
cube–src distance, `gripper_open` score, and the cube–forward-goal distance.

$\mathcal A$ and $\mathcal F$ are populated automatically at run time — no per-task reward
engineering.

---

## 9. Step 6 — The inverse skill

$$\pi_{\text{inv}} \;=\; \pi_{\text{sym}} \circ \pi_{\text{res}}$$

| Method | Distance (mm) | Success @1 cm |
|---|---|---|
| symbolic prefix only | $16.6 \pm 6.6$ | 10% |
| symbolic + random action | $18.8 \pm 17.2$ | 50% |
| **symbolic + residual RL** | $\mathbf{1.4 \pm 3.2}$ | **90%** |

---

## 10. Hyperparameter reference

**Grounding** — exactly one hyperparameter per predicate. The reward scale is not a parameter:
the reward has no scale of its own (§3.3).

All values below are $\tanh$ scales under the current convention (§3.1); they are twice the
logistic temperatures recorded in pre-2026-06-28 logs.

| Predicate | threshold | $T_p$ (train) | $T_p$ (eval) | $T_p$ relative to threshold |
|---|---|---|---|---|
| `at_pose` | $\tau = 0.01$ m | 0.030 | 0.010 | $3\tau$ |
| `gripper_open` | $w_{\min}=0.04$ m | 0.040 | 0.010 | $w_{\min}$ |
| `tcp_near` | $\delta = 0.05$ m | 0.040 | — | $0.8\,\delta$ |
| `in_region` | region box | 0.010 | — | box half-extent |

Each temperature is set on the order of its predicate's own tolerance: large enough that the
score stays differentiable near the boundary and above sensor noise, small enough that clearly
satisfied or violated states saturate. Tighter temperatures at evaluation make success a
near-Boolean check at the reported tolerance. Class defaults (used when a script does not
override) are `at_pose` 0.04, `tcp_near` 0.04, `in_region` 0.02, `gripper_open` 0.01,
`near` 0.04, `holding` 0.5.

**Decision thresholds**

| Symbol | Value | Where |
|---|---|---|
| $\theta_{\text{pre}}$ | 0.80 | precondition |
| $\theta_\Delta$ / $\Delta_{\min}$ | 0.35 / 0.15 | add & delete effects |
| $\theta$ (fence) | 0.80 | partition (7) — matched to $\theta_{\text{pre}}$ |
| BFS success | 0.98 | on $\mathcal V(s)$ |
| term reachable | 0.90 | `two_phase_inverse` |
| $d_{\max}$ | 3 | BFS depth |

**Regions** (environment-provided): `src` half-extent 1 cm — the restoration tolerance, and the
reason a precision residual exists at all; `goal` half-extent 4 cm — the scripted push only
lands the cube to ~2 cm; $z$ half-extent 5 cm so the vertical axis never dominates the box margin.

---

## 11. Summary of recent changes

1. **Unified grounding.** `PredicateResult.score` computes the $\tanh$ form (1), and $T_p$ was
   redefined as the $\tanh$ scale (all stored temperatures doubled). Score curves verified
   bit-identical; the separate reward scale $c_p$ no longer exists. Caveat: "temperature" now
   means twice what it meant in older logs.
2. **Reward = soft score, by construction.** The per-term reward scale was removed; it is
   removed entirely; the saturated contribution is
   computed as $\sigma_p(2V_p-1)$ directly from the score. One hyperparameter per predicate.
   The map $x\mapsto2x-1$ is forced, not chosen. Active term and `gripper_open` are
   bit-identical to before; two fences change only in their violated regime → retrain to report.
3. **Weighted/unweighted split.** `term_value` (thresholds) vs `term_score` (potential). Fixed
   an empty fence set on real scores; toy results byte-identical.
4. **TCP in the forward model.** Primitives move the modelled end-effector, eliminating phantom
   residuals for TCP-relative predicates.
5. **Derived partition.** $\mathcal F/\mathcal A$ computed from $s_h$, not hardcoded; matches
   the previous hardcoded values, so no retraining.
6. **Validity gate.** Failed prefix executions are re-rolled instead of corrupting the partition.
7. **Distinct predicate keys.** `slot_name="src"` / `"goal"` — the two `at_pose` groundings no
   longer collide on `at_pose(cube,target_pose)`.

## 12. Open items

- **Retrain under the unified reward** (§3.3). The active term is unchanged, but `tcp_near`'s
  fence penalty differs by up to 0.100 in the violated regime, which exploration does reach.
  Requires `stable_baselines3`, which is **not currently installed** in the environment and is
  not declared in `environment.yml`.
- **Threshold inconsistency**: the partition uses $\theta=0.80$ while `two_phase_inverse` still
  defaults `term_reachable_threshold=0.90`. Unify.
- **BFS success threshold 0.98 is unreachable** with realistic soft scores — a cube at a region
  centre scores ~0.88, so $\mathcal V$ saturates below 0.98 and the planner always reports
  `success=False` while still returning the correct prefix. Either lower it or define success
  on $\min_p V^{\pm}_p$ rather than the weighted mean.
- `ToyInversePlanner._state_key` ignores `ee_pose`. Safe today (the TCP deterministically
  follows the object); revisit if a primitive moves the TCP independently.
- Only one task family, and $\mathcal A$ reduces to a single predicate.
