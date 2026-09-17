# 10-Minute Presentation Outline

**Paper:** *Inverse Manipulation through Symbolic Planning and Residual Operator Learning*
**Venue:** 14th ICAPS Workshop on Planning and Robotics (PlanRob), oral, Dublin, 2026-06-29
**Audience:** planning + robotics researchers (assume they know STRIPS, RL/SAC, manipulation basics — don't over-explain background)

This is an outline only. Build the slides yourself; each slide below gives **Goal / Say / Show (visual) / Tips**.

---

## Timing at a glance (≈9:40, leaves buffer in a 10-min slot)

| # | Slide | Time | Visual to build |
|---|-------|------|-----------------|
| 1 | Title | 0:20 | Title + authors + teaser strip |
| 2 | Why undo skills? | 0:45 | Motivation scenario graphic |
| 3 | Trajectory- vs task-level inversion (the key idea) | 1:00 | **NEW** 2-row concept diagram (most important visual) |
| 4 | Approach overview | 1:00 | Figure 1 (redrawn) |
| 5 | Operator extraction + soft predicates | 1:00 | Sigmoid eq. + start/end table → STRIPS operator |
| 6 | Inverse target + plan + partition | 1:00 | Inverse-target checklist (3 green / 1 red) |
| 7 | Residual reward design | 1:05 | Figure 2 (two reward curves) |
| 8 | Setup + rollout story | 0:45 | Figure 3 (4-panel) or short video |
| 9 | Results | 0:55 | Table 1 → bar chart |
| 10 | Ablations | 0:40 | Table 2 |
| 11 | Conclusion + future work | 0:45 | Contributions + future bullets |

Rule of thumb: ~1 slide/minute. Resist adding more — 10 min disappears fast. Slides 5+6 can merge and slide 10 can become a backup if you run long (see "If short on time").

---

## Slide 1 — Title (0:20)
- **Goal:** Set the topic in one sentence; look confident, move on fast.
- **Say:** "I'll show how we *undo* a manipulation skill not by rewinding the motion, but by restoring the world state — combining symbolic planning with a small RL residual."
- **Show:** Title, author list, affiliations (CREATE / Università di Napoli Federico II), institutional logos. **Teaser visual:** a thin horizontal 4-frame strip from Figure 3 (initial → after push → after symbolic prefix → after RL) along the bottom. This previews the whole story before you say a word.
- **Tips:** Don't read the title aloud. One spoken sentence, then advance.

## Slide 2 — Why undo skills? (0:45)
- **Goal:** Establish the problem matters and that "undo ≠ rewind."
- **Say:** Human-robot collaboration and assembly/disassembly need to reverse executed skills. The naive idea — rewind the motor trajectory — only works for a subset of skills. The general definition: **restore the world to its pre-forward state.**
- **Show:** A simple before→after motivation graphic. Two options:
  - (a) A relatable icon scene: assembly/disassembly line or a robot returning an object to its spot, with a curved "undo" arrow.
  - (b) Cleaner for this talk: a single arrow `forward skill ⟶  ??? (undo)` with a question mark, teeing up the next slide.
- **Tips:** Keep it to ~2 sentences. This is framing, not content.

## Slide 3 — Trajectory- vs task-level inversion (1:00)  ★ most important conceptual slide
- **Goal:** Deliver the core insight and *pre-empt the obvious objection* ("you said non-prehensile but you grasp it"). This is where you win the audience.
- **Say:** For a pick-and-place, the inverse is another pick-and-place — *in-class inversion*. But a **non-prehensile push cannot be undone by rewinding the trajectory**: the gripper never grasped the cube, so retracting the arm leaves the cube where it was pushed. Restoring the state needs a *different class* of manipulation — a pick-and-place — which we call **inter-class inversion**. That's the gap between trajectory-level and task-level inversion.
- **Show:** **NEW two-row diagram (build this carefully):**
  - **Top row — "Rewind the trajectory" (✗):** cartoon of push (gripper behind cube, cube slides +x to goal); then a reverse arrow; gripper retracts to start but cube **stays at the goal**. Mark with a red ✗. Caption: *"Rewinding fails — the gripper never held the cube."*
  - **Bottom row — "Restore the state" (✓):** start from the pushed state; a *different* action (pick at goal → carry → place at source); cube back at source. Green ✓. Caption: *"Restore the pre-push state with available primitives — here pick-and-place = inter-class inversion."*
  - Label the left edge of each row: **Trajectory-level** / **Task-level**.
- **Tips:** This visual simultaneously motivates the method AND defuses the "motivation vs solution looks contradictory" question a reviewer raised. Say the words "in-class" and "inter-class" out loud — they're your framing anchors.

## Slide 4 — Approach overview (1:00)
- **Goal:** Give the whole pipeline once, so later slides are "zoom-ins."
- **Say:** Walk the pipeline left to right: demonstrations → extract a STRIPS operator via soft predicates → build the inverse target → a BFS planner restores what it can → whatever's left becomes a residual RL problem with an automatically-derived reward → execute symbolic prefix + residual policy.
- **Show:** **Redraw Figure 1** larger and cleaner. Color-code: **symbolic stages in one color** (operator extraction, inverse target, planner) and **the RL stage in a second color** (residual reward, SAC). Keep the boxes: Demos → Operator Extraction → Inverse Target `I(o)=Pre∪Del∪¬Add` → Partition by plan outcome → {Fences F, Active residual A} → Residual RL.
- **Tips:** Use a "you are here" highlight: reuse this same diagram as a small strip on slides 5–7 with the current stage highlighted. Cheap, and it keeps the audience oriented.

## Slide 5 — Operator extraction + soft predicates (1:00)
- **Goal:** How a symbolic operator falls out of demos.
- **Say:** A scene graph gives continuous object/robot states. Each predicate is *soft*: a sigmoid of a signed margin (how far from satisfied). Evaluate predicates at the start and end of each demo; average across demos. High-at-start → precondition; large positive start→end change → add effect; large negative → delete effect.
- **Show:**
  - The soft-predicate equation `V_p(s) = σ(m_p(s)/T_p)` with a tiny inline sketch of the sigmoid and the words "margin > 0 = satisfied."
  - A small **start/end score table** (3–4 predicate rows with start mean, end mean, Δ) and arrows mapping rows to **Pre / Add / Del**.
  - End with the extracted **PUSH operator** box: Pre `{AT_POSE(cube,src), TCP_NEAR(cube), GRIPPER_OPEN()}`, Add `{AT_POSE(cube,goal)}`, Del `{AT_POSE(cube,src)}`.
- **Tips:** Don't read the table cell by cell — point at one row ("this predicate is true before and false after → it's a delete effect") and let the rest land visually.

## Slide 6 — Inverse target, planning, and the partition (1:00)
- **Goal:** Show how the inverse objective is derived and split into "planner-solved" vs "RL-left."
- **Say:** The inverse target is `I(o) = Pre ∪ Del ∪ ¬Add` — preserve preconditions, restore deletes, negate adds. For PUSH that's four predicates. A BFS planner with scripted PICK/PLACE primitives satisfies **three of four**; precise repositioning `AT_POSE(cube,src)` it cannot. Restored predicates become **fences F** (must stay satisfied); the unmet one becomes the **active residual A** for RL.
- **Show:** A **checklist visual** of the four inverse-target predicates:
  - `AT_POSE(cube,src)` → ✗ red → labeled **Active residual (A)**
  - `TCP_NEAR(cube)` → ✓ green → Fence
  - `GRIPPER_OPEN()` → ✓ green → Fence
  - `¬AT_POSE(cube,goal)` → ✓ green → Fence
  - Add a small note: "F and A are populated **automatically** at the handoff state."
- **Tips:** This is the slide that frames the RL job as "one focused, automatically-identified predicate" — honest and clean. Land the phrase "no per-task reward engineering."

## Slide 7 — Residual reward design (1:05)
- **Goal:** Justify the reward shape (your ablation's punchline lives here).
- **Say:** Each term gets a signed, bounded margin `tanh(m/c)`. Two roles: **active** residual predicates use the full bipolar `tanh` (pull toward satisfaction, both directions); **fences** use a one-sided `min(0, tanh)` — silent when satisfied, penalize only when violated. Bounded ⇒ the value function doesn't diverge; one-sided fences ⇒ no incentive to "do nothing."
- **Show:** **Reuse Figure 2** — the two side-by-side reward curves (active: bipolar tanh in [−1,+1]; fence: one-sided in [−1,0] with a "silent zone"). Annotate the "silent zone" and "penalize violations only."
- **Tips:** This is your most technical slide; keep it to the two curves and the two-sentence intuition. The reward equation `R(s)=Σ_A m̃ + Σ_F min(0,m̃)` can sit small in a corner — don't derive it live.

## Slide 8 — Setup + rollout story (0:45)
- **Goal:** Ground everything in the concrete experiment; make it visual.
- **Say:** ManiSkill3 PushCube, Franka Panda, `pd-ee-delta-pos`. Forward skill = scripted push in +x. Inverse = symbolic pick-and-place prefix, then SAC residual on `AT_POSE(cube,src)`. 4-D action (XYZ + gripper), 12-D predicate-grounded observation, curriculum on handoff perturbation.
- **Show:** **Figure 3 as a left-to-right 4-panel story** (initial → after push → after symbolic prefix → after RL), with the on-frame cube-source distances if legible. **Even better: embed a 5–10 s rollout video** (you have `scripts/visualize_pushcube_full_rollout.py` and rendered artifacts) — a moving clip of the inverse executing is the single most persuasive thing in a robotics talk.
- **Tips:** If you use a video, pre-load it and test playback on the venue machine; have the 4-panel static image as fallback.

## Slide 9 — Results (0:55)
- **Goal:** The headline number, clearly.
- **Say:** Symbolic prefix alone leaves ~16.6 mm error (10% @1cm). Adding the residual policy reaches **1.4 mm mean, 90% @1cm** across 10 seeds. A random-action control hits 50% only by chance (note the huge std). A pure trajectory-inverse baseline is **inapplicable** — the forward skill is non-prehensile.
- **Show:** Convert **Table 1 into a bar chart**: x-axis = three methods (Symbolic prefix / Symbolic+random / Symbolic+RL), dual encoding — bars for mean distance (mm) with error bars, and success@1cm labeled on top. Highlight the "ours" bar. Keep the raw table small beneath if you want exact numbers.
- **Tips:** Say the 16.6 → 1.4 mm drop as your one memorable number. Explicitly call out that the trajectory-inverse baseline can't even run — it ties back to slide 3.

## Slide 10 — Ablations (0:40)
- **Goal:** Show the reward design is necessary, not arbitrary.
- **Say:** With plain `tanh` on fences, the agent gets rewarded for staying still (10%). With unbounded margins, the value function explodes and the cube ends >100 mm off (0%). Active + one-sided fences = 90%.
- **Show:** **Table 2** (Reward setting | Success@1cm | Failure mode). Optionally a tiny icon per row (😐 stays still / 💥 diverges / ✓ works).
- **Tips:** This is the most cuttable slide — fold its one sentence into slide 7 if you're over time.

## Slide 11 — Conclusion + future work (0:45)
- **Goal:** Recap the three contributions; show you know the limits.
- **Say:** (1) Skill inversion as a hybrid symbolic-continuous *restoration* problem with inverse targets derived from extracted operators. (2) A residual operator-learning framework where RL fires only when planning falls short. (3) RL rewards derived from operator effects, not task specs. Next: multiple task families for generality; richer active residuals with multiple interacting predicates; learning predicates and primitives instead of hand-specifying them.
- **Show:** Three contribution bullets (left) + future-work bullets (right). Optional closing teaser: loop the rollout video small, or repeat the slide-1 strip. End with title/authors/contact + a "Thank you / Questions" line.
- **Tips:** Land contribution (3) — "reward from operator effects, not task goals" — as your distinctive angle.

---

## If short on time (cut in this order)
1. Drop slide 10 (Ablations) → move its one-liner into slide 7.
2. Merge slides 5 and 6 into one "Method: from demos to residual" slide.
3. Trim slide 2 to a single sentence over slide 3's diagram.

## Visual consistency / design notes
- **Two-color code throughout:** symbolic = color A, RL/residual = color B. Apply it on slides 4, 6, 7, 9 so the "symbolic does the coarse part, RL does the residual" story is visible at a glance.
- **Reuse the pipeline strip** (slide 4) as a small "you-are-here" header on slides 5–7.
- Predicate names in monospace (`AT_POSE`, `GRIPPER_OPEN`) everywhere for consistency with the paper.
- Minimize text: target ≤6 lines/slide; the equations and figures carry the load.
- Pre-render Figures 1–3 and the new diagrams as high-res PNG/PDF (vector if possible) so they're crisp on a projector.

## Assets you already have (reuse, don't recreate)
- Figure 1 (pipeline), Figure 2 (reward curves), Figure 3 (rollout) — from the paper.
- Tables 1 and 2.
- Rollout rendering: `scripts/visualize_pushcube_full_rollout.py` + `artifacts/` (for the slide-8 video/strip).
- New visuals to make from scratch: slide-3 trajectory-vs-state diagram, slide-5 start/end table, slide-6 inverse-target checklist, slide-9 bar chart.

## Anticipated Q&A — prepare backup slides
- **"It's non-prehensile, but your inverse grasps — isn't that contradictory?"** → Backup slide restating in-class vs inter-class: non-prehensility rules out *trajectory* reversal; the inverse is deliberately allowed a different (here prehensile) class. (Slide 3 already plants this.)
- **"The residual is a single predicate — is RL doing much?"** → Backup: yes, it's narrow placement *here*; the framework auto-populates A/F, and richer residuals are the next experiment (slide 11).
- **"Why only one task / no external baselines?"** → Backup: workshop-stage WIP; trajectory-inverse baseline is inapplicable by construction; generality across task families is explicitly future work.
- Keep backups *after* the Thank-you slide.

## Delivery tips
- Rehearse to **9:00** so you have margin; oral sessions run strict and questions are separate.
- Memorize your one number (16.6 → 1.4 mm, 90%) and one phrase ("restore the state, not the trajectory").
- Start the rollout video early/looping so it's playing while you talk — don't click-and-wait.
