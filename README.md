# inverse-skills-dev

Development repository for the inverse skill learning.

Example:

<img width="512" height="512" alt="output-onlinegiftools" src="https://github.com/user-attachments/assets/d0ad7dae-24d8-4f1e-a328-924bf90e54dc" />


## Current environment status

- development
- simulation
- training
- logging
- evaluation

A separate real-robot FR3 runtime stack will be added later.

## Repository layout

```text
inverse-skills-dev/
├── .devcontainer/
│   └── devcontainer.json
├── docker/
│   └── Dockerfile
├── configs/
├── docs/
├── experiments/
├── notebooks/
├── scripts/
├── src/
│   └── inverse_skills/
├── artifacts/
├── checkpoints/
├── data/
├── logs/
├── docker-compose.yml
├── environment.yml
├── pyproject.toml
└── README.md
```

## Getting started

### Host prerequisites

The container runs the simulator on the GPU, so these are required, not optional:

- an NVIDIA GPU with driver 525 or newer (`nvidia-smi` to check) — the pinned
  PyTorch is a CUDA 12.8 build;
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed on the host, and a CDI spec if your Docker uses one
  (`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`, regenerate after driver updates).
  `docker-compose.yml` reserves an NVIDIA device, so the container will not start without it.
- Docker Compose v2.

Hiding the GPU does not work as a fallback: ManiSkill renders through
`sapien_cuda` even though physics runs on CPU, and drops into a CPU renderer
that crashes. Only the unit tests and the planner demo run without a GPU.

If your user id is not 1000, export it before building so that files written
into the bind-mounted working directory belong to you:

```bash
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
```

### 1. Build the container

```bash
docker compose build
```

### 2. Run the container

```bash
./scripts/run_dev_container.sh
```

### 3. Attach a new shell to the container

```bash
./scripts/attach_dev_container.sh
```

### 4. Verify Python import

```bash
python -c "import torch; import mani_skill; print('ok')"
```

### 5. Run the Symbolic Inversion + SAC training

```bash
micromamba run -n inverse-skills python scripts/planrob_inverse_rl_pushcube_full_demo_2d_action.py
```

## The stow pipeline (planner + learned operator)

The `StowCube-v1` domain demonstrates planning with an incomplete action
library: a forward skill wedges cube A in a pocket and blocks it with cube B,
and no library primitive can free A again. The planner delegates that one
predicate to a *hole*, RL learns a policy for it, and the executive runs the
mixed plan. Run the stages in order — each writes what the next one reads.

The extracted operator and a trained policy are committed under
`artifacts/stow/`, so stages 2 and 4 work immediately on a fresh clone. Stages
1 and 3 regenerate them. They are the earlier, positive-only results, kept as
published; the current code reads the committed operators with their meaning
unchanged (a test checks it). The results of the current code are
in `artifacts/signed_literals_20261111/` (see below).

### Signed literals

Every condition is a signed literal, `p` or `not p`. A predicate's score V, in
[0, 1], is read in two ways:

- **Boolean**, for the planner's symbolic state: p is true when V ≥ 0.5.
- **Robust**, for everything checked physically: extraction, preconditions and
  effects during execution, RL success, fences, handoffs and the final goal.
  `p` holds when V ≥ 0.8, and `not p` holds when V ≤ 0.2. A score in between
  holds neither way, and is reported as uncertain.

The restoration goal is the forward operator's preconditions, of both signs.
Every effect's complement is among them. For StowCube the goal now includes
`not tcp_near(cube_a)` and `not tcp_near(cube_b)`: the hand must end away from
both cubes. The inverse therefore ends with `retreat_to_safe`, a predefined
library action, not a learned skill. It is the existing rise, with the gripper
kept as it is. It was measured in simulation before it was registered
(`scripts/stow_retreat_gate.py`), and its precondition `gripper_open()` comes
from that measurement: with the gripper closed on cube A, releasing the press
moves A.

`artifacts/signed_literals_20261111/` holds both tasks run fresh with these
rules, from Stage 0 to the final evaluation. `REPORT.md` there gives the
results, and `run.py` reruns them stage by stage.

| # | Command | Time | Writes |
|---|---------|------|--------|
| 0 | `python scripts/stow_stage0_gate.py` | ~4 min | `artifacts/stow/stage0_gate.json` |
| 1 | `python scripts/stow_stage1_extract.py` | ~1 min | `artifacts/stow/stage1_operator.json` |
| 2 | `python scripts/stow_stage2_plan.py` | ~1 min | `artifacts/stow/stage2_plan.json`, `artifacts/stow/hole_spec.json` |
| 3 | `python scripts/stow_stage3_train.py --timesteps 120000` | ~17 min (GPU) | `artifacts/stow/stage3/<run id>/` |
| 4 | `python scripts/stow_stage4_execute.py artifacts/stow/stage3/1790788534/sac_hole_best` | ~15 min | `artifacts/stow/stage4_execution.json`, `artifacts/stow/learned_operator.json` |

Saved results are never overwritten: a stage stops if its output file exists. To
run a stage again, give it new output paths with `--out` (and `--spec-out` or
`--learned-out` where it has them), and point later stages at them with
`--operator` and `--spec`.

Operators are extracted with `min_max`, which asks every demonstration to show each
condition and transition, and plans are made by the `exact` mode, which judges holes
by real transitions. `mean` and `relaxed` stay available by naming them
(`OperatorExtractionConfig(mode="mean")`, `plan(..., mode="relaxed")`), for comparisons
and for the direct baseline's `unrestricted_holes`, which only the relaxed mode has.
`artifacts/default_mode_validation/` holds the comparison of the two configurations.

### How the planner chooses holes

The `exact` mode (`src/inverse_skills/operators/exact_hole_planner.py`) implements the
"Minimal Hole Planner" formulation. With k = `max_delegated`, the learning budget:

- **Candidates** are the scored literals, of either sign, that are goal literals or
  preconditions of some library action. Only these can become holes. A hole for `p`
  makes p true; a hole for `not p` makes p false. A hole can run only while its
  literal does not hold.
- **Feasible(H)**: the library and holes for H reach the goal, in any order and at any
  length, decided by exhaustive search of the real transitions.
- **Admissible**: H has at most k literals, is feasible, no literal can be dropped
  (subset-minimal), and no literal l has a *replacement*: a set of at most k literals,
  drawn from H's other literals and the candidates upstream of l, that is feasible.
  u is upstream of l when a library action that establishes l needs u, directly or
  through a chain of such actions, and nothing leads back from u to l.
- **Plan**: among the admissible sets of the smallest size, the plan with the fewest
  displaced literals, then the fewest steps. There is no length limit. If no set of
  size up to k is admissible, the answer is no plan.

What this guarantees, and what it does not:

- The library operators are assumed complete and correct under the modelled
  transition, which applies each action's own effects together with the domain's
  world constraints (implications, mutual exclusions, disturbances).
- A plan's holes are subset-minimal: no hole can be dropped. They are not always the
  fewest possible: the preference can choose two upstream holes over one, within k.
- "No plan" is exact relative to the candidate set, the budget and the symbolic
  transition model: it means no set of at most k candidates reaches the goal. The
  report says whether holes for every candidate would.
- The upstream preference reads preconditions of both signs. An action establishes
  `not p` by deleting p, by making true a literal that excludes p, or by triggering
  a disturbance that displaces p. It is not a complete causal analysis, and it does
  not show that the preferred skill is easier to learn; the Drawer's
  targeted-versus-direct experiment is its evidence.

Each stage prints `PASS` or `FAIL` against its own checks. What they verify:

- **Stage 0** — the scene behaves as designed: the forward skill seats both
  cubes, `pick(A)` is blocked while A is wedged, and a scripted press-and-drag
  frees it. Run this first on a new machine; it is the physics gate.
- **Stage 1** — the signed operator extracted from 5 demonstrations has the
  designed effects and no robot-relative effect, and every effect's complement is
  a precondition, so the restoration goal equals the preconditions. Robot-relative
  facts such as `not tcp_near(cube_a)` appear only as preconditions.
- **Stage 2** — the planner, from measured predicate scores, produces
  `pick B → place B → approach A → HOLE[clear_of_walls(A)] → pick A → place A
  → retreat_to_safe` with no equally cheap alternative, and finds no plan at all
  when holes are disallowed. Everything about the hole is derived, not given: which predicate
  to delegate (an admissible set, as above: one the goal cannot be reached without
  by real transitions, that has a soft score, and that no learnable literal upstream
  of it could replace within the budget), where the learned
  skill starts (the robot-relative predicates the library can establish), and
  what it must preserve (the causal links spanning it, of both signs). These are written to
  `hole_spec.json`, the learning problem Stage 3 trains against.
- **Stage 3** — SAC learns the hole's policy against the spec; the best
  checkpoint is selected on the operator's contract (postcondition established
  *and* fences intact).
- **Stage 4** — the learned skill is first modelled from its own executions by
  the same extractor that modelled the forward skill, and joins the library. Each
  execution must start at the handoff the plan predicts, checked before the policy
  runs (`operators/handoff.py`); a scene that fails is replaced by the next and
  recorded, and what the policy then does never decides whether a run is kept.
  The executive then runs the full plan and the ablations — library only, the
  ordering a planner without the interference model cannot rule out, and the
  re-planned inverse using the learned operator — checking every step's
  preconditions and modelled effects robustly, and reporting wherever the
  measured scene diverges from the domain's prediction.

The planner itself contains no domain vocabulary (a test enforces it). The
domain supplies facts about its world — which literals imply or exclude each
other, and which changes displace other objects (moving a cube into or out of
the pocket sweeps the mouth) — and the plan's ordering follows from them.

Supporting scripts:

```bash
python scripts/stow_planner_demo.py          # planner on a complete vs incomplete library (no GPU)
python scripts/stow_pick_limit.py            # measures the grasp limit that calibrates X_CLEAR
python scripts/stow_stage3_precondition.py artifacts/stow/stage3/1790788534/sac_hole_best
python -m pytest -q                          # no GPU needed
```

Every stage has a matching `stow_stage<N>_video.py` that renders an annotated
mp4 into `artifacts/stow/videos/` (not committed — regenerate as needed).
`scripts/stow_framework_video.py` renders the pipeline end to end without the
ablations.

Historical reference results, before signed literals (positive-only goal, one
0.8 threshold for every check), run `1790788534`, 10 evaluation seeds:

| condition | goal met | A error | B error |
|---|---|---|---|
| full: planner + learned skill | 10/10 | 1.5 mm | 0.8 mm |
| library only | 0/10 — refused at `pick(A)`, precondition unmet | | |
| without the interference model | 3/10 | 51 mm | 89 mm |
| re-planned with the learned operator | 9/10 | | |

In the full plan the domain's prediction matches the measured scene after
every step except on one seed, where the forward push jams B against A and
lifting B drags A partly out of the pocket — an interaction the model does not
contain, which the executive reports at the step that causes it. That same
seed is the re-planned condition's one failure: A ends outside the learned
operator's measured precondition, and the executive declines to run it there.
Two seeds' forward pushes fall short of the stowed state and are re-rolled,
since an inverse is only defined once the forward skill has happened.

Handoff validity is reported apart from these results: 9 of the 10 evaluation
seeds give a valid handoff (not the seed above), and 10 of the 11 scenes that
reached the handoff while the learned skill's demonstrations were collected.

The signed-literal rerun of both tasks (2026-11-11), with the restoration goal of
both signs and robust checks, is reported in
`artifacts/signed_literals_20261111/REPORT.md`. Two protocol questions it raised
are open: which Stow checkpoint the selection rule should pick, and when the final
evaluation should stop the direct baseline's policy. Its numbers are not final
until they are settled.
