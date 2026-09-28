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
1 and 3 regenerate them.

| # | Command | Time | Writes |
|---|---------|------|--------|
| 0 | `python scripts/stow_stage0_gate.py` | ~4 min | `artifacts/stow/stage0_gate.json` |
| 1 | `python scripts/stow_stage1_extract.py` | ~1 min | `artifacts/stow/stage1_operator.json` |
| 2 | `python scripts/stow_stage2_plan.py` | ~1 min | `artifacts/stow/stage2_plan.json` |
| 3 | `python scripts/stow_stage3_train.py --timesteps 120000` | ~15 min (GPU) | `artifacts/stow/stage3/<run id>/` |
| 4 | `python scripts/stow_stage4_execute.py artifacts/stow/stage3/1789749141/sac_hole_best` | ~8 min | `artifacts/stow/stage4_execution.json` |

Each stage prints `PASS` or `FAIL` against its own checks. What they verify:

- **Stage 0** — the scene behaves as designed: the forward skill seats both
  cubes, `pick(A)` is blocked while A is wedged, and a scripted press-and-drag
  frees it. Run this first on a new machine; it is the physics gate.
- **Stage 1** — the STRIPS operator extracted from 5 demonstrations matches the
  scene's intended meaning, with no spurious `tcp_near` terms.
- **Stage 2** — the planner, from measured predicate scores, produces
  `pick B → place B → approach A → HOLE[clear_of_walls(A)] → pick A → place A`,
  with the hole mid-plan, and finds no plan at all when holes are disallowed.
- **Stage 3** — SAC learns the hole's policy; the best checkpoint is selected on
  the operator's contract (postcondition established *and* fences intact).
- **Stage 4** — the executive runs the full plan on the robot and the ablations
  beside it.

Supporting scripts:

```bash
python scripts/stow_planner_demo.py          # planner on a complete vs incomplete library (no GPU)
python scripts/stow_pick_limit.py            # measures the grasp limit that calibrates X_CLEAR
python scripts/stow_stage3_precondition.py artifacts/stow/stage3/1789749141/sac_hole_best
python -m pytest -q                          # 27 tests, no GPU needed
```

Every stage has a matching `stow_stage<N>_video.py` that renders an annotated
mp4 into `artifacts/stow/videos/` (not committed — regenerate as needed).
`scripts/stow_framework_video.py` renders the pipeline end to end without the
ablations.

Reference results, all on 5 seeds: the full pipeline restores both cubes to
about 2 mm on 5/5; the library-only ablation stops at `pick(A)` with its
precondition unmet on 5/5.
