# inverse-skills-dev

Minimal development repository for the simplest publishable version of inverse skill learning.

## Current scope

The active project goal is to build the simplest publishable version first:
- inverse skills are modeled as restoration of forward preconditions and effects
- not as trajectory reversal
- no paper writing until the minimal version is implemented and tested
- environment should be reproducible across machines

## First environment target

This first setup is intentionally focused on:
- development
- simulation
- training
- logging
- evaluation

A separate real-robot FR3 runtime stack can be added later.

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

### 1. Build the container

```bash
docker compose build
```

### 2. Open a shell in the container

```bash
docker compose run --rm dev
```

### 3. Verify Python import

```bash
python -c "import torch; import mani_skill; print('ok')"
```

## Notes

- The container expects NVIDIA Docker support on the host if you want GPU access.
- The first simulation target is ManiSkill to keep the environment lightweight and reproducible.
- RLBench or ROS 2 should be isolated later if needed.
