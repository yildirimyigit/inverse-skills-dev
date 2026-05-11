# inverse-skills-dev

Development repository for the inverse skill learning.

Example:

<img width="512" height="512" alt="res_low" src="https://github.com/user-attachments/assets/1a62989e-6a55-45ab-9f3b-11b35763c8ab" />


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
