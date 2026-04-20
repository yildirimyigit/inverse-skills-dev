# Minimal internal package structure

This patch adds the first simulator-agnostic core for the simplest publishable version:

- scene graph state
- geometric predicate margins
- forward rollout logging
- forward operator extraction
- inverse restoration objective

## Copy into repo

From your repo root:

```bash
cp -r /path/to/inverse-skills-minimal-structure/src .
cp -r /path/to/inverse-skills-minimal-structure/scripts .
cp -r /path/to/inverse-skills-minimal-structure/tests .
```

## Smoke test

Inside the container:

```bash
python scripts/smoke_operator_extraction.py
pytest -q
```
