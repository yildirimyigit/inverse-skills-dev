# PlanRob Short-Paper Starter

This starter narrows the project to a planning-centric proof of concept:

1. represent scenes with an object-centric state,
2. evaluate soft predicates,
3. extract a forward operator from successful rollouts,
4. synthesize the inverse target as restoration of forward preconditions and undoing of forward effects,
5. plan a short inverse action sequence in a toy domain.

## Suggested order

Copy `src/`, `scripts/`, and `tests/` into your repo, then run:

```bash
python scripts/smoke_operator_extraction.py
python scripts/planrob_toy_demo.py
pytest -q tests/test_operator_extraction.py tests/test_toy_inverse_planner.py
```

## Expected result

- `smoke_operator_extraction.py` should learn a pick-place operator.
- `planrob_toy_demo.py` should show that a forward `push_to_target` skill can be inverted with the plan:
  - `pick(cube)`
  - `place(source)`

That gives you a minimal proof that inverse skills can be represented as operator restoration rather than trajectory reversal.
