from __future__ import annotations

import json
from pathlib import Path

from inverse_skills.operators import OperatorExtractor, RestorationObjective
from inverse_skills.toy import build_predicate_registry, make_pick_place_rollouts


def main() -> None:
    rollouts = make_pick_place_rollouts(num_rollouts=3)
    registry = build_predicate_registry()
    result = OperatorExtractor(registry).extract("pick_place", rollouts)
    objective = RestorationObjective(result.operator, registry)

    print("Learned operator:")
    print(json.dumps(result.operator.to_dict(), indent=2))
    print("\nPredicate scores:")
    print(json.dumps(result.scores, indent=2))
    print("\nInverse restoration potential:")
    print(f"  at forward final state: {objective.potential(rollouts[0].last()):.3f}")
    print(f"  at restored source state: {objective.potential(rollouts[0].first()):.3f}")

    out = Path("artifacts/smoke_operator.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.operator.to_dict(), indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
