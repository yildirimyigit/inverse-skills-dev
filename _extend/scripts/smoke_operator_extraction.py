from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inverse_skills.core import ObjectState, Pose, Region, RobotState, SceneGraph
from inverse_skills.logging import ForwardRollout
from inverse_skills.operators import OperatorExtractor, RestorationObjective
from inverse_skills.predicates import InRegionPredicate, PredicateRegistry


def make_scene(timestep: int, object_position: list[float]) -> SceneGraph:
    source = Region.from_bounds("source", lower=[-0.15, -0.15, -0.05], upper=[0.15, 0.15, 0.10])
    target = Region.from_bounds("target", lower=[0.35, -0.15, -0.05], upper=[0.65, 0.15, 0.10])
    cube = ObjectState(
        name="cube",
        semantic_class="box",
        pose=Pose(position=np.asarray(object_position, dtype=np.float32), quat_xyzw=np.array([0, 0, 0, 1], dtype=np.float32)),
    )
    robot = RobotState(q=np.zeros(7, dtype=np.float32), gripper_width=0.08)
    return SceneGraph(timestep=timestep, robot=robot, objects={"cube": cube}, regions={"source": source, "target": target})


def main() -> None:
    rollouts = []
    for i in range(3):
        start = make_scene(0, [0.0, 0.02 * i, 0.02])
        end = make_scene(50, [0.50, 0.02 * i, 0.02])
        rollouts.append(ForwardRollout(skill_name="pick_place", demo_id=f"demo_{i:03d}", scenes=[start, end]))

    registry = PredicateRegistry([
        InRegionPredicate("cube", "source"),
        InRegionPredicate("cube", "target"),
    ])

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
