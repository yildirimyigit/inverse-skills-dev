from __future__ import annotations

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
        pose=Pose(position=np.asarray(object_position, dtype=np.float32), quat_xyzw=np.array([0, 0, 0, 1], dtype=np.float32)),
    )
    robot = RobotState(q=np.zeros(7, dtype=np.float32), gripper_width=0.08)
    return SceneGraph(timestep=timestep, robot=robot, objects={"cube": cube}, regions={"source": source, "target": target})


def test_operator_extraction_for_pick_place_transition() -> None:
    rollouts = [
        ForwardRollout("pick_place", "demo_0", [make_scene(0, [0, 0, 0.02]), make_scene(50, [0.5, 0, 0.02])]),
        ForwardRollout("pick_place", "demo_1", [make_scene(0, [0, 0.02, 0.02]), make_scene(50, [0.5, 0.02, 0.02])]),
    ]
    registry = PredicateRegistry([
        InRegionPredicate("cube", "source"),
        InRegionPredicate("cube", "target"),
    ])

    result = OperatorExtractor(registry).extract("pick_place", rollouts)
    op = result.operator

    pre = {term.key for term in op.preconditions}
    add = {term.key for term in op.add_effects}
    delete = {term.key for term in op.delete_effects}

    assert "in_region(cube,source)" in pre
    assert "in_region(cube,target)" in add
    assert "in_region(cube,source)" in delete

    objective = RestorationObjective(op, registry)
    assert objective.potential(rollouts[0].first()) > objective.potential(rollouts[0].last())
