from __future__ import annotations

from inverse_skills.logging import ForwardRollout
from inverse_skills.toy.domains import make_scene, make_scene_named, make_scene_named_with_distractor
from inverse_skills.toy.simulator import ToyTabletopSimulator
from inverse_skills.toy.primitives import PrimitiveLibrary


def make_pick_place_rollouts_executable(num_rollouts: int = 3) -> list[ForwardRollout]:
    return make_pick_place_rollouts_executable_named(
        object_name="cube",
        source_name="source",
        target_name="target",
        skill_name="pick_place",
        num_rollouts=num_rollouts,
    )


def make_push_rollouts_executable(num_rollouts: int = 3) -> list[ForwardRollout]:
    return make_push_rollouts_executable_named(
        object_name="cube",
        source_name="source",
        target_name="target",
        skill_name="push_to_target",
        num_rollouts=num_rollouts,
    )


def make_pick_place_rollouts_executable_named(
    *,
    object_name: str,
    source_name: str,
    target_name: str,
    skill_name: str = "pick_place",
    num_rollouts: int = 3,
) -> list[ForwardRollout]:
    sim = ToyTabletopSimulator(PrimitiveLibrary(object_name, source_name, target_name))
    rollouts: list[ForwardRollout] = []
    for i in range(num_rollouts):
        y = 0.02 * i
        start = make_scene_named(0, object_name, [0.0, y, 0.02], source_name=source_name, target_name=target_name, skill_name=skill_name)
        result = sim.execute(
            skill_name=skill_name,
            demo_id=f"pick_exec_{i:03d}",
            start_scene=start,
            actions=[f"pick({object_name})", f"place({target_name})"],
        )
        rollouts.append(result.rollout)
    return rollouts


def make_push_rollouts_executable_named(
    *,
    object_name: str,
    source_name: str,
    target_name: str,
    skill_name: str = "push_to_target",
    num_rollouts: int = 3,
) -> list[ForwardRollout]:
    sim = ToyTabletopSimulator(PrimitiveLibrary(object_name, source_name, target_name))
    rollouts: list[ForwardRollout] = []
    for i in range(num_rollouts):
        y = 0.015 * i
        start = make_scene_named(0, object_name, [0.0, y, 0.02], source_name=source_name, target_name=target_name, skill_name=skill_name)
        result = sim.execute(
            skill_name=skill_name,
            demo_id=f"push_exec_{i:03d}",
            start_scene=start,
            actions=[f"push({target_name})"],
        )
        rollouts.append(result.rollout)
    return rollouts


def make_grasp_hold_rollouts_executable(num_rollouts: int = 3) -> list[ForwardRollout]:
    return make_grasp_hold_rollouts_executable_named(
        object_name="cube",
        source_name="source",
        target_name="target",
        skill_name="grasp_hold",
        num_rollouts=num_rollouts,
    )


def make_grasp_hold_rollouts_executable_named(
    *,
    object_name: str,
    source_name: str,
    target_name: str,
    skill_name: str = "grasp_hold",
    num_rollouts: int = 3,
) -> list[ForwardRollout]:
    sim = ToyTabletopSimulator(PrimitiveLibrary(object_name, source_name, target_name))
    rollouts: list[ForwardRollout] = []
    for i in range(num_rollouts):
        y = 0.02 * i
        start = make_scene_named(0, object_name, [0.0, y, 0.02], source_name=source_name, target_name=target_name, skill_name=skill_name)
        result = sim.execute(
            skill_name=skill_name,
            demo_id=f"grasp_hold_exec_{i:03d}",
            start_scene=start,
            actions=[f"pick({object_name})"],
        )
        rollouts.append(result.rollout)
    return rollouts


def make_push_rollouts_executable_named_with_distractor(
    *,
    object_name: str,
    source_name: str,
    target_name: str,
    distractor_name: str,
    skill_name: str = "push_restore",
    num_rollouts: int = 3,
) -> list[ForwardRollout]:
    sim = ToyTabletopSimulator(PrimitiveLibrary(object_name, source_name, target_name))
    rollouts: list[ForwardRollout] = []
    for i in range(num_rollouts):
        y = 0.015 * i
        start = make_scene_named_with_distractor(
            0,
            object_name,
            [0.0, y, 0.02],
            distractor_name=distractor_name,
            distractor_position=[0.20, 0.20, 0.02],
            source_name=source_name,
            target_name=target_name,
            skill_name=skill_name,
        )
        result = sim.execute(
            skill_name=skill_name,
            demo_id=f"push_exec_distractor_{i:03d}",
            start_scene=start,
            actions=[f"push({target_name})"],
        )
        rollouts.append(result.rollout)
    return rollouts
