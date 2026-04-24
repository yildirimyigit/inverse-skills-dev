from __future__ import annotations

from dataclasses import dataclass

from inverse_skills.core import SceneGraph
from inverse_skills.logging import ForwardRollout
from inverse_skills.toy.primitives import PrimitiveAction, PrimitiveLibrary


@dataclass
class ToyExecutionResult:
    rollout: ForwardRollout
    action_trace: list[str]


class ToyTabletopSimulator:
    """A minimal deterministic tabletop simulator.

    It is intentionally tiny. The goal is not realism, but to make the PlanRob
    proof of concept stronger by generating rollouts through executable
    primitives instead of hand-assembling only start/end states.
    """

    def __init__(self, primitive_library: PrimitiveLibrary | None = None):
        self.primitives = primitive_library or PrimitiveLibrary()

    def execute(self, skill_name: str, demo_id: str, start_scene: SceneGraph, actions: list[str]) -> ToyExecutionResult:
        current = start_scene.copy(timestep=start_scene.timestep)
        scenes = [current]
        trace: list[str] = []

        for name in actions:
            action = PrimitiveAction(name)
            current = self.primitives.apply(current, action)
            current.metadata = {**current.metadata, "last_action": name}
            scenes.append(current)
            trace.append(name)

        rollout = ForwardRollout(
            skill_name=skill_name,
            demo_id=demo_id,
            scenes=scenes,
            metadata={"action_trace": trace, "generator": "toy_tabletop_simulator_v0"},
        )
        return ToyExecutionResult(rollout=rollout, action_trace=trace)
