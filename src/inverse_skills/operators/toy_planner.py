from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from inverse_skills.core import SceneGraph
from inverse_skills.operators.restoration import RestorationObjective
from inverse_skills.toy.primitives import PrimitiveAction, PrimitiveLibrary


@dataclass
class PlanResult:
    success: bool
    actions: list[str]
    final_potential: float
    expanded_nodes: int


class ToyInversePlanner:
    def __init__(self, primitives: PrimitiveLibrary, objective: RestorationObjective, success_threshold: float = 0.98):
        self.primitives = primitives
        self.objective = objective
        self.success_threshold = success_threshold

    def plan(self, start_scene: SceneGraph, max_depth: int = 3) -> PlanResult:
        start_score = self.objective.potential(start_scene)
        if start_score >= self.success_threshold:
            return PlanResult(True, [], start_score, expanded_nodes=0)

        frontier = deque([(start_scene, [])])
        visited = {self._state_key(start_scene)}
        expanded = 0
        best_score = start_score
        best_actions: list[str] = []

        while frontier:
            scene, actions = frontier.popleft()
            expanded += 1
            current_score = self.objective.potential(scene)
            if current_score > best_score:
                best_score = current_score
                best_actions = [str(a) for a in actions]
            if current_score >= self.success_threshold:
                return PlanResult(True, [str(a) for a in actions], current_score, expanded_nodes=expanded)
            if len(actions) >= max_depth:
                continue

            for action in self.primitives.available_actions():
                next_scene = self.primitives.apply(scene, action)
                state_key = self._state_key(next_scene)
                if state_key in visited:
                    continue
                visited.add(state_key)
                frontier.append((next_scene, actions + [action]))

        return PlanResult(False, best_actions, best_score, expanded_nodes=expanded)

    def _state_key(self, scene: SceneGraph) -> tuple:
        obj_pos = tuple(
            round(v, 3)
            for v in scene.get_object(self.primitives.object_name).pose.position.tolist()
        )
        return (obj_pos, scene.robot.holding, round(scene.robot.gripper_width, 3))
