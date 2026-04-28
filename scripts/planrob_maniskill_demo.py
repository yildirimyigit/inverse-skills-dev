"""
Demonstrates that the inverse-skill operator extraction pipeline generalises
to real physics-based simulation using ManiSkill3 PickCube-v1.

A scripted top-down grasp oracle picks the cube.  Successful rollouts are fed
into the existing OperatorExtractor, which recovers the same abstract operator
structure as the grasp_hold skill in the toy domain.

The default Panda EEF orientation (pointing downward) aligns naturally with a
top-down approach, so no rotation control is required — only delta-position
actions are used (pd_ee_delta_pos, 4-DOF).

Run:
    python scripts/planrob_maniskill_demo.py

Output:
    artifacts/planrob_maniskill_demo.json
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch
import gymnasium as gym

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import mani_skill.envs  # noqa: F401 — registers environments

from inverse_skills.core import ObjectState, Pose, Region, RobotState, SceneGraph
from inverse_skills.logging import ForwardRollout
from inverse_skills.operators import OperatorExtractor, RestorationObjective
from inverse_skills.operators.toy_planner import ToyInversePlanner
from inverse_skills.predicates import GripperOpenPredicate, HoldingPredicate, InRegionPredicate, PredicateRegistry
from inverse_skills.toy.primitives import PrimitiveLibrary

_IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

# Table surface region: cube rests here before being picked up.
# PickCube randomises cube xy in [-0.1, 0.1]; cube half_size = 0.02 so z_center = 0.02.
_TABLE_LOWER = np.array([-0.15, -0.15, 0.00], dtype=np.float32)
_TABLE_UPPER = np.array([ 0.15,  0.15, 0.05], dtype=np.float32)


def _make_regions() -> dict[str, Region]:
    return {"table_surface": Region("table_surface", _TABLE_LOWER, _TABLE_UPPER)}


def _obs_to_scene(obs: dict, regions: dict[str, Region], timestep: int) -> SceneGraph:
    obj_pose = obs["extra"]["obj_pose"].squeeze().cpu().numpy()
    qpos     = obs["agent"]["qpos"].squeeze().cpu().numpy()
    is_grasp = bool(obs["extra"]["is_grasped"].squeeze().item())
    gripper_width = float(qpos[-2] + qpos[-1])

    return SceneGraph(
        timestep=timestep,
        robot=RobotState(
            q=qpos[:7].astype(np.float32),
            gripper_width=gripper_width,
            holding="cube" if is_grasp else None,
        ),
        objects={"cube": ObjectState(
            name="cube",
            semantic_class="cube",
            pose=Pose(position=obj_pose[:3].astype(np.float32), quat_xyzw=_IDENTITY_QUAT),
        )},
        regions=regions,
        metadata={"skill_name": "pick_cube"},
    )


def _run_pick_oracle(env, regions: dict[str, Region], seed: int) -> tuple[ForwardRollout, bool]:
    """Scripted top-down pick: lower EEF over cube, close gripper, lift."""
    obs, _info = env.reset(seed=seed)
    # ManiSkill leaks is_grasped=True across resets when the previous episode ended while
    # holding an object.  Two open-gripper steps (no translation) clear the flag reliably.
    for _ in range(2):
        obs, *_ = env.step(torch.tensor(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)))
    obj_pos = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy()

    scenes   = [_obs_to_scene(obs, regions, timestep=0)]
    timestep = 1
    success  = False

    def _step_toward(target: np.ndarray, n_steps: int, tol: float = 0.018,
                     gripper_open: float = 1.0) -> None:
        nonlocal obs, timestep, success
        for _ in range(n_steps):
            tcp = obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
            delta = (target - tcp).astype(np.float64)
            norm = float(np.linalg.norm(delta))
            if norm < tol:
                break
            action = np.zeros(4, dtype=np.float32)
            action[:3] = np.clip(delta / max(norm, 1e-6), -1.0, 1.0).astype(np.float32)
            action[3]  = gripper_open
            obs, _r, _done, _trunc, _info = env.step(torch.tensor(action))
            scenes.append(_obs_to_scene(obs, regions, timestep=timestep))
            timestep += 1

    def _close_gripper(n_steps: int = 8) -> None:
        nonlocal obs, timestep, success
        for _ in range(n_steps):
            action = np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float32)
            obs, _r, _done, _trunc, info = env.step(torch.tensor(action))
            scenes.append(_obs_to_scene(obs, regions, timestep=timestep))
            timestep += 1
            raw = info["success"]
            success = bool(raw.item() if hasattr(raw, "item") else raw)

    # Phase 1: move above cube
    above = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.08], dtype=np.float32)
    _step_toward(above, n_steps=20, tol=0.018)

    # Phase 2: descend to grasp height
    grasp_pos = np.array([obj_pos[0], obj_pos[1], obj_pos[2]], dtype=np.float32)
    _step_toward(grasp_pos, n_steps=20, tol=0.012)

    # Phase 3: close gripper
    _close_gripper(n_steps=10)

    # Phase 4: lift
    lift_target = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.15], dtype=np.float32)
    _step_toward(lift_target, n_steps=15, tol=0.02, gripper_open=-1.0)  # keep gripper closed

    raw = obs["extra"]["is_grasped"].squeeze().item()
    success = bool(raw)

    rollout = ForwardRollout(
        skill_name="pick_cube",
        demo_id=f"maniskill_pick_{seed:03d}",
        scenes=scenes,
        metadata={"success": success, "generator": "maniskill_pick_oracle_v0", "seed": seed},
    )
    return rollout, success


def main() -> None:
    seeds   = [0, 1, 2, 3, 4]
    regions = _make_regions()
    rollouts: list[ForwardRollout] = []
    results_per_seed: list[dict]   = []

    print("Running PickCube-v1 forward rollouts (top-down grasp oracle)...")
    env = gym.make("PickCube-v1", obs_mode="state_dict",
                   control_mode="pd_ee_delta_pos", render_mode=None,
                   max_episode_steps=80)

    for seed in seeds:
        rollout, ok = _run_pick_oracle(env, regions, seed)
        cube_start = rollout.first().objects["cube"].pose.position.tolist()
        cube_end   = rollout.last().objects["cube"].pose.position.tolist()
        holding_end = rollout.last().robot.holding
        print(f"  seed {seed}: {'OK' if ok else 'FAIL'}  "
              f"scenes={len(rollout.scenes)}  "
              f"cube z: {cube_start[2]:.3f} -> {cube_end[2]:.3f}  "
              f"holding={holding_end}")
        results_per_seed.append({"seed": seed, "success": ok,
                                  "n_scenes": len(rollout.scenes),
                                  "cube_start_z": cube_start[2],
                                  "cube_end_z": cube_end[2]})
        if ok:
            rollouts.append(rollout)

    env.close()
    n_ok = len(rollouts)
    print(f"\n{n_ok}/{len(seeds)} rollouts succeeded.")

    if n_ok == 0:
        print("No successful rollouts — operator extraction skipped.")
        return

    # ── operator extraction ────────────────────────────────────────────────────
    registry = PredicateRegistry([
        InRegionPredicate("cube", "table_surface"),
        GripperOpenPredicate(min_width=0.04),
        HoldingPredicate("cube"),
    ])

    extraction = OperatorExtractor(registry).extract("pick_cube", rollouts)
    operator   = extraction.operator

    print("\nExtracted forward operator (ManiSkill3 physics, PickCube-v1):")
    print(f"  Preconditions : {[t.key for t in operator.preconditions]}")
    print(f"  Add effects   : {[t.key for t in operator.add_effects]}")
    print(f"  Delete effects: {[t.key for t in operator.delete_effects]}")

    print("\nSynthesised inverse target:")
    for t in operator.inverse_target_terms():
        print(f"  [{t.polarity:8s}] {t.key}  (w={t.weight:.4f})")

    print("\nPredicate scores at forward final state:")
    fwd_final = rollouts[0].last()
    for key in sorted(registry._predicates):
        r = registry.get(key).evaluate(fwd_final)
        print(f"  {key}: score={r.score:.4f}  truth={r.truth}")

    print("\nComparison: toy grasp_hold operator expected structure:")
    print("  Preconditions : ['gripper_open()', 'in_region(cube,source)']")
    print("  Add effects   : ['holding(cube)']")
    print("  Delete effects: ['gripper_open()']")

    # ── save results ───────────────────────────────────────────────────────────
    out = {
        "env": "PickCube-v1 (ManiSkill3 3.0.1)",
        "skill": "pick_cube",
        "note": "Maps to grasp_hold in toy domain: pre={gripper_open,in_region}, add={holding}, del={gripper_open}",
        "control_mode": "pd_ee_delta_pos",
        "seeds_attempted": seeds,
        "n_succeeded": n_ok,
        "rollouts": results_per_seed,
        "table_region": {"lower": _TABLE_LOWER.tolist(), "upper": _TABLE_UPPER.tolist()},
        "operator": operator.to_dict(),
        "predicate_scores_at_fwd_final": {
            key: {"score": float(registry.get(key).evaluate(rollouts[0].last()).score),
                  "truth": bool(registry.get(key).evaluate(rollouts[0].last()).truth)}
            for key in sorted(registry._predicates)
        },
        "extractor_scores": extraction.scores,
    }

    out_path = Path("artifacts/planrob_maniskill_demo.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
