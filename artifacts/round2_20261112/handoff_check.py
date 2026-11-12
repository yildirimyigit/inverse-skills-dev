"""R0: does the Stow validation start exactly where Stage 4 starts the learned skill?

For each scene, two runs in two separate simulators:
  Stage 4    its own functions: `forward_once` and `run_prefix` from
             scripts/stow_stage4_execute.py, then `stow_executive.run_learned`, which
             stops at the first state where the target robustly holds
  validation `StowHoleEnv(exact_handoff=True).reset(seed)`, the start every validation
             and test episode gets (scripts/stow_stage3_train.py), then the same actions

compared bitwise (PLAN.md section 1.4):
  - the whole saved simulator state at the handoff: both cubes' poses and velocities, and
    the robot's joint positions and velocities, both gripper fingers included
  - the gripper opening (the two finger joints)
  - the contact forces between each finger and cube A
  - the trajectory under the same actions up to the stopping point: the whole saved state
    after every step, the stopping step, and the target's and fences' scores there

The actions are the scripted reference drag, so no learned policy runs on these scenes.
Exit code 1 if any scene differs.

    PYTHONNOUSERSITE=1 python artifacts/round2_20261112/handoff_check.py --spec <stow hole spec> --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import stow_stage3_train as stage3          # noqa: E402  (scripts import each other by name)
import stow_stage4_execute as stage4        # noqa: E402

from inverse_skills.envs import stow_executive as ex                      # noqa: E402
from inverse_skills.envs import stow_predicates as sp                     # noqa: E402
from inverse_skills.envs import stow_primitives as prim                   # noqa: E402
from inverse_skills.envs.stow_hole_env import StowHoleEnv, load_spec      # noqa: E402
from inverse_skills.operators.hole_planner import Action, spec_fences, spec_target  # noqa: E402

torch.set_num_threads(1)


class Drag:
    """The scripted reference drag as a policy: the same action at every step."""

    def predict(self, obs, deterministic=True):
        return stage3._DRAG, None


def flat_state(env) -> dict[str, np.ndarray]:
    """The whole saved simulator state, one array per actor and articulation."""
    out = {}
    for kind, entries in env.unwrapped.get_state_dict().items():
        for name, value in entries.items():
            out[f"{kind}/{name}"] = value.detach().cpu().numpy().copy()
    return out


def contacts(env) -> np.ndarray:
    u = env.unwrapped
    return np.stack([u.scene.get_pairwise_contact_forces(link, u.cube_a).detach().cpu().numpy().reshape(-1)
                     for link in (u.agent.finger1_link, u.agent.finger2_link)])


def fingers(env) -> np.ndarray:
    return env.unwrapped.agent.robot.get_qpos().detach().cpu().numpy().reshape(-1)[-2:].copy()


def same(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list[str]:
    """The keys whose arrays differ (bitwise)."""
    return sorted(k for k in a.keys() | b.keys() if k not in a or k not in b or not np.array_equal(a[k], b[k]))


def compare(seed: int, spec: dict, env4, envv: StowHoleEnv) -> dict:
    """Both runs for one validation seed. `env4` is a plain simulator, `envv` the validation environment."""
    target, fences = spec_target(spec), spec_fences(spec)
    contract = (target, *sorted(fences))
    prefix = [Action.from_dict(a) for a in spec["prefix"]]
    registry = sp.registry()

    obs_v, start = envv.reset(seed=seed)
    scene = seed + (start["prefix_attempts"] - 1) * 10_000          # the scene the validation start used
    obs_4, regions = stage4.forward_once(env4, scene)
    obs_4, failure = stage4.run_prefix(env4, obs_4, prefix, regions)

    row = {"seed": seed, "scene": scene, "handoff_valid": bool(start["handoff_valid"]),
           "stage4_prefix_failure": failure,
           "state_differs_at_handoff": same(flat_state(env4), flat_state(envv._env)),
           "fingers_equal": bool(np.array_equal(fingers(env4), fingers(envv._env))),
           "contacts_equal": bool(np.array_equal(contacts(env4), contacts(envv._env))),
           "finger_contact_n": np.round(np.linalg.norm(contacts(env4), axis=1), 3).tolist()}

    # Stage 4: run_learned with the drag, recording the state after every step.
    states_4 = []
    obs_4 = ex.run_learned(env4, obs_4, Drag(), target, regions, registry,
                           on_step=lambda _o: states_4.append(flat_state(env4)))
    scores_4 = {k: float(r.score) for k, r in registry.evaluate_all(prim.obs_to_scene(obs_4, regions)).items()}
    stop_4 = len(states_4) if target.robustly_holds(scores_4[target.atom]) else None

    # Validation: the same actions through the environment, for the whole 40 steps.
    trajectory, states_v = [start["scores"]], []
    for _ in range(envv.max_steps):
        obs_v, _r, _term, _trunc, info = envv.step(stage3._DRAG)
        trajectory.append(info["scores"])
        states_v.append(flat_state(envv._env))
    stop_v = stage3.stopping_point(trajectory, target)

    compared = len(states_4)                    # Stage 4 stops there; validation goes on to step 40
    row.update(
        stop_step_stage4=stop_4, stop_step_validation=stop_v,
        steps_compared=compared,
        trajectory_steps_differing=[t + 1 for t in range(compared) if same(states_4[t], states_v[t])],
        scores_at_stop_equal=(stop_v is not None and stop_4 == stop_v
                              and all(scores_4[l.atom] == trajectory[stop_v][l.atom] for l in contract)),
    )
    row["equal"] = (not row["state_differs_at_handoff"] and row["fingers_equal"] and row["contacts_equal"]
                    and not row["trajectory_steps_differing"] and stop_4 == stop_v
                    and (stop_4 is None or row["scores_at_stop_equal"]))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=list(stage3.VALIDATION_SEEDS))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; results are never overwritten")
    spec = load_spec(args.spec)
    env4, envv = prim.make_env(), StowHoleEnv(spec, seed_pool=tuple(args.seeds), exact_handoff=True)
    rows = []
    for seed in args.seeds:
        rows.append(compare(seed, spec, env4, envv))
        r = rows[-1]
        print(f"seed {seed} (scene {r['scene']}): {'equal' if r['equal'] else 'DIFFERENT'}; stop at step "
              f"{r['stop_step_stage4']} (validation {r['stop_step_validation']}); finger contact "
              f"{r['finger_contact_n']} N" + (f"; differs: {r['state_differs_at_handoff']}" if r['state_differs_at_handoff'] else ""))
    env4.close()
    envv.close()
    passed = all(r["equal"] for r in rows)
    print("\nHANDOFF EQUIVALENCE:", "PASS" if passed else "FAIL")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"spec": str(args.spec), "passed": passed, "seeds": rows}, indent=2))
    print(f"wrote {args.out}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
