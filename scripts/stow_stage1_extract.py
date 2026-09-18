"""Stage 1: extract the forward operator "stow" from scripted demonstrations.

Each demonstration spans the whole skill: first scene = the world before the
pushes, last scene = the world they leave behind (both cubes seated, arm
retracted, gripper open). Cutting anywhere inside the skill would sample the
TCP mid-contact and turn tcp_near into a spurious effect, as the PushCube
pipeline found.

Prints the per-predicate score table and the extracted operator, checks it
against what the scene is designed to mean, and writes
artifacts/stow/stage1_operator.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inverse_skills.envs import stow_predicates as sp
from inverse_skills.envs import stow_primitives as prim
from inverse_skills.logging.rollout import ForwardRollout
from inverse_skills.operators.extractor import OperatorExtractor

_EXPECTED_PRE = {"in_region(cube_a,src_a)", "in_region(cube_b,src_b)",
                 "gripper_open()", "clear_of_walls(cube_a)"}
_EXPECTED_ADD = {"in_region(cube_a,pocket)", "in_region(cube_b,mouth)"}
_EXPECTED_DEL = {"in_region(cube_a,src_a)", "in_region(cube_b,src_b)",
                 "clear_of_walls(cube_a)"}


def demonstrate(env, seed: int) -> ForwardRollout:
    obs = prim.reset(env, seed)
    regions = prim.regions(obs)
    start = prim.obs_to_scene(obs, regions)
    obs = prim.forward_stow(env, obs)
    end = prim.obs_to_scene(obs, regions, timestep=1)
    return ForwardRollout(skill_name="stow", demo_id=f"demo_{seed}", scenes=[start, end])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")],
                    default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", type=Path, default=Path("artifacts/stow/stage1_operator.json"))
    args = ap.parse_args()

    env = prim.make_env()
    rollouts = [demonstrate(env, seed) for seed in args.seeds]
    env.close()

    registry = sp.registry()
    result = OperatorExtractor(registry).extract("stow", rollouts)
    operator = result.operator

    print(f"{len(rollouts)} demonstrations, seeds {args.seeds}\n")
    print(f"{'predicate':28s} {'start':>7} {'end':>7} {'delta':>7}")
    for key in registry.keys():
        s = result.scores[key]
        print(f"{key:28s} {s['start_mean']:7.3f} {s['end_mean']:7.3f} {s['delta']:+7.3f}")

    pre = {t.key for t in operator.preconditions}
    add = {t.key for t in operator.add_effects}
    dele = {t.key for t in operator.delete_effects}
    print("\noperator 'stow'")
    print("  Pre:", ", ".join(sorted(pre)) or "-")
    print("  Add:", ", ".join(sorted(add)) or "-")
    print("  Del:", ", ".join(sorted(dele)) or "-")
    print("  inverse target:", ", ".join(
        f"{'¬' if t.polarity == 'negative' else ''}{t.key}"
        for t in operator.inverse_target_terms()))

    checks = {
        "pre_matches": pre == _EXPECTED_PRE,
        "add_matches": add == _EXPECTED_ADD,
        "del_matches": dele == _EXPECTED_DEL,
        "no_tcp_near_terms": not any(k.startswith("tcp_near") for k in pre | add | dele),
    }
    print("\nchecks:", json.dumps(checks))
    for name, ok in checks.items():
        if not ok:
            print(f"  {name}: FAILED")
    print("STAGE 1:", "PASS" if all(checks.values()) else "FAIL")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "seeds": args.seeds,
        "operator": operator.to_dict(),
        "scores": result.scores,
        "checks": checks,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
