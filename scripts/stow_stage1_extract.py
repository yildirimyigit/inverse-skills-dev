"""Stage 1: extract the forward operator "stow" from scripted demonstrations.

Each demonstration spans the whole skill: first scene = the world before the
pushes, last scene = the world they leave behind (both cubes seated, arm
retracted, gripper open). Cutting anywhere inside the skill would sample the
TCP mid-contact and turn tcp_near into a spurious effect, as the PushCube
pipeline found.

The default min_max rule asks every demonstration to show each condition and
transition. Prints the per-predicate score table and the extracted operator,
checks it against what the scene is designed to mean, and writes
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
from inverse_skills.operators.literals import neg, pos
from inverse_skills.operators.stage1_checks import signed_summary, stage1_checks, stage1_table, uncertain_at_start

# What the forward skill is designed to do, as signed effects. Only compared with the result;
# the preconditions, and so the restoration goal, are whatever the demonstrations show.
DESIGNED_EFFECTS = {pos("in_region(cube_a,pocket)"), pos("in_region(cube_b,mouth)"),
                    neg("in_region(cube_a,src_a)"), neg("in_region(cube_b,src_b)"),
                    neg("clear_of_walls(cube_a)")}


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
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; saved results are never overwritten. Choose another --out.")

    env = prim.make_env()
    rollouts = [demonstrate(env, seed) for seed in args.seeds]
    env.close()

    registry = sp.registry()
    result = OperatorExtractor(registry).extract("stow", rollouts)
    operator = result.operator

    print(f"{len(rollouts)} demonstrations, seeds {args.seeds}\n")
    for line in stage1_table(result.scores):
        print(line)

    summary = signed_summary(operator)
    print("\noperator 'stow'")
    print("  Pre:", ", ".join(summary["preconditions"]) or "-")
    print("  Eff:", ", ".join(summary["effects"]) or "-")
    print("  restoration goal G = Pre u complement(Eff):", ", ".join(summary["restoration_goal"]))
    uncertain = uncertain_at_start(result.scores)
    print("  neither reliably true nor false at the start (not in G):", ", ".join(uncertain) or "none")

    checks = stage1_checks(operator, DESIGNED_EFFECTS, set(registry.robot_relative_keys()))
    print("\nchecks:", json.dumps(checks))
    for name, ok in checks.items():
        if not ok:
            print(f"  {name}: FAILED")
    print("STAGE 1:", "PASS" if all(checks.values()) else "FAIL")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "seeds": args.seeds,
        "demonstrations": {"skill": "stow", "seeds": args.seeds,
                           "boundaries": "the whole skill: the scene before it starts and the scene it leaves"},
        "extraction": {"mode": "min_max", "true_threshold": 0.8, "false_threshold": 0.2},
        "operator": operator.to_dict(),
        "signed": summary,
        "uncertain_at_start": uncertain,
        "scores": result.scores,
        "checks": checks,
    }, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
