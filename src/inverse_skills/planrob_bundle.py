from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from inverse_skills.core import Pose
from inverse_skills.operators.extractor import OperatorExtractor
from inverse_skills.operators.parameterized import OperatorParameterizer
from inverse_skills.operators.restoration import RestorationObjective
from inverse_skills.operators.toy_planner import ToyInversePlanner
from inverse_skills.operators.two_phase import two_phase_inverse
from inverse_skills.predicates import (
    AtPosePredicate,
    GripperOpenPredicate,
    InRegionPredicate,
    PredicateRegistry,
)
from inverse_skills.toy.domains import (
    build_predicate_registry,
    build_predicate_registry_grasp_hold,
    build_predicate_registry_with_distractor,
    make_scene,
)
from inverse_skills.toy.generators import (
    make_grasp_hold_rollouts_executable,
    make_pick_place_rollouts_executable,
    make_push_rollouts_executable,
    make_push_rollouts_executable_named,
    make_push_rollouts_executable_named_with_distractor,
)
from inverse_skills.toy.primitives import PrimitiveLibrary
from inverse_skills.toy.simulator import ToyTabletopSimulator


def _run_executable_case(skill_name: str, rollouts) -> dict:
    registry = build_predicate_registry()
    extraction = OperatorExtractor(registry).extract(skill_name, rollouts)
    objective = RestorationObjective(extraction.operator, registry)
    planner = ToyInversePlanner(PrimitiveLibrary(), objective)
    result = planner.plan(rollouts[0].last(), max_depth=3)

    return {
        "skill_name": skill_name,
        "num_scenes_first_rollout": len(rollouts[0].scenes),
        "first_rollout_action_trace": rollouts[0].metadata.get("action_trace", []),
        "operator": extraction.operator.to_dict(),
        "inverse_potential_at_forward_final": objective.potential(rollouts[0].last()),
        "inverse_potential_at_forward_start": objective.potential(rollouts[0].first()),
        "per_predicate_at_forward_final": _per_predicate_scores(objective, rollouts[0].last()),
        "plan_success": result.success,
        "plan_actions": result.actions,
        "plan_final_potential": result.final_potential,
        "expanded_nodes": result.expanded_nodes,
    }


def _per_predicate_scores(objective: RestorationObjective, scene) -> dict[str, float]:
    return {
        term.key: float(objective.term_score(term, scene))
        for term in objective.terms
    }


def _run_grasp_hold_case() -> dict:
    rollouts = make_grasp_hold_rollouts_executable()
    registry = build_predicate_registry_grasp_hold()
    extraction = OperatorExtractor(registry).extract("grasp_hold", rollouts)
    objective = RestorationObjective(extraction.operator, registry)
    planner = ToyInversePlanner(PrimitiveLibrary(), objective)
    result = planner.plan(rollouts[0].last(), max_depth=3)

    return {
        "skill_name": "grasp_hold",
        "num_scenes_first_rollout": len(rollouts[0].scenes),
        "first_rollout_action_trace": rollouts[0].metadata.get("action_trace", []),
        "operator": extraction.operator.to_dict(),
        "inverse_potential_at_forward_final": objective.potential(rollouts[0].last()),
        "inverse_potential_at_forward_start": objective.potential(rollouts[0].first()),
        "per_predicate_at_forward_final": _per_predicate_scores(objective, rollouts[0].last()),
        "plan_success": result.success,
        "plan_actions": result.actions,
        "plan_final_potential": result.final_potential,
        "expanded_nodes": result.expanded_nodes,
    }


def _run_parameterized_case(*, skill_name: str, object_name: str, source_name: str, target_name: str) -> dict:
    rollouts = make_push_rollouts_executable_named(
        object_name=object_name,
        source_name=source_name,
        target_name=target_name,
        skill_name=skill_name,
    )
    registry = build_predicate_registry(object_name, source_name, target_name)
    learned = OperatorExtractor(registry).extract(skill_name, rollouts)
    template = OperatorParameterizer().parameterize(learned.operator, rollouts[0])
    return {
        "object_name": object_name,
        "source_name": source_name,
        "target_name": target_name,
        "ground_operator": learned.operator.to_dict(),
        "parameterized_template": template.to_dict(),
    }


def _run_parameterized_distractor_case(
    *,
    skill_name: str,
    object_name: str,
    source_name: str,
    target_name: str,
    distractor_name: str,
) -> dict:
    rollouts = make_push_rollouts_executable_named_with_distractor(
        object_name=object_name,
        source_name=source_name,
        target_name=target_name,
        distractor_name=distractor_name,
        skill_name=skill_name,
    )
    registry = build_predicate_registry_with_distractor(
        object_name=object_name,
        source_name=source_name,
        target_name=target_name,
        distractor_name=distractor_name,
    )
    learned = OperatorExtractor(registry).extract(skill_name, rollouts)
    template = OperatorParameterizer().parameterize(learned.operator, rollouts[0])
    return {
        "object_name": object_name,
        "source_name": source_name,
        "target_name": target_name,
        "distractor_name": distractor_name,
        "ground_operator": learned.operator.to_dict(),
        "parameterized_template": template.to_dict(),
    }


def _run_two_phase_case(skill_name: str, rollouts, registry, max_depth: int) -> dict:
    operator = OperatorExtractor(registry).extract(skill_name, rollouts).operator
    objective = RestorationObjective(operator, registry)
    planner = ToyInversePlanner(PrimitiveLibrary(), objective)
    result = two_phase_inverse(planner, rollouts[0].last(), max_depth=max_depth)
    return result.to_dict()


def _build_pose_precise_push_case() -> tuple[str, list, PredicateRegistry]:
    """Push rollouts that share a precise starting pose offset from source.center.

    The inverse target includes at_pose(cube, init_pose) with 5mm tolerance.
    BFS's place(source) teleports to source.center, which is ~6cm away — so
    at_pose is structurally unreachable by the discrete primitive library.
    """
    init_position = [0.05, 0.04, 0.02]
    init_pose = Pose(
        position=np.asarray(init_position, dtype=np.float32),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    sim = ToyTabletopSimulator(PrimitiveLibrary())
    rollouts = []
    for i in range(3):
        start = make_scene(0, init_position, skill_name="push_to_target")
        out = sim.execute("push_to_target", f"push_pose_precise_{i:03d}", start, ["push(target)"])
        rollouts.append(out.rollout)
    registry = PredicateRegistry([
        InRegionPredicate("cube", "source"),
        InRegionPredicate("cube", "target"),
        GripperOpenPredicate(min_width=0.04),
        AtPosePredicate("cube", target_pose=init_pose,
                        distance_threshold=0.005, temperature=0.001),
    ])
    return "push_to_target", rollouts, registry


def build_bundle() -> dict:
    executable = {
        "pick_place": _run_executable_case("pick_place", make_pick_place_rollouts_executable()),
        "push_to_target": _run_executable_case("push_to_target", make_push_rollouts_executable()),
        "grasp_hold": _run_grasp_hold_case(),
    }

    pose_precise_skill, pose_precise_rollouts, pose_precise_registry = _build_pose_precise_push_case()

    two_phase = {
        "pick_place_d3": _run_two_phase_case(
            "pick_place", make_pick_place_rollouts_executable(),
            build_predicate_registry(), max_depth=3,
        ),
        "push_d3": _run_two_phase_case(
            "push_to_target", make_push_rollouts_executable(),
            build_predicate_registry(), max_depth=3,
        ),
        "grasp_hold_d3": _run_two_phase_case(
            "grasp_hold", make_grasp_hold_rollouts_executable(),
            build_predicate_registry_grasp_hold(), max_depth=3,
        ),
        "push_pose_precise": _run_two_phase_case(
            pose_precise_skill, pose_precise_rollouts, pose_precise_registry, max_depth=3,
        ),
    }

    parameterized = {
        "case_cube": _run_parameterized_case(
            skill_name="push_restore",
            object_name="cube",
            source_name="source",
            target_name="target",
        ),
        "case_mug": _run_parameterized_case(
            skill_name="push_restore",
            object_name="mug",
            source_name="home",
            target_name="goal",
        ),
    }
    parameterized["templates_match"] = (
        parameterized["case_cube"]["parameterized_template"]["preconditions"]
        == parameterized["case_mug"]["parameterized_template"]["preconditions"]
        and parameterized["case_cube"]["parameterized_template"]["add_effects"]
        == parameterized["case_mug"]["parameterized_template"]["add_effects"]
        and parameterized["case_cube"]["parameterized_template"]["delete_effects"]
        == parameterized["case_mug"]["parameterized_template"]["delete_effects"]
        and parameterized["case_cube"]["parameterized_template"]["inverse_target_terms"]
        == parameterized["case_mug"]["parameterized_template"]["inverse_target_terms"]
    )

    distractor = {
        "case_cube": _run_parameterized_distractor_case(
            skill_name="push_restore",
            object_name="cube",
            source_name="source",
            target_name="target",
            distractor_name="can",
        ),
        "case_mug": _run_parameterized_distractor_case(
            skill_name="push_restore",
            object_name="mug",
            source_name="home",
            target_name="goal",
            distractor_name="bottle",
        ),
    }
    distractor["templates_match"] = (
        distractor["case_cube"]["parameterized_template"]["preconditions"]
        == distractor["case_mug"]["parameterized_template"]["preconditions"]
        and distractor["case_cube"]["parameterized_template"]["add_effects"]
        == distractor["case_mug"]["parameterized_template"]["add_effects"]
        and distractor["case_cube"]["parameterized_template"]["delete_effects"]
        == distractor["case_mug"]["parameterized_template"]["delete_effects"]
        and distractor["case_cube"]["parameterized_template"]["inverse_target_terms"]
        == distractor["case_mug"]["parameterized_template"]["inverse_target_terms"]
    )

    summary = {
        "same_class_inverse_success": executable["pick_place"]["plan_success"],
        "cross_class_inverse_success": executable["push_to_target"]["plan_success"],
        "cross_class_inverse_actions": executable["push_to_target"]["plan_actions"],
        "grasp_hold_inverse_success": executable["grasp_hold"]["plan_success"],
        "grasp_hold_inverse_actions": executable["grasp_hold"]["plan_actions"],
        "renaming_template_invariance": parameterized["templates_match"],
        "distractor_template_invariance": distractor["templates_match"],
        "pick_place_restoration_gain": (
            executable["pick_place"]["plan_final_potential"]
            - executable["pick_place"]["inverse_potential_at_forward_final"]
        ),
        "push_restoration_gain": (
            executable["push_to_target"]["plan_final_potential"]
            - executable["push_to_target"]["inverse_potential_at_forward_final"]
        ),
        "grasp_hold_restoration_gain": (
            executable["grasp_hold"]["plan_final_potential"]
            - executable["grasp_hold"]["inverse_potential_at_forward_final"]
        ),
        "push_d3_gap_closed_by_symbolic": two_phase["push_d3"]["gap_closed_by_symbolic"],
        "push_d3_gap_remaining_for_rl": two_phase["push_d3"]["gap_remaining_for_rl"],
        "pose_precise_gap_closed_by_symbolic": two_phase["push_pose_precise"]["gap_closed_by_symbolic"],
        "pose_precise_gap_remaining_for_rl": two_phase["push_pose_precise"]["gap_remaining_for_rl"],
        "pose_precise_residual_terms": two_phase["push_pose_precise"]["residual_term_keys"],
        "pose_precise_fully_solved": two_phase["push_pose_precise"]["fully_solved"],
    }

    return {
        "summary": summary,
        "executable": executable,
        "two_phase": two_phase,
        "parameterized": parameterized,
        "distractor": distractor,
    }


def bundle_markdown_table(bundle: dict) -> str:
    summary = bundle["summary"]
    exe = bundle["executable"]
    lines = [
        "## Claims",
        "",
        "| Check | Value |",
        "|---|---|",
        f"| Same-class inverse success | `{summary['same_class_inverse_success']}` |",
        f"| Cross-class inverse success | `{summary['cross_class_inverse_success']}` |",
        f"| Cross-class inverse actions | `{' -> '.join(summary['cross_class_inverse_actions'])}` |",
        f"| Grasp-hold inverse success | `{summary['grasp_hold_inverse_success']}` |",
        f"| Grasp-hold inverse actions | `{' -> '.join(summary['grasp_hold_inverse_actions'])}` |",
        f"| Template invariance across renamings | `{summary['renaming_template_invariance']}` |",
        f"| Template invariance with distractor present | `{summary['distractor_template_invariance']}` |",
        f"| Pick-place restoration gain | `{summary['pick_place_restoration_gain']:.6f}` |",
        f"| Push restoration gain | `{summary['push_restoration_gain']:.6f}` |",
        f"| Grasp-hold restoration gain | `{summary['grasp_hold_restoration_gain']:.6f}` |",
        "",
        "## Per-predicate inverse target scores at forward final state",
        "",
        "| Predicate | pick\\_place | push | grasp\\_hold |",
        "|---|---|---|---|",
    ]
    all_terms: dict[str, dict] = {}
    for skill in ("pick_place", "push_to_target", "grasp_hold"):
        for term_key, score in exe[skill]["per_predicate_at_forward_final"].items():
            all_terms.setdefault(term_key, {})[skill] = score
    for term_key, skill_scores in sorted(all_terms.items()):
        pp = f"`{skill_scores.get('pick_place', '—'):.4f}`" if 'pick_place' in skill_scores else "—"
        pu = f"`{skill_scores.get('push_to_target', '—'):.4f}`" if 'push_to_target' in skill_scores else "—"
        gh = f"`{skill_scores.get('grasp_hold', '—'):.4f}`" if 'grasp_hold' in skill_scores else "—"
        lines.append(f"| `{term_key}` | {pp} | {pu} | {gh} |")
    return "\n".join(lines) + "\n"


def bundle_latex_table(bundle: dict) -> str:
    summary = bundle["summary"]
    exe = bundle["executable"]
    rows = [
        ("Same-class inverse success", str(summary["same_class_inverse_success"])),
        ("Cross-class inverse success", str(summary["cross_class_inverse_success"])),
        ("Cross-class inverse actions", " $\\rightarrow$ ".join(summary["cross_class_inverse_actions"])),
        ("Grasp-hold inverse success", str(summary["grasp_hold_inverse_success"])),
        ("Grasp-hold inverse actions", " $\\rightarrow$ ".join(summary["grasp_hold_inverse_actions"])),
        ("Template invariance across renamings", str(summary["renaming_template_invariance"])),
        ("Template invariance with distractor present", str(summary["distractor_template_invariance"])),
        ("Pick-place restoration gain", f"{summary['pick_place_restoration_gain']:.6f}"),
        ("Push restoration gain", f"{summary['push_restoration_gain']:.6f}"),
        ("Grasp-hold restoration gain", f"{summary['grasp_hold_restoration_gain']:.6f}"),
    ]
    lines = [
        "% Main claims table",
        "\\begin{tabular}{ll}",
        "\\toprule",
        "Check & Value \\\\",
        "\\midrule",
    ]
    for check, value in rows:
        lines.append(f"{check} & {value} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "",
        "% Per-predicate comparison table",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Predicate & pick-place & push & grasp-hold \\\\",
        "\\midrule",
    ])
    all_terms: dict[str, dict] = {}
    for skill in ("pick_place", "push_to_target", "grasp_hold"):
        for term_key, score in exe[skill]["per_predicate_at_forward_final"].items():
            all_terms.setdefault(term_key, {})[skill] = score
    for term_key, skill_scores in sorted(all_terms.items()):
        pp = f"{skill_scores['pick_place']:.4f}" if 'pick_place' in skill_scores else "---"
        pu = f"{skill_scores['push_to_target']:.4f}" if 'push_to_target' in skill_scores else "---"
        gh = f"{skill_scores['grasp_hold']:.4f}" if 'grasp_hold' in skill_scores else "---"
        term_escaped = term_key.replace("_", "\\_")
        lines.append(f"\\texttt{{{term_escaped}}} & {pp} & {pu} & {gh} \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ])
    return "\n".join(lines)


def write_bundle_artifacts(bundle: dict, artifacts_dir: str = "artifacts") -> tuple[Path, Path, Path]:
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "planrob_submission_bundle.json"
    md_path = out_dir / "planrob_submission_table.md"
    tex_path = out_dir / "planrob_submission_table.tex"

    json_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    md_path.write_text(bundle_markdown_table(bundle), encoding="utf-8")
    tex_path.write_text(bundle_latex_table(bundle), encoding="utf-8")
    return json_path, md_path, tex_path
