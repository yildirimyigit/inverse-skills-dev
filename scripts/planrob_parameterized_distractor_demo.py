from __future__ import annotations

import json
from pathlib import Path

from inverse_skills.operators import OperatorExtractor, OperatorParameterizer
from inverse_skills.toy import (
    build_predicate_registry_with_distractor,
    make_push_rollouts_executable_named_with_distractor,
)


def run_case(*, skill_name: str, object_name: str, source_name: str, target_name: str, distractor_name: str) -> dict:
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


def main() -> None:
    outputs = {
        "case_cube": run_case(
            skill_name="push_restore",
            object_name="cube",
            source_name="source",
            target_name="target",
            distractor_name="can",
        ),
        "case_mug": run_case(
            skill_name="push_restore",
            object_name="mug",
            source_name="home",
            target_name="goal",
            distractor_name="bottle",
        ),
    }

    outputs["templates_match"] = (
        outputs["case_cube"]["parameterized_template"]["preconditions"]
        == outputs["case_mug"]["parameterized_template"]["preconditions"]
        and outputs["case_cube"]["parameterized_template"]["add_effects"]
        == outputs["case_mug"]["parameterized_template"]["add_effects"]
        and outputs["case_cube"]["parameterized_template"]["delete_effects"]
        == outputs["case_mug"]["parameterized_template"]["delete_effects"]
        and outputs["case_cube"]["parameterized_template"]["inverse_target_terms"]
        == outputs["case_mug"]["parameterized_template"]["inverse_target_terms"]
    )

    print(json.dumps(outputs, indent=2))
    out = Path("artifacts/planrob_parameterized_distractor_demo.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
