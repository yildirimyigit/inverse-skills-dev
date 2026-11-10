"""Every fresh output of this verification against the saved records. Comparison only.

Two references for each output, where both exist:
  saved            the result the paper's experiment recorded (artifacts/stow, artifacts/drawer)
  legacy_exact     the 2026-11-05 plain-script run with the legacy exact planner and the current
                   min_max extraction and push (artifacts/default_mode_validation/after_migration)

Values are compared exactly. Differences are listed by path, with list indices replaced by [*]
so that one field differing in many rows is listed once, with a count.

Writes comparison.json next to this file, never overwriting it.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

HERE = Path("artifacts/final_verification_fe374f0")
LEGACY = Path("artifacts/default_mode_validation/after_migration")
OUT = HERE / "comparison.json"

PAIRS = {
    "Stow Stage 1 operator": ("stow_stage1_operator.json", "artifacts/stow/stage1_operator.json",
                              LEGACY / "stow_stage1_operator.json"),
    "Drawer Stage 1 operator": ("drawer_stage1_operator.json", "artifacts/drawer/stage1_operator.json",
                                LEGACY / "drawer_stage1_operator.json"),
    "Stow Stage 2 plan": ("stow_stage2_plan.json", "artifacts/stow/stage2_plan.json", LEGACY / "stow_stage2_plan.json"),
    "Stow hole spec": ("stow_hole_spec.json", "artifacts/stow/hole_spec.json", LEGACY / "stow_hole_spec.json"),
    "Drawer Stage 2 plan": ("drawer_stage2_plan.json", "artifacts/drawer/stage2_plan.json",
                            LEGACY / "drawer_stage2_plan.json"),
    "Drawer hole spec": ("drawer_hole_spec.json", "artifacts/drawer/hole_spec.json", LEGACY / "drawer_hole_spec.json"),
    "Direct baseline plan": ("direct/plan_direct.json", "artifacts/drawer/direct/plan_direct.json",
                             LEGACY / "direct/plan_direct.json"),
    "Direct baseline spec": ("direct/hole_spec_direct.json", "artifacts/drawer/direct/hole_spec_direct.json",
                             LEGACY / "direct/hole_spec_direct.json"),
    "Stow Stage 3 input": ("stow_stage3_input.json", None, LEGACY / "stow_stage3_input.json"),
    "Drawer Stage 3 input": ("drawer_stage3_input.json", None, LEGACY / "drawer_stage3_input.json"),
    "Stow learned operator": ("stow_learned_operator.json", "artifacts/stow/learned_operator.json",
                              LEGACY / "stow_learned_operator.json"),
    "Stow Stage 4": ("stow_stage4.json", "artifacts/stow/stage4_execution.json", LEGACY / "stow_stage4.json"),
    "Drawer Stage 4": ("drawer_stage4.json", "artifacts/drawer/stage4_execution_after_push_fix.json",
                       LEGACY / "drawer_stage4.json"),
    "Targeted vs direct, seeds 3000-3049": ("drawer_final_targeted_vs_direct.json",
                                            "artifacts/drawer/final_targeted_vs_direct.json", None),
    "Stow planner demo": ("stow_planner_demo.json", "artifacts/stow/planner_demo.json",
                          LEGACY / "stow_planner_demo.json"),
}


def short(value) -> str:
    text = json.dumps(value)
    return text if len(text) <= 160 else text[:157] + "..."


def differences(fresh, reference, path: str, found: list) -> None:
    if isinstance(fresh, dict) and isinstance(reference, dict):
        for key in sorted(set(fresh) | set(reference), key=str):
            if key not in reference:
                found.append((f"{path}.{key}", "only in the fresh output", short(fresh[key])))
            elif key not in fresh:
                found.append((f"{path}.{key}", "only in the reference", short(reference[key])))
            else:
                differences(fresh[key], reference[key], f"{path}.{key}", found)
    elif isinstance(fresh, list) and isinstance(reference, list) and len(fresh) == len(reference):
        for i, (a, b) in enumerate(zip(fresh, reference)):
            differences(a, b, f"{path}[{i}]", found)
    elif fresh != reference:
        found.append((path, f"fresh {short(fresh)}", f"reference {short(reference)}"))


def compare(fresh_path: Path, reference_path) -> dict:
    if reference_path is None:
        return {"reference": None}
    reference_path = Path(reference_path)
    if not reference_path.exists():
        return {"reference": str(reference_path), "missing": True}
    fresh_text, reference_text = fresh_path.read_text(), reference_path.read_text()
    found: list = []
    differences(json.loads(fresh_text), json.loads(reference_text), "$", found)
    by_field = Counter(re.sub(r"\[\d+\]", "[*]", path) for path, *_ in found)
    return {"reference": str(reference_path), "bytes_identical": fresh_text == reference_text,
            "values_identical": not found, "differing_values": len(found),
            "differing_fields": dict(sorted(by_field.items())),
            "examples": [list(row) for row in found[:12]]}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; results are never overwritten")
    result = {}
    for name, (fresh, saved, legacy) in PAIRS.items():
        fresh_path = HERE / fresh
        result[name] = {"fresh": str(fresh_path), "saved": compare(fresh_path, saved),
                        "legacy_exact": compare(fresh_path, legacy)}
        for label in ("saved", "legacy_exact"):
            row = result[name][label]
            if row.get("reference") and not row.get("missing"):
                verdict = ("bytes identical" if row["bytes_identical"] else "values identical" if row["values_identical"]
                           else f"{row['differing_values']} values differ in {len(row['differing_fields'])} fields")
                print(f"{name:38s} vs {label:12s} {verdict}")
    OUT.write_text(json.dumps(result, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
