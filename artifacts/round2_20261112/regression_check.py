"""R0: the changed final evaluation against two earlier records, row for row.

scripts/drawer_final_evaluate.py changed in Round 2 in two ways: the part of an episode after
the handoff became the function `inverse`, and the direct policy now runs its full 100 steps.
Run with E3's checkpoints on the previous configurations 3000-3049, it must reproduce:

  targeted rows   E4's final evaluation (E4_stage4/final_targeted_vs_direct.json): the
                  targeted procedure did not change
  direct rows     the earlier full-horizon diagnostic (D_diagnostics/direct_full_horizon.json),
                  which ran the final evaluation's own code with the direct policy run for its
                  whole horizon: the rule Round 2 adopts

Every field the two rows share must be equal. Fields only the new rows have
(settle_gripper_command) are listed, not compared. Exit code 1 on any difference.

    PYTHONNOUSERSITE=1 python artifacts/round2_20261112/regression_check.py --new <json> \
        --targeted-record <E4 json> --direct-record <diagnostic json> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare(new_rows: list[dict], old_rows: list[dict]) -> dict:
    differing = []
    for new, old in zip(new_rows, old_rows):
        keys = sorted(new.keys() & old.keys())
        diff = [k for k in keys if new[k] != old[k]]
        if diff:
            differing.append({"seed": new.get("seed"), "fields": diff,
                              "new": {k: new[k] for k in diff}, "old": {k: old[k] for k in diff}})
    return {"rows": len(new_rows), "rows_in_record": len(old_rows), "rows_differing": len(differing),
            "only_in_new": sorted({k for row in new_rows for k in row} - {k for row in old_rows for k in row}),
            "differences": differing[:5], "equal": len(new_rows) == len(old_rows) and not differing}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--targeted-record", type=Path, required=True)
    ap.add_argument("--direct-record", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; results are never overwritten")
    new = json.loads(args.new.read_text())["results"]
    records = {"targeted": json.loads(args.targeted_record.read_text())["results"],
               "direct": json.loads(args.direct_record.read_text())["results"]}
    policies = {}
    for name, rows in sorted(new.items()):
        method = name.split(",")[0]
        policies[name] = {"record": str(args.targeted_record if method == "targeted" else args.direct_record),
                          **compare(rows, records[method][name])}
        p = policies[name]
        print(f"{name:22s} {p['rows']} rows: {'equal' if p['equal'] else str(p['rows_differing']) + ' differ'}"
              f"  (new fields, not compared: {', '.join(p['only_in_new']) or 'none'})")
    passed = all(p["equal"] for p in policies.values())
    print("\nREGRESSION:", "PASS" if passed else "FAIL")
    args.out.write_text(json.dumps({"new": str(args.new), "passed": passed, "policies": policies}, indent=2))
    print(f"wrote {args.out}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
