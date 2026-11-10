"""Can the minimal planner itself give the direct baseline? Planning only, no simulation.

`scripts/drawer_direct_spec.py` defines the direct baseline through the relaxed planner's
`unrestricted_holes` option, which the minimal planner does not have. This asks the minimal
planner, from each state the fresh Drawer Stage 2 run measured, for two descriptions of the
same baseline, and compares the learning problem each gives with the fresh direct spec:

  grasp_unscored     handle_grasped has no soft score, so it cannot be learned
  without_pull_open  the library has no pull_open

Writes direct_from_minimal_planner.json next to this file, never overwriting it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.operators import hole_planner as hp

HERE = Path("artifacts/final_verification_fe374f0")
OUT = HERE / "direct_from_minimal_planner.json"


def variants() -> dict:
    grasp_unscored = dd.domain()
    grasp_unscored.scored = grasp_unscored.scored - {dd.GRASPED}
    full = dd.domain()
    without_pull = replace(full, actions=[a for a in full.actions if not a.name.startswith("pull_open")])
    return {"grasp_unscored": grasp_unscored, "without_pull_open": without_pull}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; results are never overwritten")
    goal = dd.goal_from_operator(HERE / "drawer_stage1_operator.json")
    direct_text = (HERE / "direct" / "hole_spec_direct.json").read_text()
    direct = json.loads(direct_text)
    rows = []
    for seed_row in json.loads((HERE / "drawer_stage2_plan.json").read_text())["seeds"]:
        state = frozenset(seed_row["state"])
        for name, domain in variants().items():
            result = hp.plan(domain, state, *goal)
            spec = hp.hole_spec(domain, state, result, *goal)
            rows.append({"seed": seed_row["seed"], "variant": name, "plan": list(result.actions),
                         "cost": list(result.cost),
                         "spec_bytes_equal_direct": json.dumps(spec, indent=2) == direct_text,
                         "fields_differing_from_direct": sorted(k for k in set(spec) | set(direct)
                                                                if spec.get(k) != direct.get(k))})
    summary = {name: {"seeds": sum(r["variant"] == name for r in rows),
                      "spec_bytes_equal_direct": sum(r["spec_bytes_equal_direct"] for r in rows if r["variant"] == name),
                      "plans": sorted({" > ".join(r["plan"]) for r in rows if r["variant"] == name}),
                      "fields_differing": sorted({f for r in rows if r["variant"] == name
                                                  for f in r["fields_differing_from_direct"]})}
               for name in variants()}
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
