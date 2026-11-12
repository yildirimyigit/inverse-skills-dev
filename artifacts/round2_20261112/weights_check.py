"""Did Round 2's training repeat E3's? Compares the policy networks' weights.

Round 2 retrained the four targeted Drawer grasp policies with the established protocol, adding only a
second validation measure (PLAN.md section 1.6), and retrained the Stow drag with unchanged training and
a new validation and selection (section 1.4). The checkpoint files' SHA-256 differ from E3's, so this
compares what matters: every tensor of the policy, and which zip members differ.
  targeted   each SAC seed's selected checkpoint against E3's
  stow       the final (120,000-step) model against E3's; E3 kept no other checkpoint in common

    PYTHONNOUSERSITE=1 python artifacts/round2_20261112/weights_check.py

Writes R1_training/weights_vs_e3.json, never overwriting it.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import torch
from stable_baselines3 import SAC

HERE = Path("artifacts/round2_20261112")
OUT = HERE / "R1_training" / "weights_vs_e3.json"
E3 = Path("artifacts/signed_literals_20261111/E3_stage3/drawer")
E3_STOW = Path("artifacts/signed_literals_20261111/E3_stage3/stow/1794407215/sac_hole_final.zip")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; it is never overwritten")
    torch.set_num_threads(1)
    pairs = {f"targeted, SAC seed {k}": (next((HERE / "R1_training" / "drawer").glob(f"seed{k}_*/sac_hole_best.zip")),
                                         next(E3.glob(f"seed{k}_*/sac_hole_best.zip"))) for k in range(4)}
    pairs["stow, final model"] = (next((HERE / "R1_training" / "stow").glob("*/sac_hole_final.zip")), E3_STOW)
    rows = {}
    for name, (new, old) in pairs.items():
        a, b = SAC.load(new, device="cpu").policy.state_dict(), SAC.load(old, device="cpu").policy.state_dict()
        with zipfile.ZipFile(new) as za, zipfile.ZipFile(old) as zb:
            differ = sorted(n for n in set(za.namelist()) & set(zb.namelist()) if za.read(n) != zb.read(n))
        rows[name] = {"round2": str(new), "e3": str(old), "tensors": len(a),
                                 "weights_identical": a.keys() == b.keys() and all(torch.equal(a[n], b[n]) for n in a),
                                 "zip_members_differing": differ}
        print(f"{name}: weights identical {rows[name]['weights_identical']}; "
              f"zip members differing: {differ}")
    OUT.write_text(json.dumps({"about": __doc__.split(chr(10))[0], "policies": rows}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
