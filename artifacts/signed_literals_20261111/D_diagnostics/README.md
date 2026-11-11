# Diagnostics: not part of the protocol

These runs answer one question raised by E4, and replace no result.

**The question.** In E4 the fresh drag policy, selected at 4,000 steps by the unchanged rule, fails the full Stow plan on some scenes. Its learned operator also lacks the target effect. Is that the policy or the code?

**The runs.** Both use the same Stage 4 script and the same E1/E2 inputs as E4, at commit 6acc02e, with a different checkpoint each:
- `stow_stage4_historical_checkpoint.*` uses `artifacts/stow/stage3/1790788534/sac_hole_best`, the published policy.
- `stow_stage4_fresh_final_model.*` uses `E3_stage3/stow/1794407215/sac_hole_final`, the fresh run's last model at 120,000 steps. The selection rule did not choose it.

Both ran at commit 6acc02e on 2026-11-11, at 15:06 UTC, each with `PYTHONNOUSERSITE=1`. With `<run>` the run's name:

    python scripts/stow_stage4_execute.py <checkpoint> \
      --operator artifacts/signed_literals_20261111/E1_stage1/stow_stage1_operator.json \
      --spec artifacts/signed_literals_20261111/E2_stage2/stow_hole_spec.json \
      --plan artifacts/signed_literals_20261111/E2_stage2/stow_stage2_plan.json \
      --out artifacts/signed_literals_20261111/D_diagnostics/stow_stage4_<run>.json \
      --learned-out artifacts/signed_literals_20261111/D_diagnostics/stow_learned_operator_<run>.json

The console logs, `*.log`, stay local.

`direct_full_horizon.*` asks a second question, from the final evaluation. Under the protocol, the direct policies reach the complete signed goal and then lose it while settling. How do they do when run for their full 100-step training horizon, as validation runs them, instead of being stopped at the first goal step?
- The script imports the final evaluation's code and changes only that stop rule, for the direct policies alone.
- It uses the same configurations 3000–3049, the same checked checkpoints, the same 10-step settling and the same robust goal.

The full-horizon run's command is in its script's docstring. Its output records its commit.
