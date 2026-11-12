# Previous results (signed-literal rerun, 2026-11-11), labelled

Recorded under the previous protocol; kept as they were. Not comparable seed for seed with Round 2.

## Drawer, configurations 3000-3049

Previous protocol: the direct reward had the fence not handle_grasped; the direct policy stopped at the first step where the goal held, then settled.

| Policy | Complete goal after settling (protocol) | Diagnostic: direct run for its full 100 steps |
|---|---|---|
| targeted, SAC seed 0 | 49/50 |  |
| targeted, SAC seed 1 | 49/50 |  |
| targeted, SAC seed 2 | 49/50 |  |
| targeted, SAC seed 3 | 49/50 |  |
| direct, SAC seed 0 | 20/50 | 42/50 |
| direct, SAC seed 1 | 0/50 | 49/50 |
| direct, SAC seed 2 | 34/50 | 49/50 |
| direct, SAC seed 3 | 39/50 | 49/50 |

## StowCube, scenes 1000-1009

Previous protocol: checkpoint chosen on the contract at step 40, then the fewest mean steps, on the same scenes as Stage 4; it chose step 4,000.

| Checkpoint | Full | Library only | Without interference | Relearned | Target in the learned operator |
|---|---|---|---|---|---|
| protocol-selected, step 4,000 (E4) | 8/10 | 0/10 | 0/10 | 0/10 | no |
| diagnostic: the run's final model, 120,000 | 10/10 | 0/10 | 0/10 | 9/10 | yes |
| diagnostic: historical checkpoint 1790788534 | 10/10 | 0/10 | 0/10 | 9/10 | yes |
