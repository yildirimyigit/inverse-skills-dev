| Policy | Goal after settling (k/50) | 95% interval | Goal at the policy's end | Lost in settling | First ≥90% complete-goal validation | First ≥90% grasp validation | Best/final complete-goal validation | Stayed ≥90% | Selected at |
|---|---|---|---|---|---|---|---|---|---|
| targeted, SAC seed 0 | 50/50 | [93, 100]% | 50/50 | 0 | 10k | 10k | 100/100% | yes | 10k |
| targeted, SAC seed 1 | 50/50 | [93, 100]% | 50/50 | 0 | 10k | 10k | 100/100% | yes | 10k |
| targeted, SAC seed 2 | 50/50 | [93, 100]% | 50/50 | 0 | 12.5k | 12.5k | 100/100% | yes | 12.5k |
| targeted, SAC seed 3 | 50/50 | [93, 100]% | 50/50 | 0 | 10k | 12.5k | 100/100% | yes | 12.5k |
| direct, SAC seed 0 | 50/50 | [93, 100]% | 50/50 | 0 | 25k | — | 100/70% | no | 25k |
| direct, SAC seed 1 | 50/50 | [93, 100]% | 50/50 | 0 | 22.5k | — | 100/100% | yes | 22.5k |
| direct, SAC seed 2 | 50/50 | [93, 100]% | 50/50 | 0 | 22.5k | — | 100/0% | no | 22.5k |
| direct, SAC seed 3 | 50/50 | [93, 100]% | 50/50 | 0 | 25k | — | 100/100% | yes | 25k |
| targeted, over the 4 seeds | 200/200; per seed 50–50, median 50 | | | | median 10k | | | | |
| direct, over the 4 seeds | 200/200; per seed 50–50, median 50 | | | | median 23.75k | | | | |

Four SAC runs per method, each at its validation-selected checkpoint, on the same 50 configurations with identical forward initializations. Success is the complete signed restoration goal, every literal judged robustly, after the established 10 settling steps. Validation (10 configurations every 2,500 interactions, budget 30k) measures the complete goal for both methods: targeted after its scripted pull_open and settling, direct after its 100 steps and settling; the targeted checkpoint is selected on its grasp validation, as established. Wilson intervals are per policy. Four runs per method do not support strong statistical claims.
