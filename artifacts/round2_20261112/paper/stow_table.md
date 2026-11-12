Selected checkpoint: 116,000 interactions (C_exec 100%, P10 +0.176). Stage 3: PASS.

| Stage 3 | Contract where Stage 4 stops (C_exec) | 95% interval | Contract at step 40 | P10 margin at step 40 | Mean steps to the target |
|---|---|---|---|---|---|
| validation (4000-4019) | 20/20 | [84, 100]% | 20/20 | +0.176 | 4.3 |
| test (5000-5049) | 50/50 | [93, 100]% | 50/50 | +0.174 | 4.1 |

| Stage 4, scenes 5000-5049 | Complete signed goal | 95% interval | Boolean goal |
|---|---|---|---|
| Full plan: planner + learned drag + retreat_to_safe | 48/50 | [87, 99]% | 48/50 |
| Library only | 0/50 | [0, 7]% | 0/50 |
| Without the interference model | 1/50 | [0, 10]% | 1/50 |
| Relearned: the drag as an extracted library operator | 48/50 | [87, 99]% | 48/50 |

Target among the learned operator's effects: yes. Handoff validity: demonstrations 10 of 11 scenes that reached the handoff; evaluation scenes 48 of 50. Stage 4: PASS.
