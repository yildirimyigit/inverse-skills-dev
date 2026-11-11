# Retreat validation, before registration

- **Run:** `scripts/stow_retreat_gate.py` at commit `7279aad`, clean tree.
- **Output:** `retreat_gate.json`, with its console log `retreat_gate.stdout`.
- **What was tested:** the existing `retreat()` rise, run as `retreat_to_safe` runs it, with the gripper kept as it is.
- **Where it started:** from 5 states the historical inverse execution reaches, on 20 seeds each (1000–1009 and 2000–2009). The states come from the historical plan, the historical drag policy `1790788534/sac_hole_best`, and the published executive.

## Results

| Start state | Gripper at start | Passed | tcp_near(A) after | tcp_near(B) after | Cube displacement | Contact during the rise |
|---|---|---|---|---|---|---|
| after the forward skill | open | 20/20 | ≤ 0.001 | ≤ 0.001 | 0 mm | none; already above the travel height, so 0 steps |
| after place(cube_b, src_b) | open | 20/20 | ≤ 0.001 | ≤ 0.005 | ≤ 0.001 mm | none |
| after place(cube_a, src_a), the plan's end | open | 20/20 | ≤ 0.006 | ≤ 0.001 | ≤ 0.001 mm | none |
| after approach(cube_a) | closed, pressing on A | 0/20 | ≤ 0.004 | 0.000 | A: 1.4–79 mm; in 15 runs A left the pocket | none after the first step |
| after the learned drag | closed, on A | 0/20 | ≤ 0.020 | 0.000 | A: 2.6–44 mm | none after the first step |

In every open-gripper run the gripper stayed robustly open, at 0.976 or 0.982, and every other predicate kept its robust value.

## What causes the failures

The contact measurement does work: in a separate probe of seed 1003, pressing on A registered 5–23 N on the cube and 15 N on the back wall behind it. During the rise no robot link touches anything after the first step.

The cube moves during that first step, when the press is released: A is squeezed against the pocket wall and springs back once the fingers lift. So the rise is collision-free. It separates the hand from both cubes robustly in every state, at a score of 0.020 or less. It disturbs a cube only when it starts from a closed-gripper press.

## Consequence for the model

`retreat_to_safe` is registered in `stow_domain.py` as follows:

- **Preconditions:** `gripper_open()`, which the measurements justify, and not `holding(cube_a)` and not `holding(cube_b)`, because rising while holding a cube would carry it along.
- **Effects:** not `tcp_near(cube_a)` and not `tcp_near(cube_b)`. There is no gripper effect and no object effect.

Within these preconditions the action passed 60 of 60 runs. Outside them (closed gripper) it failed 40 of 40, which is why the precondition is there.

The motion itself is unchanged. The planner may use the action only where it was verified. The plan expected for the signed goal uses it after `place(cube_a, src_a)`, where the gripper is open.

The gate will be rerun at the final code in Stage 0 (E0). That run judges the runs whose start satisfies the registered preconditions, and still reports the others.
