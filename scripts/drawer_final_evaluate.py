"""Final targeted-versus-direct evaluation on the reserved configurations, seeds 3000-3049.

These seeds were never used before: not for training, checkpoint selection or tuning.

Eight frozen policies, each at its validation-selected best checkpoint, with its
SHA-256 checked against the tracked manifest before loading:
  targeted, SAC seed k   Stage 3's grasp policies: HOLE[handle_grasped], then pull_open
  direct, SAC seed k     the direct baseline: HOLE[drawer_open], nothing scripted

Every episode, the same for both methods:
  1. reset the configuration, then run the forward push_close and approach_handle,
     the current controllers, so both methods take over from the same physical
     state (checked: identical for all eight policies on every configuration);
  2. targeted: the grasp policy until handle_grasped >= 0.8, at most 40 steps (its
     training horizon), then the known pull_open, run only if the grasp holds
     (its precondition);
     direct: the drawer-opening policy until the restoration goal holds, at most
     100 steps (its training horizon);
  3. settling, identical for both: 10 steps with the arm held still and the
     gripper command unchanged;
  4. success: the restoration goal, drawer_open and not drawer_closed, measured
     after settling.

How the drawer was moved is recorded the first time it passes 50 mm open: a
two-finger pinch (handle_grasped >= 0.8), a one-finger press (one finger at
10 N or more while the pinch does not count), or other.

Also, from the validation curves recorded during training (seeds 500-509): the
interactions needed to first reach >= 90% validation success, whether it held
afterwards, and best versus final checkpoint.

Writes artifacts/drawer/final_targeted_vs_direct.json (never overwrites) and prints the table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.envs import drawer_primitives as prim
from inverse_skills.envs import drawer_skill_io as skill_io
from inverse_skills.envs.drawer_direct_env import observe as observe_direct
from inverse_skills.envs.motion import step_in_place

torch.set_num_threads(1)

SEEDS = tuple(range(3000, 3050))
SATISFIED = 0.8
GRASP_STEPS = 40          # the grasp policies' training horizon
DIRECT_STEPS = 100        # the direct policies' training horizon
SETTLE_STEPS = 10
PRESS_N = 10.0            # a finger pressing at least this hard counts as pressing
OPEN, CLOSED = "drawer_open(drawer)", "drawer_closed(drawer)"
GRASPED, STANDOFF = "handle_grasped(drawer)", "at_handle_standoff(drawer)"
STAGE3 = Path("artifacts/drawer/stage3")
DIRECT = Path("artifacts/drawer/direct")


def scores(obs, registry) -> dict[str, float]:
    return {k: float(r.score) for k, r in registry.evaluate_all(prim.obs_to_scene(obs)).items()}


def goal_met(s: dict[str, float], goal) -> bool:
    positive, negative = goal
    return all(s[p] >= SATISFIED for p in positive) and all(s[q] < SATISFIED for q in negative)


class Recorder(gym.Wrapper):
    """While `active`: counts control steps, notes when the goal first holds, the furthest
    opening, whether handle_grasped held, and how the drawer was held when it first
    passed 50 mm. Always keeps the last gripper command, for settling."""

    def __init__(self, env, registry, goal):
        super().__init__(env)
        self.registry, self.goal = registry, goal
        self.active = False
        self.gripper = 1.0

    def start(self):
        self.active, self.steps, self.goal_step = True, 0, None
        self.max_opening, self.grasp_held, self.at_50mm = 0.0, False, None

    def step(self, action):
        out = self.env.step(action)
        self.gripper = float(action[3])
        if self.active:
            obs = out[0]
            s = scores(obs, self.registry)
            self.steps += 1
            if self.goal_step is None and goal_met(s, self.goal):
                self.goal_step = self.steps
            self.max_opening = max(self.max_opening, prim.opening(obs))
            self.grasp_held = self.grasp_held or s[GRASPED] >= SATISFIED
            if self.at_50mm is None and prim.opening(obs) > 0.05:
                fingers = prim.pinch_force(obs)
                held = ("two-finger pinch" if s[GRASPED] >= SATISFIED
                        else "one-finger press" if max(fingers) >= PRESS_N else "other")
                self.at_50mm = {"how": held, "handle_grasped": round(s[GRASPED], 3),
                                "fingers_n": [round(f, 1) for f in fingers]}
        return out


def run_policy(env, obs, policy, observe, done, max_steps: int, registry):
    """Run the policy until `done` holds (checked before every step) or the budget runs out."""
    for used in range(max_steps):
        if done(scores(obs, registry)):
            return obs, used
        action, _ = policy.predict(observe(obs), deterministic=True)
        obs, *_ = env.step(torch.tensor(skill_io.command(action)))
    return obs, max_steps


def episode(env: Recorder, seed: int, method: str, policy, registry, goal) -> dict:
    obs = prim.reset(env, seed)
    obs = prim.push_close(env, obs)
    after_push = scores(obs, registry)
    obs = prim.approach_handle(env, obs)
    start = scores(obs, registry)
    row = {"seed": seed,
           "start_state": [*prim.tcp_pos(obs).tolist(), prim.opening(obs), prim.gripper_width(obs)],
           "forward_ok": after_push[CLOSED] >= SATISFIED and after_push[OPEN] < SATISFIED,
           "start_ok": start[STANDOFF] >= SATISFIED and start[GRASPED] < SATISFIED}
    if not (row["forward_ok"] and row["start_ok"]):
        row.update(success=False, failure="the push or the approach did not reach the handoff state")
        return row

    env.start()
    if method == "targeted":
        grasp = registry.get(GRASPED)
        obs, used = run_policy(env, obs, policy, lambda o: skill_io.observe(o, grasp),
                               lambda s: s[GRASPED] >= SATISFIED, GRASP_STEPS, registry)
        row["grasp_achieved"] = scores(obs, registry)[GRASPED] >= SATISFIED
        if row["grasp_achieved"]:
            obs = prim.pull_open(env, obs)
            row["pull_open_opened"] = scores(obs, registry)[OPEN] >= SATISFIED
    else:
        achieve, grasp = registry.get(OPEN), registry.get(GRASPED)
        obs, used = run_policy(env, obs, policy, lambda o: observe_direct(o, achieve, grasp),
                               lambda s: goal_met(s, goal), DIRECT_STEPS, registry)
    env.active = False
    row.update(learned_steps=used, steps_total=env.steps, steps_to_goal=env.goal_step,
               max_opening_mm=round(env.max_opening * 1000, 1), handle_grasped_held=env.grasp_held,
               at_50mm=env.at_50mm, goal_before_settling=goal_met(scores(obs, registry), goal))

    obs = step_in_place(env, obs, SETTLE_STEPS, gripper_cmd=env.gripper)
    final = scores(obs, registry)
    row.update(success=goal_met(final, goal), final_opening_mm=round(prim.opening(obs) * 1000, 1),
               final_scores={k: round(v, 3) for k, v in final.items()})
    if row["success"]:
        row["failure"] = None
    elif row["goal_before_settling"]:
        row["failure"] = "goal lost while settling"
    elif method == "targeted" and not row["grasp_achieved"]:
        row["failure"] = "grasp not achieved in 40 steps, so pull_open was not run"
    elif method == "targeted":
        row["failure"] = "pull_open did not open the drawer"
    else:
        row["failure"] = (f"goal not reached in 100 steps (furthest {row['max_opening_mm']:.0f} mm, handle_grasped "
                          f"{'held' if env.grasp_held else 'never held'})")
    return row


def evaluate(name: str, method: str, checkpoint: str, seeds, operator: str) -> tuple[str, list[dict]]:
    """One policy on every configuration, in its own process and simulator."""
    registry, goal = dp.registry(), dd.goal_from_operator(operator)
    policy = SAC.load(checkpoint, device="cpu")
    env = Recorder(prim.make_env(), registry, goal)
    rows = [episode(env, seed, method, policy, registry, goal) for seed in seeds]
    env.close()
    return name, rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoints() -> dict[str, dict]:
    """The eight selected checkpoints, each checked against its tracked manifest."""
    found = {}
    for method, folder, selected in (("targeted", STAGE3, "sac_hole_best.zip"), ("direct", DIRECT, "sac_direct_best.zip")):
        manifest = json.loads((folder / "checkpoints.json").read_text())
        for run, entry in sorted(manifest["runs"].items(), key=lambda item: item[1]["sac_seed"]):
            path = folder / run / selected
            recorded = entry["files"][selected]["sha256"]
            if sha256(path) != recorded:
                raise SystemExit(f"{path} is not the recorded checkpoint")
            found[f"{method}, SAC seed {entry['sac_seed']}"] = {"method": method, "path": str(path), "sha256": recorded}
    return found


def learning(name: str, method: str) -> dict:
    """From the validation curve recorded during training (seeds 500-509)."""
    folder, key = (STAGE3, "validation_success_rate") if method == "targeted" else (DIRECT, "validation_goal_rate")
    seed = int(name.rsplit(" ", 1)[1])
    for path in folder.glob("*/summary.json"):
        if json.loads((path.parent / "config.json").read_text())["seed"] == seed:
            history = json.loads(path.read_text())["history"]
            break
    rates = [(h["timesteps"], h[key]) for h in history]
    first = next((i for i, (_, r) in enumerate(rates) if r >= 0.9), None)
    selected = [h["timesteps"] for h in history if h.get("saved")][-1]
    return {"first_90_at_step": rates[first][0] if first is not None else None,
            "held_90_afterwards": first is not None and all(r >= 0.9 for _, r in rates[first:]),
            "best_validation": max(r for _, r in rates), "selected_at_step": selected,
            "final_validation": rates[-1][1], "curve": rates}


def summarise(rows: list[dict]) -> dict:
    wins = [r for r in rows if r["success"]]
    to_goal = sorted(r["steps_to_goal"] for r in wins)
    total = sorted(r["steps_total"] for r in wins)
    return {
        "success": len(wins), "configurations": len(rows),
        "failures": dict(Counter(r["failure"] for r in rows if not r["success"])),
        "steps_to_goal": [to_goal[0], int(np.median(to_goal)), to_goal[-1]] if wins else None,
        "steps_total": [total[0], int(np.median(total)), total[-1]] if wins else None,
        "final_opening_mm": ([min(r["final_opening_mm"] for r in wins), float(np.median([r["final_opening_mm"] for r in wins])),
                              max(r["final_opening_mm"] for r in wins)] if wins else None),
        "how_drawer_was_moved": dict(Counter(r["at_50mm"]["how"] for r in wins if r.get("at_50mm"))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=list(SEEDS))
    ap.add_argument("--operator", default="artifacts/drawer/stage1_operator.json")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("artifacts/drawer/final_targeted_vs_direct.json"))
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists; results are never overwritten")

    policies = checkpoints()
    print(f"{len(policies)} checkpoints verified; {len(args.seeds)} configurations, seeds {args.seeds[0]}-{args.seeds[-1]}")
    spawn = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=spawn) as pool:
        jobs = [pool.submit(evaluate, name, p["method"], p["path"], args.seeds, args.operator)
                for name, p in policies.items()]
        results = dict(job.result() for job in jobs)

    same_start = all(len({json.dumps(results[name][i]["start_state"]) for name in results}) == 1
                     for i in range(len(args.seeds)))
    summary = {name: summarise(rows) for name, rows in results.items()}
    curves = {name: learning(name, p["method"]) for name, p in policies.items()}

    n = len(args.seeds)
    print(f"start states identical for every policy on every configuration: {same_start}\n")
    print(f"{'policy':22s} {'success':>8} {'steps to goal':>14} {'final mm (median)':>18} {'first >=90% val':>16} "
          f"{'held':>5} {'best/final val':>14}  how the drawer was moved | failures")
    for name, s in summary.items():
        c = curves[name]
        steps = "/".join(str(k) for k in s["steps_to_goal"]) if s["steps_to_goal"] else "-"
        final = f"{s['final_opening_mm'][1]:.0f}" if s["final_opening_mm"] else "-"
        print(f"{name:22s} {s['success']:>5}/{n:<2} {steps:>14} {final:>18} {c['first_90_at_step']!s:>16} "
              f"{'yes' if c['held_90_afterwards'] else 'no':>5} {c['best_validation']:>6.0%}/{c['final_validation']:<6.0%}  "
              f"{s['how_drawer_was_moved']} | {s['failures'] or '-'}")

    args.out.write_text(json.dumps({
        "seeds": args.seeds, "goal": [sorted(g) for g in dd.goal_from_operator(args.operator)],
        "protocol": {"grasp_steps": GRASP_STEPS, "direct_steps": DIRECT_STEPS, "settle_steps": SETTLE_STEPS,
                     "success": "drawer_open and not drawer_closed after settling", "press_n": PRESS_N},
        "checkpoints": policies, "start_states_identical": same_start,
        "summary": summary, "learning": curves, "results": results}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
