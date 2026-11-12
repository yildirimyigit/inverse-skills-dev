"""Round 2's frozen protocol (artifacts/round2_20261112/PLAN.md).

The direct baseline's reward B', the two checkpoint-selection rules, the Stow stopping point
and margins, the full-horizon direct evaluation, the targeted policies' complete-goal
validation measure, the seed sets, and the Stow validation handoff's equivalence with
Stage 4's. Simulation runs on training scenes only.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.envs import drawer_primitives as dprim
from inverse_skills.envs import stow_primitives as sprim
from inverse_skills.envs.drawer_direct_env import DrawerDirectEnv, direct_reward
from inverse_skills.envs.drawer_hole_env import DrawerHoleEnv, load_spec
from inverse_skills.envs.stow_hole_env import StowHoleEnv
from inverse_skills.operators.literals import join, neg, pos

ROOT = Path(__file__).resolve().parent.parent
E2 = ROOT / "artifacts/signed_literals_20261111/E2_stage2"
OPEN, CLOSED, STANDOFF, GRASPED = dd.OPEN, dd.CLOSED, dd.STANDOFF, dd.GRASPED
GOAL_SETS = ({OPEN}, {CLOSED, STANDOFF, GRASPED})            # the Drawer's signed restoration goal
GOAL = join(*GOAL_SETS)
CLEAR, SRC_B, MOUTH_B = "clear_of_walls(cube_a)", "in_region(cube_b,src_b)", "in_region(cube_b,mouth)"
STOW_TARGET, STOW_FENCES = pos(CLEAR), {pos(SRC_B), neg(MOUTH_B)}


def script(name: str):
    """A script from scripts/, imported by name, as the scripts import each other."""
    if str(ROOT / "scripts") not in sys.path:
        sys.path.insert(0, str(ROOT / "scripts"))
    return importlib.import_module(name)


def round2(name: str):
    """A script from the Round 2 folder."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "artifacts/round2_20261112" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Fixed:
    """A stand-in policy: the same action at every step."""

    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return self.action, None


class Counting(gym.Wrapper):
    """Counts the simulator steps taken through it."""

    def __init__(self, env):
        super().__init__(env)
        self.count = 0

    def step(self, action):
        self.count += 1
        return self.env.step(action)


def drawer(open_, closed=0.05, standoff=0.05, grasped=0.05) -> dict[str, float]:
    return {OPEN: open_, CLOSED: closed, STANDOFF: standoff, GRASPED: grasped}


def stow(clear, src_b=0.95, mouth_b=0.05) -> dict[str, float]:
    return {CLEAR: clear, SRC_B: src_b, MOUTH_B: mouth_b}


def direct_spec() -> dict:
    """The E2 direct spec with its fences removed, as the Round 2 spec script writes it."""
    spec = json.loads((E2 / "direct/hole_spec_direct.json").read_text())
    spec["preserve"], spec["preserve_absent"] = [], []
    return spec


# --- B' ------------------------------------------------------------------------------------------


def test_b_prime_pays_the_dense_target_and_one_where_the_whole_goal_holds():
    target = pos(OPEN)
    assert direct_reward(target, GOAL, drawer(0.9)) == pytest.approx(1.8)                 # 0.8 + 1
    assert direct_reward(target, GOAL, drawer(0.8, 0.2, 0.2, 0.2)) == pytest.approx(1.6)  # thresholds included
    assert direct_reward(target, GOAL, drawer(0.79)) == pytest.approx(0.58)               # drawer_open not robust
    assert direct_reward(target, GOAL, drawer(0.9, grasped=0.21)) == pytest.approx(0.8)   # a negative literal not robust
    # A grasp costs nothing unless it breaks the goal.
    assert direct_reward(target, GOAL, drawer(0.5, grasped=0.95)) == direct_reward(target, GOAL, drawer(0.5))
    # Per step, every goal state earns more than any state outside the goal.
    assert min(direct_reward(target, GOAL, drawer(v, 0.2, 0.2, 0.2)) for v in (0.8, 0.9, 1.0)) > \
        max(direct_reward(target, GOAL, drawer(v, grasped=0.9)) for v in (0.0, 0.5, 1.0))


def test_the_direct_environment_refuses_a_spec_with_fences():
    fenced = json.loads((E2 / "direct/hole_spec_direct.json").read_text())
    assert fenced["preserve_absent"] == [GRASPED]          # the previous protocol's fence
    with pytest.raises(ValueError, match="no fences"):
        DrawerDirectEnv(fenced, GOAL_SETS)


# --- checkpoint selection --------------------------------------------------------------------------


def test_direct_selection_uses_the_settled_rate_and_the_earliest_among_equals():
    train = script("drawer_direct_train")
    history = [{"timesteps": 2500, "validation_goal_rate": 1.0, "validation_settled_goal_rate": 0.6},
               {"timesteps": 5000, "validation_goal_rate": 0.7, "validation_settled_goal_rate": 0.9},
               {"timesteps": 7500, "validation_goal_rate": 0.9, "validation_settled_goal_rate": 0.9}]
    assert train.select_checkpoint(history)["timesteps"] == 5000


def test_stow_selection_ranks_the_stop_contract_then_p10_then_the_latest_never_speed():
    stage3 = script("stow_stage3_train")
    history = [  # the old rule (contract at step 40, then fewest mean steps) picks 4,000
        {"timesteps": 4000, "contract_rate_at_stop": 1.0, "p10_margin_at_40": 0.02, "contract_rate_at_40": 1.0, "mean_steps": 3.7},
        {"timesteps": 8000, "contract_rate_at_stop": 0.95, "p10_margin_at_40": 0.15, "contract_rate_at_40": 1.0, "mean_steps": 3.0},
        {"timesteps": 12000, "contract_rate_at_stop": 1.0, "p10_margin_at_40": 0.10, "contract_rate_at_40": 0.9, "mean_steps": 6.0},
        {"timesteps": 16000, "contract_rate_at_stop": 1.0, "p10_margin_at_40": 0.10, "contract_rate_at_40": 0.85, "mean_steps": 9.0}]
    assert stage3.select(history)["timesteps"] == 16000
    assert stage3.select(history[:3])["timesteps"] == 12000


# --- the Stow stopping point and margins ---------------------------------------------------------


def test_the_stopping_point_is_the_first_robust_target_and_the_contract_is_judged_there():
    stage3 = script("stow_stage3_train")
    trajectory = [stow(0.1), stow(0.5), stow(0.85), stow(0.95)]
    judged = stage3.judge(trajectory, STOW_TARGET, STOW_FENCES)
    assert judged["stop_step"] == 2 and judged["contract_at_stop"] and judged["contract_at_40"]
    assert judged["margin_at_40"] == pytest.approx(0.15) and judged["margin_at_stop"] == pytest.approx(0.05)
    assert stage3.stopping_point([stow(0.9), stow(0.1)], STOW_TARGET) == 0           # the handoff itself counts
    # A fence broken at the stop fails the contract, even if it is whole again at step 40.
    judged = stage3.judge([stow(0.1), stow(0.85, src_b=0.7), stow(0.9)], STOW_TARGET, STOW_FENCES)
    assert not judged["contract_at_stop"] and judged["contract_at_40"]
    # A target never reached fails.
    judged = stage3.judge([stow(0.1), stow(0.5), stow(0.79)], STOW_TARGET, STOW_FENCES)
    assert judged["stop_step"] is None and not judged["contract_at_stop"] and not judged["contract_at_40"]


def test_margins_have_both_signs_and_the_contract_holds_exactly_when_the_minimum_is_not_negative():
    stage3 = script("stow_stage3_train")
    assert stage3.margin(pos("p"), 0.9) == pytest.approx(0.1)
    assert stage3.margin(neg("p"), 0.05) == pytest.approx(0.15)
    values = (0.0, 0.1, 0.19, 0.2, 0.21, 0.5, 0.79, 0.8, 0.81, 1.0)
    for clear in values:
        for src_b in values:
            for mouth_b in values:
                judged = stage3.judge([stow(clear, src_b, mouth_b)], STOW_TARGET, STOW_FENCES)
                assert judged["contract_at_40"] == (judged["margin_at_40"] >= 0.0)
    episodes = [{"contract_at_stop": True, "contract_at_40": True, "margin_at_40": m, "stop_step": 3,
                 "object_x_mm": 50.0, "handoff_valid": True} for m in np.linspace(-0.1, 0.2, 20)]
    summary = stage3.summarize(episodes, 40)
    assert summary["p10_margin_at_40"] == pytest.approx(np.percentile(np.linspace(-0.1, 0.2, 20), 10))


# --- seeds -------------------------------------------------------------------------------------------


def test_the_seed_sets_are_disjoint_even_through_re_rolls_and_the_audit_is_clean():
    stage3, stage4 = script("stow_stage3_train"), script("stow_stage4_execute")
    grasp, final = script("drawer_stage3_train"), script("drawer_final_evaluate")

    def disjoint(*sets):
        residues = [{s % 10_000 for s in group} for group in sets]
        return all(not (a & b) for i, a in enumerate(residues) for b in residues[i + 1:])

    assert disjoint(stage3.TRAIN_SEEDS, stage3.VALIDATION_SEEDS, stage3.TEST_SEEDS, stage4.EXTRACTION_SEEDS,
                    range(1000, 1010))
    assert disjoint(grasp.TRAIN_SEEDS, grasp.VALIDATION_SEEDS, grasp.TEST_SEEDS, range(2000, 2020),
                    range(3000, 3050), final.SEEDS)
    audit = json.loads((ROOT / "artifacts/round2_20261112/seed_audit.json").read_text())["candidates"]
    assert all(entry["clean"] for entry in audit.values())
    assert audit["stow_validation"]["seeds"] == f"{stage3.VALIDATION_SEEDS[0]}-{stage3.VALIDATION_SEEDS[-1]}"
    assert audit["stow_final_test"]["seeds"] == f"{stage3.TEST_SEEDS[0]}-{stage3.TEST_SEEDS[-1]}"
    assert audit["drawer_final_test"]["seeds"] == f"{final.SEEDS[0]}-{final.SEEDS[-1]}"


# --- simulation, training scenes only --------------------------------------------------------------


def test_the_direct_environment_pays_b_prime_at_every_step_and_never_ends_before_step_100():
    env = DrawerDirectEnv(direct_spec(), GOAL_SETS, seed_pool=(0,))
    try:
        env.reset(seed=0)
        for t in range(1, 101):
            _obs, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0, 0.0, 1.0]))
            assert not terminated and truncated == (t == 100)
            assert reward == pytest.approx(direct_reward(env.target, env.goal, info["scores"]))
        # The goal term on a measured goal state: what the scripted inverse leaves.
        raw = env._env
        obs = dprim.approach_handle(raw, dprim.push_close(raw, dprim.reset(raw, 0)))
        obs = dprim.pull_open(raw, dprim.test_grasp(raw, obs))
        scores = env._scores(obs)
        assert env.goal_met(scores)
        assert env._reward(scores) == pytest.approx(2.0 * scores[OPEN] - 1.0 + 1.0)
    finally:
        env.close()


@pytest.fixture(scope="module")
def drawer_start():
    """A Drawer scene at the handoff (training scene 0), stepped through a counter."""
    raw = Counting(dprim.make_env())
    obs = dprim.approach_handle(raw, dprim.push_close(raw, dprim.reset(raw, 0)))
    yield raw, obs, raw.unwrapped.get_state_dict()
    raw.close()


def test_the_final_evaluation_runs_the_direct_policy_for_all_100_steps_then_settles(drawer_start, monkeypatch):
    final = script("drawer_final_evaluate")
    raw, obs, state = drawer_start
    raw.unwrapped.set_state_dict(state)
    raw.count = 0
    monkeypatch.setattr(final, "goal_met", lambda _scores, _goal: True)    # the old rule would stop at once
    registry = dp.registry()
    row = final.inverse(final.Recorder(raw, registry, GOAL_SETS), obs, "direct", Fixed([0.0, 0.0, 0.0, 1.0]),
                        registry, GOAL_SETS)
    assert row["learned_steps"] == 100 and raw.count == 100 + final.SETTLE_STEPS


def test_a_targeted_policy_that_never_grasps_gets_no_pull_open_and_the_same_settling(drawer_start):
    final = script("drawer_final_evaluate")
    raw, obs, state = drawer_start
    raw.unwrapped.set_state_dict(state)
    raw.count = 0
    registry = dp.registry()
    row = final.inverse(final.Recorder(raw, registry, GOAL_SETS), obs, "targeted", Fixed([0.0, 0.0, 1.0, 1.0]),
                        registry, GOAL_SETS)
    assert not row["grasp_achieved"] and "pull_open_opened" not in row and not row["success"]
    assert row["learned_steps"] == final.GRASP_STEPS and raw.count == final.GRASP_STEPS + final.SETTLE_STEPS


def test_the_targeted_complete_goal_validation_runs_the_final_evaluation_procedure():
    grasp = script("drawer_stage3_train")
    env = DrawerHoleEnv(load_spec(E2 / "drawer_hole_spec.json"), seed_pool=(0,))
    try:
        measured = grasp.task_rollout(env, (0,), Fixed([0.0, 0.0, 1.0, 1.0]), dp.registry(), GOAL_SETS)
    finally:
        env.close()
    episode = measured["episodes"][0]
    assert measured["success_rate"] == 0.0 and episode["grasp_achieved"] is False
    assert episode["learned_steps"] == 40 and episode["failure"].startswith("grasp not achieved")


def test_the_stow_validation_start_is_stage_4s_handoff_bitwise():
    check = round2("handoff_check")
    spec = load_spec(E2 / "stow_hole_spec.json")
    env4, envv = sprim.make_env(), StowHoleEnv(spec, seed_pool=(0,), exact_handoff=True)
    try:
        row = check.compare(0, spec, env4, envv)
    finally:
        env4.close()
        envv.close()
    assert row["stage4_prefix_failure"] is None and row["handoff_valid"]
    assert row["state_differs_at_handoff"] == [] and row["fingers_equal"] and row["contacts_equal"]
    assert row["stop_step_stage4"] is not None and row["stop_step_stage4"] == row["stop_step_validation"]
    assert row["trajectory_steps_differing"] == [] and row["scores_at_stop_equal"] and row["equal"]
