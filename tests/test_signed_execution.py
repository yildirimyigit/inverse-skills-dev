"""Signed literals on the robot: the learning environment and the executives, in simulation.

These run the real simulator, a few seconds each. They check the machinery, not a learned
policy: scripted controllers and fixed actions stand in for policies.
"""

from __future__ import annotations

import numpy as np
import pytest

from inverse_skills.envs import drawer_domain as dd
from inverse_skills.envs import drawer_executive as dex
from inverse_skills.envs import drawer_predicates as dp
from inverse_skills.envs import drawer_primitives as dprim
from inverse_skills.envs import stow_domain as sd
from inverse_skills.envs import stow_executive as sex
from inverse_skills.envs import stow_primitives as sprim
from inverse_skills.envs.drawer_hole_env import DrawerHoleEnv
from inverse_skills.operators import hole_planner as hp
from inverse_skills.operators.literals import ROBUST_FALSE, neg, pos
from inverse_skills.operators.restoration import hole_reward

OPEN, CLOSED, STANDOFF, GRASPED = dd.OPEN, dd.CLOSED, dd.STANDOFF, dd.GRASPED
DRAWER_GOAL = ({OPEN}, {CLOSED, STANDOFF, GRASPED})          # the signed goal Stage 1 extracts
DRAWER_CONTROLLERS = {"approach_handle": dprim.approach_handle, "grasp_handle": dprim.test_grasp,
                      "pull_open": dprim.pull_open}
STOW_GOAL = ({"clear_of_walls(cube_a)", "gripper_open()", "in_region(cube_a,src_a)", "in_region(cube_b,src_b)"},
             {"in_region(cube_a,pocket)", "in_region(cube_b,mouth)", "tcp_near(cube_a)", "tcp_near(cube_b)"})


class Fixed:
    """A stand-in policy: the same action at every step."""

    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return self.action, None


# --- the reward ---------------------------------------------------------------------------------


def test_the_reward_rises_toward_either_target_and_charges_only_broken_fences():
    levels = [0.05, 0.2, 0.5, 0.8, 0.95]
    toward_p = [hole_reward(pos("t"), [], {"t": v}) for v in levels]
    toward_not_p = [hole_reward(neg("t"), [], {"t": v}) for v in levels]
    assert toward_p == sorted(toward_p) and toward_not_p == sorted(toward_not_p, reverse=True)
    assert toward_p[0] == pytest.approx(-toward_not_p[0])                 # the same formula, mirrored
    kept = {"t": 0.5, "f": 0.9, "n": 0.1}
    assert hole_reward(pos("t"), [pos("f"), neg("n")], kept) == pytest.approx(0.0)        # kept fences cost nothing
    broken = {"t": 0.5, "f": 0.3, "n": 0.7}
    assert hole_reward(pos("t"), [pos("f"), neg("n")], broken) == pytest.approx(-0.4 - 0.4)
    with pytest.raises(ValueError):
        hole_reward(pos("t"), [neg("t")], {"t": 0.5})


# --- a negative target in the drawer's learning environment ------------------------------------


def test_the_drawer_environment_learns_toward_a_negative_target():
    """A spec asking the hand to leave the standoff, keeping the handle not grasped. The episode
    starts at the standoff; rising with the gripper open must raise the reward, which is the
    framework's signed formula, until the target robustly holds."""
    approach = next(a for a in dd.domain().actions if a.name == "approach_handle(drawer)")
    spec = {"hole": f"HOLE[not {STANDOFF}]", "target": neg(STANDOFF).to_dict(), "achieve": STANDOFF,
            "object": "drawer", "start": [], "start_absent": [], "preserve": [], "preserve_absent": [GRASPED],
            "prefix": [approach.to_dict()], "suffix": [], "plan": [], "hole_index": 1, "cost": []}
    env = DrawerHoleEnv(spec, seed_pool=(0,))
    try:
        observation, info = env.reset(seed=0)
        start_score = info["scores"][STANDOFF]
        assert start_score > ROBUST_FALSE and info["handoff_valid"]          # not yet achieved
        assert observation[0] < 0                                            # the signed margin: not there yet
        rewards, scores = [], []
        for _ in range(env.max_steps):
            observation, reward, _, truncated, info = env.step(np.array([0.0, 0.0, 1.0, 1.0]))
            rewards.append(reward)
            scores.append(info["scores"][STANDOFF])
            assert reward == pytest.approx(hole_reward(neg(STANDOFF), {neg(GRASPED)}, info["scores"]))
            if truncated:
                break
        assert info["postcondition"] and scores[-1] <= ROBUST_FALSE and info["fences_held"]
        assert rewards[-1] > rewards[0] and observation[0] > 0
    finally:
        env.close()


# --- the drawer executive and a mixed signed goal -----------------------------------------------


@pytest.fixture(scope="module")
def drawer_env():
    env = dprim.make_env()
    yield env
    env.close()


def test_the_executive_judges_a_mixed_signed_goal_robustly(drawer_env):
    obs = dprim.push_close(drawer_env, dprim.reset(drawer_env, 0))
    domain = dd.domain(extra_actions=[dd.scripted_grasp()])
    plan = hp.plan(domain, hp.symbolic_state(dex.scores(obs, dp.registry())), *DRAWER_GOAL)
    assert plan.actions == ("approach_handle(drawer)", "grasp_handle(drawer)", "pull_open(drawer)")
    obs, report = dex.execute_plan(drawer_env, obs, list(plan.steps), domain, DRAWER_GOAL, DRAWER_CONTROLLERS)
    assert all(step.ok for step in report.steps)
    assert report.goal_met and report.goal_met_boolean and report.goal_unmet == []
    assert report.final_scores[STANDOFF] <= ROBUST_FALSE and report.final_scores[GRASPED] <= ROBUST_FALSE
    # A goal that also wants the hand at the standoff is not met, and says which literal fails.
    _, other = dex.execute_plan(drawer_env, obs, [], domain, ({OPEN, STANDOFF}, {CLOSED}), DRAWER_CONTROLLERS)
    assert not other.goal_met and other.goal_unmet == [STANDOFF]


def test_the_executive_runs_a_learned_step_that_owes_a_negative_literal(drawer_env):
    """After a scripted grasp, a learned step for `not handle_grasped`: a fixed action that opens
    the gripper and rises. The executive stops it once the literal robustly holds and checks the
    effect its model predicts, the grasp gone."""
    obs = dprim.push_close(drawer_env, dprim.reset(drawer_env, 1))
    domain = dd.domain(extra_actions=[dd.scripted_grasp()])
    by_name = {a.name: a for a in domain.actions}
    release = domain.hole(neg(GRASPED), available=set())
    assert release.delete == {GRASPED} and release.add == frozenset()
    steps = [by_name["approach_handle(drawer)"], by_name["grasp_handle(drawer)"], release]
    obs, report = dex.execute_plan(drawer_env, obs, steps, domain, (set(), {GRASPED}), DRAWER_CONTROLLERS,
                                   learned={neg(GRASPED): Fixed([0.0, 0.0, 1.0, 1.0])})
    assert [s.ok for s in report.steps] == [True, True, True]
    assert report.final_scores[GRASPED] <= ROBUST_FALSE and report.goal_met


# --- the stow library's retreat -----------------------------------------------------------------


def test_retreat_to_safe_is_modelled_as_measured_and_placed_by_the_planner():
    action = sd.retreat_to_safe()
    assert action.precondition_literals == {pos("gripper_open()"), neg("holding(cube_a)"), neg("holding(cube_b)")}
    assert action.effect_literals == {neg("tcp_near(cube_a)"), neg("tcp_near(cube_b)")}
    assert action in sd.domain().actions
    plan = hp.plan(sd.domain(), sd.stowed_state(), *STOW_GOAL)
    assert plan.actions[-1] == "retreat_to_safe" and plan.planning_cost == (1, 0, 7)


def test_retreat_to_safe_runs_through_the_executive_and_leaves_both_cubes():
    env = sprim.make_env()
    try:
        obs = sprim.forward_stow(env, sprim.reset(env, 1000))
        regions = sprim.regions(obs)
        domain = sd.domain()
        by_name = {a.name: a for a in domain.actions}
        steps = [by_name["pick(cube_b)"], by_name["place(cube_b,src_b)"], by_name["retreat_to_safe"]]
        obs, report = sex.execute_plan(env, obs, steps, regions=regions, domain=domain,
                                       goal=({"in_region(cube_b,src_b)", "gripper_open()"}, {"tcp_near(cube_b)"}))
        assert [s.ok for s in report.steps] == [True, True, True]
        assert report.final_scores["tcp_near(cube_b)"] <= ROBUST_FALSE
        assert report.final_scores["gripper_open()"] >= 0.8 and report.goal_met
    finally:
        env.close()
