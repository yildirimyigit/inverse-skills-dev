"""End-to-end inverse skill on PushCube-v1 with multi-predicate residual RL.

The symbolic inverse performs the coarse place(source): it picks the cube,
carries it back, and opens the gripper near the source. The RL phase then
maximizes a generic, framework-derived reward built from the inverse-target
predicates' own normalized signed margins, with one mode per term:

  R(s) = Σ_active  sign · tanh(margin/scale)         in [-1, +1]
       + Σ_fences  min(0, sign · tanh(margin/scale)) in [-1,  0]

For PushCube the inverse-target terms and modes are
  • at_pose(cube, init_pose)        sign=+1, scale=3·tolerance, bipolar  (active residual)
  • gripper_open()                  sign=+1, scale=min_width,   fence
  • ¬at_pose(cube, forward_goal)    sign=-1, scale=tolerance,   fence
  • tcp_near(cube)                  sign=+1, scale=5cm,          fence    (forward-push precondition)

The active residual provides bipolar shaping toward satisfaction. Fences are
silent when the predicate holds and only impose negative reward when it is
violated — the literal reading of "preconditions and delete-effects should
hold" with no perverse incentive to over-saturate a fence at the expense of
the residual. RL keeps 4D action authority; the reward, not the action space,
enforces non-violation.

Outputs:
  artifacts/out/<run_id>/planrob_inverse_rl_pushcube_full_*.{json,png,zip}

NOTE: filename retains "_2d_action" for git history continuity; the action
space is back to 4D (xyz delta + gripper).
"""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import gymnasium as gym
import matplotlib
import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

matplotlib.use("Agg")

warnings.filterwarnings("ignore")

import mani_skill.envs  # noqa: E402,F401  (registers PushCube)

from inverse_skills.predicates import GripperOpenPredicate, TcpNearObjectPredicate  # noqa: E402
from inverse_skills.operators.restoration import signed_margin_reward  # noqa: E402

# Reuse helpers + RL config from canonical PickCube demo + PushCube oracle.
_spec = importlib.util.spec_from_file_location("demo", "scripts/planrob_inverse_rl_demo.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)

_spec_pc = importlib.util.spec_from_file_location("pc", "scripts/planrob_inverse_rl_pushcube_demo.py")
pc = importlib.util.module_from_spec(_spec_pc)
_spec_pc.loader.exec_module(pc)


# ── Config (inherited from Phase 2 + PushCube oracle) ────────────────────────


_SAC_SEED = demo._SAC_SEED
_EVAL_SEEDS = demo._EVAL_SEEDS
_CURRICULUM_START_TOL = demo._CURRICULUM_START_TOL
_CURRICULUM_END_TOL = demo._CURRICULUM_END_TOL
_CURRICULUM_SCHEDULE_STEPS = demo._CURRICULUM_SCHEDULE_STEPS
# Keep the proven 0.2 scale (matches the canonical demo). At 0.4 a single bad
# step can shove the cube past every fence's saturation range, after which
# gradient = 0 and the policy can't recover. "Gentle pushes" = small per-step
# authority + many steps; max_steps=60 below provides the time budget.
_ACTION_SCALE_XYZ = 0.2
_PERTURBATION_RANGE_M = demo._PERTURBATION_RANGE_M
_IDENTITY_QUAT = demo._IDENTITY_QUAT
_TABLE_LOWER = demo._TABLE_LOWER
_TABLE_UPPER = demo._TABLE_UPPER

# PushCube oracle config. The true ManiSkill goal is a 20cm +X displacement;
# for several seeds that leaves the cube outside the reachable workspace of
# the scripted pick/place inverse. This 2D correction experiment assumes the
# symbolic inverse has actually restored the cube near source, so it keeps the
# forward skill in the reachable fixed-displacement regime by default.
# Override pc's 3cm default — 10cm gives a visibly meaningful push for the
# video while staying inside Panda's reachable workspace for every PushCube-v1
# init_pos (cube ends at most ~+0.20 in x, well clear of the IK-stall regime
# at +0.18+ that broke env-goal-x mode). Fixed displacement = same residual
# distribution every episode → reliable training distribution.
_PUSH_DISPLACEMENT_M = 0.10
_FORWARD_PUSH_SCALE = pc._FORWARD_PUSH_SCALE
_USE_ENV_GOAL_FOR_PUSH = True

# Diagnostics for gripper_open, which the symbolic inverse should restore
# before RL starts.
_GRIPPER_OPEN_TEMP_TRAIN = 0.02   # diagnostic score during training episodes
_GRIPPER_OPEN_TEMP_EVAL = 0.005
_GRIPPER_OPEN_MIN_WIDTH = 0.04    # gripper_width threshold for "open"
_TCP_NEAR_THRESHOLD_M = 0.05       # tcp considered "near" cube within 5cm
_TCP_NEAR_TEMP = 0.02
_MAX_STEPS = 60                    # room for residual correction after release
_TERMINATION_TOL_EPS = 1e-9
_OBSERVATION_MODE = "predicate_grounded"
_ABSOLUTE_OBS_LABELS = (
    "cube_x", "cube_y", "cube_z",
    "tcp_x", "tcp_y", "tcp_z",
    "gripper_width",
    "goal_x", "goal_y", "goal_z",
)
_PREDICATE_GROUNDED_OBS_LABELS = (
    "goal_minus_cube_x_by_tol",
    "goal_minus_cube_y_by_tol",
    "goal_minus_cube_z_by_tol",
    "cube_minus_tcp_x_by_tol",
    "cube_minus_tcp_y_by_tol",
    "cube_minus_tcp_z_by_tol",
    "gripper_width_by_max",
    "at_pose_score",
    "at_pose_margin_by_tol",
    "cube_goal_dist_by_tol",
    "gripper_open_score",
    "fwd_goal_minus_cube_dist_by_tol",
)
_OBS_LABELS_BY_MODE = {
    "absolute": _ABSOLUTE_OBS_LABELS,
    "predicate_grounded": _PREDICATE_GROUNDED_OBS_LABELS,
}
_MAX_GRIPPER_WIDTH_M = 0.08

# Perturbation curriculum. Start deterministic (no handoff noise), ramp to the
# full ±2cm range so the agent first encounters a clean residual signal and
# only later has to generalize across handoff variability.
_PERTURBATION_CURRICULUM_START_M = 0.0
_PERTURBATION_CURRICULUM_END_M = _PERTURBATION_RANGE_M
_PERTURBATION_CURRICULUM_SCHEDULE_STEPS = 200_000

# Saturation-scale multiplier for the *active* residual term. Fences saturate
# fast (scale=tolerance) so they only contribute gradient when violated. The
# active term needs a wider linear region so gradient remains usable across
# the full residual range (cube up to ~6 cm off init); without this, tanh
# kills the gradient past ~3·tolerance and the policy gets stuck.
_ACTIVE_RESIDUAL_SCALE_FACTOR = 3.0

# Scripted-prefix validity thresholds. After each scripted phase the env
# checks the cube's actual position against the expected one. If either
# phase fails its check, reset() re-samples the seed and re-runs the
# scripted setup. This guarantees the RL phase always starts from a clean
# handoff, at the cost of variable wall time per reset.
_FORWARD_PUSH_TOLERANCE_FRAC = 0.30   # cube within 30% of push_dx of forward goal xy
_FORWARD_PUSH_TOLERANCE_MIN_M = 0.02  # but at least 2cm
_SYMBOLIC_HANDOFF_TOLERANCE_M = 0.015  # cube within 1.5cm of init+perturbation
_MAX_SCRIPTED_ATTEMPTS = 50


# ── Env ──────────────────────────────────────────────────────────────────────


class PushCubeRecoveryFullEnv(gym.Env):
    """PushCube-v1 inverse env with the generic multi-predicate residual reward.

    Reward       = Σ_p sign_p · margin_p(scene) / scale_p
                   over the inverse-target predicates
                   {at_pose(cube, init), gripper_open, ¬at_pose(cube, fwd_goal)}.
    Termination  = V_at_pose ≥ success_threshold AND V_gripper_open ≥ success_threshold
                   (only after curriculum reaches final tolerance).
    Action       = 4D xyz delta + gripper command (full pd_ee_delta_pos space).
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = _MAX_STEPS, success_threshold: float = 0.50,
                 atpose_tolerance: float = _CURRICULUM_START_TOL,
                 atpose_temperature: float = 0.015,
                 action_scale_xyz: float = _ACTION_SCALE_XYZ,
                 perturbation_range_m: float = _PERTURBATION_RANGE_M,
                 push_displacement_m: float = _PUSH_DISPLACEMENT_M,
                 use_env_goal_for_push: bool = _USE_ENV_GOAL_FOR_PUSH,
                 observation_mode: str = _OBSERVATION_MODE,
                 render_mode: str | None = None):
        super().__init__()
        if observation_mode not in _OBS_LABELS_BY_MODE:
            valid = ", ".join(sorted(_OBS_LABELS_BY_MODE))
            raise ValueError(f"Unknown observation_mode={observation_mode!r}; expected one of: {valid}")
        self._env = gym.make(
            "PushCube-v1", obs_mode="state_dict",
            control_mode="pd_ee_delta_pos", render_mode=render_mode,
            max_episode_steps=600,
        )
        self.max_steps = max_steps
        self.success_threshold = success_threshold
        self._current_tolerance = atpose_tolerance
        self.atpose_temperature = atpose_temperature
        self.action_scale_xyz = action_scale_xyz
        self.perturbation_range_m = perturbation_range_m
        self.push_displacement_m = push_displacement_m
        self.use_env_goal_for_push = use_env_goal_for_push
        self.observation_mode = observation_mode
        self.regions = {"table_surface": demo.Region("table_surface", _TABLE_LOWER, _TABLE_UPPER)}

        # Predicates — the gripper_open and tcp_near ones are non-mutating, set once.
        self.gripper_open_train = GripperOpenPredicate(
            min_width=_GRIPPER_OPEN_MIN_WIDTH, temperature=_GRIPPER_OPEN_TEMP_TRAIN)
        self.gripper_open_eval = GripperOpenPredicate(
            min_width=_GRIPPER_OPEN_MIN_WIDTH, temperature=_GRIPPER_OPEN_TEMP_EVAL)
        self.tcp_near_pred = TcpNearObjectPredicate(
            object_name="cube",
            distance_threshold=_TCP_NEAR_THRESHOLD_M,
            temperature=_TCP_NEAR_TEMP,
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(self.obs_labels()),), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.init_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.forward_goal_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.at_pose_pred: demo.AtPosePredicate | None = None
        self.at_pose_eval: demo.AtPosePredicate | None = None
        self.at_pose_forward_goal_pred: demo.AtPosePredicate | None = None
        self._inverse_target_terms: list[tuple] = []
        self._last_obs = None
        self._step_count = 0
        self._episode_tolerance = float(atpose_tolerance)
        self._last_perturbation: np.ndarray = np.zeros(2, dtype=np.float32)
        self._last_forward_goal_pos: np.ndarray | None = None
        self._last_forward_push_displacement_m = float(push_displacement_m)

    def set_curriculum_tolerance(self, value: float) -> None:
        self._current_tolerance = float(value)

    def set_curriculum_perturbation(self, value: float) -> None:
        self.perturbation_range_m = max(0.0, float(value))

    def obs_labels(self) -> tuple[str, ...]:
        return _OBS_LABELS_BY_MODE[self.observation_mode]

    def _termination_enabled(self) -> bool:
        return self._episode_tolerance <= _CURRICULUM_END_TOL + _TERMINATION_TOL_EPS

    def _normalized_at_pose_margin(self, margin: float) -> float:
        return float(margin) / max(float(self._episode_tolerance), 1e-6)

    def _resolve_forward_push_displacement(self, obs, cube_init: np.ndarray) -> float:
        """Use ManiSkill's per-episode absolute goal if available, otherwise
        fall back to the legacy fixed PushCube displacement."""
        goal_pos_tensor = obs.get("extra", {}).get("goal_pos")
        if self.use_env_goal_for_push and goal_pos_tensor is not None:
            goal_pos = goal_pos_tensor.squeeze()[:3].cpu().numpy().astype(np.float32)
            self._last_forward_goal_pos = goal_pos.copy()
            self._last_forward_push_displacement_m = float(goal_pos[0] - cube_init[0])
        else:
            self._last_forward_goal_pos = None
            self._last_forward_push_displacement_m = float(self.push_displacement_m)
        return self._last_forward_push_displacement_m

    # --- forward + symbolic phases (same as plain PushCube) ---------------

    def _run_forward_push(self, obs):
        cube_init = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()
        push_dx = self._resolve_forward_push_displacement(obs, cube_init)
        behind_push = cube_init.copy(); behind_push[0] -= 0.06; behind_push[2] = cube_init[2] + 0.005
        obs = pc._step_toward_scaled(self._env, obs, behind_push, 30, 0.012, gripper_cmd=-1.0, scale=1.0)
        push_tgt = cube_init.copy(); push_tgt[0] += push_dx; push_tgt[2] += 0.005
        obs = pc._step_toward_scaled(self._env, obs, push_tgt, 30, 0.005, gripper_cmd=-1.0,
                                      scale=_FORWARD_PUSH_SCALE)
        tcp = obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
        up = tcp.copy(); up[2] += 0.10
        obs = pc._step_toward_scaled(self._env, obs, up, 20, 0.012, gripper_cmd=-1.0, scale=1.0)
        obs = pc._step_in_place(self._env, obs, 8, gripper_cmd=1.0)
        return obs

    def _run_symbolic_inverse(self, obs):
        cube_now = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()
        # EEF-only motion (no cube held yet) keeps full scale=1.0 for speed.
        above_cube = cube_now.copy(); above_cube[2] += 0.10
        obs = pc._step_toward_scaled(self._env, obs, above_cube, 20, 0.012, gripper_cmd=1.0, scale=1.0)
        obs = pc._step_toward_scaled(self._env, obs, cube_now.copy(), 20, 0.012, gripper_cmd=1.0, scale=1.0)
        # Close gripper, then dwell extra steps so the grasp force settles
        # before any motion that could shake the cube loose.
        obs = pc._step_in_place(self._env, obs, 10, gripper_cmd=-1.0)
        obs = pc._step_in_place(self._env, obs, 5, gripper_cmd=-1.0)

        # All cube-held motion uses scale=0.5 with proportionally larger step
        # budgets. Lower scale → gentler controller demand per step → less
        # cube slip during the long carry. Each step still early-exits at
        # `tol` so short displacements are unchanged in wall time.
        lift_h = cube_now.copy(); lift_h[2] += 0.15
        obs = pc._step_toward_scaled(self._env, obs, lift_h, 25, 0.02, gripper_cmd=-1.0, scale=0.5)
        over_init = np.array([self.init_pos[0], self.init_pos[1], lift_h[2]], dtype=np.float32)
        # Carry budget sized for the env-goal-x case (~20cm traverse) at
        # scale=0.5. _step_toward_scaled early-exits at tol, so short hops
        # are not penalized.
        obs = pc._step_toward_scaled(self._env, obs, over_init, 80, 0.012, gripper_cmd=-1.0, scale=0.5)
        if self.perturbation_range_m > 0.0:
            dx = float(self.np_random.uniform(-self.perturbation_range_m, self.perturbation_range_m))
            dy = float(self.np_random.uniform(-self.perturbation_range_m, self.perturbation_range_m))
        else:
            dx = dy = 0.0
        self._last_perturbation = np.array([dx, dy], dtype=np.float32)
        handoff = self.init_pos.copy()
        handoff[0] += dx; handoff[1] += dy; handoff[2] += 0.005
        obs = pc._step_toward_scaled(self._env, obs, handoff, 30, 0.012, gripper_cmd=-1.0, scale=0.5)
        obs = pc._step_in_place(self._env, obs, 10, gripper_cmd=1.0)
        obs = pc._step_in_place(self._env, obs, 10, gripper_cmd=1.0)
        return obs

    # --- gym API ------------------------------------------------------------

    def _attempt_scripted_setup(self, try_seed: int) -> tuple[dict, dict]:
        """Run one full scripted setup (env reset + forward push + symbolic
        inverse) and validate each phase. Returns the post-symbolic obs and
        an info dict with per-phase validity flags and error metrics.

        The caller (reset) loops over this until a fully-valid attempt is
        produced, so the RL phase always starts from a clean handoff.
        """
        obs, _ = self._env.reset(seed=try_seed)
        for _ in range(2):
            obs, *_ = self._env.step(torch.tensor(np.array([0., 0., 0., 1.], dtype=np.float32)))

        self._episode_tolerance = float(self._current_tolerance)
        self.init_pos = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().copy()
        init_pose = demo.Pose(position=self.init_pos.astype(np.float32), quat_xyzw=_IDENTITY_QUAT)
        self.at_pose_pred = demo.AtPosePredicate(
            "cube", target_pose=init_pose,
            distance_threshold=self._episode_tolerance, temperature=self.atpose_temperature)
        self.at_pose_eval = demo.AtPosePredicate(
            "cube", target_pose=init_pose,
            distance_threshold=_CURRICULUM_END_TOL, temperature=0.005)

        # Resolve THIS episode's forward push displacement before building the
        # forward-goal predicate. _run_forward_push() will call this again —
        # the function is idempotent.
        push_dx = self._resolve_forward_push_displacement(obs, self.init_pos)

        # Forward-goal predicate (the negated add-effect of the forward push).
        self.forward_goal_pos = self.init_pos.copy()
        self.forward_goal_pos[0] += float(push_dx)
        forward_goal_pose = demo.Pose(position=self.forward_goal_pos.astype(np.float32),
                                       quat_xyzw=_IDENTITY_QUAT)
        self.at_pose_forward_goal_pred = demo.AtPosePredicate(
            "cube", target_pose=forward_goal_pose,
            distance_threshold=self._episode_tolerance, temperature=self.atpose_temperature)

        active_scale = max(_ACTIVE_RESIDUAL_SCALE_FACTOR * self._episode_tolerance, 1e-6)
        fence_scale = max(self._episode_tolerance, 1e-6)
        self._inverse_target_terms = [
            (self.at_pose_pred,              +1.0, active_scale,             "bipolar"),
            (self.gripper_open_train,        +1.0, _GRIPPER_OPEN_MIN_WIDTH,  "fence"),
            (self.at_pose_forward_goal_pred, -1.0, fence_scale,              "fence"),
            (self.tcp_near_pred,             +1.0, _TCP_NEAR_THRESHOLD_M,    "fence"),
        ]

        # --- Forward push + check ----
        obs = self._run_forward_push(obs)
        cube_after_forward = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy()
        forward_target_xy = np.array([self.init_pos[0] + push_dx, self.init_pos[1]],
                                      dtype=np.float32)
        forward_err = float(np.linalg.norm(cube_after_forward[:2] - forward_target_xy))
        forward_tol = max(_FORWARD_PUSH_TOLERANCE_MIN_M,
                          _FORWARD_PUSH_TOLERANCE_FRAC * abs(push_dx))
        forward_ok = forward_err < forward_tol

        info = {
            "scripted_seed": int(try_seed),
            "push_dx_m": float(push_dx),
            "forward_err_m": forward_err,
            "forward_tol_m": float(forward_tol),
            "forward_ok": bool(forward_ok),
            "symbolic_err_m": float("nan"),
            "symbolic_ok": False,
            "scripted_valid": False,
        }
        if not forward_ok:
            return obs, info

        # --- Symbolic inverse + check ----
        obs = self._run_symbolic_inverse(obs)
        cube_after_symbolic = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy()
        handoff_target = self.init_pos + np.array(
            [self._last_perturbation[0], self._last_perturbation[1], 0.005],
            dtype=np.float32,
        )
        symbolic_err = float(np.linalg.norm(cube_after_symbolic - handoff_target))
        symbolic_ok = symbolic_err < _SYMBOLIC_HANDOFF_TOLERANCE_M

        info["symbolic_err_m"] = symbolic_err
        info["symbolic_ok"] = bool(symbolic_ok)
        info["scripted_valid"] = bool(forward_ok and symbolic_ok)
        return obs, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Retry the scripted prefix until both phases pass their validity
        # checks. The first attempt uses the caller-provided seed (if any);
        # subsequent attempts draw from self.np_random, so the retry sequence
        # is deterministic given the input seed.
        explicit_seed = seed
        attempt_log: list[dict] = []
        for attempt in range(_MAX_SCRIPTED_ATTEMPTS):
            try_seed = (explicit_seed if (attempt == 0 and explicit_seed is not None)
                        else int(self.np_random.integers(0, 2**31 - 1)))
            obs, attempt_info = self._attempt_scripted_setup(try_seed)
            attempt_log.append(attempt_info)
            if attempt_info["scripted_valid"]:
                break
        else:
            raise RuntimeError(
                f"Could not find a valid scripted scenario after "
                f"{_MAX_SCRIPTED_ATTEMPTS} attempts. Last attempt: {attempt_info}"
            )

        self._last_obs = obs
        self._step_count = 0
        initial_scene = demo._obs_to_scene(obs, self.regions)
        initial_at_pose = self.at_pose_pred.evaluate(initial_scene)
        initial_reward = signed_margin_reward(initial_scene, self._inverse_target_terms)
        return self._encode(obs), {
            "phase": "rl_start",
            "seed": explicit_seed,
            "scripted_seed": attempt_info["scripted_seed"],
            "scripted_attempts": len(attempt_log),
            "scripted_attempt_log": attempt_log,
            "forward_err_m": attempt_info["forward_err_m"],
            "symbolic_err_m": attempt_info["symbolic_err_m"],
            "init_pose": self.init_pos.tolist(),
            "forward_goal_pos_used_for_fence": self.forward_goal_pos.tolist(),
            "perturbation_xy": self._last_perturbation.tolist(),
            "perturbation_range_m": self.perturbation_range_m,
            "current_tolerance_m": self._current_tolerance,
            "episode_tolerance_m": self._episode_tolerance,
            "initial_at_pose_score": float(initial_at_pose.score),
            "initial_at_pose_margin_m": float(initial_at_pose.margin),
            "initial_reward": float(initial_reward),
            "termination_enabled": self._termination_enabled(),
            "use_env_goal_for_push": self.use_env_goal_for_push,
            "forward_goal_pos": None if self._last_forward_goal_pos is None
                else self._last_forward_goal_pos.tolist(),
            "forward_push_displacement_m": self._last_forward_push_displacement_m,
        }

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 4:
            raise ValueError(f"Expected 4D action (xyz delta + gripper), got shape {np.asarray(action).shape}")
        maniskill_action = np.array([
            a[0] * self.action_scale_xyz,
            a[1] * self.action_scale_xyz,
            a[2] * self.action_scale_xyz,
            a[3],
        ], dtype=np.float32)
        obs, _r, _term, _trunc, info = self._env.step(torch.tensor(maniskill_action))
        self._last_obs = obs
        self._step_count += 1

        scene = demo._obs_to_scene(obs, self.regions)
        at_pose_result = self.at_pose_pred.evaluate(scene)
        at_pose_score = float(at_pose_result.score)
        gripper_result = self.gripper_open_train.evaluate(scene)
        gripper_score = float(gripper_result.score)
        forward_goal_result = self.at_pose_forward_goal_pred.evaluate(scene)
        tcp_near_result = self.tcp_near_pred.evaluate(scene)
        cube_pos = scene.objects["cube"].pose.position
        distance = float(np.linalg.norm(cube_pos - self.init_pos))
        cube_to_forward_goal = float(np.linalg.norm(cube_pos - self.forward_goal_pos))

        # Generic predicate-derived reward — sum of normalized signed margins
        # over the inverse-target terms.
        reward = signed_margin_reward(scene, self._inverse_target_terms)

        termination_enabled = self._termination_enabled()
        terminated = (
            termination_enabled
            and at_pose_score >= self.success_threshold
            and gripper_score >= self.success_threshold
        )
        truncated = self._step_count >= self.max_steps
        return self._encode(obs), float(reward), terminated, truncated, {
            "v_residual": at_pose_score,
            "at_pose_margin_m": float(at_pose_result.margin),
            "normalized_at_pose_margin": float(at_pose_result.margin / max(self._episode_tolerance, 1e-6)),
            "v_gripper_open": gripper_score,
            "gripper_open_margin_m": float(gripper_result.margin),
            "v_at_pose_forward_goal": float(forward_goal_result.score),
            "fwd_goal_margin_m": float(forward_goal_result.margin),
            "cube_to_forward_goal_m": cube_to_forward_goal,
            "v_tcp_near": float(tcp_near_result.score),
            "tcp_near_margin_m": float(tcp_near_result.margin),
            "distance": distance,
            "current_tolerance_m": self._current_tolerance,
            "episode_tolerance_m": self._episode_tolerance,
            "termination_enabled": termination_enabled,
            "success": bool(terminated),
            "rl_action": a.tolist(),
            "maniskill_action": maniskill_action.tolist(),
            "step": self._step_count,
        }

    def _encode(self, obs) -> np.ndarray:
        if self.observation_mode == "predicate_grounded":
            return self._encode_predicate_grounded(obs)
        if self.observation_mode == "absolute":
            return self._encode_absolute(obs)
        raise RuntimeError(f"Unhandled observation_mode={self.observation_mode!r}")

    def _encode_absolute(self, obs) -> np.ndarray:
        cube_pos = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy()
        tcp_pos = obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
        qpos = obs["agent"]["qpos"].squeeze().cpu().numpy()
        gw = float(qpos[-2] + qpos[-1])
        return np.concatenate([cube_pos, tcp_pos, [gw], self.init_pos]).astype(np.float32)

    def _encode_predicate_grounded(self, obs) -> np.ndarray:
        cube_pos = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().astype(np.float32)
        tcp_pos = obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy().astype(np.float32)
        qpos = obs["agent"]["qpos"].squeeze().cpu().numpy()
        gw = float(qpos[-2] + qpos[-1])
        scene = demo._obs_to_scene(obs, self.regions)
        at_pose_result = self.at_pose_pred.evaluate(scene)
        gripper_result = self.gripper_open_train.evaluate(scene)
        scale = max(float(self._episode_tolerance), 1e-6)
        goal_minus_cube = (self.init_pos - cube_pos) / scale
        cube_minus_tcp = (cube_pos - tcp_pos) / scale
        cube_goal_dist = float(np.linalg.norm(cube_pos - self.init_pos)) / scale
        cube_fwd_goal_dist = float(np.linalg.norm(cube_pos - self.forward_goal_pos)) / scale
        return np.concatenate([
            goal_minus_cube.astype(np.float32),
            cube_minus_tcp.astype(np.float32),
            [gw / _MAX_GRIPPER_WIDTH_M],
            [float(at_pose_result.score)],
            [self._normalized_at_pose_margin(at_pose_result.margin)],
            [cube_goal_dist],
            [float(gripper_result.score)],
            [cube_fwd_goal_dist],
        ]).astype(np.float32)

    def close(self):
        self._env.close()


# ── Eval that reports gripper state too ──────────────────────────────────────


def _eval_episode_full(env: PushCubeRecoveryFullEnv, policy_fn, seed: int, do_rl: bool,
                        do_release: bool = True) -> dict:
    """Same as demo._eval_episode but additionally records gripper state at
    the post-RL and post-release moments."""
    obs, _info_reset = env.reset(seed=seed)
    rl_steps = 0
    if do_rl and policy_fn is not None:
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, _r, terminated, truncated, info = env.step(action)
            rl_steps += 1

    # Pre-release diagnostics
    pre_scene = demo._obs_to_scene(env._last_obs, env.regions)
    v_at_pose_pre = float(env.at_pose_eval.evaluate(pre_scene).score)
    v_gripper_pre = float(env.gripper_open_eval.evaluate(pre_scene).score)
    cube_pre = pre_scene.objects["cube"].pose.position
    pre_dist_mm = float(np.linalg.norm(cube_pre - env.init_pos)) * 1000.0
    pre_gripper_width = float(pre_scene.robot.gripper_width)

    if do_release:
        # Same slow-open release motion as the canonical demo.
        ms_obs = env._last_obs
        for _ in range(15):
            a = torch.tensor(np.array([0.0, 0.0, 0.0, 0.3], dtype=np.float32))
            ms_obs, *_ = env._env.step(a)
        for _ in range(3):
            a = torch.tensor(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            ms_obs, *_ = env._env.step(a)
        tcp = ms_obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy()
        retract = tcp.copy().astype(np.float32); retract[2] += 0.05
        ms_obs = pc._step_toward_scaled(env._env, ms_obs, retract, 10, 0.01, gripper_cmd=1.0, scale=1.0)
        env._last_obs = ms_obs

    post_scene = demo._obs_to_scene(env._last_obs, env.regions)
    cube_final = post_scene.objects["cube"].pose.position
    post_dist_mm = float(np.linalg.norm(cube_final - env.init_pos)) * 1000.0
    v_at_pose_post = float(env.at_pose_eval.evaluate(post_scene).score)
    v_gripper_post = float(env.gripper_open_eval.evaluate(post_scene).score)
    post_gripper_width = float(post_scene.robot.gripper_width)

    return {
        "seed": seed,
        "rl_steps": rl_steps,
        "pre_release_dist_mm": pre_dist_mm,
        "pre_release_v_at_pose": v_at_pose_pre,
        "pre_release_v_gripper_open": v_gripper_pre,
        "pre_release_gripper_width_mm": pre_gripper_width * 1000.0,
        "post_release_dist_mm": post_dist_mm,
        "v_at_pose_post": v_at_pose_post,
        "v_gripper_open_post": v_gripper_post,
        "post_release_gripper_width_mm": post_gripper_width * 1000.0,
        "distance_mm": post_dist_mm,
        "v_after_release": v_at_pose_post,  # for compatibility with _summarize
        "cube_final": cube_final.tolist(),
        "init_pose": env.init_pos.tolist(),
    }


def evaluate_with_release(env, model_predict_fn, seeds, label, do_rl=True):
    eps = [_eval_episode_full(env, model_predict_fn, s, do_rl=do_rl, do_release=True) for s in seeds]
    return demo._summarize(label, eps)


def evaluate_no_release(env, model_predict_fn, seeds, label, do_rl=True):
    """Eval that skips the scripted release — directly measure cube position
    after RL terminates. The strongest claim: 'RL alone solves the full inverse'."""
    eps = [_eval_episode_full(env, model_predict_fn, s, do_rl=do_rl, do_release=False) for s in seeds]
    return demo._summarize(label, eps)


# ── Policy I/O tracing ───────────────────────────────────────────────────────


def _policy_obs_summary(obs_vec: np.ndarray, labels: tuple[str, ...]) -> dict:
    obs_vec = np.asarray(obs_vec, dtype=np.float32)
    return {
        "raw": {label: float(value) for label, value in zip(labels, obs_vec)},
        "vector": obs_vec.astype(float).tolist(),
    }


def _physical_state_summary(env: PushCubeRecoveryFullEnv) -> dict:
    obs = env._last_obs
    cube = obs["extra"]["obj_pose"].squeeze()[:3].cpu().numpy().astype(float)
    tcp = obs["extra"]["tcp_pose"].squeeze()[:3].cpu().numpy().astype(float)
    qpos = obs["agent"]["qpos"].squeeze().cpu().numpy()
    goal = env.init_pos.astype(float)
    gripper_width = float(qpos[-2] + qpos[-1])
    cube_goal = cube - goal
    tcp_cube = tcp - cube
    return {
        "cube_pos_m": cube.tolist(),
        "tcp_pos_m": tcp.tolist(),
        "goal_pos_m": goal.tolist(),
        "gripper_width_m": gripper_width,
        "cube_minus_goal_m": cube_goal.tolist(),
        "tcp_minus_cube_m": tcp_cube.tolist(),
        "cube_goal_xy_dist_m": float(np.linalg.norm(cube_goal[:2])),
        "cube_goal_dist_m": float(np.linalg.norm(cube_goal)),
        "tcp_cube_xy_dist_m": float(np.linalg.norm(tcp_cube[:2])),
        "tcp_cube_dist_m": float(np.linalg.norm(tcp_cube)),
    }


def _unit_xy(delta_xy: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(delta_xy))
    if norm < 1e-8:
        return np.zeros(2, dtype=np.float32)
    return np.clip(delta_xy / norm, -1.0, 1.0).astype(np.float32)


def _pad_xy_to_4d(xy: np.ndarray, gripper: float = 1.0) -> np.ndarray:
    """Promote a 2D xy tracer action to the 4D action space (z=0, gripper=open)."""
    return np.array([float(xy[0]), float(xy[1]), 0.0, float(gripper)], dtype=np.float32)


def _trace_policy_action(policy: str, obs_vec: np.ndarray, rng: np.random.Generator,
                         physical: dict,
                         model: SAC | None = None) -> np.ndarray:
    if model is not None:
        action, _ = model.predict(obs_vec, deterministic=True)
        return np.asarray(action, dtype=np.float32)
    if policy == "zero":
        return np.zeros(4, dtype=np.float32)
    if policy == "random":
        return rng.uniform(-1.0, 1.0, size=4).astype(np.float32)

    cube_xy = np.asarray(physical["cube_pos_m"][:2], dtype=np.float32)
    tcp_xy = np.asarray(physical["tcp_pos_m"][:2], dtype=np.float32)
    goal_xy = np.asarray(physical["goal_pos_m"][:2], dtype=np.float32)

    if policy == "toward-cube":
        return _pad_xy_to_4d(_unit_xy(cube_xy - tcp_xy))

    if policy == "push-to-goal":
        goal_delta = goal_xy - cube_xy
        goal_dir = _unit_xy(goal_delta)
        if np.linalg.norm(goal_dir) < 1e-8:
            return np.zeros(4, dtype=np.float32)
        behind_cube = cube_xy - goal_dir * 0.035
        if np.linalg.norm(behind_cube - tcp_xy) > 0.012:
            return _pad_xy_to_4d(_unit_xy(behind_cube - tcp_xy))
        return _pad_xy_to_4d(goal_dir)

    raise ValueError(f"Unknown trace policy: {policy}")


def trace_policy_io(seed: int = 1000, steps: int = _MAX_STEPS,
                    policy: str = "zero", model_path: Path | None = None,
                    out_path: Path | None = None,
                    observation_mode: str = _OBSERVATION_MODE) -> dict:
    """Print and optionally save the exact policy input/output sequence.

    The policy input is the 10D vector from `_encode()`. This tracer also prints
    derived relative features that the MLP currently has to infer by subtraction.
    """
    env = PushCubeRecoveryFullEnv(
        max_steps=steps,
        atpose_tolerance=_CURRICULUM_END_TOL,
        action_scale_xyz=_ACTION_SCALE_XYZ,
        perturbation_range_m=_PERTURBATION_RANGE_M,
        push_displacement_m=_PUSH_DISPLACEMENT_M,
        observation_mode=observation_mode,
    )
    env.set_curriculum_tolerance(_CURRICULUM_END_TOL)
    model = SAC.load(model_path, env=env) if model_path is not None else None
    rng = np.random.default_rng(_SAC_SEED)

    obs, reset_info = env.reset(seed=seed)
    print("Policy observation vector:")
    for idx, label in enumerate(env.obs_labels()):
        print(f"  obs[{idx}] = {label}")
    reset_physical = _physical_state_summary(env)
    print(f"\nTrace seed={seed} policy={policy if model is None else f'model:{model_path}'}")
    print(
        f"observation_mode={env.observation_mode}\n"
        f"reset: init={np.round(env.init_pos, 4).tolist()}  "
        f"perturb_xy={reset_info['perturbation_xy']}  "
        f"initial_dist={reset_physical['cube_goal_dist_m'] * 1000.0:.1f}mm  "
        f"initial_score={reset_info['initial_at_pose_score']:.3f}"
    )
    print("initial_policy_obs=", np.array2string(obs, precision=4, suppress_small=False))
    print(
        "\nstep  cube_goal_xy(mm)  dist(mm)  tcp_cube_xy(mm)  "
        "action_xyzg                              reward    V_atP  V_grip  V_fwd  done"
    )

    trace = {
        "reset": reset_info,
        "observation_mode": env.observation_mode,
        "obs_labels": list(env.obs_labels()),
        "steps": [],
    }
    terminated = truncated = False
    for step_i in range(steps):
        before_policy = _policy_obs_summary(obs, env.obs_labels())
        before_physical = _physical_state_summary(env)
        action = _trace_policy_action(policy, obs, rng, physical=before_physical, model=model)
        next_obs, reward, terminated, truncated, info = env.step(action)
        after_policy = _policy_obs_summary(next_obs, env.obs_labels())
        after_physical = _physical_state_summary(env)
        done = terminated or truncated
        cube_goal_xy_mm = np.asarray(after_physical["cube_minus_goal_m"][:2]) * 1000.0
        tcp_cube_xy_mm = np.asarray(after_physical["tcp_minus_cube_m"][:2]) * 1000.0
        print(
            f"{step_i + 1:>4d}  "
            f"[{cube_goal_xy_mm[0]:>6.1f},{cube_goal_xy_mm[1]:>6.1f}]  "
            f"{after_physical['cube_goal_dist_m'] * 1000.0:>7.1f}  "
            f"[{tcp_cube_xy_mm[0]:>6.1f},{tcp_cube_xy_mm[1]:>6.1f}]  "
            f"[{action[0]:>6.3f},{action[1]:>6.3f},{action[2]:>6.3f},{action[3]:>6.3f}]  "
            f"{reward:>8.4f}  "
            f"{info['v_residual']:>5.3f}  "
            f"{info['v_gripper_open']:>6.3f}  "
            f"{info['v_at_pose_forward_goal']:>5.3f}  "
            f"{'T' if done else '-'}"
        )
        trace["steps"].append({
            "step": step_i + 1,
            "policy_obs_before": before_policy,
            "physical_before": before_physical,
            "action_xy": action.astype(float).tolist(),
            "maniskill_action": info["maniskill_action"],
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": {
                key: (float(value) if isinstance(value, (np.floating, float)) else value)
                for key, value in info.items()
            },
            "policy_obs_after": after_policy,
            "physical_after": after_physical,
        })
        obs = next_obs
        if done:
            break

    env.close()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
        print(f"\nSaved trace JSON: {out_path}")
    return trace


# ── Training ─────────────────────────────────────────────────────────────────


class PerturbationCurriculumCallback(BaseCallback):
    """Linearly ramps handoff perturbation from start to end over schedule_steps,
    then holds at end. Lets the agent first see a clean residual signal and
    only later generalize across handoff variability."""

    def __init__(self, env: PushCubeRecoveryFullEnv, start_m: float, end_m: float,
                 schedule_steps: int):
        super().__init__()
        self._env_ref = env
        self.start_m = float(start_m)
        self.end_m = float(end_m)
        self.schedule_steps = int(schedule_steps)

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / max(1, self.schedule_steps))
        value = self.start_m + (self.end_m - self.start_m) * progress
        self._env_ref.set_curriculum_perturbation(value)
        return True


def train(total_timesteps: int, checkpoint_path: Path, best_checkpoint_path: Path
          ) -> tuple[SAC, demo.CheckpointAndLogCallback, PushCubeRecoveryFullEnv]:
    env = PushCubeRecoveryFullEnv(
        max_steps=_MAX_STEPS,
        atpose_tolerance=_CURRICULUM_START_TOL,
        action_scale_xyz=_ACTION_SCALE_XYZ,
        perturbation_range_m=_PERTURBATION_CURRICULUM_START_M,
        push_displacement_m=_PUSH_DISPLACEMENT_M,
    )
    model = SAC(
        "MlpPolicy", env, verbose=0,
        learning_rate=3e-4, buffer_size=100_000,
        learning_starts=1_000, batch_size=256,
        tau=0.005, gamma=0.99,
        policy_kwargs={"net_arch": [256, 256]},
        seed=_SAC_SEED,
    )
    log_cb = demo.CheckpointAndLogCallback(
        checkpoint_path=checkpoint_path,
        best_checkpoint_path=best_checkpoint_path,
        checkpoint_every=50_000, log_every=5_000, best_window=50, env=env,
    )
    curr_cb = demo.CurriculumCallback(
        env, start_tol=_CURRICULUM_START_TOL, end_tol=_CURRICULUM_END_TOL,
        schedule_steps=_CURRICULUM_SCHEDULE_STEPS,
    )
    perturb_cb = PerturbationCurriculumCallback(
        env,
        start_m=_PERTURBATION_CURRICULUM_START_M,
        end_m=_PERTURBATION_CURRICULUM_END_M,
        schedule_steps=_PERTURBATION_CURRICULUM_SCHEDULE_STEPS,
    )
    callbacks = CallbackList([log_cb, curr_cb, perturb_cb])
    print(f"Training SAC for {total_timesteps} timesteps "
          f"(PushCube-v1, generic multi-predicate residual reward)...")
    print(f"  reward = active(at_pose, bipolar) + fences(gripper_open, ¬at_pose(fwd_goal), tcp_near, all min(·,0))  "
          f"termination after tol={_CURRICULUM_END_TOL*1000:.0f}mm "
          f"with V_at_pose ≥ {env.success_threshold:.2f} AND V_gripper ≥ {env.success_threshold:.2f}")
    push_mode = "env_goal_x" if env.use_env_goal_for_push else f"fixed_{_PUSH_DISPLACEMENT_M*100:.1f}cm"
    print(f"  obs_mode={env.observation_mode}  "
          f"push_mode={push_mode}  fallback_push_disp={_PUSH_DISPLACEMENT_M*100:.1f}cm  "
          f"push_scale={_FORWARD_PUSH_SCALE}  "
          f"action=4D(xyz_scale={_ACTION_SCALE_XYZ}, gripper)  max_steps={_MAX_STEPS}  "
          f"tol curriculum {_CURRICULUM_START_TOL*1000:.0f}->{_CURRICULUM_END_TOL*1000:.0f}mm  "
          f"perturb curriculum {_PERTURBATION_CURRICULUM_START_M*1000:.0f}->{_PERTURBATION_CURRICULUM_END_M*1000:.0f}mm "
          f"over {_PERTURBATION_CURRICULUM_SCHEDULE_STEPS} steps")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    print(f"Training done. {len(log_cb.episode_returns)} episodes completed.")
    return model, log_cb, env


# ── Main ────────────────────────────────────────────────────────────────────


def main() -> None:
    from time import time
    identifier = f"{int(time())}"
    out_dir = Path("artifacts") / f"out/{identifier}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "planrob_inverse_rl_pushcube_full_model_final.zip"
    best_path = out_dir / "planrob_inverse_rl_pushcube_full_model_best.zip"
    curve_path = out_dir / "planrob_inverse_rl_pushcube_full_curve.png"
    json_path = out_dir / "planrob_inverse_rl_pushcube_full_demo.json"

    base_env = PushCubeRecoveryFullEnv(
        max_steps=_MAX_STEPS, atpose_tolerance=_CURRICULUM_END_TOL,
        action_scale_xyz=_ACTION_SCALE_XYZ, perturbation_range_m=_PERTURBATION_RANGE_M,
        push_displacement_m=_PUSH_DISPLACEMENT_M,
    )
    base_env.set_curriculum_tolerance(_CURRICULUM_END_TOL)

    print(f"Evaluating symbolic-only baseline on {len(_EVAL_SEEDS)} seeds...")
    sym = evaluate_with_release(base_env, None, _EVAL_SEEDS, "symbolic_only", do_rl=False)
    print(f"  symbolic_only:  dist {sym['distance_mm_mean']:.1f}±{sym['distance_mm_std']:.1f}mm  "
          f"V {sym['v_after_release_mean']:.3f}  success@1cm {sym['success_rate_at_1cm']:.1%}")

    print(f"Evaluating random-RL baseline on {len(_EVAL_SEEDS)} seeds...")
    rng = np.random.default_rng(0)
    rnd = evaluate_with_release(base_env, lambda _o: rng.uniform(-1, 1, size=4).astype(np.float32),
                                 _EVAL_SEEDS, "random_rl", do_rl=True)
    print(f"  random_rl:      dist {rnd['distance_mm_mean']:.1f}±{rnd['distance_mm_std']:.1f}mm  "
          f"V {rnd['v_after_release_mean']:.3f}  success@1cm {rnd['success_rate_at_1cm']:.1%}")
    base_env.close()

    final_model, callback, env = train(1_000_000, ckpt_path, best_path)
    final_model.save(ckpt_path)
    demo.plot_curve(callback, curve_path)

    env.set_curriculum_tolerance(_CURRICULUM_END_TOL)

    def predict_fn(obs):
        action, _ = final_model.predict(obs, deterministic=True)
        return action

    print(f"\nEvaluating trained_final WITH scripted release on {len(_EVAL_SEEDS)} seeds...")
    final_with = evaluate_with_release(env, predict_fn, _EVAL_SEEDS, "trained_final_with_release", do_rl=True)
    print(f"  with_release:   dist {final_with['distance_mm_mean']:.1f}±{final_with['distance_mm_std']:.1f}mm  "
          f"V {final_with['v_after_release_mean']:.3f}  success@1cm {final_with['success_rate_at_1cm']:.1%}")

    print(f"\nEvaluating trained_final WITHOUT scripted release on {len(_EVAL_SEEDS)} seeds...")
    final_without = evaluate_no_release(env, predict_fn, _EVAL_SEEDS, "trained_final_no_release", do_rl=True)
    print(f"  no_release:     dist {final_without['distance_mm_mean']:.1f}±{final_without['distance_mm_std']:.1f}mm  "
          f"V {final_without['v_after_release_mean']:.3f}  success@1cm {final_without['success_rate_at_1cm']:.1%}")

    if best_path.exists():
        best_model = SAC.load(best_path, env=env)
        def best_predict(obs):
            action, _ = best_model.predict(obs, deterministic=True)
            return action
        best_with = evaluate_with_release(env, best_predict, _EVAL_SEEDS, "trained_best_with_release", do_rl=True)
        print(f"\n  best_with_release:  dist {best_with['distance_mm_mean']:.1f}±{best_with['distance_mm_std']:.1f}mm  "
              f"V {best_with['v_after_release_mean']:.3f}  success@1cm {best_with['success_rate_at_1cm']:.1%}")
    else:
        best_with = None

    env.close()

    payload = {
        "phase": "pushcube_full",
        "phase_description": "PushCube-v1 + 4D RL action with generic multi-predicate residual reward (sum of normalized signed margins over inverse-target terms)",
        "sac_seed": _SAC_SEED,
        "eval_seeds": _EVAL_SEEDS,
        "baseline_symbolic_only": sym,
        "baseline_random_rl": rnd,
        "trained_final_with_release": final_with,
        "trained_final_no_release": final_without,
        "trained_best_with_release": best_with,
        "training": {
            "n_episodes_logged": len(callback.episode_returns),
            "first50_mean_final_v": float(np.mean(callback.episode_v_finals[:50])) if callback.episode_v_finals else None,
            "last50_mean_final_v": float(np.mean(callback.episode_v_finals[-50:])) if callback.episode_v_finals else None,
            "last50_mean_distance_mm": float(np.mean(callback.episode_distances[-50:])) * 1000.0
                if callback.episode_distances else None,
            "best_rolling_v": float(callback.best_rolling_v),
            "best_rolling_v_at_step": int(callback.best_at_step),
        },
        "config": {
            "env": "PushCube-v1",
            "algorithm": "SAC",
            "reward": "active term in [-1,+1] (bipolar tanh) + fences in [-1,0] (one-sided tanh: penalize violation, silent when satisfied)",
            "reward_terms": [
                {"key": "at_pose(cube, init)",         "sign": +1.0, "scale": "active_residual_scale_factor * episode_tolerance_m", "mode": "bipolar", "role": "active residual"},
                {"key": "gripper_open()",              "sign": +1.0, "scale": "gripper_open_min_width_m",                           "mode": "fence",   "role": "fence (precondition restored by symbolic prefix)"},
                {"key": "at_pose(cube, forward_goal)", "sign": -1.0, "scale": "episode_tolerance_m",                                 "mode": "fence",   "role": "fence (negated add-effect)"},
                {"key": "tcp_near(cube)",              "sign": +1.0, "scale": "tcp_near_threshold_m",                                "mode": "fence",   "role": "fence (forward-push precondition; prevents run-away)"},
            ],
            "active_residual_scale_factor": _ACTIVE_RESIDUAL_SCALE_FACTOR,
            "tcp_near_threshold_m": _TCP_NEAR_THRESHOLD_M,
            "tcp_near_temperature": _TCP_NEAR_TEMP,
            "observation_mode": env.observation_mode,
            "observation_features": list(env.obs_labels()),
            "logged_v_residual": "V_at_pose sigmoid score, used for callback logging and termination",
            "termination": "V_at_pose ≥ 0.50 AND V_gripper_open ≥ 0.50, only when episode_tolerance_m ≤ final curriculum tolerance",
            "total_timesteps": 1_000_000,
            "max_steps_per_episode": _MAX_STEPS,
            "forward_push_mode": "env_goal_x" if _USE_ENV_GOAL_FOR_PUSH else "fixed_displacement",
            "use_env_goal_for_push": _USE_ENV_GOAL_FOR_PUSH,
            "fallback_push_displacement_m": _PUSH_DISPLACEMENT_M,
            "push_action_scale": _FORWARD_PUSH_SCALE,
            "rl_action_dim": 4,
            "rl_action_space": "xyz EEF delta + gripper command (full pd_ee_delta_pos)",
            "atpose_tolerance_curriculum_start_m": _CURRICULUM_START_TOL,
            "atpose_tolerance_curriculum_end_m": _CURRICULUM_END_TOL,
            "atpose_tolerance_curriculum_schedule_steps": _CURRICULUM_SCHEDULE_STEPS,
            "perturbation_curriculum_start_m": _PERTURBATION_CURRICULUM_START_M,
            "perturbation_curriculum_end_m": _PERTURBATION_CURRICULUM_END_M,
            "perturbation_curriculum_schedule_steps": _PERTURBATION_CURRICULUM_SCHEDULE_STEPS,
            "atpose_temperature_train": 0.015,
            "atpose_temperature_eval": 0.005,
            "gripper_open_temperature_train": _GRIPPER_OPEN_TEMP_TRAIN,
            "gripper_open_temperature_eval": _GRIPPER_OPEN_TEMP_EVAL,
            "gripper_open_min_width_m": _GRIPPER_OPEN_MIN_WIDTH,
            "buffer_size": 100_000, "batch_size": 256, "gamma": 0.99,
            "net_arch": [256, 256],
            "success_threshold": 0.50,
            "termination_enabled_after_tolerance_m": _CURRICULUM_END_TOL,
            "action_scale_xyz": _ACTION_SCALE_XYZ,
            "handoff_perturbation_range_m_final": _PERTURBATION_RANGE_M,
            "release_motion_for_with_release_eval": "slow_open_v1 (15 steps cmd=0.3 + 3 settle)",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {json_path}")


def _main_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-io", action="store_true",
                        help="Print policy input/action/reward transitions instead of training.")
    parser.add_argument("--trace-policy", choices=("zero", "random", "toward-cube", "push-to-goal"),
                        default="zero")
    parser.add_argument("--trace-model", type=Path, default=None,
                        help="Optional SAC .zip checkpoint. Overrides --trace-policy.")
    parser.add_argument("--observation-mode", choices=tuple(_OBS_LABELS_BY_MODE),
                        default=_OBSERVATION_MODE,
                        help="Use 'absolute' to trace old 10D checkpoints.")
    parser.add_argument("--trace-seed", type=int, default=1000)
    parser.add_argument("--trace-steps", type=int, default=_MAX_STEPS)
    parser.add_argument("--trace-out", type=Path, default=None)
    args = parser.parse_args()

    if args.trace_io:
        trace_policy_io(
            seed=args.trace_seed,
            steps=args.trace_steps,
            policy=args.trace_policy,
            model_path=args.trace_model,
            out_path=args.trace_out,
            observation_mode=args.observation_mode,
        )
        return

    main()


if __name__ == "__main__":
    _main_cli()
