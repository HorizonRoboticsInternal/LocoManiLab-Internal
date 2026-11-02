"""ALMI parity against mjlab-loco's own deployment code, tick by tick.

``sim2real/rl_policy/observations/almi.py`` and ``checkpoints/almi``'s exported
ONNX reimplement a layout that already has a working reference implementation --
mjlab-loco's ``deploy_mujoco/sim2sim.py`` + ``policies.py``, which is what runs
on the deploy box. The two live in incompatible environments, so the comparison
is done against a recorded closed-loop rollout: see
``tests/data/gen_almi_sim2sim_parity.py`` for how to regenerate the fixture.

This is the test that pins down the four things most likely to be silently wrong
in a port like this, none of which raise on their own:

* joint ordering (the 29-joint blocks, and the 17-joint upper reference),
* ``q - q_default`` in the proprioception block versus **absolute** angles in
  ``q_ref``/``q_ref_error``,
* the history's time direction (oldest -> newest, current frame last),
* ``waist_rpy`` being derived from ``q_ref`` rather than commanded separately.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import
from sim2real.rl_policy.observations.almi import (
    LAYOUT_DIMS,
    AlmiCommands,
    _AlmiCore,
)

_DATA = Path(__file__).parent / "data"
_CHECKPOINTS = Path(__file__).parents[1] / "checkpoints" / "almi"

#: One (fixture, checkpoint dir) pair per shipped observation layout. The two
#: halves of a pair are regenerated TOGETHER: the npz records what mjlab-loco's
#: own implementation produced for that specific checkpoint, so pairing a
#: fixture with a different checkpoint compares two things that never met.
#: Both layouts are exercised because both are still runnable -- whole_body is
#: the registered default and every ALMI dataset to date was collected under it.
CASES = {
    "whole_body": (_DATA / "almi_sim2sim_parity.npz", _CHECKPOINTS / "flat_dual"),
    "half_body": (
        _DATA / "almi_sim2sim_parity_half_body.npz",
        _CHECKPOINTS / "flat_dual_20260817",
    ),
}

# Teacher-forced, the observation is pure concatenation of fixture values plus
# one subtraction -- except projected_gravity, which each side reaches by its own
# route (a float64 rotation matrix there, quat_rotate_inverse here). A few
# float32 ulps on values of order 1 is what that costs; anything structural is
# orders of magnitude larger.
OBS_ATOL = 1e-6
# The action goes through four 512-wide layers traced to ONNX and re-run under a
# different runtime. Checked against a float64 reference during export: the ONNX
# graph is at least as accurate as the torch module it came from, so this is
# rounding, not drift. Free-running, that rounding also re-enters the next
# tick's observation through prev_action, which is why the two passes exist.
ACTION_RTOL = 1e-4
ACTION_ATOL = 1e-4
# The free-running pass feeds its own rounding back in, so its error compounds
# with the rollout AND scales with how big the actions get -- which is a
# property of the checkpoint, not of this code: the whole_body rollout peaks at
# |action| 7.5 and lands at 5.3e-6, the half_body one peaks at 78 and lands at
# 3.0e-4. Bounded here at 2e-3, ~7x the worst measured, and still three orders
# of magnitude below anything structural: a mis-sliced observation or a history
# advancing at the wrong rate moves the action by O(0.1-1), not by 1e-4. The
# teacher-forced pass above is the tight one and is what pins the layout.
FREE_RUN_ACTION_RTOL = 2e-3
FREE_RUN_ACTION_ATOL = 2e-3
FREE_RUN_OBS_ATOL = 2e-3


class _FakeStateProcessor:
    def __init__(self, joint_names):
        self.joint_names = list(joint_names)
        self.joint_pos = np.zeros(len(joint_names), dtype=np.float32)
        self.joint_vel = np.zeros(len(joint_names), dtype=np.float32)
        self.root_ang_vel_b = np.zeros(3, dtype=np.float32)
        self.root_quat_w = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class _FakeActionManager:
    def __init__(self, joint_names):
        self.joint_names = list(joint_names)


class _FakeEnv:
    """The slice of ``BasePolicy`` an ALMI observation actually reads."""

    def __init__(self, joint_names, default_joint_pos):
        self.state_processor = _FakeStateProcessor(joint_names)
        self.action_manager = _FakeActionManager(joint_names)
        self.joint_names_simulation = list(joint_names)
        self.default_dof_angles = np.asarray(default_joint_pos, dtype=np.float32)
        self.num_actions = len(joint_names)
        self.almi_commands = None
        self.total_inference_cnt = 0


@pytest.fixture(scope="module", params=sorted(CASES), ids=sorted(CASES))
def case(request):
    """One layout's (fixture path, checkpoint dir), skipped if not present."""
    npz, checkpoint_dir = CASES[request.param]
    if not npz.is_file():
        pytest.skip(f"parity fixture not generated: {npz}")
    return request.param, npz, checkpoint_dir


@pytest.fixture(scope="module")
def fixture_data(case):
    layout, npz, _ = case
    data = np.load(npz, allow_pickle=False)
    # The fixture's own widths have to be the layout it is filed under --
    # otherwise a regeneration against the wrong checkpoint would quietly turn
    # this into a second copy of the other case's test.
    assert _layout_of(data) == layout, (
        f"{npz.name} holds {_layout_of(data)!r} observations but is registered "
        f"as {layout!r}"
    )
    return data


@pytest.fixture(scope="module")
def onnx_branches(case):
    _, _, checkpoint_dir = case
    lower = checkpoint_dir / "lower.onnx"
    upper = checkpoint_dir / "upper.onnx"
    if not (lower.is_file() and upper.is_file()):
        pytest.skip(
            f"ALMI ONNX not exported into {checkpoint_dir}; run "
            "scripts/export_almi_onnx.py"
        )
    import onnxruntime as ort

    return (
        ort.InferenceSession(str(lower), providers=["CPUExecutionProvider"]),
        ort.InferenceSession(str(upper), providers=["CPUExecutionProvider"]),
    )


def _layout_of(data) -> str:
    """Which observation layout the fixture was recorded under.

    Read off the recorded widths rather than stored as a flag: the fixture is
    regenerated from mjlab-loco whenever the layout changes, and a flag written
    by hand at that moment is exactly the thing that would be forgotten. An
    unknown width is a hard error -- guessing here would compare this
    implementation against a reference it never ran.
    """
    lower, upper = int(data["lower_obs"].shape[1]), int(data["upper_obs"].shape[1])
    for name, dims in LAYOUT_DIMS.items():
        if dims == (lower, upper):
            return name
    raise AssertionError(
        f"fixture has lower_obs={lower}, upper_obs={upper}, which matches no "
        f"layout in LAYOUT_DIMS ({LAYOUT_DIMS}) -- regenerate it, or add the "
        "layout to sim2real/rl_policy/observations/almi.py"
    )


def _replay(data, onnx_branches, *, teacher_forced: bool):
    """Run the recorded rollout through this repo's observation + ONNX path.

    ``teacher_forced`` decides what ``prev_action`` holds. Forced, it is the
    reference's own recorded action, so every observation is a pure function of
    fixture values and any disagreement is a real layout bug. Free-running, it
    is this implementation's ONNX output, so the ~1e-5 float32 delta between the
    two runtimes re-enters the observation and compounds -- which is what makes
    the free-running action check meaningful and the observation check loose.
    """
    lower_session, upper_session = onnx_branches
    joint_names = [str(name) for name in data["joint_names"]]
    env = _FakeEnv(joint_names, data["default_joint_pos"])
    core = _AlmiCore(
        env,
        joint_names=joint_names,
        lower_joint_names=joint_names[:12],
        upper_joint_names=joint_names[12:29],
        history_steps=5,
        layout=_layout_of(data),
    )

    ticks = data["joint_pos"].shape[0]
    lower_obs = np.zeros_like(data["lower_obs"])
    upper_obs = np.zeros_like(data["upper_obs"])
    history = np.zeros_like(data["lower_history"])
    action = np.zeros_like(data["action"])
    waist_rpy = np.zeros_like(data["waist_rpy"])

    state_dict = {"action": np.zeros(len(joint_names), dtype=np.float32)}
    for t in range(ticks):
        env.state_processor.joint_pos = data["joint_pos"][t]
        env.state_processor.joint_vel = data["joint_vel"][t]
        env.state_processor.root_ang_vel_b = data["base_ang_vel"][t]
        env.state_processor.root_quat_w = data["root_quat"][t]
        env.almi_commands = AlmiCommands(
            twist=data["twist"][t],
            # Zero on purpose: a real reference must overwrite it, and if
            # validated() ever stops doing that the recorded waist command
            # goes to zero here rather than quietly staying plausible.
            waist_rpy=np.zeros(3, dtype=np.float32),
            ee_pos=data["ee_pos"][t],
            ee_quat=data["ee_quat"][t],
            upper_ref_joint_pos=data["q_ref"][t],
        )
        env.total_inference_cnt = t
        core.update_once(state_dict)

        lower_obs[t] = core.output("lower_obs")
        upper_obs[t] = core.output("upper_obs")
        history[t] = core.output("lower_history").reshape(5, -1)
        waist_rpy[t] = env.almi_commands.validated().waist_rpy

        lower_action = lower_session.run(
            None,
            {
                "obs": lower_obs[t][None, :],
                "lower_actor_history": history[t].reshape(1, -1),
            },
        )[0].reshape(-1)
        upper_action = upper_session.run(None, {"obs": upper_obs[t][None, :]})[
            0
        ].reshape(-1)
        action[t] = np.concatenate([lower_action, upper_action])
        # Bounded exactly as gen_almi_sim2sim_parity.py bounds it before feeding
        # the next tick -- the rollout drives a body that never responds, so an
        # unbounded feedback path lets a branch that leans on prev_action run
        # away to 1e30 and turns every later tick into noise-vs-noise. The value
        # travels in the fixture rather than being repeated here.
        clip = float(data["feedback_clip"])
        state_dict["action"] = np.clip(
            data["action"][t] if teacher_forced else action[t], -clip, clip
        )

    return {
        "lower_obs": lower_obs,
        "upper_obs": upper_obs,
        "lower_history": history,
        "action": action,
        "waist_rpy": waist_rpy,
    }


@pytest.fixture(scope="module")
def replayed(fixture_data, onnx_branches):
    """Teacher-forced: observations are exactly comparable."""
    return _replay(fixture_data, onnx_branches, teacher_forced=True)


@pytest.fixture(scope="module")
def free_running(fixture_data, onnx_branches):
    """Closed loop on this implementation's own actions."""
    return _replay(fixture_data, onnx_branches, teacher_forced=False)


def test_projected_gravity_convention_matches(fixture_data, replayed):
    """The obs derives gravity from the root quaternion; the reference derived
    it from the pelvis rotation matrix. Same vector or the whole thing is off."""
    np.testing.assert_allclose(
        replayed["lower_obs"][:, 3:6],
        fixture_data["projected_gravity"],
        atol=OBS_ATOL,
    )


def test_lower_obs_matches_sim2sim(fixture_data, replayed):
    np.testing.assert_allclose(
        replayed["lower_obs"], fixture_data["lower_obs"], atol=OBS_ATOL
    )


def test_upper_obs_matches_sim2sim(fixture_data, replayed):
    np.testing.assert_allclose(
        replayed["upper_obs"], fixture_data["upper_obs"], atol=OBS_ATOL
    )


def test_history_matches_sim2sim(fixture_data, replayed):
    """Includes the opening ticks, where the buffer is backfilled with the first
    frame rather than padded with zeros."""
    np.testing.assert_allclose(
        replayed["lower_history"], fixture_data["lower_history"], atol=OBS_ATOL
    )


def test_history_is_oldest_to_newest(fixture_data, replayed):
    """The last history frame is the current lower_obs -- not the first."""
    np.testing.assert_allclose(
        replayed["lower_history"][:, -1, :], replayed["lower_obs"], atol=OBS_ATOL
    )


def test_waist_command_is_derived_from_the_joint_reference(fixture_data, replayed):
    """q_ref order is (yaw, roll, pitch); the command is (roll, pitch, yaw)."""
    q_ref = fixture_data["q_ref"]
    np.testing.assert_allclose(replayed["waist_rpy"], q_ref[:, [1, 2, 0]], atol=0)
    np.testing.assert_allclose(
        replayed["waist_rpy"], fixture_data["waist_rpy"], atol=0
    )


def test_action_matches_sim2sim(fixture_data, replayed):
    """One-step: the two runtimes agree on the action for identical inputs."""
    np.testing.assert_allclose(
        replayed["action"],
        fixture_data["action"],
        rtol=ACTION_RTOL,
        atol=ACTION_ATOL,
    )


def test_free_running_rollout_does_not_diverge(fixture_data, free_running):
    """64 ticks of closed loop, this implementation feeding back its own action.

    The looser tolerance is the price of the loop: ONNX-vs-torch rounding enters
    prev_action and rides forward. What this catches that the teacher-forced
    pass cannot is any error that only compounds -- a history that advances at
    the wrong rate, or an action fed back one tick late.
    """
    np.testing.assert_allclose(
        free_running["action"],
        fixture_data["action"],
        rtol=FREE_RUN_ACTION_RTOL,
        atol=FREE_RUN_ACTION_ATOL,
    )
    np.testing.assert_allclose(
        free_running["lower_obs"], fixture_data["lower_obs"], atol=FREE_RUN_OBS_ATOL
    )
    np.testing.assert_allclose(
        free_running["upper_obs"], fixture_data["upper_obs"], atol=FREE_RUN_OBS_ATOL
    )
