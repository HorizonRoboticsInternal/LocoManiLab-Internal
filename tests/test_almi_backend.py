"""AlmiBackend: the recording contract, the stick mapping, and the EE frame.

The parity test next door proves the observation layout matches mjlab-loco. This
one covers what the *backend* adds on top -- turning an operator into a command
-- which has no upstream reference to compare against.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import
from g1_bridge.backends import get_backend_class
from g1_bridge.backends.almi import AlmiBackend
from g1_bridge.modality import layout_for
from g1_bridge.recording.frames import build_blob_frame
from g1_bridge.recording.schema import layout_sources
from sim2real.rl_policy.families import policy_config_path

#: Filled by the host in recording/frames.py, not by any backend.
HOST_INJECTED_SOURCES = {"root_orientation", "projected_gravity"}


@pytest.fixture(scope="module")
def policy_config():
    return yaml.safe_load(policy_config_path("almi").read_text())


#: The default stance, in the order policy.yaml declares. Kept here rather than
#: read from the config so a bad edit to `default_joint_pos` fails the test.
DEFAULT_JOINT_POS = np.array(
    [-0.312, 0, 0, 0.669, -0.363, 0] * 2 + [0, 0, 0]
    + [0.2, 0.2, 0, 0.6, 0, 0, 0]
    + [0.2, -0.2, 0, 0.6, 0, 0, 0],
    dtype=np.float32,
)


class _FakeTracking:
    def __init__(self, policy_config):
        from sim2real.config.robots import get_robot_cfg

        self.policy_config = policy_config
        self.almi_commands = None
        # The backend forward-kinematics its neutral reference out of these.
        self.robot_cfg = get_robot_cfg("g1")
        self.default_dof_angles = DEFAULT_JOINT_POS.copy()


class _FakeCtx:
    """The slice of Sim2RealContext AlmiBackend touches outside step_policy."""

    def __init__(self, policy_config):
        self.tracking = _FakeTracking(policy_config)
        self.left_trigger = 0.0
        self.right_trigger = 0.0
        self._qpos = np.arange(29, dtype=np.float32) * 0.01
        #: Distinct from the body qpos above so a test can tell the hand block
        #: of observation.state apart from the arms next to it.
        self._hands = {
            "left": np.full(7, 0.5, dtype=np.float32),
            "right": np.full(7, -0.5, dtype=np.float32),
        }

    def read_joint_state(self):
        return self._qpos, np.zeros(29, dtype=np.float32)

    def read_hand_state(self):
        return self._hands


def _loaded_backend(policy_config):
    backend = AlmiBackend()
    ctx = _FakeCtx(policy_config)
    backend.on_load(ctx, cfg={}, res=_Resources())
    backend.reset(ctx)
    return backend, ctx


class _Resources:
    extra = {"gmr_relay": None}


def _payload(*, torso_pos, torso_quat, left_pos, right_pos, joint_pos=None):
    names = ["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    identity = [1.0, 0.0, 0.0, 0.0]
    joint_names = list(yaml.safe_load(policy_config_path("almi").read_text())[
        "joint_names_simulation"
    ])
    if joint_pos is None:
        joint_pos = np.zeros(len(joint_names), dtype=np.float32)
    return {
        "body_names": names,
        "body_pos_w": [torso_pos, left_pos, right_pos],
        "body_quat_w": [torso_quat, identity, identity],
        "joint_names": joint_names,
        "joint_pos": np.asarray(joint_pos, dtype=np.float32).tolist(),
    }


def test_registry_resolves_almi():
    cls = get_backend_class("almi")
    assert cls is AlmiBackend
    assert cls.name == "almi"
    assert cls.uses_gmr is True
    assert cls.vla_action_space == "twist_ee"


def test_teleop_sample_covers_exactly_the_modality_sources(policy_config):
    """The recording schema and the sample that fills it must not drift apart."""
    backend, ctx = _loaded_backend(policy_config)
    backend._have_reference = True
    backend._driving = True  # a POSE tick: IDLE ticks are not recorded (schema v7)
    backend._commands = backend._commands.__class__(
        twist=np.array([0.1, 0.2, 0.3, 0.65], np.float32),
        waist_rpy=np.zeros(3, np.float32),
        ee_pos=np.zeros(6, np.float32),
        ee_quat=np.array([1, 0, 0, 0, 1, 0, 0, 0], np.float32),
        upper_ref_joint_pos=np.zeros(17, np.float32),
    ).validated()

    sample = backend.teleop_sample(ctx)
    expected = set(layout_sources(layout_for("almi"))) - HOST_INJECTED_SOURCES
    assert set(sample) == expected


def test_recorded_frame_has_the_declared_widths(policy_config):
    backend, ctx = _loaded_backend(policy_config)
    backend._have_reference = True
    backend._driving = True  # a POSE tick: IDLE ticks are not recorded (schema v7)
    backend._commands = backend._commands.__class__(
        twist=np.array([0.1, 0.2, 0.3, 0.65], np.float32),
        waist_rpy=np.zeros(3, np.float32),
        ee_pos=np.arange(6, dtype=np.float32) * 0.05,
        ee_quat=np.array([1, 0, 0, 0, 1, 0, 0, 0], np.float32),
        upper_ref_joint_pos=np.arange(17, dtype=np.float32) * 0.01,
    ).validated()

    frame = build_blob_frame(
        layout_for("almi"),
        backend.teleop_sample(ctx),
        image=np.zeros((4, 4, 3), np.uint8),
        root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
    )
    assert frame is not None
    # 43, not 29: schema v5 records the measured fingers alongside the body
    # joints, in sonic's interleaved order.
    assert frame["observation.state"].shape == (43,)
    assert frame["observation.root_orientation"].shape == (4,)
    assert frame["observation.projected_gravity"].shape == (3,)
    assert frame["action.twist"].shape == (4,)
    assert frame["action.upper_ref"].shape == (34,)
    assert frame["action.hands"].shape == (14,)

    # The hands land BETWEEN the arms, not appended after them -- the one
    # property that distinguishes this layout from an equally-plausible
    # 43-entry one, and that nothing downstream could detect if it were wrong.
    # _FakeCtx reports constant 0.5/-0.5 fingers against a ramp of body joints,
    # so each block is identifiable on sight.
    np.testing.assert_allclose(frame["observation.state"][22:29], 0.5)
    np.testing.assert_allclose(frame["observation.state"][36:43], -0.5)
    np.testing.assert_allclose(
        frame["observation.state"][29:36], ctx._qpos[22:29]  # right arm
    )
    # action.upper_ref is [waist_rpy(3) | ee_pos(6) | ee_quat(8) | q_ref(17)]
    np.testing.assert_allclose(
        frame["action.upper_ref"][3:9], np.arange(6, dtype=np.float32) * 0.05
    )
    np.testing.assert_allclose(
        frame["action.upper_ref"][17:34], np.arange(17, dtype=np.float32) * 0.01
    )


def test_nothing_is_recorded_before_the_first_retargeted_frame(policy_config):
    """The readiness gate: a dataset must not open with a synthesized reference."""
    backend, ctx = _loaded_backend(policy_config)
    sample = backend.teleop_sample(ctx)
    assert sample == {"upper_ref_joint_pos": None}
    assert (
        build_blob_frame(
            layout_for("almi"),
            sample,
            image=np.zeros((4, 4, 3), np.uint8),
            root_quat=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        is None
    )


def test_ee_pose_is_expressed_in_the_torso_frame(policy_config):
    """A torso rotated 90 deg about z must not change a hand held out in front
    of it -- that is the whole point of reporting in the torso frame."""
    backend, _ctx = _loaded_backend(policy_config)
    half = np.sqrt(0.5)

    flat = backend._ee_from_payload(
        _payload(
            torso_pos=[0.0, 0.0, 1.0],
            torso_quat=[1.0, 0.0, 0.0, 0.0],
            left_pos=[0.2, 0.2, 1.0],
            right_pos=[0.2, -0.2, 1.0],
        )
    )[0]
    # Same limb geometry, whole robot yawed +90 deg and translated.
    yawed = backend._ee_from_payload(
        _payload(
            torso_pos=[3.0, -1.0, 1.0],
            torso_quat=[half, 0.0, 0.0, half],
            left_pos=[3.0 - 0.2, -1.0 + 0.2, 1.0],
            right_pos=[3.0 + 0.2, -1.0 + 0.2, 1.0],
        )
    )[0]
    np.testing.assert_allclose(flat, [0.2, 0.2, 0.0, 0.2, -0.2, 0.0], atol=1e-6)
    np.testing.assert_allclose(yawed, flat, atol=1e-6)


def test_waist_command_tracks_the_clipped_joint_reference(policy_config):
    """Clipping happens on q_ref, so the derived waist command stays in sync."""
    backend, _ctx = _loaded_backend(policy_config)

    class _Relay:
        @staticmethod
        def latest():
            joint_names = list(policy_config["joint_names_simulation"])
            joint_pos = np.zeros(len(joint_names), dtype=np.float32)
            # waist_yaw far outside the trained +/-2.5, waist_roll inside.
            joint_pos[joint_names.index("waist_yaw_joint")] = 9.0
            joint_pos[joint_names.index("waist_roll_joint")] = 0.2
            return _payload(
                torso_pos=[0.0, 0.0, 1.0],
                torso_quat=[1.0, 0.0, 0.0, 0.0],
                left_pos=[0.2, 0.2, 1.0],
                right_pos=[0.2, -0.2, 1.0],
                joint_pos=joint_pos,
            )

    backend._relay = _Relay()
    backend._update_reference_from_gmr()
    assert backend._have_reference

    commands = backend._commands.__class__(
        twist=np.array([0.0, 0.0, 0.0, 0.65], np.float32),
        waist_rpy=np.zeros(3, np.float32),
        ee_pos=backend._ee_pos,
        ee_quat=backend._ee_quat,
        upper_ref_joint_pos=backend._q_ref,
    ).validated()

    assert backend._q_ref[0] == pytest.approx(2.5)  # yaw, clipped
    assert backend._q_ref[1] == pytest.approx(0.2)  # roll, untouched
    # waist_rpy is (roll, pitch, yaw) read back out of q_ref (yaw, roll, pitch).
    np.testing.assert_allclose(commands.waist_rpy, [0.2, 0.0, 2.5], atol=1e-6)


def test_sticks_map_to_the_trained_twist_ranges(policy_config):
    backend, ctx = _loaded_backend(policy_config)

    class _Tick:
        buttons = {"left_axis": (0.0, 1.0), "right_axis": (0.0, 0.0)}

    forward = backend._twist_from_sticks(_Tick(), driving=True)
    assert forward[0] == pytest.approx(0.8)  # full left-stick forward -> vx max
    assert forward[1] == pytest.approx(0.0)
    assert forward[2] == pytest.approx(0.0)

    class _Turn:
        buttons = {"left_axis": (0.0, 0.0), "right_axis": (-1.0, 0.0)}

    turning = backend._twist_from_sticks(_Turn(), driving=True)
    assert turning[2] == pytest.approx(1.8)  # -rx -> +wz

    class _Idle:
        buttons = {"left_axis": (1.0, 1.0), "right_axis": (1.0, 0.0)}

    parked = backend._twist_from_sticks(_Idle(), driving=False)
    np.testing.assert_allclose(parked[:3], 0.0)


def test_stick_deadzone_rejects_thumb_rest(policy_config):
    backend, ctx = _loaded_backend(policy_config)

    class _Rest:
        buttons = {"left_axis": (0.1, 0.1), "right_axis": (0.1, 0.0)}

    np.testing.assert_allclose(
        backend._twist_from_sticks(_Rest(), driving=True)[:3], 0.0
    )


def test_episode_opens_at_the_robots_standing_height(policy_config):
    """ALMI must open at the same posture every other controller holds.

    Regression test for the 0.65 default, which was a 15 cm crouch: it left the
    knee at 1.4 rad, where the policy is only marginally stable (the residual
    drift varied between runs). The value is derived from the robot cfg's own
    standing root height and clipped into the trained band, so it tracks the
    robot rather than a constant.
    """
    backend, ctx = _loaded_backend(policy_config)
    natural = float(backend._fk.default_root_height)
    hi = float(backend._twist_max()[3])
    assert backend._base_height == pytest.approx(min(natural, hi))
    assert backend._base_height > 0.7, "opening posture is still a crouch"
    # And the first tick commands it, rather than something else.
    class _Idle:
        buttons = {"left_axis": (0.0, 0.0), "right_axis": (0.0, 0.0)}

    assert backend._twist_from_sticks(_Idle(), driving=True)[3] == pytest.approx(
        backend._base_height
    )


def test_base_height_override_is_honoured(policy_config):
    """`--set almi_base_height=...` for an operator who wants to start low."""
    backend = AlmiBackend()
    ctx = _FakeCtx(policy_config)
    backend.on_load(ctx, cfg={"almi_base_height": 0.55}, res=_Resources())
    backend.reset(ctx)
    assert backend._base_height == pytest.approx(0.55)
    # ...still clipped into the trained band.
    backend.on_load(ctx, cfg={"almi_base_height": 99.0}, res=_Resources())
    backend.reset(ctx)
    assert backend._base_height == pytest.approx(float(backend._twist_max()[3]))


def test_base_height_integrates_and_holds(policy_config):
    """Height is a posture, not a velocity: it survives releasing the stick and
    survives disengaging, and it never leaves the trained band."""
    backend, ctx = _loaded_backend(policy_config)
    # From mid-band, so there is room to move in both directions.
    backend._base_height = 0.60
    start = backend._base_height

    class _Up:
        buttons = {"left_axis": (0.0, 0.0), "right_axis": (0.0, 1.0)}

    for _ in range(10):
        backend._twist_from_sticks(_Up(), driving=True)
    raised = backend._base_height
    assert raised > start

    class _Centered:
        buttons = {"left_axis": (0.0, 0.0), "right_axis": (0.0, 0.0)}

    twist = backend._twist_from_sticks(_Centered(), driving=True)
    assert backend._base_height == pytest.approx(raised)
    assert twist[3] == pytest.approx(raised)
    # Held across a disengage too -- the velocity channels zero, the posture does not.
    parked = backend._twist_from_sticks(_Centered(), driving=False)
    assert parked[3] == pytest.approx(raised)
    # And a thumb left on the stick while disengaged must not walk it away.
    for _ in range(50):
        backend._twist_from_sticks(_Up(), driving=False)
    assert backend._base_height == pytest.approx(raised)

    for _ in range(10000):
        backend._twist_from_sticks(_Up(), driving=True)
    lo, hi = backend._twist_min(), backend._twist_max()
    assert lo[3] <= backend._base_height <= hi[3]


def test_turning_suppresses_height_integration(policy_config):
    """Both live on the right stick; a sideways thumb must not walk the height."""
    backend, ctx = _loaded_backend(policy_config)

    class _TurnAndTilt:
        buttons = {"left_axis": (0.0, 0.0), "right_axis": (1.0, 1.0)}

    before = backend._base_height
    for _ in range(20):
        backend._twist_from_sticks(_TurnAndTilt(), driving=True)
    assert backend._base_height == pytest.approx(before)


def test_neutral_reference_is_a_real_self_consistent_pose(policy_config):
    """The no-GMR neutral must be the standing pose, not a set of constants.

    This is a regression test for the bug that made ALMI walk backwards out of
    the box: the neutral used ``deploy_mujoco``'s slope-rig EE defaults while
    leaving ``q_ref`` to fall back to the live pose, so the EE target and the
    joint reference disagreed on every tick and the upper branch saturated
    (|action| ~10 instead of ~0.6). Measured cost: 1.52 m of backward walking
    per 20 s. The three fields must come from ONE pose.
    """
    backend, _ctx = _loaded_backend(policy_config)
    q_ref, ee_pos, ee_quat = backend._neutral

    # The reference IS the default stance's upper joints, in policy.yaml order.
    upper_names = policy_config["observation"]["upper_obs"]["upper_obs"][
        "upper_joint_names"
    ]
    all_names = policy_config["joint_names_simulation"]
    expected = np.array(
        [DEFAULT_JOINT_POS[all_names.index(n)] for n in upper_names], np.float32
    )
    np.testing.assert_allclose(q_ref, expected, atol=1e-6)

    # The hands hang at the sides of a standing robot: below the torso origin
    # and out to either side. The slope-rig constants put them at z=+0.20.
    assert ee_pos[2] < 0.0 and ee_pos[5] < 0.0, f"hands are not hanging: {ee_pos}"
    assert ee_pos[1] > 0.1 and ee_pos[4] < -0.1, f"hands are not to the sides: {ee_pos}"
    # A real wrist orientation, not identity.
    assert abs(float(ee_quat[0])) < 0.999, "left wrist quat is identity"

    # reset() must hand that pose to the first tick, not None.
    for value in (backend._q_ref, backend._ee_pos, backend._ee_quat):
        assert value is not None
    np.testing.assert_allclose(backend._q_ref, q_ref, atol=0)

    # ...and it stays out of the dataset until a real operator frame arrives.
    assert backend._have_reference is False


def test_neutral_command_is_internally_consistent(policy_config):
    """waist_rpy derived from the neutral q_ref, and everything in range."""
    backend, _ctx = _loaded_backend(policy_config)
    q_ref, ee_pos, ee_quat = backend._neutral
    commands = backend._commands.__class__(
        twist=np.array([0.0, 0.0, 0.0, 0.65], np.float32),
        waist_rpy=np.zeros(3, np.float32),
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        upper_ref_joint_pos=q_ref,
    ).validated()
    np.testing.assert_allclose(commands.waist_rpy, q_ref[[1, 2, 0]], atol=0)
    lo = backend._limits["ee_pos_min"]
    hi = backend._limits["ee_pos_max"]
    assert np.all(commands.ee_pos >= lo) and np.all(commands.ee_pos <= hi), (
        f"the standing pose sits outside the trained EE box: {commands.ee_pos}"
    )


def test_missing_command_limits_are_rejected(policy_config):
    """The clip box is what keeps a retargeted operator in distribution -- a
    checkpoint without one must fail loudly, not silently run unclipped."""
    stripped = {k: v for k, v in policy_config.items() if k != "almi_command_limits"}
    backend = AlmiBackend()
    with pytest.raises(ValueError, match="almi_command_limits"):
        backend.on_load(_FakeCtx(stripped), cfg={}, res=_Resources())
