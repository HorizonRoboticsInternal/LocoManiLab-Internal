"""RealRobotHost: the state layout the whole real-robot path depends on.

The host slices joints and IMU straight out of a ``RobotState``. If those offsets
are ever wrong the failure is silent and expensive -- the policy observes garbage
and the recorded dataset looks structurally fine -- so they are pinned here against
the layout ``G1RobotIO.read_state`` / ``ZMQRobotIO._message_to_state`` produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim2real.rl_policy.robot_io.base import RobotState

from g1_bridge.hosts.real import RealRobotHost

NUM_JOINTS = 29


class FakeRobotIO:
    """Hands out prepared states; records the last command written."""

    def __init__(self, states):
        self._states = list(states)
        self.commands = []
        self.closed = False

    def read_state(self):
        return self._states.pop(0) if self._states else None

    def write_command(self, q_target, dq_target, tau_ff, kp, kd):
        self.commands.append((q_target, dq_target, tau_ff, kp, kd))

    def close(self):
        self.closed = True


def make_state(tick=1, joint_offset=0.0, quat=(1.0, 0.0, 0.0, 0.0)):
    qpos = np.zeros(7 + NUM_JOINTS, dtype=np.float32)
    qvel = np.zeros(6 + NUM_JOINTS, dtype=np.float32)
    qpos[3:7] = quat
    qpos[7:] = np.arange(NUM_JOINTS, dtype=np.float32) + joint_offset
    qvel[3:6] = (0.1, 0.2, 0.3)
    qvel[6:] = -np.arange(NUM_JOINTS, dtype=np.float32)
    return RobotState(qpos=qpos, qvel=qvel,
                      joint_torque=np.zeros(NUM_JOINTS, dtype=np.float32), tick=tick)


def build_host(states):
    host = RealRobotHost.__new__(RealRobotHost)  # no TeleopSession: that needs a policy
    host._io = FakeRobotIO(states)
    host._joint_count = NUM_JOINTS
    host._state = None
    host._stale_ticks = 0
    host._skipped = 0
    host._expected_dt_ms = 20.0        # a 50 Hz state stream, as the bridge runs
    return host


def test_reads_slice_joints_and_imu_out_of_the_state() -> None:
    host = build_host([make_state()])
    assert host.poll() is True

    qpos, qvel = host.read_joint_state()
    assert qpos.shape == (NUM_JOINTS,) and qvel.shape == (NUM_JOINTS,)
    np.testing.assert_allclose(qpos, np.arange(NUM_JOINTS))
    np.testing.assert_allclose(qvel, -np.arange(NUM_JOINTS))

    quat, ang_vel = host.read_pelvis_state()
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(ang_vel, [0.1, 0.2, 0.3], rtol=1e-6)


def test_pelvis_quaternion_is_normalized() -> None:
    host = build_host([make_state(quat=(2.0, 0.0, 0.0, 0.0))])
    host.poll()
    quat, _ = host.read_pelvis_state()
    assert np.isclose(np.linalg.norm(quat), 1.0)


def test_degenerate_quaternion_falls_back_to_identity() -> None:
    host = build_host([make_state(quat=(0.0, 0.0, 0.0, 0.0))])
    host.poll()
    quat, _ = host.read_pelvis_state()
    np.testing.assert_allclose(quat, [1.0, 0.0, 0.0, 0.0])


def test_root_pos_is_zero_and_body_pose_is_refused() -> None:
    # A G1 has no odometry: reporting a made-up root position would be worse than
    # refusing, and read_root_pos's zeros are only safe because of where they are
    # consumed (see its docstring).
    host = build_host([make_state()])
    host.poll()
    np.testing.assert_allclose(host.read_root_pos(), np.zeros(3))
    with pytest.raises(NotImplementedError):
        host.read_body_pose_w("pelvis")


def test_clock_comes_from_the_state_tick() -> None:
    host = build_host([make_state(tick=1234)])
    host.poll()
    assert host.clock_ms() == 1234


def test_reads_before_any_state_raise_rather_than_return_zeros() -> None:
    host = build_host([])
    with pytest.raises(RuntimeError):
        host.read_joint_state()


def test_gaps_in_the_state_stream_are_counted() -> None:
    # A dropped packet is a tick the policy never saw. It has to be counted, not
    # silently absorbed: on hardware it is felt as a twitch, not seen as an error.
    host = build_host([make_state(tick=100), make_state(tick=120), make_state(tick=180)])
    host.poll()
    host.poll()
    assert host.skipped == 0, "consecutive 20 ms packets are in step"
    host.poll()
    assert host.skipped == 2, "60 ms of silence at 50 Hz means two packets went missing"


def test_poll_next_returns_false_when_the_stream_stops() -> None:
    host = build_host([make_state(tick=1)])
    assert host.poll_next(timeout_s=0.05) is True
    assert host.poll_next(timeout_s=0.05) is False


def test_repeated_tick_counts_as_stale() -> None:
    # A frozen bridge looks exactly like a stationary robot to PushRobotIO, so the
    # host judges freshness on the state's own tick and eventually gives up.
    host = build_host([make_state(tick=7), make_state(tick=7), None])
    assert host.poll() is True
    assert host.poll() is False
    assert host.stale is False
    host._stale_ticks = 5
    assert host.stale is True


def test_no_pico_button_can_start_or_discard_a_recording() -> None:
    """Recording on hardware is keyboard-only ('c'/'x').

    'Y' used to toggle it and 'Y+A' used to discard, on a bare rising edge. The
    operator's hands are on the sticks all session and 'Y' sits inside the
    A+B+X+Y engage chord, so a brush against it started a recording that
    announced itself with one green line in a scrolling log -- and a second
    brush saved it (observed on an amo run, 2026-08-12).

    This asserts the absence, which is the only way an absence stays: the
    resolver still exists for sim2real's own --controller pico path, so
    re-wiring it here is a two-line change that would otherwise look harmless.
    """
    import inspect

    from g1_bridge.hosts import real as real_mod

    assert not hasattr(real_mod.RealRobotRunner, "_drain_pico_episode_input")

    source = inspect.getsource(real_mod.RealRobotRunner)
    assert "resolve_pico_episode_command" not in source, (
        "the PICO episode binding is back in RealRobotRunner; recording on "
        "hardware is keyboard-only on purpose"
    )
    # The keyboard route must still be there -- this test is about which input
    # reaches on_episode_command, not about removing the callback.
    assert "resolve_keyboard_episode_command" in source
    assert "on_episode_command" in inspect.signature(
        real_mod.RealRobotRunner.__init__
    ).parameters
