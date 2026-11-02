"""The operator key flow: heft's [y/n] gate + heft's record keys, sim's PICO chords.

The keys are the whole hardware UX, and every mistake in them is expensive in a way
tests are cheap: arming from the wrong pose bakes a bad yaw alignment into a whole
session, and an `x` that means "stop the robot" instead of "drop this episode" is a
fall. So the transitions are pinned here rather than discovered on a real G1.
"""

from __future__ import annotations

import numpy as np
import pytest

from g1_bridge.hosts.real import RealRobotRunner


class FakeController:
    """sim2real's ControllerBase surface, fed by the test."""

    def __init__(self):
        self.keys: list[str] = []
        self.mode = None
        self.closed = False

    def press(self, *keys):
        self.keys.extend(keys)

    def get_control_mode(self):
        mode, self.mode = self.mode, None
        return mode

    def get_extra_keys(self):
        keys, self.keys = self.keys, []
        return keys

    def close(self):
        self.closed = True


class FakeHost:
    """Just enough host for the arming checks: a standing-still robot."""

    def __init__(self, qvel=0.0):
        self.qvel = qvel

    def read_joint_state(self):
        return np.zeros(29), np.full(29, self.qvel)

    def read_pelvis_state(self):
        return np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)


class FakeSession:
    def __init__(self):
        self.tracking = type("T", (), {
            "set_zero_mode": lambda self, source=None: None,
            "set_init_mode": lambda self, source=None: None,
            "set_policy_mode": lambda self, source=None: None,
        })()
        self.keypresses: list[int] = []

    def handle_keypress(self, keycode):
        self.keypresses.append(keycode)


def make_runner():
    """A runner with the loop and the hardware cut away -- keys and stages only."""
    runner = RealRobotRunner.__new__(RealRobotRunner)
    runner.session = FakeSession()
    runner.host = FakeHost()
    runner.rate_hz = 50.0
    runner.dex3_driver = None
    runner.controller = FakeController()
    runner.on_tick = None
    runner.episode_commands: list[str] = []
    runner.on_episode_command = runner.episode_commands.append
    runner._armed = False
    runner._overruns = 0
    runner._ticks = 0
    runner._stage = "confirm"
    runner._ramp_started_tick = None
    runner._prompted = False
    runner._last_prompt = 0.0
    runner._last_refuse = 0.0
    runner._stdin = False              # no terminal in tests
    runner._stdin_settings = None
    runner._keys_are_local = True      # FakeController stands in for the local one
    runner.armed_calls = []
    runner.arm = lambda: (runner.armed_calls.append(runner._ticks),
                          setattr(runner, "_armed", True),
                          setattr(runner, "_stage", "armed"))
    return runner


def step(runner, ticks=1):
    for _ in range(ticks):
        runner._drain_operator_input()
        runner._advance_stage()
        runner._ticks += 1


def test_gate_then_ramp_then_arm() -> None:
    runner = make_runner()
    step(runner)
    assert runner._stage == "confirm"

    runner.controller.press("y")            # heft's [Y/n] gate
    step(runner)
    assert runner._stage == "ramping"

    step(runner, RealRobotRunner._RAMP_TICKS)
    assert runner._stage == "ready"

    runner.controller.press("a")            # heft's "press A to enter tracking policy"
    step(runner)
    assert runner._armed and runner._stage == "armed"


def test_arming_before_the_robot_is_standing_is_refused() -> None:
    runner = make_runner()
    runner.controller.press("a")            # still at zero torque
    step(runner)
    assert not runner._armed

    runner.controller.press("y")
    step(runner)
    runner.controller.press("a")            # mid-ramp
    step(runner)
    assert not runner._armed, "arming mid-ramp would calibrate the yaw alignment on a half-way pose"


def test_enter_confirms_before_the_gate_and_records_after() -> None:
    runner = make_runner()
    runner.controller.press("enter")
    step(runner)
    assert runner._stage == "ramping" and runner.episode_commands == []

    step(runner, RealRobotRunner._RAMP_TICKS)
    runner.controller.press("enter")        # heft's keyboard_control binds ENTER to 'c'
    step(runner)
    assert runner.episode_commands == ["toggle"]


def test_record_keys_are_hefts() -> None:
    runner = make_runner()
    runner._stage = "armed"
    runner.controller.press("c", "x")
    step(runner)
    assert runner.episode_commands == ["toggle", "discard"]


def test_estop_rearms_the_gate() -> None:
    runner = make_runner()
    runner._stage = "armed"
    runner._armed = True
    RealRobotRunner.zero(runner)
    assert not runner._armed
    assert runner._stage == "confirm", "after an e-stop, one keypress must not reach a live policy"


def test_arming_a_moving_robot_is_refused() -> None:
    # The yaw alignment is frozen from the pose at this instant, so arming a robot
    # that is still moving freezes a pose it is about to leave.
    runner = make_runner()
    runner.host = FakeHost(qvel=3.0)
    runner._stage = "ready"
    runner.controller.press("a")
    step(runner)
    assert not runner._armed

    runner.host = FakeHost(qvel=0.1)
    runner.controller.press("a")
    step(runner)
    assert runner._armed


def test_the_keyboard_pane_can_only_record() -> None:
    """Keys from the ZMQ pane are recording-only.

    That pane publishes on a socket any process on the machine can bind to, so a
    key arriving from it must never be able to arm the robot or drop it to zero
    torque -- those are answered on the controller's own terminal, where the
    prompts asking for them are printed.
    """
    runner = make_runner()
    runner._keys_are_local = False
    runner._stage = "ready"

    for key in ("y", "a", "o", "i", "e", "p"):
        runner.controller.press(key)
    step(runner)
    assert not runner._armed, "the pane must not be able to arm"
    assert runner._stage == "ready", "the pane must not be able to change stage"
    assert runner.session.keypresses == [], "the pane must not reach the mode manager"
    assert runner.episode_commands == []

    runner.controller.press("c", "x")
    step(runner)
    assert runner.episode_commands == ["toggle", "discard"]


def test_q_quits() -> None:
    runner = make_runner()
    runner.controller.press("q")
    with pytest.raises(KeyboardInterrupt):
        step(runner)


def test_e_and_p_reach_the_mode_manager_fallbacks() -> None:
    runner = make_runner()
    runner._stage = "armed"
    runner.controller.press("e", "p")
    step(runner)
    assert runner.session.keypresses == [ord("E"), ord("P")]
