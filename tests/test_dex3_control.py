from __future__ import annotations

import numpy as np
import pytest

from sim2real.utils.common import Dex3HandStateMessage, HandGripMessage
from sim2real.utils.dex3 import (
    DEX3_CLOSE_POSE,
    DEX3_JOINT_RANGES,
    DEX3_NUM_MOTORS,
    DEX3_THUMB_OPPOSITION_RAD,
    Dex3HandDriver,
    DryRunDex3HandDriver,
    HandGripConsumer,
    grip_to_joint_targets,
)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("grip", [0.0, 0.4, 1.0])
def test_grip_mapping_matches_close_pose_scaled_by_grip(side: str, grip: float) -> None:
    targets = grip_to_joint_targets(grip, side)

    assert targets.shape == (DEX3_NUM_MOTORS,)
    assert targets[0] == pytest.approx(DEX3_THUMB_OPPOSITION_RAD)
    close_pose = DEX3_CLOSE_POSE[side]
    for idx in range(1, DEX3_NUM_MOTORS):
        assert targets[idx] == pytest.approx(close_pose[idx] * grip)


@pytest.mark.parametrize("side", ["left", "right"])
def test_close_pose_fits_within_joint_ranges(side: str) -> None:
    # heft's kCloseLeft/kCloseRight were tuned on real hardware independently
    # of this reference asset's joint ranges; confirm they're still mutually
    # consistent (this is what caught the previous, mismatched hand revision).
    ranges = DEX3_JOINT_RANGES[side]
    close_pose = DEX3_CLOSE_POSE[side]
    for idx in range(DEX3_NUM_MOTORS):
        lower, upper = ranges[idx]
        assert lower <= close_pose[idx] <= upper


@pytest.mark.parametrize(
    ("grip", "side", "error"),
    [
        (-0.1, "left", "within"),
        (1.1, "left", "within"),
        (np.nan, "left", "finite"),
        (0.5, "up", "side"),
    ],
)
def test_grip_mapping_rejects_invalid_input(grip: float, side: str, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        grip_to_joint_targets(grip, side)


def test_dry_run_driver_records_last_grips_and_targets() -> None:
    driver = DryRunDex3HandDriver(hand_dir="double")
    driver.apply_grips(0.2, 0.9)

    assert driver.last_grips == pytest.approx((0.2, 0.9))
    feedback = driver.read_feedback()
    assert set(feedback) == {"left", "right"}
    np.testing.assert_allclose(feedback["left"], grip_to_joint_targets(0.2, "left"))
    np.testing.assert_allclose(feedback["right"], grip_to_joint_targets(0.9, "right"))
    np.testing.assert_allclose(driver.read_targets()["left"], feedback["left"])
    driver.close()  # should not raise


def test_dry_run_driver_single_hand_only_tracks_that_side() -> None:
    driver = DryRunDex3HandDriver(hand_dir="left")
    driver.apply_grips(0.5, 0.5)

    assert set(driver.read_feedback()) == {"left"}


def test_consumer_decodes_and_applies_to_dry_run_driver() -> None:
    driver = DryRunDex3HandDriver(hand_dir="double")
    consumer = HandGripConsumer(driver)

    decoded = consumer.consume(
        HandGripMessage(timestamp_ns=7, left_grip=0.3, right_grip=0.6).to_bytes()
    )

    assert decoded.timestamp_ns == 7
    assert driver.last_grips == pytest.approx((0.3, 0.6))


class FakeMotorCmd:
    def __init__(self) -> None:
        self.mode = 0
        self.q = 0.0
        self.dq = 0.0
        self.tau = 0.0
        self.kp = 0.0
        self.kd = 0.0


class FakeHandCmd:
    def __init__(self) -> None:
        self.motor_cmd = [FakeMotorCmd() for _ in range(DEX3_NUM_MOTORS)]


class FakeMotorState:
    def __init__(self, q: float = 0.0) -> None:
        self.q = q


class FakeHandState:
    def __init__(self) -> None:
        self.motor_state = [FakeMotorState() for _ in range(DEX3_NUM_MOTORS)]


class FakePublisher:
    def __init__(self, topic: str, msg_type) -> None:
        self.topic = topic
        self.msg_type = msg_type
        self.sent: list[FakeHandCmd] = []
        self.inited = False

    def Init(self) -> None:
        self.inited = True

    def Write(self, cmd: FakeHandCmd) -> None:
        self.sent.append(cmd)


class FakeSubscriber:
    def __init__(self, topic: str, msg_type) -> None:
        self.topic = topic
        self.msg_type = msg_type
        self.handler = None

    def Init(self, handler, queue_len: int) -> None:
        self.handler = handler


def _make_fake_driver(**kwargs) -> tuple[Dex3HandDriver, dict[str, FakePublisher], list[tuple[int, str]]]:
    publishers: dict[str, FakePublisher] = {}
    subscribers: dict[str, FakeSubscriber] = {}
    init_calls: list[tuple[int, str]] = []

    def channel_factory_initialize(domain_id: int, interface: str) -> None:
        init_calls.append((domain_id, interface))

    def channel_publisher_cls(topic: str, msg_type):
        pub = FakePublisher(topic, msg_type)
        publishers[topic] = pub
        return pub

    def channel_subscriber_cls(topic: str, msg_type):
        sub = FakeSubscriber(topic, msg_type)
        subscribers[topic] = sub
        return sub

    driver = Dex3HandDriver(
        channel_factory_initialize=channel_factory_initialize,
        channel_publisher_cls=channel_publisher_cls,
        channel_subscriber_cls=channel_subscriber_cls,
        hand_cmd_cls=FakeHandCmd,
        hand_state_cls=FakeHandState,
        hand_cmd_factory=FakeHandCmd,
        **kwargs,
    )
    return driver, publishers, init_calls


def test_driver_initializes_dds_channels_for_both_hands() -> None:
    driver, publishers, init_calls = _make_fake_driver(domain_id=3, interface="eth1")
    try:
        assert init_calls == [(3, "eth1")]
        assert set(publishers) == {"rt/dex3/left/cmd", "rt/dex3/right/cmd"}
        assert all(pub.inited for pub in publishers.values())
    finally:
        driver.close()


def test_driver_clamps_first_command_by_max_delta_q() -> None:
    driver, publishers, _ = _make_fake_driver(max_delta_q=0.1)
    try:
        driver.apply_grips(1.0, 0.0)
        sent = publishers["rt/dex3/left/cmd"].sent
        assert len(sent) == 1
        # measured state starts at zero, so the first command must not exceed
        # +/- max_delta_q even though the raw grip target is much larger.
        for idx in range(1, DEX3_NUM_MOTORS):
            assert abs(sent[0].motor_cmd[idx].q) <= 0.1 + 1e-6
        assert sent[0].motor_cmd[0].q == pytest.approx(DEX3_THUMB_OPPOSITION_RAD)
    finally:
        driver.close()


def test_driver_uses_thumb_gains_only_for_first_motor() -> None:
    driver, publishers, _ = _make_fake_driver(kp=1.0, kd=0.2, thumb_kp=2.0, thumb_kd=0.5)
    try:
        driver.apply_grips(0.0, 0.0)
        sent = publishers["rt/dex3/left/cmd"].sent[0]
        assert sent.motor_cmd[0].kp == pytest.approx(2.0)
        assert sent.motor_cmd[0].kd == pytest.approx(0.5)
        for idx in range(1, DEX3_NUM_MOTORS):
            assert sent.motor_cmd[idx].kp == pytest.approx(1.0)
            assert sent.motor_cmd[idx].kd == pytest.approx(0.2)
    finally:
        driver.close()


def test_driver_close_sends_relax_command_with_zero_gains() -> None:
    driver, publishers, _ = _make_fake_driver()
    driver.close()

    for pub in publishers.values():
        assert len(pub.sent) == 1
        for motor in pub.sent[0].motor_cmd:
            assert motor.kp == 0.0
            assert motor.kd == 0.0


def test_dex3_hand_state_message_round_trip() -> None:
    message = Dex3HandStateMessage(
        timestamp_ns=123,
        left_target=np.arange(DEX3_NUM_MOTORS, dtype=np.float32),
        left_measured=np.zeros(DEX3_NUM_MOTORS, dtype=np.float32),
        right_target=np.full(DEX3_NUM_MOTORS, 0.5, dtype=np.float32),
        right_measured=np.full(DEX3_NUM_MOTORS, -0.5, dtype=np.float32),
    )

    decoded = Dex3HandStateMessage.from_bytes(message.to_bytes())

    assert decoded.timestamp_ns == 123
    np.testing.assert_allclose(decoded.left_target, message.left_target)
    np.testing.assert_allclose(decoded.right_measured, message.right_measured)


def test_dex3_hand_state_message_defaults_to_zero_hands() -> None:
    message = Dex3HandStateMessage(timestamp_ns=1)
    for arr in (
        message.left_target,
        message.left_measured,
        message.right_target,
        message.right_measured,
    ):
        assert arr.shape == (DEX3_NUM_MOTORS,)
        np.testing.assert_allclose(arr, np.zeros(DEX3_NUM_MOTORS))


def test_dex3_hand_state_message_rejects_wrong_size_payload() -> None:
    with pytest.raises(ValueError, match="expected 7 joint values"):
        Dex3HandStateMessage(left_target=np.zeros(6))

    with pytest.raises(ValueError, match="invalid size"):
        Dex3HandStateMessage.from_bytes(b"short")
