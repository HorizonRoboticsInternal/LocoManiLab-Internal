from __future__ import annotations

import numpy as np
import zmq

from sim2real.rl_policy.recording.extras.dex3_extra import Dex3HandsRecordingExtra
from sim2real.utils.common import Dex3HandStateMessage
from sim2real.utils.dex3 import DEX3_NUM_MOTORS, DryRunDex3HandDriver


def test_dex3_extra_features_declare_28_dim_fields() -> None:
    extra = Dex3HandsRecordingExtra(policy=None, port=0)
    try:
        features = extra.features()
        assert features["observation.hand_state"]["shape"] == (2 * DEX3_NUM_MOTORS,)
        assert features["action.hand_joints"]["shape"] == (2 * DEX3_NUM_MOTORS,)
        assert len(features["observation.hand_state"]["names"]) == 2 * DEX3_NUM_MOTORS
    finally:
        extra.close()


def test_dex3_extra_falls_back_to_zeros_when_nothing_published() -> None:
    extra = Dex3HandsRecordingExtra(policy=None, host="127.0.0.1", port=0)
    try:
        result = extra.compute({"step_index": 0})
        np.testing.assert_allclose(
            result["observation.hand_state"], np.zeros(2 * DEX3_NUM_MOTORS)
        )
        np.testing.assert_allclose(
            result["action.hand_joints"], np.zeros(2 * DEX3_NUM_MOTORS)
        )
    finally:
        extra.close()


def test_dex3_extra_reads_published_state_over_zmq() -> None:
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")

    extra = Dex3HandsRecordingExtra(policy=None, host="127.0.0.1", port=port)
    try:
        message = Dex3HandStateMessage(
            timestamp_ns=1,
            left_target=np.full(DEX3_NUM_MOTORS, 0.1, dtype=np.float32),
            left_measured=np.full(DEX3_NUM_MOTORS, 0.2, dtype=np.float32),
            right_target=np.full(DEX3_NUM_MOTORS, 0.3, dtype=np.float32),
            right_measured=np.full(DEX3_NUM_MOTORS, 0.4, dtype=np.float32),
        )

        # PUB/SUB needs the subscription to propagate before the message lands;
        # retry briefly instead of a fixed sleep to keep this test fast/robust.
        result = {}
        for _ in range(200):
            pub.send(message.to_bytes())
            result = extra.compute({"step_index": 0})
            if result["observation.hand_state"][0] != 0.0:
                break

        np.testing.assert_allclose(
            result["observation.hand_state"],
            np.concatenate([message.left_measured, message.right_measured]),
        )
        np.testing.assert_allclose(
            result["action.hand_joints"],
            np.concatenate([message.left_target, message.right_target]),
        )
    finally:
        extra.close()
        pub.close(0)


class _FakePolicy:
    def __init__(self, dex3_driver) -> None:
        self.dex3_driver = dex3_driver


def test_dex3_extra_prefers_in_process_driver_over_zmq() -> None:
    driver = DryRunDex3HandDriver(hand_dir="double")
    driver.apply_grips(0.25, 0.75)

    extra = Dex3HandsRecordingExtra(_FakePolicy(driver), port=0)
    try:
        # No ZMQ socket should even be opened when an in-process driver is present.
        assert extra._socket is None

        result = extra.compute({"step_index": 0})
        feedback = driver.read_feedback()
        np.testing.assert_allclose(
            result["observation.hand_state"],
            np.concatenate([feedback["left"], feedback["right"]]),
        )
        np.testing.assert_allclose(
            result["action.hand_joints"],
            np.concatenate([feedback["left"], feedback["right"]]),
        )
    finally:
        extra.close()  # should not raise even with no socket to close
