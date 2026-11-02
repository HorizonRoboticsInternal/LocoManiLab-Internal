"""Dex3 hand joint angles -- ``observation.hand_state``/``action.hand_joints``.

Records the Dex3 hand's 7-per-hand measured joint angles and commanded targets
alongside the generic body state, mirroring ``heft_extra.py``'s feature shape.

Two sources, tried in order:

1. In-process ``policy.dex3_driver`` (``BasePolicyArgs.dex3_hand_dir``) -- the
   normal path. ``build_lerobot_recorder`` adds this extra automatically
   whenever in-process Dex3 control is enabled, reading straight from the same
   ``Dex3HandDriver`` instance driving the hand; no ZMQ hop needed.
2. A ``Dex3HandStateMessage`` (see ``sim2real.utils.common``) published by a
   standalone ``scripts/dex3_control.py --publish-state-addr`` process, for the
   case where the hand is being bring-up/dry-run tested independently of a
   running tracking policy.

If neither source is available, this extra falls back to zeros for the full
recording rather than raising, so enabling it by mistake produces an
obviously-flat dataset instead of a crash.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import zmq
from loguru import logger

from sim2real.rl_policy.recording.extras.base import RecordingExtra
from sim2real.utils.common import PORTS, Dex3HandStateMessage
from sim2real.utils.dex3 import DEX3_JOINT_NAMES, DEX3_NUM_MOTORS

_JOINT_NAMES = [f"dex3_left_{name}" for name in DEX3_JOINT_NAMES] + [
    f"dex3_right_{name}" for name in DEX3_JOINT_NAMES
]
_ZERO_HAND = np.zeros(DEX3_NUM_MOTORS, dtype=np.float32)


class Dex3HandsRecordingExtra(RecordingExtra, extra_name="dex3"):
    def __init__(
        self,
        policy: Any,
        host: str = "localhost",
        port: int = PORTS["dex3_state"],
    ) -> None:
        super().__init__(policy)
        self._driver = getattr(policy, "dex3_driver", None)
        self._latest: Dex3HandStateMessage | None = None
        self._socket = None

        if self._driver is not None:
            logger.info("Dex3HandsRecordingExtra reading from in-process dex3_driver")
            return

        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        try:
            self._socket.connect(f"tcp://{host}:{port}")
            logger.info("Dex3HandsRecordingExtra connected to tcp://{}:{}", host, port)
        except Exception as exc:
            logger.warning(
                "Dex3HandsRecordingExtra ZMQ connect failed, hand fields will be zero: {}", exc
            )

    def _poll(self) -> None:
        try:
            raw = self._socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return
        try:
            self._latest = Dex3HandStateMessage.from_bytes(raw)
        except Exception as exc:
            logger.debug("Dex3HandsRecordingExtra decode error: {}", exc)

    def features(self) -> dict[str, dict]:
        return {
            "observation.hand_state": {
                "dtype": "float32",
                "shape": (2 * DEX3_NUM_MOTORS,),
                "names": list(_JOINT_NAMES),
            },
            "action.hand_joints": {
                "dtype": "float32",
                "shape": (2 * DEX3_NUM_MOTORS,),
                "names": [f"target_{n}" for n in _JOINT_NAMES],
            },
        }

    def compute(self, frame: dict[str, Any]) -> dict[str, Any]:
        if self._driver is not None:
            targets = self._driver.read_targets()
            measured = self._driver.read_feedback()
            return {
                "observation.hand_state": np.concatenate(
                    [measured.get("left", _ZERO_HAND), measured.get("right", _ZERO_HAND)]
                ),
                "action.hand_joints": np.concatenate(
                    [targets.get("left", _ZERO_HAND), targets.get("right", _ZERO_HAND)]
                ),
            }

        self._poll()
        state = self._latest
        if state is None:
            return {
                "observation.hand_state": np.concatenate([_ZERO_HAND, _ZERO_HAND]),
                "action.hand_joints": np.concatenate([_ZERO_HAND, _ZERO_HAND]),
            }
        return {
            "observation.hand_state": np.concatenate(
                [state.left_measured, state.right_measured]
            ),
            "action.hand_joints": np.concatenate([state.left_target, state.right_target]),
        }

    def close(self) -> None:
        if self._socket is None:
            return
        try:
            self._socket.close(0)
        except Exception as exc:
            logger.debug("Failed to stop Dex3HandsRecordingExtra socket cleanly: {}", exc)
