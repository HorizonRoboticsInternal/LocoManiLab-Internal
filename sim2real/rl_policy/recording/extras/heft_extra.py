"""heft controller Dex3 hand state -- ``observation.hand_state``/``action.hand_joints``.

Ported from LocoManiLab's original HeFT integration
(the original exporter's ``_add_data_frame`` +
``vla/features.py``'s ``get_features(..., include_hand=True)``): heft records the
Dex3 hand's 7-per-hand measured joint angles and commanded targets alongside the
generic body state.

Wire format: JSON messages on the ``heft_debug`` ZMQ topic (``topic_bytes +
json_bytes``, see that fork's ``vla/state_bus.py``), published by
``src/deploy.py --publish-state-addr`` -- a controller-side patch heft's fork adds on
top of this repo's ``BasePolicy``. Plain upstream ``sim2real`` does not publish this
topic on its own, so this extra only produces real (non-zero) values when the process
you're recording is a heft-fork build that runs with that flag; on any other build it
falls back to zeros for the full recording, matching the same defensive default the
original exporter uses (``proprio.get("hand_left_q", [0.0] * 7)``) rather than
raising, so a family that mistakenly enables this extra gets an obviously-flat
dataset rather than a crash.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import zmq
from loguru import logger

from sim2real.rl_policy.recording.extras.base import RecordingExtra

STATE_TOPIC = "heft_debug"
DEFAULT_HEFT_STATE_PORT = 5558
DEX3_MOTORS_PER_HAND = 7

_DEX3_JOINT_NAMES = [f"dex3_left_j{i}" for i in range(DEX3_MOTORS_PER_HAND)] + [
    f"dex3_right_j{i}" for i in range(DEX3_MOTORS_PER_HAND)
]
_ZERO_HAND = [0.0] * DEX3_MOTORS_PER_HAND


class HeftHandsRecordingExtra(RecordingExtra, extra_name="heft-hands"):
    def __init__(
        self,
        policy: Any,
        host: str = "localhost",
        port: int = DEFAULT_HEFT_STATE_PORT,
    ) -> None:
        super().__init__(policy)
        self._latest: dict | None = None

        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, STATE_TOPIC)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        try:
            self._socket.connect(f"tcp://{host}:{port}")
            logger.info("HeftHandsRecordingExtra connected to tcp://{}:{}", host, port)
        except Exception as exc:
            logger.warning(
                "HeftHandsRecordingExtra ZMQ connect failed, hand fields will be zero: {}", exc
            )

    def _poll(self) -> None:
        try:
            raw = self._socket.recv(zmq.NOBLOCK)
        except zmq.Again:
            return
        try:
            self._latest = json.loads(raw[len(STATE_TOPIC):].decode())
        except Exception as exc:
            logger.debug("HeftHandsRecordingExtra decode error: {}", exc)

    def features(self) -> dict[str, dict]:
        return {
            "observation.hand_state": {
                "dtype": "float32",
                "shape": (2 * DEX3_MOTORS_PER_HAND,),
                "names": list(_DEX3_JOINT_NAMES),
            },
            "action.hand_joints": {
                "dtype": "float32",
                "shape": (2 * DEX3_MOTORS_PER_HAND,),
                "names": [f"target_{n}" for n in _DEX3_JOINT_NAMES],
            },
        }

    def compute(self, frame: dict[str, Any]) -> dict[str, Any]:
        self._poll()
        proprio = self._latest or {}
        return {
            "observation.hand_state": np.concatenate(
                [
                    np.asarray(proprio.get("hand_left_q", _ZERO_HAND), dtype=np.float32),
                    np.asarray(proprio.get("hand_right_q", _ZERO_HAND), dtype=np.float32),
                ]
            ),
            "action.hand_joints": np.concatenate(
                [
                    np.asarray(proprio.get("hand_left_target", _ZERO_HAND), dtype=np.float32),
                    np.asarray(proprio.get("hand_right_target", _ZERO_HAND), dtype=np.float32),
                ]
            ),
        }

    def close(self) -> None:
        try:
            self._socket.close(0)
        except Exception as exc:
            logger.debug("Failed to stop HeftHandsRecordingExtra socket cleanly: {}", exc)
