"""Minimal ZMQ subscriber for the existing hand-grip channel (see utils.common.HandGripMessage).

Separate from ``utils.dh116s.HandGripConsumer`` (which drives a physical DH116S hand) --
this just exposes the latest ``(left_grip, right_grip)`` pair for recording, with no
actuator dependency.
"""

from __future__ import annotations

import zmq
from loguru import logger

from sim2real.utils.common import PORTS, HandGripMessage


class HandGripClient:
    def __init__(self, connect: str = f"tcp://127.0.0.1:{PORTS['hand_grip']}") -> None:
        self._connect = connect
        self._available = True
        self._latest: tuple[float, float] | None = None

        self._zmq_context = zmq.Context.instance()
        self._socket = self._zmq_context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")

        try:
            self._socket.connect(connect)
        except Exception as exc:
            self._available = False
            self._socket.close(0)
            logger.warning(f"Hand-grip ZMQ subscriber unavailable, continuing without it: {exc}")
        else:
            logger.info("Hand-grip ZMQ subscriber connected to {}", connect)

    def read(self) -> tuple[float, float] | None:
        if not self._available:
            return self._latest
        try:
            raw = self._socket.recv(flags=zmq.DONTWAIT)
        except zmq.Again:
            return self._latest
        try:
            message = HandGripMessage.from_bytes(raw)
        except Exception as exc:
            logger.debug(f"Hand-grip ZMQ decode error: {exc}")
            return self._latest
        self._latest = (message.left_grip, message.right_grip)
        return self._latest

    def close(self) -> None:
        try:
            self._socket.close(0)
        except Exception as exc:
            logger.debug(f"Failed to stop hand-grip subscriber cleanly: {exc}")
