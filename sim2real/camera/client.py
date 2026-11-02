"""Non-blocking ZMQ SUB client for the generic camera channel.

Uses RCVHWM + drain-to-latest rather than ``zmq.CONFLATE``: CONFLATE can leave a
subscriber that connects *after* the publisher has already sent messages permanently
starved (the conflate buffer only fills on receipt, so a late joiner sees nothing until
the next fresh publish) -- draining to the newest of whatever is queued avoids that
class of race at the cost of a small, bounded receive buffer instead of a single slot.
"""

from __future__ import annotations

import zmq
from loguru import logger

from sim2real.camera.messages import decode_camera_frame
from sim2real.utils.common import PORTS


class CameraClient:
    def __init__(
        self,
        connect: str = f"tcp://127.0.0.1:{PORTS['camera']}",
        hwm: int = 5,
    ) -> None:
        self._connect = connect
        self._available = True
        self._latest: dict | None = None

        self._zmq_context = zmq.Context.instance()
        self._socket = self._zmq_context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVHWM, int(hwm))
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")

        try:
            self._socket.connect(connect)
        except Exception as exc:
            self._available = False
            self._socket.close(0)
            logger.warning(f"Camera ZMQ subscriber unavailable, continuing without it: {exc}")
        else:
            logger.info("Camera ZMQ subscriber connected to {}", connect)

    def read(self) -> dict | None:
        """Drain any queued messages and return the most recent decoded frame, or None."""
        if not self._available:
            return self._latest

        while True:
            try:
                raw = self._socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break
            try:
                self._latest = decode_camera_frame(raw)
            except Exception as exc:
                logger.debug(f"Camera ZMQ decode error: {exc}")

        return self._latest

    def close(self) -> None:
        try:
            self._socket.close(0)
        except Exception as exc:
            logger.debug(f"Failed to stop camera subscriber cleanly: {exc}")
