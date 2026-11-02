"""Shared ZMQ hand-grip consumer, decoupled from any specific hand hardware."""

from __future__ import annotations

import time

from sim2real.utils.common import HandGripMessage


class HandGripConsumer:
    """Decode hand-grip messages and apply them to a grip-driven hand driver."""

    def __init__(self, driver) -> None:
        self.driver = driver
        self.last_message: HandGripMessage | None = None
        self.last_receive_monotonic: float | None = None

    def consume(
        self,
        data: bytes,
        *,
        received_monotonic: float | None = None,
    ) -> HandGripMessage:
        message = HandGripMessage.from_bytes(data)
        self.driver.apply_grips(message.left_grip, message.right_grip)
        self.last_message = message
        self.last_receive_monotonic = (
            time.monotonic() if received_monotonic is None else float(received_monotonic)
        )
        return message

    def is_stale(self, *, now_monotonic: float, timeout_s: float) -> bool:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        return (
            self.last_receive_monotonic is not None
            and now_monotonic - self.last_receive_monotonic >= timeout_s
        )
