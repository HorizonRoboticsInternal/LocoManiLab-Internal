"""RobotIO for a host that already owns the robot.

``ZMQRobotIO`` and ``G1RobotIO`` both assume the runtime is the one talking to
the robot -- over a socket or over DDS. A host that already has the state in
hand and wants the command back in hand needs neither: an in-process simulator
tick, a test harness, a VLA closed-loop driver.

Usage is strictly ordered, once per policy tick::

    io.push_state(RobotState(...))   # what the robot is doing now
    tracking.step()                  # runtime reads it, writes a command
    q, dq, tau, kp, kd = io.last_command

``read_state`` returns the same state until the next ``push_state``, so a tick
that forgets to push re-runs the policy on stale state rather than blocking --
the same failure mode as a stalled socket, and deliberately so: the runtime's
own staleness handling stays in charge.
"""

from __future__ import annotations

import numpy as np

from sim2real.rl_policy.robot_io.base import RobotIO, RobotState


class PushRobotIO(RobotIO):
    """State pushed in by the host, command read back off ``last_command``."""

    def __init__(self) -> None:
        self._latest_state: RobotState | None = None
        #: The most recent ``write_command`` arguments, or None before the
        #: runtime has produced one. Tuple of (q_target, dq_target, tau_ff, kp, kd).
        self.last_command: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
        ] | None = None

    def push_state(self, state: RobotState) -> None:
        self._latest_state = state

    def read_state(self) -> RobotState | None:
        return self._latest_state

    def write_command(self, q_target, dq_target, tau_ff, kp, kd) -> None:
        self.last_command = (
            np.asarray(q_target, dtype=np.float32),
            np.asarray(dq_target, dtype=np.float32),
            np.asarray(tau_ff, dtype=np.float32),
            np.asarray(kp, dtype=np.float32),
            np.asarray(kd, dtype=np.float32),
        )

    def close(self) -> None:
        return None
