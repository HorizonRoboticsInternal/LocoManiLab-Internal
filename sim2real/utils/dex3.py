"""Dex3 hand control helpers for normalized PICO grip commands.

Drives the Unitree Dex3 hand (7 actuated joints per hand: thumb_0/1/2,
index_0/1, middle_0/1) over the same DDS bus used for arm/leg control
(`unitree_sdk2py`), on topics `rt/dex3/{left,right}/cmd` and
`rt/dex3/{left,right}/state`, using `unitree_hg.msg.dds_.HandCmd_`/
`HandState_`. Unlike DH116S (a third-party CAN-FD hand), Dex3 needs no
separate vendor SDK download -- only the official `unitree_sdk2py` package.
"""

from __future__ import annotations

import importlib
import threading
from typing import Literal

import numpy as np

from sim2real.utils.hand_grip_consumer import HandGripConsumer

__all__ = [
    "Dex3HandDriver",
    "DryRunDex3HandDriver",
    "HandGripConsumer",
    "compute_pd_torques",
    "grip_to_joint_targets",
]

HandSide = Literal["left", "right", "double"]

DEX3_NUM_MOTORS = 7
DEX3_JOINT_NAMES = (
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "index_0",
    "index_1",
    "middle_0",
    "middle_1",
)

# Joint ranges (radians), matching this Dex3 hand's reference MJCF
# (assets/g1_unified/robot.xml -- see that directory's README). Nothing reads
# them off the asset any more: the runtime hand splice that did (and the
# separate reference asset it needed) is gone, load_sim_model selects the
# hand-ful MJCF instead. Keep these in sync by hand if it ever changes. q=0 is the
# open pose; left/right ranges mirror in sign around it. Order matches
# DEX3_JOINT_NAMES: thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1.
DEX3_JOINT_RANGES: dict[str, tuple[tuple[float, float], ...]] = {
    "left": (
        (-1.0472, 1.0472),
        (-0.724312, 1.0472),
        (0.0, 1.74533),
        (-1.5708, 0.0),
        (-1.74533, 0.0),
        (-1.5708, 0.0),
        (-1.74533, 0.0),
    ),
    "right": (
        (-1.0472, 1.0472),
        (-1.0472, 0.724312),
        (-1.74533, 0.0),
        (0.0, 1.5708),
        (0.0, 1.74533),
        (0.0, 1.5708),
        (0.0, 1.74533),
    ),
}

# Fully-closed joint targets (q at grip=1.0), ported from heft's
# Dex3HandBridge (the original HeFT integration, g1_sim2real/src/g1_udp_bridge.cpp,
# kCloseLeft/kCloseRight) -- a grasp pose hand-tuned on real Dex3 hardware,
# not derived from the joint's own range limits. thumb_0 (the opposition/
# abduction joint) is pinned at 0 in every state, matching heft. Order
# matches DEX3_JOINT_NAMES.
DEX3_CLOSE_POSE: dict[str, tuple[float, ...]] = {
    "left": (0.0, 0.724, 1.05, -1.5, -1.5, -1.5, -1.5),
    "right": (0.0, -0.724, -1.05, 1.5, 1.5, 1.5, 1.5),
}
DEX3_THUMB_OPPOSITION_RAD = 0.0

DEFAULT_DOMAIN_ID = 0
DEFAULT_INTERFACE = "eth0"
# Real-hardware gains (Dex3HandDriver): the physical motor is its own
# position-servo closed loop; these are heft's Dex3HandBridge values
# (g1_sim2real/src/g1_udp_bridge.cpp HandConfig), tuned directly on real
# Dex3 hardware. heft uses a single kd for every motor including the thumb
# (only kp is split, per Unitree's own example) -- so thumb_kd == kd here.
DEFAULT_KP = 1.5
DEFAULT_KD = 0.1
DEFAULT_THUMB_KP = 2.0
DEFAULT_THUMB_KD = 0.1
DEFAULT_MAX_DELTA_Q = 0.25

# Dex3SimHandBridge (sim2real.sim_env.utils.bridge) defaults to these same
# gains. Verified empirically: they reach the close pose within about a
# second under the reference asset's actuator ctrlrange cap (+/-0.7 Nm for
# every joint but thumb_0), with some overshoot/ringing (heft's kd=0.1 is
# tuned against the real motor's own internal damping, which the sim's PD
# doesn't reproduce) -- pass --dex3-kd/--dex3-thumb-kd on base_sim.py with a
# higher value than this if that ringing is undesirable for a given use case.


def compute_pd_torques(
    q_target: np.ndarray,
    q_measured: np.ndarray,
    qvel_measured: np.ndarray,
    *,
    kp: float = DEFAULT_KP,
    kd: float = DEFAULT_KD,
    thumb_kp: float = DEFAULT_THUMB_KP,
    thumb_kd: float = DEFAULT_THUMB_KD,
) -> np.ndarray:
    """PD torque for the 7 Dex3 motors (index 0 = thumb_0, its own gains)."""
    q_target = np.asarray(q_target, dtype=np.float32)
    q_measured = np.asarray(q_measured, dtype=np.float32)
    qvel_measured = np.asarray(qvel_measured, dtype=np.float32)
    if not (q_target.shape == q_measured.shape == qvel_measured.shape == (DEX3_NUM_MOTORS,)):
        raise ValueError(f"expected shape ({DEX3_NUM_MOTORS},) arrays")

    kp_vec = np.full(DEX3_NUM_MOTORS, kp, dtype=np.float32)
    kd_vec = np.full(DEX3_NUM_MOTORS, kd, dtype=np.float32)
    kp_vec[0] = thumb_kp
    kd_vec[0] = thumb_kd
    return kp_vec * (q_target - q_measured) - kd_vec * qvel_measured


def grip_to_joint_targets(grip: float, side: Literal["left", "right"]) -> np.ndarray:
    """Map a normalized grip in [0, 1] to the Dex3 seven-motor grasp pose."""
    grip_value = float(grip)
    if not np.isfinite(grip_value) or not 0.0 <= grip_value <= 1.0:
        raise ValueError("grip must be finite and within [0, 1]")
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    close_pose = np.asarray(DEX3_CLOSE_POSE[side], dtype=np.float32)
    targets = (close_pose * grip_value).astype(np.float32)
    # thumb_0 is the opposition/abduction joint; hold it fixed instead of
    # modulating it with the trigger, matching heft's Dex3HandBridge (and the
    # DH116S precedent of keeping the thumb alignment joint at a constant
    # grasp pose).
    targets[0] = DEX3_THUMB_OPPOSITION_RAD
    return targets


def _pack_hand_motor_mode(*, enabled: bool) -> int:
    """Pack a Dex3 MotorCmd_.mode byte (bit 4 = FOC-enable status)."""
    status = 1 if enabled else 0
    return status << 4


class Dex3HandDriver:
    """Drives the Unitree Dex3 hand over DDS from normalized grip commands."""

    def __init__(
        self,
        *,
        hand_dir: HandSide = "double",
        domain_id: int = DEFAULT_DOMAIN_ID,
        interface: str = DEFAULT_INTERFACE,
        kp: float = DEFAULT_KP,
        kd: float = DEFAULT_KD,
        thumb_kp: float = DEFAULT_THUMB_KP,
        thumb_kd: float = DEFAULT_THUMB_KD,
        max_delta_q: float = DEFAULT_MAX_DELTA_Q,
        channel_factory_initialize=None,
        channel_publisher_cls=None,
        channel_subscriber_cls=None,
        hand_cmd_cls=None,
        hand_state_cls=None,
        hand_cmd_factory=None,
    ) -> None:
        if hand_dir not in {"left", "right", "double"}:
            raise ValueError("hand_dir must be 'left', 'right', or 'double'")
        if float(max_delta_q) <= 0:
            raise ValueError("max_delta_q must be > 0")

        self.hand_dir = hand_dir
        self.kp = float(kp)
        self.kd = float(kd)
        self.thumb_kp = float(thumb_kp)
        self.thumb_kd = float(thumb_kd)
        self.max_delta_q = float(max_delta_q)
        self._lock = threading.Lock()
        self._publishers: dict[str, object] = {}
        self._subscribers: dict[str, object] = {}
        self._last_state_q: dict[str, np.ndarray] = {}
        self._last_target_q: dict[str, np.ndarray] = {}

        need_defaults = None in (
            channel_factory_initialize,
            channel_publisher_cls,
            channel_subscriber_cls,
            hand_cmd_cls,
            hand_state_cls,
            hand_cmd_factory,
        )
        if need_defaults:
            channel_module = importlib.import_module("unitree_sdk2py.core.channel")
            idl_module = importlib.import_module("unitree_sdk2py.idl.unitree_hg.msg.dds_")
            # HandCmd_'s `motor_cmd` field is a variable-length IDL sequence,
            # not a fixed array: it starts empty unless built through this
            # factory, which pre-allocates exactly DEX3_NUM_MOTORS entries
            # (mirrors the official SDK's own low-level examples).
            default_module = importlib.import_module("unitree_sdk2py.idl.default")
            channel_factory_initialize = channel_factory_initialize or channel_module.ChannelFactoryInitialize
            channel_publisher_cls = channel_publisher_cls or channel_module.ChannelPublisher
            channel_subscriber_cls = channel_subscriber_cls or channel_module.ChannelSubscriber
            hand_cmd_cls = hand_cmd_cls or idl_module.HandCmd_
            hand_state_cls = hand_state_cls or idl_module.HandState_
            hand_cmd_factory = hand_cmd_factory or default_module.unitree_hg_msg_dds__HandCmd_

        self._hand_cmd_cls = hand_cmd_cls
        self._hand_cmd_factory = hand_cmd_factory
        channel_factory_initialize(domain_id, interface)

        sides = ("left", "right") if hand_dir == "double" else (hand_dir,)
        for side in sides:
            self._last_state_q[side] = np.zeros(DEX3_NUM_MOTORS, dtype=np.float32)
            self._last_target_q[side] = np.zeros(DEX3_NUM_MOTORS, dtype=np.float32)

            publisher = channel_publisher_cls(f"rt/dex3/{side}/cmd", hand_cmd_cls)
            publisher.Init()
            self._publishers[side] = publisher

            subscriber = channel_subscriber_cls(f"rt/dex3/{side}/state", hand_state_cls)
            subscriber.Init(self._make_state_handler(side), 10)
            self._subscribers[side] = subscriber

    def _make_state_handler(self, side: str):
        def _handler(message) -> None:
            values = np.asarray(
                [motor.q for motor in message.motor_state[:DEX3_NUM_MOTORS]],
                dtype=np.float32,
            )
            with self._lock:
                self._last_state_q[side] = values

        return _handler

    def apply_grips(self, left_grip: float, right_grip: float) -> None:
        grips = {"left": left_grip, "right": right_grip}
        for side, publisher in self._publishers.items():
            target = grip_to_joint_targets(grips[side], side)
            with self._lock:
                measured = self._last_state_q[side]
                clamped = np.clip(
                    target,
                    measured - self.max_delta_q,
                    measured + self.max_delta_q,
                ).astype(np.float32)
                self._last_target_q[side] = clamped
            self._send_targets(publisher, clamped)

    def _send_targets(self, publisher, targets: np.ndarray) -> None:
        cmd = self._hand_cmd_factory()
        for idx, q in enumerate(targets):
            motor = cmd.motor_cmd[idx]
            motor.mode = _pack_hand_motor_mode(enabled=True)
            motor.q = float(q)
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = self.thumb_kp if idx == 0 else self.kp
            motor.kd = self.thumb_kd if idx == 0 else self.kd
        publisher.Write(cmd)

    def read_feedback(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {side: q.copy() for side, q in self._last_state_q.items()}

    def read_targets(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {side: q.copy() for side, q in self._last_target_q.items()}

    def close(self) -> None:
        for publisher in self._publishers.values():
            self._relax(publisher)
        self._publishers.clear()
        self._subscribers.clear()

    def _relax(self, publisher) -> None:
        cmd = self._hand_cmd_factory()
        for idx in range(DEX3_NUM_MOTORS):
            motor = cmd.motor_cmd[idx]
            motor.mode = _pack_hand_motor_mode(enabled=False)
            motor.q = 0.0
            motor.dq = 0.0
            motor.tau = 0.0
            motor.kp = 0.0
            motor.kd = 0.0
        try:
            publisher.Write(cmd)
        except Exception:
            pass


class DryRunDex3HandDriver:
    """No-DDS driver used to validate transport and mapping."""

    def __init__(self, hand_dir: HandSide = "double") -> None:
        self.hand_dir = hand_dir
        self.last_grips: tuple[float, float] | None = None
        self.last_targets: dict[str, np.ndarray] = {}

    def apply_grips(self, left_grip: float, right_grip: float) -> None:
        self.last_grips = (float(left_grip), float(right_grip))
        sides = ("left", "right") if self.hand_dir == "double" else (self.hand_dir,)
        values = {"left": left_grip, "right": right_grip}
        self.last_targets = {
            side: grip_to_joint_targets(values[side], side) for side in sides
        }

    def read_feedback(self) -> dict[str, np.ndarray]:
        return {side: target.copy() for side, target in self.last_targets.items()}

    def read_targets(self) -> dict[str, np.ndarray]:
        return self.read_feedback()

    def close(self) -> None:
        pass
