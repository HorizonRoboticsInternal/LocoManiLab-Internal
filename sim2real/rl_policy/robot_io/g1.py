from __future__ import annotations

import importlib
from importlib.machinery import EXTENSION_SUFFIXES
import subprocess
import sys
import time
from types import ModuleType
from typing import Callable

import numpy as np
from loguru import logger

from sim2real.config.robots.base import RobotCfg
from sim2real.rl_policy.robot_io.base import RobotIO, RobotState
from sim2real.utils.common import UNITREE_LEGGED_CONST


def _is_native_extension(module: ModuleType) -> bool:
    """Whether ``module`` is the expected pybind extension, not a namesake.

    ``unitree-interface==0.1.0`` on PyPI currently resolves to a pure-Python
    package with a different factory signature. The deployment binding is a
    compiled ``unitree_interface*.so``. Module objects without ``__file__`` are
    accepted so injected test doubles keep working.
    """
    origin = getattr(module, "__file__", None)
    return origin is None or any(str(origin).endswith(suffix) for suffix in EXTENSION_SUFFIXES)


class _Sdk2G1Robot:
    """The existing SDK2 DDS path without the separate ZMQ bridge process."""

    def __init__(self, *, interface: str, domain_id: int, joint_count: int) -> None:
        try:
            channel_module = importlib.import_module("unitree_sdk2py.core.channel")
            idl_module = importlib.import_module(
                "unitree_sdk2py.idl.unitree_hg.msg.dds_"
            )
            default_module = importlib.import_module("unitree_sdk2py.idl.default")
            crc_module = importlib.import_module("unitree_sdk2py.utils.crc")
        except ImportError as exc:
            raise ImportError(
                "unitree_sdk2py is required for robot_io='inline' with robot='g1'. "
                "Install the robot-g1 optional dependency."
            ) from exc

        self._joint_count = joint_count
        channel_module.ChannelFactoryInitialize(domain_id, interface)
        self._state_subscriber = channel_module.ChannelSubscriber(
            "rt/lowstate", idl_module.LowState_
        )
        self._state_subscriber.Init(handler=None, queueLen=0)
        self._command_publisher = channel_module.ChannelPublisher(
            "rt/lowcmd", idl_module.LowCmd_
        )
        self._command_publisher.Init()
        self._low_command = default_module.unitree_hg_msg_dds__LowCmd_()
        self._crc = crc_module.CRC()
        self._initialize_low_command()

    def _initialize_low_command(self) -> None:
        command = self._low_command
        command.level_flag = UNITREE_LEGGED_CONST["LOWLEVEL"]
        command.gpio = 0
        command.mode_machine = 5
        command.mode_pr = 0
        for motor in command.motor_cmd:
            motor.mode = 1
            motor.q = UNITREE_LEGGED_CONST["PosStopF"]
            motor.kp = 0.0
            motor.dq = UNITREE_LEGGED_CONST["VelStopF"]
            motor.kd = 0.0
            motor.tau = 0.0

    def set_control_mode(self, _mode: object) -> None:
        # G1RobotIO only requests PR. This is the same value real_bridge.py uses.
        self._low_command.mode_pr = 0

    def read_low_state(self):
        return self._state_subscriber.Read()

    @staticmethod
    def create_zero_command():
        # G1RobotIO fills every field before passing this back to us.
        from types import SimpleNamespace

        return SimpleNamespace()

    def write_low_command(self, command) -> None:
        if any(
            np.asarray(values).size != self._joint_count
            for values in (
                command.q_target,
                command.dq_target,
                command.tau_ff,
                command.kp,
                command.kd,
            )
        ):
            raise ValueError(
                f"inline G1 command must have {self._joint_count} entries per field"
            )
        for index in range(self._joint_count):
            motor = self._low_command.motor_cmd[index]
            motor.q = float(command.q_target[index])
            motor.dq = float(command.dq_target[index])
            motor.tau = float(command.tau_ff[index])
            motor.kp = float(command.kp[index])
            motor.kd = float(command.kd[index])
        self._low_command.crc = self._crc.Crc(self._low_command)
        self._command_publisher.Write(self._low_command)

    def close(self) -> None:
        for channel in (self._state_subscriber, self._command_publisher):
            close = getattr(channel, "Close", None)
            if callable(close):
                close()


class G1RobotIO(RobotIO):
    """Inline G1 DDS backend, using native binding or SDK2 directly."""

    def __init__(
        self,
        robot_cfg: RobotCfg,
        *,
        interface: str,
        sdk_module: ModuleType | None = None,
        motion_switcher_factory: Callable[[], object] | None = None,
        channel_factory_initialize: Callable[[int, str], None] | None = None,
        motion_release_delay_s: float = 1.0,
    ) -> None:
        self._joint_count = len(robot_cfg.joint_names)
        self._robot = None
        try:
            # The Python SDK2 motion-switcher and the C++ inline SDK share the
            # same DDS domain.  Release the high-level motion service before
            # even importing the inline SDK, because its module import creates
            # a DDS participant and can make CycloneDDS reject SDK2 RPC topics.
            if (
                motion_switcher_factory is None
                and channel_factory_initialize is None
            ):
                self._switch_to_debug_mode_in_subprocess(
                    domain_id=int(robot_cfg.domain_id),
                    interface=interface,
                    release_delay_s=motion_release_delay_s,
                )
            else:
                self._switch_to_debug_mode(
                    domain_id=int(robot_cfg.domain_id),
                    interface=interface,
                    motion_switcher_factory=motion_switcher_factory,
                    channel_factory_initialize=channel_factory_initialize,
                    release_delay_s=motion_release_delay_s,
                )
            if sdk_module is None:
                try:
                    native_module = importlib.import_module("unitree_interface")
                except ImportError:
                    native_module = None
                if native_module is not None and _is_native_extension(native_module):
                    sdk_module = native_module
                elif native_module is not None:
                    logger.warning(
                        "Ignoring non-native unitree_interface at {}; using "
                        "unitree_sdk2py directly",
                        getattr(native_module, "__file__", "(unknown)"),
                    )

            if sdk_module is not None:
                self._robot = sdk_module.create_robot(
                    interface,
                    sdk_module.RobotType.G1,
                    sdk_module.MessageType.HG,
                )
                self._robot.set_control_mode(sdk_module.ControlMode.PR)
                self._io_backend = "unitree_interface"
            else:
                self._robot = _Sdk2G1Robot(
                    interface=interface,
                    domain_id=int(robot_cfg.domain_id),
                    joint_count=self._joint_count,
                )
                self._robot.set_control_mode("PR")
                self._io_backend = "unitree_sdk2py-direct"
            logger.info("G1 inline RobotIO backend: {}", self._io_backend)
            # Preserve the existing inline startup behavior: prime the SDK once.
            self._robot.read_low_state()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _switch_to_debug_mode_in_subprocess(
        *, domain_id: int, interface: str, release_delay_s: float
    ) -> None:
        command = [
            sys.executable,
            "-m",
            "sim2real.rl_policy.robot_io.g1_debug_mode",
            "--domain-id",
            str(domain_id),
            "--interface",
            interface,
            "--release-delay-s",
            str(release_delay_s),
        ]
        try:
            subprocess.run(command, check=True, timeout=30.0)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "Failed to switch the G1 to debug mode before inline SDK startup"
            ) from exc

    def _switch_to_debug_mode(
        self,
        *,
        domain_id: int,
        interface: str,
        motion_switcher_factory: Callable[[], object] | None,
        channel_factory_initialize: Callable[[int, str], None] | None,
        release_delay_s: float,
    ) -> None:
        if release_delay_s < 0:
            raise ValueError("motion_release_delay_s must be >= 0")
        if motion_switcher_factory is None or channel_factory_initialize is None:
            try:
                motion_switcher_module = importlib.import_module(
                    "unitree_sdk2py.comm.motion_switcher.motion_switcher_client"
                )
                channel_module = importlib.import_module("unitree_sdk2py.core.channel")
            except ImportError as exc:
                raise ImportError(
                    "unitree_sdk2py is required to switch inline G1 control "
                    "to debug mode. Install the robot-g1 optional dependency."
                ) from exc
            if motion_switcher_factory is None:
                motion_switcher_factory = motion_switcher_module.MotionSwitcherClient
            if channel_factory_initialize is None:
                channel_factory_initialize = channel_module.ChannelFactoryInitialize

        channel_factory_initialize(domain_id, interface)
        self._motion_switcher = motion_switcher_factory()
        self._motion_switcher.SetTimeout(5.0)
        self._motion_switcher.Init()

        while True:
            status, result = self._motion_switcher.CheckMode()
            if status != 0 or result is None:
                raise RuntimeError(
                    "Failed to query the G1 motion mode before inline control: "
                    f"status={status} result={result!r}"
                )
            if not result.get("name"):
                logger.info("G1 inline robot is in debug mode")
                return

            logger.info("Releasing active G1 motion mode: {}", result.get("name"))
            release_status, _ = self._motion_switcher.ReleaseMode()
            if release_status != 0:
                raise RuntimeError(
                    "Failed to release the active G1 motion mode before inline "
                    f"control: status={release_status} mode={result.get('name')!r}"
                )
            if release_delay_s > 0:
                time.sleep(release_delay_s)

    def read_state(self) -> RobotState | None:
        low_state = self._robot.read_low_state()
        if low_state is None:
            return None

        if hasattr(low_state, "imu_state"):
            # unitree_sdk2py's raw DDS message (the x86 direct path).
            imu = low_state.imu_state
            motors = low_state.motor_state
            quaternion = imu.quaternion
            gyroscope = imu.gyroscope
            joint_pos = [motors[i].q for i in range(self._joint_count)]
            joint_vel = [motors[i].dq for i in range(self._joint_count)]
            joint_torque_values = [
                motors[i].tau_est for i in range(self._joint_count)
            ]
            tick = int(getattr(low_state, "tick", time.time_ns() // 1_000_000))
        else:
            # Native unitree_interface's normalized state objects.
            quaternion = low_state.imu.quat
            gyroscope = low_state.imu.omega
            joint_pos = low_state.motor.q[: self._joint_count]
            joint_vel = low_state.motor.dq[: self._joint_count]
            joint_torque_values = low_state.motor.tau_est[: self._joint_count]
            tick = int(time.time_ns() // 1_000_000)

        qpos = np.zeros(7 + self._joint_count, dtype=np.float32)
        qvel = np.zeros(6 + self._joint_count, dtype=np.float32)
        qpos[3:7] = np.asarray(quaternion, dtype=np.float32)
        qpos[7:] = np.asarray(joint_pos, dtype=np.float32)
        qvel[3:6] = np.asarray(gyroscope, dtype=np.float32)
        qvel[6:] = np.asarray(joint_vel, dtype=np.float32)
        joint_torque = np.asarray(joint_torque_values, dtype=np.float32).copy()
        return RobotState(
            qpos=qpos,
            qvel=qvel,
            joint_torque=joint_torque,
            tick=tick,
        )

    def write_command(
        self,
        q_target: np.ndarray,
        dq_target: np.ndarray,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
    ) -> None:
        command = self._robot.create_zero_command()
        command.q_target = np.asarray(q_target, dtype=np.float32).copy()
        command.dq_target = np.asarray(dq_target, dtype=np.float32).copy()
        command.tau_ff = np.asarray(tau_ff, dtype=np.float32).copy()
        command.kp = np.asarray(kp, dtype=np.float32).copy().tolist()
        command.kd = np.asarray(kd, dtype=np.float32).copy().tolist()
        self._robot.write_low_command(command)

    def close(self) -> None:
        close = getattr(self._robot, "close", None)
        if callable(close):
            close()
