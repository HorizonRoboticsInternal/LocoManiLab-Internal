import numpy as np
import mujoco
import zmq

from loguru import logger

from sim2real.config.robots.base import RobotCfg
from sim2real.utils.common import LowStateMessage, LowCmdMessage, PORTS, HandGripMessage
from sim2real.utils.dex3 import (
    DEFAULT_KD,
    DEFAULT_KP,
    DEFAULT_THUMB_KD,
    DEFAULT_THUMB_KP,
    DEX3_JOINT_NAMES,
    compute_pd_torques,
    grip_to_joint_targets,
)
from sim2real.utils.strings import resolve_matching_names_values


class SimulationBridge:

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
        robot_cfg: RobotCfg,
    ):
        self.robot_cfg = robot_cfg
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.torques = np.zeros(self.mj_model.nu)

        # ZMQ communication setup
        self.zmq_context = zmq.Context.instance()

        self.low_state_port = self.robot_cfg.low_state_port
        low_state_bind_addr = self.robot_cfg.low_state_bind_addr
        low_state_endpoint = f"tcp://{low_state_bind_addr}:{self.low_state_port}"
        self.low_state_pub = self.zmq_context.socket(zmq.PUB)
        self.low_state_pub.setsockopt(zmq.SNDHWM, 1)
        self.low_state_pub.setsockopt(zmq.LINGER, 0)
        self.low_state_pub.bind(low_state_endpoint)

        self.low_cmd_port = self.robot_cfg.low_cmd_port
        low_cmd_host = self.robot_cfg.low_cmd_host
        low_cmd_endpoint = f"tcp://{low_cmd_host}:{self.low_cmd_port}"
        self.low_cmd_sub = self.zmq_context.socket(zmq.SUB)
        self.low_cmd_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.low_cmd_sub.setsockopt(zmq.CONFLATE, 1)
        self.low_cmd_sub.setsockopt(zmq.RCVTIMEO, 0)
        self.low_cmd_sub.setsockopt(zmq.LINGER, 0)
        self.low_cmd_sub.connect(low_cmd_endpoint)

        total_joints = len(self.robot_cfg.joint_names)
        self.cmd_q = np.zeros(total_joints, dtype=np.float32)
        self.cmd_dq = np.zeros(total_joints, dtype=np.float32)
        self.cmd_tau = np.zeros(total_joints, dtype=np.float32)
        self.cmd_kp = np.zeros(total_joints, dtype=np.float32)
        self.cmd_kd = np.zeros(total_joints, dtype=np.float32)
        self.has_received_command = False

        self.init_joint_indices()

    def init_joint_indices(self):
        joint_names_mujoco = [
            self.mj_model.joint(i).name for i in range(self.mj_model.njnt)
        ]
        actuator_names_mujoco = [
            self.mj_model.actuator(i).name for i in range(self.mj_model.nu)
        ]
        self.joint_indices_unitree = []
        self.qpos_adrs = []
        self.qvel_adrs = []
        self.act_adrs = []

        for name in self.robot_cfg.joint_names:
            if name not in joint_names_mujoco or name not in actuator_names_mujoco:
                continue
            self.joint_indices_unitree.append(self.robot_cfg.joint_names.index(name))

            joint_idx = joint_names_mujoco.index(name)
            self.qpos_adrs.append(self.mj_model.jnt_qposadr[joint_idx])
            self.qvel_adrs.append(self.mj_model.jnt_dofadr[joint_idx])
            self.act_adrs.append(actuator_names_mujoco.index(name))
        
        root_joint_idx = None
        for root_joint_name in self.robot_cfg.root_joint_names:
            if root_joint_name in joint_names_mujoco:
                root_joint_idx = joint_names_mujoco.index(root_joint_name)
                break
        if root_joint_idx is None:
            raise ValueError("No root joint found in the MuJoCo model.")
        self.root_qpos_adr = self.mj_model.jnt_qposadr[root_joint_idx]
        self.root_qvel_adr = self.mj_model.jnt_dofadr[root_joint_idx]

        joint_effort_limit_dict = self.robot_cfg.joint_effort_limit
        joint_indices, joint_names_matched, joint_effort_limit = (
            resolve_matching_names_values(
                joint_effort_limit_dict,
                joint_names_mujoco,
                preserve_order=True,
                strict=False,
            )
        )
        self.joint_effort_limit_mjc = np.array(joint_effort_limit)
        self.joint_idx_in_ctrl = np.array(
            [actuator_names_mujoco.index(name) for name in joint_names_matched]
        )

    def compute_torques(self):
        self.torques[:] = 0.0
        self._poll_low_cmd()

        if self.has_received_command:
            for unitree_idx, qpos_addr, qvel_addr, act_addr in zip(
                self.joint_indices_unitree,
                self.qpos_adrs,
                self.qvel_adrs,
                self.act_adrs,
            ):
                q_des = self.cmd_q[unitree_idx]
                dq_des = self.cmd_dq[unitree_idx]
                tau_ff = self.cmd_tau[unitree_idx]
                kp = self.cmd_kp[unitree_idx]
                kd = self.cmd_kd[unitree_idx]

                self.torques[act_addr] = (
                    tau_ff
                    + kp * (q_des - self.mj_data.qpos[qpos_addr])
                    + kd * (dq_des - self.mj_data.qvel[qvel_addr])
                )
        # Set the torque limit
        self.torques[self.joint_idx_in_ctrl] = np.clip(
            self.torques[self.joint_idx_in_ctrl],
            -self.joint_effort_limit_mjc,
            self.joint_effort_limit_mjc,
        )

    def _poll_low_cmd(self):
        """Non-blocking command subscriber that keeps the most recent message."""
        if self.low_cmd_sub is None:
            return

        updated = False
        while True:
            try:
                data = self.low_cmd_sub.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break

            try:
                low_cmd = LowCmdMessage.from_bytes(data)
            except Exception as exc:
                logger.warning(f"Failed to decode low command message: {exc}")
                continue

            if low_cmd.q_target.size != len(self.robot_cfg.joint_names):
                logger.warning(
                    "Received low command with unexpected size {}",
                    low_cmd.q_target.size,
                )
                continue

            self.cmd_q[:] = low_cmd.q_target
            self.cmd_dq[:] = low_cmd.dq_target
            self.cmd_tau[:] = low_cmd.tau_ff
            self.cmd_kp[:] = low_cmd.kp
            self.cmd_kd[:] = low_cmd.kd
            updated = True

        if updated:
            self.has_received_command = True

    def publish_low_state(self):
        if self.mj_data is None:
            return

        joint_pos_partial = self.mj_data.qpos[self.qpos_adrs]
        joint_vel_partial = self.mj_data.qvel[self.qvel_adrs]
        joint_torque_partial = self.mj_data.actuator_force[self.act_adrs]

        joint_pos_full = np.zeros(len(self.robot_cfg.joint_names), dtype=np.float32)
        joint_vel_full = np.zeros(len(self.robot_cfg.joint_names), dtype=np.float32)
        joint_tau_full = np.zeros(len(self.robot_cfg.joint_names), dtype=np.float32)
        for mjc_idx, unitree_idx in enumerate(self.joint_indices_unitree):
            joint_pos_full[unitree_idx] = joint_pos_partial[mjc_idx]
            joint_vel_full[unitree_idx] = joint_vel_partial[mjc_idx]
            joint_tau_full[unitree_idx] = joint_torque_partial[mjc_idx]

        # quaternion: w, x, y, z
        root_quat_w = self.mj_data.qpos[self.root_qpos_adr + 3:self.root_qpos_adr+7]

        # angular velocity: x, y, z
        root_ang_vel_b = self.mj_data.qvel[self.root_qvel_adr + 3:self.root_qvel_adr+6]
        low_state_msg = LowStateMessage(
            quaternion=root_quat_w,
            gyroscope=root_ang_vel_b,
            joint_positions=joint_pos_full,
            joint_velocities=joint_vel_full,
            joint_torques=joint_tau_full,
            tick=int(self.mj_data.time * 1e3),
        )

        try:
            self.low_state_pub.send(low_state_msg.to_bytes(), flags=zmq.DONTWAIT)
        except zmq.Again:
            pass


class Dex3SimHandBridge:
    """Drives Dex3 hand joints in the sim via local PD control, from the same
    ``hand_grip`` ZMQ stream the real-hardware ``Dex3HandDriver`` consumes.

    Independent of ``SimulationBridge``/``RobotCfg.joint_names`` -- resolves
    hand joint qpos/qvel/actuator addresses by name directly against the
    loaded ``mj_model`` (populated by ``add_dex3_hands``), so it composes with
    the 29-DOF body control without touching its protocol or joint list.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        *,
        hand_grip_connect: str = f"tcp://127.0.0.1:{PORTS['hand_grip']}",
        kp: float = DEFAULT_KP,
        kd: float = DEFAULT_KD,
        thumb_kp: float = DEFAULT_THUMB_KP,
        thumb_kd: float = DEFAULT_THUMB_KD,
    ) -> None:
        self.mj_model = mj_model
        self.kp = kp
        self.kd = kd
        self.thumb_kp = thumb_kp
        self.thumb_kd = thumb_kd
        joint_names_mujoco = [mj_model.joint(i).name for i in range(mj_model.njnt)]
        actuator_names_mujoco = [mj_model.actuator(i).name for i in range(mj_model.nu)]

        self.qpos_adrs: dict[str, list[int]] = {}
        self.qvel_adrs: dict[str, list[int]] = {}
        self.act_adrs: dict[str, list[int]] = {}
        for side in ("left", "right"):
            qpos_adrs: list[int] = []
            qvel_adrs: list[int] = []
            act_adrs: list[int] = []
            for joint_name in DEX3_JOINT_NAMES:
                full_name = f"{side}_hand_{joint_name}_joint"
                if full_name not in joint_names_mujoco:
                    break
                # Actuator naming is not consistent across the G1 MJCFs: the
                # hand-ful model names hand actuators without the "_joint"
                # suffix its joints carry. Same two-candidate lookup
                # robocasa's build_joint_mapping does for the body joints.
                act_name = next(
                    (
                        candidate
                        for candidate in (f"{side}_hand_{joint_name}", full_name)
                        if candidate in actuator_names_mujoco
                    ),
                    None,
                )
                if act_name is None:
                    break
                joint_idx = joint_names_mujoco.index(full_name)
                qpos_adrs.append(int(mj_model.jnt_qposadr[joint_idx]))
                qvel_adrs.append(int(mj_model.jnt_dofadr[joint_idx]))
                act_adrs.append(actuator_names_mujoco.index(act_name))
            if len(qpos_adrs) == len(DEX3_JOINT_NAMES):
                self.qpos_adrs[side] = qpos_adrs
                self.qvel_adrs[side] = qvel_adrs
                self.act_adrs[side] = act_adrs

        if not self.qpos_adrs:
            raise ValueError(
                "No Dex3 hand joints found in mj_model; load the model with "
                "load_sim_model(..., with_dex3_hands=True)."
            )

        self.zmq_context = zmq.Context.instance()
        self.hand_grip_sub = self.zmq_context.socket(zmq.SUB)
        self.hand_grip_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.hand_grip_sub.setsockopt(zmq.CONFLATE, 1)
        self.hand_grip_sub.setsockopt(zmq.RCVTIMEO, 0)
        self.hand_grip_sub.setsockopt(zmq.LINGER, 0)
        self.hand_grip_sub.connect(hand_grip_connect)

        self.last_grips = {"left": 0.0, "right": 0.0}

    def _poll_hand_grip(self) -> None:
        while True:
            try:
                data = self.hand_grip_sub.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                break
            try:
                message = HandGripMessage.from_bytes(data)
            except Exception as exc:
                logger.warning(f"Failed to decode hand-grip message: {exc}")
                continue
            self.last_grips["left"] = message.left_grip
            self.last_grips["right"] = message.right_grip

    def apply_torques(self, mj_data: mujoco.MjData, torques: np.ndarray) -> None:
        """Write PD torques for the hand's actuators into ``torques`` in-place."""
        self._poll_hand_grip()
        for side, act_adrs in self.act_adrs.items():
            q_target = grip_to_joint_targets(self.last_grips[side], side)
            q_measured = mj_data.qpos[self.qpos_adrs[side]]
            qvel_measured = mj_data.qvel[self.qvel_adrs[side]]
            torques[act_adrs] = compute_pd_torques(
                q_target,
                q_measured,
                qvel_measured,
                kp=self.kp,
                kd=self.kd,
                thumb_kp=self.thumb_kp,
                thumb_kd=self.thumb_kd,
            )

    def close(self) -> None:
        try:
            self.hand_grip_sub.close(0)
        except Exception as exc:
            logger.debug(f"Failed to stop hand-grip subscriber cleanly: {exc}")
