"""Multi-target (head + left wrist + right wrist) whole-body IK, vendored
from Psi0's real-robot deployment
(``real/teleop/robot_control/multi_target_ik.py`` -- ``PinkIKSolver`` /
``G1_29_BodyIK.solve_lower_ik``), verbatim except:

* ``G1_JOINT_NAME_MAP`` inlined (was imported from a sibling
  ``robot_arm_joints.py`` module not vendored here) -- identical to
  ``composite.py``'s own ``_JOINT_NAMES`` (G1_CFG.joint_names) order: 6 left
  leg, 6 right leg, 3 waist, 7 left arm, 7 right arm.
* Uses ``g1_body29_hand14_virtual.urdf`` (this directory) -- the same URDF
  ``g1_arm_ik.py`` uses, but with 4 virtual joints appended/inserted
  (``ik_pelvis_z_joint`` between world/pelvis, ``ik_torso_{yaw,pitch,roll}
  _joint`` between waist_pitch and torso_link) so the IK solver can treat
  pelvis height and torso orientation as free variables it solves for,
  jointly with the arm/wrist targets -- this is what makes it output
  ``(h_new, rpy_new)`` at all, unlike the simpler 2-arm-only ``g1_arm_ik.py``
  (``G1ArmIK``), which only ever solves arm/wrist joint angles against a
  fixed torso.

Used by ``robocasa/g1_host/bridge.py``'s "amo" backend to drive
``torso_height``/``torso_roll``/``torso_pitch``/``torso_yaw`` (AMO's
command[3..6]) from the operator's PICO head + 3-point wrist tracking --
``g1_arm_ik.py``/``G1ArmIK`` is unchanged and still used for the actual
arm/wrist joint-angle targets.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import numpy as np
import pink
import pinocchio as pin
from pink import solve_ik as pink_solve_ik
from pink.limits import VelocityLimit
from pink.tasks.task import Task

# Identical to composite.py's _JOINT_NAMES / G1_CFG.joint_names order.
G1_JOINT_NAME_MAP = {
    0: "left_hip_pitch_joint", 1: "left_hip_roll_joint", 2: "left_hip_yaw_joint",
    3: "left_knee_joint", 4: "left_ankle_pitch_joint", 5: "left_ankle_roll_joint",
    6: "right_hip_pitch_joint", 7: "right_hip_roll_joint", 8: "right_hip_yaw_joint",
    9: "right_knee_joint", 10: "right_ankle_pitch_joint", 11: "right_ankle_roll_joint",
    12: "waist_yaw_joint", 13: "waist_roll_joint", 14: "waist_pitch_joint",
    15: "left_shoulder_pitch_joint", 16: "left_shoulder_roll_joint",
    17: "left_shoulder_yaw_joint", 18: "left_elbow_joint",
    19: "left_wrist_roll_joint", 20: "left_wrist_pitch_joint", 21: "left_wrist_yaw_joint",
    22: "right_shoulder_pitch_joint", 23: "right_shoulder_roll_joint",
    24: "right_shoulder_yaw_joint", 25: "right_elbow_joint",
    26: "right_wrist_roll_joint", 27: "right_wrist_pitch_joint", 28: "right_wrist_yaw_joint",
}

AMO_CONFIG_DEFAULTS = {
    "AMO_IK_POS_W": "1.0",
    "AMO_IK_ORI_W": "0.5",
    # Psi0's own value -- head weighted above either arm.
    #
    # This was lowered to 1.0 for a while, together with a stiffer REG_RPY
    # below, to stop the robot squatting whenever the operator raised the
    # controllers at VR-engage: the solver was dragging pelvis height down as a
    # "cheap" way to satisfy the targets. That turned out to be a frame bug,
    # not a weighting problem -- the arm targets were being handed to Pink in
    # G1ArmIK's fixed-root frame while Pink's own pelvis floats at
    # ``_amo_torso_h``, so every arm target sat ~torso_h too low and the solver
    # closed the gap by squatting (see the pelvis_in_pink_world correction in
    # ``amo.py::_tick_amo_arms``). With the real cause fixed, the de-tuning is
    # no longer buying anything, so both weights are back at Psi0 parity.
    #
    # If squat-on-engage ever returns, suspect the frame handling first; these
    # weights only mask it.
    "AMO_IK_W_HEAD": "2.5",
    "AMO_IK_W_LEFT": "1.0",
    "AMO_IK_W_RIGHT": "1.0",
    "AMO_IK_LM": "1e-3",
    "AMO_IK_QP_DAMP": "1e-4",
    "AMO_IK_REG_Q": "0.0005",
    # REG_RPY is Psi0's own value; REG_H is deliberately 10x Psi0's 0.0004.
    #
    # Both were raised together as part of the squat-on-engage workaround (see
    # AMO_IK_W_HEAD above). The orientation half is back at parity now that the
    # frame bug behind that symptom is fixed. Pelvis HEIGHT keeps the stiffer
    # value: it is the channel with a hard 0.20-1.20 m travel and the one whose
    # drift is most visible to the operator, so it stays the conservative one
    # until there is a reason on hardware to loosen it. Both are env-var
    # overridable if torso follow-through feels too stiff.
    "AMO_IK_REG_H": "0.004",
    "AMO_IK_REG_RPY": "0.006",
    "AMO_IK_VEL_PZ": "8.0",
    "AMO_IK_VEL_YAW": "0.30",
    "AMO_IK_VEL_PITCH": "0.30",
    "AMO_IK_VEL_ROLL": "0.30",
    "AMO_IK_NO_LIMITS": "0",
    "AMO_QP_SOLVER": "quadprog",
    "AMO_SILENT": "1",
}
for _k, _v in AMO_CONFIG_DEFAULTS.items():
    os.environ.setdefault(_k, str(_v))


def quat_to_euler(quat: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    pitch = np.copysign(np.pi / 2, sinp) if np.abs(sinp) >= 1 else np.arcsin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def rotation_matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    sy = -R[2, 0]
    pitch = math.asin(np.clip(sy, -1.0, 1.0))
    roll = math.atan2(R[2, 1], R[2, 2])
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw], dtype=np.float64)


class FrameFrobeniusTask(Task):
    """Aligns a frame's position + orientation using a Frobenius-norm
    rotation error: e = [(p_des - p), vec(R_des - R)]."""

    def __init__(self, frame_name: str, model: pin.Model, data: pin.Data,
                 pos_w: float = 1.0, ori_w: float = 0.5,
                 lm_damping: float = 1e-2, gain: float = 1.0):
        super().__init__(
            cost=np.hstack([np.sqrt(pos_w) * np.ones(3), np.sqrt(ori_w) * np.ones(9)]),
            gain=gain, lm_damping=lm_damping,
        )
        self._frame_name = frame_name
        self._frame_id = model.getFrameId(frame_name)
        self._model = model
        self._data = data
        self._R_des = np.eye(3)
        self._p_des = np.zeros(3)

    def set_target(self, p_des: np.ndarray, R_des: np.ndarray):
        self._p_des = np.asarray(p_des, dtype=np.float64).reshape(3)
        self._R_des = np.asarray(R_des, dtype=np.float64).reshape(3, 3)

    def compute_error(self, configuration):
        oMf = configuration.get_transform_frame_to_world(self._frame_name)
        e_p = self._p_des - oMf.translation
        e_R = (self._R_des - oMf.rotation).reshape(9)
        return np.hstack([e_p, e_R]).astype(np.float64)

    def compute_jacobian(self, configuration):
        model = configuration.model
        data = configuration.data
        q = configuration.q
        fid = model.getFrameId(self._frame_name)
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)
        J6 = pin.computeFrameJacobian(model, data, q, fid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Jv = J6[:3, :]
        Jw = J6[3:, :]
        R = configuration.get_transform_frame_to_world(self._frame_name).rotation
        JR = np.zeros((9, Jw.shape[1]), dtype=np.float64)
        for i in range(Jw.shape[1]):
            wx, wy, wz = Jw[:, i]
            w_hat = np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])
            JR[:, i] = (-w_hat @ R).reshape(9)
        return np.vstack([-Jv, JR])

    def __repr__(self):
        return f"FrameFrobeniusTask(frame={self._frame_name})"


class RPYHRegularizationTask(Task):
    """Keeps torso orientation (yaw/pitch/roll) and pelvis height close to a
    reference -- otherwise the head/wrist targets alone underdetermine them."""

    def __init__(self, model: pin.Model, data: pin.Data,
                 torso_frame: str = "torso_link", pelvis_frame: str = "pelvis",
                 w_height: float = 5e-2, w_rpy: float = 5e-2,
                 lm_damping: float = 1e-3, gain: float = 1.0,
                 jname_pz=None, jname_yaw=None, jname_pitch=None, jname_roll=None):
        super().__init__(
            cost=np.array([np.sqrt(w_height), np.sqrt(w_rpy), np.sqrt(w_rpy), np.sqrt(w_rpy)], dtype=np.float64),
            gain=gain, lm_damping=lm_damping,
        )
        self._model = model
        self._data = data
        self._fid_torso = model.getFrameId(torso_frame)
        self._fid_pelvis = model.getFrameId(pelvis_frame)
        self._h_ref = 0.75
        self._rpy_ref = np.zeros(3, dtype=np.float64)

        def _v_idx(jname):
            try:
                jid = model.getJointId(jname)
                return model.joints[jid].idx_v if jid > 0 else None
            except Exception:
                return None

        self._iv_pz = _v_idx(jname_pz)
        self._iv_yaw = _v_idx(jname_yaw)
        self._iv_pitch = _v_idx(jname_pitch)
        self._iv_roll = _v_idx(jname_roll)

    def set_reference(self, h_ref: float, rpy_ref: np.ndarray):
        self._h_ref = float(h_ref)
        self._rpy_ref = np.asarray(rpy_ref, dtype=np.float64).reshape(3)

    def compute_error(self, configuration):
        oMt = configuration.get_transform_frame_to_world(self._model.frames[self._fid_torso].name)
        oMp = configuration.get_transform_frame_to_world(self._model.frames[self._fid_pelvis].name)
        R = oMt.rotation
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arcsin(-R[2, 0])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        e_h = self._h_ref - oMp.translation[2]
        e_rpy = self._rpy_ref - np.array([yaw, pitch, roll])
        return np.hstack([e_h, e_rpy]).astype(np.float64)

    def compute_jacobian(self, configuration):
        J = np.zeros((4, configuration.model.nv), dtype=np.float64)
        if self._iv_pz is not None:
            J[0, self._iv_pz] = 1.0
        if self._iv_yaw is not None:
            J[1, self._iv_yaw] = 1.0
        if self._iv_pitch is not None:
            J[2, self._iv_pitch] = 1.0
        if self._iv_roll is not None:
            J[3, self._iv_roll] = 1.0
        return J

    def __repr__(self):
        return f"RPYHRegularizationTask(h_ref={self._h_ref:.3f})"


class ArmPostureTask(Task):
    """Regularizes the controlled (arm/wrist) joints toward a reference
    posture, so the QP doesn't pick an arbitrary null-space solution."""

    def __init__(self, model: pin.Model, data: pin.Data,
                 ctrl_q_ids: np.ndarray, ctrl_v_ids: np.ndarray, w_vec: np.ndarray):
        super().__init__(cost=np.sqrt(w_vec).astype(np.float64), gain=1.0, lm_damping=1e-3)
        self._model = model
        self._data = data
        self._q_ids = np.asarray(ctrl_q_ids, dtype=int)
        self._v_ids = np.asarray(ctrl_v_ids, dtype=int)
        self._q_ref = None

    def set_reference(self, q_ref: np.ndarray):
        self._q_ref = np.asarray(q_ref, dtype=np.float64).copy()

    def compute_error(self, configuration):
        q = configuration.q
        if self._q_ref is None:
            self._q_ref = q.copy()
        return (q[self._q_ids] - self._q_ref[self._q_ids]).astype(np.float64)

    def compute_jacobian(self, configuration):
        nv = configuration.model.nv
        J = np.zeros((len(self._q_ids), nv), dtype=np.float64)
        for i, vid in enumerate(self._v_ids):
            J[i, vid] = 1.0
        return J

    def __repr__(self):
        return f"ArmPostureTask(n_ctrl={len(self._q_ids)})"


class PinkIKSolver:
    def __init__(self, urdf_path: str, frame_names: dict, controlled_joint_names: list):
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

        self.frame_left_name = frame_names["left"]
        self.frame_right_name = frame_names["right"]
        self.frame_head_name = frame_names["head"]
        self._ensure_frame(self.frame_left_name)
        self._ensure_frame(self.frame_right_name)
        self._ensure_frame(self.frame_head_name)
        self._ensure_frame("torso_link")
        self._ensure_frame("pelvis")

        self.jname_ik_pz = "ik_pelvis_z_joint"
        self.jname_ik_yaw = "ik_torso_yaw_joint"
        self.jname_ik_pitch = "ik_torso_pitch_joint"
        self.jname_ik_roll = "ik_torso_roll_joint"

        self.ctrl_jids = [self._joint_id_from_name(n) for n in controlled_joint_names]
        self.ctrl_dof_ids = np.concatenate([
            np.arange(self.model.joints[jid].idx_v, self.model.joints[jid].idx_v + self.model.joints[jid].nv, dtype=int)
            for jid in self.ctrl_jids
        ])
        self.ctrl_q_ids = np.array([self.model.joints[jid].idx_q for jid in self.ctrl_jids], dtype=int)
        self.nd = self.model.nv

        self.pos_w = float(os.environ.get("AMO_IK_POS_W", 1.0))
        self.ori_w = float(os.environ.get("AMO_IK_ORI_W", 0.5))
        self.target_weights = {
            "head": float(os.environ.get("AMO_IK_W_HEAD", 1.0)),
            "left": float(os.environ.get("AMO_IK_W_LEFT", 1.0)),
            "right": float(os.environ.get("AMO_IK_W_RIGHT", 1.0)),
        }
        self.lm_damping = float(os.environ.get("AMO_IK_LM", 1e-3))
        self.qp_damping = float(os.environ.get("AMO_IK_QP_DAMP", 1e-4))
        self.reg_q = float(os.environ.get("AMO_IK_REG_Q", 5e-4))
        self.reg_h = float(os.environ.get("AMO_IK_REG_H", 4e-3))
        self.reg_rpy = float(os.environ.get("AMO_IK_REG_RPY", 2e-2))
        self.dt = 0.02

        self._tasks = {
            "left": FrameFrobeniusTask(self.frame_left_name, self.model, self.data, self.pos_w, self.ori_w),
            "right": FrameFrobeniusTask(self.frame_right_name, self.model, self.data, self.pos_w, self.ori_w),
            "head": FrameFrobeniusTask(self.frame_head_name, self.model, self.data, self.pos_w, self.ori_w),
        }

        try:
            self.model.velocityLimit = np.zeros(self.nd, dtype=np.float64)
        except Exception:
            self.model.velocityLimit = np.zeros(self.nd).astype(np.float64)

        def set_v_limit(joint_name, cap):
            try:
                jid = self.model.getJointId(joint_name)
                if jid > 0:
                    self.model.velocityLimit[self.model.joints[jid].idx_v] = float(cap)
            except Exception:
                pass

        upper_joints = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        ]
        wrist_joints = [
            "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]
        for jn in upper_joints:
            set_v_limit(jn, 8.0)
        for jn in wrist_joints:
            set_v_limit(jn, 12.0)
        set_v_limit(self.jname_ik_pz, float(os.environ.get("AMO_IK_VEL_PZ", 8.0)))
        set_v_limit(self.jname_ik_yaw, float(os.environ.get("AMO_IK_VEL_YAW", 0.3)))
        set_v_limit(self.jname_ik_pitch, float(os.environ.get("AMO_IK_VEL_PITCH", 0.3)))
        set_v_limit(self.jname_ik_roll, float(os.environ.get("AMO_IK_VEL_ROLL", 0.3)))
        self._vel_limit = VelocityLimit(self.model)

        w_upper = self.reg_q
        w_vec = np.array([*([w_upper] * 8), *([w_upper] * 6)], dtype=np.float64)
        self._arm_posture_task = ArmPostureTask(self.model, self.data, self.ctrl_q_ids, self.ctrl_dof_ids, w_vec=w_vec)
        self._rpyh_task = RPYHRegularizationTask(
            self.model, self.data, torso_frame="torso_link", pelvis_frame="pelvis",
            w_height=self.reg_h, w_rpy=self.reg_rpy, lm_damping=self.lm_damping, gain=1.0,
            jname_pz=self.jname_ik_pz, jname_yaw=self.jname_ik_yaw,
            jname_pitch=self.jname_ik_pitch, jname_roll=self.jname_ik_roll,
        )

    def solve(self, q_init: np.ndarray, targets: dict, substeps: int = 8,
              h0: Optional[float] = None, rpy0: Optional[np.ndarray] = None) -> tuple:
        cfg = pink.Configuration(self.model, self.data, q_init.copy())
        active_tasks = []
        base_cost = np.hstack([np.sqrt(self.pos_w) * np.ones(3), np.sqrt(self.ori_w) * np.ones(9)])
        for key, task in self._tasks.items():
            if key not in targets:
                continue
            p_des, R_des = targets[key]
            task.set_target(p_des, R_des)
            task.cost[:] = np.sqrt(float(self.target_weights.get(key, 1.0))) * base_cost
            active_tasks.append(task)

        self._arm_posture_task.set_reference(cfg.q)
        active_tasks.append(self._arm_posture_task)

        if h0 is None or rpy0 is None:
            oMt = cfg.get_transform_frame_to_world("torso_link")
            oMp = cfg.get_transform_frame_to_world("pelvis")
            roll, pitch, yaw = rotation_matrix_to_rpy(oMt.rotation)
            h_ref = float(oMp.translation[2])
            rpy_ref = np.array([yaw, pitch, roll])
        else:
            # rpy0 arrives as [roll, pitch, yaw] (what solve() returns and the
            # caller feeds back); the regularisation task's reference is
            # [yaw, pitch, roll] -- see RPYHRegularizationTask.compute_error and
            # the h0-is-None branch above. Passing it through unpermuted swapped
            # roll and yaw on every warm-started solve.
            rpy0 = np.asarray(rpy0, dtype=np.float64).reshape(3)
            h_ref = float(h0)
            rpy_ref = np.array([rpy0[2], rpy0[1], rpy0[0]], dtype=np.float64)
        self._rpyh_task.set_reference(h_ref, rpy_ref)
        active_tasks.append(self._rpyh_task)

        constraints = [self._vel_limit] if os.environ.get("AMO_IK_NO_LIMITS", "0") != "1" else []
        dt_sub = self.dt / max(1, int(substeps))
        solver_name = os.environ.get("AMO_QP_SOLVER", "quadprog")
        for _ in range(substeps):
            v = pink_solve_ik(cfg, active_tasks, dt=dt_sub, solver=solver_name,
                               damping=self.qp_damping, limits=constraints, safety_break=False)
            cfg.integrate_inplace(v, dt_sub)

        oMp = cfg.get_transform_frame_to_world("pelvis")
        oMt = cfg.get_transform_frame_to_world("torso_link")
        roll, pitch, yaw = rotation_matrix_to_rpy(oMt.rotation)
        h_new = float(oMp.translation[2])
        rpy_new = np.array([roll, pitch, yaw])
        return cfg.q.copy(), h_new, rpy_new

    def assemble_q(self, joint_pos: np.ndarray, base_xyz=None, base_quat_wxyz=None) -> np.ndarray:
        """Assemble a full URDF-order q from a 29-dim joint_pos (composite.py's
        _JOINT_NAMES order == G1_JOINT_NAME_MAP above), pelvis xyz, and pelvis
        quaternion (w,x,y,z)."""
        q = np.zeros(self.model.nq, dtype=np.float64)
        name_to_pos = {}
        for idx, joint_name in G1_JOINT_NAME_MAP.items():
            name_to_pos[joint_name] = float(joint_pos[idx])

        for jname, val in name_to_pos.items():
            jid = self.model.getJointId(jname)
            if jid <= 0 or jid >= len(self.model.joints):
                continue
            j = self.model.joints[jid]
            if j.nq == 1:
                q[j.idx_q] = val

        try:
            if base_xyz is not None:
                jid_h = self.model.getJointId(self.jname_ik_pz)
                if 0 < jid_h < len(self.model.joints):
                    iq = int(self.model.joints[jid_h].idx_q)
                    if 0 <= iq < self.model.nq:
                        q[iq] = float(base_xyz[2])
            if base_quat_wxyz is not None:
                roll, pitch, yaw = quat_to_euler(np.asarray(base_quat_wxyz, dtype=np.float64))
                for jn, val in [
                    (self.jname_ik_yaw, yaw), (self.jname_ik_pitch, pitch), (self.jname_ik_roll, roll),
                ]:
                    jid = self.model.getJointId(jn)
                    if 0 < jid < len(self.model.joints):
                        iq = int(self.model.joints[jid].idx_q)
                        if 0 <= iq < self.model.nq:
                            q[iq] = val
        except Exception:
            pass
        return q

    def _ensure_frame(self, name: str) -> int:
        fid = self.model.getFrameId(name)
        if fid == len(self.model.frames):
            raise RuntimeError(f"URDF frame not found: {name}")
        return fid

    def _joint_id_from_name(self, jname: str) -> int:
        jid = self.model.getJointId(jname)
        if jid == 0:
            raise RuntimeError(f"URDF joint not found: {jname}")
        return jid
