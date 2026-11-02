"""
SONIC Controller (WBT SONIC)
NVIDIA GR00T-WholeBodyControl SONIC模型部署控制器

SONIC采用Encoder-Decoder架构:
- Encoder: 输入1762D (运动参考), 输出64D token
- Decoder: 输入994D (token + 机器人历史状态), 输出29D action

g1 mode (mode_id=0): 全身29关节动作跟踪

模型路径:
- Encoder: ~/sjy/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_encoder.onnx
- Decoder: ~/sjy/GR00T-WholeBodyControl/gear_sonic_deploy/policy/release/model_decoder.onnx
"""

import numpy as np
import onnxruntime as ort

from .base_controller import BaseController
from ..utils.trt_utils import create_ort_session
from ..utils.bydmimic_utils import (
    get_gravity_orientation,
    mujoco_to_isaaclab_reindex,
    isaaclab_to_mujoco_reindex,
)

# SONIC default angles (from policy_parameters.hpp) - MuJoCo order
SONIC_DEFAULT_ANGLES_MUJOCO = np.array([
    -0.312,  # left_hip_pitch_joint
    0.0,     # left_hip_roll_joint
    0.0,     # left_hip_yaw_joint
    0.669,   # left_knee_joint
    -0.363,  # left_ankle_pitch_joint
    0.0,     # left_ankle_roll_joint
    -0.312,  # right_hip_pitch_joint
    0.0,     # right_hip_roll_joint
    0.0,     # right_hip_yaw_joint
    0.669,   # right_knee_joint
    -0.363,  # right_ankle_pitch_joint
    0.0,     # right_ankle_roll_joint
    0.0,     # waist_yaw_joint
    0.0,     # waist_roll_joint
    0.0,     # waist_pitch_joint
    0.2,     # left_shoulder_pitch_joint
    0.2,     # left_shoulder_roll_joint
    0.0,     # left_shoulder_yaw_joint
    0.6,     # left_elbow_joint
    0.0,     # left_wrist_roll_joint
    0.0,     # left_wrist_pitch_joint
    0.0,     # left_wrist_yaw_joint
    0.2,     # right_shoulder_pitch_joint
    -0.2,    # right_shoulder_roll_joint
    0.0,     # right_shoulder_yaw_joint
    0.6,     # right_elbow_joint
    0.0,     # right_wrist_roll_joint
    0.0,     # right_wrist_pitch_joint
    0.0,     # right_wrist_yaw_joint
], dtype=np.float32)


# ---------------------------------------------------------------------------
# SONIC-specific PD parameters (from policy_parameters.hpp)
# ---------------------------------------------------------------------------
_NATURAL_FREQ = 10 * 2 * np.pi
_DAMPING_RATIO = 2.0

_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425

_S_5020 = _ARMATURE_5020 * _NATURAL_FREQ ** 2
_S_7520_14 = _ARMATURE_7520_14 * _NATURAL_FREQ ** 2
_S_7520_22 = _ARMATURE_7520_22 * _NATURAL_FREQ ** 2
_S_4010 = _ARMATURE_4010 * _NATURAL_FREQ ** 2

_D_5020 = 2 * _DAMPING_RATIO * _ARMATURE_5020 * _NATURAL_FREQ
_D_7520_14 = 2 * _DAMPING_RATIO * _ARMATURE_7520_14 * _NATURAL_FREQ
_D_7520_22 = 2 * _DAMPING_RATIO * _ARMATURE_7520_22 * _NATURAL_FREQ
_D_4010 = 2 * _DAMPING_RATIO * _ARMATURE_4010 * _NATURAL_FREQ

_E_5020 = 25.0
_E_7520_14 = 88.0
_E_7520_22 = 139.0
_E_4010 = 5.0

# SONIC kps (MuJoCo joint order, same as SONIC IsaacLab)
SONIC_KPS = np.array([
    _S_7520_22,   _S_7520_22,   _S_7520_14,   _S_7520_22,   2*_S_5020,    2*_S_5020,    # left leg
    _S_7520_22,   _S_7520_22,   _S_7520_14,   _S_7520_22,   2*_S_5020,    2*_S_5020,    # right leg
    _S_7520_14,   2*_S_5020,    2*_S_5020,                                                # waist
    _S_5020,      _S_5020,      _S_5020,      _S_5020,      _S_5020,      _S_4010,  _S_4010,  # left arm
    _S_5020,      _S_5020,      _S_5020,      _S_5020,      _S_5020,      _S_4010,  _S_4010,  # right arm
], dtype=np.float32)

SONIC_KDS = np.array([
    _D_7520_22,   _D_7520_22,   _D_7520_14,   _D_7520_22,   2*_D_5020,    2*_D_5020,
    _D_7520_22,   _D_7520_22,   _D_7520_14,   _D_7520_22,   2*_D_5020,    2*_D_5020,
    _D_7520_14,   2*_D_5020,    2*_D_5020,
    _D_5020,      _D_5020,      _D_5020,      _D_5020,      _D_5020,      _D_4010,  _D_4010,
    _D_5020,      _D_5020,      _D_5020,      _D_5020,      _D_5020,      _D_4010,  _D_4010,
], dtype=np.float32)

SONIC_ACTION_SCALE = np.array([
    0.25*_E_7520_22/_S_7520_22, 0.25*_E_7520_22/_S_7520_22, 0.25*_E_7520_14/_S_7520_14,
    0.25*_E_7520_22/_S_7520_22, 0.25*_E_5020/_S_5020,       0.25*_E_5020/_S_5020,
    0.25*_E_7520_22/_S_7520_22, 0.25*_E_7520_22/_S_7520_22, 0.25*_E_7520_14/_S_7520_14,
    0.25*_E_7520_22/_S_7520_22, 0.25*_E_5020/_S_5020,       0.25*_E_5020/_S_5020,
    0.25*_E_7520_14/_S_7520_14, 0.25*_E_5020/_S_5020,       0.25*_E_5020/_S_5020,
    0.25*_E_5020/_S_5020,       0.25*_E_5020/_S_5020,       0.25*_E_5020/_S_5020,
    0.25*_E_5020/_S_5020,       0.25*_E_5020/_S_5020,       0.25*_E_4010/_S_4010,  0.25*_E_4010/_S_4010,
    0.25*_E_5020/_S_5020,       0.25*_E_5020/_S_5020,       0.25*_E_5020/_S_5020,
    0.25*_E_5020/_S_5020,       0.25*_E_5020/_S_5020,       0.25*_E_4010/_S_4010,  0.25*_E_4010/_S_4010,
], dtype=np.float32)

# SONIC parameters in IsaacLab order (converted from MuJoCo)
SONIC_DEFAULT_ANGLES = SONIC_DEFAULT_ANGLES_MUJOCO[mujoco_to_isaaclab_reindex]
SONIC_KPS_ISAAC = SONIC_KPS[mujoco_to_isaaclab_reindex]
SONIC_KDS_ISAAC = SONIC_KDS[mujoco_to_isaaclab_reindex]
SONIC_ACTION_SCALE_ISAAC = SONIC_ACTION_SCALE[mujoco_to_isaaclab_reindex]


# ---------------------------------------------------------------------------
# Encoder observation layout — selected per-instance by the loaded encoder's
# input dimension (see WBTControllerSonic.__init__). The flat slot order matches
# the ``encoder_observations`` listing order in the SONIC deploy
# ``observation_config*.yaml`` (offsets accumulate by dimension), so each named
# slot below is exactly the [start, end) span the ONNX expects.
#
# Three checkpoint variants ship today and share the SAME decoder I/O (994->29),
# so only the encoder layout differs (see sonic_ckpt.py):
#   * release      (old ckpt) — 1762D, smpl block is 10-frame, has root_z slots.
#   * low_latency  (new ckpt) — 1247D, smpl block is 4-frame, root_z dropped.
#   * v1_1                    — 1751D, smpl block is 10-frame, root_z dropped,
#                               anchors heading-normalized.
#
# The low-latency layout reuses the release slot KEY NAMES (even though they are
# semantically step1/4-frame there) so controller code can look slots up by the
# same key regardless of variant; only the spans (and the smpl frame count)
# differ.
# ---------------------------------------------------------------------------
ENCODER_OBS_LAYOUT_RELEASE = {
    'encoder_mode_4': (0, 4),
    'motion_joint_positions_10frame_step5': (4, 294),
    'motion_joint_velocities_10frame_step5': (294, 584),
    'motion_root_z_position_10frame_step5': (584, 594),
    'motion_root_z_position': (594, 595),
    'motion_anchor_orientation': (595, 601),
    'motion_anchor_orientation_10frame_step5': (601, 661),
    'motion_joint_positions_lowerbody_10frame_step5': (661, 781),
    'motion_joint_velocities_lowerbody_10frame_step5': (781, 901),
    'vr_3point_local_target': (901, 910),
    'vr_3point_local_orn_target': (910, 922),
    'smpl_joints_10frame_step1': (922, 1642),
    'smpl_anchor_orientation_10frame_step1': (1642, 1702),
    'motion_joint_positions_wrists_10frame_step1': (1702, 1762),
}
ENCODER_DIM_RELEASE = 1762

# Low-latency (new) variant — derived from observation_config_low_latency.yaml.
# Differences vs release: root_z slots removed; the three smpl slots are
# 4-frame instead of 10-frame (288/24/24 vs 720/60/60).
ENCODER_OBS_LAYOUT_LOW_LATENCY = {
    'encoder_mode_4': (0, 4),
    'motion_joint_positions_10frame_step5': (4, 294),
    'motion_joint_velocities_10frame_step5': (294, 584),
    'motion_anchor_orientation_10frame_step5': (584, 644),
    'motion_anchor_orientation': (644, 650),
    'motion_joint_positions_lowerbody_10frame_step5': (650, 770),
    'motion_joint_velocities_lowerbody_10frame_step5': (770, 890),
    'vr_3point_local_target': (890, 899),
    'vr_3point_local_orn_target': (899, 911),
    'smpl_joints_10frame_step1': (911, 1199),
    'smpl_anchor_orientation_10frame_step1': (1199, 1223),
    'motion_joint_positions_wrists_10frame_step1': (1223, 1247),
}
ENCODER_DIM_LOW_LATENCY = 1247

# v1.1 variant — derived from sonic_v1_1/observation_config.yaml. It is the
# release layout minus the two root_z slots, with the two anchor slots in
# low_latency's order (multi-frame first), and the three smpl slots back to
# 10-frame. Same slot KEY NAMES again, so controller code is variant-agnostic.
#
# ⚠️ The two anchor slots are heading-normalized in v1.1 (the reference root
# expressed in the robot's YAW-ONLY frame, not its full body frame). This
# controller does not compute them -- it receives them through
# ``data['motion_anchor_orientation']`` (see update_obs), and the caller that
# does compute them is tele_humanoid_benchmark's robocasa/g1_host/wbt.py, which
# supplies body-frame ones. So __init__ REFUSES a v1.1 encoder outright; this
# table exists to make that refusal a specific message instead of a bare
# "unsupported dim", and to be ready if wbt.py ever learns the heading frame.
# The bridge runtime implements it properly today -- see
# g1_bridge/sonic_utils/encoder_obs.py's sonic_g1_v1_1_encoder_input.
#
# Slot KEY NAMES stay the release/low_latency spellings (as the other two
# layouts do) so _build_encoder_obs can look slots up variant-agnostically.
ENCODER_OBS_LAYOUT_V1_1 = {
    'encoder_mode_4': (0, 4),
    'motion_joint_positions_10frame_step5': (4, 294),
    'motion_joint_velocities_10frame_step5': (294, 584),
    'motion_anchor_orientation_10frame_step5': (584, 644),
    'motion_anchor_orientation': (644, 650),
    'motion_joint_positions_lowerbody_10frame_step5': (650, 770),
    'motion_joint_velocities_lowerbody_10frame_step5': (770, 890),
    'vr_3point_local_target': (890, 899),
    'vr_3point_local_orn_target': (899, 911),
    'smpl_joints_10frame_step1': (911, 1631),
    'smpl_anchor_orientation_10frame_step1': (1631, 1691),
    'motion_joint_positions_wrists_10frame_step1': (1691, 1751),
}
ENCODER_DIM_V1_1 = 1751

# encoder input dim -> (layout dict, smpl future-frame count).
ENCODER_LAYOUTS = {
    ENCODER_DIM_RELEASE: ENCODER_OBS_LAYOUT_RELEASE,
    ENCODER_DIM_LOW_LATENCY: ENCODER_OBS_LAYOUT_LOW_LATENCY,
    ENCODER_DIM_V1_1: ENCODER_OBS_LAYOUT_V1_1,
}

# Backwards-compatible module aliases (the release variant is the historical
# default). Controllers use the per-instance ``self.encoder_layout`` / numbers.
ENCODER_OBS_LAYOUT = ENCODER_OBS_LAYOUT_RELEASE
ENCODER_DIM = ENCODER_DIM_RELEASE

# Decoder observation layout (994D) — identical across ckpt variants.
DECODER_OBS_LAYOUT = {
    'token_state': (0, 64),
    'his_base_angular_velocity_10frame_step1': (64, 94),
    'his_body_joint_positions_10frame_step1': (94, 384),
    'his_body_joint_velocities_10frame_step1': (384, 674),
    'his_last_actions_10frame_step1': (674, 964),
    'his_gravity_dir_10frame_step1': (964, 994),
}

DECODER_DIM = 994
TOKEN_DIM = 64
HISTORY_SIZE = 10


class WBTControllerSonic(BaseController):
    """SONIC (GR00T-WholeBodyControl) 控制器 - g1模式 (mode_id=0)

    Encoder输入 (g1 mode):
    - motion_joint_positions_10frame_step5 (290D)
    - motion_joint_velocities_10frame_step5 (290D)
    - motion_anchor_orientation_10frame_step5 (60D)

    Decoder输入:
    - token_state (64D) from encoder
    - his_base_angular_velocity_10frame_step1 (30D)
    - his_body_joint_positions_10frame_step1 (290D)
    - his_body_joint_velocities_10frame_step1 (290D)
    - his_last_actions_10frame_step1 (290D)
    - his_gravity_dir_10frame_step1 (30D)
    """

    def __init__(self, encoder_path: str, decoder_path: str, use_trt: bool = False):
        super().__init__()

        # Load ONNX models
        self.encoder_session = create_ort_session(encoder_path, use_trt=use_trt)
        self.decoder_session = create_ort_session(decoder_path, use_trt=use_trt)

        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.encoder_output_name = self.encoder_session.get_outputs()[0].name
        self.decoder_input_name = self.decoder_session.get_inputs()[0].name
        self.decoder_output_name = self.decoder_session.get_outputs()[0].name

        # Pick the encoder obs layout that matches the loaded ONNX so old
        # (release, 1762D) and new (low-latency, 1247D) checkpoints are both
        # supported without a code change at the call site. The decoder is
        # identical (994->29) across variants, so only the encoder is keyed.
        enc_in_dim = self.encoder_session.get_inputs()[0].shape[-1]
        self.encoder_dim = int(enc_in_dim) if isinstance(enc_in_dim, int) else ENCODER_DIM
        self.encoder_layout = ENCODER_LAYOUTS.get(self.encoder_dim)
        if self.encoder_layout is None:
            raise ValueError(
                f"Unsupported SONIC encoder input dim {self.encoder_dim} "
                f"(model: {encoder_path}); known layouts: {sorted(ENCODER_LAYOUTS)}"
            )
        # v1.1 wants heading-normalized anchors, and this runtime does not
        # compute anchors at all -- set_control_var() takes them from its
        # caller, and the only caller (tele_humanoid_benchmark's
        # robocasa/g1_host/wbt.py) supplies body-frame ones. Feeding those to a
        # heading-trained encoder is undetectable at runtime: right dimension,
        # wrong frame, robot quietly tracks the wrong thing. Refuse instead.
        # Registering the layout above is only so the failure is THIS message
        # rather than a confusing "unsupported dim".
        if self.encoder_dim == ENCODER_DIM_V1_1:
            raise ValueError(
                f"{encoder_path} is the sonic_v1_1 encoder, which expects "
                "heading-normalized motion anchors; this WBT runtime is fed "
                "body-frame ones and has no way to tell. Use the bridge "
                "runtime instead (--controller-runtime bridge with "
                "--policy-config checkpoints/sonic/v1_1/g1/policy.yaml), which "
                "implements the heading normalization."
            )

        # PD parameters (IsaacLab order for internal computation)
        self.kps_isaac = SONIC_KPS_ISAAC.copy()
        self.kds_isaac = SONIC_KDS_ISAAC.copy()
        self.action_scale_isaac = SONIC_ACTION_SCALE_ISAAC.copy()
        self.default_dof_pos_isaac = SONIC_DEFAULT_ANGLES.copy()  # Already in IsaacLab order

        # MuJoCo order versions for output conversion
        self.kps = SONIC_KPS.copy()
        self.kds = SONIC_KDS.copy()
        self.action_scale = SONIC_ACTION_SCALE.copy()
        self.default_dof_pos = SONIC_DEFAULT_ANGLES_MUJOCO.copy()

        # Token state from encoder
        self.token_state = np.zeros(TOKEN_DIM, dtype=np.float32)

        # Decoder history buffers (10 frames, step 1)
        self.his_base_ang_vel = np.zeros((HISTORY_SIZE, 3), dtype=np.float32)
        self.his_body_joint_pos = np.zeros((HISTORY_SIZE, 29), dtype=np.float32)
        self.his_body_joint_vel = np.zeros((HISTORY_SIZE, 29), dtype=np.float32)
        self.his_last_actions = np.zeros((HISTORY_SIZE, 29), dtype=np.float32)
        self.his_gravity_dir = np.zeros((HISTORY_SIZE, 3), dtype=np.float32)
        self.his_gravity_dir[:] = [0, 0, -1]  # Default gravity

        # Encoder motion buffers (10 frames for encoder input)
        # For streaming mocap (no future data), all 10 slots hold the current frame,
        # matching the C++ play=false behavior.
        self.motion_joint_pos_buf = np.zeros((HISTORY_SIZE, 29), dtype=np.float32)
        self.motion_joint_vel_buf = np.zeros((HISTORY_SIZE, 29), dtype=np.float32)
        self.motion_anchor_ori_buf = np.zeros((HISTORY_SIZE, 6), dtype=np.float32)

        # Root Z position from reference motion (pelvis height, ~0.75m standing)
        self.motion_root_z = 0.0

    def reset(self):
        """Reset controller state"""
        self.token_state.fill(0)

        # Reset history buffers
        self.his_base_ang_vel.fill(0)
        self.his_body_joint_pos.fill(0)
        self.his_body_joint_vel.fill(0)
        self.his_last_actions.fill(0)
        self.his_gravity_dir[:] = [0, 0, -1]

        # Reset motion buffers
        self.motion_joint_pos_buf.fill(0)
        self.motion_joint_vel_buf.fill(0)
        self.motion_anchor_ori_buf.fill(0)
        self.motion_root_z = 0.0

    def set_control_var(self, data: dict):
        """设置控制变量 (encoder输入)

        Args:
            data: dict with keys:
                - joint_pos_29: (29,) 全身关节位置 (IsaacLab order, raw angles)
                - joint_vel_29: (29,) 全身关节速度 (IsaacLab order)
                - motion_anchor_orientation: (6,) anchor朝向 (rotation matrix前2列flatten)
                - root_z: float, 参考运动根节点Z坐标 (骨盆高度)

        For streaming mocap (no future frames available), all 10 encoder slots
        are filled with the current frame. This matches the C++ play=false behavior.
        """
        if 'joint_pos_29' in data:
            self.motion_joint_pos_buf[:] = data['joint_pos_29']
        if 'joint_vel_29' in data:
            self.motion_joint_vel_buf[:] = data['joint_vel_29']
        if 'motion_anchor_orientation' in data:
            self.motion_anchor_ori_buf[:] = data['motion_anchor_orientation']
        if 'root_z' in data:
            self.motion_root_z = float(data['root_z'])

    def _build_encoder_obs(self) -> np.ndarray:
        """构建encoder观测向量 - g1 mode (mode_id=0).

        Length is ``self.encoder_dim`` (1762 release / 1247 low-latency) and
        slots come from ``self.encoder_layout`` so both ckpt variants work.
        """
        layout = self.encoder_layout
        enc_obs = np.zeros(self.encoder_dim, dtype=np.float32)

        # encoder_mode_4: [mode_id, 0, 0, 0], g1 mode = 0
        # This is NOT one-hot encoding! It's mode_id (0 for g1) + zero padding
        enc_obs[0] = 0.0  # mode_id = 0 for g1 mode

        # motion_joint_positions_10frame_step5 (290D = 10×29)
        start, end = layout['motion_joint_positions_10frame_step5']
        enc_obs[start:end] = self.motion_joint_pos_buf.flatten()

        # motion_joint_velocities_10frame_step5 (290D)
        start, end = layout['motion_joint_velocities_10frame_step5']
        enc_obs[start:end] = self.motion_joint_vel_buf.flatten()

        # motion_anchor_orientation_10frame_step5 (60D = 10×6)
        start, end = layout['motion_anchor_orientation_10frame_step5']
        enc_obs[start:end] = self.motion_anchor_ori_buf.flatten()

        # motion_root_z_position slots — release only (dropped in low-latency).
        if 'motion_root_z_position_10frame_step5' in layout:
            start, end = layout['motion_root_z_position_10frame_step5']
            enc_obs[start:end] = self.motion_root_z
        if 'motion_root_z_position' in layout:
            start, end = layout['motion_root_z_position']
            enc_obs[start] = self.motion_root_z

        # motion_anchor_orientation (6D) - current frame
        start, end = layout['motion_anchor_orientation']
        enc_obs[start:end] = self.motion_anchor_ori_buf[0]

        return enc_obs

    def _build_decoder_obs(self) -> np.ndarray:
        """构建decoder观测向量 (994D)"""
        dec_obs = np.zeros(DECODER_DIM, dtype=np.float32)

        # token_state (64D)
        start, end = DECODER_OBS_LAYOUT['token_state']
        dec_obs[start:end] = self.token_state

        # his_base_angular_velocity_10frame_step1 (30D)
        start, end = DECODER_OBS_LAYOUT['his_base_angular_velocity_10frame_step1']
        dec_obs[start:end] = self.his_base_ang_vel.flatten()

        # his_body_joint_positions_10frame_step1 (290D)
        start, end = DECODER_OBS_LAYOUT['his_body_joint_positions_10frame_step1']
        dec_obs[start:end] = self.his_body_joint_pos.flatten()

        # his_body_joint_velocities_10frame_step1 (290D)
        start, end = DECODER_OBS_LAYOUT['his_body_joint_velocities_10frame_step1']
        dec_obs[start:end] = self.his_body_joint_vel.flatten()

        # his_last_actions_10frame_step1 (290D)
        start, end = DECODER_OBS_LAYOUT['his_last_actions_10frame_step1']
        dec_obs[start:end] = self.his_last_actions.flatten()

        # his_gravity_dir_10frame_step1 (30D)
        start, end = DECODER_OBS_LAYOUT['his_gravity_dir_10frame_step1']
        dec_obs[start:end] = self.his_gravity_dir.flatten()

        return dec_obs

    def _update_history(self, robot_state: dict):
        """更新历史buffer

        Args:
            robot_state: dict with keys:
                - joint_pos: (29,) mujoco order
                - joint_vel: (29,) mujoco order
                - quaternion: (4,) wxyz
                - gyroscope: (3,) angular velocity
        """
        qj_mujoco = robot_state['joint_pos']      # mujoco order
        dqj_mujoco = robot_state['joint_vel']     # mujoco order
        quat = robot_state['quaternion']   # wxyz
        ang_vel = robot_state['gyroscope']

        # Convert to IsaacLab order for SONIC
        qj_isaac = qj_mujoco[mujoco_to_isaaclab_reindex]
        dqj_isaac = dqj_mujoco[mujoco_to_isaaclab_reindex]

        # Roll history (oldest out, newest in)
        self.his_base_ang_vel = np.roll(self.his_base_ang_vel, -1, axis=0)
        self.his_body_joint_pos = np.roll(self.his_body_joint_pos, -1, axis=0)
        self.his_body_joint_vel = np.roll(self.his_body_joint_vel, -1, axis=0)
        self.his_gravity_dir = np.roll(self.his_gravity_dir, -1, axis=0)

        # Update newest frame
        self.his_base_ang_vel[-1] = ang_vel

        # body_joint_positions: (qj - default_angles) in IsaacLab order
        body_q = qj_isaac - self.default_dof_pos_isaac
        self.his_body_joint_pos[-1] = body_q

        # body_joint_velocities in IsaacLab order
        self.his_body_joint_vel[-1] = dqj_isaac

        # gravity_dir
        gravity_dir = get_gravity_orientation(quat)
        self.his_gravity_dir[-1] = gravity_dir

    def step(self, robot) -> tuple:
        """执行一步控制

        Args:
            robot: 机器人对象或状态字典

        Returns:
            tuple: (target_q, kps, kds)
        """
        # Get robot state
        if isinstance(robot, dict):
            robot_state = robot
        else:
            robot_state = robot.get_state()

        # Update history buffers
        self._update_history(robot_state)

        # Run encoder to get token
        enc_obs = self._build_encoder_obs()
        enc_input = enc_obs.reshape(1, -1).astype(np.float32)
        enc_output = self.encoder_session.run(
            [self.encoder_output_name],
            {self.encoder_input_name: enc_input}
        )
        self.token_state = enc_output[0].squeeze()

        # Run decoder to get action
        dec_obs = self._build_decoder_obs()
        dec_input = dec_obs.reshape(1, -1).astype(np.float32)
        dec_output = self.decoder_session.run(
            [self.decoder_output_name],
            {self.decoder_input_name: dec_input}
        )
        action = dec_output[0].squeeze()

        # Update action history (roll + new action at end)
        self.his_last_actions = np.roll(self.his_last_actions, -1, axis=0)
        self.his_last_actions[-1] = action

        # Convert action to target joint positions (in IsaacLab order)
        target_q_isaac = self.default_dof_pos_isaac + action * self.action_scale_isaac

        # Convert to MuJoCo order for robot output
        target_q = target_q_isaac[isaaclab_to_mujoco_reindex]

        return target_q, self.kps, self.kds

    def decode_with_token(self, robot, token_state) -> tuple:
        """Run the decoder only, with an EXTERNALLY supplied motion token.

        VLA deployment path: the GR00T policy predicts ``motion_token`` (the same
        64-D latent the encoder would normally produce), so here we skip the
        encoder, inject ``token_state``, and run just the decoder to get the
        29-joint target. History buffers are still updated from the live robot
        state so the decoder has its temporal context. Mirrors ``step`` minus the
        encoder.

        Args:
            robot: robot object or state dict (joint_pos/joint_vel/quaternion/gyroscope).
            token_state: (64,) motion token from the VLA policy.
        """
        robot_state = robot if isinstance(robot, dict) else robot.get_state()
        self._update_history(robot_state)

        self.token_state = np.asarray(token_state, dtype=np.float32).reshape(-1)

        dec_obs = self._build_decoder_obs()
        dec_input = dec_obs.reshape(1, -1).astype(np.float32)
        dec_output = self.decoder_session.run(
            [self.decoder_output_name],
            {self.decoder_input_name: dec_input}
        )
        action = dec_output[0].squeeze()

        self.his_last_actions = np.roll(self.his_last_actions, -1, axis=0)
        self.his_last_actions[-1] = action

        target_q_isaac = self.default_dof_pos_isaac + action * self.action_scale_isaac
        target_q = target_q_isaac[isaaclab_to_mujoco_reindex]
        return target_q, self.kps, self.kds
