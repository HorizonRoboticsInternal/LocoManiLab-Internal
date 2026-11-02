"""SONIC controller extension for the ``teleop`` encoder mode (mode_id=1).

SONIC has three encoder modes (see
``gear_sonic_deploy/policy/release/observation_config.yaml``):

  mode_id=0 ("g1")     — full body motion tracking (used by wbt_controller_sonic
                          as shipped, fed by the kinematic planner output)
  mode_id=1 ("teleop") — VR 3-point upper-body + planner lower-body
                          (this file)
  mode_id=2 ("smpl")   — full SMPL joint tracking

Required encoder observations for mode_id=1:
  - encoder_mode_4                                 (4D)
  - motion_joint_positions_lowerbody_10frame_step5 (120D = 10 frames × 12 leg joints)
  - motion_joint_velocities_lowerbody_10frame_step5 (120D)
  - vr_3point_local_target                         (9D  = 3 points × xyz)
  - vr_3point_local_orn_target                     (12D = 3 points × quat wxyz)
  - motion_anchor_orientation                      (6D)

This wrapper extends :class:`WBTControllerSonic` and adds:
  - ``set_vr_3point(target_9, orn_12)``    new input channel for PICO data
  - ``set_lowerbody_control_var(...)``     planner lower-body data
  - overridden ``_build_encoder_obs()`` that fills the teleop slots only
"""

from __future__ import annotations

import numpy as np

from .wbt_controller_sonic import (
    HISTORY_SIZE,
    WBTControllerSonic,
)


# G1 has 12 leg joints (left hip×3, knee, ankle×2 + right hip×3, knee, ankle×2).
# In *MuJoCo* body-tree order these are indices 0..11.
N_LEG_JOINTS = 12

# IsaacLab-order indices of the 12 MuJoCo leg joints (in MuJoCo order).
#
# Derivation: for each MuJoCo leg index, look up the joint name in our local
# ``mujoco_joint_names`` list, then find its position in
# ``isaaclab_joint_names``.  Both lists are in ``wbt_vendored.utils.bydmimic_utils``.
#
#     MuJoCo  0 (L hip pitch)   -> IsaacLab  0
#     MuJoCo  1 (L hip roll)    -> IsaacLab  3
#     MuJoCo  2 (L hip yaw)     -> IsaacLab  6
#     MuJoCo  3 (L knee)        -> IsaacLab  9
#     MuJoCo  4 (L ankle pitch) -> IsaacLab 13
#     MuJoCo  5 (L ankle roll)  -> IsaacLab 17
#     MuJoCo  6 (R hip pitch)   -> IsaacLab  1
#     MuJoCo  7 (R hip roll)    -> IsaacLab  4
#     MuJoCo  8 (R hip yaw)     -> IsaacLab  7
#     MuJoCo  9 (R knee)        -> IsaacLab 10
#     MuJoCo 10 (R ankle pitch) -> IsaacLab 14
#     MuJoCo 11 (R ankle roll)  -> IsaacLab 18
#
# Matches SONIC C++ deploy's ``lower_body_joint_mujoco_order_in_isaaclab_index``
# in gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp:92.
#
# The previous (broken) value [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15] picked
# up shoulder/waist joints as "leg joints" — VR_3PT walking was unusable
# because the encoder was reading mis-labelled inputs.
_LOWER_BODY_ISAAC_INDICES_IN_MUJOCO_ORDER = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18], dtype=np.int64
)


class WBTControllerSonicTeleop(WBTControllerSonic):
    """SONIC with encoder mode_id=1 (teleop).

    Usage::

        ctrl = WBTControllerSonicTeleop(encoder_path=..., decoder_path=...)
        # Each policy tick:
        ctrl.set_lowerbody_control_var(joint_pos_29_isaac, joint_vel_29_isaac,
                                       motion_anchor_orientation_6d)
        ctrl.set_vr_3point(vr_3point_local_target_9, vr_3point_local_orn_target_12)
        target_q, kps, kds = ctrl.step(robot_state)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lower-body planner reference (10 frames × 12 leg joints in IsaacLab order).
        # Filled by set_lowerbody_control_var() — defaults to SONIC default pose.
        default_lower = self.default_dof_pos_isaac[
            _LOWER_BODY_ISAAC_INDICES_IN_MUJOCO_ORDER
        ]
        self.motion_joint_pos_lowerbody_buf = np.tile(
            default_lower, (HISTORY_SIZE, 1)
        ).astype(np.float32)
        self.motion_joint_vel_lowerbody_buf = np.zeros(
            (HISTORY_SIZE, N_LEG_JOINTS), dtype=np.float32
        )

        # VR 3-point inputs — default to zero (SONIC will treat as
        # "no teleop input", upper-body output will be close to default pose).
        self._vr_3point_target = np.zeros(9, dtype=np.float32)
        self._vr_3point_orn_target = np.zeros(12, dtype=np.float32)
        # Initialise quaternions to identity (wxyz = [1, 0, 0, 0]) for each
        # of the 3 points so the encoder doesn't see all-zero quats.
        for i in range(3):
            self._vr_3point_orn_target[i * 4] = 1.0

    def reset(self):
        super().reset()
        default_lower = self.default_dof_pos_isaac[
            _LOWER_BODY_ISAAC_INDICES_IN_MUJOCO_ORDER
        ]
        self.motion_joint_pos_lowerbody_buf[:] = default_lower
        self.motion_joint_vel_lowerbody_buf.fill(0)
        self._vr_3point_target.fill(0)
        self._vr_3point_orn_target.fill(0)
        for i in range(3):
            self._vr_3point_orn_target[i * 4] = 1.0

    # ------------------------------------------------------------------ #
    # New input channels for teleop mode
    # ------------------------------------------------------------------ #
    def set_lowerbody_control_var(self, data: dict):
        """Set the lower-body planner reference.

        Args:
            data: dict with optional keys (IsaacLab order, full 29-D arrays):
                - ``joint_pos_29``: (29,) — slice [LOWER_BODY_INDICES] used
                - ``joint_vel_29``: (29,)
                - ``motion_anchor_orientation``: (6,)
        """
        if "joint_pos_29" in data:
            self.motion_joint_pos_lowerbody_buf[:] = np.asarray(
                data["joint_pos_29"], dtype=np.float32
            )[_LOWER_BODY_ISAAC_INDICES_IN_MUJOCO_ORDER]
        if "joint_vel_29" in data:
            self.motion_joint_vel_lowerbody_buf[:] = np.asarray(
                data["joint_vel_29"], dtype=np.float32
            )[_LOWER_BODY_ISAAC_INDICES_IN_MUJOCO_ORDER]
        if "motion_anchor_orientation" in data:
            self.motion_anchor_ori_buf[:] = np.asarray(
                data["motion_anchor_orientation"], dtype=np.float32
            )

    def set_vr_3point(
        self,
        vr_3point_local_target_9: np.ndarray,
        vr_3point_local_orn_target_12: np.ndarray,
    ) -> None:
        """Push the latest calibrated 3-point VR pose from PICO."""
        self._vr_3point_target = np.asarray(
            vr_3point_local_target_9, dtype=np.float32
        ).copy()
        self._vr_3point_orn_target = np.asarray(
            vr_3point_local_orn_target_12, dtype=np.float32
        ).copy()

    # ------------------------------------------------------------------ #
    # Override encoder observation builder for mode_id=1
    # ------------------------------------------------------------------ #
    def _build_encoder_obs(self) -> np.ndarray:
        layout = self.encoder_layout
        enc_obs = np.zeros(self.encoder_dim, dtype=np.float32)

        # encoder_mode_4: [mode_id, 0, 0, 0] — teleop = 1
        enc_obs[0] = 1.0

        # motion_joint_positions_lowerbody_10frame_step5 (120D)
        start, end = layout["motion_joint_positions_lowerbody_10frame_step5"]
        enc_obs[start:end] = self.motion_joint_pos_lowerbody_buf.flatten()

        # motion_joint_velocities_lowerbody_10frame_step5 (120D)
        start, end = layout["motion_joint_velocities_lowerbody_10frame_step5"]
        enc_obs[start:end] = self.motion_joint_vel_lowerbody_buf.flatten()

        # vr_3point_local_target (9D)
        start, end = layout["vr_3point_local_target"]
        enc_obs[start:end] = self._vr_3point_target

        # vr_3point_local_orn_target (12D)
        start, end = layout["vr_3point_local_orn_target"]
        enc_obs[start:end] = self._vr_3point_orn_target

        # motion_anchor_orientation (6D) — current frame
        start, end = layout["motion_anchor_orientation"]
        enc_obs[start:end] = self.motion_anchor_ori_buf[0]

        return enc_obs
