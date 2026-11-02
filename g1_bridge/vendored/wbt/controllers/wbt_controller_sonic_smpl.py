"""SONIC controller extension for the ``smpl`` encoder mode (mode_id=2).

This is the full-body teleop mode where the policy receives:
  - 24 SMPL joint positions (10 frames × 24 × 3 = 720D)
  - SMPL anchor orientation (10 frames × 6D = 60D)
  - 6 G1 wrist joint positions in IsaacLab order (10 frames × 6 = 60D)

Encoder obs slot layout (from ENCODER_OBS_LAYOUT in wbt_controller_sonic.py):
  - ``encoder_mode_4``                         (4D, [2,0,0,0])
  - ``smpl_joints_10frame_step1``              offset 922-1642   (720D)
  - ``smpl_anchor_orientation_10frame_step1``  offset 1642-1702  (60D)
  - ``motion_joint_positions_wrists_10frame_step1`` offset 1702-1762 (60D)

Data wiring (driven by WholeBodyG1Composite._update_pose_reference):

    ctrl.set_pose_control_var({
        "smpl_joints_local": np.ndarray (24, 3),   # robot frame, root-relative
        "anchor_6d": np.ndarray (6,),               # row-major flat first-two-cols
        "wrist_joints": np.ndarray (6,),            # [L_roll, R_roll, L_pitch,
                                                    #  R_pitch, L_yaw, R_yaw]
    })

The wrist-joint vector is ordered by IsaacLab indices {23, 24, 25, 26, 27, 28}
to match the encoder's expected order — see
``gear_sonic_deploy/.../policy_parameters.hpp:88``
(``wrist_joint_isaaclab_order_in_isaaclab_index = {23, 24, 25, 26, 27, 28}``).
"""

from __future__ import annotations

import numpy as np

from .wbt_controller_sonic import (
    HISTORY_SIZE,
    WBTControllerSonic,
)


N_SMPL_JOINTS = 24
N_WRIST_JOINTS = 6   # L/R × {roll, pitch, yaw}


class WBTControllerSonicSmpl(WBTControllerSonic):
    """SONIC with encoder mode_id=2 (full-body SMPL teleop)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Number of smpl future frames the active encoder expects: 10 for the
        # release ckpt, 4 for low-latency (smpl_joints block / (24 joints × 3)).
        s, e = self.encoder_layout['smpl_joints_10frame_step1']
        self.smpl_frames = (e - s) // (N_SMPL_JOINTS * 3)

        # 10-frame histories — sliced to ``smpl_frames`` at encode time.
        # smpl_joints_local: (HISTORY_SIZE, 24, 3)
        self._smpl_joints_buf = np.zeros(
            (HISTORY_SIZE, N_SMPL_JOINTS, 3), dtype=np.float32
        )
        # smpl_anchor: (HISTORY_SIZE, 6)
        self._smpl_anchor_buf = np.tile(
            np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            (HISTORY_SIZE, 1),
        )
        # wrist joint positions: (HISTORY_SIZE, 6)
        self._wrist_joints_buf = np.zeros(
            (HISTORY_SIZE, N_WRIST_JOINTS), dtype=np.float32
        )

    def reset(self):
        super().reset()
        self._smpl_joints_buf.fill(0)
        self._smpl_anchor_buf[:] = np.array(
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32
        )
        self._wrist_joints_buf.fill(0)

    # ------------------------------------------------------------------ #
    def set_pose_control_var(self, data: dict) -> None:
        """Push the latest full-body POSE reference.

        Args:
            data: dict with optional keys:
              - ``smpl_joints_local``: (24, 3) — robot-frame, root-relative
              - ``anchor_6d``:        (6,)
              - ``wrist_joints``:     (6,)  IsaacLab wrist order
        """
        if "smpl_joints_local" in data and data["smpl_joints_local"] is not None:
            j = np.asarray(data["smpl_joints_local"], dtype=np.float32)
            # Tile to all 10 history frames so first encoder build is clean.
            # Subsequent calls just replace all rows with the latest — the
            # encoder doesn't see real history here, only "current state
            # repeated", which matches how set_lowerbody_control_var works.
            self._smpl_joints_buf[:] = j[None, :, :]
        if "anchor_6d" in data and data["anchor_6d"] is not None:
            a = np.asarray(data["anchor_6d"], dtype=np.float32)
            self._smpl_anchor_buf[:] = a[None, :]
        if "wrist_joints" in data and data["wrist_joints"] is not None:
            w = np.asarray(data["wrist_joints"], dtype=np.float32)
            self._wrist_joints_buf[:] = w[None, :]

    # ------------------------------------------------------------------ #
    def _build_encoder_obs(self) -> np.ndarray:
        layout = self.encoder_layout
        nf = self.smpl_frames  # 10 (release) or 4 (low-latency)
        enc_obs = np.zeros(self.encoder_dim, dtype=np.float32)

        # encoder_mode_4: [mode_id, 0, 0, 0] — smpl = 2
        enc_obs[0] = 2.0

        # All history rows hold the current frame (streaming, no future), so the
        # first ``nf`` rows are the right slice for either frame count.
        start, end = layout["smpl_joints_10frame_step1"]
        enc_obs[start:end] = self._smpl_joints_buf[:nf].reshape(-1)

        start, end = layout["smpl_anchor_orientation_10frame_step1"]
        enc_obs[start:end] = self._smpl_anchor_buf[:nf].reshape(-1)

        start, end = layout["motion_joint_positions_wrists_10frame_step1"]
        enc_obs[start:end] = self._wrist_joints_buf[:nf].reshape(-1)

        return enc_obs
