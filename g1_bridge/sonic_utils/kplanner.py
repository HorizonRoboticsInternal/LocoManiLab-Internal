"""Python port of GR00T-SONIC's kinematic planner (kPlanner).

Mirrors ``LocalMotionPlannerBase`` / ``LocalMotionPlannerONNX`` from
``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/``, just enough to feed
the wbt SONIC controller in-process (no DDS, no TensorRT).

Input ONNX model: ``planner_sonic.onnx`` (V2, 11 inputs, 27 modes).

Storage convention: ``planner_motion_50hz`` is a ``[N, 36]`` float32 array of
MuJoCo qpos rows ``[root_xyz(3), root_quat_wxyz(4), 29 joints in MuJoCo body
tree order]`` — the same layout as the planner's ``context_mujoco_qpos``
input.  Reordering to IsaacLab (the order SONIC expects) is done at SONIC
hand-off time, not in the buffer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# G1 has 29 actuated body joints + 7-dim free joint (xyz + wxyz) = 36 qpos dims.
G1_NUM_MOTOR = 29
QPOS_DIM = 7 + G1_NUM_MOTOR

# Default standing height for context init — taken from
# PlannerConfig::default_height in localmotion_kplanner.hpp.
DEFAULT_STANDING_HEIGHT = 0.788740


class LocomotionMode:
    """Subset of ``LocomotionMode`` from localmotion_kplanner.hpp."""

    IDLE = 0
    SLOW_WALK = 1
    WALK = 2
    RUN = 3
    SQUAT = 4
    KNEEL_TWO_LEGS = 5
    KNEEL = 6
    LYING = 7
    CRAWLING = 8
    IDEL_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17


# Modes where the planner does *not* replan on speed/direction change —
# only on mode/facing/height change. Mirrors ``is_static_motion_mode``.
STATIC_MODES = frozenset(
    {
        LocomotionMode.IDLE,
        LocomotionMode.SQUAT,
        LocomotionMode.KNEEL_TWO_LEGS,
        LocomotionMode.KNEEL,
        LocomotionMode.LYING,
        LocomotionMode.IDEL_BOXING,
    }
)


@dataclass
class MovementState:
    """Per-tick locomotion command passed from input to planner."""

    locomotion_mode: int = LocomotionMode.IDLE
    movement_direction: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )
    facing_direction: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float32)
    )
    movement_speed: float = -1.0  # -1 = use mode default
    height: float = -1.0  # -1 = disabled

    def copy(self) -> "MovementState":
        return MovementState(
            locomotion_mode=self.locomotion_mode,
            movement_direction=self.movement_direction.copy(),
            facing_direction=self.facing_direction.copy(),
            movement_speed=float(self.movement_speed),
            height=float(self.height),
        )


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between two wxyz quaternions."""
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        out = q0 + t * (q1 - q0)
        n = float(np.linalg.norm(out))
        return out / n if n > 1e-12 else q0.copy()
    theta_0 = np.arccos(np.clip(d, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    if sin_theta_0 < 1e-9:
        return q0.copy()
    theta = theta_0 * t
    s0 = np.cos(theta) - d * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


class SonicKPlanner:
    """In-process Python kPlanner wrapping ``planner_sonic.onnx``.

    Owns no MuJoCo state — callers pass the robot pose to ``initialize`` and
    drive ``update_planning`` on whatever schedule they like (the C++ deploy
    runs it on a dedicated 10 Hz thread; we just throttle it inline).
    """

    LOOK_AHEAD_STEPS = 2  # 0.04 s at 50 Hz

    def __init__(
        self,
        planner_path: str,
        default_height: float = DEFAULT_STANDING_HEIGHT,
        seed: int = 1234,
        allowed_tokens_mask: Optional[Sequence[int]] = None,
        providers: Optional[Sequence[str]] = None,
    ):
        if not os.path.exists(planner_path):
            raise FileNotFoundError(f"kPlanner ONNX not found: {planner_path}")
        # Imported lazily so the module is cheap to import for users who only
        # need default_pos / sonic / almi.
        import onnxruntime as ort

        if providers is None:
            # GPU (CUDA) with CPU fallback when onnxruntime-gpu is installed;
            # this 739MB planner is the heaviest SONIC inference, so the GPU
            # speedup matters most here.
            from g1_bridge.vendored.wbt.utils.trt_utils import (
                preferred_providers,
            )

            providers = preferred_providers()
        so = ort.SessionOptions()
        so.log_severity_level = 3  # silence per-inference ScatterND WARNING spam
        self.session = ort.InferenceSession(
            planner_path,
            sess_options=so,
            providers=list(providers),
        )
        self.default_height = float(default_height)
        self.seed = int(seed)

        # Reusable input buffers (kept on the planner instance; new arrays each
        # session.run() call is also fine, this just keeps allocations down).
        self._context = np.zeros((1, 4, QPOS_DIM), dtype=np.float32)
        self._mode = np.zeros((1,), dtype=np.int64)
        self._target_vel = np.array([-1.0], dtype=np.float32)
        self._height = np.array([-1.0], dtype=np.float32)
        self._mov_dir = np.zeros((1, 3), dtype=np.float32)
        self._face_dir = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        self._seed = np.array([self.seed], dtype=np.int64)
        # Advanced inputs we don't use — held at safe defaults.
        self._has_specific = np.zeros((1, 1), dtype=np.int64)
        self._specific_pos = np.zeros((1, 4, 3), dtype=np.float32)
        self._specific_head = np.zeros((1, 4), dtype=np.float32)
        if allowed_tokens_mask is None:
            # C++ default: [1]*6 + [0]*5 — allow 6..11 tokens.
            allowed_tokens_mask = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        self._allowed_tokens = np.array([list(allowed_tokens_mask)], dtype=np.int64)

        # Output buffer.
        self.planner_motion_50hz: Optional[np.ndarray] = None  # [N, 36]
        self.gen_frame = 0
        self.initialized = False

    # ------------------------------------------------------------------ #
    # Inference + resampling
    # ------------------------------------------------------------------ #
    def _run_inference(self) -> np.ndarray:
        outputs = self.session.run(
            None,
            {
                "context_mujoco_qpos": self._context,
                "target_vel": self._target_vel,
                "mode": self._mode,
                "movement_direction": self._mov_dir,
                "facing_direction": self._face_dir,
                "random_seed": self._seed,
                "has_specific_target": self._has_specific,
                "specific_target_positions": self._specific_pos,
                "specific_target_headings": self._specific_head,
                "allowed_pred_num_tokens": self._allowed_tokens,
                "height": self._height,
            },
        )
        qpos_padded = outputs[0][0]  # [N_max, 36]
        n_pred = int(outputs[1][0])
        if n_pred <= 0 or n_pred > qpos_padded.shape[0]:
            raise RuntimeError(f"kPlanner returned invalid num_pred_frames={n_pred}")
        if not np.isfinite(qpos_padded[:n_pred]).all():
            raise RuntimeError("kPlanner output contains NaN/Inf")
        return qpos_padded[:n_pred].astype(np.float32, copy=True)

    @staticmethod
    def _resample_30hz_to_50hz(qpos_30hz: np.ndarray) -> np.ndarray:
        """Resample a 30Hz qpos sequence to 50Hz with linear/slerp interp."""
        n30 = qpos_30hz.shape[0]
        n50 = int(np.floor(n30 / 30.0 * 50.0))
        if n50 <= 0:
            n50 = 1
        out = np.zeros((n50, QPOS_DIM), dtype=np.float32)
        for f in range(n50):
            f30 = (f / 50.0) * 30.0
            f0 = int(np.floor(f30))
            f1 = min(f0 + 1, n30 - 1)
            w1 = f30 - f0
            w0 = 1.0 - w1
            out[f, 0:3] = w0 * qpos_30hz[f0, 0:3] + w1 * qpos_30hz[f1, 0:3]
            out[f, 3:7] = _slerp(qpos_30hz[f0, 3:7], qpos_30hz[f1, 3:7], w1)
            out[f, 7:QPOS_DIM] = (
                w0 * qpos_30hz[f0, 7:QPOS_DIM] + w1 * qpos_30hz[f1, 7:QPOS_DIM]
            )
        return out

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def initialize(
        self,
        base_quat_wxyz: np.ndarray,
        joint_positions_mujoco: np.ndarray,
        mode: int = LocomotionMode.IDLE,
    ) -> bool:
        """Seed the context at standing default + run the first inference.

        Mirrors ``LocalMotionPlannerBase::Initialize`` — all 4 context frames
        are placed at (0, 0, default_height) with the robot's actual spawn
        yaw (from *base_quat_wxyz*) and current joint positions.
        ``_face_dir`` is derived from the spawn quaternion so the kPlanner's
        heading matches the real robot from the very first tick — no twitch.
        """
        if joint_positions_mujoco.shape != (G1_NUM_MOTOR,):
            raise ValueError(
                f"joint_positions_mujoco must be (29,), got {joint_positions_mujoco.shape}"
            )
        # Derive yaw + forward direction from the spawn quaternion so context
        # frames and _face_dir are consistent from the start.
        w, x, y, z = (
            float(base_quat_wxyz[0]),
            float(base_quat_wxyz[1]),
            float(base_quat_wxyz[2]),
            float(base_quat_wxyz[3]),
        )
        fwd_x = 1.0 - 2.0 * (y * y + z * z)
        fwd_y = 2.0 * (x * y + w * z)
        spawn_yaw = float(np.arctan2(fwd_y, fwd_x))
        spawn_facing = np.array(
            [np.cos(spawn_yaw), np.sin(spawn_yaw), 0.0], dtype=np.float32
        )
        self._context.fill(0.0)
        for n in range(4):
            self._context[0, n, 0] = 0.0
            self._context[0, n, 1] = 0.0
            self._context[0, n, 2] = self.default_height
            self._context[0, n, 3] = float(base_quat_wxyz[0])  # qw
            self._context[0, n, 4] = float(base_quat_wxyz[1])  # qx
            self._context[0, n, 5] = float(base_quat_wxyz[2])  # qy
            self._context[0, n, 6] = float(base_quat_wxyz[3])  # qz
            self._context[0, n, 7:QPOS_DIM] = joint_positions_mujoco
        self._mode[0] = int(mode)
        self._mov_dir[:] = 0.0
        self._face_dir[0, :] = spawn_facing
        self._target_vel[0] = -1.0
        self._height[0] = -1.0

        try:
            qpos_30hz = self._run_inference()
            self.planner_motion_50hz = self._resample_30hz_to_50hz(qpos_30hz)
            self.gen_frame = 0
            self.initialized = True
            return True
        except Exception as e:
            print(f"[kPlanner] initialize() failed: {e}")
            self.initialized = False
            self.planner_motion_50hz = None
            return False

    def update_planning(
        self,
        current_frame: int,
        movement_state: MovementState,
        seed: Optional[int] = None,
    ) -> bool:
        """Run a replan with new MovementState, refresh planner_motion_50hz.

        Mirrors ``LocalMotionPlannerBase::UpdatePlanning``.  Context is
        sampled from the *current* planner_motion at 30 Hz starting at
        ``current_frame + LOOK_AHEAD_STEPS``.  After this call,
        ``planner_motion_50hz`` is freshly populated and the caller should
        reset its frame cursor to 0.
        """
        if not self.initialized or self.planner_motion_50hz is None:
            return False
        n_motion = self.planner_motion_50hz.shape[0]
        self.gen_frame = current_frame + self.LOOK_AHEAD_STEPS
        # Build 4-frame context at 30Hz from current planner motion.
        for n in range(4):
            t = self.gen_frame / 50.0 + n / 30.0
            f_50hz = t * 50.0
            f0 = int(np.floor(f_50hz))
            f0 = min(max(f0, 0), n_motion - 1)
            f1 = min(f0 + 1, n_motion - 1)
            w1 = f_50hz - f0
            w0 = 1.0 - w1
            self._context[0, n, 0:3] = (
                w0 * self.planner_motion_50hz[f0, 0:3]
                + w1 * self.planner_motion_50hz[f1, 0:3]
            )
            self._context[0, n, 3:7] = _slerp(
                self.planner_motion_50hz[f0, 3:7],
                self.planner_motion_50hz[f1, 3:7],
                w1,
            )
            self._context[0, n, 7:QPOS_DIM] = (
                w0 * self.planner_motion_50hz[f0, 7:QPOS_DIM]
                + w1 * self.planner_motion_50hz[f1, 7:QPOS_DIM]
            )

        self._mode[0] = int(movement_state.locomotion_mode)
        self._target_vel[0] = float(movement_state.movement_speed)
        self._height[0] = float(movement_state.height)
        self._mov_dir[0, :] = movement_state.movement_direction.astype(np.float32)
        self._face_dir[0, :] = movement_state.facing_direction.astype(np.float32)
        if seed is not None:
            self.seed = int(seed)
            self._seed[0] = self.seed

        try:
            qpos_30hz = self._run_inference()
            self.planner_motion_50hz = self._resample_30hz_to_50hz(qpos_30hz)
            return True
        except Exception as e:
            print(f"[kPlanner] update_planning() failed: {e}")
            return False

    def sample_frame(self, frame: int) -> np.ndarray:
        """Return a single qpos row (36,), clamped to the last frame."""
        if self.planner_motion_50hz is None:
            raise RuntimeError("kPlanner not initialised")
        n = self.planner_motion_50hz.shape[0]
        idx = min(max(frame, 0), n - 1)
        return self.planner_motion_50hz[idx].copy()

    def sample_joint_velocity(self, frame: int) -> np.ndarray:
        """Finite-diff joint velocities at ``frame`` (MuJoCo joint order)."""
        if self.planner_motion_50hz is None:
            raise RuntimeError("kPlanner not initialised")
        n = self.planner_motion_50hz.shape[0]
        i0 = min(max(frame, 0), n - 1)
        i1 = min(i0 + 1, n - 1)
        if i0 == i1:
            return np.zeros(G1_NUM_MOTOR, dtype=np.float32)
        dq = (
            self.planner_motion_50hz[i1, 7:QPOS_DIM]
            - self.planner_motion_50hz[i0, 7:QPOS_DIM]
        )
        return (dq * 50.0).astype(np.float32, copy=False)

    @property
    def num_frames(self) -> int:
        return (
            0
            if self.planner_motion_50hz is None
            else int(self.planner_motion_50hz.shape[0])
        )
