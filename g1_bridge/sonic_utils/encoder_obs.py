"""``g1_input`` observations for the sonic checkpoints that ship as a separate
encoder + decoder ONNX pair (not sim2real's usual single fused
``policy.onnx``) -- see ``g1_bridge/bridge_runtime.py``'s
``SplitOnnxBridgeTracking`` for the runtime that consumes them.

Two variants live here, and they must stay in one module: the two monkeypatches
below (``_append_history_wbt`` / ``_gravity_reset_wbt``) are import side effects
that have to apply exactly once, and splitting the classes across modules would
turn "which one got imported first" into a load-bearing accident.

* ``sonic_g1_low_latency_encoder_input`` -- 1247D, the default. Documented
  immediately below; supports the g1 and smpl sub-encoders.
* ``sonic_g1_v1_1_encoder_input`` -- 1751D, heading-normalized, 10-frame smpl.
  Documented at the bottom of this module.

The rest of this docstring is about the low_latency variant.

The low_latency encoder is a **unitoken** model: ONE 1247D graph containing
three encoders (g1 / teleop / smpl), selected at runtime by the single
scalar ``encoder_mode_4[0]`` (mode_id 0 / 1 / 2). Verified by md5: the
encoder this bridge loads is bit-identical to the official
``gear_sonic_deploy/policy/low_latency/model_encoder.onnx``. So switching
teleop styles needs no second checkpoint -- only a different mode_id plus
that mode's own input blocks. This module supports two of the three:

* ``"g1"`` (mode_id=0, the default): reference = 29 joint angles + anchor
  orientation. Driven by kPlanner in PLANNER mode, or by a frozen
  self-anchored default pose otherwise.
* ``"smpl"`` (mode_id=2): reference = the operator's own SMPL body pose,
  i.e. POSE-mode "follow me" teleop. Selected per-tick by the composite
  via ``set_encoder_mode()``/``set_pose_reference()``.

sim2real's own ``sonic.py`` observations module already has the building
blocks for a mode-conditioned "unitoken" encoder input (``sonic_encoder_index``,
``sonic_command_multi_future_nonflat``, ``sonic_motion_anchor_ori_b_mf_nonflat``)
and a finished example of assembling one for the *smpl* mode
(``sonic_smpl_official_encoder_input``, 1762D) -- but no equivalent for the
*g1* mode used by GR00T-WholeBodyControl/gear_sonic_deploy's low_latency
checkpoint (encoder input 1247D). This fills that gap without touching
sim2real's own source tree, registering into the same ``Observation``
registry (namespace ``"sonic"``) purely as an import side effect.

Field order/sizes mirror gear_sonic_deploy's
``policy/release/observation_config_low_latency.yaml`` (``encoder.encoder_observations``)
and ``GetObservationRegistry()`` in
``src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp`` in
GR00T-WholeBodyControl/gear_sonic_deploy -- cross-
checked against humanoid_benchmark's own reimplementation of this exact
layout, ``WBTControllerSonic._build_encoder_obs`` in
``g1_bridge/vendored/wbt/controllers/
wbt_controller_sonic.py`` (``ENCODER_OBS_LAYOUT_LOW_LATENCY``):

    offset  block                                        dim   g1   smpl
    ------  -------------------------------------------  ----  ---  ----
         0  encoder_mode_4                                  4  0.0   2.0
         4  motion_joint_positions_10frame_step1          290   X
       294  motion_joint_velocities_10frame_step1         290   X
       584  motion_anchor_orientation_10frame_step1        60   X
       644  motion_anchor_orientation                       6   X
       650  motion_joint_positions/velocities_lowerbody   240
       890  vr_3point_local_target / _orn_target           21
       911  smpl_joints_4frame_step1                      288        X
      1199  smpl_anchor_orientation_4frame_step1           24        X
      1223  motion_joint_positions_wrists_4frame_step1     24        X
                                                         ----
                                                         1247

Blocks not marked for the active mode stay zero -- matching
``WBTControllerSonic._build_encoder_obs`` (g1) and
``WBTControllerSonicSmpl._build_encoder_obs`` (smpl) byte-for-byte. Those
are tele's own deployment, proven to hold a stable stand / track the
operator with this exact checkpoint in this exact sim. (The real C++
deployment's ``InitializeObservationFunctions()`` does gather every
configured block regardless of mode -- filling the unused ones with real
data was tried here and neither helps nor is needed.)

The smpl blocks are 4-frame, not 10 (that is the whole "low latency" of
this export). As in WBTControllerSonicSmpl, all 4 frames carry the CURRENT
frame: this is a live operator stream, so there is no future to look ahead
at, mirroring the C++ ``operator_state.play == false`` behavior.
"""

from __future__ import annotations

import numpy as np

from sim2real.rl_policy.observations.base import Observation
from sim2real.rl_policy.observations.sonic import (
    _SonicMotionObservation,
    sonic_encoder_index,
    sonic_motion_anchor_ori_b_mf_nonflat,
)
from sim2real.utils.math import (
    matrix_from_quat,
    quat_conjugate,
    quat_mul,
    yaw_quat,
)

ENCODER_DIM = 1247
_MODE_DIM = 4
_POS_DIM = 290
_VEL_DIM = 290
_ANCHOR_DIM = 60
_ANCHOR_CURRENT_DIM = 6
_G1_BLOCK_DIM = _MODE_DIM + _POS_DIM + _VEL_DIM + _ANCHOR_DIM + _ANCHOR_CURRENT_DIM  # 650

# encoder_mode_4[0] values, from observation_config_low_latency.yaml's
# encoder_modes list.
MODE_G1 = 0.0
MODE_TELEOP = 1.0
MODE_SMPL = 2.0

# --------------------------------------------------------------------- #
# teleop-mode (mode_id=1) block spans -- VR 3-point upper body over a
# kPlanner lower body. Identical offsets in ENCODER_OBS_LAYOUT_LOW_LATENCY
# and ENCODER_OBS_LAYOUT_V1_1 (only the smpl blocks past 911 differ between
# the two variants), so one implementation serves both classes; the module
# asserts that below rather than trusting this comment.
# --------------------------------------------------------------------- #
_TELEOP_LOWER_POS_SLICE = (650, 770)  # 10 frames x 12 leg joints
_TELEOP_LOWER_VEL_SLICE = (770, 890)
_TELEOP_VR_3PT_POS_SLICE = (890, 899)  # 3 points x xyz
_TELEOP_VR_3PT_ORN_SLICE = (899, 911)  # 3 points x quat wxyz
_TELEOP_ANCHOR_SLICE = (644, 650)      # current-frame anchor, inside the g1 span
_N_LEG_JOINTS = 12
_TELEOP_FRAMES = (
    _TELEOP_LOWER_POS_SLICE[1] - _TELEOP_LOWER_POS_SLICE[0]
) // _N_LEG_JOINTS  # 10

#: Which 12 of the 29 IsaacLab-ordered joints are legs, in the MuJoCo leg
#: order the encoder slot expects. Imported rather than re-derived: getting
#: this wrong picks up shoulder/waist joints as "legs" and VR_3PT walking is
#: unusable, which is exactly the bug the vendored table's comment records.
from g1_bridge.vendored.wbt.controllers.wbt_controller_sonic_teleop import (  # noqa: E402
    _LOWER_BODY_ISAAC_INDICES_IN_MUJOCO_ORDER as _LEG_IDX,
)

#: Identity quaternion (wxyz) per 3-point, the at-rest value
#: WBTControllerSonicTeleop initialises its buffer to -- an all-zero quat is
#: not a rotation and the encoder has never seen one in training.
_IDENTITY_VR_3PT_ORN = np.tile(
    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), 3
)


def _assert_teleop_slices_agree() -> None:
    """The teleop slots must sit at the same offsets in both variants, or the
    shared builder below silently writes into the wrong block. Checked against
    the authoritative layout tables at import time rather than trusted."""
    from g1_bridge.vendored.wbt.controllers.wbt_controller_sonic import (
        ENCODER_OBS_LAYOUT_LOW_LATENCY,
        ENCODER_OBS_LAYOUT_V1_1,
    )

    expected = {
        "motion_joint_positions_lowerbody_10frame_step5": _TELEOP_LOWER_POS_SLICE,
        "motion_joint_velocities_lowerbody_10frame_step5": _TELEOP_LOWER_VEL_SLICE,
        "vr_3point_local_target": _TELEOP_VR_3PT_POS_SLICE,
        "vr_3point_local_orn_target": _TELEOP_VR_3PT_ORN_SLICE,
        "motion_anchor_orientation": _TELEOP_ANCHOR_SLICE,
    }
    for layout_name, layout in (
        ("low_latency", ENCODER_OBS_LAYOUT_LOW_LATENCY),
        ("v1_1", ENCODER_OBS_LAYOUT_V1_1),
    ):
        for key, span in expected.items():
            if layout[key] != span:
                raise AssertionError(
                    f"teleop slot {key!r} moved in the {layout_name} layout: "
                    f"{layout[key]} != {span}"
                )


_assert_teleop_slices_agree()


def _build_teleop_obs(dim, motion, anchor, vr_3point_target, vr_3point_orn) -> np.ndarray:
    """Encoder input for mode_id=1, shared by both checkpoint variants.

    Port of ``WBTControllerSonicTeleop._build_encoder_obs`` with one
    deliberate difference: the vendored controller's
    ``set_lowerbody_control_var`` replicates ONE planner frame across all ten
    slots (its caller, tele's wbt.py, only ever hands it the current frame),
    whereas the motion buffer here already carries the genuine step5 future
    that the g1 block feeds -- the same quantity these slots name. Feeding the
    real lookahead costs nothing and is what ``10frame_step5`` means.
    """
    out = np.zeros(dim, dtype=np.float32)
    out[0] = MODE_TELEOP

    # (batch, frames, 29) IsaacLab-ordered; batch is always 1 here, exactly as
    # _compute_g1's flatten assumes.
    pos = motion._select(motion.ref_joint_pos_future)[0]
    vel = motion._select(motion.ref_joint_vel_future)[0]
    start, end = _TELEOP_LOWER_POS_SLICE
    out[start:end] = pos[:_TELEOP_FRAMES, _LEG_IDX].reshape(-1)
    start, end = _TELEOP_LOWER_VEL_SLICE
    out[start:end] = vel[:_TELEOP_FRAMES, _LEG_IDX].reshape(-1)

    start, end = _TELEOP_VR_3PT_POS_SLICE
    out[start:end] = vr_3point_target
    start, end = _TELEOP_VR_3PT_ORN_SLICE
    out[start:end] = vr_3point_orn

    # Current-frame anchor. Taken from the same object the g1 path uses, so
    # v1_1 automatically gets its heading-normalized flavour and low_latency
    # its body-frame one.
    start, end = _TELEOP_ANCHOR_SLICE
    out[start:end] = anchor.anchor_ori[0, 0].reshape(-1)
    return out

# smpl-mode block spans, from ENCODER_OBS_LAYOUT_LOW_LATENCY in
# wbt_controller_sonic.py (which reuses the release variant's 10frame slot
# KEY NAMES even though these are semantically 4-frame here).
_SMPL_JOINTS_SLICE = (911, 1199)   # 4 frames x 24 SMPL joints x 3
_SMPL_ANCHOR_SLICE = (1199, 1223)  # 4 frames x 6
_SMPL_WRIST_SLICE = (1223, 1247)   # 4 frames x 6 G1 wrist joints
_N_SMPL_JOINTS = 24
_N_WRIST_JOINTS = 6
_SMPL_FRAMES = (_SMPL_JOINTS_SLICE[1] - _SMPL_JOINTS_SLICE[0]) // (_N_SMPL_JOINTS * 3)  # 4

# Identity anchor (first two columns of I3, row-major) -- the at-rest value
# WBTControllerSonicSmpl initialises/resets its anchor buffer to.
_IDENTITY_ANCHOR6 = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)


# --------------------------------------------------------------------- #
# History-buffer startup semantics: align sim2real's sonic history obs
# with the proven-working WBTControllerSonic (import side effect, applied
# to sim2real's classes at runtime -- its source tree stays untouched,
# same zero-invasion convention as the rest of g1_bridge).
#
# sim2real's _SonicHistoryObservation._append_history bulk-fills ALL
# history slots with the first post-reset sample ("you've been in this
# pose for the whole window"); WBTControllerSonic starts from a zeroed
# history and only ever rolls the newest frame in ("you were at rest and
# just now moved"), with the gravity history zero-state being [0,0,-1]
# (upright), not [0,0,0]. The two behaviors were confirmed to produce
# diverging actions from tick 0 by an offline A/B on randomly perturbed
# states (identical ONNX pair, identical inputs otherwise) -- and WBT's
# variant is the one proven to recover/stand in-env, so match it exactly.
# --------------------------------------------------------------------- #
import sim2real.rl_policy.observations.sonic as _sonic_obs_mod


def _append_history_wbt(self, value) -> None:
    self.history = np.roll(self.history, -1, axis=0)
    self.history[-1, :] = value
    self._history_initialized = True


_sonic_obs_mod._SonicHistoryObservation._append_history = _append_history_wbt


def _gravity_reset_wbt(self) -> None:
    self.history[:] = self.down  # [0, 0, -1]: at-rest upright, matches WBT
    self._history_initialized = False


_sonic_obs_mod.sonic_projected_gravity_history.reset = _gravity_reset_wbt


class sonic_g1_low_latency_encoder_input(Observation, namespace="sonic"):
    """Full 1247D unitoken encoder input, built for whichever encoder mode
    the composite currently selects (see ``set_encoder_mode``); blocks
    belonging to the other modes stay zero. Matches the proven-working
    ``WBTControllerSonic._build_encoder_obs`` (g1) /
    ``WBTControllerSonicSmpl._build_encoder_obs`` (smpl).

    The g1 path reuses sim2real's own mode-index and motion observation
    classes internally (constructed directly, not through the registry, so
    their well-tested update/quaternion-alignment logic isn't reimplemented
    here) -- only the *positions* and *velocities* halves of the 10-frame
    command are needed as separate contiguous blocks (unlike
    ``sonic_command_multi_future_nonflat``, which interleaves them
    per-frame), so those two are read directly off the shared
    ``_SonicMotionObservation`` helper instead.

    The smpl path needs no sim2real observation class at all: its three
    blocks come straight from the composite via ``set_pose_reference()``,
    fed by tele's own ``pico_vr_source.process_smpl_pose_full`` -- the same
    source WBT's POSE mode uses. (sim2real does ship smpl observation
    classes, but they read a ``RealtimeSmplMotionBuffer``, and a Tracking
    has exactly one motion buffer -- which the g1 path already owns. Taking
    the data directly keeps both modes live on a single Tracking instance
    instead of needing a second one.)
    """

    #: Sub-encoders this class can populate; the composite reads it rather than
    #: type-checking, so a new variant only has to declare what it supports.
    SUPPORTED_ENCODER_MODES = ("g1", "smpl", "teleop")

    #: Frame for ``set_pose_reference``'s ``anchor_6d``. This checkpoint's slot
    #: is ``smpl_anchor_orientation_*`` (body frame); v1_1's is the heading one.
    POSE_ANCHOR_FRAME = "body"

    def __init__(
        self,
        future_steps,
        joint_names: str = ".*",
        root_body_name: str = "pelvis",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mode = sonic_encoder_index(env=self.env, width=_MODE_DIM, mode_id=MODE_G1)
        self._motion = _SonicMotionObservation(
            env=self.env,
            future_steps=future_steps,
            joint_names=joint_names,
            root_body_name=root_body_name,
        )
        self._anchor = sonic_motion_anchor_ori_b_mf_nonflat(
            env=self.env,
            future_steps=future_steps,
            root_body_name=root_body_name,
        )
        self._encoder_mode = "g1"
        self._reset_pose_reference()
        self._reset_vr_3point()

    # ------------------------------------------------------------------ #
    # smpl-mode (POSE) reference, pushed by the composite each tick.
    # ------------------------------------------------------------------ #
    def _reset_pose_reference(self) -> None:
        self._pose_smpl_joints = np.zeros((_N_SMPL_JOINTS, 3), dtype=np.float32)
        self._pose_anchor6 = _IDENTITY_ANCHOR6.copy()
        self._pose_wrist_joints = np.zeros(_N_WRIST_JOINTS, dtype=np.float32)

    def _reset_vr_3point(self) -> None:
        self._vr_3point_target = np.zeros(9, dtype=np.float32)
        self._vr_3point_orn = _IDENTITY_VR_3PT_ORN.copy()

    def set_vr_3point(self, vr_3point_target, vr_3point_orn_target) -> None:
        """Latest calibrated 3-point PICO pose for teleop mode: (9,) left
        wrist / right wrist / neck positions and (12,) their quaternions
        (wxyz) -- exactly ``WBTControllerSonicTeleop.set_vr_3point``'s
        contract."""
        self._vr_3point_target = np.asarray(
            vr_3point_target, dtype=np.float32
        ).reshape(9)
        self._vr_3point_orn = np.asarray(
            vr_3point_orn_target, dtype=np.float32
        ).reshape(12)

    def set_encoder_mode(self, mode: str) -> None:
        """Select which of the unitoken encoder's sub-encoders to build for:
        ``"g1"`` (mode_id=0), ``"teleop"`` (mode_id=1) or ``"smpl"``
        (mode_id=2)."""
        if mode not in self.SUPPORTED_ENCODER_MODES:
            raise ValueError(
                f"encoder mode must be one of {self.SUPPORTED_ENCODER_MODES}, "
                f"got {mode!r}"
            )
        self._encoder_mode = mode

    def set_pose_reference(self, smpl_joints_local, anchor_6d, wrist_joints) -> None:
        """Latest operator pose for smpl mode. Shapes: (24, 3) root-local
        SMPL joint positions, (6,) anchor orientation, (6,) G1 wrist joint
        angles in IsaacLab wrist order -- exactly
        ``WBTControllerSonicSmpl.set_pose_control_var``'s contract."""
        self._pose_smpl_joints = np.asarray(smpl_joints_local, dtype=np.float32).reshape(
            _N_SMPL_JOINTS, 3
        )
        self._pose_anchor6 = np.asarray(anchor_6d, dtype=np.float32).reshape(6)
        self._pose_wrist_joints = np.asarray(wrist_joints, dtype=np.float32).reshape(
            _N_WRIST_JOINTS
        )

    def reset(self) -> None:
        self._mode.reset()
        self._motion.reset()
        self._anchor.reset()
        self._encoder_mode = "g1"
        self._reset_pose_reference()
        self._reset_vr_3point()

    def update(self, data) -> None:
        self._mode.update(data)
        self._motion.update(data)
        self._anchor.update(data)

    def compute(self) -> np.ndarray:
        if self._encoder_mode == "smpl":
            return self._compute_smpl()
        if self._encoder_mode == "teleop":
            return self._compute_teleop()
        return self._compute_g1()

    def _compute_teleop(self) -> np.ndarray:
        """teleop mode (mode_id=1) -- VR_3PT: kPlanner legs, PICO arms. The
        g1 and smpl blocks stay zero."""
        return _build_teleop_obs(
            ENCODER_DIM,
            self._motion,
            self._anchor,
            self._vr_3point_target,
            self._vr_3point_orn,
        )

    def _compute_smpl(self) -> np.ndarray:
        """smpl mode (mode_id=2) -- POSE "follow me" teleop. Port of
        ``WBTControllerSonicSmpl._build_encoder_obs``: three blocks, every
        frame holding the current one (live stream, no lookahead). The g1
        blocks stay zero."""
        out = np.zeros(ENCODER_DIM, dtype=np.float32)
        out[0] = MODE_SMPL

        start, end = _SMPL_JOINTS_SLICE
        out[start:end] = np.tile(self._pose_smpl_joints, (_SMPL_FRAMES, 1, 1)).reshape(-1)

        start, end = _SMPL_ANCHOR_SLICE
        out[start:end] = np.tile(self._pose_anchor6, _SMPL_FRAMES)

        start, end = _SMPL_WRIST_SLICE
        out[start:end] = np.tile(self._pose_wrist_joints, _SMPL_FRAMES)
        return out

    def _compute_g1(self) -> np.ndarray:
        out = np.zeros(ENCODER_DIM, dtype=np.float32)
        offset = 0

        mode = self._mode.compute()
        out[offset : offset + _MODE_DIM] = mode
        offset += _MODE_DIM

        pos = self._motion._select(self._motion.ref_joint_pos_future).reshape(-1)
        out[offset : offset + _POS_DIM] = pos
        offset += _POS_DIM

        vel = self._motion._select(self._motion.ref_joint_vel_future).reshape(-1)
        out[offset : offset + _VEL_DIM] = vel
        offset += _VEL_DIM

        anchor = self._anchor.compute()
        out[offset : offset + _ANCHOR_DIM] = anchor
        offset += _ANCHOR_DIM

        # current-frame anchor orientation (future_steps[0] == step 0;
        # anchor_ori shape is (batch=1, n_frames, 3, 2), so [0, 0] is
        # batch-0/frame-0) -- filled unconditionally by WBTControllerSonic
        # even in g1 mode, see module docstring
        out[offset : offset + _ANCHOR_CURRENT_DIM] = self._anchor.anchor_ori[0, 0].reshape(-1)
        offset += _ANCHOR_CURRENT_DIM

        # remaining ENCODER_DIM - _G1_BLOCK_DIM entries stay zero (teleop/smpl
        # blocks, which g1 mode does not use -- matches the proven-working
        # WBTControllerSonic layout, see module docstring)
        return out


# ===================================================================== #
# sonic_v1_1 -- the third GEAR-SONIC variant (nvidia/GEAR-SONIC@sonic_v1_1)
# ===================================================================== #
#
# Same split encoder/decoder shape as low_latency and the SAME decoder
# (obs_dict[1,994] -> action[1,29]), so everything below the token is
# unchanged: proprioception, joint order, gains, action scale. Only the
# encoder differs -- obs_dict[1,1751] -> encoded_tokens[1,64] (verified on
# the downloaded graphs, not inferred from the config).
#
# Three differences from low_latency, all inside the g1 block:
#
#   1. The reference is sampled at **step5** (future_steps [0,5,..,45]),
#      not step1 -- v1_1 reverts to the release variant's horizon. This is
#      set in policy.yaml, not here, but it is why the two classes cannot
#      share one manifest.
#   2. The anchor orientations are normalized by the robot's **heading**
#      (yaw-only) frame instead of its full body frame -- see
#      _SonicMotionAnchorOriHeading below. This is the "robot-heading-
#      normalized" in the upstream release note.
#   3. The smpl blocks are 10-frame (720/60/60) instead of 4-frame
#      (288/24/24), which is what grows 1247 -> 1751. They are NOT
#      implemented here (see the class docstring).
#
# Block layout, from sonic_v1_1/observation_config.yaml's
# encoder_observations order x model_config.yaml's
# env_config.obs.group_obs_dims.tokenizer (offsets accumulate by dim):
#
#     offset  block                                             dim   g1
#     ------  ------------------------------------------------  ----  --
#          0  encoder_mode_4                                       4   X
#          4  motion_joint_positions_10frame_step5               290   X
#        294  motion_joint_velocities_10frame_step5              290   X
#        584  motion_anchor_orientation_heading_10frame_step5     60   X
#        644  motion_anchor_orientation_heading                    6   X
#        650  motion_joint_positions_lowerbody_10frame_step5     120
#        770  motion_joint_velocities_lowerbody_10frame_step5    120
#        890  vr_3point_local_target                               9
#        899  vr_3point_local_orn_target                          12
#        911  smpl_joints_10frame_step1                          720
#       1631  smpl_anchor_orientation_heading_10frame_step1       60
#       1691  motion_joint_positions_wrists_10frame_step1         60
#                                                               ----
#                                                               1751
#
# i.e. ENCODER_OBS_LAYOUT_RELEASE minus the two root_z slots (11 dims),
# with the two anchor slots in low_latency's order (multi-frame first).
# The g1 span is the same 650 dims as low_latency, so out[650:] stays zero.

ENCODER_DIM_V1_1 = 1751

# smpl-mode block spans. Same three slots as low_latency and the same key names
# upstream gives them, but 10-frame instead of 4 -- that growth (288/24/24 ->
# 720/60/60) is the whole 1247 -> 1751 difference.
_V11_SMPL_JOINTS_SLICE = (911, 1631)   # 10 frames x 24 SMPL joints x 3
_V11_SMPL_ANCHOR_SLICE = (1631, 1691)  # 10 frames x 6
_V11_SMPL_WRIST_SLICE = (1691, 1751)   # 10 frames x 6 G1 wrist joints
_V11_SMPL_FRAMES = (
    _V11_SMPL_JOINTS_SLICE[1] - _V11_SMPL_JOINTS_SLICE[0]
) // (_N_SMPL_JOINTS * 3)  # 10

# The current-frame anchor at 644 belongs to the *teleop* mode's
# required_observations, not g1's -- but it is filled here anyway, exactly as
# the low_latency path fills its counterpart: WBTControllerSonic fills it in
# g1 mode too, and the C++ deployment gathers every configured block
# regardless of mode. Keeping the two variants byte-identical in this respect
# means a v1_1-vs-low_latency diff only ever shows the intended changes.
_V11_G1_BLOCK_DIM = _G1_BLOCK_DIM  # 650 -- identical span to low_latency


class _SonicMotionAnchorOriHeading(sonic_motion_anchor_ori_b_mf_nonflat):
    """``motion_anchor_orientation_heading*`` -- the reference root
    orientation expressed in the robot's **heading** frame.

    sim2real's ``sonic_motion_anchor_ori_b_mf_nonflat`` computes the *body*
    frame version, ``conj(robot_root_quat) * ref_root_quat``. v1_1 was
    trained against the heading frame instead: only the robot's yaw is
    removed, so the reference no longer counter-rotates when the torso
    pitches or rolls. That is the whole point of the variant -- it makes the
    encoder input invariant to the robot's absolute facing without also
    coupling it to lean.

    Implemented as an override of ``update()`` that lets the parent run
    first and then recomputes ``anchor_ori``. The recompute is a handful of
    quaternion ops over 10 frames; paying it twice buys us the parent's
    lazy ``_heading_offset`` initialisation and ``motion_obs`` bookkeeping
    verbatim, rather than a copy of them that can drift.

    ``_heading_offset`` (set in the parent's ``reset()``) is a DIFFERENT
    thing and is deliberately left alone: it aligns the reference motion's
    initial yaw to the robot's at episode start. Training applies both, so
    both have to be here.
    """

    def update(self, data) -> None:
        super().update(data)
        ref_root_quat_w = self._select(self.ref_root_quat_future_w)
        heading_offset = np.broadcast_to(
            self._heading_offset.reshape(1, 1, 4),
            ref_root_quat_w.shape,
        )
        ref_root_quat_w = quat_mul(heading_offset, ref_root_quat_w)
        # The only line that differs from the parent: yaw-only robot frame.
        robot_heading_quat_w = yaw_quat(
            self.state_processor.root_quat_w.reshape(1, 4)
        )
        robot_heading_quat_w = np.broadcast_to(
            robot_heading_quat_w.reshape(1, 1, 4),
            ref_root_quat_w.shape,
        )
        rel_quat = quat_mul(quat_conjugate(robot_heading_quat_w), ref_root_quat_w)
        self.anchor_ori = matrix_from_quat(rel_quat)[..., :, :2]


class sonic_g1_v1_1_encoder_input(Observation, namespace="sonic"):
    """Full 1751D unitoken encoder input for the sonic_v1_1 checkpoint.

    Supports the same two sub-encoders as the low_latency class -- ``"g1"``
    (mode_id=0) and ``"smpl"`` (mode_id=2) -- selected per-tick by the composite
    via ``set_encoder_mode()``. Differences from that class, both consequences
    of the checkpoint rather than choices made here:

    * anchor orientations are normalized by the robot's **heading** (yaw-only)
      frame, not its body frame. For the g1 blocks that is
      ``_SonicMotionAnchorOriHeading``; for the smpl block it is the caller's
      job, declared through ``POSE_ANCHOR_FRAME`` below.
    * the smpl blocks are **10-frame** (720/60/60), not 4.

    The teleop sub-encoder (mode_id=1, the VR 3-point one) is wired up on both
    classes as of 2026-08-26 -- see ``_build_teleop_obs``, whose five slots sit
    at identical offsets in both layouts. Before that it raised: an all-zero
    teleop block is a perfectly valid 1751-vector that the graph will happily
    turn into a token, and the robot would follow it into nonsense with nothing
    in the log to say why. That reasoning still stands for any mode a variant
    genuinely cannot populate, which is what ``SUPPORTED_ENCODER_MODES`` is
    for.
    """

    #: Which sub-encoders this class can actually populate. The composite reads
    #: it to decide whether POSE is offerable -- see SonicBackend.
    SUPPORTED_ENCODER_MODES = ("g1", "smpl", "teleop")

    #: Frame the caller must express ``anchor_6d`` in when pushing a pose
    #: reference. v1_1's slot is ``smpl_anchor_orientation_heading_*``, so the
    #: robot's yaw is divided out and its pitch/roll are not -- see
    #: ``g1_runtime.relative_anchor6``. Read by ``_push_pose_reference``; the
    #: two frames agree while the operator is upright, which is why this is a
    #: declared property and not something to eyeball in sim.
    POSE_ANCHOR_FRAME = "heading"

    def __init__(
        self,
        future_steps,
        joint_names: str = ".*",
        root_body_name: str = "pelvis",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mode = sonic_encoder_index(env=self.env, width=_MODE_DIM, mode_id=MODE_G1)
        self._motion = _SonicMotionObservation(
            env=self.env,
            future_steps=future_steps,
            joint_names=joint_names,
            root_body_name=root_body_name,
        )
        self._anchor = _SonicMotionAnchorOriHeading(
            env=self.env,
            future_steps=future_steps,
            root_body_name=root_body_name,
        )
        self._encoder_mode = "g1"
        self._reset_pose_reference()
        self._reset_vr_3point()

    def _reset_pose_reference(self) -> None:
        self._pose_smpl_joints = np.zeros((_N_SMPL_JOINTS, 3), dtype=np.float32)
        self._pose_anchor6 = _IDENTITY_ANCHOR6.copy()
        self._pose_wrist_joints = np.zeros(_N_WRIST_JOINTS, dtype=np.float32)

    def _reset_vr_3point(self) -> None:
        self._vr_3point_target = np.zeros(9, dtype=np.float32)
        self._vr_3point_orn = _IDENTITY_VR_3PT_ORN.copy()

    def set_vr_3point(self, vr_3point_target, vr_3point_orn_target) -> None:
        """Latest calibrated 3-point PICO pose for teleop mode: (9,) left
        wrist / right wrist / neck positions and (12,) their quaternions
        (wxyz) -- exactly ``WBTControllerSonicTeleop.set_vr_3point``'s
        contract."""
        self._vr_3point_target = np.asarray(
            vr_3point_target, dtype=np.float32
        ).reshape(9)
        self._vr_3point_orn = np.asarray(
            vr_3point_orn_target, dtype=np.float32
        ).reshape(12)

    def set_encoder_mode(self, mode: str) -> None:
        if mode not in self.SUPPORTED_ENCODER_MODES:
            raise ValueError(
                f"encoder mode must be one of {self.SUPPORTED_ENCODER_MODES} "
                f"for sonic_v1_1, got {mode!r}"
            )
        self._encoder_mode = mode

    def set_pose_reference(self, smpl_joints_local, anchor_6d, wrist_joints) -> None:
        """Latest operator pose for smpl mode. Same contract as the low_latency
        class -- (24, 3) root-local SMPL joints, (6,) anchor, (6,) G1 wrist
        angles in IsaacLab wrist order -- except the anchor must be in the
        heading frame (``POSE_ANCHOR_FRAME``)."""
        self._pose_smpl_joints = np.asarray(smpl_joints_local, dtype=np.float32).reshape(
            _N_SMPL_JOINTS, 3
        )
        self._pose_anchor6 = np.asarray(anchor_6d, dtype=np.float32).reshape(6)
        self._pose_wrist_joints = np.asarray(wrist_joints, dtype=np.float32).reshape(
            _N_WRIST_JOINTS
        )

    def reset(self) -> None:
        self._mode.reset()
        self._motion.reset()
        self._anchor.reset()
        self._encoder_mode = "g1"
        self._reset_pose_reference()
        self._reset_vr_3point()

    def update(self, data) -> None:
        self._mode.update(data)
        self._motion.update(data)
        self._anchor.update(data)

    def compute(self) -> np.ndarray:
        if self._encoder_mode == "smpl":
            return self._compute_smpl()
        if self._encoder_mode == "teleop":
            return self._compute_teleop()
        return self._compute_g1()

    def _compute_teleop(self) -> np.ndarray:
        """teleop mode (mode_id=1) -- VR_3PT: kPlanner legs, PICO arms. The
        g1 and smpl blocks stay zero."""
        return _build_teleop_obs(
            ENCODER_DIM_V1_1,
            self._motion,
            self._anchor,
            self._vr_3point_target,
            self._vr_3point_orn,
        )

    def _compute_smpl(self) -> np.ndarray:
        """smpl mode (mode_id=2) -- POSE "follow me" teleop. Same shape as the
        low_latency path with 10 frames instead of 4: every frame carries the
        CURRENT one, because this is a live operator stream with no future to
        look ahead at (the C++ deployment's ``operator_state.play == false``
        behaviour). The g1 and teleop blocks stay zero."""
        out = np.zeros(ENCODER_DIM_V1_1, dtype=np.float32)
        out[0] = MODE_SMPL

        start, end = _V11_SMPL_JOINTS_SLICE
        out[start:end] = np.tile(
            self._pose_smpl_joints, (_V11_SMPL_FRAMES, 1, 1)
        ).reshape(-1)

        start, end = _V11_SMPL_ANCHOR_SLICE
        out[start:end] = np.tile(self._pose_anchor6, _V11_SMPL_FRAMES)

        start, end = _V11_SMPL_WRIST_SLICE
        out[start:end] = np.tile(self._pose_wrist_joints, _V11_SMPL_FRAMES)
        return out

    def _compute_g1(self) -> np.ndarray:
        out = np.zeros(ENCODER_DIM_V1_1, dtype=np.float32)
        offset = 0

        out[offset : offset + _MODE_DIM] = self._mode.compute()
        offset += _MODE_DIM

        pos = self._motion._select(self._motion.ref_joint_pos_future).reshape(-1)
        out[offset : offset + _POS_DIM] = pos
        offset += _POS_DIM

        vel = self._motion._select(self._motion.ref_joint_vel_future).reshape(-1)
        out[offset : offset + _VEL_DIM] = vel
        offset += _VEL_DIM

        out[offset : offset + _ANCHOR_DIM] = self._anchor.compute()
        offset += _ANCHOR_DIM

        # current-frame anchor (batch 0, frame 0) -- see _V11_G1_BLOCK_DIM
        out[offset : offset + _ANCHOR_CURRENT_DIM] = self._anchor.anchor_ori[0, 0].reshape(-1)
        offset += _ANCHOR_CURRENT_DIM

        assert offset == _V11_G1_BLOCK_DIM
        # out[650:911] stays zero -- the teleop block, which g1 mode does not
        # use (_compute_teleop fills it instead).
        return out
