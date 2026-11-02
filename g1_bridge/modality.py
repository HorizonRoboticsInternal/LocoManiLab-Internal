"""LeRobot recording schema for every G1BridgeComposite backend.

Pure data -- no `sim2real`/`pinocchio` import, just `ModalityGroup`/
`ModalityBlob`/`ModalityLayout` (the generic machinery, defined in
``robocasa.utils.g1_data_collection``) instantiated per backend. Kept
separate from ``composite.py`` (which has the runtime logic --
``get_teleop_data_sample()`` methods etc.) so the schema is trivially
importable/inspectable without the full sim2real environment, and so there
is exactly one place that defines what each backend records -- see
``G1_BACKEND_TELEOP_KEYS`` at the bottom.

All backends now go through ``G1BridgeComposite`` ("G1_BRIDGE") --
``WholeBodyG1Composite``/``AMOG1Composite`` (g1_bridge/
wbt_g1_composite.py / amo_g1_composite.py) stay registered for
``robocasa.demos.demo_empty_teleop``'s own standalone (non-recording) teleop
mode, but no longer drive data collection, so their schemas
(``AMO_MODALITY_BLOBS`` used to live in ``robocasa.utils.g1_data_collection``
as "shared") are consolidated here instead.
"""

from __future__ import annotations

from g1_bridge.recording.schema import ModalityBlob, ModalityGroup, ModalityLayout

# --------------------------------------------------------------------------- #
# The hand command, identical for every backend: 7 Dex3 finger target angles per
# side, in the actuator order the driver takes (thumb_0,1,2, index_0,1,
# middle_0,1) -- see sim2real.utils.dex3.grip_to_joint_targets, which is what
# produces them on all five.
#
# ONE object, referenced by each layout, rather than four identical literals.
# The names are the reason: a policy server returns whatever key its embodiment
# config declares, and that config is generated from a dataset's modality.json,
# so a group renamed in one backend and not the others produces a checkpoint
# whose grasp silently does not reach the fingers (see
# Sim2RealBackend.vla_hand_targets, which now carries three spellings for
# exactly this reason). Sharing the definition makes divergence impossible
# rather than merely discouraged.
#
# `*_hand_joints` matches what sonic has always called these in its own
# modality.json; the blob backends used the shorter `*_hand` until 2026-08-17.
# The STATE groups stay `left_hand`/`right_hand` -- they are a different
# quantity (measured fingers, not commanded) and modality.json keeps state and
# action in separate namespaces, so the two never collide.
HANDS_ACTION_BLOB = ModalityBlob(
    "action.hands",
    "action",
    (
        ModalityGroup("left_hand_joints", 7, "left_hand_action"),
        ModalityGroup("right_hand_joints", 7, "right_hand_action"),
    ),
)

#: The 43-entry ``observation.state`` the tracking backends record: 29 body
#: joints with the measured 7+7 fingers INTERLEAVED between the arms.
#:
#: The interleaving is not arbitrary -- it is sonic's own order (see
#: recording.frames.sonic_state_43), and matching it is the entire point.
#: Appending both hands after both arms reads as the same 43-wide float32
#: column, so two backends could disagree about which seven entries are a hand
#: with nothing downstream able to tell: not the width, not the dtype, not
#: modality.json's own group table, which would look correct on both sides
#: because each describes itself accurately.
BODY_AND_HANDS_STATE_GROUPS = (
    ModalityGroup("left_leg", 6, "left_leg"),
    ModalityGroup("right_leg", 6, "right_leg"),
    ModalityGroup("waist", 3, "waist"),
    ModalityGroup("left_arm", 7, "left_arm"),
    ModalityGroup("left_hand", 7, "left_hand_state"),
    ModalityGroup("right_arm", 7, "right_arm"),
    ModalityGroup("right_hand", 7, "right_hand_state"),
)

# --------------------------------------------------------------------------- #
# sonic: the one backend with a real PICO/SONIC teleop reference and multiple
# ModeManager modes (OFF/POSE/PLANNER/FROZEN/POSE_PAUSE/PLANNER_VR_3PT) --
# numbered exactly as upstream's StreamMode since 2026-08-26 -- plain-tuple/legacy
# mode (one LeRobot column per signal, see TELEOP_FEATURE_SPECS in
# robocasa.utils.g1_data_collection), not a modality blob: keeps the full
# target_q_29 + vr_3pt/smpl/wrist + stream_mode/locomotion_mode/
# facing_direction schema, plus the GR00T unitree_g1_sonic action fields
# (motion_token + left/right hand joints) so the dataset matches what the
# pre-G1_BRIDGE collection path produced. locomotion_mode/facing_direction
# now come from the ported kPlanner's MovementState in PLANNER mode (see
# composite.py's _step_kplanner_pose); they stay at the MovementState
# default (IDLE/+x) otherwise.
#
# ``left_hand_joints``/``right_hand_joints`` are COMMANDS (the trigger-driven
# Dex3 targets), like every other backend's hand action -- see
# backends/sonic.teleop_sample. Datasets collected before 2026-08-11 have the
# measured finger angles in those columns instead, and the 43-D
# ``observation.state`` of a pre-2026-08-11 dataset is mis-ordered as well (real
# host: arms/hands transposed; sim host: MJCF finger order). ``schema_version``
# in ``meta/info.json``'s ``collection_info`` is what tells the two apart --
# nothing about the columns themselves does.
# --------------------------------------------------------------------------- #
#
# schema v6 (2026-08-22): ``target_q_29`` -> ``target_q_43`` (``action.wbc`` now
# carries the commanded fingers in the state's slots), plus the signals
# NVlabs' collector records that this side had available but never wrote:
# ``smpl_pose``, the three kPlanner MovementState fields beyond
# locomotion_mode/facing, the raw triggers, a PICO sample counter and the
# engage-time pelvis quaternion. See recording.features SCHEMA_VERSION bullet 6.
# --------------------------------------------------------------------------- #
SONIC_TELEOP_KEYS = (
    "target_q_43",
    "motion_token",
    "left_hand_joints",
    "right_hand_joints",
    "locomotion_mode",
    "facing_direction",
    "stream_mode",
    "vr_3point_pos",
    "vr_3point_orn",
    "smpl_joints_local",
    "smpl_root_quat",
    "wrist_joints",
    "smpl_pose",
    "planner_movement",
    "planner_speed",
    "planner_height",
    "left_trigger",
    "right_trigger",
    "pico_frame_index",
    "init_base_quat",
)

# --------------------------------------------------------------------------- #
# amo: command-velocity legs/waist (RL policy) + PICO 3-point arm-IK +
# trigger-driven hand grip. Joint-grouped modality-blob schema matching the
# reference GR00T-format AMO dataset's field grouping exactly
# (G1WholebodyBendPickMP-v0/meta/modality.json): state = current left/right
# hand(7 each)/arm(7 each) + torso rpy(3)/height(1), packed into one "states"
# blob; action = the corresponding commanded targets, same groups, plus
# torso_vx/vy/vyaw and the integrated target_yaw heading (relative to the
# episode's own start heading -- see backends/amo._ensure_yaw_anchor), packed
# into one "action" blob -- EXCEPT the hands, which as of schema v5 sit in the
# shared ``action.hands`` blob like every other backend's, so amo's own
# "action" column is the 22 non-hand entries. See _amo_teleop_data_sample() for
# exactly what each key is.
# --------------------------------------------------------------------------- #
AMO_MODALITY_BLOBS = (
    ModalityBlob(
        # "observation.state", not "states": the blob's feature_name IS the LeRobot
        # column name, and consumers that classify columns by the LeRobot constant
        # (PRTS matches OBS_STATE = "observation.state") silently see NO state at all
        # under any other name -- it surfaced as `torch.cat(): expected a non-empty list
        # of Tensors`, not as "unknown column". heft/sonic already use the standard name;
        # amo was the only one off-convention. Datasets collected before 2026-08-07 still
        # carry the old `states` column.
        "observation.state",
        "state",
        (
            ModalityGroup("left_hand", 7, "left_hand_state"),
            ModalityGroup("right_hand", 7, "right_hand_state"),
            ModalityGroup("left_arm", 7, "left_arm_state"),
            ModalityGroup("right_arm", 7, "right_arm_state"),
            # rpy/height STATE = the torso command in force during the PREVIOUS
            # tick (schema v7; Psi0's definition for its G1 wholebody datasets).
            # A command, so every host has it -- the real layout no longer drops
            # `height`. The measured attitude is the root_orientation /
            # projected_gravity blobs below.
            ModalityGroup("rpy", 3, "rpy_state"),
            ModalityGroup("height", 1, "height_state"),
        ),
    ),
    # Measured pelvis attitude (schema v7) -- the same two blobs heft/almi
    # record. AMO's own balance policy observes root roll/pitch
    # (sim2real/rl_policy/observations/amo.py), and the rpy STATE group above
    # is a torso-pose COMMAND (previous tick's), so without these the dataset
    # had no measured attitude at all. build_blob_frame computes both on every
    # host (IMU on hardware, robosuite base quat in sim); they were simply not
    # declared here before.
    ModalityBlob(
        "observation.root_orientation",
        "state",
        (
            ModalityGroup(
                "root_orientation", 4, "root_orientation", rotation_type="quaternion"
            ),
        ),
    ),
    ModalityBlob(
        "observation.projected_gravity",
        "state",
        (ModalityGroup("projected_gravity", 3, "projected_gravity"),),
    ),
    ModalityBlob(
        "action",
        "action",
        (
            ModalityGroup("left_arm", 7, "left_arm_action"),
            ModalityGroup("right_arm", 7, "right_arm_action"),
            ModalityGroup("rpy", 3, "rpy_action"),
            ModalityGroup("height", 1, "height_action"),
            ModalityGroup("torso_vx", 1, "torso_vx"),
            ModalityGroup("torso_vy", 1, "torso_vy"),
            ModalityGroup("torso_vyaw", 1, "torso_vyaw"),
            ModalityGroup("target_yaw", 1, "target_yaw"),
        ),
    ),
    # Split out of the blob above (where they were its first 14 entries) so the
    # hands live under `action.hands` for every backend alike. amo was the only
    # one that packed them in with the locomotion command, which meant a
    # consumer wanting just the grasp had to know amo's offsets specifically.
    HANDS_ACTION_BLOB,
)
AMO_MODALITY_LAYOUT = ModalityLayout(
    readiness_source="left_arm_action", blobs=AMO_MODALITY_BLOBS
)

# --------------------------------------------------------------------------- #
# heft/scalebfm: full-body GMR-retargeted tracking policies -- state =
# current joint state (leg/waist/arm groups, sliced from the canonical
# 29-order qpos, plus the measured fingers -- 43 state entries in the same
# body-then-hands order sonic uses; joint velocity intentionally not recorded);
# action = the GMR
# reference motion actually pushed into the policy this tick (root pos(3)/
# quat(4) + 29 joint angles, from the GMR payload's own "qpos" -- see
# pico_retarget_pub.py._build_payload; verified against Axellwppr/
# motion_tracking@sim2real's VRMotionSource._parse_qpos_frames and
# zengweishuai/ScaleBFM's MotionTrackingEnv, both of which use the same
# root_pos/root_quat/joint_pos representation) + the PICO trigger-driven hand
# command, reported as 7-per-side target joint angles (grip_to_joint_targets,
# same as AMO's hand action -- not a raw 0/1 trigger scalar, and not fed
# through either policy in this bridge's simulation wiring today -- hands are
# a separate trigger-driven PD loop, same as AMO/SONIC). See composite.py's
# _generic_teleop_data_sample() for exactly what each key is.
# --------------------------------------------------------------------------- #
GENERIC_TRACKING_MODALITY_BLOBS = (
    ModalityBlob(
        "observation.state",
        "state",
        BODY_AND_HANDS_STATE_GROUPS,
    ),
    ModalityBlob(
        "observation.root_orientation",
        "state",
        (
            ModalityGroup(
                "root_orientation", 4, "root_orientation", rotation_type="quaternion"
            ),
        ),
    ),
    ModalityBlob(
        "observation.projected_gravity",
        "state",
        (ModalityGroup("projected_gravity", 3, "projected_gravity"),),
    ),
    ModalityBlob(
        "action.ref_qpos",
        "action",
        (
            ModalityGroup("ref_root_pos", 3, "ref_root_pos"),
            ModalityGroup(
                "ref_root_quat", 4, "ref_root_quat", rotation_type="quaternion"
            ),
            ModalityGroup("ref_joint_pos", 29, "ref_joint_pos"),
        ),
    ),
    HANDS_ACTION_BLOB,
)
GENERIC_TRACKING_MODALITY_LAYOUT = ModalityLayout(
    readiness_source="ref_joint_pos", blobs=GENERIC_TRACKING_MODALITY_BLOBS
)

# --------------------------------------------------------------------------- #
# almi: dual-branch policy whose command is split by body half -- a twist for
# the legs and a retargeted pose reference for the waist and arms. State is the
# same limb-grouped joint state the tracking backends record.
#
# The action side is exactly ALMI's own policy input, which is the point: what
# gets recorded IS the training target, so a VLA replacing the operator has to
# emit precisely these fields. `waist_rpy`, `ee_pos`/`ee_quat` and
# `ref_joint_pos` all come from one retargeted frame and are individually
# redundant -- the upper policy consumes all three, so all three are stored
# rather than making anything downstream re-derive them (the EE pose in
# particular would need a G1 forward-kinematics model to recover from the joint
# reference alone).
#
# Note `waist_rpy` is *derived* from `ref_joint_pos`, not commanded separately:
# AlmiCommands.validated() re-reads it out of the reference's waist entries,
# mirroring how mjlab's WaistRpyCommand syncs off the same motion frame. The two
# columns therefore always agree, and a replay that edits one must edit both.
#
# What is deliberately NOT recorded: the 29-D joint target the two branches
# produced. It is a function of this command plus the checkpoint, so storing it
# would add a second, redundant action space -- see AlmiBackend.teleop_sample.
# --------------------------------------------------------------------------- #
ALMI_MODALITY_BLOBS = (
    ModalityBlob(
        "observation.state",
        "state",
        BODY_AND_HANDS_STATE_GROUPS,
    ),
    ModalityBlob(
        "observation.root_orientation",
        "state",
        (
            ModalityGroup(
                "root_orientation", 4, "root_orientation", rotation_type="quaternion"
            ),
        ),
    ),
    ModalityBlob(
        "observation.projected_gravity",
        "state",
        (ModalityGroup("projected_gravity", 3, "projected_gravity"),),
    ),
    ModalityBlob(
        "action.twist",
        "action",
        (
            ModalityGroup("lin_vel", 2, "twist_lin_vel"),
            ModalityGroup("ang_vel_z", 1, "twist_ang_vel_z"),
            # A commanded posture in metres, not a velocity -- the one twist
            # channel that does not go to zero when the operator disengages.
            ModalityGroup("base_height", 1, "twist_base_height"),
        ),
    ),
    ModalityBlob(
        "action.upper_ref",
        "action",
        (
            ModalityGroup("waist_rpy", 3, "waist_rpy"),
            ModalityGroup("left_ee_pos", 3, "left_ee_pos"),
            ModalityGroup("right_ee_pos", 3, "right_ee_pos"),
            ModalityGroup(
                "left_ee_quat", 4, "left_ee_quat", rotation_type="quaternion"
            ),
            ModalityGroup(
                "right_ee_quat", 4, "right_ee_quat", rotation_type="quaternion"
            ),
            ModalityGroup("ref_joint_pos", 17, "upper_ref_joint_pos"),
        ),
    ),
    HANDS_ACTION_BLOB,
)
ALMI_MODALITY_LAYOUT = ModalityLayout(
    readiness_source="upper_ref_joint_pos", blobs=ALMI_MODALITY_BLOBS
)

# Sample keys/layout G1BridgeComposite.get_teleop_data_sample() actually
# returns, keyed by --backend -- the single source of truth consumed by the
# unified collection script (robocasa/scripts/collect_demos_g1.py) the same
# way robocasa.utils.g1_data_collection's own CONTROLLER_TELEOP_KEYS drove
# the old tele-native-only collect_demos_g1.py.
G1_BACKEND_TELEOP_KEYS = {
    "sonic": SONIC_TELEOP_KEYS,
    "heft": GENERIC_TRACKING_MODALITY_LAYOUT,
    "scalebfm": GENERIC_TRACKING_MODALITY_LAYOUT,
    "amo": AMO_MODALITY_LAYOUT,
    "almi": ALMI_MODALITY_LAYOUT,
}

# --------------------------------------------------------------------------- #
# Real-robot layouts. As of schema v7 IDENTICAL to the sim ones for every
# backend: the one group that used to differ -- amo's `height` STATE, which was
# the sim-only `read_root_pos()[2]` (no odometry on a G1) -- is now the
# previous tick's height COMMAND (Psi0's definition), which every host has.
# Everything else was always available on hardware: finger joints come from
# the Dex3 driver's DDS feedback (see Sim2RealContext.read_hand_state),
# arm/leg/waist state from the robot's own state stream, the root quaternion
# from the IMU, and every teleop.* field from the PICO source.
#
# The names are kept so callers (collect_real, tests) have one place to ask
# "what does the real host record"; a future hardware-only gap goes here,
# rebuilt by NAME (a by-index rebuild once silently dropped amo's hands when a
# third blob was added).
# --------------------------------------------------------------------------- #
AMO_REAL_MODALITY_BLOBS = AMO_MODALITY_BLOBS
AMO_REAL_MODALITY_LAYOUT = AMO_MODALITY_LAYOUT

#: Per-backend layouts for a hardware host. Anything absent here records exactly
#: what it records in sim.
G1_BACKEND_TELEOP_KEYS_REAL = {
    **G1_BACKEND_TELEOP_KEYS,
    "amo": AMO_REAL_MODALITY_LAYOUT,
}


def layout_for(backend: str, *, real: bool = False):
    """The recording layout for one backend on one kind of host."""
    table = G1_BACKEND_TELEOP_KEYS_REAL if real else G1_BACKEND_TELEOP_KEYS
    return table[backend]
