"""The LeRobot feature set a G1 teleop dataset carries, host-agnostic.

Which columns exist is a property of the *controller*, not of the host: a heft
dataset recorded on hardware must carry the same ``observation.state(29)`` /
``action.ref_qpos(36)`` / ``action.hands(14)`` columns as one recorded in sim, or
nothing downstream (LocoManiLab's variant detection, every model's data config) can
treat the two as the same thing.

What legitimately differs between hosts is only what the host can observe:

* ``SIM_ONLY_FEATURES`` -- end-effector pose (robosuite computes it), and the
  reward/done pair (a hardware run has no ``_check_success``). The sim collector
  passes these as ``extra_features``; the real one does not.
* the ego camera's frame size, which is the render resolution in sim and the
  camera's own in the real world. Hence ``ego_view_feature(h, w)`` rather than a
  constant -- the *name* and dtype are fixed, the geometry is not.
"""

from __future__ import annotations

import pathlib

#: modality.json for the legacy plain-tuple (SONIC) schema: a static asset rather
#: than something generated from blob offsets, and now shipped WITH the recording
#: layer instead of with the benchmark -- a real-robot sonic run needs exactly the
#: same file, and reaching into robocasa for it is the dependency this package
#: exists to not have.
LEGACY_MODALITY_ASSET = pathlib.Path(__file__).resolve().parent / "assets" / "G1_modality.json"

#: Control rate and recording fps. 50 Hz everywhere: the sim's --control-freq
#: default, the real controller's rl_rate, and the rate the reference-motion
#: contract is defined at.
FPS = 50

#: Stamped into every dataset's ``collection_info`` by ``make_dataset``. Bump it
#: whenever the MEANING of an existing column changes without its name, dtype or
#: shape changing -- those are the changes nothing downstream can detect, and
#: mixing two of them in one training set is silent.
#:
#:   1  (implicit; absent from the field) everything collected before 2026-08-11.
#:   2  sonic only: ``observation.state``'s 43 entries are now assembled
#:      identically on both hosts -- 29 body joints with the 7+7 fingers
#:      interleaved between the arms, fingers in Dex3 actuator order (see
#:      recording.frames.sonic_state_43). Previously the real host transposed
#:      the right arm with the left hand, and the sim host emitted MJCF finger
#:      order (thumb, middle, index). Also: ``teleop.{left,right}_hand_joints``
#:      are the commanded Dex3 targets, not the measured finger angles.
#:      The blob-schema backends (amo/heft/scalebfm/almi) are unaffected -- they
#:      never used either code path -- but carry the field too, so that "no
#:      schema_version" always means "collected before this was tracked".
#:   3  heft/scalebfm only: ``action.ref_qpos``'s ``ref_root_pos``/
#:      ``ref_root_quat`` are now in the EPISODE's own frame (its first recorded
#:      reference is the origin; yaw and xy removed, height/roll/pitch kept) --
#:      see backends/generic._episode_relative_root. Previously they carried the
#:      operator's absolute GMR-world pose, whose origin is per-GMR-process, so
#:      each episode continued from where the previous one stopped.
#:      ``ref_joint_pos`` and every other column are unchanged.
#:   4  heft/scalebfm only: ``action.ref_qpos``'s ``ref_root_quat`` is now the
#:      ``w >= 0`` representative. v3 recorded whatever sign the continuous
#:      stream happened to be on, and a turn past pi crosses into the negative
#:      half of the double cover and stays there -- on
#:      IThorLivingRoomBasketToSofa (a ~2.8 rad turn) that was 44/51 episodes
#:      and 74.6% of all frames, which moved the column's statistics from
#:      mean/std ``+0.854/0.277`` to ``-0.384/0.811``. Since q and -q are the
#:      same rotation, a v3 set trains a regression on two opposite labels for
#:      one heading and it learns their mean: a root that never turns (measured
#:      on a 20k-step GR00T checkpoint: <2 cm and <0.09 rad over 1800 steps).
#:      A v3 dataset can be repaired in place -- flip the sign of every
#:      ``ref_root_quat`` row with ``w < 0`` and re-derive meta/stats.json --
#:      because no information is lost, only the representative.
#:   5  Hands, everywhere. Three changes that together make one hand schema
#:      instead of five, and unlike 2-4 this one DOES change names and shapes:
#:      a v5 dataset is structurally distinguishable from a v4, and the two
#:      cannot be merged.
#:        * heft/scalebfm/almi gain the MEASURED fingers in
#:          ``observation.state`` (``left_hand``/``right_hand``, 7 each, after
#:          the arms -- so state goes 29 -> 43, the same 43 sonic has always
#:          had). They previously recorded the hand COMMAND and nothing else,
#:          which left a policy unable to see its own grasp: it could predict
#:          a close but never observe whether the fingers got there.
#:        * every blob backend's hand ACTION groups are renamed
#:          ``left_hand``/``right_hand`` -> ``left_hand_joints``/
#:          ``right_hand_joints``, matching what sonic's modality.json has
#:          always called them.
#:        * amo's hands move out of its combined ``action`` blob (where they
#:          were entries 0:14) into ``action.hands``, as on every other
#:          backend. Its ``action`` column therefore shrinks 36 -> 22.
#:      Why it matters beyond tidiness: an embodiment config is generated from
#:      a dataset's modality.json and then BAKED INTO the checkpoint, so a
#:      group name that differs per backend becomes a server returning a key
#:      the deploy path does not recognise -- which is silent, because an
#:      unrecognised hand key falls back to the operator's trigger and an eval
#:      has no operator. That cost 0/5 on a heft checkpoint whose grasp was
#:      correct to 2% (see Sim2RealBackend.vla_hand_targets).
#:   6  sonic only (the legacy plain-tuple schema). Brings the column set to
#:      NVlabs' own collector (gear_sonic/scripts/run_data_exporter.py) where
#:      the signal exists on this side, plus fixes two columns whose meaning
#:      differed from upstream under the same name:
#:        * ``action.wbc`` is 43 wide, not 29: the 29 body targets with the 7+7
#:          COMMANDED fingers interleaved in the state's own slots
#:          (frames.sonic_action_43) -- ``action.wbc[i]`` and
#:          ``observation.state[i]`` are now the same joint for every i, as
#:          upstream. ``teleop.{left,right}_hand_joints`` are unchanged.
#:        * ``observation.eef_state`` is FK of ``observation.state`` -- the
#:          wrist-yaw links in the PELVIS frame, quaternions WXYZ
#:          (recording.g1_fk, verified against upstream's pinocchio model to
#:          1e-15) -- on BOTH hosts, and computed at the same tick as the state.
#:          Before v6 only the sim wrote it, from robosuite: WORLD frame, XYZW,
#:          and 0-1 control tick older than the state beside it (robosuite
#:          observables sample at a substep whose phase depends on the reset;
#:          measured on one set: some episodes aligned at the same row, some
#:          one row earlier). A hardware sonic dataset gains the column.
#:        * new columns: ``teleop.smpl_pose`` (63, SMPL body pose axis-angle
#:          from the PICO stream; zeros when no SMPL is streaming -- upstream
#:          does the same), ``teleop.planner_movement`` (3) /
#:          ``teleop.planner_speed`` (1) / ``teleop.planner_height`` (1) (the
#:          kPlanner MovementState fields that were not recorded; -1 = "use
#:          mode default", as in MovementState), ``teleop.{left,right}_trigger``
#:          (1, the raw PICO trigger in [0,1] -- so ANY hand-command generator
#:          can be re-derived offline; the recorded ``*_hand_joints`` are one
#:          such derivation), ``teleop.pico_frame_index`` (int64: how many
#:          distinct PICO samples the session had seen at this tick),
#:          ``observation.init_base_quat`` (4, wxyz: the pelvis quaternion at
#:          engage, OFF -> anything; constant within a session, identity if
#:          never engaged).
#:      Datasets collected before v6 can be brought to v6 WITHOUT re-collection
#:      except for the new signals, which are backfilled with the defaults
#:      above (the triggers are recovered from the hand command). The blob
#:      backends are unaffected.
#:   7  amo + almi (the blob backends; sonic/heft/scalebfm unchanged).
#:        * amo: the ``rpy``/``height`` STATE groups mean "the torso command in
#:          force during the previous tick" (Psi0's definition for its G1
#:          wholebody datasets -- raw_to_lerobot.py build_obs), on BOTH hosts.
#:          Before: the measured waist joints and robosuite's root-frame z,
#:          which sits 0.793 m below the physical pelvis (recorded ~-0.04 m
#:          against a 0.75 m action) -- and the real host dropped `height`
#:          entirely (observation.state 31 wide). Now 32 wide everywhere.
#:        * amo gains ``observation.root_orientation`` (4, wxyz) and
#:          ``observation.projected_gravity`` (3) -- the measured attitude,
#:          as heft/almi record. Structurally detectable (new columns).
#:        * amo/almi take their per-episode state in ``on_episode_start``
#:          (heading anchor + torso integrators; the almi recording gate +
#:          base height) instead of only in ``reset()``, which on hardware
#:          runs once per session: real datasets before this had episode N
#:          opening with episode N-1's heading offset / dialled-in height /
#:          held arm reference. Sim datasets are unaffected.
#:        * almi no longer records non-POSE (IDLE) ticks unless
#:          ``sim2real_record_idle`` is set -- the contract g1_generic.json
#:          states and heft already implemented. Datasets before this carry
#:          runs of frames with a zero twist and a frozen upper reference
#:          wherever the operator idled; they can be filtered offline
#:          (``action.twist[0:3] == 0`` with an unchanged ``action.upper_ref``).
#:   8  heft/scalebfm only: ``action.ref_qpos``'s ``ref_root_pos``/``ref_root_quat``
#:      are now the root AS THE TRACKING POLICY SEES IT -- the raw GMR root
#:      passed through the motion buffer's segment transform (yaw-only rotation
#:      + xy-only translation that re-attaches each new segment onto the pose the
#:      buffer was holding; ``RealtimeMotionBuffer.transform_root_pose``), then
#:      made episode-relative exactly as v3 defined. Before v8 the RAW stream's
#:      root was recorded: continuous within a POSE segment, but on every POSE
#:      re-entry (A+X after an IDLE pause) or XR reconnect it jumped by however
#:      far the operator had moved in the meantime, while the robot -- and the
#:      image, and the state -- stayed put; deploy (RefQposAdapter, one segment
#:      per episode) has no counterpart to that jump. Heights, roll/pitch and
#:      the 29 joints are unchanged. Datasets before v8 carry those jumps
#:      wherever the operator paused mid-episode; they are recognisable as a
#:      single-frame xy/yaw step in ref_root_pos with a static leg state, and
#:      the frame after it equal to the standing default pose.
#:   9  sonic only, and a RENUMBERING rather than a new column:
#:      ``teleop.stream_mode`` now uses upstream's StreamMode values exactly.
#:      Until v9 our ManagerMode put VR_3PT at **4**, which is upstream's
#:      POSE_PAUSE; upstream's VR 3-point mode is **5**. So a pre-v9 sonic
#:      dataset and an upstream one disagree about what a 4 means, and nothing
#:      in the column says which convention wrote it -- only this number does.
#:      Now: 0 OFF / 1 POSE / 2 PLANNER / 3 PLANNER_FROZEN_UPPER /
#:      4 POSE_PAUSE / 5 PLANNER_VR_3PT, both sides.
#:      Two behaviour changes ride along, both of which put values in the
#:      column that could not appear before: POSE_PAUSE exists at all (hold the
#:      left menu button in POSE), and VR_3PT is actually implemented -- before
#:      v9 it silently degraded to the g1 sub-encoder, so pre-v9 frames marked
#:      4 are "operator waving, robot holding its standing reference", not
#:      3-point teleop. Repairing a pre-v9 set is a value remap on one column
#:      (4 -> 5) plus dropping those frames; no re-collection is needed for the
#:      other columns.
SCHEMA_VERSION = 9

#: What ``make_dataset`` actually stamps: the LAST version whose change touched
#: THIS backend's columns. The history above is one global sequence, but its
#: entries are per-backend; stamping the global head on every dataset would
#: make a sonic set collected today (8) refuse to merge (prepare's
#: gate_schema_version) with a sonic set repaired to v6 yesterday, although
#: not one sonic column changed in v7/v8. Same backend + same number here ==
#: same column semantics, which is exactly the gate's question.
BACKEND_SCHEMA_VERSION = {
    "sonic": 9,
    "amo": 7,
    "almi": 7,
    "heft": 8,
    "scalebfm": 8,
}


def schema_version_for(backend) -> int:
    """The schema_version a dataset of ``backend`` is stamped with."""
    return int(BACKEND_SCHEMA_VERSION.get(backend, SCHEMA_VERSION))

#: Sim render size. The simulator draws at this resolution, so it is what a
#: `--sim` dataset carries.
EGO_H, EGO_W = 240, 320

#: What the real collector stores by default: the G1 camera's own resolution, so
#: hardware datasets keep every pixel the sensor produced and nothing is thrown
#: away before anyone has decided it should be. Same 4:3 aspect as the sim, so a
#: model that resizes sees the same framing either way; a config that crops at a
#: fixed pixel size does not, and has to be checked before mixing the two.
REAL_EGO_H, REAL_EGO_W = 480, 640

VIDEO_INFO = {
    "video.fps": FPS,
    "video.codec": "h264",
    "video.pix_fmt": "yuv420p",
    "video.is_depth_map": False,
    "has_audio": False,
}


def ego_view_feature(height: int = EGO_H, width: int = EGO_W, fps: int = FPS) -> dict:
    """The ego camera feature. Stored as VIDEO only.

    LeRobot buffers each frame as a temporary PNG under ``<root>/images/`` and
    encodes them into one h264 mp4 per episode on save, deleting that episode's
    PNG dir afterwards. Nothing should survive in ``images/`` -- see
    ``purge_image_cache``, which sweeps what an aborted episode leaves behind.
    """
    return {
        "dtype": "video",
        "shape": (height, width, 3),
        "names": ["height", "width", "channel"],
        "video_info": {**VIDEO_INFO, "video.fps": fps},
    }


#: Columns only a simulator can fill. The real collector omits them; every
#: consumer that matters keys off the state/action blobs instead.
SIM_ONLY_FEATURES = {
    "observation.eef_state": {"dtype": "float64", "shape": (14,)},
    "next.reward": {"dtype": "float32", "shape": (1,)},
    "next.done": {"dtype": "bool", "shape": (1,)},
}

#: Legacy (SONIC-family) fields, used by the plain-tuple / TELEOP_FEATURE_SPECS
#: schema mode below. ModalityLayout-mode controllers (amo, heft/scalebfm) build
#: their own observation.state equivalent as part of their state blob instead --
#: see ModalityBlob.
LEGACY_BASE_FEATURES = {
    "observation.state": {"dtype": "float64", "shape": (43,)},
    # FK of observation.state (pelvis frame, wxyz) -- both hosts, schema v6+.
    # Same spec as SIM_ONLY_FEATURES' entry on purpose: the sim host still
    # passes that dict as extra_features, and build_features merges by name.
    "observation.eef_state": {"dtype": "float64", "shape": (14,)},
    "observation.root_orientation": {"dtype": "float64", "shape": (4,)},
    "observation.projected_gravity": {"dtype": "float64", "shape": (3,)},
    "annotation.human.task_description": {"dtype": "int64", "shape": (1,)},
}

#: One entry per key a controller's ``get_teleop_data_sample()`` may return:
#: sample key -> (LeRobot feature name, feature spec). Which subset a given
#: backend actually populates is declared by its entry in
#: ``G1_BACKEND_TELEOP_KEYS``, so a controller that can't produce a signal simply
#: doesn't get that column, instead of a column silently full of zeros. This is
#: the legacy plain-tuple mode, still used by sonic.
TELEOP_FEATURE_SPECS = {
    # schema v6: 43 wide (body 29 + commanded fingers in the state's slots), see
    # frames.sonic_action_43. The pre-v6 key was "target_q_29" -> (29,).
    "target_q_43": ("action.wbc", {"dtype": "float64", "shape": (43,)}),
    "motion_token": ("action.motion_token", {"dtype": "float64", "shape": (64,)}),
    "left_hand_joints": (
        "teleop.left_hand_joints",
        {"dtype": "float32", "shape": (7,)},
    ),
    "right_hand_joints": (
        "teleop.right_hand_joints",
        {"dtype": "float32", "shape": (7,)},
    ),
    "stream_mode": ("teleop.stream_mode", {"dtype": "float32", "shape": (1,)}),
    "locomotion_mode": ("teleop.locomotion_mode", {"dtype": "float32", "shape": (1,)}),
    "facing_direction": (
        "teleop.facing_direction",
        {"dtype": "float32", "shape": (3,)},
    ),
    "smpl_joints_local": ("teleop.smpl_joints", {"dtype": "float32", "shape": (72,)}),
    "smpl_root_quat": ("teleop.smpl_root_quat", {"dtype": "float32", "shape": (4,)}),
    "wrist_joints": ("teleop.wrist_joints", {"dtype": "float32", "shape": (6,)}),
    "vr_3point_pos": (
        "teleop.vr_3pt_position",
        {"dtype": "float32", "shape": (9,)},
    ),
    "vr_3point_orn": (
        "teleop.vr_3pt_orientation",
        {"dtype": "float32", "shape": (12,)},
    ),
    # --- schema v6 additions (see SCHEMA_VERSION bullet 6) ---
    "smpl_pose": ("teleop.smpl_pose", {"dtype": "float32", "shape": (63,)}),
    "planner_movement": ("teleop.planner_movement", {"dtype": "float32", "shape": (3,)}),
    "planner_speed": ("teleop.planner_speed", {"dtype": "float32", "shape": (1,)}),
    "planner_height": ("teleop.planner_height", {"dtype": "float32", "shape": (1,)}),
    "left_trigger": ("teleop.left_trigger", {"dtype": "float32", "shape": (1,)}),
    "right_trigger": ("teleop.right_trigger", {"dtype": "float32", "shape": (1,)}),
    "pico_frame_index": ("teleop.pico_frame_index", {"dtype": "int64", "shape": (1,)}),
    "init_base_quat": ("observation.init_base_quat", {"dtype": "float64", "shape": (4,)}),
}


def build_features(
    teleop_keys,
    *,
    ego_hw: tuple[int, int] = (EGO_H, EGO_W),
    fps: int = FPS,
    extra_features: dict | None = None,
) -> tuple[dict, object | None]:
    """``(features, modality_blobs_or_None)`` for one run's controller.

    ``teleop_keys`` is either a ``ModalityLayout`` (blob mode: each blob becomes
    one concatenated LeRobot column, and a real modality.json is generated from
    the blobs' group offsets) or a plain tuple (legacy mode: one column per
    signal, modality.json is a static copied asset -- the caller handles that).
    """
    from g1_bridge.recording.schema import ModalityLayout, blob_features

    features = {"observation.images.ego_view": ego_view_feature(*ego_hw, fps=fps)}
    features.update(dict(extra_features or {}))
    if isinstance(teleop_keys, ModalityLayout):
        features.update(blob_features(teleop_keys.blobs))
        return features, teleop_keys.blobs
    features.update({k: dict(v) for k, v in LEGACY_BASE_FEATURES.items()})
    for key in teleop_keys:
        feature_name, spec = TELEOP_FEATURE_SPECS[key]
        features[feature_name] = dict(spec)
    return features, None
