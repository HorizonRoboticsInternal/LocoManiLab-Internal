"""Assembling one LeRobot frame from a controller's teleop sample -- host-agnostic.

The host supplies only what it alone can observe: the ego image, the robot's root
orientation, and (in sim) the extra columns of ``SIM_ONLY_FEATURES``. Everything
else in a frame comes from ``session.teleop_sample()``, which is identical code on
both hosts -- which is exactly why a real dataset and a sim dataset are the same
thing to a trainer.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from g1_bridge.recording.schema import (
    ModalityLayout,
    build_blob_frame_fields,
)


def projected_gravity(base_quat_wxyz):
    """Gravity direction expressed in the robot base frame (proprio signal)."""
    r = sRot.from_quat(np.asarray(base_quat_wxyz)[[1, 2, 3, 0]])  # wxyz -> xyzw
    return r.inv().apply([0.0, 0.0, -1.0]).astype(np.float64)


def sonic_state_43(host):
    """The legacy/SONIC schema's ``observation.state``: 29 body joints with the
    7+7 measured finger joints INTERLEAVED between the arms.

    Layout, matching ``assets/G1_modality.json`` exactly::

        0:15   legs (12) + waist (3)      canonical 29-order 0:15
        15:22  left arm                   canonical 29-order 15:22
        22:29  left hand                  Dex3 order, read_hand_state()["left"]
        29:36  right arm                  canonical 29-order 22:29
        36:43  right hand                 Dex3 order, read_hand_state()["right"]

    Both hosts build it from the same two host reads -- ``read_joint_state()``
    (29, ``G1_JOINT_NAMES`` order) and ``read_hand_state()`` (7 per side, in
    ``DEX3_JOINT_NAMES`` order) -- which is what makes a real and a sim sonic
    dataset the same 43 columns. That was not true before 2026-08-11, in two
    independent ways, and both are silent (the shapes always matched):

    * the real collector concatenated ``[body29, left7, right7]``, so its
      ``left_hand`` block (22:29) actually held the RIGHT ARM and its
      ``right_arm`` block (29:36) held the left hand.
    * the sim collector passed ``obs["robot0_joint_pos"]`` straight through.
      That is MJCF body-tree order, in which each hand's joints run
      thumb_0,1,2, **middle_0,1, index_0,1** -- whereas ``read_hand_state()``,
      ``grip_to_joint_targets`` and the Dex3 driver all use thumb_0,1,2,
      **index_0,1, middle_0,1**. So sonic's own ``teleop.*_hand_joints``
      columns disagreed with the hand blocks of the state vector beside them.

    Reading the fingers through ``read_hand_state()`` on both hosts fixes the
    second one by construction: there is now one finger ordering in the whole
    dataset, and it is the hardware's.
    """
    qpos, _qvel = host.read_joint_state()
    hands = host.read_hand_state() or {}
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)

    def finger(side):
        v = hands.get(side)
        # No hands attached (--dex3 none) is a legitimate configuration.
        return np.zeros(7) if v is None else np.asarray(v, dtype=np.float64).reshape(-1)

    return np.concatenate([q[0:22], finger("left"), q[22:29], finger("right")])


def sonic_action_43(target_q29, left_hand7, right_hand7):
    """The legacy/SONIC schema's ``action.wbc`` (schema_version >= 6): the 29
    commanded body joints with the 7+7 COMMANDED finger targets interleaved in
    exactly the slots ``sonic_state_43`` gives the measured fingers, so
    ``action.wbc[i]`` and ``observation.state[i]`` are the same joint for every
    ``i`` -- the invariant NVlabs' collector has (its ``action.wbc`` is
    ``last_action`` + ``last_{left,right}_hand_action`` assembled through the
    same 43-name robot model as the state).

    Until schema v5 this column was the 29 body targets alone; the hand command
    lived only in ``teleop.{left,right}_hand_joints`` (it still does -- those
    columns are unchanged, the 43-wide ``wbc`` just carries them too).

    A missing hand (``--dex3 none``, or a tick before the trigger is read) is
    zero-filled, like every other absent signal.
    """
    q = np.asarray(target_q29, dtype=np.float64).reshape(-1)
    if q.shape[0] != 29:
        raise ValueError(f"target_q29 must have 29 entries, got {q.shape[0]}")

    def hand(v):
        return np.zeros(7) if v is None else np.asarray(v, dtype=np.float64).reshape(7)

    return np.concatenate([q[0:22], hand(left_hand7), q[22:29], hand(right_hand7)])


#: Quaternion (wxyz) written for ``observation.init_base_quat`` when the
#: controller has not captured one (no engage yet): identity, not zeros -- a
#: zero quaternion is not a rotation and would poison any consumer that
#: normalises/slerps it.
IDENTITY_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def build_blob_frame(
    layout: ModalityLayout,
    teleop: dict,
    *,
    image,
    root_quat,
    extra_fields: dict | None = None,
    require_ready: bool = True,
):
    """One frame in blob mode (heft/scalebfm/amo), or None to skip this tick.

    Returns None while the controller's readiness sentinel is still None (before
    its first real command), so the caller can skip the frame -- unless
    ``require_ready=False`` (plumbing-only checks), in which case a still-missing
    readiness signal is zero-filled like any other absent key instead of skipping
    the frame: that proves the controller constructs and steps without crashing
    even when no real reference (GMR payload, live PICO data, ...) has arrived --
    a separate concern from whether the recording plumbing itself works.
    """
    if require_ready and teleop.get(layout.readiness_source) is None:
        return None
    values = dict(teleop)
    values["root_orientation"] = np.asarray(root_quat).astype(np.float32)
    values["projected_gravity"] = projected_gravity(root_quat).astype(np.float32)
    frame = {"observation.images.ego_view": np.ascontiguousarray(image)}
    frame.update(extra_fields or {})
    frame.update(build_blob_frame_fields(values, layout.blobs))
    return frame


def build_legacy_frame(
    teleop_keys,
    teleop: dict,
    *,
    image,
    state,
    root_quat,
    task_idx: int = 0,
    extra_fields: dict | None = None,
    require_ready: bool = True,
    feature_specs=None,
):
    """One frame in legacy plain-tuple mode (the SONIC family).

    ``state`` is that mode's 43-DoF ``observation.state`` (29 body + 14 hand),
    which the host reads directly rather than getting from the teleop sample.

    ``observation.eef_state`` is derived from ``state`` here (schema_version >= 6):
    FK of the wrist-yaw links in the pelvis frame, wxyz -- see ``g1_fk``. It
    OVERRIDES whatever the host put in ``extra_fields`` under that name (the sim
    host's robosuite world-frame / xyzw value, which is what sonic datasets
    carried before v6), so both hosts record one definition, and it is the one
    NVlabs' collector uses.
    """
    from g1_bridge.recording.features import TELEOP_FEATURE_SPECS
    from g1_bridge.recording.g1_fk import eef_state_from_state43

    specs = feature_specs or TELEOP_FEATURE_SPECS
    if require_ready and teleop.get(teleop_keys[0]) is None:
        return None
    state = np.asarray(state).astype(np.float64)
    frame = {
        "observation.images.ego_view": np.ascontiguousarray(image),
        "observation.state": state,
        "observation.root_orientation": np.asarray(root_quat).astype(np.float64),
        "observation.projected_gravity": projected_gravity(root_quat),
        "annotation.human.task_description": np.array([task_idx], dtype=np.int64),
    }
    frame.update(extra_fields or {})
    frame["observation.eef_state"] = eef_state_from_state43(state)
    for key in teleop_keys:
        feature_name, spec = specs[key]
        v = teleop.get(key)
        if v is not None:
            frame[feature_name] = np.asarray(v).astype(spec["dtype"]).reshape(spec["shape"])
        elif feature_name == "observation.init_base_quat":
            frame[feature_name] = IDENTITY_QUAT_WXYZ.copy()
        else:
            frame[feature_name] = np.zeros(spec["shape"], dtype=spec["dtype"])
    return frame
