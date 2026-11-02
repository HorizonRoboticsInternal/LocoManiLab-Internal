"""MuJoCo-side primitives for the G1 composite controllers.

The sim half of what used to be one ``g1_bridge/g1_runtime.py``: every function
here takes a ``sim`` and reads or writes ``sim.data`` directly, so it can only
run inside robosuite/MuJoCo. It stayed behind when the host-agnostic half of
``g1_bridge`` moved into the g1_backend distribution (see README.md). A
real-robot host implements the same *reads* against the robot's own state stream
instead, which is exactly why these could not travel with it.

Sim-free helpers that used to sit alongside these (``G1_JOINT_NAMES``, the
quaternion math, ``resolve_ckpt_path``, ``policy_due``, ``is_real_pico_source``)
are still in ``g1_bridge.g1_runtime`` and are imported from there.

Everything is a free function over ``(sim, ...)`` -- no controller state, no
``self`` -- so the composites keep owning their own fields and just delegate.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Sequence

import numpy as np

from g1_bridge.g1_runtime import G1_JOINT_NAMES


# --------------------------------------------------------------------------- #
# Joint / freejoint address mapping
# --------------------------------------------------------------------------- #
class JointMapping(NamedTuple):
    """MuJoCo addresses for one ordered list of joint names.

    Every list is the same length as the requested names, with ``None`` in the
    slots whose joint or actuator this particular model does not carry -- the
    G1 variants differ in which joints they expose, so callers must tolerate
    holes rather than assume all 29 are present.
    """

    qpos_addrs: List[Optional[int]]
    qvel_addrs: List[Optional[int]]
    actuator_ids: List[Optional[int]]
    present: np.ndarray  # bool mask: joint AND actuator both exist


def build_joint_mapping(sim, robot_model, joint_names: Sequence[str] = G1_JOINT_NAMES):
    """Resolve qpos/qvel addresses and actuator ids for ``joint_names``.

    Actuator naming is not consistent across the G1 MJCFs, so each joint is
    looked up first without the ``_joint`` suffix and then with it.
    """
    prefix = robot_model.naming_prefix
    qpos_addrs, qvel_addrs, act_ids, present = [], [], [], []
    model_joint_names = set(sim.model.joint_names)
    actuator_names = set(sim.model.actuator_names)
    for jname in joint_names:
        full_joint = prefix + jname
        full_act = None
        for candidate in (prefix + jname.replace("_joint", ""), prefix + jname):
            if candidate in actuator_names:
                full_act = candidate
                break
        if full_joint in model_joint_names:
            qpos_addrs.append(sim.model.get_joint_qpos_addr(full_joint))
            qvel_addrs.append(sim.model.get_joint_qvel_addr(full_joint))
        else:
            qpos_addrs.append(None)
            qvel_addrs.append(None)
        act_ids.append(sim.model.actuator_name2id(full_act) if full_act is not None else None)
        present.append(full_joint in model_joint_names and full_act is not None)
    return JointMapping(qpos_addrs, qvel_addrs, act_ids, np.array(present, dtype=bool))


def build_freejoint_mapping(sim, robot_model):
    """Locate the freejoint on the robot's root body.

    Returns ``(qpos_addr, qvel_addr)``, or ``(None, None)`` when the robot is
    mounted on a fixed base (no freejoint) or the root body is absent.
    """
    prefix = robot_model.naming_prefix
    root_body = prefix + robot_model._root_body
    try:
        root_id = sim.model.body_name2id(root_body)
    except (ValueError, KeyError):
        return None, None
    for j_id in range(sim.model.njnt):
        # mjJNT_FREE == 0
        if sim.model.jnt_type[j_id] == 0 and sim.model.jnt_bodyid[j_id] == root_id:
            return int(sim.model.jnt_qposadr[j_id]), int(sim.model.jnt_dofadr[j_id])
    return None, None


# --------------------------------------------------------------------------- #
# Sim state readers
# --------------------------------------------------------------------------- #
def read_joint_state(sim, qpos_addrs, qvel_addrs):
    """Body joint positions/velocities in the mapping's order; 0.0 for holes."""
    n = len(qpos_addrs)
    qpos = np.zeros(n, dtype=np.float32)
    qvel = np.zeros(n, dtype=np.float32)
    for i, (qp, qv) in enumerate(zip(qpos_addrs, qvel_addrs)):
        if qp is not None:
            qpos[i] = sim.data.qpos[qp]
            qvel[i] = sim.data.qvel[qv]
    return qpos, qvel


def read_pelvis_state(sim, free_qpos_addr, free_qvel_addr):
    """Return (quaternion[wxyz], body-local angular velocity)."""
    if free_qpos_addr is not None:
        quat = np.array(sim.data.qpos[free_qpos_addr + 3 : free_qpos_addr + 7], dtype=np.float32)
        # MuJoCo's freejoint qvel layout: [linear_world(3), angular_local(3)]
        ang_vel = np.array(sim.data.qvel[free_qvel_addr + 3 : free_qvel_addr + 6], dtype=np.float32)
        # Guard against degenerate quat.
        n = np.linalg.norm(quat)
        quat = quat / n if n > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return quat, ang_vel
    return (
        np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )


def read_root_pos(sim, free_qpos_addr):
    if free_qpos_addr is not None:
        return np.array(sim.data.qpos[free_qpos_addr : free_qpos_addr + 3], dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def read_body_pose_w(sim, robot_model, body_name):
    """World position + normalized wxyz quaternion of a named MuJoCo body (this
    robot's own naming_prefix applied) -- read straight off ``sim.data.xpos``/
    ``xquat``, which MuJoCo already populates via forward kinematics every step,
    so this is exact (not an approximation) for any body in the model, not just
    the pelvis free joint ``read_pelvis_state`` covers."""
    full_name = robot_model.naming_prefix + body_name
    body_id = sim.model.body_name2id(full_name)
    pos = np.array(sim.data.xpos[body_id], dtype=np.float32)
    quat = np.array(sim.data.xquat[body_id], dtype=np.float32)
    n = np.linalg.norm(quat)
    quat = quat / n if n > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return pos, quat


#: Bodies whose geoms count as "the foot" for ground-contact detection. Not
#: every G1 MJCF models toe links (g1-mode_13_15.xml does not), so both are
#: looked up leniently.
_FOOT_BODY_SUFFIXES = ("ankle_roll_link", "toe_link")


def build_foot_geom_ids(sim, robot_model):
    """Geom ids belonging to this robot's feet, for ``feet_in_contact``.

    Rebuilt per ``env.reset()`` alongside the other address maps -- geom ids are
    model-relative and robocasa hands out a genuinely new model every reset.
    """
    prefix = robot_model.naming_prefix
    body_ids = set()
    for side in ("left", "right"):
        for suffix in _FOOT_BODY_SUFFIXES:
            try:
                body_ids.add(sim.model.body_name2id(f"{prefix}{side}_{suffix}"))
            except (ValueError, KeyError):
                continue
    if not body_ids:
        return frozenset()
    return frozenset(g for g in range(sim.model.ngeom) if sim.model.geom_bodyid[g] in body_ids)


def feet_in_contact(sim, foot_geom_ids) -> bool:
    """True while at least one foot geom is in contact with anything.

    Robocasa spawns the G1 slightly above the floor, so it free-falls for the
    first few control steps of every episode -- measured at ~13 cm / ~9.1 m/s^2,
    with zero foot contacts throughout. Anything that wants to capture the
    robot's settled standing pose has to wait for this.

    Returns True when the foot geoms could not be resolved at all, so a G1
    variant this doesn't recognise degrades to "don't block" rather than
    hanging forever.
    """
    if not foot_geom_ids:
        return True
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        if contact.geom1 in foot_geom_ids or contact.geom2 in foot_geom_ids:
            return True
    return False


# --------------------------------------------------------------------------- #
# Actuation
# --------------------------------------------------------------------------- #
def write_body_torques(sim, actuator_ids, target_q, kps, kds, qpos, qvel):
    """PD-control the body joints: ``tau = kp*(q* - q) - kd*dq``.

    Writes ``sim.data.ctrl`` directly (these composites bypass robosuite's
    part-controller machinery entirely) and skips joints with no actuator.
    """
    torque = kps * (target_q - qpos) - kds * qvel
    for i, act_id in enumerate(actuator_ids):
        if act_id is not None:
            sim.data.ctrl[act_id] = float(torque[i])
    return torque


