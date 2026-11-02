"""Dex3 finger PD control for G1BridgeComposite.

Structurally the same trigger-to-curl PD loop as
``wbt_g1_composite.WholeBodyG1Composite._write_gripper_torques`` /
``_build_hand_mapping``, but sourcing the close-pose targets and PD gains
from ``sim2real.utils.dex3`` (heft's real-hardware-tuned values) instead of
tele's vendored joint-range-derived ``_HAND_LEFT_CLOSE_TARGETS`` /
``_HAND_RIGHT_CLOSE_TARGETS`` -- so a sim2real-deployed policy and
sim2real-collected teleop data agree on what "closed" means. Both repos share
byte-identical G1+hand MJCF (verified), so the joint names below need no
reindexing against tele's actuator/joint layout.
"""

from __future__ import annotations

import numpy as np

from sim2real.utils.dex3 import DEX3_JOINT_NAMES, compute_pd_torques, grip_to_joint_targets

_HAND_LEFT_NAMES = tuple(f"left_hand_{name}" for name in DEX3_JOINT_NAMES)
_HAND_RIGHT_NAMES = tuple(f"right_hand_{name}" for name in DEX3_JOINT_NAMES)


def build_hand_mapping(sim, robot_model) -> dict:
    """Look up hand actuator + joint qpos/qvel addresses once.

    Returns ``{"left": {"act_ids": [...], "qpos": [...], "qvel": [...]},
    "right": {...}}``; entries are ``None`` for any joint/actuator absent
    from the loaded MJCF (e.g. a robot variant without hands).
    """
    prefix = robot_model.naming_prefix
    mapping = {"left": {"act_ids": [], "qpos": [], "qvel": []}, "right": {"act_ids": [], "qpos": [], "qvel": []}}
    for side, names in (("left", _HAND_LEFT_NAMES), ("right", _HAND_RIGHT_NAMES)):
        store = mapping[side]
        for name in names:
            act_id = None
            for full_act in (prefix + name, prefix + name + "_joint"):
                try:
                    act_id = sim.model.actuator_name2id(full_act)
                    break
                except Exception:
                    pass
            qp_addr, qv_addr = None, None
            try:
                jid = sim.model.joint_name2id(prefix + name + "_joint")
                qp_addr = sim.model.jnt_qposadr[jid]
                qv_addr = sim.model.jnt_dofadr[jid]
            except Exception:
                pass
            store["act_ids"].append(act_id)
            store["qpos"].append(qp_addr)
            store["qvel"].append(qv_addr)
    return mapping


def write_gripper_torques(sim, hand_mapping: dict, left_command, right_command) -> None:
    """PD-drive both hands to a target and write torques into ``sim.data.ctrl``.

    Each command is either a scalar grip in [0, 1] -- the operator's trigger,
    mapped through ``DEX3_CLOSE_POSE`` -- or the seven joint angles themselves.
    A policy predicts the joint angles, and a grip scalar cannot represent what
    it predicts: one number can only slide along a single open-to-closed pose.

    No-op for a side whose mapping has no resolved actuators (older config /
    hand-less robot variant).
    """
    for side, command in (("left", left_command), ("right", right_command)):
        addrs = hand_mapping[side]
        if not any(a is not None for a in addrs["act_ids"]):
            continue
        command = np.asarray(command, dtype=np.float32).reshape(-1)
        if command.size == 1:
            q_target = grip_to_joint_targets(float(np.clip(command[0], 0.0, 1.0)), side)
        else:
            q_target = command
        q_measured = np.array(
            [sim.data.qpos[a] if a is not None else 0.0 for a in addrs["qpos"]],
            dtype=np.float32,
        )
        qvel_measured = np.array(
            [sim.data.qvel[a] if a is not None else 0.0 for a in addrs["qvel"]],
            dtype=np.float32,
        )
        torques = compute_pd_torques(q_target, q_measured, qvel_measured)
        for act_id, torque in zip(addrs["act_ids"], torques):
            if act_id is not None:
                sim.data.ctrl[act_id] = float(torque)
