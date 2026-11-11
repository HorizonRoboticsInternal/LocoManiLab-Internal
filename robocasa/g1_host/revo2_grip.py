"""Revo2 hand PD, retaining SIMPLE's named joint order and calibrated poses."""
import json
from pathlib import Path
import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion

FINGERS = ('thumb_metacarpal', 'thumb_proximal', 'thumb_distal', 'index_proximal',
           'index_distal', 'middle_proximal', 'middle_distal', 'ring_proximal',
           'ring_distal', 'pinky_proximal', 'pinky_distal')
KP = np.array([2., 3., 1.5, 3., 1.5, 3., 1.5, 3., 1.5, 3., 1.5])
KD = .3


def build_hand_mapping(sim, robot_model):
    mapping = {}
    poses = json.loads((Path(xml_path_completion(robot_model.xml_path)).parent / 'grip_poses.json').read_text())
    for side in ('left', 'right'):
        names = [robot_model.naming_prefix + f'{side}_hand_{finger}_joint' for finger in FINGERS]
        joints = [sim.model.joint_name2id(n) for n in names]
        acts = [sim.model.actuator_name2id(n) for n in names]
        mapping[side] = dict(qpos=sim.model.jnt_qposadr[joints], qvel=sim.model.jnt_dofadr[joints],
                             act_ids=acts, ranges=sim.model.jnt_range[joints].copy(), poses=poses)
    return mapping


def write_gripper_torques(sim, mapping, left_command, right_command):
    for side, command in (('left', left_command), ('right', right_command)):
        m = mapping[side]
        q = np.asarray(command, dtype=float).reshape(-1)
        if q.size == 1:
            q = np.asarray(m['poses']['closed' if q[0] > m['poses'].get('threshold', .5) else 'open'])
        if q.shape != (11,) or not np.isfinite(q).all():
            raise ValueError('Revo2 hand target must contain 11 finite joint angles')
        q = np.clip(q, m['ranges'][:, 0], m['ranges'][:, 1])
        torque = KP * (q - sim.data.qpos[m['qpos']]) - KD * sim.data.qvel[m['qvel']]
        limits = sim.model.actuator_ctrlrange[m['act_ids']]
        sim.data.ctrl[m['act_ids']] = np.clip(torque, limits[:, 0], limits[:, 1])
