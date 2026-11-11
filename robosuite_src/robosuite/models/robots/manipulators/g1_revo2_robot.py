"""G1 with Revo2 hands and the calibrated rot0 camera/backpack assembly."""
import json
from pathlib import Path
import numpy as np
from .g1_robot import G1FloatingBody
from robosuite.utils.mjcf_utils import xml_path_completion


class G1Revo2(G1FloatingBody):
    """51 actuated joints; independent body/hand name mappings at the host."""
    xml_path = 'robots/g1_revo2/robot.xml'
    hand_type = 'revo2'

    @property
    def init_qpos(self):
        poses = json.loads((Path(xml_path_completion(self.xml_path)).parent / 'grip_poses.json').read_text())
        finger_names = ('thumb_metacarpal', 'thumb_proximal', 'thumb_distal', 'index_proximal',
                        'index_distal', 'middle_proximal', 'middle_distal', 'ring_proximal',
                        'ring_distal', 'pinky_proximal', 'pinky_distal')
        values = {}
        for side in ('left', 'right'):
            values.update({f'{side}_hand_{n}_joint': q for n, q in zip(finger_names, poses['open'])})
            values.update({f'{side}_hip_pitch_joint': -.312, f'{side}_knee_joint': .669,
                           f'{side}_ankle_pitch_joint': -.363, f'{side}_shoulder_pitch_joint': .6,
                           f'{side}_shoulder_roll_joint': .12 if side == 'left' else -.12})
        return np.array([values.get(j.removeprefix(self.naming_prefix), 0.) for j in self.joints])

    @property
    def _eef_name(self):
        return {'left': 'left_hand_base_link', 'right': 'right_hand_base_link'}

    @property
    def physical_root_body(self):
        # add_mobile_base moves the physical pelvis into this fixed mount;
        # the freejoint frame is at floor height, unlike SIMPLE's pelvis root.
        return "manipulator_mount" if self.base is not None else self.naming_prefix + "pelvis"
