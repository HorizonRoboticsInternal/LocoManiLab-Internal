import numpy as np

from robosuite.models.robots.manipulators.legged_manipulator_model import (
    LeggedManipulatorModel,
)
from robosuite.utils.mjcf_utils import xml_path_completion


# G1's MJCF joint name groups. Used to override the substring-based
# classification in ManipulatorModel.update_{joints,actuators}, which keys off
# "torso" / "head" / "leg" / "mobile" — none of which appear in G1's names.
_G1_LEG_KEYS = ("_hip_", "_knee_", "_ankle_")
_G1_TORSO_KEYS = ("waist_",)
_G1_HEAD_KEYS = ()  # G1's head_link has no actuated joint


def _is_leg(name):
    return any(k in name for k in _G1_LEG_KEYS)


def _is_torso(name):
    return any(name.startswith(k) or ("_" + k) in name for k in _G1_TORSO_KEYS)


class G1(LeggedManipulatorModel):
    """Unitree G1 humanoid (29 body DoF + 14 hand DoF, with hands from g1_29dof_with_hand.xml)."""

    arms = ["right", "left"]
    xml_path = "robots/g1/robot.xml"

    def __init__(self, idn=0):
        super().__init__(xml_path_completion(self.xml_path), idn=idn)

    # ------------------------------------------------------------------ #
    # Override joint/actuator classification — G1's names don't contain
    # the "leg"/"torso"/"head" substrings the base class greps for.
    # ------------------------------------------------------------------ #
    def update_joints(self):
        for joint in self.all_joints:
            bare = joint.replace(self.naming_prefix, "")
            if _is_leg(bare):
                self._legs_joints.append(joint)
            elif _is_torso(bare):
                self._torso_joints.append(joint)
            else:
                self._arms_joints.append(joint)

    def update_actuators(self):
        for actuator in self.all_actuators:
            bare = actuator.replace(self.naming_prefix, "")
            if _is_leg(bare):
                self._legs_actuators.append(actuator)
            elif _is_torso(bare):
                self._torso_actuators.append(actuator)
            else:
                self._arms_actuators.append(actuator)

    @property
    def default_base(self):
        return "FloatingLeggedBase"

    @property
    def default_gripper(self):
        # G1's hands are baked into the MJCF as multi-finger actuators.
        # No separate gripper to attach — use NullGripper via None.
        return {"right": None, "left": None}

    @property
    def default_controller_config(self):
        # All G1 variants share default_g1.json (a composite config)
        # via _get_robot_name("G1*") -> "g1".
        return {
            "right": "default_g1",
            "left": "default_g1",
            "torso": "default_g1",
        }

    @property
    def init_qpos(self):
        # robosuite excludes <freejoint> from joint enumeration, so this array
        # covers only the 43 regular joints: 12 legs + 3 waist + 14 arms +
        # 14 hands, in MJCF body-tree order.  (Subclasses that strip leg
        # joints override this with a shorter array.)
        init = np.zeros(43, dtype=np.float64)
        # leg defaults:
        init[0] = -0.312
        init[3] = 0.669
        init[4] = -0.363
        init[6] = -0.312
        init[9] = 0.669
        init[10] = -0.363
        # waist (12..14) zero
        # arms (15..28):
        init[15] = 0.2
        init[16] = 0.2
        init[18] = 0.6
        init[22] = 0.2
        init[23] = -0.2
        init[25] = 0.6
        return init

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.30, -0.1, 0.0),
            "empty": (-0.29, 0, 0.0),
            "table": lambda table_length: (-0.15 - table_length / 2, 0, 0.0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "bimanual"

    @property
    def _eef_name(self):
        # Last arm link before the hand. NullGripper attaches here.
        return {"right": "right_wrist_yaw_link", "left": "left_wrist_yaw_link"}


class G1FixedLowerBody(G1):
    """G1 with legs locked and free joint removed. Stands rigidly at base_xpos."""

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        # Strip the lower body. _remove_joint_actuation does substring match.
        self._remove_joint_actuation("hip")
        self._remove_joint_actuation("knee")
        self._remove_joint_actuation("ankle")
        self._remove_free_joint()

    @property
    def bottom_offset(self):
        # add_mobile_base (NoActuationBase) already inserts a 'support' body at
        # +_base_offset above pelvis, so manipulator_mount ends up at the
        # standing pelvis height.  Setting bottom_offset to zero makes
        # set_base_xpos(anchor) put pelvis exactly at anchor — and the support
        # offset alone supplies the standing height.
        import numpy as np

        return np.zeros(3)

    @property
    def init_qpos(self):
        # After removal: 3 waist + 7 left arm + 7 left hand + 7 right arm + 7 right hand = 31
        # MJCF body-tree order (depth-first):
        #   waist_yaw, waist_roll, waist_pitch,
        #   left_shoulder_pitch, _roll, _yaw, left_elbow,
        #     left_wrist_roll, _pitch, _yaw,
        #     left_hand_thumb_0, _1, _2, left_hand_middle_0, _1, left_hand_index_0, _1,
        #   right_shoulder_pitch, _roll, _yaw, right_elbow,
        #     right_wrist_roll, _pitch, _yaw,
        #     right_hand_thumb_0, _1, _2, right_hand_index_0, _1, right_hand_middle_0, _1
        init = np.zeros(31, dtype=np.float64)
        # waist all zero
        # left shoulder/elbow
        init[3] = 0.2  # left_shoulder_pitch
        init[4] = 0.2  # left_shoulder_roll
        init[6] = 0.6  # left_elbow
        # right shoulder/elbow (index offset = 3 waist + 7 left arm + 7 left hand = 17)
        init[17] = 0.2  # right_shoulder_pitch
        init[18] = -0.2  # right_shoulder_roll
        init[20] = 0.6  # right_elbow
        return init

    @property
    def default_base(self):
        return "NoActuationBase"

    @property
    def base_xpos_offset(self):
        # FixedLowerBody pelvis sits at z ≈ 0.793 (set by the body pos in robot.xml),
        # so offset z=0; floating-base placement adds extra clearance instead.
        return {
            "bins": (-0.30, -0.1, 0.0),
            "empty": (-0.29, 0, 0.0),
            "table": lambda table_length: (-0.15 - table_length / 2, 0, 0.0),
        }


class G1FloatingBody(G1):
    """G1 with everything intact: freejoint + 29 body actuators + 14 hand actuators.

    Intended for whole-body controllers (Sonic / Gentle / AMO / WBT) that need
    leg control to maintain balance.  Without a balance-aware policy, the robot
    will fall over.
    """

    def __init__(self, idn=0):
        super().__init__(idn=idn)
        # No removal — legs stay actuated, freejoint stays.

    @property
    def init_qpos(self):
        # robosuite's _ref_joint_pos_indexes ignores <freejoint> elements; the
        # free joint's initial pose comes from set_robot_base instead.  So this
        # array only covers the 43 regular joints: 12 legs + 3 waist + 14 arms
        # + 14 hands, in MJCF body-tree order.
        init = np.zeros(43, dtype=np.float64)
        # leg defaults (SONIC_DEFAULT_ANGLES_MUJOCO indices 0..11):
        init[0] = -0.312  # left_hip_pitch
        init[3] = 0.669  # left_knee
        init[4] = -0.363  # left_ankle_pitch
        init[6] = -0.312  # right_hip_pitch
        init[9] = 0.669  # right_knee
        init[10] = -0.363  # right_ankle_pitch
        # Low "arms at the sides" rest pose. shoulder_pitch=0 does NOT drop the
        # arms vertically in this joint convention (it leaves them bent forward at
        # belly height); a positive shoulder_pitch rotates the whole arm down.
        # These values MUST match WholeBodyG1Composite._ARM_REST_POSE (the OFF-mode
        # hold) so the initial / paused-before-first-keypress pose is identical to
        # the settled disengaged pose -- no visible arm drop on start.
        # NOTE on indexing: this 43-D array is in MJCF body-tree (depth-first)
        # order, so the HANDS are interleaved between the arms -- left arm 15..21,
        # left hand 22..28, right arm 29..35, right hand 36..42. (This differs from
        # the wbt controller's 29-D body target, where the 14 arm joints are
        # contiguous at 15..28; do not copy indices between the two.)
        init[15] = 0.6  # left_shoulder_pitch
        init[16] = 0.12  # left_shoulder_roll
        init[29] = 0.6  # right_shoulder_pitch
        init[30] = -0.12  # right_shoulder_roll
        return init

    @property
    def default_base(self):
        # NoActuationBase: inserts a passive 'support' body at +_base_offset.
        # robocasa's CAM_CONFIGS attaches the agentview cameras to
        # 'mobilebase0_support' so we need this body in the tree.  The
        # freejoint on pelvis is preserved (we patched add_mobile_base in
        # robot_model.py so floating-base humanoids stay floating).
        return "NoActuationBase"

    @property
    def bottom_offset(self):
        # NoActuationBase puts 'support' (and thus manipulator_mount) at
        # +_base_offset above the freejoint coordinate frame; with
        # bottom_offset=0 the complete physical pelvis (geometry and inertia)
        # ends at the standing pelvis height.
        return np.zeros(3)

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.30, -0.1, 0.0),
            "empty": (-0.29, 0, 0.0),
            "table": lambda table_length: (-0.15 - table_length / 2, 0, 0.0),
        }
