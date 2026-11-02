"""Scene 205: close the two open top TV-cabinet drawers in order."""

import math

import numpy as np

from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomCloseTopRightThenTopLeftDrawer(
    IThorMultiStageCompositeMixin,
    IThorLivingRoomEnv,
):
    """Close the top-right TV drawer, then walk left and close the top-left."""

    SUPPORTED_SCENE_IDS = (205,)
    SPEC_TASK = "close_top_right_then_top_left_drawer"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    INITIAL_POSE_JITTER = False

    SIDE_TABLE_BODY = "stand_9985276433646fd9350dc0cc79ad70c5_1_0_0"
    DRESSER_BODY = "chestofdrawers_bcf35268e2027afed18909dcf850d56a_1_0_0"
    TOP_RIGHT_DRAWER_JOINT = (
        "chestofdrawers_bcf35268e2027afed18909dcf850d56a_1_8_0_" "DrawerMesh31_4"
    )
    TOP_LEFT_DRAWER_JOINT = (
        "chestofdrawers_bcf35268e2027afed18909dcf850d56a_1_2_0_joint_1"
    )
    BOTTOM_DRAWER_JOINTS = (
        "chestofdrawers_bcf35268e2027afed18909dcf850d56a_1_4_0_" "DrawerMesh11_2",
        "chestofdrawers_bcf35268e2027afed18909dcf850d56a_1_6_0_" "DrawerMesh21_3",
    )

    def __init__(self, *args, scene_ids=None, **kwargs):
        scene_ids = self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids
        kwargs.setdefault("initialization_noise", None)
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        return get_ithor_livingroom_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "robot_pose",
                "init_open_fraction",
                "success_open_fraction",
                "success_close_fraction",
                "hold_seconds",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                name="close_top_right_drawer",
                success="_right_closed_while_left_open",
                hold_steps=5,
            ),
            IThorAtomicTask(
                name="walk_left_and_close_top_left_drawer",
                success="_both_top_drawers_closed",
            ),
        )

    def _required_stage_hold_steps(self, task):
        if task is self._atomic_tasks[-1]:
            return math.ceil(
                float(self._get_task_spec()["hold_seconds"]) * self.control_freq
            )
        return super()._required_stage_hold_steps(task)

    def _spawn_clearable_fixture_roots(self):
        roots = super()._spawn_clearable_fixture_roots()
        protected = {
            self.sim.model.body_name2id(self.SIDE_TABLE_BODY),
            self.sim.model.body_name2id(self.DRESSER_BODY),
        }
        return roots - protected

    def _setup_references(self):
        super()._setup_references()
        self._top_right_drawer_joint_id = self.sim.model.joint_name2id(
            self.TOP_RIGHT_DRAWER_JOINT
        )
        self._top_left_drawer_joint_id = self.sim.model.joint_name2id(
            self.TOP_LEFT_DRAWER_JOINT
        )

    def _set_joint_fraction(self, joint_name, fraction):
        joint_id = self.sim.model.joint_name2id(joint_name)
        low, high = self.sim.model.jnt_range[joint_id]
        self.sim.data.set_joint_qpos(
            joint_name,
            low + float(fraction) * (high - low),
        )
        velocity_addr = self.sim.model.get_joint_qvel_addr(joint_name)
        self.sim.data.qvel[velocity_addr] = 0

    def _before_object_placement(self):
        super()._before_object_placement()
        initial = float(self._get_task_spec()["init_open_fraction"])
        self._set_joint_fraction(self.TOP_RIGHT_DRAWER_JOINT, initial)
        self._set_joint_fraction(self.TOP_LEFT_DRAWER_JOINT, initial)
        for joint_name in self.BOTTOM_DRAWER_JOINTS:
            self._set_joint_fraction(joint_name, 0.0)
        self.sim.forward()

    def _drawer_fraction(self, joint_name, joint_id):
        low, high = self.sim.model.jnt_range[joint_id]
        qpos = float(self.sim.data.get_joint_qpos(joint_name))
        return float(np.clip((qpos - low) / (high - low), 0.0, 1.0))

    def _top_right_drawer_fraction(self):
        return self._drawer_fraction(
            self.TOP_RIGHT_DRAWER_JOINT,
            self._top_right_drawer_joint_id,
        )

    def _top_left_drawer_fraction(self):
        return self._drawer_fraction(
            self.TOP_LEFT_DRAWER_JOINT,
            self._top_left_drawer_joint_id,
        )

    def _right_closed_while_left_open(self):
        spec = self._get_task_spec()
        return bool(
            self._top_right_drawer_fraction() <= float(spec["success_close_fraction"])
            and self._top_left_drawer_fraction() >= float(spec["success_open_fraction"])
        )

    def _both_top_drawers_closed(self):
        threshold = float(self._get_task_spec()["success_close_fraction"])
        return bool(
            self._top_right_drawer_fraction() <= threshold
            and self._top_left_drawer_fraction() <= threshold
        )

    def _success_region_bounds(self):
        return []

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "close the top-right drawer under the television first, then "
                "walk left and close the top-left drawer"
            ),
            task_id=type(self).__name__,
            first_target_joint=self.TOP_RIGHT_DRAWER_JOINT,
            second_target_joint=self.TOP_LEFT_DRAWER_JOINT,
            success_definition=(
                "both top TV-cabinet drawers start fully open; close the right "
                "drawer to 10 percent or less while the left remains open, then "
                "close the left drawer to 10 percent or less and hold both "
                "closed for 1 second"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomCloseTopRightThenTopLeftDrawer"]
