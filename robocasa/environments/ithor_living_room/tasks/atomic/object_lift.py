"""Shared fixed-THOR-object support for living-room lift tasks."""

import os

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
)
from robocasa.environments.ithor_living_room.tasks.atomic.object_to_basket import (
    FixedObjectToBasketMixin,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class FixedObjectLiftMixin(FixedObjectToBasketMixin):
    """Adapt a fixed THOR MJCF and place it using task-specific YAML."""

    TARGET_XML = None
    SUPPORTED_SCENE_IDS = (202, 207, 209, 210, 212, 226)

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _asset_path(self, relative_path):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", relative_path)
        )

    @property
    def REMOTE_NAME(self):
        return self.TARGET_NAME

    @property
    def REMOTE_XML(self):
        return self.TARGET_XML

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "robot_pose",
                    "object_pos",
                    "object_yaw",
                    "lift_height",
                    "source_category",
                    "source_wrapper_pos",
                    "source_wrapper_quat",
                    "source_bbox_pos",
                    "source_bbox_size",
                    "source_scale",
                ),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _get_obj_cfgs(self):
        adapter = self._get_object_adapter()
        return [
            dict(
                name=self.TARGET_NAME,
                obj_groups=self._asset_path(self.TARGET_XML),
                object_scale=float(adapter["source_scale"]),
            )
        ]

    def _get_scene_surface_calibration(self):
        return None

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        if pos.shape != (3,) or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: robot_pose needs XYZ + yaw"
            )
        return pos, float(pose["yaw"])

    def _place_robot_at_target(self):
        # The calibrated scene pose is supplied by _get_scene_robot_pose_calibration.
        return None

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualLivingRoomLiftSampler")

    def _is_scene_valid(self):
        return IThorLivingRoomEnv._is_scene_valid(self)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.MIN_LIFT_DELTA = float(self._get_task_spec()["lift_height"])
        if self.MIN_LIFT_DELTA <= 0.0:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: lift_height must be positive"
            )
        self.HOLD_SECONDS = float(self._get_task_spec().get("hold_seconds", 1.0))

    def _create_objects(self):
        self._remote_temp_xmls = []
        try:
            super()._create_objects()
        finally:
            for path in self._remote_temp_xmls:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            self._remote_temp_xmls = []

    def _set_target_pose_from_spec(self):
        spec = self._get_task_spec()
        set_free_joint_bottom_at(
            self,
            self.objects[self.TARGET_NAME],
            spec["object_pos"],
            yaw=spec["object_yaw"],
        )
        self.sim.forward()

    def _constrain_initial_jitter_candidate(self):
        # These task object poses are fixed by YAML, including after Q resets.
        self._set_target_pose_from_spec()

    def _reset_internal(self):
        super()._reset_internal()
        self._set_target_pose_from_spec()
        # Record the reference after applying the task-specific table placement.
        self._lift_initial_z = self._target_root_z()
        self._lift_hold_steps = 0


__all__ = ["FixedObjectLiftMixin"]
