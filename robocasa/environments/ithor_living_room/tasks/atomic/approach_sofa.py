"""iTHOR living-room navigation: walk to the calibrated sofa."""

import numpy as np

from robocasa.environments.ithor_atomic.navigation import RoomNavMixin
from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomApproachSofa(RoomNavMixin, IThorLivingRoomEnv):
    TARGET_CATEGORY = "Sofa"
    SUPPORTED_SCENE_IDS = (202, 207, 209, 210, 212, 226)
    SPEC_TASK = "approach_sofa"
    SUCCESS_DISTANCE = 0.1

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "target_fixture",
                    "robot_pose",
                    "target_xy",
                    "target_tolerance",
                ),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _get_scene_surface_calibration(self):
        # Navigation has no placed props; use scene analysis for the floor.
        return None

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        if pos.shape != (3,) or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: " "robot_pose needs XYZ + yaw"
            )
        return pos, float(pose["yaw"])

    def _navigation_fixture_query(self):
        return dict(id=self._get_task_spec()["target_fixture"])

    def _setup_task_fixtures(self):
        spec = self._get_task_spec()
        self.target_xy = np.asarray(spec["target_xy"], dtype=float)
        self.target_tolerance = np.asarray(spec["target_tolerance"], dtype=float)
        self.show_target_region = bool(spec.get("show_target_region", False))
        hold_seconds = float(spec.get("hold_seconds", 1.0))
        self.HOLD_STEPS = int(round(hold_seconds * self.control_freq))
        if (
            self.target_xy.shape != (2,)
            or self.target_tolerance.shape != (2,)
            or np.any(self.target_tolerance < 0.0)
            or self.HOLD_STEPS < 1
            or hold_seconds <= 0.0
        ):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: invalid navigation calibration"
            )
        super()._setup_task_fixtures()
