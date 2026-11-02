"""Small reusable building blocks for living-room composite tasks."""

import os

import numpy as np

from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class ExplicitCompositeSpecMixin:
    """Load one explicit scene/task calibration without deriving positions."""

    REQUIRED_SPEC_FIELDS = ()

    def _get_task_spec(self):
        key = (self.scene_name, self.SPEC_TASK)
        if getattr(self, "_composite_spec_key", None) != key:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=self.REQUIRED_SPEC_FIELDS,
            )
            self._composite_spec_key = key
        return self.task_spec

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

    def _asset_path(self, relative_path):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", relative_path)
        )


__all__ = ["ExplicitCompositeSpecMixin"]
