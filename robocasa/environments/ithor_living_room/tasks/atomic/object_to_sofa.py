"""Shared fixed-object MJCF adapter for living-room sofa tasks."""

import os

from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
)
from robocasa.environments.ithor_living_room.tasks.atomic.object_to_basket import (
    FixedObjectToBasketMixin,
)
from robocasa.environments.ithor_living_room.tasks.atomic.raised_work_surface import (
    TableSizedRaisedSurfaceMixin,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class FixedObjectToSofaMixin(FixedObjectToBasketMixin):
    """Load and place one fixed THOR object using task-specific YAML."""

    def _asset_path(self, relative_path):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", relative_path)
        )

    @property
    def REMOTE_NAME(self):
        return self.SOURCE_NAME

    @property
    def REMOTE_XML(self):
        return self.SOURCE_XML

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "sofa_fixture",
                    "robot_pose",
                    "object_pos",
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
                name=self.SOURCE_NAME,
                obj_groups=self._asset_path(self.SOURCE_XML),
                object_scale=float(adapter["source_scale"]),
            )
        ]

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

    def _reset_internal(self):
        # Skip BasketToSofa's basket-specific placement while preserving the
        # living-room reset path inherited by that task.
        IThorLivingRoomEnv._reset_internal(self)
        if self.SOURCE_NAME in self.objects:
            spec = self._get_task_spec()
            set_free_joint_bottom_at(
                self,
                self.objects[self.SOURCE_NAME],
                spec["object_pos"],
                yaw=float(spec.get("object_yaw", 0.0)),
            )
            self.sim.forward()


class RaisedFixedObjectToSofaMixin(TableSizedRaisedSurfaceMixin):
    """Add a table-sized riser under a fixed sofa-task source object."""

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self._setup_table_sized_work_surface([self._get_task_spec()["object_pos"]])

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and getattr(self, "work_surface_region", None) is not None
        )

    def _reset_internal(self):
        super()._reset_internal()
        spec = self._get_task_spec()
        set_free_joint_bottom_at(
            self,
            self.objects[self.SOURCE_NAME],
            self._work_surface_bottom_position(spec["object_pos"]),
            yaw=float(spec.get("object_yaw", 0.0)),
        )
        self.sim.forward()


__all__ = ["FixedObjectToSofaMixin", "RaisedFixedObjectToSofaMixin"]
