"""iTHOR living-room: stack one explicitly placed book on another."""

import numpy as np

from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
    set_free_joint_oriented_bottom_at,
)
from robocasa.environments.ithor_living_room.tasks.atomic.object_to_basket import (
    FixedObjectToBasketMixin,
)
from robocasa.environments.ithor_living_room.tasks.atomic.raised_work_surface import (
    TableSizedRaisedSurfaceMixin,
)
from robocasa.environments.ithor_living_room.tasks.atomic.remote_to_basket import (
    IThorLivingRoomRemoteToBasket,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomStackBookOnBook(
    TableSizedRaisedSurfaceMixin,
    FixedObjectToBasketMixin,
    IThorLivingRoomRemoteToBasket,
):
    """Place the moving book flat and stably on the base book."""

    SPEC_TASK = "stack_book_on_book"
    BASE_BOOK_NAME = "base_book"
    MOVING_BOOK_NAME = "moving_book"
    REMOTE_XML = (
        "models/assets/objects/thor/Bedroom Objects/Book/Prefabs/" "Book_3/Book_3.xml"
    )
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.04

    @property
    def REMOTE_NAME(self):
        return self.MOVING_BOOK_NAME

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "work_surface_support",
                    "work_surface_top_z",
                    "base_book_pos",
                    "base_book_yaw",
                    "moving_book_pos",
                    "moving_book_yaw",
                    "moving_book_pitch",
                    "object_jitter_radius",
                    "robot_pose",
                    "xy_tolerance",
                    "orientation_cosine",
                    "release_distance",
                    "hold_seconds",
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

    def _setup_task_fixtures(self):
        IThorLivingRoomEnv._setup_task_fixtures(self)
        spec = self._get_task_spec()
        self.WORK_SURFACE_TOP_Z = float(spec["work_surface_top_z"])
        self.INITIAL_OBJECT_JITTER_RADIUS = float(spec["object_jitter_radius"])
        if self.INITIAL_OBJECT_JITTER_RADIUS < 0.0:
            raise ValueError(f"{self.scene_name}/{self.SPEC_TASK}: invalid jitter")
        self._setup_table_sized_work_surface(
            [spec["base_book_pos"], spec["moving_book_pos"]],
            fixture_id=spec["work_surface_support"]["fixture"],
        )
        self.XY_TOLERANCE = float(spec["xy_tolerance"])
        self.ORIENTATION_COSINE = float(spec["orientation_cosine"])
        self.release_distance = float(spec["release_distance"])
        self.HOLD_SECONDS = float(spec["hold_seconds"])
        if (
            min(self.XY_TOLERANCE, self.release_distance, self.HOLD_SECONDS) <= 0.0
            or not 0.0 < self.ORIENTATION_COSINE <= 1.0
        ):
            raise ValueError(f"{self.scene_name}/{self.SPEC_TASK}: invalid threshold")
        self.set_robot_base_pose(*self._get_scene_robot_pose_calibration())

    def _get_obj_cfgs(self):
        scale = float(self._get_task_spec()["source_scale"])
        path = self._asset_path(self.REMOTE_XML)
        return [
            dict(name=self.BASE_BOOK_NAME, obj_groups=path, object_scale=scale),
            dict(name=self.MOVING_BOOK_NAME, obj_groups=path, object_scale=scale),
        ]

    def _is_scene_valid(self):
        return bool(
            IThorLivingRoomEnv._is_scene_valid(self)
            and getattr(self, "work_surface_region", None) is not None
        )

    def _initial_jitter_candidate_is_valid(self):
        return all(
            self.object_on_region(
                name,
                self.work_surface_region,
                z_tol=0.08,
                require_contact=False,
            )
            for name in (self.BASE_BOOK_NAME, self.MOVING_BOOK_NAME)
        )

    def _reset_internal(self):
        IThorLivingRoomEnv._reset_internal(self)
        spec = self._get_task_spec()
        set_free_joint_bottom_at(
            self,
            self.objects[self.BASE_BOOK_NAME],
            spec["base_book_pos"],
            yaw=spec["base_book_yaw"],
        )
        set_free_joint_oriented_bottom_at(
            self,
            self.objects[self.MOVING_BOOK_NAME],
            spec["moving_book_pos"],
            yaw=spec["moving_book_yaw"],
            pitch=spec["moving_book_pitch"],
        )
        self.sim.forward()
        self._success_hold_steps = 0

    def _raw_success(self):
        base_points = self.object_bbox_points(self.BASE_BOOK_NAME)
        moving_points = self.object_bbox_points(self.MOVING_BOOK_NAME)
        base_center = 0.5 * (np.min(base_points, axis=0) + np.max(base_points, axis=0))
        moving_center = 0.5 * (
            np.min(moving_points, axis=0) + np.max(moving_points, axis=0)
        )
        return bool(
            self.check_contact(
                self.objects[self.BASE_BOOK_NAME],
                self.objects[self.MOVING_BOOK_NAME],
            )
            and np.linalg.norm(moving_center[:2] - base_center[:2]) <= self.XY_TOLERANCE
            and float(np.min(moving_points[:, 2]))
            >= float(np.max(base_points[:, 2])) - 0.03
            and self._book_face_alignment() >= self.ORIENTATION_COSINE
            and self.object_released(self.MOVING_BOOK_NAME, th=self.release_distance)
            and self.object_on_region(
                self.BASE_BOOK_NAME,
                self.work_surface_region,
                z_tol=0.05,
                require_contact=True,
            )
        )

    def _book_face_alignment(self):
        """Treat cover-up and cover-down stacking as equally valid."""
        base_matrix = np.asarray(
            self.sim.data.get_body_xmat(self.objects[self.BASE_BOOK_NAME].root_body),
            dtype=float,
        ).reshape(3, 3)
        moving_matrix = np.asarray(
            self.sim.data.get_body_xmat(self.objects[self.MOVING_BOOK_NAME].root_body),
            dtype=float,
        ).reshape(3, 3)
        return abs(float(np.dot(base_matrix[:, 2], moving_matrix[:, 2])))

    def _setup_scene(self):
        IThorLivingRoomEnv._setup_scene(self)
        self._success_hold_steps = 0
        self.HOLD_STEPS = max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def _post_action(self, action):
        ret = IThorLivingRoomEnv._post_action(self, action)
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return ret

    def _check_success(self):
        return self._success_hold_steps >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = IThorLivingRoomEnv.get_ep_meta(self)
        ep_meta.update(
            lang="stack one book on top of the other book",
            target_object=self.MOVING_BOOK_NAME,
            target_support=self.BASE_BOOK_NAME,
        )
        return ep_meta


__all__ = ["IThorLivingRoomStackBookOnBook"]
