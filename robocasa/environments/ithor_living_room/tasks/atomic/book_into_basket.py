"""iTHOR living-room: place a book into a basket on a raised surface."""

from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.environments.ithor_living_room.task_spec_utils import (
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


class IThorLivingRoomBookIntoBasket(
    TableSizedRaisedSurfaceMixin,
    FixedObjectToBasketMixin,
    IThorLivingRoomRemoteToBasket,
):
    """Pick up the explicitly placed book and put it in the basket."""

    REMOTE_NAME = "book"
    REMOTE_XML = (
        "models/assets/objects/thor/Bedroom Objects/Book/Prefabs/" "Book_3/Book_3.xml"
    )
    SPEC_TASK = "book_into_basket"
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.04

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "work_surface_support",
                    "work_surface_top_z",
                    "basket_pos",
                    "basket_yaw",
                    "basket_contact_sink",
                    "book_pos",
                    "book_yaw",
                    "book_pitch",
                    "object_jitter_radius",
                    "robot_pose",
                    "success_xy_tolerance",
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
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        self.WORK_SURFACE_TOP_Z = float(spec["work_surface_top_z"])
        self.INITIAL_OBJECT_JITTER_RADIUS = float(spec["object_jitter_radius"])
        if self.INITIAL_OBJECT_JITTER_RADIUS < 0.0:
            raise ValueError(f"{self.scene_name}/{self.SPEC_TASK}: invalid jitter")
        self._setup_table_sized_work_surface(
            [spec["book_pos"], spec["basket_pos"]],
            fixture_id=spec["work_surface_support"]["fixture"],
        )

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
            for name in (self.BASKET_NAME, self.REMOTE_NAME)
        )

    def _reset_internal(self):
        IThorLivingRoomEnv._reset_internal(self)
        spec = self._get_task_spec()
        set_free_joint_oriented_bottom_at(
            self,
            self.objects[self.BASKET_NAME],
            spec["basket_pos"],
            yaw=spec["basket_yaw"],
            pitch=0.0,
            contact_sink=spec["basket_contact_sink"],
        )
        set_free_joint_oriented_bottom_at(
            self,
            self.objects[self.REMOTE_NAME],
            spec["book_pos"],
            yaw=spec["book_yaw"],
            pitch=spec["book_pitch"],
        )
        self.sim.forward()
        self._success_hold_steps = 0

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick up the book and place it in the basket"
        return ep_meta


__all__ = ["IThorLivingRoomBookIntoBasket"]
