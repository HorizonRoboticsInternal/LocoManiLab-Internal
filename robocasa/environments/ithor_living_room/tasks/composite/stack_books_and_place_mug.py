"""Stack the right book on the left book, then place a mug on the stack."""

import numpy as np

from robocasa.environments.ithor_atomic.pick_place import SupportContactPlacementMixin
from robocasa.environments.ithor_atomic.task_assets import (
    MUG_MODEL,
    MUG_SCALE,
    asset_xml_path,
)
from robocasa.environments.ithor_composite.progress import CompositeStageReporterMixin
from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_oriented_bottom_at,
)
from robocasa.environments.ithor_living_room.tasks.atomic.stack_book_on_book import (
    IThorLivingRoomStackBookOnBook,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_specs,
)


class IThorLivingRoomStackBooksAndPlaceMug(
    CompositeStageReporterMixin,
    SupportContactPlacementMixin,
    IThorLivingRoomStackBookOnBook,
):
    """Require the book stack before the mug is placed on its top book."""

    ATOMIC_SKILL = "StackBooksThenPlaceMug"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "stack_books_then_place_mug"
    COMPOSITE_ORDERING = ("stack_right_book_on_left_book", "place_mug_on_books")
    SPEC_TASK = "stack_books_and_place_mug"
    TASK_LANG = (
        "stack the book on your right on top of the book on your left, "
        "then place the mug on top of the two stacked books"
    )
    SUPPORTED_SCENE_IDS = (202, 207, 209, 210, 212, 226)
    SCENE_SPECS = get_ithor_livingroom_task_specs(
        SPEC_TASK, required=("mug_pos", "mug_yaw", "stack_hold_seconds", "hold_seconds")
    )
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    MUG_NAME = "mug"
    MUG_HEIGHT_TOLERANCE = 0.01
    MUG_UPRIGHT_COSINE = 0.94

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.fixture_refs["work_surface"] = self.work_surface_fixture
        self.region_refs["work_surface"] = self.work_surface_region

    def _get_obj_cfgs(self):
        return super()._get_obj_cfgs() + [
            dict(
                name=self.MUG_NAME,
                obj_groups=asset_xml_path(MUG_MODEL),
                object_scale=MUG_SCALE,
            )
        ]

    def _place_stack_objects(self):
        spec = self._get_task_spec()
        for name in (self.BASE_BOOK_NAME, self.MOVING_BOOK_NAME, self.MUG_NAME):
            position = self._work_surface_bottom_position(spec[f"{name}_pos"])
            set_free_joint_oriented_bottom_at(
                self,
                self.objects[name],
                position,
                yaw=spec[f"{name}_yaw"],
                pitch=spec["moving_book_pitch"]
                if name == self.MOVING_BOOK_NAME
                else 0.0,
                contact_sink=0.0,
            )
            if not self.place_object_on_support_contact(
                name,
                self.work_surface_region,
                self.work_surface_fixture,
                xy=position[:2],
                inward_fractions=(0.0,),
                probe_step=0.0005,
            ):
                raise RuntimeError(
                    f"{self.scene_name}: {name} cannot rest on the work surface"
                )

    def _before_object_placement(self):
        super()._before_object_placement()
        self._place_stack_objects()

    def _setup_scene(self):
        super()._setup_scene()
        self._stack_collision_geom_ids = {}
        self._stack_hold_steps = max(
            1,
            int(
                round(
                    float(self._get_task_spec()["stack_hold_seconds"])
                    * self.control_freq
                )
            ),
        )
        self._reset_task_progress()

    def _reset_internal(self):
        IThorLivingRoomEnv._reset_internal(self)
        self._reset_task_progress()
        if (
            self.count_body_collisions()
            or not self.robot_footprint_is_on_floor(self._robot_base_xy(), radius=0.25)
            or self._books_stacked()
        ):
            raise RuntimeError(f"{self.scene_name}: invalid initial book-stack layout")

    def _reset_task_progress(self):
        self._stack_stage = 0
        self._stack_hold = 0
        self._success_hold_steps = 0

    def _books_stacked(self):
        # Hand proximity changes while placing the mug; it must not erase an
        # intact book stack and force the robot back into the first stage.
        base_points = self.object_bbox_points(self.BASE_BOOK_NAME)
        moving_points = self.object_bbox_points(self.MOVING_BOOK_NAME)
        base_center = (base_points.min(axis=0) + base_points.max(axis=0)) / 2
        moving_center = (moving_points.min(axis=0) + moving_points.max(axis=0)) / 2
        return bool(
            self.check_contact(
                self.objects[self.BASE_BOOK_NAME], self.objects[self.MOVING_BOOK_NAME]
            )
            and np.linalg.norm(moving_center[:2] - base_center[:2]) <= self.XY_TOLERANCE
            and moving_points[:, 2].min() >= base_points[:, 2].max() - 0.03
            and self._book_face_alignment() >= self.ORIENTATION_COSINE
            and self.object_on_region(
                self.BASE_BOOK_NAME, self.work_surface_region,
                z_tol=0.05, require_contact=True,
            )
        )

    def _mug_touches_top_book(self):
        return self.check_contact(
            self.objects[self.MUG_NAME], self.objects[self.MOVING_BOOK_NAME]
        )

    def _collision_z_bounds(self, name):
        """Use physical surfaces; the book's authored bbox has a vertical offset."""
        if name not in self._stack_collision_geom_ids:
            bodies = self._body_tree_ids([self.obj_body_id[name]])
            self._stack_collision_geom_ids[name] = tuple(self._collidable_geoms(bodies))
        bounds = []
        for geom_id in self._stack_collision_geom_ids[name]:
            aabb = np.asarray(self.sim.model.geom_aabb[geom_id])
            vertical = np.asarray(self.sim.data.geom_xmat[geom_id]).reshape(3, 3)[2]
            center_z = self.sim.data.geom_xpos[geom_id][2] + vertical @ aabb[:3]
            half_z = np.abs(vertical) @ aabb[3:]
            bounds.append((center_z - half_z, center_z + half_z))
        return np.min(bounds, axis=0)[0], np.max(bounds, axis=0)[1]

    def _mug_on_top_book(self):
        if not self._mug_touches_top_book():
            return False
        book_points = self.object_bbox_points(self.MOVING_BOOK_NAME)
        mug_points = self.object_bbox_points(self.MUG_NAME)
        _, book_top = self._collision_z_bounds(self.MOVING_BOOK_NAME)
        mug_bottom, _ = self._collision_z_bounds(self.MUG_NAME)
        rotation = np.asarray(
            self.sim.data.get_body_xmat(self.objects[self.MOVING_BOOK_NAME].root_body)
        ).reshape(3, 3)
        origin = book_points.mean(axis=0)
        book_local = (book_points - origin) @ rotation
        mug_local = (mug_points - origin) @ rotation
        # Include the mug footprint so side contact or a mug perched off an edge fails.
        contained = np.all(
            mug_local[:, :2].min(axis=0) >= book_local[:, :2].min(axis=0) - 0.005
        ) and np.all(
            mug_local[:, :2].max(axis=0) <= book_local[:, :2].max(axis=0) + 0.005
        )
        mug_rotation = np.asarray(
            self.sim.data.get_body_xmat(self.objects[self.MUG_NAME].root_body)
        ).reshape(3, 3)
        return bool(
            contained
            and abs(mug_bottom - book_top) <= self.MUG_HEIGHT_TOLERANCE
            and mug_rotation[2, 2] >= self.MUG_UPRIGHT_COSINE
            and self.object_released(self.MUG_NAME, th=self.release_distance)
        )

    def _raw_success(self):
        return bool(
            self._books_stacked()
            and self._mug_on_top_book()
            and self.object_released(self.MOVING_BOOK_NAME, th=self.release_distance)
        )

    def _post_action(self, action):
        ret = IThorLivingRoomEnv._post_action(self, action)
        books_stacked = self._books_stacked()
        if self._stack_stage == 0:
            valid = (
                books_stacked
                and self.object_released(self.MOVING_BOOK_NAME, th=self.release_distance)
                and not self._mug_touches_top_book()
            )
            self._stack_hold = self._stack_hold + 1 if valid else 0
            self._success_hold_steps = 0
            if self._stack_hold >= self._stack_hold_steps:
                self._stack_stage = 1
                self._report_stage_complete(1, 2, next_stage="place_mug_on_books")
            return ret
        if not books_stacked:
            self._reset_task_progress()
            return ret
        valid = self._raw_success()
        previously_complete = self._check_success()
        self._success_hold_steps = self._success_hold_steps + 1 if valid else 0
        if self._check_success() and not previously_complete:
            self._report_stage_complete(2, 2)
        return ret

    def _check_success(self):
        return self._stack_stage == 1 and self._success_hold_steps >= self.HOLD_STEPS

    def get_ep_meta(self):
        meta = IThorLivingRoomEnv.get_ep_meta(self)
        meta.update(
            lang=self.TASK_LANG,
            composite_stages=list(self.COMPOSITE_ORDERING),
            composite_ordering="strict",
            left_book=self.BASE_BOOK_NAME,
            right_book=self.MOVING_BOOK_NAME,
            target_object=self.MUG_NAME,
            target_support=self.MOVING_BOOK_NAME,
            success_hold_seconds=self.HOLD_SECONDS,
            success_definition=(
                "first stack the right book on the left book with no mug on top; "
                "then keep both books stacked on the table and the released mug "
                "upright on the upper book for 1 second"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomStackBooksAndPlaceMug"]
