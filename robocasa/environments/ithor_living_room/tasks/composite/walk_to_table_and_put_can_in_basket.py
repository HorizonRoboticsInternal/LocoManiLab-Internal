"""Navigate to a table, then place its can into its basket."""

import numpy as np
import robocasa.utils.object_utils as OU
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
)
from robocasa.environments.ithor_living_room.tasks.composite.explicit_spec import (
    ExplicitCompositeSpecMixin,
)
from robocasa.environments.ithor_living_room.tasks.atomic.raised_work_surface import (
    TableSizedRaisedSurfaceMixin,
)


class IThorLivingRoomWalkToTableAndPutCanInBasket(
    CompositeStageReporterMixin,
    TableSizedRaisedSurfaceMixin,
    ExplicitCompositeSpecMixin,
    IThorLivingRoomEnv,
):
    ATOMIC_SKILL = "ApproachPickPlace"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "navigate_then_can_to_basket"
    COMPOSITE_ORDERING = "sequential"
    SPEC_TASK = "walk_to_table_and_put_can_in_basket"
    SUPPORTED_SCENE_IDS = (202, 207, 209, 212, 226)
    REQUIRED_SPEC_FIELDS = (
        "work_surface_support",
        "table_fixture",
        "robot_pose",
        "basket_pos",
        "basket_yaw",
        "basket_scale",
        "can_pos",
        "can_yaw",
        "approach_distance",
        "minimum_start_distance",
        "success_xy_tolerance",
        "hold_seconds",
    )
    BASKET_XML = "models/assets/objects/lightwheel/basket/Basket039/model.xml"
    APPROACH_HOLD_STEPS = 5
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.04

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        self.table_fixture = self.get_fixture(id=spec["table_fixture"])
        if self.table_fixture is None:
            return
        self.fixture_refs["navigation_table"] = self.table_fixture
        self.WORK_SURFACE_TOP_Z = float(spec.get("work_surface_top_z", 0.75))
        self._setup_table_sized_work_surface(
            [spec["basket_pos"], spec["can_pos"]],
            fixture_id=spec["table_fixture"],
        )
        self.set_robot_base_pose(*self._get_scene_robot_pose_calibration())

    def _get_obj_cfgs(self):
        spec = self._get_task_spec()
        return [
            dict(
                name="basket",
                obj_groups=self._asset_path(self.BASKET_XML),
                object_scale=float(spec["basket_scale"]),
            ),
            dict(name="can", obj_groups="can"),
        ]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualLivingRoomCanBasketSampler")

    def _is_scene_valid(self):
        return bool(
            IThorLivingRoomEnv._is_scene_valid(self)
            and getattr(self, "table_fixture", None) is not None
            and getattr(self, "work_surface_region", None) is not None
        )

    def _before_object_placement(self):
        super()._before_object_placement()
        spec = self._get_task_spec()
        for name in ("basket", "can"):
            set_free_joint_bottom_at(
                self,
                self.objects[name],
                self._work_surface_bottom_position(spec[f"{name}_pos"]),
                yaw=spec[f"{name}_yaw"],
            )
        self.sim.forward()

    def _constrain_initial_jitter_candidate(self):
        return None

    def _initial_jitter_allowed_contact_bodies(self, name):
        del name
        return set(self._fixture_body_ids(self.work_surface_fixture))

    def _initial_jitter_candidate_is_valid(self):
        return all(
            self.object_on_region(name, self.work_surface_region, require_contact=False)
            for name in ("basket", "can")
        )

    def _setup_scene(self):
        super()._setup_scene()
        spec = self._get_task_spec()
        self._start_distance = self.robot_fixture_collision_distance(
            self.table_fixture, max_distance=4.0
        )
        if not (
            np.isfinite(self._start_distance)
            and self._start_distance > float(spec["minimum_start_distance"])
        ):
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: invalid distant table start"
            )
        self.HOLD_STEPS = max(
            1, int(round(float(spec["hold_seconds"]) * self.control_freq))
        )
        self._route_stage = 0
        self._approach_hold = 0
        self._success_hold = 0

    def _robot_is_near_table(self):
        distance = self.robot_fixture_collision_distance(
            self.table_fixture, max_distance=1.5
        )
        return bool(
            np.isfinite(distance)
            and distance <= float(self._get_task_spec()["approach_distance"])
        )

    def _can_is_packed(self):
        spec = self._get_task_spec()
        return OU.check_obj_in_receptacle(
            self, "can", "basket", th=float(spec["success_xy_tolerance"])
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self._route_stage == 0:
            near = self._robot_is_near_table()
            self._approach_hold = self._approach_hold + 1 if near else 0
            if self._approach_hold >= self.APPROACH_HOLD_STEPS:
                self._route_stage = 1
                self._report_stage_complete(
                    1,
                    2,
                    next_stage="put_can_in_basket",
                )
            self._success_hold = 0
            return ret
        valid = self._can_is_packed()
        self._success_hold = self._success_hold + 1 if valid else 0
        return ret

    def _check_success(self):
        return self._route_stage == 1 and self._success_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            lang="walk to the table, then put the can into the basket",
            start_distance=float(self._start_distance),
            composite_stages=[
                "walk to the named table",
                "put the can in the basket",
            ],
        )
        return ep_meta


__all__ = ["IThorLivingRoomWalkToTableAndPutCanInBasket"]
