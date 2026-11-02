"""Kitchen composites that approach a table before manipulating its props."""

from __future__ import annotations

import numpy as np

from robocasa.environments.ithor_atomic.pick_place import (
    SupportContactPlacementMixin,
)
from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.environments.ithor_kitchen.tasks.atomic.apple_on_plate import (
    IThorKitchenAppleOnPlate,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.lift_objects import (
    KitchenLiftObjectMixin,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
    get_ithor_kitchen_task_specs,
)


_KITCHEN_SCENE_IDS = (6, 7, 13, 24, 25)
_COMMON_SPEC_FIELDS = (
    "target_fixture",
    "support_body",
    "robot_pose",
    "approach_distance",
    "minimum_start_distance",
    "hold_seconds",
)
_LIFT_SPEC_FIELDS = _COMMON_SPEC_FIELDS + (
    "lift_height",
    "support_region",
    "object_pos",
    "object_quat",
)
_PLATE_SPEC_FIELDS = _COMMON_SPEC_FIELDS + ("plate_pos", "source_pos")


class _KitchenApproachTableGateMixin(CompositeStageReporterMixin):
    """Require an exact distant table approach before the local operation."""

    ATOMIC_SKILL = "NavigateThenManipulate"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "navigate_then_manipulate"
    APPROACH_HOLD_STEPS = 5
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    SCENE_SPECS = None
    REQUIRED_SPEC_FIELDS = _COMMON_SPEC_FIELDS

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_kitchen_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=self.REQUIRED_SPEC_FIELDS,
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _navigation_spec(self):
        return self._get_task_spec()

    def _get_scene_robot_pose_calibration(self):
        pose = self._navigation_spec()["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        if pos.shape != (3,) or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}: approach_table robot_pose needs XYZ + yaw"
            )
        return pos, float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        navigation = self._navigation_spec()
        stable_id = navigation["target_fixture"]
        self.approach_table_fixture = self.get_fixture(id=stable_id)
        if (
            self.approach_table_fixture is None
            or self.approach_table_fixture.stable_id != stable_id
        ):
            raise LookupError(
                f"{self.scene_name}: missing exact approach table {stable_id}"
            )
        support_body = navigation["support_body"]
        if support_body not in self.approach_table_fixture.body_names:
            raise ValueError(
                f"{self.scene_name}: operation support {support_body} does not "
                f"belong to approach target {stable_id}"
            )
        if hasattr(self, "work_surface_fixture") and (
            support_body not in self.work_surface_fixture.body_names
        ):
            raise ValueError(
                f"{self.scene_name}: lift surface differs from approach table"
            )
        self.fixture_refs["approach_table"] = self.approach_table_fixture

    def _setup_scene(self):
        super()._setup_scene()
        self._place_objects_before_approach_calibration()
        pose = self._navigation_spec()["robot_pose"]
        xy = np.asarray(pose["pos"][:2], dtype=float)
        yaw = float(pose["yaw"])
        if hasattr(self, "_robot_free_joint_z"):
            delattr(self, "_robot_free_joint_z")
        placed = bool(
            self.robot_footprint_is_on_floor(xy, radius=0.25)
            and self._set_robot_free_pose(xy, yaw)
        )
        if placed:
            placed = self._calibrate_robot_free_joint_height()
        if not placed:
            raise RuntimeError(
                f"{self.scene_name}: cannot place the distant table start"
            )
        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])

    def _place_objects_before_approach_calibration(self):
        raise NotImplementedError

    def _validate_approach_start(self):
        """Validate after task objects and the robot have reached reset poses."""
        start_xy = self._robot_base_xy()
        if not self.robot_footprint_is_on_floor(start_xy, radius=0.25):
            raise RuntimeError(
                f"{self.scene_name}: approach table start footprint leaves Floor"
            )
        if self.count_body_collisions():
            raise RuntimeError(
                f"{self.scene_name}: approach table start has a collision"
            )
        distance = self.robot_fixture_collision_distance(
            self.approach_table_fixture,
            max_distance=5.0,
        )
        minimum = float(self._navigation_spec()["minimum_start_distance"])
        approach = float(self._navigation_spec()["approach_distance"])
        if (
            not np.isfinite(distance)
            or not 0.0 < approach < minimum
            or distance < minimum
            or distance < approach + 0.15
        ):
            raise RuntimeError(
                f"{self.scene_name}: invalid distant table start "
                f"(distance={distance:.3f}, minimum={minimum:.3f}, "
                f"approach={approach:.3f})"
            )
        self._approach_start_distance = float(distance)
        self._approach_distance = approach

    def _raw_operation_success(self):
        raise NotImplementedError

    def _robot_near_table(self):
        distance = self.robot_fixture_collision_distance(
            self.approach_table_fixture,
            max_distance=1.5,
        )
        return bool(np.isfinite(distance) and distance <= self._approach_distance)

    def _reset_internal(self):
        super()._reset_internal()
        self._validate_approach_start()
        hold_seconds = float(self._get_task_spec()["hold_seconds"])
        self._operation_hold_steps = max(
            1, int(round(hold_seconds * self.control_freq))
        )
        self._reset_task_progress()
        if self._raw_operation_success():
            raise RuntimeError(f"{self.scene_name}: composite starts successful")

    def _reset_task_progress(self):
        """Clear both composite and inherited atomic success state."""
        parent = getattr(super(), "_reset_task_progress", None)
        if callable(parent):
            parent()
        self._approach_stage = 0
        self._approach_hold = 0
        self._operation_hold = 0
        self._lift_hold_steps = 0
        self._success_hold_steps = 0

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self._approach_stage == 0:
            ready = self._robot_near_table()
            self._approach_hold = self._approach_hold + 1 if ready else 0
            if self._approach_hold >= self.APPROACH_HOLD_STEPS:
                self._approach_stage = 1
                self._report_stage_complete(
                    1,
                    2,
                    next_stage="complete_table_manipulation",
                )
            self._operation_hold = 0
            return ret

        valid = self._raw_operation_success()
        self._operation_hold = self._operation_hold + 1 if valid else 0
        return ret

    def _check_success(self):
        return bool(
            self._approach_stage == 1
            and self._operation_hold >= self._operation_hold_steps
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            lang=self.TASK_LANG,
            approach_fixture=self.approach_table_fixture.stable_id,
            approach_distance=float(self._approach_distance),
            approach_start_distance=float(self._approach_start_distance),
            composite_stages=[
                "walk to the named table",
                "complete the local table manipulation and hold its result",
            ],
        )
        return ep_meta


class _KitchenApproachTableLiftMixin(_KitchenApproachTableGateMixin):
    """Use the atomic lift-height predicate after reaching the table."""

    ATOMIC_SKILL = "NavigateThenLiftObject"
    COMPOSITE_KIND = "navigate_then_lift"
    REQUIRED_SPEC_FIELDS = _LIFT_SPEC_FIELDS

    def _place_objects_before_approach_calibration(self):
        self._place_lift_target()

    def _raw_operation_success(self):
        return KitchenLiftObjectMixin._raw_success(self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["success_definition"] = (
            f"after reaching the table, keep the {self.TARGET_LANG} at least "
            f"{self.min_lift_delta:g} meters above its reset height for "
            f"{self.lift_hold_seconds:g} seconds"
        )
        return ep_meta


class _KitchenApproachTablePlateMixin(_KitchenApproachTableGateMixin):
    """Use the atomic object-on-plate predicate after reaching the table."""

    ATOMIC_SKILL = "NavigateThenPickPlace"
    COMPOSITE_KIND = "navigate_then_pick_place"
    REQUIRED_SPEC_FIELDS = _PLATE_SPEC_FIELDS

    def _place_objects_before_approach_calibration(self):
        self._place_task_objects()

    def _configured_object_positions(self):
        spec = self._get_task_spec()
        plate_pos = np.asarray(spec["plate_pos"], dtype=float)
        source_pos = np.asarray(spec["source_pos"], dtype=float)
        if plate_pos.shape != (3,) or source_pos.shape != (3,):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: "
                "plate_pos and source_pos must contain XYZ"
            )
        return plate_pos, source_pos

    def _raw_operation_success(self):
        return IThorKitchenAppleOnPlate._raw_success(self)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["success_definition"] = (
            f"after reaching the table, the {self.SOURCE_LANG} must contact "
            "the plate near its center and remain there for "
            f"{self.success_hold_seconds:g} seconds"
        )
        return ep_meta


class IThorKitchenApproachTableAndLiftApple(
    _KitchenApproachTableLiftMixin,
    KitchenLiftObjectMixin,
    SupportContactPlacementMixin,
    IThorKitchenEnv,
):
    TARGET_NAME = "apple"
    TARGET_GROUP = "apple"
    TARGET_LANG = "apple"
    SPEC_TASK = "approach_table_and_lift_apple"
    TASK_LANG = "walk to the table, then lift the apple"
    COMPOSITE_ORDERING = ("approach_table", "lift_apple")
    SUPPORTED_SCENE_IDS = _KITCHEN_SCENE_IDS
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK, required=_LIFT_SPEC_FIELDS)


class IThorKitchenApproachTableAndLiftSaltShaker(
    _KitchenApproachTableLiftMixin,
    KitchenLiftObjectMixin,
    SupportContactPlacementMixin,
    IThorKitchenEnv,
):
    TARGET_NAME = "salt_shaker"
    TARGET_LANG = "salt shaker"
    TARGET_XML = (
        "models/assets/objects/lightwheel/salt_and_pepper_shaker/"
        "SaltShaker002/model.xml"
    )
    SPEC_TASK = "approach_table_and_lift_salt_shaker"
    TASK_LANG = "walk to the table, then lift the salt shaker"
    COMPOSITE_ORDERING = ("approach_table", "lift_salt_shaker")
    SUPPORTED_SCENE_IDS = _KITCHEN_SCENE_IDS
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK, required=_LIFT_SPEC_FIELDS)


class IThorKitchenApproachTableAndLiftBowl(
    _KitchenApproachTableLiftMixin,
    KitchenLiftObjectMixin,
    SupportContactPlacementMixin,
    IThorKitchenEnv,
):
    TARGET_NAME = "bowl"
    TARGET_GROUP = "bowl"
    TARGET_LANG = "bowl"
    SPEC_TASK = "approach_table_and_lift_bowl"
    TASK_LANG = "walk to the table, then lift the bowl"
    COMPOSITE_ORDERING = ("approach_table", "lift_bowl")
    SUPPORTED_SCENE_IDS = _KITCHEN_SCENE_IDS
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK, required=_LIFT_SPEC_FIELDS)


class IThorKitchenApproachTableAndPutAppleOnPlate(
    _KitchenApproachTablePlateMixin,
    IThorKitchenAppleOnPlate,
):
    SPEC_TASK = "approach_table_and_put_apple_on_plate"
    TASK_LANG = "walk to the table, then put the apple on the plate"
    COMPOSITE_ORDERING = ("approach_table", "put_apple_on_plate")
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK, required=_PLATE_SPEC_FIELDS)


class IThorKitchenApproachTableAndPutBreadOnPlate(
    _KitchenApproachTablePlateMixin,
    IThorKitchenAppleOnPlate,
):
    SOURCE_NAME = "bread"
    SOURCE_GROUP = "bread"
    SOURCE_LANG = "bread"
    SPEC_TASK = "approach_table_and_put_bread_on_plate"
    TASK_LANG = "walk to the table, then put the bread on the plate"
    COMPOSITE_ORDERING = ("approach_table", "put_bread_on_plate")
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK, required=_PLATE_SPEC_FIELDS)


class IThorKitchenApproachTableAndPutCheeseOnPlate(
    _KitchenApproachTablePlateMixin,
    IThorKitchenAppleOnPlate,
):
    SOURCE_NAME = "cheese"
    SOURCE_GROUP = "cheese"
    SOURCE_LANG = "cheese"
    SPEC_TASK = "approach_table_and_put_cheese_on_plate"
    TASK_LANG = "walk to the table, then put the cheese on the plate"
    COMPOSITE_ORDERING = ("approach_table", "put_cheese_on_plate")
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK, required=_PLATE_SPEC_FIELDS)


class IThorKitchenApproachTableAndPutMugOnPlate(
    _KitchenApproachTablePlateMixin,
    IThorKitchenAppleOnPlate,
):
    SOURCE_NAME = "mug"
    SOURCE_GROUP = "mug"
    SOURCE_LANG = "mug"
    SPEC_TASK = "approach_table_and_put_mug_on_plate"
    TASK_LANG = "walk to the table, then put the mug on the plate"
    COMPOSITE_ORDERING = ("approach_table", "put_mug_on_plate")
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK, required=_PLATE_SPEC_FIELDS)


__all__ = [
    "IThorKitchenApproachTableAndLiftApple",
    "IThorKitchenApproachTableAndLiftSaltShaker",
    "IThorKitchenApproachTableAndLiftBowl",
    "IThorKitchenApproachTableAndPutAppleOnPlate",
    "IThorKitchenApproachTableAndPutBreadOnPlate",
    "IThorKitchenApproachTableAndPutCheeseOnPlate",
    "IThorKitchenApproachTableAndPutMugOnPlate",
]
