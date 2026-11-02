"""Kitchen composites that navigate to a fixture before operating it."""

from __future__ import annotations

import numpy as np

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.articulation import (
    IThorKitchenCloseCabinet,
    IThorKitchenCloseDrawer,
    IThorKitchenOpenCabinet,
    IThorKitchenOpenDrawer,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.bowl_to_sink import (
    IThorKitchenBowlToSink,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.cup_to_sink import (
    IThorKitchenCupToSink,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.pan_table_to_stove import (
    IThorKitchenPanTableToStove,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.stand_canned_drink_upright import (
    IThorKitchenStandCannedDrinkUpright,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_specs,
)


_KITCHEN_SCENE_IDS = (6, 7, 13, 24, 25)
_NAVIGATION_FIELDS = (
    "approach_fixture",
    "robot_pose",
    "approach_distance",
    "minimum_start_distance",
    "hold_seconds",
)
_CAN_FIELDS = _NAVIGATION_FIELDS + (
    "support_region",
    "can_pos",
    "can_quat",
    "show_support",
    "upright_tolerance_degrees",
    "support_z_tolerance",
    "release_distance",
)
_PAN_FIELDS = _NAVIGATION_FIELDS + (
    "target_region",
    "pan_pos",
    "pan_quat",
    "show_target_region",
    "support_z_tolerance",
    "pan_base_local_xy",
    "pan_base_radius",
)
_SINK_FIELDS = _NAVIGATION_FIELDS + (
    "object_pos",
    "sink_fixture",
    "sink_region",
)
_ARTICULATION_FIELDS = _NAVIGATION_FIELDS + (
    "target_fixture",
    "target_joint",
    "init_open_fraction",
)


class _KitchenApproachFixtureGateMixin(CompositeStageReporterMixin):
    """Require both navigation and the inherited atomic predicate."""

    ATOMIC_SKILL = "NavigateThenManipulate"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "navigate_then_manipulate"
    APPROACH_HOLD_STEPS = 5
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    SCENE_SPECS = None

    def _navigation_spec(self):
        spec = self._get_task_spec()
        missing = [field for field in _NAVIGATION_FIELDS if field not in spec]
        if missing:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: missing navigation fields "
                f"{missing}"
            )
        return spec

    def _get_scene_robot_pose_calibration(self):
        pose = self._navigation_spec()["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        if pos.shape != (3,) or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: robot_pose needs XYZ + yaw"
            )
        return pos, float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        stable_id = self._navigation_spec()["approach_fixture"]
        self.approach_fixture = self.get_fixture(id=stable_id)
        if (
            self.approach_fixture is None
            or self.approach_fixture.stable_id != stable_id
        ):
            raise LookupError(
                f"{self.scene_name}/{self.SPEC_TASK}: missing approach fixture "
                f"{stable_id}"
            )
        self.fixture_refs["approach_target"] = self.approach_fixture

    def _validate_approach_start(self):
        start_xy = self._robot_base_xy()
        if not self.robot_footprint_is_on_floor(start_xy, radius=0.25):
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: start footprint leaves Floor"
            )
        if (
            self.count_body_collisions()
            and not self._calibrate_robot_free_joint_height()
        ):
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: start has a collision"
            )
        distance = self.robot_fixture_collision_distance(
            self.approach_fixture,
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
                f"{self.scene_name}/{self.SPEC_TASK}: invalid navigation start "
                f"(distance={distance:.3f}, minimum={minimum:.3f}, "
                f"approach={approach:.3f})"
            )
        self._approach_start_distance = float(distance)
        self._approach_distance = approach

    def _raw_operation_success(self):
        raise NotImplementedError

    def _robot_near_approach_fixture(self):
        distance = self.robot_fixture_collision_distance(
            self.approach_fixture,
            max_distance=1.5,
        )
        return bool(np.isfinite(distance) and distance <= self._approach_distance)

    def _reset_internal(self):
        super()._reset_internal()
        self._validate_approach_start()
        hold_seconds = float(self._navigation_spec()["hold_seconds"])
        self._operation_hold_steps = max(
            1, int(round(hold_seconds * self.control_freq))
        )
        self._reset_task_progress()
        if self._raw_operation_success():
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: composite starts successful"
            )

    def _reset_task_progress(self):
        parent = getattr(super(), "_reset_task_progress", None)
        if callable(parent):
            parent()
        self._approach_stage = 0
        self._approach_hold = 0
        self._operation_hold = 0
        self._success_hold = 0
        self._success_hold_steps = 0

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self._approach_stage == 0:
            ready = self._robot_near_approach_fixture()
            self._approach_hold = self._approach_hold + 1 if ready else 0
            if self._approach_hold >= self.APPROACH_HOLD_STEPS:
                self._approach_stage = 1
                self._report_stage_complete(
                    1,
                    2,
                    next_stage="complete_fixture_manipulation",
                )

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
            approach_fixture=self.approach_fixture.stable_id,
            approach_distance=float(self._approach_distance),
            approach_start_distance=float(self._approach_start_distance),
            composite_stages=list(self.COMPOSITE_ORDERING),
            success_definition=(
                "reach the named fixture and continuously hold the corresponding "
                "atomic task predicate; completion order is unrestricted"
            ),
        )
        return ep_meta


class IThorKitchenApproachTableAndStandCannedDrinkUpright(
    _KitchenApproachFixtureGateMixin,
    IThorKitchenStandCannedDrinkUpright,
):
    SPEC_TASK = "approach_table_and_stand_canned_drink_upright"
    TASK_LANG = 'walk to the table and stand the canned drink upright'
    SUPPORTED_SCENE_IDS = _KITCHEN_SCENE_IDS
    COMPOSITE_ORDERING = ("approach_table", "stand_canned_drink_upright")
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK, required=_CAN_FIELDS)

    def _raw_operation_success(self):
        return IThorKitchenStandCannedDrinkUpright._raw_success(self)


class IThorKitchenApproachStoveAndPutPan(
    _KitchenApproachFixtureGateMixin,
    IThorKitchenPanTableToStove,
):
    SPEC_TASK = "approach_stove_and_put_pan"
    TASK_LANG = 'walk to the stove and put the pan on the stove'
    SUPPORTED_SCENE_IDS = _KITCHEN_SCENE_IDS
    COMPOSITE_ORDERING = ("approach_stove", "put_pan_on_stove")
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK, required=_PAN_FIELDS)

    def _raw_operation_success(self):
        return IThorKitchenPanTableToStove._raw_success(self)


class IThorKitchenApproachSinkAndPutBowl(
    _KitchenApproachFixtureGateMixin,
    IThorKitchenBowlToSink,
):
    SPEC_TASK = "approach_sink_and_put_bowl"
    TASK_LANG = 'walk to the sink and put the bowl into the sink'
    SUPPORTED_SCENE_IDS = _KITCHEN_SCENE_IDS
    COMPOSITE_ORDERING = ("approach_sink", "put_bowl_in_sink")
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK, required=_SINK_FIELDS)

    def _raw_operation_success(self):
        return IThorKitchenBowlToSink._raw_success(self)


class IThorKitchenApproachSinkAndPutCup(
    _KitchenApproachFixtureGateMixin,
    IThorKitchenCupToSink,
):
    SPEC_TASK = "approach_sink_and_put_cup"
    TASK_LANG = 'walk to the sink and put the cup into the sink'
    SUPPORTED_SCENE_IDS = _KITCHEN_SCENE_IDS
    COMPOSITE_ORDERING = ("approach_sink", "put_cup_in_sink")
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK, required=_SINK_FIELDS)

    def _raw_operation_success(self):
        return IThorKitchenBowlToSink._raw_success(self)


class _KitchenApproachArticulationMixin(_KitchenApproachFixtureGateMixin):
    ATOMIC_SKILL = "NavigateThenArticulate"
    COMPOSITE_KIND = "navigate_then_articulate"

    def _raw_operation_success(self):
        return RoomOpenArticulation._raw_success(self)

    def _finalize_articulation_robot_pose(self, distance):
        del distance
        self.sim.forward()
        return bool(
            self.count_body_collisions() == 0
            or self._calibrate_robot_free_joint_height()
        )


class IThorKitchenApproachDrawerAndOpen(
    _KitchenApproachArticulationMixin,
    IThorKitchenOpenDrawer,
):
    SPEC_TASK = "approach_drawer_and_open"
    TASK_LANG = 'walk to the drawer and open it'
    SUPPORTED_SCENE_IDS = _KITCHEN_SCENE_IDS
    COMPOSITE_ORDERING = ("approach_drawer", "open_drawer")
    SCENE_SPECS = get_ithor_kitchen_task_specs(
        SPEC_TASK,
        required=_ARTICULATION_FIELDS + ("success_open_fraction",),
    )


class IThorKitchenApproachDrawerAndClose(
    _KitchenApproachArticulationMixin,
    IThorKitchenCloseDrawer,
):
    SPEC_TASK = "approach_drawer_and_close"
    TASK_LANG = 'walk to the drawer and close it'
    SUPPORTED_SCENE_IDS = _KITCHEN_SCENE_IDS
    COMPOSITE_ORDERING = ("approach_drawer", "close_drawer")
    SCENE_SPECS = get_ithor_kitchen_task_specs(
        SPEC_TASK,
        required=_ARTICULATION_FIELDS + ("success_close_fraction",),
    )


class IThorKitchenApproachCabinetAndOpen(
    _KitchenApproachArticulationMixin,
    IThorKitchenOpenCabinet,
):
    SPEC_TASK = "approach_cabinet_and_open"
    TASK_LANG = 'walk to the cabinet and open it'
    SUPPORTED_SCENE_IDS = _KITCHEN_SCENE_IDS
    COMPOSITE_ORDERING = ("approach_cabinet", "open_cabinet")
    SCENE_SPECS = get_ithor_kitchen_task_specs(
        SPEC_TASK,
        required=_ARTICULATION_FIELDS + ("success_open_fraction",),
    )


class IThorKitchenApproachCabinetAndClose(
    _KitchenApproachArticulationMixin,
    IThorKitchenCloseCabinet,
):
    SPEC_TASK = "approach_cabinet_and_close"
    TASK_LANG = 'walk to the cabinet and close it'
    SUPPORTED_SCENE_IDS = _KITCHEN_SCENE_IDS
    COMPOSITE_ORDERING = ("approach_cabinet", "close_cabinet")
    SCENE_SPECS = get_ithor_kitchen_task_specs(
        SPEC_TASK,
        required=_ARTICULATION_FIELDS + ("success_close_fraction",),
    )


__all__ = [
    "IThorKitchenApproachTableAndStandCannedDrinkUpright",
    "IThorKitchenApproachStoveAndPutPan",
    "IThorKitchenApproachSinkAndPutBowl",
    "IThorKitchenApproachSinkAndPutCup",
    "IThorKitchenApproachDrawerAndOpen",
    "IThorKitchenApproachDrawerAndClose",
    "IThorKitchenApproachCabinetAndOpen",
    "IThorKitchenApproachCabinetAndClose",
]
