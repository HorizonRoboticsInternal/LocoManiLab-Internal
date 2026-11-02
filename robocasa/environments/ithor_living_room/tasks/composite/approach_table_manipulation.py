"""Navigate to the calibrated table before performing its atomic operation."""

from copy import deepcopy

import numpy as np

from robocasa.environments.ithor_composite.progress import CompositeStageReporterMixin
from robocasa.environments.ithor_living_room.tasks.atomic.approach_table import (
    IThorLivingRoomApproachTable,
)
from robocasa.environments.ithor_living_room.tasks.atomic.lift_dumbbell import (
    IThorLivingRoomLiftDumbbell,
)
from robocasa.environments.ithor_living_room.tasks.atomic.lift_statue import (
    IThorLivingRoomLiftStatue,
)
from robocasa.environments.ithor_living_room.tasks.atomic.lift_watering_can import (
    IThorLivingRoomLiftWateringCan,
)
from robocasa.environments.ithor_living_room.tasks.atomic.remote_to_basket import (
    IThorLivingRoomRemoteToBasket,
)
from robocasa.environments.ithor_living_room.tasks.atomic.candle_to_basket import (
    IThorLivingRoomCandleToBasket,
)
from robocasa.environments.ithor_living_room.tasks.atomic.tissue_box_to_basket import (
    IThorLivingRoomTissueBoxToBasket,
)
from robocasa.environments.ithor_living_room.tasks.atomic.book_into_basket import (
    IThorLivingRoomBookIntoBasket,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
    get_ithor_livingroom_task_specs,
)


def _table_operation_specs(atomic_task, task_name):
    """Combine atomic and navigation defaults with task-specific YAML poses."""
    specs = {}
    overrides = get_ithor_livingroom_task_specs(task_name)
    for scene_id in IThorLivingRoomApproachTable.SUPPORTED_SCENE_IDS:
        scene = f"FloorPlan{scene_id}"
        spec = get_ithor_livingroom_task_spec(scene, atomic_task.SPEC_TASK)
        navigation = get_ithor_livingroom_task_spec(scene, "approach_table")
        spec["operation_robot_pose"] = spec["robot_pose"]
        spec["robot_pose"] = navigation["robot_pose"]
        spec["table_fixture"] = navigation["target_fixture"]
        spec.update(overrides.get(scene, {}))
        specs[scene] = spec
    return specs


class _ApproachTableGateMixin(CompositeStageReporterMixin):
    """Two ordered stages with independent consecutive hold counters."""

    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L2"
    COMPOSITE_ORDERING = "sequential"
    SUPPORTED_SCENE_IDS = IThorLivingRoomApproachTable.SUPPORTED_SCENE_IDS
    APPROACH_DISTANCE = 0.35
    MINIMUM_START_DISTANCE = 0.50
    APPROACH_HOLD_STEPS = 5
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = deepcopy(self.SCENE_SPECS[self.scene_name])
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        # BookIntoBasket sets its jitter from YAML; keep composite starts fixed.
        self.INITIAL_OBJECT_JITTER_RADIUS = 0.0
        stable_id = self._get_task_spec()["table_fixture"]
        self.approach_table_fixture = self.get_fixture(id=stable_id)
        if (
            self.approach_table_fixture is None
            or self.approach_table_fixture.stable_id != stable_id
        ):
            raise LookupError(f"{self.scene_name}: missing exact table {stable_id}")
        support = getattr(self, "base_support_fixture", self.approach_table_fixture)
        if support.stable_id != stable_id:
            raise ValueError(f"{self.scene_name}: operation is on a different table")
        self.fixture_refs["approach_table"] = self.approach_table_fixture

    def _reset_internal(self):
        super()._reset_internal()
        self._operation_hold_steps = max(1, int(round(self.control_freq)))
        self._reset_task_progress()
        self._approach_start_distance = self.robot_fixture_collision_distance(
            self.approach_table_fixture, max_distance=5.0
        )
        if (
            not np.isfinite(self._approach_start_distance)
            or self._approach_start_distance < self.MINIMUM_START_DISTANCE
            or not self.robot_footprint_is_on_floor(self._robot_base_xy(), radius=0.25)
            or self.count_body_collisions()
        ):
            raise RuntimeError(f"{self.scene_name}: invalid distant table spawn")

    def _reset_task_progress(self):
        self._approach_stage = 0
        self._approach_hold = 0
        self._operation_hold = 0
        self._operation_armed = False
        self._completion_reported = False
        self._lift_hold_steps = 0
        self._success_hold_steps = 0

    def _robot_near_table(self):
        distance = self.robot_fixture_collision_distance(
            self.approach_table_fixture, max_distance=1.5
        )
        return bool(np.isfinite(distance) and distance <= self.APPROACH_DISTANCE)

    def _post_action(self, action):
        ret = super()._post_action(action)
        valid = self._raw_operation_success()
        if self._approach_stage == 0:
            near = self._robot_near_table()
            self._approach_hold = self._approach_hold + 1 if near else 0
            if self._approach_hold >= self.APPROACH_HOLD_STEPS:
                self._approach_stage = 1
                # A goal already achieved before arrival must be undone first.
                self._operation_armed = not valid
                self._report_stage_complete(1, 2, next_stage=self.OPERATION_STAGE)
            self._operation_hold = 0
            return ret

        if not valid:
            self._operation_armed = True
        self._operation_hold = (
            self._operation_hold + 1 if valid and self._operation_armed else 0
        )
        if self._check_success() and not self._completion_reported:
            self._completion_reported = True
            self._report_stage_complete(2, 2)
        return ret

    def _check_success(self):
        return bool(
            self._approach_stage == 1
            and self._operation_hold >= self._operation_hold_steps
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        spec = self._get_task_spec()
        ep_meta.update(
            lang=self.TASK_LANG,
            composite_stages=["walk_to_table", self.OPERATION_STAGE],
            composite_ordering=self.COMPOSITE_ORDERING,
            composite_stage=self._approach_stage,
            approach_fixture=self.approach_table_fixture.stable_id,
            approach_distance=self.APPROACH_DISTANCE,
            approach_hold_steps=self.APPROACH_HOLD_STEPS,
            approach_start_distance=float(self._approach_start_distance),
            operation_robot_pose=deepcopy(spec["operation_robot_pose"]),
            operation_hold_seconds=1.0,
            success_definition=self.SUCCESS_DEFINITION,
        )
        return ep_meta


class _ApproachTableLiftMixin(_ApproachTableGateMixin):
    ATOMIC_SKILL = "NavigateThenLiftObject"
    COMPOSITE_KIND = "navigate_then_lift"
    SUCCESS_DEFINITION = (
        "Reach the named table, then keep the object at least 0.30 m above "
        "its reset height for 1.0 s."
    )

    def _raw_operation_success(self):
        return bool(self._target_root_z() >= self._lift_initial_z + 0.30)


class _ApproachTableBasketMixin(_ApproachTableGateMixin):
    ATOMIC_SKILL = "NavigateThenPickPlace"
    COMPOSITE_KIND = "navigate_then_place_in_basket"
    SUCCESS_DEFINITION = (
        "Reach the named table, then keep the object touching the basket, "
        "within the atomic task's horizontal root-distance tolerance and "
        "with all end effectors beyond its release distance, for 1.0 s."
    )

    def _raw_operation_success(self):
        """Use the corresponding atomic placement predicate without extra gates."""
        return self._raw_success()


class IThorLivingRoomApproachTableAndLiftDumbbell(
    _ApproachTableLiftMixin, IThorLivingRoomLiftDumbbell
):
    SPEC_TASK = "approach_table_and_lift_dumbbell"
    TASK_LANG = "walk to the table, then lift the dumbbell by 30 cm"
    OPERATION_STAGE = "lift_dumbbell"
    SCENE_SPECS = _table_operation_specs(IThorLivingRoomLiftDumbbell, SPEC_TASK)


class IThorLivingRoomApproachTableAndLiftStatue(
    _ApproachTableLiftMixin, IThorLivingRoomLiftStatue
):
    SPEC_TASK = "approach_table_and_lift_statue"
    TASK_LANG = "walk to the table, then lift the statue by 30 cm"
    OPERATION_STAGE = "lift_statue"
    SCENE_SPECS = _table_operation_specs(IThorLivingRoomLiftStatue, SPEC_TASK)


class IThorLivingRoomApproachTableAndLiftWateringCan(
    _ApproachTableLiftMixin, IThorLivingRoomLiftWateringCan
):
    SPEC_TASK = "approach_table_and_lift_watering_can"
    TASK_LANG = "walk to the table, then lift the watering can by 30 cm"
    OPERATION_STAGE = "lift_watering_can"
    SCENE_SPECS = _table_operation_specs(IThorLivingRoomLiftWateringCan, SPEC_TASK)


class IThorLivingRoomApproachTableAndPutRemoteInBasket(
    _ApproachTableBasketMixin, IThorLivingRoomRemoteToBasket
):
    SPEC_TASK = "approach_table_and_put_remote_in_basket"
    TASK_LANG = (
        "walk to the table, then put the remote control into the basket on the table"
    )
    OPERATION_STAGE = "put_remote_in_basket"
    SCENE_SPECS = _table_operation_specs(IThorLivingRoomRemoteToBasket, SPEC_TASK)


class IThorLivingRoomApproachTableAndPutCandleInBasket(
    _ApproachTableBasketMixin, IThorLivingRoomCandleToBasket
):
    SPEC_TASK = "approach_table_and_put_candle_in_basket"
    TASK_LANG = "walk to the table, then put the candle into the basket on the table"
    OPERATION_STAGE = "put_candle_in_basket"
    SCENE_SPECS = _table_operation_specs(IThorLivingRoomCandleToBasket, SPEC_TASK)


class IThorLivingRoomApproachTableAndPutTissueBoxInBasket(
    _ApproachTableBasketMixin, IThorLivingRoomTissueBoxToBasket
):
    SPEC_TASK = "approach_table_and_put_tissue_box_in_basket"
    TASK_LANG = (
        "walk to the table, then put the tissue box into the basket on the table"
    )
    OPERATION_STAGE = "put_tissue_box_in_basket"
    SCENE_SPECS = _table_operation_specs(IThorLivingRoomTissueBoxToBasket, SPEC_TASK)


class IThorLivingRoomApproachTableAndPutBookInBasket(
    _ApproachTableBasketMixin, IThorLivingRoomBookIntoBasket
):
    SPEC_TASK = "approach_table_and_put_book_in_basket"
    TASK_LANG = "walk to the table, then put the book into the basket on the table"
    OPERATION_STAGE = "put_book_in_basket"
    SCENE_SPECS = _table_operation_specs(IThorLivingRoomBookIntoBasket, SPEC_TASK)


__all__ = [
    "IThorLivingRoomApproachTableAndLiftDumbbell",
    "IThorLivingRoomApproachTableAndLiftStatue",
    "IThorLivingRoomApproachTableAndLiftWateringCan",
    "IThorLivingRoomApproachTableAndPutRemoteInBasket",
    "IThorLivingRoomApproachTableAndPutCandleInBasket",
    "IThorLivingRoomApproachTableAndPutTissueBoxInBasket",
    "IThorLivingRoomApproachTableAndPutBookInBasket",
]
