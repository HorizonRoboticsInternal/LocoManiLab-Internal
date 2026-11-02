"""Two-stage open/close cycles composed from Bathroom articulation atomics."""

from __future__ import annotations

import copy

from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_spec,
)

from ..articulation import _BathroomToiletLid, _StableCloseToiletLid
from ..shower_door import _BathroomShowerDoor
from ..storage_articulation import _BathroomStorageArticulation


_ALL_SCENE_IDS = (401, 405, 407, 415, 418)


def _articulation_cycle_specs(scene_ids, first_task, lang):
    specs = {}
    for scene_id in scene_ids:
        scene_name = f"FloorPlan{scene_id}"
        spec = copy.deepcopy(
            get_ithor_bathroom_task_spec(scene_name, first_task)
        )
        spec["lang"] = lang
        specs[scene_name] = spec
    return specs


class _BathroomArticulationCycleMixin(CompositeStageReporterMixin):
    """Require both opposite joint thresholds in the declared order."""

    ATOMIC_SKILL = "ArticulationCycle"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "articulation_cycle"
    COMPOSITE_ORDERING = ("first_joint_state", "second_joint_state")
    FIRST_BEHAVIOR = None
    SECOND_BEHAVIOR = None
    FIRST_HOLD_STEPS = 5
    HOLD_STEPS = 5

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_task_progress()

    def _reset_task_progress(self):
        # Fixed-scene collection restores physics without _reset_internal().
        self._success_hold = 0
        self._cycle_stage = 0
        self._first_state_hold = 0
        self._second_state_hold = 0

    def _joint_matches(self, behavior):
        fractions = [joint.fraction(self.sim) for joint in self._success_joints()]
        if behavior == "open":
            return all(
                fraction >= self.SUCCESS_OPEN_FRACTION for fraction in fractions
            )
        if behavior == "close":
            return all(
                fraction <= self.SUCCESS_CLOSE_FRACTION for fraction in fractions
            )
        raise ValueError(f"unsupported articulation behavior {behavior!r}")

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self._cycle_stage == 0:
            first_done = self._joint_matches(self.FIRST_BEHAVIOR)
            self._first_state_hold = (
                self._first_state_hold + 1 if first_done else 0
            )
            if self._first_state_hold >= self.FIRST_HOLD_STEPS:
                self._cycle_stage = 1
                self._success_hold = 0
                self._report_stage_complete(
                    1,
                    2,
                    next_stage=f"{self.SECOND_BEHAVIOR}_the_same_joint",
                )
            self._second_state_hold = 0
            return ret

        second_done = self._joint_matches(self.SECOND_BEHAVIOR)
        self._second_state_hold = (
            self._second_state_hold + 1 if second_done else 0
        )
        return ret

    def _check_success(self):
        return bool(
            self._cycle_stage == 1
            and self._second_state_hold >= self.HOLD_STEPS
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            {
                "lang": self.scene_task_spec()["lang"],
                "stage_order": [
                    f"{self.FIRST_BEHAVIOR} the named joint to its atomic threshold",
                    f"{self.SECOND_BEHAVIOR} the same joint to its atomic threshold",
                ],
            }
        )
        return ep_meta


class IThorBathroomOpenThenCloseToiletLid(
    _BathroomArticulationCycleMixin,
    _BathroomToiletLid,
    IThorBathroomEnv,
):
    BEHAVIOR = FIRST_BEHAVIOR = "open"
    SECOND_BEHAVIOR = "close"
    SPEC_TASK = "open_then_close_toilet_lid"
    COMPOSITE_ORDERING = ("open_toilet_lid", "close_toilet_lid")
    SUPPORTED_SCENE_IDS = _ALL_SCENE_IDS
    SCENE_SPECS = _articulation_cycle_specs(
        SUPPORTED_SCENE_IDS,
        "toilet_lid",
        "open the toilet lid, then close it",
    )


class IThorBathroomCloseThenOpenToiletLid(
    _BathroomArticulationCycleMixin,
    _StableCloseToiletLid,
    _BathroomToiletLid,
    IThorBathroomEnv,
):
    BEHAVIOR = FIRST_BEHAVIOR = "close"
    SECOND_BEHAVIOR = "open"
    CLOSE_START_FRACTION = 1.00
    SPEC_TASK = "close_then_open_toilet_lid"
    COMPOSITE_ORDERING = ("close_toilet_lid", "open_toilet_lid")
    SUPPORTED_SCENE_IDS = _ALL_SCENE_IDS
    SCENE_SPECS = _articulation_cycle_specs(
        SUPPORTED_SCENE_IDS,
        "toilet_lid",
        "close the toilet lid, then open it",
    )


class IThorBathroomOpenThenCloseShowerDoor(
    _BathroomArticulationCycleMixin,
    _BathroomShowerDoor,
    IThorBathroomEnv,
):
    BEHAVIOR = FIRST_BEHAVIOR = "open"
    SECOND_BEHAVIOR = "close"
    SPEC_TASK = "open_then_close_shower_door"
    COMPOSITE_ORDERING = ("open_shower_door", "close_shower_door")
    SUPPORTED_SCENE_IDS = (407, 418)
    SCENE_SPECS = _articulation_cycle_specs(
        SUPPORTED_SCENE_IDS,
        "open_shower_door",
        "open the shower door, then close it",
    )


class IThorBathroomCloseThenOpenShowerDoor(
    _BathroomArticulationCycleMixin,
    _BathroomShowerDoor,
    IThorBathroomEnv,
):
    BEHAVIOR = FIRST_BEHAVIOR = "close"
    SECOND_BEHAVIOR = "open"
    SPEC_TASK = "close_then_open_shower_door"
    COMPOSITE_ORDERING = ("close_shower_door", "open_shower_door")
    SUPPORTED_SCENE_IDS = (407, 418)
    SCENE_SPECS = _articulation_cycle_specs(
        SUPPORTED_SCENE_IDS,
        "close_shower_door",
        "close the shower door, then open it",
    )


class IThorBathroomOpenThenCloseCabinetDoor(
    _BathroomArticulationCycleMixin,
    _BathroomStorageArticulation,
    IThorBathroomEnv,
):
    BEHAVIOR = FIRST_BEHAVIOR = "open"
    SECOND_BEHAVIOR = "close"
    SPEC_TASK = "open_then_close_cabinet_door"
    TARGET_JOINT_TYPES = ("hinge",)
    COMPOSITE_ORDERING = ("open_cabinet_door", "close_cabinet_door")
    SUPPORTED_SCENE_IDS = (405, 407, 418)
    SCENE_SPECS = _articulation_cycle_specs(
        SUPPORTED_SCENE_IDS,
        "open_cabinet_door",
        "open the bathroom cabinet door, then close it",
    )


class IThorBathroomCloseThenOpenCabinetDoor(
    _BathroomArticulationCycleMixin,
    _BathroomStorageArticulation,
    IThorBathroomEnv,
):
    BEHAVIOR = FIRST_BEHAVIOR = "close"
    SECOND_BEHAVIOR = "open"
    SPEC_TASK = "close_then_open_cabinet_door"
    TARGET_JOINT_TYPES = ("hinge",)
    COMPOSITE_ORDERING = ("close_cabinet_door", "open_cabinet_door")
    SUPPORTED_SCENE_IDS = (405, 407, 418)
    SCENE_SPECS = _articulation_cycle_specs(
        SUPPORTED_SCENE_IDS,
        "close_cabinet_door",
        "close the bathroom cabinet door, then open it",
    )


class IThorBathroomOpenThenCloseLowerDrawer(
    _BathroomArticulationCycleMixin,
    _BathroomStorageArticulation,
    IThorBathroomEnv,
):
    BEHAVIOR = FIRST_BEHAVIOR = "open"
    SECOND_BEHAVIOR = "close"
    SPEC_TASK = "open_then_close_lower_drawer"
    TARGET_JOINT_TYPES = ("slide",)
    COMPOSITE_ORDERING = ("open_lower_drawer", "close_lower_drawer")
    SUPPORTED_SCENE_IDS = (415,)
    SCENE_SPECS = _articulation_cycle_specs(
        SUPPORTED_SCENE_IDS,
        "open_lower_drawer",
        "open the lower bathroom drawer, then close it",
    )


class IThorBathroomCloseThenOpenLowerDrawer(
    _BathroomArticulationCycleMixin,
    _BathroomStorageArticulation,
    IThorBathroomEnv,
):
    BEHAVIOR = FIRST_BEHAVIOR = "close"
    SECOND_BEHAVIOR = "open"
    SPEC_TASK = "close_then_open_lower_drawer"
    TARGET_JOINT_TYPES = ("slide",)
    COMPOSITE_ORDERING = ("close_lower_drawer", "open_lower_drawer")
    SUPPORTED_SCENE_IDS = (415,)
    SCENE_SPECS = _articulation_cycle_specs(
        SUPPORTED_SCENE_IDS,
        "close_lower_drawer",
        "close the lower bathroom drawer, then open it",
    )


__all__ = [
    "IThorBathroomCloseThenOpenCabinetDoor",
    "IThorBathroomCloseThenOpenLowerDrawer",
    "IThorBathroomCloseThenOpenShowerDoor",
    "IThorBathroomCloseThenOpenToiletLid",
    "IThorBathroomOpenThenCloseCabinetDoor",
    "IThorBathroomOpenThenCloseLowerDrawer",
    "IThorBathroomOpenThenCloseShowerDoor",
    "IThorBathroomOpenThenCloseToiletLid",
]
