"""Lift exact native Bathroom objects from calibrated work surfaces."""

from __future__ import annotations

from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_specs,
    get_supported_scene_ids,
)

from .brush_support import _BathroomScrubBrushSupportMixin
from .object_manipulation import _BathroomNativeObjectTask


class _BathroomNativeLiftTask(_BathroomNativeObjectTask):
    """Private implementation shared by exact-object lift tasks."""

    ATOMIC_SKILL = "LiftObject"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "lift"
    HOLD_STEPS = 5
    MIN_LIFT_DELTA = 0.30
    GRASP_HEIGHT_ALLOWANCE = 0.10

    SOURCE_LABEL = None

    def _source_root_z(self):
        body_id = self.sim.model.body_name2id(self.source_fixture.body_name)
        return float(self.sim.data.body_xpos[body_id, 2])

    def _reset_internal(self):
        super()._reset_internal()
        self._initial_source_z = self._source_root_z()
        self._grasped_near_source = False
        self._success_hold = 0

    def _raw_success(self):
        grasped = self._g1_hand_grasping_fixture(self.source_fixture)
        height = self._source_root_z()
        if (
            grasped
            and height
            <= self._initial_source_z + self.GRASP_HEIGHT_ALLOWANCE
        ):
            self._grasped_near_source = True
        return bool(
            self._grasped_near_source
            and grasped
            and height >= self._initial_source_z + self.MIN_LIFT_DELTA
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold = (
            self._success_hold + 1 if self._raw_success() else 0
        )
        return ret

    def _check_success(self):
        return self._success_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(self._native_object_ep_meta())
        ep_meta.update(
            {
                "lang": self.scene_task_spec()["lang"],
                "success_definition": (
                    f"the named {self.SOURCE_LABEL} must first be grasped at "
                    f"its reset height, then remain grasped at least "
                    f"{self.MIN_LIFT_DELTA:.2f} m higher for "
                    f"{self.HOLD_STEPS} control steps"
                ),
            }
        )
        return ep_meta


class IThorBathroomLiftSoapBottle(
    _BathroomNativeLiftTask,
    IThorBathroomEnv,
):
    """Pick up the exact native soap bottle."""

    SPEC_TASK = "lift_soap_bottle"
    SOURCE_LABEL = "soap bottle"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK,
        required=(
            "source_fixture",
            "surface_fixture",
            "surface_region",
            "source_xy",
            "remove_fixtures",
            "auxiliary_placements",
            "robot_pose",
            "lang",
        ),
    )


class IThorBathroomLiftToiletPaperRoll(
    _BathroomNativeLiftTask,
    IThorBathroomEnv,
):
    """Pick up the exact native full toilet-paper roll."""

    SPEC_TASK = "lift_toilet_paper_roll"
    SOURCE_LABEL = "toilet-paper roll"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK,
        required=(
            "source_fixture",
            "surface_fixture",
            "surface_region",
            "source_xy",
            "remove_fixtures",
            "auxiliary_placements",
            "robot_pose",
            "lang",
        ),
    )


class IThorBathroomLiftSprayBottle(
    _BathroomNativeLiftTask,
    IThorBathroomEnv,
):
    """Pick up the exact native spray bottle."""

    SPEC_TASK = "lift_spray_bottle"
    SOURCE_LABEL = "spray bottle"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK,
        required=(
            "source_fixture",
            "surface_fixture",
            "surface_region",
            "source_xy",
            "remove_fixtures",
            "auxiliary_placements",
            "robot_pose",
            "lang",
        ),
    )


class IThorBathroomLiftScrubBrush(
    _BathroomScrubBrushSupportMixin,
    _BathroomNativeLiftTask,
    IThorBathroomEnv,
):
    """Pick up the exact native scrub brush."""

    SPEC_TASK = "lift_scrub_brush"
    SOURCE_LABEL = "scrub brush"
    SOURCE_STOOL_FIELD = "atomic_source_stool"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK,
        required=(
            "source_fixture",
            "surface_fixture",
            "surface_region",
            "source_xy",
            "source_quat",
            "remove_fixtures",
            "auxiliary_placements",
            "robot_pose",
            "lang",
        ),
    )

    # These overrides belong only to the atomic task. Composite consumers of
    # lift_scrub_brush retain their original bathtub-rim calibration.
    for _scene_name, _spec in SCENE_SPECS.items():
        _layout = _spec.get(SOURCE_STOOL_FIELD)
        if _layout is not None:
            _surface_id = f"{_scene_name}:spawned:brush_stool"
            _spec.update(
                surface_fixture=_surface_id,
                surface_region=f"{_surface_id}:region:support",
                source_xy=_layout["xy"],
                source_quat=[1.0, 0.0, 0.0, 0.0],
                robot_pose=_layout["robot_pose"],
                remove_fixtures=_layout["remove_fixtures"],
            )
    del _scene_name, _spec, _layout, _surface_id


__all__ = [
    "IThorBathroomLiftScrubBrush",
    "IThorBathroomLiftSoapBottle",
    "IThorBathroomLiftSprayBottle",
    "IThorBathroomLiftToiletPaperRoll",
]
