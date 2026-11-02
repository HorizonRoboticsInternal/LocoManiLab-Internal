"""Bathroom navigation-to-manipulation tasks composed from atomics."""

from __future__ import annotations

import copy

import numpy as np

from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_spec,
)

from ..articulation import _BathroomToiletLid, _StableCloseToiletLid
from ..bathtub_tasks import _BathroomSpongeInBathtubTask
from ..lift import _BathroomNativeLiftTask
from ..pick_place import _BathroomToiletPaperRollIntoBinTask
from ..reorientation import _BathroomNativeUprightTask
from ..shower_door import _BathroomShowerDoor
from ..storage_articulation import _BathroomStorageArticulation
from .brush_support import _BathroomScrubBrushLiftTask


_ALL_SCENE_IDS = (401, 405, 407, 415, 418)
_NAVIGATION_SPEC_FIELDS = (
    "fixture",
    "start_pose",
    "minimum_start_distance",
    "remove_fixtures",
    "lang",
)


def _approach_operation_specs(
    scene_ids,
    navigation_task,
    operation_task,
    lang,
):
    specs = {}
    for scene_id in scene_ids:
        scene_name = f"FloorPlan{scene_id}"
        operation = copy.deepcopy(
            get_ithor_bathroom_task_spec(scene_name, operation_task)
        )
        navigation = copy.deepcopy(
            get_ithor_bathroom_task_spec(
                scene_name,
                navigation_task,
                required=_NAVIGATION_SPEC_FIELDS,
            )
        )
        operation["navigation"] = navigation
        operation["lang"] = lang
        specs[scene_name] = operation
    return specs


def _garbage_can_toilet_roll_specs():
    """Compose atomics where available and keep missing layouts task-local."""

    lang = "walk to the garbage can, then put the toilet-paper roll inside"
    specs = _approach_operation_specs(
        (401, 405, 407),
        "approach_garbage_can",
        "toilet_paper_roll_into_bin",
        lang,
    )
    # The composite begins from its garbage-can navigation stance, not the
    # atomic manipulation stance. Keep FloorPlan401 unchanged as the frozen
    # baseline, but use the composite's target-specific stance elsewhere.
    for scene_name in ("FloorPlan405", "FloorPlan407"):
        specs[scene_name]["robot_pose"] = copy.deepcopy(
            specs[scene_name]["navigation"]["start_pose"]
        )
    # FloorPlan415 and FloorPlan418 have calibrated toilet-roll lifts but no
    # standalone bin task. Derive their source workspaces from the frozen
    # atomics, then add only the composite's bin placement. This preserves all
    # atomic task specs and leaves other composites unchanged.
    local_bin_layouts = {
        "FloorPlan415": {
            "bin_fixture": "FloorPlan415:bin_15_13",
            "bin_region": "FloorPlan415:bin_15_13:region:receptacle:0",
            "bin_region_z_offset": -0.04,
            "bin_xy": [-2.10, -2.70],
        },
        "FloorPlan418": {
            "bin_fixture": "FloorPlan418:bin_18_0",
            "bin_region": "FloorPlan418:bin_18_0:region:receptacle:0",
            "bin_region_z_offset": -0.04,
            "bin_xy": [-1.90, -3.75],
        },
    }
    # FloorPlan415 moves the bin and robot together toward the paper roll.
    # FloorPlan418 only moves the bin against the south wall; its robot
    # pose stays unchanged. Both poses face their relocated bin centers.
    composite_bin_start_poses = {
        "FloorPlan415": {"xy": [-2.10, -3.70], "yaw": 1.570796},
        "FloorPlan418": {"xy": [-1.90, -2.25], "yaw": -1.570796},
    }
    for scene_name, bin_layout in local_bin_layouts.items():
        operation = copy.deepcopy(
            get_ithor_bathroom_task_spec(scene_name, "lift_toilet_paper_roll")
        )
        operation.update(bin_layout)
        operation["source_quat"] = [0.0, 0.0, 0.70710678, 0.70710678]
        operation["navigation"] = copy.deepcopy(
            get_ithor_bathroom_task_spec(
                scene_name,
                "approach_garbage_can",
                required=_NAVIGATION_SPEC_FIELDS,
            )
        )
        operation["navigation"]["start_pose"] = copy.deepcopy(
            composite_bin_start_poses[scene_name]
        )
        operation["robot_pose"] = copy.deepcopy(operation["navigation"]["start_pose"])
        operation["lang"] = lang
        specs[scene_name] = operation
    return specs


class _ApproachThenOperationMixin(CompositeStageReporterMixin):
    """Latch one unchanged navigation atomic before an unchanged operation."""

    ATOMIC_SKILL = "NavigateThenManipulate"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "navigate_then_manipulate"
    COMPOSITE_ORDERING = ("approach_fixture", "complete_operation")
    APPROACH_HOLD_STEPS = 5
    HOLD_STEPS = 5
    SUCCESS_DISTANCE = 0.35
    INITIAL_STATE_TOLERANCE = 0.03
    INITIAL_ROBOT_JITTER_RADIUS = 0.0

    def _navigation_spec(self):
        return self.scene_task_spec()["navigation"]

    def _approach_start_pose(self):
        pose = self._navigation_spec()["start_pose"]
        if not isinstance(pose, dict) or set(pose) != {"xy", "yaw"}:
            raise ValueError(
                f"{self.scene_name}: navigation start_pose needs exactly xy and yaw"
            )
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        if xy.shape != (2,) or not np.all(np.isfinite(xy)) or not np.isfinite(yaw):
            raise ValueError(f"{self.scene_name}: invalid approach start pose")
        return xy, yaw

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        navigation = self._navigation_spec()
        stable_id = navigation["fixture"]
        self.approach_fixture = self.get_fixture(id=stable_id)
        if (
            self.approach_fixture is None
            or self.approach_fixture.stable_id != stable_id
        ):
            raise LookupError(
                f"{self.scene_name}: missing exact approach fixture {stable_id}"
            )
        removal_ids = navigation["remove_fixtures"]
        if not isinstance(removal_ids, (list, tuple)) or isinstance(removal_ids, str):
            raise ValueError("navigation remove_fixtures must be a list")
        if len(removal_ids) != len(set(removal_ids)):
            raise ValueError("navigation remove_fixtures must be unique")
        protected_ids = {
            fixture.stable_id
            for fixture in self.fixture_refs.values()
            if hasattr(fixture, "stable_id")
        }
        protected_ids.add(stable_id)
        already_removed = set(getattr(self, "_removed_fixture_ids", ()))
        already_removed.update(getattr(self, "rim_clutter_fixture_ids", ()))
        removals = []
        for removal_id in removal_ids:
            if removal_id in protected_ids:
                raise ValueError(
                    f"{self.scene_name}: approach removal overlaps task fixture "
                    f"{removal_id}"
                )
            if removal_id in already_removed:
                continue
            fixture = self.get_fixture(id=removal_id)
            if fixture is None or fixture.stable_id != removal_id:
                raise LookupError(
                    f"{self.scene_name}: missing approach removal {removal_id}"
                )
            removals.append(fixture)
        if removals:
            self.remove_semantic_fixtures(removals)
        self.removed_approach_fixture_ids = list(removal_ids)
        self.fixture_refs["approach_target"] = self.approach_fixture
        xy, yaw = self._approach_start_pose()
        self.set_robot_base_pose([xy[0], xy[1], 0.0], yaw)

    def _setup_scene(self):
        super()._setup_scene()
        xy, yaw = self._approach_start_pose()
        if hasattr(self, "_robot_free_joint_z"):
            delattr(self, "_robot_free_joint_z")
        if not (
            self.robot_footprint_is_on_floor(xy, radius=0.25)
            and self._set_robot_free_pose(xy, yaw)
            and self._calibrate_robot_free_joint_height()
        ):
            raise RuntimeError(f"{self.scene_name}: invalid approach start pose")
        self.sim.forward()
        if self.count_body_collisions():
            raise RuntimeError(f"{self.scene_name}: approach start collides")
        distance = self.robot_fixture_collision_distance(
            self.approach_fixture, max_distance=5.0
        )
        minimum = min(
            0.70,
            float(self._navigation_spec()["minimum_start_distance"]),
        )
        if not np.isfinite(distance) or distance < minimum:
            raise RuntimeError(
                f"{self.scene_name}: approach starts {distance:.3f} m from "
                f"its target; expected at least {minimum:.3f} m"
            )
        if distance <= self.SUCCESS_DISTANCE:
            raise RuntimeError(f"{self.scene_name}: approach starts at its goal")
        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        self._approach_start_distance = float(distance)

    def _reset_internal(self):
        super()._reset_internal()
        self._operation_initial_state = self._capture_operation_initial_state()
        self._reset_task_progress()

    def _reset_task_progress(self):
        # Fixed-scene collection restores physics without _reset_internal().
        parent_reset = getattr(super(), "_reset_task_progress", None)
        if callable(parent_reset):
            parent_reset()
        self._approach_stage = 0
        self._approach_hold = 0
        self._composite_operation_hold = 0

    def _capture_operation_initial_state(self):
        raise NotImplementedError

    def _operation_initial_state_intact(self):
        raise NotImplementedError

    def _near_approach_fixture(self):
        distance = self.robot_fixture_collision_distance(
            self.approach_fixture, max_distance=1.0
        )
        return bool(np.isfinite(distance) and distance <= self.SUCCESS_DISTANCE)

    def _on_approach_complete(self):
        if hasattr(self, "_success_hold"):
            self._success_hold = 0
        if hasattr(self, "_success_hold_steps"):
            self._success_hold_steps = 0
        if hasattr(self, "_grasped_near_source"):
            self._grasped_near_source = False

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self._approach_stage == 0:
            ready = (
                self._near_approach_fixture() and self._operation_initial_state_intact()
            )
            self._approach_hold = self._approach_hold + 1 if ready else 0
            if self._approach_hold >= self.APPROACH_HOLD_STEPS:
                self._approach_stage = 1
                self._on_approach_complete()
                self._report_stage_complete(
                    1, 2, next_stage="complete_the_atomic_operation"
                )
            self._composite_operation_hold = 0
            return ret

        valid = self._raw_success()
        self._composite_operation_hold = (
            self._composite_operation_hold + 1 if valid else 0
        )
        return ret

    def _check_success(self):
        return bool(
            self._approach_stage == 1
            and self._composite_operation_hold >= self.HOLD_STEPS
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            {
                "lang": self.scene_task_spec()["lang"],
                "approach_fixture": self.approach_fixture.stable_id,
                "approach_distance": self.SUCCESS_DISTANCE,
                "approach_start_distance": self._approach_start_distance,
                "removed_approach_fixtures": list(self.removed_approach_fixture_ids),
                "stage_order": [
                    "reach the atomic navigation target while the operation "
                    "state remains unchanged",
                    "complete the atomic operation and hold its result",
                ],
            }
        )
        return ep_meta


class _ApproachThenArticulationMixin(_ApproachThenOperationMixin):
    """Use the exact target-joint fraction as the preserved reset state."""

    def _capture_operation_initial_state(self):
        return float(self.target_semantic_joint.fraction(self.sim))

    def _operation_initial_state_intact(self):
        fraction = float(self.target_semantic_joint.fraction(self.sim))
        return bool(
            abs(fraction - self._operation_initial_state)
            <= self.INITIAL_STATE_TOLERANCE
            and not self._raw_success()
        )


class _ApproachThenNativeFixtureMixin(_ApproachThenOperationMixin):
    """Preserve an exact native object's reset center until navigation."""

    def _capture_operation_initial_state(self):
        return np.asarray(
            self.source_fixture.world_center(self.sim), dtype=float
        ).copy()

    def _operation_initial_state_intact(self):
        center = np.asarray(self.source_fixture.world_center(self.sim), dtype=float)
        return bool(
            np.linalg.norm(center - self._operation_initial_state)
            <= self.INITIAL_STATE_TOLERANCE
            and self._fixture_released(self.source_fixture)
            and not self._raw_success()
        )


class _ApproachThenSpawnedObjectMixin(_ApproachThenOperationMixin):
    """Preserve a task-spawned object's reset center until navigation."""

    def _capture_operation_initial_state(self):
        return np.asarray(self.object_pos(self.SOURCE_NAME), dtype=float).copy()

    def _operation_initial_state_intact(self):
        center = np.asarray(self.object_pos(self.SOURCE_NAME), dtype=float)
        return bool(
            np.linalg.norm(center - self._operation_initial_state)
            <= self.INITIAL_STATE_TOLERANCE
            and self._source_is_released()
            and not self._raw_success()
        )


class _CompositeSpongeStartPoseMixin:
    """Use an explicit, composite-local rim stance where the atomic is frozen."""

    def _place_robot_at_north_rim(self):
        pose = self.scene_task_spec().get("composite_operation_robot_pose")
        if pose is None:
            return super()._place_robot_at_north_rim()
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        self._clear_robot_height_cache()
        if not (
            xy.shape == (2,)
            and self.robot_footprint_is_on_floor(xy, radius=0.25)
            and self._set_robot_free_pose(xy, yaw)
            and self._calibrate_robot_free_joint_height()
            and self.count_body_collisions() == 0
        ):
            return False
        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        return True


class IThorBathroomApproachToiletAndOpenToiletLid(
    _ApproachThenArticulationMixin,
    _BathroomToiletLid,
    IThorBathroomEnv,
):
    BEHAVIOR = "open"
    SPEC_TASK = "approach_toilet_and_open_toilet_lid"
    COMPOSITE_ORDERING = ("approach_toilet", "open_toilet_lid")
    SUPPORTED_SCENE_IDS = _ALL_SCENE_IDS
    SCENE_SPECS = _approach_operation_specs(
        SUPPORTED_SCENE_IDS,
        "approach_toilet",
        "toilet_lid",
        "walk to the toilet, then open the toilet lid",
    )


class IThorBathroomApproachToiletAndCloseToiletLid(
    _ApproachThenArticulationMixin,
    _StableCloseToiletLid,
    _BathroomToiletLid,
    IThorBathroomEnv,
):
    BEHAVIOR = "close"
    # Preserve the existing fully-open start outside the calibrated 401 case.
    CLOSE_START_FRACTION = 1.00
    SPEC_TASK = "approach_toilet_and_close_toilet_lid"
    COMPOSITE_ORDERING = ("approach_toilet", "close_toilet_lid")
    SUPPORTED_SCENE_IDS = _ALL_SCENE_IDS
    SCENE_SPECS = _approach_operation_specs(
        SUPPORTED_SCENE_IDS,
        "approach_toilet",
        "toilet_lid",
        "walk to the toilet, then close the toilet lid",
    )

    def _setup_scene(self):
        # At 100% the 401 lid leans back past vertical, so a downward push
        # cannot start closing it. Keep this override instance- and scene-local
        # to protect already-collected atomics and other composite tasks.
        self.CLOSE_START_FRACTION = (
            0.80
            if self.scene_name == "FloorPlan401"
            else type(self).CLOSE_START_FRACTION
        )
        super()._setup_scene()
        if self.scene_name == "FloorPlan401":
            joint_id = self.sim.model.joint_name2id(self.target_joint)
            dof_address = int(self.sim.model.jnt_dofadr[joint_id])
            # Resolve static friction tightly enough to prevent gravity creep
            # during navigation, without increasing friction or locking the lid.
            self.sim.model.dof_solref[dof_address] = (0.002, 1.0)
            self.sim.model.dof_solimp[dof_address] = (0.9999, 0.9999, 0.001, 0.5, 2.0)
            self.sim.forward()


class IThorBathroomNavigateToBathingAreaAndOpenShowerDoor(
    _ApproachThenArticulationMixin,
    _BathroomShowerDoor,
    IThorBathroomEnv,
):
    BEHAVIOR = "open"
    SPEC_TASK = "navigate_to_bathing_area_and_open_shower_door"
    COMPOSITE_ORDERING = ("navigate_to_bathing_area", "open_shower_door")
    SUPPORTED_SCENE_IDS = (407, 418)
    SCENE_SPECS = _approach_operation_specs(
        SUPPORTED_SCENE_IDS,
        "navigate_to_bathing_area",
        "open_shower_door",
        "walk to the shower, then open the shower door",
    )


class IThorBathroomNavigateToBathingAreaAndCloseShowerDoor(
    _ApproachThenArticulationMixin,
    _BathroomShowerDoor,
    IThorBathroomEnv,
):
    BEHAVIOR = "close"
    SPEC_TASK = "navigate_to_bathing_area_and_close_shower_door"
    COMPOSITE_ORDERING = ("navigate_to_bathing_area", "close_shower_door")
    SUPPORTED_SCENE_IDS = (407, 418)
    SCENE_SPECS = _approach_operation_specs(
        SUPPORTED_SCENE_IDS,
        "navigate_to_bathing_area",
        "close_shower_door",
        "walk to the shower, then close the shower door",
    )


class IThorBathroomApproachSinkAndOpenCabinetDoor(
    _ApproachThenArticulationMixin,
    _BathroomStorageArticulation,
    IThorBathroomEnv,
):
    BEHAVIOR = "open"
    SPEC_TASK = "approach_sink_and_open_cabinet_door"
    TARGET_JOINT_TYPES = ("hinge",)
    COMPOSITE_ORDERING = ("approach_sink", "open_cabinet_door")
    SUPPORTED_SCENE_IDS = (405, 407, 418)
    SCENE_SPECS = _approach_operation_specs(
        SUPPORTED_SCENE_IDS,
        "approach_sink",
        "open_cabinet_door",
        "walk to the sink cabinet, then open its door halfway",
    )
    # Keep FloorPlan405's atomic target while validating the articulated
    # sweep from its calibrated collision-free approach-start pose.
    SCENE_SPECS["FloorPlan405"].update(
        get_ithor_bathroom_task_spec("FloorPlan405", SPEC_TASK)
    )


class IThorBathroomApproachSinkAndCloseCabinetDoor(
    _ApproachThenArticulationMixin,
    _BathroomStorageArticulation,
    IThorBathroomEnv,
):
    BEHAVIOR = "close"
    SPEC_TASK = "approach_sink_and_close_cabinet_door"
    TARGET_JOINT_TYPES = ("hinge",)
    COMPOSITE_ORDERING = ("approach_sink", "close_cabinet_door")
    SUPPORTED_SCENE_IDS = (405, 407, 418)
    SCENE_SPECS = _approach_operation_specs(
        SUPPORTED_SCENE_IDS,
        "approach_sink",
        "close_cabinet_door",
        "walk to the sink cabinet, then close its door",
    )
    # Keep the atomic targets while using composite-only collision-free
    # validation poses for the collected close configurations.
    for _scene_name in ("FloorPlan405", "FloorPlan407"):
        SCENE_SPECS[_scene_name].update(
            get_ithor_bathroom_task_spec(_scene_name, SPEC_TASK)
        )


class _ApproachThenLiftTask(
    _ApproachThenNativeFixtureMixin,
    _BathroomNativeLiftTask,
):
    ATOMIC_SKILL = "NavigateThenLiftObject"


class IThorBathroomNavigateToBathingAreaAndLiftSoapBottle(
    _ApproachThenLiftTask, IThorBathroomEnv
):
    SPEC_TASK = "navigate_to_bathing_area_and_lift_soap_bottle"
    SOURCE_LABEL = "soap bottle"
    COMPOSITE_ORDERING = ("navigate_to_bathing_area", "lift_soap_bottle")
    SUPPORTED_SCENE_IDS = (401,)
    SCENE_SPECS = _approach_operation_specs(
        SUPPORTED_SCENE_IDS,
        "navigate_to_bathing_area",
        "lift_soap_bottle",
        "walk to the bathtub, then pick up the soap bottle",
    )


class IThorBathroomApproachSinkAndLiftSoapBottle(
    _ApproachThenLiftTask, IThorBathroomEnv
):
    SPEC_TASK = "approach_sink_and_lift_soap_bottle"
    SOURCE_LABEL = "soap bottle"
    COMPOSITE_ORDERING = ("approach_sink", "lift_soap_bottle")
    SUPPORTED_SCENE_IDS = (405, 407, 418)
    SCENE_SPECS = _approach_operation_specs(
        SUPPORTED_SCENE_IDS,
        "approach_sink",
        "lift_soap_bottle",
        "walk to the sink, then pick up the soap bottle",
    )


class IThorBathroomApproachSinkAndLiftToiletPaperRoll(
    _ApproachThenLiftTask, IThorBathroomEnv
):
    SPEC_TASK = "approach_sink_and_lift_toilet_paper_roll"
    SOURCE_LABEL = "toilet-paper roll"
    COMPOSITE_ORDERING = ("approach_sink", "lift_toilet_paper_roll")
    SUPPORTED_SCENE_IDS = (405, 407, 418)
    SCENE_SPECS = _approach_operation_specs(
        SUPPORTED_SCENE_IDS,
        "approach_sink",
        "lift_toilet_paper_roll",
        "walk to the sink, then pick up the toilet-paper roll",
    )


class IThorBathroomApproachSinkAndLiftSprayBottle(
    _ApproachThenLiftTask, IThorBathroomEnv
):
    SPEC_TASK = "approach_sink_and_lift_spray_bottle"
    SOURCE_LABEL = "spray bottle"
    COMPOSITE_ORDERING = ("approach_sink", "lift_spray_bottle")
    SUPPORTED_SCENE_IDS = (405, 407, 418)
    SCENE_SPECS = _approach_operation_specs(
        SUPPORTED_SCENE_IDS,
        "approach_sink",
        "lift_spray_bottle",
        "walk to the sink, then pick up the spray bottle",
    )


class IThorBathroomNavigateToBathingAreaAndLiftScrubBrush(
    _ApproachThenNativeFixtureMixin, _BathroomScrubBrushLiftTask, IThorBathroomEnv
):
    ATOMIC_SKILL = "NavigateThenLiftObject"
    SPEC_TASK = "navigate_to_bathing_area_and_lift_scrub_brush"
    SOURCE_LABEL = "scrub brush"
    COMPOSITE_ORDERING = ("navigate_to_bathing_area", "lift_scrub_brush")
    SUPPORTED_SCENE_IDS = (405, 407, 415)
    SCENE_SPECS = _approach_operation_specs(
        SUPPORTED_SCENE_IDS,
        "navigate_to_bathing_area",
        "lift_scrub_brush",
        "walk to the bathing area, then pick up the scrub brush",
    )
    # Match the atomic 405 brush placement in this composite's private spec;
    # keep the already-collected atomic and all other tasks unchanged.
    SCENE_SPECS["FloorPlan405"].update(
        get_ithor_bathroom_task_spec("FloorPlan405", SPEC_TASK)
    )
    # Atomic episodes have already been collected. Apply the stable stool
    # layout only to this composite's independent 407 spec, never the atomic.
    SCENE_SPECS["FloorPlan407"].update(
        get_ithor_bathroom_task_spec("FloorPlan407", SPEC_TASK)
    )
    # The 415 rim brush is too far from its collision-free working stance.
    # Override only this composite; its atomic and the other scenes are frozen.
    SCENE_SPECS["FloorPlan415"].update(
        get_ithor_bathroom_task_spec("FloorPlan415", SPEC_TASK)
    )


class IThorBathroomApproachSinkAndLiftScrubBrush(
    _ApproachThenLiftTask, IThorBathroomEnv
):
    SPEC_TASK = "approach_sink_and_lift_scrub_brush"
    SOURCE_LABEL = "scrub brush"
    COMPOSITE_ORDERING = ("approach_sink", "lift_scrub_brush")
    SUPPORTED_SCENE_IDS = (418,)
    SCENE_SPECS = _approach_operation_specs(
        SUPPORTED_SCENE_IDS,
        "approach_sink",
        "lift_scrub_brush",
        "walk to the sink, then pick up the scrub brush",
    )

    # Restore the 9207dbc8 layout only for this composite; freeze the atomic spec.
    SCENE_SPECS["FloorPlan418"]["source_xy"] = [-2.49, -3.65]
    SCENE_SPECS["FloorPlan418"]["robot_pose"] = {
        "xy": [-1.80, -3.65],
        "yaw": 3.140581,
    }


class IThorBathroomNavigateToBathingAreaAndStandSoapBottleUpright(
    _ApproachThenNativeFixtureMixin,
    _BathroomNativeUprightTask,
    IThorBathroomEnv,
):
    ATOMIC_SKILL = "NavigateThenReorientObject"
    SPEC_TASK = "navigate_to_bathing_area_and_stand_soap_bottle_upright"
    SOURCE_LABEL = "soap bottle"
    COMPOSITE_ORDERING = (
        "navigate_to_bathing_area",
        "stand_soap_bottle_upright",
    )
    SUPPORTED_SCENE_IDS = (401,)
    SCENE_SPECS = _approach_operation_specs(
        SUPPORTED_SCENE_IDS,
        "navigate_to_bathing_area",
        "stand_soap_bottle_upright",
        "walk to the bathtub, then stand the soap bottle upright",
    )


class IThorBathroomNavigateToBathingAreaAndPutSpongeInBathtub(
    _CompositeSpongeStartPoseMixin,
    _ApproachThenSpawnedObjectMixin,
    _BathroomSpongeInBathtubTask,
    IThorBathroomEnv,
):
    ATOMIC_SKILL = "NavigateThenPickPlace"
    SPEC_TASK = "navigate_to_bathing_area_and_put_sponge_in_bathtub"
    COMPOSITE_ORDERING = (
        "navigate_to_bathing_area",
        "put_sponge_in_bathtub",
    )
    # FloorPlan418 has a shower enclosure, not a bathtub; it intentionally
    # remains unsupported rather than repurposing this bathtub-specific task.
    SUPPORTED_SCENE_IDS = (401, 405, 407, 415)
    SCENE_SPECS = _approach_operation_specs(
        SUPPORTED_SCENE_IDS,
        "navigate_to_bathing_area",
        "put_sponge_in_bathtub",
        "walk to the bathtub, then put the sponge in it",
    )
    # Keep frozen atomics unchanged while using collision-free rim stances for
    # these navigation composites only.
    SCENE_SPECS["FloorPlan405"]["composite_operation_robot_pose"] = {
        "xy": [-1.89, 0.83],
        "yaw": 0.0,
    }
    SCENE_SPECS["FloorPlan415"]["composite_operation_robot_pose"] = {
        "xy": [-1.42, -0.83],
        "yaw": 3.141593,
    }


class IThorBathroomApproachGarbageCanAndPutToiletPaperRollIntoBin(
    _ApproachThenNativeFixtureMixin,
    _BathroomToiletPaperRollIntoBinTask,
    IThorBathroomEnv,
):
    ATOMIC_SKILL = "NavigateThenPickPlace"
    SPEC_TASK = "approach_garbage_can_and_put_toilet_paper_roll_into_bin"
    COMPOSITE_ORDERING = (
        "approach_garbage_can",
        "put_toilet_paper_roll_into_bin",
    )
    SUPPORTED_SCENE_IDS = (401, 405, 407, 415, 418)
    SCENE_SPECS = _garbage_can_toilet_roll_specs()


__all__ = [
    "IThorBathroomApproachGarbageCanAndPutToiletPaperRollIntoBin",
    "IThorBathroomApproachSinkAndCloseCabinetDoor",
    "IThorBathroomApproachSinkAndLiftScrubBrush",
    "IThorBathroomApproachSinkAndLiftSoapBottle",
    "IThorBathroomApproachSinkAndLiftSprayBottle",
    "IThorBathroomApproachSinkAndLiftToiletPaperRoll",
    "IThorBathroomApproachSinkAndOpenCabinetDoor",
    "IThorBathroomApproachToiletAndCloseToiletLid",
    "IThorBathroomApproachToiletAndOpenToiletLid",
    "IThorBathroomNavigateToBathingAreaAndCloseShowerDoor",
    "IThorBathroomNavigateToBathingAreaAndLiftScrubBrush",
    "IThorBathroomNavigateToBathingAreaAndLiftSoapBottle",
    "IThorBathroomNavigateToBathingAreaAndOpenShowerDoor",
    "IThorBathroomNavigateToBathingAreaAndPutSpongeInBathtub",
    "IThorBathroomNavigateToBathingAreaAndStandSoapBottleUpright",
]
