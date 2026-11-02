"""Scene-local sink routine composed from the sponge and native bottle atoms."""

from copy import deepcopy
from dataclasses import replace

import numpy as np

from robocasa.environments.ithor_atomic.pick_place import SupportContactPlacementMixin
from robocasa.environments.ithor_atomic.task_assets import SPONGE_MODEL
from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.environments.ithor_composite.progress import CompositeStageReporterMixin
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_spec,
)

from ..lift import _BathroomNativeLiftTask
from ..object_manipulation import _NativeBathroomFixtureMixin
from ..sink_tasks import IThorBathroomPutSpongeInSink


class IThorBathroomApproachSinkAndPutSpongeInSinkThenLiftSoapBottle(
    CompositeStageReporterMixin,
    _NativeBathroomFixtureMixin,
    SupportContactPlacementMixin,
    SceneSpecMixin,
    IThorBathroomEnv,
):
    """Approach the sink, put the sponge on its bottom, then lift the bottle."""

    ATOMIC_SKILL = "ThreeSkillRoutine"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    SPEC_TASK = "approach_sink_put_sponge_lift_soap_bottle"
    SUPPORTED_SCENE_IDS = (401,)
    COMPOSITE_KIND = "ordered_sink_routine"
    COMPOSITE_ORDERING = ("approach_sink", "put_sponge_in_sink", "lift_soap_bottle")
    APPROACH_HOLD_STEPS = 5
    SUCCESS_DISTANCE = 0.35
    MIN_LIFT_DELTA = 0.10
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    SCENE_SPECS = {
        "FloorPlan401": deepcopy(
            IThorBathroomPutSpongeInSink.SCENE_SPECS["FloorPlan401"]
        )
    }
    # The two near corners provide support clear of the raised outer lip.
    SCENE_SPECS["FloorPlan401"]["source_region"]["center"] = [-2.79, 1.28, 1.025]
    SCENE_SPECS["FloorPlan401"].update(
        navigation=deepcopy(
            get_ithor_bathroom_task_spec("FloorPlan401", "approach_sink")
        ),
        bottle_fixture="FloorPlan401:Soap_Bottle_1_10",
        bottle_support_center=[-2.79, 1.82, 1.025],
        bottle_support_half_size=[0.08, 0.07, 0.03],
        # Point the native bottle's +Y axis upward on the sink rim.
        bottle_quat=[0.70710678, 0.70710678, 0.0, 0.0],
        lang=(
            "Walk to the sink, put the sponge in the sink, "
            "then pick up the soap bottle."
        ),
    )
    SOURCE_NAME = IThorBathroomPutSpongeInSink.SOURCE_NAME
    SOURCE_SCALE = IThorBathroomPutSpongeInSink.SOURCE_SCALE
    INITIAL_POSE_JITTER = False
    # These atomic helpers have no lifecycle / super() calls. Reuse them without
    # inheriting a public environment, which would create a catalog alias.
    _make_region = IThorBathroomPutSpongeInSink._make_region
    _get_obj_cfgs = IThorBathroomPutSpongeInSink._get_obj_cfgs
    _raw_success = IThorBathroomPutSpongeInSink._raw_success
    _success_region_bounds = IThorBathroomPutSpongeInSink._success_region_bounds
    _source_root_z = _BathroomNativeLiftTask._source_root_z

    def _get_scene_robot_pose_calibration(self):
        pose = self.scene_task_spec()["navigation"]["start_pose"]
        return np.array([*pose["xy"], 0.0]), float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.sink_fixture = self._exact_fixture(spec["sink_fixture"])
        if self.sink_fixture is None:
            raise LookupError(f"{self.scene_name}: missing exact sink")
        self.sink_bottom_geoms = tuple(spec["sink_bottom_geoms"])
        self.source_region = self._make_region("support", spec["source_region"])
        self.sink_region = self._make_region("receptacle", spec["sink_region"])
        self.fixture_refs["sink"] = self.sink_fixture
        self.region_refs["sink_rim"] = self.source_region
        self.region_refs["sink_receptacle"] = self.sink_region
        self.source_fixture = self._exact_fixture(spec["bottle_fixture"])
        if self.source_fixture is None:
            raise LookupError(f"{self.scene_name}: missing exact soap bottle")
        self.bottle_support_region = replace(
            self.source_region,
            stable_id=f"{self.sink_fixture.stable_id}:region:support:routine_bottle",
            _world_center=np.asarray(spec["bottle_support_center"], dtype=float),
            half_size=np.asarray(spec["bottle_support_half_size"], dtype=float),
        )
        self.fixture_refs["soap_bottle"] = self.source_fixture
        self.region_refs["bottle_support"] = self.bottle_support_region
        pose = spec["navigation"]["start_pose"]
        self.set_robot_base_pose([*pose["xy"], 0.0], pose["yaw"])

    def _setup_scene(self):
        super()._setup_scene()
        self._set_exact_robot_pose(
            self.scene_task_spec()["navigation"]["start_pose"], "sink routine"
        )

    def _reset_internal(self):
        super()._reset_internal()
        spec = self.scene_task_spec()
        if not self.place_object_on_support_contact(
            self.SOURCE_NAME,
            self.source_region,
            fixture=self.sink_fixture,
            inward_fractions=(0.0,),
            probe_above=0.01,
            probe_below=0.02,
            probe_step=0.0002,
            penetration_tolerance=0.0003,
        ):
            raise RuntimeError(f"{self.scene_name}: sponge has no sink-rim support")
        self._set_fixture_quaternion(self.source_fixture, spec["bottle_quat"])
        if not self._place_fixture_on_support(
            self.source_fixture,
            self.bottle_support_region,
            self.sink_fixture,
            spec["bottle_support_center"][:2],
            contract_name="routine_bottle_sink_rim",
            probe_above=0.01,
            probe_below=0.03,
            probe_step=0.0002,
        ):
            raise RuntimeError(f"{self.scene_name}: bottle has no sink-rim support")
        if self.count_body_collisions() or not self.robot_footprint_is_on_floor(
            self._robot_base_xy(), radius=0.25
        ):
            raise RuntimeError(f"{self.scene_name}: unsafe sink routine spawn")
        self._initial_sink_distance = self.robot_fixture_collision_distance(
            self.sink_fixture, max_distance=10.0
        )
        if (
            not np.isfinite(self._initial_sink_distance)
            or self._initial_sink_distance
            < spec["navigation"]["minimum_start_distance"]
        ):
            raise RuntimeError(
                f"{self.scene_name}: sink routine starts too near its goal"
            )
        self._reset_task_progress()

    def _reset_task_progress(self):
        # Fixed-scene collection calls this hook after restoring initial physics.
        self._initial_source_z = self._source_root_z()
        self._success_hold_steps = 0
        self._routine_stage = 0
        self._routine_hold = 0
        self._routine_armed = True
        self._routine_reported_success = False

    def _bottle_is_lifted(self):
        return self._source_root_z() >= self._initial_source_z + self.MIN_LIFT_DELTA

    def _stage_goal(self):
        if self._routine_stage == 0:
            distance = self.robot_fixture_collision_distance(
                self.sink_fixture, max_distance=10.0
            )
            return bool(np.isfinite(distance) and distance <= self.SUCCESS_DISTANCE)
        if self._routine_stage == 1:
            return self._raw_success()  # The unchanged sponge / sink-bottom predicate.
        return bool(self._bottle_is_lifted() and self._raw_success())

    def _advance_sink_routine(self):
        goal = self._stage_goal()
        target_reached = self._bottle_is_lifted() if self._routine_stage == 2 else goal
        if not target_reached:
            self._routine_armed = True
        self._routine_hold = (
            self._routine_hold + 1 if goal and self._routine_armed else 0
        )
        # Sponge placement retains its atomic one-second contact requirement.
        required = (
            self.APPROACH_HOLD_STEPS if self._routine_stage == 0 else self.HOLD_STEPS
        )
        if self._routine_hold < required:
            return
        if self._routine_stage < 2:
            self._routine_stage += 1
            self._routine_hold = 0
            # A preplaced sponge must leave the bottom and be placed again.
            # A prelifted bottle must drop below the lift threshold and rise again;
            # interrupting sponge contact alone cannot arm the final stage.
            self._routine_armed = not (
                self._bottle_is_lifted()
                if self._routine_stage == 2
                else self._stage_goal()
            )
            self._report_stage_complete(
                self._routine_stage, 3, self.COMPOSITE_ORDERING[self._routine_stage]
            )
        elif not self._routine_reported_success:
            self._routine_reported_success = True
            self._report_stage_complete(3, 3)

    def _post_action(self, action):
        self._advance_sink_routine()
        return super()._post_action(action)

    def _check_success(self):
        return bool(self._routine_stage == 2 and self._routine_hold >= self.HOLD_STEPS)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=self.scene_task_spec()["lang"],
            source_object=self.SOURCE_NAME,
            source_model=SPONGE_MODEL,
            source_scale=self.SOURCE_SCALE,
            source_support_region=self.source_region.stable_id,
            target_fixture=self.sink_fixture.stable_id,
            target_contact_geoms=list(self.sink_bottom_geoms),
            success_requires_release=False,
            stage_order=list(self.COMPOSITE_ORDERING),
            approach_hold_steps=self.APPROACH_HOLD_STEPS,
            sponge_hold_steps=self.HOLD_STEPS,
            final_hold_steps=self.HOLD_STEPS,
            navigation_target_fixture=self.sink_fixture.stable_id,
            navigation_success_distance=self.SUCCESS_DISTANCE,
            initial_sink_collision_distance=float(self._initial_sink_distance),
            bottle_fixture=self.source_fixture.stable_id,
            bottle_support_region=self.bottle_support_region.stable_id,
            initial_bottle_root_z=self._initial_source_z,
            minimum_lift_delta=self.MIN_LIFT_DELTA,
            final_grasp_required=False,
            success_definition=(
                "approach within 0.35 m of the sink for 5 control steps, then keep "
                "the sponge in contact with its bottom for 1 second, then lift "
                f"the soap bottle at least {self.MIN_LIFT_DELTA:.2f} m above its "
                "reset height; maintain "
                "this height and sponge-bottom contact for 1 second, without a "
                "grasp requirement"
            ),
        )
        return meta
