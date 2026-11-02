"""Ordered scene-410 routines with task-local copies of the drawer atoms."""

from copy import deepcopy
from dataclasses import replace

from robocasa.environments.ithor_atomic.navigation import ExactSceneNavigation
from robocasa.environments.ithor_atomic.pick_place import resolve_task_support_region
from robocasa.environments.ithor_composite.progress import CompositeStageReporterMixin
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv

from ..drawer_pnp import (
    IThorBathroomTakeSoapBottleFromDrawer,
    IThorBathroomToiletPaperRollIntoDrawer,
    _BathroomDrawerTask,
)
from ..reorientation import _BathroomNativeUprightTask


class _OrderedDrawerRoutineMixin(CompositeStageReporterMixin):
    """Credit three goals in order; precompleted goals must be undone first."""

    ATOMIC_SKILL = "OrderedRoutine"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    COMPOSITE_KIND = "drawer_routine"
    INTERMEDIATE_HOLD_STEPS = 5

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_task_progress()

    def _reset_task_progress(self):
        # Also called after fixed-scene collection restores the initial physics.
        self._success_hold = 0
        self._routine_stage = 0
        self._routine_hold = 0
        self._routine_armed = not self._stage_goal(0)
        self._routine_reported_success = False

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._advance_drawer_routine()
        return ret

    def _advance_drawer_routine(self):
        goal = self._stage_goal(self._routine_stage)
        if not goal:
            self._routine_armed = True
        self._routine_hold = (
            self._routine_hold + 1 if goal and self._routine_armed else 0
        )
        required = (
            self.HOLD_STEPS
            if self._routine_stage == 2
            else self.INTERMEDIATE_HOLD_STEPS
        )
        if self._routine_hold < required:
            return
        if self._routine_stage < 2:
            self._routine_stage += 1
            self._routine_hold = 0
            self._routine_armed = not self._stage_goal(self._routine_stage)
            self._report_stage_complete(
                self._routine_stage,
                3,
                next_stage=self.COMPOSITE_ORDERING[self._routine_stage],
            )
        elif not self._routine_reported_success:
            self._routine_reported_success = True
            self._report_stage_complete(3, 3)

    def _check_success(self):
        return bool(
            self._routine_stage == 2
            and self._routine_hold >= self.HOLD_STEPS
            and self._stage_goal(2)
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            stage_order=list(self.COMPOSITE_ORDERING),
            intermediate_hold_steps=self.INTERMEDIATE_HOLD_STEPS,
            final_hold_steps=self.HOLD_STEPS,
        )
        return meta


class IThorBathroomOpenDrawerThenPutToiletPaperRollAndCloseDrawer(
    _OrderedDrawerRoutineMixin, _BathroomDrawerTask, IThorBathroomEnv
):
    """Open the sink drawer, put the roll on its bottom, then close it."""

    SUPPORTED_SCENE_IDS = (410,)
    COMPOSITE_ORDERING = (
        "open_drawer",
        "put_toilet_paper_roll_in_drawer",
        "close_drawer",
    )
    _source_contacts_drawer_bottom = (
        IThorBathroomToiletPaperRollIntoDrawer._source_contacts_drawer_bottom
    )
    SUCCESS_OPEN_FRACTION = 0.8
    SUCCESS_CLOSE_FRACTION = 0.2
    SCENE_SPECS = {
        "FloorPlan410": deepcopy(
            IThorBathroomToiletPaperRollIntoDrawer.SCENE_SPECS["FloorPlan410"]
        )
    }
    SCENE_SPECS["FloorPlan410"]["lang"] = (
        "Open the drawer under the sink, put the toilet paper roll inside, "
        "then close the drawer."
    )
    SUCCESS_DESCRIPTION = (
        "open the drawer to at least 80%, rest the roll on its bottom, then "
        "close the same drawer to at most 20% with bottom contact maintained"
    )

    def _setup_scene(self):
        super()._setup_scene()
        # Only this routine starts 20% open; atomic layouts and robot pose stay intact.
        initial_qpos = self.drawer_joint.closed_qpos + 0.25 * (
            self.drawer_joint.open_qpos - self.drawer_joint.closed_qpos
        )
        self.sim.data.set_joint_qpos(self.drawer_joint.name, initial_qpos)
        self.sim.data.set_joint_qvel(self.drawer_joint.name, 0.0)
        self.sim.forward()

    def _stage_goal(self, index):
        fraction = self.drawer_joint.fraction(self.sim)
        if index == 0:
            return fraction >= self.SUCCESS_OPEN_FRACTION
        if index == 1:
            return bool(
                fraction >= self.SUCCESS_OPEN_FRACTION
                and IThorBathroomToiletPaperRollIntoDrawer._raw_success(self)
            )
        return self._raw_success()

    def _raw_success(self):
        return bool(
            self.drawer_joint.fraction(self.sim) <= self.SUCCESS_CLOSE_FRACTION
            and IThorBathroomToiletPaperRollIntoDrawer._raw_success(self)
        )


class IThorBathroomTakeSoapBottleFromDrawerThenStandOnSinkAndApproachToilet(
    _OrderedDrawerRoutineMixin, _BathroomDrawerTask, IThorBathroomEnv
):
    """Lift the drawer bottle, release it upright on the counter, approach toilet."""

    SUPPORTED_SCENE_IDS = (410,)
    COMPOSITE_ORDERING = (
        "take_soap_bottle_from_drawer",
        "stand_soap_bottle_on_sink_counter",
        "approach_toilet",
    )
    SCENE_SPECS = {
        "FloorPlan410": deepcopy(
            IThorBathroomTakeSoapBottleFromDrawer.SCENE_SPECS["FloorPlan410"]
        )
    }
    SCENE_SPECS["FloorPlan410"].update(
        lang=(
            "Take the soap bottle out of the drawer, stand it upright on the "
            "vanity top and release it, then walk to the toilet."
        ),
        toilet_fixture="FloorPlan410:Toilet_1_4",
        # Keep region 0 as the primary reference; placement accepts all four patches.
        # Drawer staging uses drawer_support_region, so its source pose is unchanged.
        surface_region="FloorPlan410:CounterTopMesh_fb983d72_22:region:receptacle:0",
    )
    SUCCESS_DESCRIPTION = (
        "lift the drawer bottle at least 0.30 m, release it upright on the sink "
        "counter, then approach within 0.35 m of the toilet while it stays upright"
    )
    SOURCE_IN_DRAWER = True
    COUNTER_SUPPORT_Z_OFFSET = -0.012
    COUNTER_REGION_IDS = tuple(
        f"FloorPlan410:CounterTopMesh_fb983d72_22:region:receptacle:{index}"
        for index in range(4)
    )
    MIN_LIFT_DELTA = IThorBathroomTakeSoapBottleFromDrawer.MIN_LIFT_DELTA
    _source_root_z = IThorBathroomTakeSoapBottleFromDrawer._source_root_z
    SUPPORT_PLANE_TOLERANCE = _BathroomNativeUprightTask.SUPPORT_PLANE_TOLERANCE
    _fixture_bottom_is_on_support_plane = (
        _BathroomNativeUprightTask._fixture_bottom_is_on_support_plane
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        # The actual upward counter contacts are at z=0.806 m, 12 mm below
        # the exported shallow region's lower face. Calibrate only our copy.
        raw_regions = {
            region.stable_id: region
            for region in self.work_surface_fixture.regions.values()
        }
        counter_regions = []
        for stable_id in self.COUNTER_REGION_IDS:
            region = resolve_task_support_region(raw_regions[stable_id])
            offset = region._world_offset.copy()
            offset[2] += self.COUNTER_SUPPORT_Z_OFFSET
            counter_regions.append(replace(region, _world_offset=offset))
        self.counter_support_regions = tuple(counter_regions)
        self.work_surface_region = self.counter_support_regions[0]
        self.region_refs["task_surface"] = self.work_surface_region
        self.target_fixture = self._exact_fixture(
            self.scene_task_spec()["toilet_fixture"]
        )
        self.fixture_refs["target_toilet"] = self.target_fixture
        self._navigation_success_distance = ExactSceneNavigation.SUCCESS_DISTANCE
        # Native Soap_Bottle_10's base-to-pump direction is its local +Y axis.
        self._target_upright_axis = (1, 1)

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.target_fixture is not None)

    def _reset_task_progress(self):
        self._initial_source_z = self._source_root_z()
        super()._reset_task_progress()
        self._initial_toilet_distance = self.robot_fixture_collision_distance(
            self.target_fixture, max_distance=10.0
        )

    def _bottle_is_on_counter(self):
        # The narrow rim can physically support the bottle despite bbox overhang.
        # Test the center against the patch union, preserving gaps around the sink.
        center = self.source_fixture.world_center(self.sim)
        support_bodies = self._fixture_body_ids(
            self.work_surface_fixture, include_siblings=True
        )
        contact_distances = self._fixture_contact_distances(
            self.source_fixture, support_bodies
        )
        return bool(
            self._fixture_is_upright(self.source_fixture, self._target_upright_axis)
            and any(
                region.contains_xy(center, self.sim)
                for region in self.counter_support_regions
            )
            and self._fixture_bottom_is_on_support_plane()
            and contact_distances
            and min(contact_distances) >= -self.CONTACT_PENETRATION_TOLERANCE
            and not self._fixture_has_illegal_penetration(
                self.source_fixture, self.work_surface_fixture.body_names
            )
            and self._fixture_released(self.source_fixture)
        )

    def _stage_goal(self, index):
        if index == 0:
            return IThorBathroomTakeSoapBottleFromDrawer._raw_success(self)
        if index == 1:
            return self._bottle_is_on_counter()
        return self._raw_success()

    def _raw_success(self):
        return bool(
            self._bottle_is_on_counter()
            and ExactSceneNavigation._robot_is_near_target(self)
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            navigation_target_fixture=self.target_fixture.stable_id,
            initial_toilet_collision_distance=float(self._initial_toilet_distance),
            navigation_success_distance=self._navigation_success_distance,
            placement_surface=self.work_surface_fixture.stable_id,
            placement_region=self.work_surface_region.stable_id,
            placement_regions=[
                region.stable_id for region in self.counter_support_regions
            ],
        )
        return meta


TASK_CLASSES = (
    IThorBathroomOpenDrawerThenPutToiletPaperRollAndCloseDrawer,
    IThorBathroomTakeSoapBottleFromDrawerThenStandOnSinkAndApproachToilet,
)

__all__ = [cls.__name__ for cls in TASK_CLASSES]
