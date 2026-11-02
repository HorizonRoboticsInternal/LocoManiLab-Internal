"""Ordered Scene 315 routines with task-local desk layouts and progress."""

from copy import deepcopy

import numpy as np

from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.environments.ithor_composite.progress import CompositeStageReporterMixin
from robocasa.models.scenes.ithor_bedroom_task_specs import get_ithor_bedroom_task_spec

from .assets import (
    BASKET_MODEL,
    BASKET_SCALE,
    CAN_MODEL,
    CAN_SCALE,
    LIFT_BASKET_MODEL,
    LIFT_BASKET_SCALE,
    MUG_MODEL,
    MUG_SCALE,
    asset_xml_path,
)
from .composite_base import BedroomCompositeObjectMixin
from .trash_can import BedroomTrashCanLayoutMixin, IThorBedroomCanIntoTrashCan


def _spec(lang, stages):
    spec = get_ithor_bedroom_task_spec("FloorPlan315", "can_into_trash_can")
    navigation = get_ithor_bedroom_task_spec("FloorPlan315", "approach_trash_can")
    spec.update(
        lang=lang,
        start_pose=deepcopy(navigation["start_pose"]),
        minimum_start_distance=navigation["minimum_start_distance"],
        composite_ordering=list(stages),
        approach_direction=[0.0, 1.0],
        row_center_x=-1.30,
    )
    return {"FloorPlan315": spec}


class _Scene315RoutineMixin(
    CompositeStageReporterMixin,
    BedroomTrashCanLayoutMixin,
    BedroomCompositeObjectMixin,
    SceneSpecMixin,
):
    SUPPORTED_SCENE_IDS = (315,)
    TASK_FAMILY = "composite"
    ATOMIC_SKILL = "ThreeSkillRoutine"
    DIFFICULTY_LEVEL = "L3"
    COMPOSITE_KIND = "ordered_bedroom_routine"
    APPROACH_HOLD_STEPS = 5
    SUCCESS_DISTANCE = 0.35
    INITIAL_POSE_JITTER = False
    OBJECT_GAP = 0.04
    MIN_LIFT_DELTA = 0.30

    def _get_scene_robot_pose_calibration(self):
        pose = self.scene_task_spec()["start_pose"]
        return np.array([*pose["xy"], 0.0]), float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.work_surface_fixture = self.require_fixture(spec["source_fixture"])
        self.work_surface_region = self.work_surface_fixture.regions[
            spec["source_region"]
        ]
        self.bin_bottom_geoms = tuple(spec["bin_bottom_geoms"])
        self.fixture_refs["work_surface"] = self.work_surface_fixture
        self.region_refs["work_surface"] = self.work_surface_region

    def _get_obj_cfgs(self):
        return [
            dict(
                name=name,
                obj_groups=name,
                object_scale=scale,
                info={"mjcf_path": asset_xml_path(model)},
            )
            for name, (model, scale) in self.OBJECT_ASSETS.items()
        ]

    def _setup_scene(self):
        super()._setup_scene()
        position, yaw = self._get_scene_robot_pose_calibration()
        if not (
            self.robot_footprint_is_on_floor(position[:2], radius=0.25)
            and self._set_robot_free_pose(position[:2], yaw)
            and self._calibrate_robot_free_joint_height()
            and self.count_body_collisions() == 0
        ):
            raise RuntimeError(f"{self.scene_name}: invalid routine navigation start")
        self.robot_base_pos = position
        self.robot_base_ori = np.array([0.0, 0.0, yaw])

    def _apply_object_placements(self, placements):
        super()._apply_object_placements(placements)
        region = self.work_surface_region
        center = region.world_pose(self.sim)[0]
        outward = np.asarray(self.scene_task_spec()["approach_direction"], dtype=float)
        lateral = np.array([-outward[1], outward[0]])
        names = tuple(self.OBJECT_ASSETS)
        # Stage every object away from the desk before probing real support.
        for i, name in enumerate(names):
            self._set_object_bottom_at(name, center[:2], 2.5 + 0.4 * i)
            self.sim.forward()
            self._orient_object_long_axis_crosswise(name, region)
        widths = [2 * self._object_projected_half_extent(n, lateral) for n in names]
        depths = [self._object_projected_half_extent(n, outward) for n in names]
        total = sum(widths) + self.OBJECT_GAP * (len(names) - 1)
        if total + 2 * self.OBJECT_EDGE_MARGIN > 2 * self.support_region_radial_extent(
            region, lateral, sim=self.sim
        ):
            raise RuntimeError(f"{self.scene_name}: routine objects exceed desk width")
        depth = self.support_region_radial_extent(region, outward, sim=self.sim)
        row_depth = depth - max(depths) - self.OBJECT_EDGE_MARGIN
        row_center = center[:2].copy()
        row_center[0] = self.scene_task_spec()["row_center_x"]
        cursor = -total / 2
        for name, width in zip(names, widths):
            xy = row_center + outward * row_depth + lateral * (cursor + width / 2)
            if not self.place_object_on_support_contact(
                name,
                region,
                fixture=self.work_surface_fixture,
                xy=xy,
                inward_fractions=(0.0,),
                probe_step=0.001,
            ):
                raise RuntimeError(
                    f"{self.scene_name}: no contacted desk pose for {name}"
                )
            cursor += width + self.OBJECT_GAP

    def _surface_layout_frame(self, region, operator_xy=None):
        # Orient using the desk's operation side, independently of the far spawn.
        center = region.world_pose(self.sim)[0]
        direction = np.asarray(
            self.scene_task_spec()["approach_direction"], dtype=float
        )
        return center, direction, np.array([-direction[1], direction[0]])

    def _reset_internal(self):
        super()._reset_internal()
        self._initial_positions = {
            name: self.object_pos(name).copy() for name in self.OBJECT_ASSETS
        }
        self._lift_initial_z = float(self.object_pos("basket")[2])
        self._reset_task_progress()
        target = (
            self.bin_fixture
            if self.COMPOSITE_ORDERING[0] == "approach_trash_can"
            else self.work_surface_fixture
        )
        self._start_distance = self.robot_fixture_collision_distance(
            target, max_distance=10.0
        )
        if self._start_distance < self.scene_task_spec()["minimum_start_distance"]:
            raise RuntimeError(f"{self.scene_name}: routine starts too near its target")
        if self.count_body_collisions() or not self.robot_footprint_is_on_floor(
            self._robot_base_xy(), radius=0.25
        ):
            raise RuntimeError(f"{self.scene_name}: unsafe routine reset")

    def _reset_task_progress(self):
        self._sequence_stage = 0
        self._sequence_hold = 0
        self._final_stage_reported = False

    def _near(self, fixture):
        distance = self.robot_fixture_collision_distance(fixture, max_distance=4.0)
        return distance <= self.SUCCESS_DISTANCE

    def _unchanged(self, names):
        return all(
            np.linalg.norm(self.object_pos(name) - self._initial_positions[name])
            <= 0.03
            and self.object_released(name, th=0.18)
            for name in names
        )

    def _advance_sequence(self, valid):
        self._sequence_hold = self._sequence_hold + 1 if valid else 0
        required = (
            self.HOLD_STEPS if self._sequence_stage == 2 else self.APPROACH_HOLD_STEPS
        )
        if self._sequence_hold < required:
            return
        if self._sequence_stage < 2:
            self._sequence_stage += 1
            self._sequence_hold = 0
            self._report_stage_complete(
                self._sequence_stage,
                3,
                next_stage=self.COMPOSITE_ORDERING[self._sequence_stage],
            )
        elif not self._final_stage_reported:
            self._final_stage_reported = True
            self._report_stage_complete(3, 3)

    def _post_action(self, action):
        self._advance_sequence(self._stage_valid())
        return super()._post_action(action)

    def _check_success(self):
        return self._sequence_stage == 2 and self._sequence_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        navigation_target = (
            self.bin_fixture
            if self.COMPOSITE_ORDERING[0] == "approach_trash_can"
            else self.work_surface_fixture
        )
        meta.update(
            target_fixture_id=navigation_target.stable_id,
            navigation_target_fixture_id=navigation_target.stable_id,
            lang=self.scene_task_spec()["lang"],
            composite_stages=list(self.COMPOSITE_ORDERING),
            composite_ordering=list(self.COMPOSITE_ORDERING),
            completed_stages=self._sequence_stage + int(self._check_success()),
            source_surface=self.work_surface_region.stable_id,
            start_distance=float(self._start_distance),
            approach_distance=self.SUCCESS_DISTANCE,
            object_assets={
                name: {"model": model, "scale": scale}
                for name, (model, scale) in self.OBJECT_ASSETS.items()
            },
            success_definition=self.SUCCESS_DEFINITION,
            requires_grasp=False,
        )
        return meta


class IThorBedroomApproachTrashCanAndPutCanInTrashThenLiftBasket(
    _Scene315RoutineMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "approach_trash_can_and_put_can_in_trash_then_lift_basket"
    COMPOSITE_ORDERING = ("approach_trash_can", "put_can_in_trash", "lift_basket")
    SCENE_SPECS = _spec(
        "walk to the trash can, put the can into it, then lift the basket",
        COMPOSITE_ORDERING,
    )
    OBJECT_ASSETS = {
        "basket": (LIFT_BASKET_MODEL, LIFT_BASKET_SCALE),
        "can": (CAN_MODEL, CAN_SCALE),
    }
    SOURCE_NAME = "can"
    _can_contacts_trash_can_bottom = (
        IThorBedroomCanIntoTrashCan._can_contacts_trash_can_bottom
    )
    SUCCESS_DEFINITION = (
        "approach the native trash can within 0.35 m for 5 steps with both objects "
        "unchanged; keep the can touching its bottom for 5 steps while the basket "
        "waits on the desk; then lift the basket "
        "at least 0.30 m for 1.0 s while the can stays against the bin bottom, "
        "without requiring hand grasp detection"
    )

    def _stage_valid(self):
        if self._sequence_stage == 0:
            return self._near(self.bin_fixture) and self._unchanged(("can", "basket"))
        if self._sequence_stage == 1:
            return self._can_contacts_trash_can_bottom() and self._unchanged(
                ("basket",)
            )
        height = float(self.object_pos("basket")[2])
        return bool(
            self._can_contacts_trash_can_bottom()
            and height >= self._lift_initial_z + self.MIN_LIFT_DELTA
        )


class IThorBedroomApproachDeskAndPutMugThenCanInBasket(
    _Scene315RoutineMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "approach_desk_and_put_mug_then_can_in_basket"
    COMPOSITE_ORDERING = ("approach_desk", "put_mug_in_basket", "put_can_in_basket")
    SCENE_SPECS = _spec(
        "walk to the desk, put the mug in the basket, then put the can in the basket",
        COMPOSITE_ORDERING,
    )
    OBJECT_ASSETS = {
        "basket": (BASKET_MODEL, BASKET_SCALE),
        "mug": (MUG_MODEL, MUG_SCALE),
        "can": (CAN_MODEL, CAN_SCALE),
    }
    SUCCESS_DEFINITION = (
        "approach the desk within 0.35 m for 5 steps with all objects unchanged; "
        "release the mug inside and contacting the supported basket for 5 steps "
        "while the can waits on the desk; then release the can inside and contacting "
        "the basket with the mug still inside, keeping both placed for 1.0 s"
    )

    def _placed(self, name):
        return bool(
            self._receptacle_upright_released_on_region(
                "basket", self.work_surface_region
            )
            and self._object_in_named_receptacle(
                name,
                "basket",
                containment="center",
                bottom_tolerance=0.035 if name == "mug" else 0.005,
            )
            and self.object_released(name, th=0.18)
        )

    def _stage_valid(self):
        if self._sequence_stage == 0:
            return self._near(self.work_surface_fixture) and self._unchanged(
                tuple(self.OBJECT_ASSETS)
            )
        if self._sequence_stage == 1:
            return self._placed("mug") and self._unchanged(("can",))
        return self._placed("mug") and self._placed("can")


__all__ = [
    "IThorBedroomApproachTrashCanAndPutCanInTrashThenLiftBasket",
    "IThorBedroomApproachDeskAndPutMugThenCanInBasket",
]
