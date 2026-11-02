"""Ergonomic navigation-then-manipulation tasks for iTHOR bedrooms."""

from __future__ import annotations

import numpy as np

from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)

from robocasa.environments.ithor_atomic.lift import RoomLiftMixin
from robocasa.environments.ithor_atomic.pick_place import (
    SupportContactPlacementMixin,
    resolve_task_support_region,
)
from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_spec,
)

from .assets import (
    BASKET_MODEL,
    BASKET_SCALE,
    BOXED_FOOD_MODEL,
    BOXED_FOOD_SCALE,
    CAN_MODEL,
    CAN_SCALE,
    LIFT_BASKET_MODEL,
    LIFT_BASKET_SCALE,
    MUG_MODEL,
    MUG_SCALE,
    asset_xml_path,
)
from .object_in_receptacle import (
    BedroomPnPMixin,
    BedroomRightHandSourceLayoutMixin,
    BedroomSimpleReceptaclePlacementMixin,
)
from .surface_placement import (
    align_object_long_axis_parallel_to_surface_edge,
)


_WALK_SPEC_FIELDS = (
    "fixture",
    "region",
    "far_start_pose",
    "minimum_start_distance",
    "approach_distance",
    "remove_fixtures",
    "lang",
)

_BEDROOM_SCENE_IDS = (302, 306, 310, 323, 329)


def _use_atomic_desk_start_for_323(scene_name, spec):
    """Reuse the navigation atomic start without changing other desk scenes."""
    if scene_name == "FloorPlan323":
        spec["far_start_pose"] = get_ithor_bedroom_task_spec(
            scene_name, "approach_desk", required=("start_pose",)
        )["start_pose"]


def _desk_walk_pnp_specs(lang):
    """Copy atomic desk-walk calibrations for these three PnP tasks only."""
    specs = {}
    for scene_id in _BEDROOM_SCENE_IDS:
        scene_name = f"FloorPlan{scene_id}"
        spec = get_ithor_bedroom_task_spec(
            scene_name,
            "secondary_workspace",
            required=tuple(field for field in _WALK_SPEC_FIELDS if field != "lang"),
        )
        workspace = get_ithor_bedroom_task_spec(scene_name, "grasp_workspace")
        if (spec["fixture"], spec["region"]) != (
            workspace["fixture"],
            workspace["region"],
        ):
            raise ValueError(f"{scene_name}: PnP and desk-walk surfaces differ")
        _use_atomic_desk_start_for_323(scene_name, spec)
        spec["lang"] = lang
        specs[scene_name] = spec
    return specs


def _all_desk_walk_lift_specs(lang):
    specs = {}
    for scene_id in _BEDROOM_SCENE_IDS:
        scene_name = f"FloorPlan{scene_id}"
        spec = get_ithor_bedroom_task_spec(
            scene_name,
            "secondary_workspace",
            required=tuple(field for field in _WALK_SPEC_FIELDS if field != "lang"),
        )
        lift_spec = get_ithor_bedroom_task_spec(
            scene_name,
            "lift_basket",
            required=("fixture", "region"),
        )
        if (spec["fixture"], spec["region"]) != (
            lift_spec["fixture"],
            lift_spec["region"],
        ):
            raise ValueError(
                f"{scene_name}: composite and atomic basket lift surfaces differ"
            )
        _use_atomic_desk_start_for_323(scene_name, spec)
        spec["lang"] = lang
        specs[scene_name] = spec
    return specs


def _surface_maps(specs):
    return (
        {scene_name: spec["fixture"] for scene_name, spec in specs.items()},
        {scene_name: spec["region"] for scene_name, spec in specs.items()},
    )


class _BedroomWalkGateMixin(CompositeStageReporterMixin):
    """Require an exact distant approach before a local table operation."""

    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L2"
    COMPOSITE_KIND = "navigate_then_manipulate"
    APPROACH_HOLD_STEPS = 5
    HOLD_STEPS = 5
    INITIAL_POSE_TOLERANCE = 0.03
    MIN_WORK_HEIGHT = 0.50
    MAX_WORK_HEIGHT = 0.90
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0

    def _set_far_start_pose(self):
        spec = self.scene_task_spec()
        pose = spec["far_start_pose"]
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        if xy.shape != (2,):
            raise ValueError("far_start_pose.xy must contain exactly two values")
        if hasattr(self, "_robot_free_joint_z"):
            delattr(self, "_robot_free_joint_z")
        if not (
            self.robot_footprint_is_on_floor(xy, radius=0.25)
            and self._set_robot_free_pose(xy, yaw)
            and self._calibrate_robot_free_joint_height()
            and self.count_body_collisions() == 0
        ):
            return False
        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        start_distance = self.robot_fixture_collision_distance(
            self.work_surface_fixture,
            max_distance=4.0,
        )
        minimum = float(spec["minimum_start_distance"])
        if not np.isfinite(start_distance) or start_distance < minimum:
            return False
        center, _ = self.work_surface_region.world_pose(self.sim)
        if not self.MIN_WORK_HEIGHT <= float(center[2]) <= self.MAX_WORK_HEIGHT:
            raise RuntimeError(
                f"{self.scene_name}: composite work surface height "
                f"{float(center[2]):.3f} m leaves the ergonomic range"
            )
        self._walk_start_distance = float(start_distance)
        self._walk_approach_distance = float(spec["approach_distance"])
        return True

    def _place_robot_at_surface(self):
        return self._set_far_start_pose()

    def _robot_near_work_surface(self):
        distance = self.robot_fixture_collision_distance(
            self.work_surface_fixture,
            max_distance=1.5,
        )
        return bool(np.isfinite(distance) and distance <= self._walk_approach_distance)

    def _walk_initial_names(self):
        return (self.SOURCE_NAME, self.RECEPTACLE_NAME)

    def _record_walk_initial_state(self):
        self._walk_initial_positions = {
            name: np.asarray(
                self.sim.data.get_body_xpos(self.objects[name].root_body),
                dtype=float,
            ).copy()
            for name in self._walk_initial_names()
        }
        source_position = self._walk_initial_positions[self.SOURCE_NAME]
        self._navigation_start_source_horizontal_distance = float(
            np.linalg.norm(source_position[:2] - self._robot_base_xy())
        )

    def _walk_initial_state_intact(self):
        if self._raw_operation_success():
            return False
        for name, initial in self._walk_initial_positions.items():
            current = np.asarray(
                self.sim.data.get_body_xpos(self.objects[name].root_body),
                dtype=float,
            )
            if float(np.linalg.norm(current - initial)) > self.INITIAL_POSE_TOLERANCE:
                return False
            if not self.object_released(name, th=self.RELEASE_DISTANCE):
                return False
        return True

    def _reset_walk_progress(self):
        self._walk_stage = 0
        self._walk_approach_hold = 0
        self._walk_operation_hold = 0

    def _reset_internal(self):
        super()._reset_internal()
        self._record_walk_initial_state()
        self._reset_walk_progress()
        if self._raw_operation_success():
            raise RuntimeError(f"{self.scene_name}: composite starts successful")

    def _advance_walk_progress(self):
        near = self._robot_near_work_surface()
        if self._walk_stage == 0:
            ready = near and self._walk_initial_state_intact()
            self._walk_approach_hold = self._walk_approach_hold + 1 if ready else 0
            if self._walk_approach_hold >= self.APPROACH_HOLD_STEPS:
                self._walk_stage = 1
                self._report_stage_complete(
                    1,
                    2,
                    next_stage="complete_manipulation",
                )
            self._walk_operation_hold = 0
            return
        valid = near and self._raw_operation_success()
        self._walk_operation_hold = self._walk_operation_hold + 1 if valid else 0

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._advance_walk_progress()
        return ret

    def _check_success(self):
        return bool(
            self._walk_stage == 1 and self._walk_operation_hold >= self.HOLD_STEPS
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["start_distance"] = float(self._walk_start_distance)
        ep_meta["navigation_start_source_horizontal_distance"] = float(
            self._navigation_start_source_horizontal_distance
        )
        ep_meta["approach_distance"] = float(self._walk_approach_distance)
        ep_meta["stage_order"] = [
            "approach the named work surface while the objects stay at reset",
            "complete the local manipulation while remaining at that surface",
        ]
        return ep_meta


class _BedroomWalkPnPTask(
    _BedroomWalkGateMixin,
    BedroomSimpleReceptaclePlacementMixin,
    BedroomRightHandSourceLayoutMixin,
    BedroomPnPMixin,
):
    """Private implementation for navigation followed by one placement."""

    ATOMIC_SKILL = "NavigateThenPickPlace"
    COMPOSITE_ORDERING = ("approach_work_surface", "place_in_receptacle")
    RECEPTACLE_XY_CONTAINMENT = "center"
    STAND_DISTANCES = (0.40, 0.45, 0.48, 0.58, 0.68, 0.80)

    def _raw_operation_success(self):
        return bool(
            BedroomSimpleReceptaclePlacementMixin._raw_place_success(self)
            and self._receptacle_is_upright_on_support()
        )


class _BedroomWalkLiftBasketTask(
    CompositeStageReporterMixin,
    SupportContactPlacementMixin,
    SceneSpecMixin,
    RoomLiftMixin,
):
    """Private exact-support navigation followed by a true basket lift."""

    ATOMIC_SKILL = "NavigateThenLiftObject"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "navigate_then_lift"
    COMPOSITE_ORDERING = ("approach_work_surface", "lift_basket")
    TARGET_NAME = "basket"
    TARGET_XML = asset_xml_path(LIFT_BASKET_MODEL)
    TARGET_SCALE = LIFT_BASKET_SCALE
    MIN_LIFT_DELTA = 0.30
    GRASP_HEIGHT_ALLOWANCE = 0.10
    APPROACH_HOLD_STEPS = 5
    HOLD_STEPS = 5
    INITIAL_POSE_TOLERANCE = 0.03
    PLACEMENT_MARGIN = 0.01
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0

    def _get_obj_cfgs(self):
        return [
            {
                "name": self.TARGET_NAME,
                "obj_groups": self.TARGET_XML,
                "object_scale": self.TARGET_SCALE,
                "placement": {
                    "region": getattr(self, "lift_region", None),
                    "margin": self.PLACEMENT_MARGIN,
                    "rotation": 0.0,
                    "strategy": "edge",
                },
            }
        ]

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.lift_fixture = self.get_fixture(id=spec["fixture"])
        raw_region = None
        if self.lift_fixture is not None:
            raw_region = next(
                (
                    region
                    for region in self.get_region(
                        fixture=self.lift_fixture,
                        return_all=True,
                    )
                    if region.stable_id == spec["region"]
                ),
                None,
            )
        self.lift_region = (
            resolve_task_support_region(raw_region) if raw_region is not None else None
        )
        if self.lift_fixture is not None:
            self.fixture_refs["lift_surface"] = self.lift_fixture
        if self.lift_region is not None:
            self.region_refs["lift_surface"] = self.lift_region
        removal_ids = tuple(
            dict.fromkeys((*spec.get("seating", ()), *spec["remove_fixtures"]))
        )
        removals = [self.get_fixture(id=stable_id) for stable_id in removal_ids]
        missing = [
            stable_id
            for stable_id, fixture in zip(removal_ids, removals)
            if fixture is None
        ]
        if missing:
            raise ValueError(
                f"{self.scene_name}: unknown exact lift removals {missing}"
            )
        self.removed_walk_lift_fixture_ids = list(removal_ids)
        if removals:
            self.remove_semantic_fixtures(removals)

    def _place_robot_at_target(self):
        spec = self.scene_task_spec()
        pose = spec["far_start_pose"]
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        if xy.shape != (2,):
            raise ValueError("far_start_pose.xy must contain exactly two values")
        if hasattr(self, "_robot_free_joint_z"):
            delattr(self, "_robot_free_joint_z")
        if not (
            self.robot_footprint_is_on_floor(xy, radius=0.25)
            and self._set_robot_free_pose(xy, yaw)
            and self._calibrate_robot_free_joint_height()
            and self.count_body_collisions() == 0
        ):
            raise RuntimeError(f"{self.scene_name}: invalid distant basket start")
        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        self._walk_start_distance = self.robot_fixture_collision_distance(
            self.lift_fixture,
            max_distance=4.0,
        )
        if not np.isfinite(
            self._walk_start_distance
        ) or self._walk_start_distance < float(spec["minimum_start_distance"]):
            raise RuntimeError(f"{self.scene_name}: basket start is too near")
        self._walk_approach_distance = float(spec["approach_distance"])
        height = float(self.lift_region.world_pose(self.sim)[0][2])
        if not 0.50 <= height <= 0.90:
            raise RuntimeError(
                f"{self.scene_name}: lift surface height {height:.3f} m is invalid"
            )

    def _reset_internal(self):
        super()._reset_internal()
        align_object_long_axis_parallel_to_surface_edge(
            self, self.TARGET_NAME, self.lift_region
        )
        self.sim.forward()
        basket_xy, _ = self._lift_target_pose()
        if not self.place_object_on_support_contact(
            self.TARGET_NAME,
            self.lift_region,
            fixture=self.lift_fixture,
            xy=basket_xy,
        ):
            raise RuntimeError(
                f"{self.scene_name}: no stable basket pose on walk surface"
            )
        self._lift_initial_z = self._target_root_z()
        basket_position = np.asarray(
            self.sim.data.get_body_xpos(self.objects[self.TARGET_NAME].root_body),
            dtype=float,
        )
        self._navigation_start_source_horizontal_distance = float(
            np.linalg.norm(basket_position[:2] - self._robot_base_xy())
        )
        center, _ = self.lift_region.world_pose(self.sim)
        preferred = self.scene_task_spec().get("approach_direction")
        directions = self.support_region_long_edge_approach_directions(
            self.lift_region,
            toward_xy=self._robot_base_xy() if preferred is None else None,
            sim=self.sim,
        )
        if preferred is None:
            direction = np.asarray(directions[0], dtype=float)
        else:
            preferred = np.asarray(preferred, dtype=float)
            direction = np.asarray(
                max(
                    directions,
                    key=lambda candidate: float(np.dot(candidate, preferred)),
                ),
                dtype=float,
            )
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        extent = self.support_region_radial_extent(
            self.lift_region,
            direction,
            sim=self.sim,
        )
        operation_reference_xy = center[:2] + direction * (
            extent + float(self.STAND_DISTANCE)
        )
        self._operation_reference_xy = operation_reference_xy
        self._operation_source_horizontal_distance = float(
            np.linalg.norm(basket_position[:2] - operation_reference_xy)
        )
        self._basket_initial_position = np.asarray(
            self.sim.data.get_body_xpos(self.objects[self.TARGET_NAME].root_body),
            dtype=float,
        ).copy()
        self._grasped_at_source_height = False
        self._walk_stage = 0
        self._walk_approach_hold = 0
        self._walk_lift_hold = 0

    def _robot_near_lift_surface(self):
        distance = self.robot_fixture_collision_distance(
            self.lift_fixture,
            max_distance=1.5,
        )
        return bool(np.isfinite(distance) and distance <= self._walk_approach_distance)

    def _basket_initial_state_intact(self):
        current = np.asarray(
            self.sim.data.get_body_xpos(self.objects[self.TARGET_NAME].root_body),
            dtype=float,
        )
        return bool(
            float(np.linalg.norm(current - self._basket_initial_position))
            <= self.INITIAL_POSE_TOLERANCE
            and self.object_released(self.TARGET_NAME, th=0.18)
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        near = self._robot_near_lift_surface()
        if self._walk_stage == 0:
            ready = near and self._basket_initial_state_intact()
            self._walk_approach_hold = self._walk_approach_hold + 1 if ready else 0
            if self._walk_approach_hold >= self.APPROACH_HOLD_STEPS:
                self._walk_stage = 1
                self._report_stage_complete(
                    1,
                    2,
                    next_stage="lift_basket",
                )
            self._walk_lift_hold = 0
            return ret

        height = self._target_root_z()
        grasped = self.g1_hand_grasping_object(self.TARGET_NAME)
        if grasped and height <= self._lift_initial_z + self.GRASP_HEIGHT_ALLOWANCE:
            self._grasped_at_source_height = True
        valid = bool(
            near
            and self._grasped_at_source_height
            and grasped
            and height >= self._lift_initial_z + self.MIN_LIFT_DELTA
        )
        self._walk_lift_hold = self._walk_lift_hold + 1 if valid else 0
        return ret

    def _check_success(self):
        return bool(self._walk_stage == 1 and self._walk_lift_hold >= self.HOLD_STEPS)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.scene_task_spec()["lang"]
        ep_meta["start_distance"] = float(self._walk_start_distance)
        ep_meta["navigation_start_source_horizontal_distance"] = float(
            self._navigation_start_source_horizontal_distance
        )
        ep_meta["operation_reference_xy"] = self._operation_reference_xy.tolist()
        ep_meta["operation_source_horizontal_distance"] = float(
            self._operation_source_horizontal_distance
        )
        ep_meta["approach_distance"] = float(self._walk_approach_distance)
        ep_meta["removed_fixtures"] = list(self.removed_walk_lift_fixture_ids)
        ep_meta["stage_order"] = [
            "approach the basket work surface while the basket stays at reset",
            "grasp the basket near reset height and lift it 0.30 m nearby",
        ]
        return ep_meta


class IThorBedroomApproachDeskAndLiftBasket(
    _BedroomWalkLiftBasketTask,
    IThorBedroomEnv,
):
    """Approach the main desk and lift its basket."""

    SPEC_TASK = "approach_desk_and_lift_basket"
    SUPPORTED_SCENE_IDS = _BEDROOM_SCENE_IDS
    SCENE_SPECS = _all_desk_walk_lift_specs("walk to the desk and lift the basket")


class IThorBedroomApproachDeskAndPutCanInBasket(
    _BedroomWalkPnPTask,
    IThorBedroomEnv,
):
    SPEC_TASK = "approach_desk_and_put_can_in_basket"
    SUPPORTED_SCENE_IDS = _BEDROOM_SCENE_IDS
    SCENE_SPECS = _desk_walk_pnp_specs("walk to the desk and put the can in the basket")
    WORKSPACE_SPEC_TASK = "grasp_workspace"
    WORK_SURFACE_FIXTURES, WORK_SURFACE_REGIONS = _surface_maps(SCENE_SPECS)
    SOURCE_GROUP, RECEPTACLE_GROUP = "can", "basket"
    SOURCE_MJCF_PATH = asset_xml_path(CAN_MODEL)
    RECEPTACLE_MJCF_PATH = asset_xml_path(BASKET_MODEL)
    SOURCE_SCALE, RECEPTACLE_SCALE = CAN_SCALE, BASKET_SCALE


class IThorBedroomApproachDeskAndPutMugInBasket(
    _BedroomWalkPnPTask,
    IThorBedroomEnv,
):
    SPEC_TASK = "approach_desk_and_put_mug_in_basket"
    SUPPORTED_SCENE_IDS = _BEDROOM_SCENE_IDS
    SCENE_SPECS = _desk_walk_pnp_specs("walk to the desk and put the mug in the basket")
    WORKSPACE_SPEC_TASK = "grasp_workspace"
    WORK_SURFACE_FIXTURES, WORK_SURFACE_REGIONS = _surface_maps(SCENE_SPECS)
    SOURCE_GROUP, RECEPTACLE_GROUP = "mug", "basket"
    SOURCE_MJCF_PATH = asset_xml_path(MUG_MODEL)
    RECEPTACLE_MJCF_PATH = asset_xml_path(BASKET_MODEL)
    SOURCE_SCALE, RECEPTACLE_SCALE = MUG_SCALE, BASKET_SCALE
    RECEPTACLE_BOTTOM_TOLERANCE = 0.035


class IThorBedroomApproachDeskAndPutBoxedFoodInBasket(
    _BedroomWalkPnPTask,
    IThorBedroomEnv,
):
    SPEC_TASK = "approach_desk_and_put_boxed_food_in_basket"
    SUPPORTED_SCENE_IDS = _BEDROOM_SCENE_IDS
    SCENE_SPECS = _desk_walk_pnp_specs(
        "walk to the desk and put the boxed food in the basket"
    )
    WORKSPACE_SPEC_TASK = "grasp_workspace"
    WORK_SURFACE_FIXTURES, WORK_SURFACE_REGIONS = _surface_maps(SCENE_SPECS)
    SOURCE_GROUP, RECEPTACLE_GROUP = "boxed_food", "basket"
    SOURCE_MJCF_PATH = asset_xml_path(BOXED_FOOD_MODEL)
    RECEPTACLE_MJCF_PATH = asset_xml_path(BASKET_MODEL)
    SOURCE_SCALE, RECEPTACLE_SCALE = BOXED_FOOD_SCALE, BASKET_SCALE


__all__ = [
    "IThorBedroomApproachDeskAndLiftBasket",
    "IThorBedroomApproachDeskAndPutBoxedFoodInBasket",
    "IThorBedroomApproachDeskAndPutCanInBasket",
    "IThorBedroomApproachDeskAndPutMugInBasket",
]
