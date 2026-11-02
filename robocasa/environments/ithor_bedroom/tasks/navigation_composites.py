"""Ordered routes composed from the three Bedroom navigation atomics."""

from __future__ import annotations

import copy

import numpy as np

from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_spec,
)


_SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)
_NAVIGATION_SPEC_FIELDS = (
    "fixture",
    "start_pose",
    "minimum_start_distance",
    "remove_fixtures",
    "lang",
)


def _navigation_route_specs(first_task, second_task):
    specs = {}
    for scene_id in _SUPPORTED_SCENE_IDS:
        scene_name = f"FloorPlan{scene_id}"
        first = copy.deepcopy(
            get_ithor_bedroom_task_spec(
                scene_name,
                first_task,
                required=_NAVIGATION_SPEC_FIELDS,
            )
        )
        second = copy.deepcopy(
            get_ithor_bedroom_task_spec(
                scene_name,
                second_task,
                required=_NAVIGATION_SPEC_FIELDS,
            )
        )
        if first["fixture"] == second["fixture"]:
            raise ValueError(
                f"{scene_name}: a Bedroom navigation route needs two "
                "distinct fixtures"
            )
        removal_ids = list(
            dict.fromkeys(
                (*first["remove_fixtures"], *second["remove_fixtures"])
            )
        )
        specs[scene_name] = {
            "first_fixture": first["fixture"],
            "second_fixture": second["fixture"],
            "start_pose": copy.deepcopy(first["start_pose"]),
            "minimum_start_distance": float(first["minimum_start_distance"]),
            "remove_fixtures": removal_ids,
            "success_distance": 0.35,
            "minimum_stage_travel": 0.30,
            "lang": f'{first["lang"]}, then {second["lang"]}',
        }
    if tuple(specs) != tuple(
        f"FloorPlan{scene_id}" for scene_id in _SUPPORTED_SCENE_IDS
    ):
        raise AssertionError("Bedroom navigation route scene map is incomplete")
    return specs


class _BedroomNavigationRouteMixin(
    CompositeStageReporterMixin,
    SceneSpecMixin,
):
    """Require two exact navigation targets in the declared order."""

    ATOMIC_SKILL = "OrderedNavigation"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "two_stage_navigation"
    COMPOSITE_ORDERING = ("first_navigation", "second_navigation")
    FIRST_HOLD_STEPS = 5
    HOLD_STEPS = 5
    INITIAL_ROBOT_JITTER_RADIUS = 0.0

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.first_fixture = self.get_fixture(id=spec["first_fixture"])
        self.second_fixture = self.get_fixture(id=spec["second_fixture"])
        if self.first_fixture is None or self.second_fixture is None:
            return
        if self.first_fixture.stable_id == self.second_fixture.stable_id:
            raise ValueError(f"{self.scene_name}: route targets must be distinct")
        removal_ids = spec["remove_fixtures"]
        if not isinstance(removal_ids, (list, tuple)) or isinstance(
            removal_ids, str
        ):
            raise ValueError("remove_fixtures must be a list")
        removal_ids = tuple(removal_ids)
        if len(removal_ids) != len(set(removal_ids)):
            raise ValueError(f"{self.scene_name}: duplicate route removal")
        forbidden = {
            self.first_fixture.stable_id,
            self.second_fixture.stable_id,
        }
        if forbidden.intersection(removal_ids):
            raise ValueError(f"{self.scene_name}: route target cannot be removed")
        removals = []
        for stable_id in removal_ids:
            fixture = self.get_fixture(id=stable_id)
            if fixture is None or fixture.stable_id != stable_id:
                raise LookupError(
                    f"{self.scene_name}: missing route removal {stable_id}"
                )
            removals.append(fixture)
        if removals:
            self.remove_semantic_fixtures(removals)
        self.removed_route_fixture_ids = list(removal_ids)
        self.fixture_refs["first_navigation_target"] = self.first_fixture
        self.fixture_refs["second_navigation_target"] = self.second_fixture
        pose = spec["start_pose"]
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError(f"{self.scene_name}: invalid route start pose")
        self.set_robot_base_pose([xy[0], xy[1], 0.0], yaw)

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.first_fixture is not None
            and self.second_fixture is not None
        )

    def _setup_scene(self):
        super()._setup_scene()
        pose = self.scene_task_spec()["start_pose"]
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        if not (
            self.robot_footprint_is_on_floor(xy, radius=0.25)
            and self._set_robot_free_pose(xy, yaw)
            and self._calibrate_robot_free_joint_height()
        ):
            raise RuntimeError(f"{self.scene_name}: invalid route start pose")
        self.sim.forward()
        if self.count_body_collisions():
            raise RuntimeError(f"{self.scene_name}: route start collides")
        start_distance = self.robot_fixture_collision_distance(
            self.first_fixture, max_distance=5.0
        )
        required = float(self.scene_task_spec()["minimum_start_distance"])
        if not np.isfinite(start_distance) or start_distance < required:
            raise RuntimeError(
                f"{self.scene_name}: route starts {start_distance:.3f} m from "
                f"the first target; expected at least {required:.3f} m"
            )
        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        self._route_start_distance = float(start_distance)
        self._reset_route_progress()

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_route_progress()

    def _reset_route_progress(self):
        self._route_stage = 0
        self._first_hold = 0
        self._second_hold = 0
        self._first_arrival_xy = None

    def _near_fixture(self, fixture):
        distance = self.robot_fixture_collision_distance(fixture, max_distance=1.0)
        return bool(
            np.isfinite(distance)
            and distance <= float(self.scene_task_spec()["success_distance"])
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self._route_stage == 0:
            self._first_hold = (
                self._first_hold + 1 if self._near_fixture(self.first_fixture) else 0
            )
            if self._first_hold >= self.FIRST_HOLD_STEPS:
                self._route_stage = 1
                self._first_arrival_xy = self._robot_base_xy().copy()
                self._report_stage_complete(
                    1, 2, next_stage="navigate_to_second_target"
                )
            self._second_hold = 0
            return ret

        travelled = float(
            np.linalg.norm(self._robot_base_xy() - self._first_arrival_xy)
        )
        valid = bool(
            travelled >= float(self.scene_task_spec()["minimum_stage_travel"])
            and self._near_fixture(self.second_fixture)
        )
        self._second_hold = self._second_hold + 1 if valid else 0
        return ret

    def _check_success(self):
        return self._route_stage == 1 and self._second_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        spec = self.scene_task_spec()
        ep_meta.update(
            {
                "lang": spec["lang"],
                "first_fixture": self.first_fixture.stable_id,
                "second_fixture": self.second_fixture.stable_id,
                "start_distance": self._route_start_distance,
                "success_distance": float(spec["success_distance"]),
                "minimum_stage_travel": float(spec["minimum_stage_travel"]),
                "removed_fixtures": list(self.removed_route_fixture_ids),
                "stage_order": [
                    "reach the first named fixture and remain nearby",
                    "move away and reach the second named fixture",
                ],
            }
        )
        return ep_meta


class IThorBedroomBedThenDesk(_BedroomNavigationRouteMixin, IThorBedroomEnv):
    SCENE_SPECS = _navigation_route_specs("approach_bed", "approach_desk")
    SUPPORTED_SCENE_IDS = _SUPPORTED_SCENE_IDS


class IThorBedroomDeskThenBed(_BedroomNavigationRouteMixin, IThorBedroomEnv):
    SCENE_SPECS = _navigation_route_specs("approach_desk", "approach_bed")
    SUPPORTED_SCENE_IDS = _SUPPORTED_SCENE_IDS


class IThorBedroomBedThenStorage(_BedroomNavigationRouteMixin, IThorBedroomEnv):
    SCENE_SPECS = _navigation_route_specs("approach_bed", "approach_storage")
    SUPPORTED_SCENE_IDS = _SUPPORTED_SCENE_IDS


class IThorBedroomStorageThenBed(_BedroomNavigationRouteMixin, IThorBedroomEnv):
    SCENE_SPECS = _navigation_route_specs("approach_storage", "approach_bed")
    SUPPORTED_SCENE_IDS = _SUPPORTED_SCENE_IDS


class IThorBedroomDeskThenStorage(_BedroomNavigationRouteMixin, IThorBedroomEnv):
    SCENE_SPECS = _navigation_route_specs("approach_desk", "approach_storage")
    SUPPORTED_SCENE_IDS = _SUPPORTED_SCENE_IDS


class IThorBedroomStorageThenDesk(_BedroomNavigationRouteMixin, IThorBedroomEnv):
    SCENE_SPECS = _navigation_route_specs("approach_storage", "approach_desk")
    SUPPORTED_SCENE_IDS = _SUPPORTED_SCENE_IDS


__all__ = [
    "IThorBedroomBedThenDesk",
    "IThorBedroomBedThenStorage",
    "IThorBedroomDeskThenBed",
    "IThorBedroomDeskThenStorage",
    "IThorBedroomStorageThenBed",
    "IThorBedroomStorageThenDesk",
]
