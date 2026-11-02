"""Scene-calibrated navigation through an ordered list of fixture goals."""

import re
from collections.abc import Mapping

import numpy as np

from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_composite.base import IThorAtomicTask


class MultiStageNavigationMixin(SceneSpecMixin):
    """Visit two or more fixture goals in declared order."""

    ATOMIC_SKILL = "NavigationSequence"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "multi_stage_navigation"
    COMPOSITE_ORDERING = "sequential"
    STAGE_NAMES = ()
    SUCCESS_DISTANCE = 0.35
    START_CLEARANCE = 0.15
    MIN_STAGE_TRAVEL = 0.30
    INTERMEDIATE_HOLD_STEPS = 5
    HOLD_STEPS = 5
    INITIAL_POSE_JITTER = False

    def _get_atomic_tasks(self):
        stage_names = tuple(self.STAGE_NAMES)
        if len(stage_names) < 2:
            raise ValueError(
                f"{type(self).__name__}: STAGE_NAMES must contain at least two "
                "navigation stages"
            )
        route_lengths = {
            len(spec.get("stages", ())) for spec in self.SCENE_SPECS.values()
        }
        if route_lengths != {len(stage_names)}:
            raise ValueError(
                f"{type(self).__name__}: every scene spec must contain "
                f"{len(stage_names)} stages, got {sorted(route_lengths)}"
            )
        stage_slugs = []
        for stage_name in stage_names:
            slug = re.sub(r"[^a-z0-9]+", "_", str(stage_name).lower()).strip("_")
            if not slug:
                raise ValueError(
                    f"{type(self).__name__}: invalid stage name {stage_name!r}"
                )
            stage_slugs.append(slug)

        navigation_hold_seconds = self._navigation_hold_seconds()
        tasks = []
        for index, (stage_name, slug) in enumerate(zip(stage_names, stage_slugs)):
            task_name = f"navigate_to_{slug}"
            if stage_slugs.count(slug) > 1:
                task_name = f"{task_name}_{index + 1}"
            tasks.append(
                IThorAtomicTask(
                    name=task_name,
                    success="_active_navigation_stage_complete",
                    task_cls="RootXYNavigation",
                    hold_steps=(
                        None
                        if navigation_hold_seconds is not None
                        else (
                            self.INTERMEDIATE_HOLD_STEPS
                            if index < len(stage_names) - 1
                            else (
                                None
                                if self.SUCCESS_HOLD_SECONDS is not None
                                else self.HOLD_STEPS
                            )
                        )
                    ),
                    on_complete="_record_navigation_stage_completion",
                    metadata={
                        "skill": "navigation",
                        "route_index": index,
                        "target": str(stage_name),
                        "success_distance": self._stage_success_distance(index),
                    },
                )
            )
        return tuple(tasks)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        robot_pos, robot_yaw = self._configured_robot_pose()
        self.set_robot_base_pose(robot_pos, robot_yaw)
        self.route_fixtures = [
            self.register_fixture_ref(
                f"route_target_{index + 1}",
                {"id": fixture_id},
            )
            for index, fixture_id in enumerate(self.scene_task_spec()["stages"])
        ]

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and len(self.route_fixtures) == len(self.STAGE_NAMES)
            and all(fixture is not None for fixture in self.route_fixtures)
        )

    def _setup_scene(self):
        super()._setup_scene()
        self._route_target_positions = [
            fixture.world_center(self.sim)[:2].copy() for fixture in self.route_fixtures
        ]
        self._validate_configured_robot_pose()

    def _reset_composite_progress(self):
        super()._reset_composite_progress()
        self._previous_stage_reached_xy = None

    def _navigation_hold_seconds(self):
        value = getattr(self, "NAVIGATION_HOLD_SECONDS", None)
        if value is None:
            return None
        seconds = float(value)
        if not np.isfinite(seconds) or seconds <= 0.0:
            raise ValueError(
                f"{type(self).__name__}: NAVIGATION_HOLD_SECONDS must be "
                "finite and positive"
            )
        return seconds

    def _required_stage_hold_steps(self, task):
        seconds = self._navigation_hold_seconds()
        if seconds is None:
            return super()._required_stage_hold_steps(task)
        return max(
            1,
            int(round(seconds * self.control_freq)),
        )

    def _stage_success_distance(self, stage_index):
        stage_name = str(self.STAGE_NAMES[stage_index]).lower()
        overrides = getattr(self, "NAVIGATION_SUCCESS_DISTANCES", {})
        distance = float(overrides.get(stage_name, self.SUCCESS_DISTANCE))
        if not np.isfinite(distance) or distance < 0.0:
            raise ValueError(
                f"{type(self).__name__}: invalid success distance for "
                f"stage {stage_name!r}"
            )
        return distance

    def _fixture_planar_distance(self, fixture):
        """Return root-XY distance to the fixture's oriented XY footprint."""

        robot_xy = np.asarray(self._robot_base_xy(), dtype=float)
        fixture_xy = np.asarray(fixture.world_center(self.sim), dtype=float)[:2]
        delta = robot_xy - fixture_xy
        yaw = float(fixture.yaw)
        c, s = np.cos(-yaw), np.sin(-yaw)
        local_delta = np.array(
            [
                c * delta[0] - s * delta[1],
                s * delta[0] + c * delta[1],
            ]
        )
        half_size = (
            fixture.half_size
            if fixture.local_half_size is None
            else fixture.local_half_size
        )
        outside = np.maximum(
            np.abs(local_delta) - np.asarray(half_size, dtype=float)[:2],
            0.0,
        )
        return float(np.linalg.norm(outside))

    def _fixture_is_near(self, fixture):
        distance = self._fixture_planar_distance(fixture)
        success_distance = self._stage_success_distance(self.stage_index)
        return bool(np.isfinite(distance) and distance <= success_distance)

    def _configured_robot_pose(self):
        pose = self.scene_task_spec()["robot_pose"]
        task_name = getattr(self, "SPEC_TASK", type(self).__name__)
        if not isinstance(pose, Mapping) or "pos" not in pose or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}/{task_name}: robot_pose needs pos + yaw"
            )
        pos = np.asarray(pose["pos"], dtype=float)
        yaw = float(pose["yaw"])
        if pos.shape != (3,) or not np.isfinite(pos).all() or not np.isfinite(yaw):
            raise ValueError(
                f"{self.scene_name}/{task_name}: robot_pose must contain "
                "finite XYZ + yaw"
            )
        return pos, yaw

    def _validate_configured_robot_pose(self):
        # Refresh contacts and geom distances after reset before auditing the
        # exact model pose configured during scene construction.
        self.sim.forward()
        configured_pos, configured_yaw = self._configured_robot_pose()
        actual_xy = self._robot_base_xy()
        actual_yaw = float(self._robot_base_yaw())
        configured_model_xy = np.asarray(self.robot_base_pos, dtype=float)[:2]
        yaw_error = float(
            np.arctan2(
                np.sin(actual_yaw - configured_yaw),
                np.cos(actual_yaw - configured_yaw),
            )
        )
        if not np.allclose(
            configured_model_xy, configured_pos[:2], atol=1e-5
        ) or not np.isclose(yaw_error, 0.0, atol=1e-5):
            raise RuntimeError(
                f"{self.scene_name}: G1 reset pose does not match YAML robot_pose"
            )
        if not self.robot_footprint_is_on_floor(actual_xy, radius=0.25):
            raise RuntimeError(
                f"{self.scene_name}: YAML robot_pose leaves the G1 footprint off-floor"
            )
        collisions = self.count_body_collisions()
        if collisions:
            raise RuntimeError(
                f"{self.scene_name}: YAML robot_pose has {collisions} G1 collisions"
            )
        min_distance = float(self._stage_success_distance(0) + self.START_CLEARANCE)
        fixture = self.route_fixtures[0]
        distance = self._fixture_planar_distance(fixture)
        if not np.isfinite(distance) or distance <= min_distance:
            raise RuntimeError(
                f"{self.scene_name}: YAML robot_pose is only {distance:.3f} m "
                f"from first route target {fixture.stable_id}; needs > "
                f"{min_distance:.3f} m"
            )

    def _active_navigation_stage_complete(self):
        fixture = self.route_fixtures[self.stage_index]
        if not self._fixture_is_near(fixture):
            return False
        if self.stage_index == 0:
            return True
        if self._previous_stage_reached_xy is None:
            return False
        travel = float(
            np.linalg.norm(self._robot_base_xy() - self._previous_stage_reached_xy)
        )
        return travel >= float(self.MIN_STAGE_TRAVEL)

    def _record_navigation_stage_completion(self):
        self._previous_stage_reached_xy = self._robot_base_xy().copy()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.TASK_LANG
        ep_meta["route_fixtures"] = [
            fixture.stable_id for fixture in self.route_fixtures
        ]
        ep_meta["route_target_positions"] = [
            [float(value) for value in target]
            for target in self._route_target_positions
        ]
        robot_pos, robot_yaw = self._configured_robot_pose()
        ep_meta["route_robot_pose"] = {
            "pos": [float(value) for value in robot_pos],
            "yaw": float(robot_yaw),
        }
        ep_meta["composite_stages"] = [
            f"walk to the named {stage_name}" for stage_name in self.STAGE_NAMES
        ]
        route_hold_steps = [
            self._required_stage_hold_steps(task)
            for task in self._atomic_tasks
        ]
        ep_meta["route_hold_steps"] = route_hold_steps
        stage_distances = [
            self._stage_success_distance(index)
            for index in range(len(self.STAGE_NAMES))
        ]
        ep_meta["route_success_distances"] = stage_distances
        distance_summary = ", ".join(
            f"{stage_name}: {distance:.2f} m"
            for stage_name, distance in zip(self.STAGE_NAMES, stage_distances)
        )
        ep_meta["success_definition"] = (
            f"visit all {len(self.route_fixtures)} named fixtures in order; remain "
            "within the configured planar root distance of each fixture footprint "
            f"({distance_summary}) for the configured stage holds "
            f"{route_hold_steps} control steps and move at least "
            f"{self.MIN_STAGE_TRAVEL:.2f} m between consecutive stages"
        )
        return ep_meta


__all__ = ["MultiStageNavigationMixin"]
