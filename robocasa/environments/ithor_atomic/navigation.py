"""iTHOR semantic navigation atomic tasks."""

import numpy as np
from robosuite.utils.mjcf_utils import new_site

from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin


class RoomNavMixin:
    """Navigate to the vicinity of a semantic fixture."""

    ATOMIC_SKILL = "ApproachFurniture"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "navigation"

    TARGET_CATEGORY: str = None
    HOLD_STEPS: int = 1
    SUCCESS_DISTANCE: float = 0.1
    SHOW_TARGET_REGION: bool = False
    TARGET_REGION_SITE_NAME: str = "navigation_target_region"

    def _navigation_fixture_query(self):
        """Return the semantic query used to resolve the navigation target."""
        return dict(category=self.TARGET_CATEGORY)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.target_fixture = self.register_fixture_ref(
            "navigation_target", self._navigation_fixture_query()
        )
        target_xy = np.asarray(
            getattr(self, "target_xy", self.target_fixture.center[:2]), dtype=float
        )
        target_tolerance = getattr(self, "target_tolerance", None)
        show_region = bool(getattr(self, "show_target_region", self.SHOW_TARGET_REGION))
        if show_region and target_tolerance is not None:
            self.mujoco_arena.worldbody.append(
                new_site(
                    name=self.TARGET_REGION_SITE_NAME,
                    type="box",
                    pos=[*target_xy, 0.012],
                    size=[*np.asarray(target_tolerance, dtype=float), 0.01],
                    rgba=[0.1, 0.9, 0.2, 0.35],
                    group=0,
                )
            )

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.get_fixture(**self._navigation_fixture_query()) is not None
        )

    def _setup_scene(self):
        super()._setup_scene()
        self._in_range_steps = 0
        self._furniture_center = self.target_fixture.world_center(self.sim)[:2]
        self._navigation_target_xy = np.asarray(
            getattr(self, "target_xy", self._furniture_center), dtype=float
        )
        default_tolerance = (
            np.asarray(self.target_fixture.half_size[:2], dtype=float)
            + self.SUCCESS_DISTANCE
        )
        self._navigation_target_tolerance = np.asarray(
            getattr(self, "target_tolerance", default_tolerance), dtype=float
        )
        if self._navigation_target_xy.shape != (2,):
            raise ValueError("navigation target_xy must contain exactly two values")
        if self._navigation_target_tolerance.shape != (2,):
            raise ValueError(
                "navigation target_tolerance must contain exactly two values"
            )
        if np.any(self._navigation_target_tolerance < 0.0):
            raise ValueError("navigation target_tolerance must be nonnegative")

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self._robot_is_near_target_fixture():
            self._in_range_steps += 1
        else:
            self._in_range_steps = 0
        return ret

    def _robot_is_near_target_fixture(self):
        delta = np.abs(self._robot_base_xy() - self._navigation_target_xy)
        return bool(np.all(delta <= self._navigation_target_tolerance))

    def _check_success(self):
        return self._in_range_steps >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["target_pos"] = [float(v) for v in self._navigation_target_xy]
        ep_meta["target_tolerance"] = [
            float(v) for v in self._navigation_target_tolerance
        ]
        ep_meta["lang"] = (
            f"walk over to the {self.TARGET_CATEGORY.lower()} " f"and stand next to it"
        )
        return ep_meta


class ExactSceneNavigation(SceneSpecMixin):
    """Navigate from a distant spawn into an exact fixture's vicinity."""

    ATOMIC_SKILL = "Navigation"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "navigation"
    HOLD_STEPS = 5
    SUCCESS_DISTANCE = 0.35
    MAX_START_HEADING_ERROR_DEGREES = 20.0
    SHOW_TARGET_REGION = False
    INITIAL_ROBOT_JITTER_RADIUS = 0.0

    def _navigation_start_config(self):
        spec = self.scene_task_spec()
        pose = spec["start_pose"]
        if not isinstance(pose, dict) or set(pose) != {"xy", "yaw"}:
            raise ValueError(
                f"{self.scene_name}: start_pose must contain exactly xy and yaw"
            )
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        if xy.shape != (2,) or not np.all(np.isfinite(xy)) or not np.isfinite(yaw):
            raise ValueError(f"{self.scene_name}: invalid navigation start_pose")

        minimum_distance = float(spec["minimum_start_distance"])
        if (
            not np.isfinite(minimum_distance)
            or minimum_distance < self._navigation_success_distance + 0.15
        ):
            raise ValueError(
                f"{self.scene_name}: minimum_start_distance must exceed the goal "
                "distance by at least 0.15 m"
            )

        removal_ids = spec["remove_fixtures"]
        if not isinstance(removal_ids, (list, tuple)) or isinstance(
            removal_ids, str
        ):
            raise ValueError(f"{self.scene_name}: remove_fixtures must be a list")
        removal_ids = tuple(str(stable_id) for stable_id in removal_ids)
        if len(removal_ids) != len(set(removal_ids)):
            raise ValueError(f"{self.scene_name}: duplicate remove_fixtures entry")
        return xy, yaw, minimum_distance, removal_ids

    def _navigation_fixture_query(self):
        return {"id": self.scene_task_spec()["fixture"]}

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self._navigation_success_distance = float(
            self.scene_task_spec().get("success_distance", self.SUCCESS_DISTANCE)
        )
        if (
            not np.isfinite(self._navigation_success_distance)
            or self._navigation_success_distance <= 0.0
        ):
            raise ValueError(
                f"{self.scene_name}: success_distance must be finite and positive"
            )
        self.target_fixture = self.register_fixture_ref(
            "navigation_target", self._navigation_fixture_query()
        )
        start_xy, start_yaw, _, removal_ids = self._navigation_start_config()
        if self.target_fixture.stable_id in removal_ids:
            raise ValueError(f"{self.scene_name}: navigation target cannot be removed")
        removals = []
        removal_body_names = set()
        target_body_names = set(self.target_fixture.body_names)
        for stable_id in removal_ids:
            fixture = self.get_fixture(id=stable_id)
            if fixture is None or fixture.stable_id != stable_id:
                raise LookupError(
                    f"{self.scene_name}: missing exact removal fixture {stable_id!r}"
                )
            if not fixture.body_name:
                raise ValueError(
                    f"{self.scene_name}: removal fixture {stable_id!r} has no scene body"
                )
            if fixture.body_name in removal_body_names:
                raise ValueError(
                    f"{self.scene_name}: multiple removals resolve to body "
                    f"{fixture.body_name!r}"
                )
            if (
                fixture.body_name in target_body_names
                or self.target_fixture.body_name in fixture.body_names
            ):
                raise ValueError(
                    f"{self.scene_name}: removal fixture {stable_id!r} overlaps "
                    "the navigation target body"
                )
            removal_body_names.add(fixture.body_name)
            removals.append(fixture)
        removed_bodies = self.remove_semantic_fixtures(removals)
        missing_bodies = {
            fixture.body_name
            for fixture in removals
            if fixture.body_name not in removed_bodies
        }
        if missing_bodies:
            raise RuntimeError(
                f"{self.scene_name}: could not remove exact scene bodies "
                f"{sorted(missing_bodies)}"
            )
        self._navigation_removed_fixture_ids = list(removal_ids)
        self.set_robot_base_pose([start_xy[0], start_xy[1], 0.0], start_yaw)

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.get_fixture(**self._navigation_fixture_query()) is not None
        )

    def _setup_scene(self):
        super()._setup_scene()
        self._reset_task_progress()
        self._navigation_target_xy = self.target_fixture.world_center(self.sim)[:2]
        start_xy, start_yaw, minimum_distance, _ = self._navigation_start_config()
        if not self.robot_footprint_is_on_floor(start_xy, radius=0.25):
            raise RuntimeError(
                f"{self.scene_name}: navigation start footprint is not fully on Floor"
            )
        if not self._set_robot_free_pose(start_xy, start_yaw):
            raise RuntimeError(f"{self.scene_name}: robot has no free base joint")
        if not self._calibrate_robot_free_joint_height():
            raise RuntimeError(
                f"{self.scene_name}: navigation start has an unresolved collision"
            )
        self.sim.forward()
        if self.count_body_collisions() != 0:
            raise RuntimeError(f"{self.scene_name}: navigation start collides with scene")

        actual_distance = self.robot_fixture_collision_distance(
            self.target_fixture, max_distance=max(5.0, minimum_distance + 1.0)
        )
        if not np.isfinite(actual_distance) or actual_distance < minimum_distance:
            raise RuntimeError(
                f"{self.scene_name}: navigation start collision distance "
                f"{actual_distance:.3f} m is below the required "
                f"{minimum_distance:.3f} m"
            )
        self.robot_base_pos = np.array([start_xy[0], start_xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, start_yaw])
        self._navigation_start_fixture_distance = float(actual_distance)
        self._navigation_minimum_start_distance = float(minimum_distance)
        self._navigation_start_center_distance = float(
            np.linalg.norm(start_xy - self._navigation_target_xy)
        )
        desired_yaw = float(
            np.arctan2(
                self._navigation_target_xy[1] - start_xy[1],
                self._navigation_target_xy[0] - start_xy[0],
            )
        )
        heading_error = float(
            np.arctan2(
                np.sin(start_yaw - desired_yaw), np.cos(start_yaw - desired_yaw)
            )
        )
        self._navigation_start_heading_error_degrees = abs(
            float(np.rad2deg(heading_error))
        )
        if (
            self._navigation_start_heading_error_degrees
            > self.MAX_START_HEADING_ERROR_DEGREES
        ):
            raise RuntimeError(
                f"{self.scene_name}: navigation start faces "
                f"{self._navigation_start_heading_error_degrees:.1f} degrees away "
                "from the target"
            )
        if self._robot_is_near_target():
            raise RuntimeError(
                f"{self.scene_name}: navigation start is already at the goal"
            )

    def _reset_task_progress(self):
        parent = getattr(super(), "_reset_task_progress", None)
        if callable(parent):
            parent()
        self._in_range_steps = 0

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._in_range_steps = (
            self._in_range_steps + 1 if self._robot_is_near_target() else 0
        )
        return ret

    def _robot_is_near_target(self):
        geom_distance = self.robot_fixture_collision_distance(
            self.target_fixture, max_distance=1.0
        )
        return bool(
            np.isfinite(geom_distance)
            and geom_distance <= self._navigation_success_distance
        )

    def _check_success(self):
        return self._in_range_steps >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.scene_task_spec()["lang"]
        ep_meta["target_fixture"] = self.target_fixture.stable_id
        ep_meta["target_pos"] = [float(v) for v in self._navigation_target_xy]
        ep_meta["navigation_start_pose"] = {
            "xy": [float(v) for v in self.robot_base_pos[:2]],
            "yaw": float(self.robot_base_ori[2]),
        }
        ep_meta["navigation_start_fixture_distance"] = float(
            self._navigation_start_fixture_distance
        )
        ep_meta["navigation_start_center_distance"] = float(
            self._navigation_start_center_distance
        )
        ep_meta["navigation_start_heading_error_degrees"] = float(
            self._navigation_start_heading_error_degrees
        )
        ep_meta["minimum_start_distance"] = float(
            self._navigation_minimum_start_distance
        )
        ep_meta["removed_fixtures"] = list(self._navigation_removed_fixture_ids)
        ep_meta["success_definition"] = (
            "robot collision geometry must remain within "
            f"{self._navigation_success_distance:.2f} m of the named target fixture "
            f"for {self.HOLD_STEPS} consecutive control steps"
        )
        return ep_meta


__all__ = ["ExactSceneNavigation", "RoomNavMixin"]
