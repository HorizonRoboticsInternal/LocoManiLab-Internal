"""Walk to a calibrated window in a Holodeck scene."""

import mujoco
import numpy as np

from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckApproachWindow(HolodeckEnv):
    """Navigate from a calibrated start and stand near a scene window."""

    ATOMIC_SKILL = "ApproachWindow"
    TASK_FAMILY = "navigation"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "approach_window"
    HOLD_STEPS = 10
    SUCCESS_RADIUS = 2.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    MIN_INITIAL_DISTANCE = 0.65

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=("window_body", "robot_pose", "success_radius"),
        )
        self.window_body = self.task_spec["window_body"]
        self.window_fixture = next(
            (
                fixture
                for fixture in self.fixtures.values()
                if fixture.body_name == self.window_body
            ),
            None,
        )
        if self.window_fixture is not None:
            self.fixture_refs["approach_window"] = self.window_fixture
        pose = self.task_spec["robot_pose"]
        self.set_robot_base_pose(np.asarray(pose["pos"], dtype=float), float(pose["yaw"]))

    def _is_scene_valid(self):
        # The baked model is not assigned to ``self.sim`` during construction.
        return bool(self.window_body)

    def success_distance(self):
        """Return the nearest robot-to-window collision-geometry distance."""
        window_body_id = self.sim.model.body_name2id(self.window_body)
        window_geoms = self._collidable_geoms(
            self._body_tree_ids([window_body_id])
        )
        robot_body_id = self.sim.model.body_name2id(
            self.robots[0].robot_model.root_body
        )
        robot_geoms = self._collidable_geoms(self._body_tree_ids([robot_body_id]))
        if not window_geoms or not robot_geoms:
            return np.inf

        fromto = np.empty(6, dtype=np.float64)
        distances = [
            float(
                mujoco.mj_geomDistance(
                    self.sim.model._model,
                    self.sim.data._data,
                    robot_geom,
                    window_geom,
                    # The initial-distance guard must distinguish far starts
                    # from the success threshold instead of saturating at it.
                    np.inf,
                    fromto,
                )
            )
            for robot_geom in robot_geoms
            for window_geom in window_geoms
        ]
        distances = [distance for distance in distances if distance >= 0]
        return min(distances) if distances else np.inf

    def _setup_scene(self):
        super()._setup_scene()
        if self.success_distance() < self.MIN_INITIAL_DISTANCE:
            raise RuntimeError(
                f"{self.scene_name}: robot starts too close to the window; "
                f"minimum distance is {self.MIN_INITIAL_DISTANCE:.2f} m"
            )
        self._in_range_steps = 0

    def _reset_internal(self):
        super()._reset_internal()
        self._in_range_steps = 0

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self.success_distance() < self.success_radius:
            self._in_range_steps += 1
        else:
            self._in_range_steps = 0
        return ret

    @property
    def success_radius(self):
        return float(self.task_spec.get("success_radius", self.SUCCESS_RADIUS))

    def _check_success(self):
        return self._in_range_steps >= self.HOLD_STEPS

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = self.task_spec.get(
            "lang", "walk to the target window and stand nearby"
        )
        return meta


__all__ = ["HolodeckApproachWindow"]
