"""Walk to the calibrated office trash can in a Holodeck scene."""

import mujoco
import numpy as np

from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckApproachTrashCan(HolodeckEnv):
    """Navigate from a calibrated start and stand near the office trash can."""

    ATOMIC_SKILL = "ApproachTrashCan"
    TASK_FAMILY = "navigation"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "approach_trash_can"
    HOLD_STEPS = 10
    SUCCESS_RADIUS = 1.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    MIN_INITIAL_DISTANCE = 0.75

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=("trash_can_body", "robot_pose", "success_radius"),
        )
        print(
            f"[holodeck] trash can body={self.task_spec['trash_can_body']} "
            f"scene={self.scene_name}; target position is baked in the scene XML",
            flush=True,
        )
        self.trash_can_fixture = next(
            (
                fixture
                for fixture in self.fixtures.values()
                if fixture.body_name == self.task_spec["trash_can_body"]
            ),
            None,
        )
        self.trash_can_body = self.task_spec["trash_can_body"]
        if self.trash_can_fixture is not None:
            self.fixture_refs["approach_trash_can"] = self.trash_can_fixture
        # The baked trash can is a target body, but it is not guaranteed to be
        # represented in the semantic fixture registry.  The calibrated robot
        # pose must therefore not depend on finding a fixture wrapper.
        pose = self.task_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        # The baked MuJoCo model is not assigned to ``self.sim`` yet when
        # scene validity is checked during construction.  Body existence is
        # verified lazily on the first distance calculation instead.
        return bool(self.trash_can_body)

    def _robot_base_xy(self):
        root_body = self.robots[0].robot_model.root_body
        return np.asarray(self.sim.data.get_body_xpos(root_body), dtype=float)[:2]

    def success_distance(self):
        """Return the nearest distance between robot and trash-can geometry."""
        target_body_id = self.sim.model.body_name2id(self.trash_can_body)
        target_geoms = self._collidable_geoms(
            self._body_tree_ids([target_body_id])
        )
        robot_body_id = self.sim.model.body_name2id(
            self.robots[0].robot_model.root_body
        )
        robot_geoms = self._collidable_geoms(
            self._body_tree_ids([robot_body_id])
        )
        if not target_geoms or not robot_geoms:
            return np.inf

        fromto = np.empty(6, dtype=np.float64)
        distances = [
            float(
                mujoco.mj_geomDistance(
                    self.sim.model._model,
                    self.sim.data._data,
                    robot_geom,
                    target_geom,
                    np.inf,
                    fromto,
                )
            )
            for robot_geom in robot_geoms
            for target_geom in target_geoms
        ]
        finite = [distance for distance in distances if np.isfinite(distance)]
        return max(0.0, min(finite)) if finite else np.inf

    def _setup_scene(self):
        super()._setup_scene()
        if self.trash_can_body not in self.sim.model.body_names:
            raise RuntimeError(
                f"{self.scene_name}: calibrated trash can body is missing: "
                f"{self.trash_can_body}"
            )
        if self.success_distance() < self.MIN_INITIAL_DISTANCE:
            raise RuntimeError(
                f"{self.scene_name}: robot starts too close to the trash can; "
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
        meta["lang"] = "walk to the office trash can and stand nearby"
        return meta


__all__ = ["HolodeckApproachTrashCan"]
