"""Lift the native Scene 329 baseball bat clear of the floor."""

import mujoco
import numpy as np

from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv


class IThorBedroomLiftBaseballBat(SceneSpecMixin, IThorBedroomEnv):
    TASK_FAMILY = "lift"
    ATOMIC_SKILL = "LiftObject"
    DIFFICULTY_LEVEL = "L1"
    SUPPORTED_SCENE_IDS = (329,)
    TARGET_STABLE_ID = "FloorPlan329:BaseballBat_1_13"
    SCENE_SPECS = {
        "FloorPlan329": {
            "fixture": TARGET_STABLE_ID,
            "robot_pose": {"xy": [0.735, 0.737], "yaw": 2.44947676},
            "lang": "Pick up the baseball bat.",
        },
    }
    INITIAL_POSE_JITTER = False
    # Ignore the contact margin and millimetre-scale solver noise at the floor.
    MIN_GROUND_CLEARANCE = 0.005

    def _get_scene_robot_pose_calibration(self):
        pose = self.scene_task_spec()["robot_pose"]
        return np.array([*pose["xy"], 0.0]), float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.bat_fixture = self.get_fixture(id=self.TARGET_STABLE_ID)
        if self.bat_fixture is None:
            raise RuntimeError(f"{self.scene_name}: baseball bat is missing")

    def _reset_internal(self):
        super()._reset_internal()
        self._bat_hold_steps = 0
        self._floor_geom_id = self.sim.model.geom_name2id("floor")
        bodies = self._fixture_body_ids(self.bat_fixture, include_siblings=True)
        self._bat_collision_meshes = []
        for geom_id in self._geom_ids_for_bodies(bodies):
            model = self.sim.model
            if not (model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]):
                continue
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                raise RuntimeError("Expected the native baseball bat collision mesh")
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            self._bat_collision_meshes.append(
                (geom_id, model.mesh_vert[start : start + count].copy())
            )
        if not self._bat_collision_meshes:
            raise RuntimeError("Baseball bat has no active collision geometry")

    def _bat_ground_clearance(self):
        """Minimum signed distance of the entire collision mesh to Floor."""
        floor_pos = self.sim.data.geom_xpos[self._floor_geom_id]
        normal = self.sim.data.geom_xmat[self._floor_geom_id].reshape(3, 3)[:, 2]
        clearance = np.inf
        for geom_id, vertices in self._bat_collision_meshes:
            rotation = self.sim.data.geom_xmat[geom_id].reshape(3, 3)
            position = self.sim.data.geom_xpos[geom_id]
            distances = vertices @ (rotation.T @ normal) + np.dot(
                position - floor_pos, normal
            )
            clearance = min(clearance, float(np.min(distances)))
        return clearance

    def _bat_is_off_ground(self):
        return self._bat_ground_clearance() > self.MIN_GROUND_CLEARANCE

    def _post_action(self, action):
        self._bat_hold_steps = (
            self._bat_hold_steps + 1 if self._bat_is_off_ground() else 0
        )
        return super()._post_action(action)

    def _check_success(self):
        return self._bat_hold_steps >= self.HOLD_STEPS

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = self.scene_task_spec()["lang"]
        meta["target_fixture_id"] = self.TARGET_STABLE_ID
        meta["minimum_ground_clearance"] = self.MIN_GROUND_CLEARANCE
        meta["requires_grasp"] = False
        return meta


__all__ = ["IThorBedroomLiftBaseballBat"]
