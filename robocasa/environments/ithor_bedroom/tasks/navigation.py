"""Exact-fixture Bedroom navigation tasks."""

from robocasa.environments.ithor_atomic.navigation import ExactSceneNavigation
from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_specs,
    get_supported_scene_ids,
)
from robocasa.utils import env_utils as EnvUtils


class IThorBedroomApproachBed(ExactSceneNavigation, IThorBedroomEnv):
    SPEC_TASK = "approach_bed"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK,
        required=(
            "fixture",
            "start_pose",
            "minimum_start_distance",
            "remove_fixtures",
            "lang",
        ),
    )


class IThorBedroomApproachBedsideWindow(ExactSceneNavigation, IThorBedroomEnv):
    SPEC_TASK = "approach_bedside_window"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK,
        required=(
            "fixture",
            "start_pose",
            "minimum_start_distance",
            "remove_fixtures",
            "lang",
            "robot_root_height",
        ),
    )

    def _calibrate_robot_free_joint_height(self):
        """Ground this task's G1 from its calibrated, above-floor root height."""
        joint = self._get_robot_free_joint()
        if joint is None:
            return False
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[2] = float(self.scene_task_spec()["robot_root_height"])
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.forward()
        qpos_address = self.sim.model.get_joint_qpos_addr(joint)[0]
        if EnvUtils._snap_freejoint_to_ground(self, qpos_address) is None:
            return False
        self._robot_free_joint_z = float(self.sim.data.get_joint_qpos(joint)[2])
        self._robot_default_free_joint_z = self._robot_free_joint_z
        return self.count_body_collisions() == 0


class IThorBedroomApproachDesk(ExactSceneNavigation, IThorBedroomEnv):
    SPEC_TASK = "approach_desk"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK,
        required=(
            "fixture",
            "start_pose",
            "minimum_start_distance",
            "remove_fixtures",
            "lang",
        ),
    )


class IThorBedroomNavigateToStorage(ExactSceneNavigation, IThorBedroomEnv):
    SPEC_TASK = "approach_storage"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK,
        required=(
            "fixture",
            "start_pose",
            "minimum_start_distance",
            "remove_fixtures",
            "lang",
        ),
    )


__all__ = [
    "IThorBedroomApproachBed",
    "IThorBedroomApproachBedsideWindow",
    "IThorBedroomApproachDesk",
    "IThorBedroomNavigateToStorage",
]
