"""Bedroom drawer articulation tasks."""

from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_specs,
    get_supported_scene_ids,
)

from .articulation_base import (
    BedroomArticulationMixin,
    BedroomTwentyEightyArticulationMixin,
)


class IThorBedroomOpenDrawer(
    BedroomTwentyEightyArticulationMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    ATOMIC_SKILL = "OpenDrawer"
    SPEC_TASK = "drawer"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK, required=("fixture", "joint", "target", "lang")
    )
    BEHAVIOR = "open"

    # Captured from ExactSceneArticulation's search with G1FloatingBody,
    # scene seed 0, and the drawer initially 20% open (before XY jitter).
    _SCENE_329_ROBOT_POS = (-1.2044756044721552, -1.74338068629375, 0.0)
    _SCENE_329_ROBOT_YAW = -1.5707963267948966

    def _finalize_articulation_robot_pose(self, distance):
        # Restore the default if a later reset selects another scene.
        self.INITIAL_ROBOT_JITTER_RADIUS = super().INITIAL_ROBOT_JITTER_RADIUS
        if self.scene_name != "FloorPlan329":
            return super()._finalize_articulation_robot_pose(distance)
        self.INITIAL_ROBOT_JITTER_RADIUS = 0.0
        if not (
            self._set_robot_free_pose(
                self._SCENE_329_ROBOT_POS[:2], self._SCENE_329_ROBOT_YAW
            )
            and self._calibrate_robot_free_joint_height()
        ):
            return False
        self.set_robot_base_pose(self._SCENE_329_ROBOT_POS, self._SCENE_329_ROBOT_YAW)
        return self.count_body_collisions() == 0

    def _constrain_initial_jitter_candidate(self):
        super()._constrain_initial_jitter_candidate()
        if self.scene_name == "FloorPlan329":
            # Fixed-task collection applies its own XY jitter after reset.
            # Preserve the settled height and orientation when constraining it.
            joint = self._get_robot_free_joint()
            qpos = self.sim.data.get_joint_qpos(joint).copy()
            qpos[:2] = self._SCENE_329_ROBOT_POS[:2]
            self.sim.data.set_joint_qpos(joint, qpos)
            self.sim.forward()


class IThorBedroomCloseDrawer(
    BedroomTwentyEightyArticulationMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    ATOMIC_SKILL = "CloseDrawer"
    SPEC_TASK = "drawer"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK, required=("fixture", "joint", "target", "lang")
    )
    BEHAVIOR = "close"


__all__ = ["IThorBedroomOpenDrawer", "IThorBedroomCloseDrawer"]
