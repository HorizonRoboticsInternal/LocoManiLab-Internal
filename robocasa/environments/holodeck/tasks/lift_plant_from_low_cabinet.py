"""Lift the calibrated potted plant from Holodeck scene 1477."""

import numpy as np

from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckLiftPlantFromLowCabinet(HolodeckEnv):
    """Lift the potted plant on the round side table by 30 cm."""

    ATOMIC_SKILL = "PickObject"
    TASK_FAMILY = "pick"
    SUPPORTED_SCENE_IDS = (1477,)
    SPEC_TASK = "lift_plant_from_low_cabinet"
    HOLD_SECONDS = 1.0
    MIN_LIFT_DELTA = 0.30
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    USE_DEFAULT_ASSET_LAYOUT = False
    NOISE_FREE_ROBOT_SPAWN = True

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_baked_scene_fixtures(self):
        spec = get_holodeck_task_spec(
            self.scene_name, self.SPEC_TASK, required=("remove_scene_bodies",)
        )
        removed = self.remove_scene_bodies(spec["remove_scene_bodies"])
        missing = set(spec["remove_scene_bodies"]) - removed
        if missing:
            raise RuntimeError(
                f"{self.scene_name}: configured floor obstacle bodies are missing: "
                f"{sorted(missing)}"
            )
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=("source_body", "target_body", "target_pose", "robot_pose"),
        )
        pose = self.task_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        return all(
            self.find_scene_body(name) is not None
            for name in (
                self.task_spec["source_body"],
                self.task_spec["target_body"],
            )
        )

    def _get_obj_cfgs(self):
        return []

    def _target_body_id(self):
        return self.sim.model.body_name2id(self.task_spec["target_body"])

    def _target_joint_name(self):
        joint_id = int(self.sim.model.body_jntadr[self._target_body_id()])
        if joint_id < 0 or int(self.sim.model.jnt_type[joint_id]) != 0:
            raise RuntimeError(
                f"{self.scene_name}: plant target must have a free root joint"
            )
        return self.sim.model.joint_id2name(joint_id)

    def _set_target_pose(self):
        pose = self.task_spec["target_pose"]
        joint_name = self._target_joint_name()
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(joint_name, qpos)
        self.sim.data.set_joint_qvel(joint_name, np.zeros(6))

    def _plant_root_z(self):
        return float(self.sim.data.body_xpos[self._target_body_id()][2])

    def _reset_internal(self):
        super()._reset_internal()
        self._set_target_pose()
        self.sim.forward()
        self._plant_initial_z = self._plant_root_z()
        self._success_hold = 0

    @property
    def hold_steps(self):
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def _post_action(self, action):
        ret = super()._post_action(action)
        lifted = bool(
            self._plant_root_z() >= self._plant_initial_z + self.MIN_LIFT_DELTA
        )
        self._success_hold = self._success_hold + 1 if lifted else 0
        return ret

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = "pick up the potted plant from the side table."
        meta[
            "success_definition"
        ] = "the plant must remain at least 0.30 m above its reset height for 1.0 s"
        return meta


__all__ = ["HolodeckLiftPlantFromLowCabinet"]
