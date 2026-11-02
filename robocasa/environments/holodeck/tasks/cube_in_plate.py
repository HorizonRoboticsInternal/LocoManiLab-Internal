"""Place the calibrated Rubik's cube into a plate in Holodeck scene 1477."""

import numpy as np

from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckCubeInPlate(HolodeckEnv):
    """Place the coffee-table Rubik's cube inside the adjacent plate."""

    ATOMIC_SKILL = "PlaceObject"
    TASK_FAMILY = "place"
    SUPPORTED_SCENE_IDS = (1477,)
    SPEC_TASK = "cube_in_plate"
    HOLD_SECONDS = 1.0
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
                f"{self.scene_name}: configured extra plate bodies are missing: "
                f"{sorted(missing)}"
            )
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "target_body",
                "target_pose",
                "plate_body",
                "plate_pose",
                "table_body",
                "robot_pose",
            ),
        )
        pose = self.task_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        return all(
            self.find_scene_body(self.task_spec[key]) is not None
            for key in ("target_body", "plate_body", "table_body")
        )

    def _get_obj_cfgs(self):
        return []

    def _body_id(self, spec_key):
        return self.sim.model.body_name2id(self.task_spec[spec_key])

    def _set_baked_pose(self, spec_key):
        body_id = self._body_id(spec_key.replace("_pose", "_body"))
        joint_id = int(self.sim.model.body_jntadr[body_id])
        if joint_id < 0 or int(self.sim.model.jnt_type[joint_id]) != 0:
            raise RuntimeError(
                f"{self.scene_name}: {spec_key} body must have a free root joint"
            )
        joint_name = self.sim.model.joint_id2name(joint_id)
        pose = self.task_spec[spec_key]
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(joint_name, qpos)
        self.sim.data.set_joint_qvel(joint_name, np.zeros(6))

    def _cube_contacts_body(self, spec_key):
        cube_bodies = self._body_tree_ids([self._body_id("target_body")])
        other_bodies = self._body_tree_ids([self._body_id(spec_key)])
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body1 in cube_bodies and body2 in other_bodies) or (
                body2 in cube_bodies and body1 in other_bodies
            ):
                return True
        return False

    def _cube_placed(self):
        return bool(
            self._cube_contacts_body("plate_body")
            and not self._cube_contacts_body("table_body")
        )

    def _reset_internal(self):
        super()._reset_internal()
        self._set_baked_pose("plate_pose")
        self._set_baked_pose("target_pose")
        self.sim.forward()
        self._success_hold = 0
        if self._cube_placed():
            raise RuntimeError(f"{self.scene_name}: cube starts on the plate")

    @property
    def hold_steps(self):
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold = self._success_hold + 1 if self._cube_placed() else 0
        return ret

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = "put the Rubik's cube into the plate"
        meta["success_definition"] = (
            "the cube must remain in contact with the target plate and out of "
            "contact with the coffee table for 1.0 s; no grasp or release required"
        )
        return meta


__all__ = ["HolodeckCubeInPlate"]
