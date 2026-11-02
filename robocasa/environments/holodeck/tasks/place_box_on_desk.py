"""Place the calibrated cardboard box onto the Holodeck desk."""

import os

import numpy as np

from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.environments.holodeck.tasks.pick_box_from_floor import (
    HolodeckPickBoxFromFloor,
)
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckPlaceBoxOnDesk(HolodeckEnv):
    """Move the floor cardboard box onto the calibrated desk surface."""

    ATOMIC_SKILL = "PickPlace"
    TASK_FAMILY = "pick_place"
    BOX_ASSET = HolodeckPickBoxFromFloor.BOX_ASSET
    BOX_SCALE = 1.5
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "place_box_on_desk"
    DESK_BODY = "desk_aeb6ab35aea76da67db0d3cd45d3e821_1_0_0"
    HOLD_SECONDS = 1.0
    # This task uses deterministic calibrated poses for both robot and box.
    INITIAL_POSE_JITTER = False
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    NOISE_FREE_ROBOT_SPAWN = True
    USE_DEFAULT_ASSET_LAYOUT = False
    CONTACT_MARGIN = 0.003

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_baked_scene_fixtures(self):
        spec = get_holodeck_task_spec(self.scene_name, self.SPEC_TASK)
        remove_scene_bodies = tuple(spec.get("remove_scene_bodies", ()))
        if remove_scene_bodies:
            removed = self.remove_scene_bodies(remove_scene_bodies)
            missing = set(remove_scene_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured box scene bodies are missing: "
                    f"{sorted(missing)}"
                )
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "robot_pose",
                "box_pose",
                "box_scale",
                "table_body",
            ),
        )
        self.desk_body = self.task_spec["table_body"]
        self.desk_fixture = self.get_fixture(id=self.desk_body)
        if self.desk_fixture is None:
            raise RuntimeError(f"{self.scene_name}: desk body is missing")
        self.fixture_refs["place_box_on_desk"] = self.desk_fixture
        pose = self.task_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        # The scene-local YAML selects the desk for every supported scene.
        # Keep the legacy DESK_BODY constant for the train_117 contract, but
        # do not reject valid scene-local desks during model construction.
        return getattr(self, "desk_fixture", None) is not None

    def _get_obj_cfgs(self):
        return []

    def _create_objects(self):
        super()._create_objects()
        box_xml = HolodeckPickBoxFromFloor._make_box_xml()
        try:
            box = MJCFObject(
                name="box",
                mjcf_path=box_xml,
                scale=float(self.task_spec["box_scale"]),
            )
        finally:
            for path in (box_xml,):
                if os.path.exists(path):
                    os.remove(path)
        self.objects[box.name] = box
        self.model.merge_objects([box])

    def _set_box_pose(self):
        box = self.objects["box"]
        qpos = self.sim.data.get_joint_qpos(box.joints[0]).copy()
        pose = self.task_spec["box_pose"]
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(box.joints[0], qpos)
        self.sim.data.set_joint_qvel(box.joints[0], np.zeros(6))
        self.sim.forward()
        points, _ = self._body_geom_points([box.root_body])
        if not len(points):
            raise RuntimeError(f"{self.scene_name}: box has no compiled geometry")
        # The box starts on the floor, ready for the pick-and-place motion.
        qpos[2] -= float(np.min(points[:, 2]))
        self.sim.data.set_joint_qpos(box.joints[0], qpos)
        self.sim.data.set_joint_qvel(box.joints[0], np.zeros(6))
        self.sim.forward()

    def _reset_internal(self):
        super()._reset_internal()
        self._set_box_pose()
        self._initial_box_pos = self.object_pos("box").copy()
        self._success_hold = 0
        self._desk_body_ids = set(
            self._body_tree_ids([self.sim.model.body_name2id(self.desk_body)])
        )
        if self._box_on_desk():
            raise RuntimeError(f"{self.scene_name}: box starts on desk")

    def _box_contacts_desk(self):
        box_body = self.sim.model.body_name2id(self.objects["box"].root_body)
        box_bodies = set(self._body_tree_ids([box_body]))
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body1 in box_bodies and body2 in self._desk_body_ids) or (
                body2 in box_bodies and body1 in self._desk_body_ids
            ):
                return True
        return False

    def _box_on_desk(self):
        """Return whether the box is in contact with the desk."""
        return bool(self._box_contacts_desk())

    def _post_action(self, action):
        _, done, info = super()._post_action(action)
        placed = self._box_on_desk()
        self._success_hold = self._success_hold + 1 if placed else 0
        success = self._check_success()
        info["task_success"] = bool(success)
        info["success"] = bool(success)
        info["success_hold"] = self._success_hold
        return float(success), done, info

    @property
    def hold_steps(self):
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = "pick up the cardboard box and place it on the desk"
        meta["source_object"] = "box"
        meta["target_object"] = self.desk_body
        meta["success_definition"] = (
            "the box must contact the desk continuously "
            f"for {self.HOLD_SECONDS:.1f} s"
        )
        return meta


__all__ = ["HolodeckPlaceBoxOnDesk"]
