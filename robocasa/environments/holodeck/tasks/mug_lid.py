"""Open or close a jar lid on a calibrated Holodeck office tabletop."""

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

import robocasa
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.environments.holodeck.tasks._laptop_layout import (
    LAPTOP_CLEARED_FOR_JAR_SCENES,
    get_open_laptop_layout,
    replace_baked_laptop,
    spawn_calibrated_laptop,
)
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


def make_empty_jar_xml(source):
    tree = ET.parse(source)
    root = tree.getroot()
    for body in list(root.iter("body")):
        for child in list(body):
            if child.tag == "body" and child.get("name") == "Jar004_Prop":
                body.remove(child)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".xml", dir=os.path.dirname(source), delete=False
    ) as file:
        tree.write(file, encoding="utf-8", xml_declaration=True)
        return file.name

class _HolodeckMugLidMixin:
    """Shared scene setup for the two directions of the jar-lid task."""

    MUG_ASSET = "objects/lightwheel/jar/Jar004/model.xml"
    LID_ASSET = "objects/lightwheel/jar/Jar004/Jar004_Lid/model.xml"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    HOLD_SECONDS = 1.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    USE_DEFAULT_ASSET_LAYOUT = False
    LAPTOP_CLEARED_SCENES = LAPTOP_CLEARED_FOR_JAR_SCENES
    NOISE_FREE_ROBOT_SPAWN = True
    LID_DAMPING = None
    USE_EMPTY_JAR = False

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_baked_scene_fixtures(self):
        self._laptop_layout = get_open_laptop_layout(self.scene_name)
        replace_baked_laptop(self, self._laptop_layout)
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        self.laptop_fixture = spawn_calibrated_laptop(self, self._laptop_layout)
        super()._setup_task_fixtures()
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "table_body",
                "robot_pose",
                "object_scale",
                "mug_pose",
                *self.SPEC_REQUIRED,
            ),
        )
        # The train_1664 desk is an unclassified Objaverse body. It has a
        # calibrated physical tabletop but is absent from the semantic
        # registry, so keep an exact XML-body fallback for this task.
        self.table_body = self.task_spec["table_body"]
        tables = self.get_fixture(return_all=True)
        self.table_fixture = next(
            (
                fixture
                for fixture in tables
                if fixture.body_name == self.table_body
            ),
            None,
        )
        self._has_calibrated_table_body = (
            self.find_scene_body(self.table_body) is not None
        )
        if self.table_fixture is not None:
            self.fixture_refs["table"] = self.table_fixture
        if self._has_calibrated_table_body:
            pose = self.task_spec["robot_pose"]
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )
        else:
            self._scene_invalid_reason = (
                f"missing calibrated table body {self.table_body!r}"
            )

    def _is_scene_valid(self):
        return self._has_calibrated_table_body

    def _setup_scene(self):
        super()._setup_scene()
        if self.LID_DAMPING is None:
            return
        qvel_addr = self.sim.model.get_joint_qvel_addr(
            self.objects["mug_lid"].joints[0]
        )
        if isinstance(qvel_addr, tuple):
            start, end = qvel_addr
        else:
            start, end = qvel_addr, qvel_addr + 1
        self.sim.model.dof_damping[start:end] = np.maximum(
            self.sim.model.dof_damping[start:end], float(self.LID_DAMPING)
        )
        self.sim.data.qvel[start:end] = 0.0

    def _get_obj_cfgs(self):
        return []

    def _create_objects(self):
        super()._create_objects()
        root = robocasa.models.assets_root
        scale = float(self.task_spec["object_scale"])
        mug_path = os.path.join(root, self.MUG_ASSET)
        temporary_mug_path = None
        if self.USE_EMPTY_JAR:
            temporary_mug_path = make_empty_jar_xml(mug_path)
            mug_path = temporary_mug_path

        try:
            mug = MJCFObject(
                name="mug",
                mjcf_path=mug_path,
                scale=scale,
            )
        finally:
            if temporary_mug_path and os.path.exists(temporary_mug_path):
                os.remove(temporary_mug_path)
        lid = MJCFObject(
            name="mug_lid",
            mjcf_path=os.path.join(root, self.LID_ASSET),
            scale=scale,
        )
        self.objects[mug.name] = mug
        self.objects[lid.name] = lid
        self.model.merge_objects([mug, lid])

    def _reset_internal(self):
        super()._reset_internal()
        self._place(self.objects["mug"], self.task_spec["mug_pose"])
        self._place(
            self.objects["mug_lid"],
            self.task_spec[self.LID_START_POSE_KEY],
        )
        self.sim.forward()
        self._reset_task_progress()

    def _place(self, obj, pose):
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        qvel_addr = self.sim.model.get_joint_qvel_addr(obj.joints[0])
        if isinstance(qvel_addr, tuple):
            self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = 0.0
        else:
            self.sim.data.qvel[qvel_addr] = 0.0

    def _reset_task_progress(self):
        parent = getattr(super(), "_reset_task_progress", None)
        if callable(parent):
            parent()
        self._success_hold = 0
        if self.SUCCESS_MODE == "close":
            mug_pos = self.sim.data.get_body_xpos(self.objects["mug"].root_body)
            closed_pos = np.asarray(
                self.task_spec["closed_lid_pose"]["pos"], dtype=float
            )
            self._closed_lid_offset = closed_pos - mug_pos

    def _post_action(self, action):
        ret = super()._post_action(action)
        completed = (
            self._lid_is_open()
            if self.SUCCESS_MODE == "open"
            else self._lid_is_closed()
        )
        self._success_hold = self._success_hold + 1 if completed else 0
        return ret

    def _lid_offset(self):
        mug_pos = self.sim.data.get_body_xpos(self.objects["mug"].root_body)
        lid_pos = self.sim.data.get_body_xpos(self.objects["mug_lid"].root_body)
        return np.asarray(lid_pos - mug_pos, dtype=float)

    def _lid_is_open(self):
        return not self._objects_in_contact("mug_lid", "mug")

    def _lid_is_closed(self):
        return bool(
            np.linalg.norm(self._lid_offset() - self._closed_lid_offset)
            <= float(self.task_spec["close_distance"])
        )

    def _objects_in_contact(self, first_name, second_name):
        first_bodies = self._body_tree_ids([self.obj_body_id[first_name]])
        second_bodies = self._body_tree_ids([self.obj_body_id[second_name]])
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            bodies = {
                int(self.sim.model.geom_bodyid[contact.geom1]),
                int(self.sim.model.geom_bodyid[contact.geom2]),
            }
            if (bodies & first_bodies) and (bodies & second_bodies):
                return True
        return False

    @property
    def hold_steps(self):
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())


class HolodeckOpenMugLid(_HolodeckMugLidMixin, HolodeckEnv):
    """Lift the removable lid away from a tabletop jar."""

    ATOMIC_SKILL = "RemoveLid"
    TASK_FAMILY = "pick"
    SPEC_TASK = "open_mug_lid"
    SPEC_REQUIRED = ("lid_pose",)
    LID_START_POSE_KEY = "lid_pose"
    SUCCESS_MODE = "open"
    USE_EMPTY_JAR = True

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = "remove the lid from the jar"
        meta["success_definition"] = (
            "the lid must remain out of contact with the jar for the hold interval"
        )
        return meta


class HolodeckCloseMugLid(_HolodeckMugLidMixin, HolodeckEnv):
    """Pick up a nearby jar lid and place it back on the jar."""

    ATOMIC_SKILL = "CloseLid"
    TASK_FAMILY = "pick"
    SPEC_TASK = "close_mug_lid"
    SPEC_REQUIRED = ("open_lid_pose", "closed_lid_pose", "close_distance")
    LID_START_POSE_KEY = "open_lid_pose"
    SUCCESS_MODE = "close"
    LID_DAMPING = 0.05
    # The 2025 desk has insufficient clear tabletop area for the lid to be
    # placed beside the jar while retaining the laptop. Keep the laptop
    # removal scoped to this task; the other jar tasks retain their layout.
    LAPTOP_CLEARED_SCENES = ("train_1664", "train_2025")

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = "put the lid back on the jar"
        meta["success_definition"] = (
            "the lid must remain within the calibrated closed-pose tolerance "
            "for the hold interval"
        )
        return meta


__all__ = ["HolodeckOpenMugLid", "HolodeckCloseMugLid"]
