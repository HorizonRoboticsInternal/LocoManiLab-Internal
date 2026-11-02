"""Place an enlarged sugar cube into an open Jar004 on a Holodeck desk."""

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

import robocasa
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.environments.holodeck.tasks._laptop_layout import (
    get_open_laptop_layout,
    replace_baked_laptop,
    spawn_calibrated_laptop,
)
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec
from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion

def make_empty_jar_xml():
    """Create a visible, empty Jar004 variant without altering the source asset."""
    source = os.path.join(robocasa.models.assets_root, HolodeckSugarCubeInJar.JAR_ASSET)
    tree = ET.parse(source)
    root = tree.getroot()

    # Remove only the opaque contents / filling, not the transparent jar shell.
    for body in list(root.iter("body")):
        for child in list(body):
            if child.tag == "body" and child.get("name") == "Jar004_Prop":
                body.remove(child)

    # Jar004 relies on its opaque filling for most of its silhouette.  Once that
    # body is removed, the original 0.2-alpha glass shell is effectively
    # invisible in the Holodeck renderer.  Keep it slightly tinted but opaque
    # enough to remain visible from the task camera.
    clear_mat = root.find(".//material[@name='Jar004_Clear_material']")
    if clear_mat is not None:
        clear_mat.set("rgba", "0.35 0.45 0.50 0.95")
        clear_mat.set("specular", "0.55")
        clear_mat.set("shininess", "0.65")

    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".xml",
        dir=os.path.dirname(source),  # important: keep relative mesh paths valid
        delete=False,
    ) as file:
        tree.write(file, encoding="utf-8", xml_declaration=True)
        return file.name

class HolodeckSugarCubeInJar(HolodeckEnv):
    """Pick up SugarCube002 and release it inside an open Jar004."""

    ATOMIC_SKILL = "PickPlaceInside"
    TASK_FAMILY = "pick_place"
    JAR_ASSET = "objects/lightwheel/jar/Jar004/model.xml"
    LID_ASSET = "objects/lightwheel/jar/Jar004/Jar004_Lid/model.xml"
    SUGAR_ASSET = "objects/lightwheel/sugar_cube/SugarCube002/model.xml"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "sugar_cube_in_jar"
    HOLD_SECONDS = 1.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    USE_DEFAULT_ASSET_LAYOUT = False
    # Keep the tabletop clear for the jar, lid, and sugar cube layout.
    LAPTOP_CLEARED_SCENES = ("train_1664", "train_2025")
    NOISE_FREE_ROBOT_SPAWN = True
    # SugarCube002 is a 14 mm cube at unit scale.  A 3x scale gives the G1
    # fingers a 42 mm grasp target while retaining clearance inside Jar004.
    SUGAR_SCALE = 3.0
    # The bbox and collision mesh are calibrated independently. Allow a small
    # boundary tolerance while still requiring every bbox corner to be inside.
    JAR_BBOX_TOLERANCE = 0.001

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
                    f"{self.scene_name}: configured sugar-jar scene bodies are missing: "
                    f"{sorted(missing)}"
                )
        self._add_missing_desk_support(spec)
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
                "jar_scale",
                "jar_pose",
                "lid_pose",
                "sugar_pose",
            ),
        )
        table_body = self.task_spec["table_body"]
        semantic_fixtures = getattr(self.semantic_scene, "fixtures", {}).values()
        self.table_fixture = next(
            (fixture for fixture in semantic_fixtures if fixture.body_name == table_body),
            None,
        )
        if self.table_fixture is None:
            self.table_fixture = next(
                (
                    fixture
                    for fixture in getattr(self, "fixtures", {}).values()
                    if getattr(fixture, "body_name", None) == table_body
                ),
                None,
            )
        if self.table_fixture is None:
            tables = self.get_fixture(capabilities=("support",), return_all=True) or ()
            self.table_fixture = next(
                (
                    fixture
                    for fixture in tables
                    if getattr(fixture, "body_name", None) == table_body
                ),
                None,
            )
        self.table_support_region = None
        if self.table_fixture is not None:
            regions = self.get_region(
                fixture=self.table_fixture,
                kind="support",
                return_all=True,
            )
            if regions:
                self.table_support_region = max(
                    regions,
                    key=lambda region: float(np.prod(region.half_size[:2])),
                )
            if self.table_support_region is None:
                self.table_support_region = self._calibrated_support_region()
            self.fixture_refs["table"] = self.table_fixture
            if self.table_support_region is not None:
                self.region_refs["table_support"] = self.table_support_region
            pose = self.task_spec["robot_pose"]
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )

    def _calibrated_support_region(self):
        support_z = self.task_spec.get("support_z")
        if support_z is None or not isinstance(self.table_fixture, SemanticFixture):
            return None
        layout_xy = np.asarray(
            [
                self.task_spec[name]["pos"][:2]
                for name in ("jar_pose", "lid_pose", "sugar_pose")
            ],
            dtype=float,
        )
        center_xy = (layout_xy.min(axis=0) + layout_xy.max(axis=0)) / 2.0
        half_xy = (layout_xy.max(axis=0) - layout_xy.min(axis=0)) / 2.0 + 0.05
        center = np.array([center_xy[0], center_xy[1], float(support_z)])
        region = SemanticRegion(
            stable_id=f"{self.table_fixture.stable_id}:region:task_support",
            fixture_id=self.table_fixture.stable_id,
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.array([half_xy[0], half_xy[1], 0.02]),
            _world_center=center,
            _world_yaw=0.0,
        )
        self.table_fixture.regions[region.stable_id] = region
        self.table_fixture.capabilities.add("support")
        return region

    def _add_missing_desk_support(self, spec):
        """Add task-local support under the calibrated objects on open desk meshes."""
        if self.scene_name not in ("train_1664", "train_2025"):
            return
        body = self.find_scene_body(spec["table_body"])
        if body is None:
            raise RuntimeError(
                f"{self.scene_name}: missing desk for sugar-jar support: "
                f"{spec['table_body']}"
            )
        support_top_z = float(spec["support_top_z"])
        body_pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        body_quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
        from robosuite.utils import transform_utils as T

        body_rot = T.quat2mat(T.convert_quat(body_quat, to="xyzw"))
        local_quat = T.convert_quat(T.mat2quat(body_rot.T), to="wxyz")
        supports = (
            ("jar", spec["jar_pose"], (0.12, 0.12)),
            ("lid", spec["lid_pose"], (0.12, 0.12)),
            ("sugar", spec["sugar_pose"], (0.035, 0.035)),
        )
        for name, pose, half_size in supports:
            object_pos = np.asarray(pose["pos"], dtype=float)
            support_center = np.array(
                [object_pos[0], object_pos[1], support_top_z - 0.005]
            )
            local_pos = body_rot.T @ (support_center - body_pos)
            geom_name = f"{self.scene_name}_sugar_{name}_support"
            if body.find(f"./geom[@name='{geom_name}']") is not None:
                continue
            ET.SubElement(
                body,
                "geom",
                name=geom_name,
                type="box",
                pos=" ".join(str(float(value)) for value in local_pos),
                quat=" ".join(str(float(value)) for value in local_quat),
                size=f"{half_size[0]} {half_size[1]} 0.005",
                contype="8",
                conaffinity="15",
                group="0",
                friction="0.9 0.9 0.001",
                solimp="0.95 0.98 0.001",
                solref="0.004 1",
            )

    def _is_scene_valid(self):
        return self.table_fixture is not None and self.table_support_region is not None

    def _get_obj_cfgs(self):
        return []

    def _create_objects(self):
        super()._create_objects()
        root = robocasa.models.assets_root
        jar_xml_path = make_empty_jar_xml()
        try:
            jar = MJCFObject(
                name="jar",
                mjcf_path=jar_xml_path,
                scale=float(self.task_spec["jar_scale"]),
            )
        finally:
            if os.path.exists(jar_xml_path):
                os.remove(jar_xml_path)
        lid = MJCFObject(
            name="jar_lid",
            mjcf_path=os.path.join(root, self.LID_ASSET),
            scale=float(self.task_spec["jar_scale"]),
        )
        sugar = MJCFObject(
            name="sugar_cube",
            mjcf_path=os.path.join(root, self.SUGAR_ASSET),
            scale=self.SUGAR_SCALE,
        )
        for obj in (jar, lid, sugar):
            self.objects[obj.name] = obj
        self.model.merge_objects([jar, lid, sugar])

    def _reset_internal(self):
        super()._reset_internal()
        self._place(self.objects["jar"], self.task_spec["jar_pose"])
        self._place(self.objects["jar_lid"], self.task_spec["lid_pose"])
        self._place(self.objects["sugar_cube"], self.task_spec["sugar_pose"])
        self.sim.forward()
        self._success_hold = 0
        if not self._initial_layout_valid():
            raise RuntimeError(f"{self.scene_name}: sugar-cube jar reset is invalid")

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

    def _initial_layout_valid(self):
        support_center, _ = self.table_support_region.world_pose(self.sim)
        return bool(
            all(
                abs(self.object_bottom_z(name) - float(support_center[2])) <= 0.02
                for name in ("jar", "jar_lid", "sugar_cube")
            )
            and all(
                self.table_support_region.contains_xy(
                    self.object_pos(name), self.sim
                )
                for name in ("jar", "jar_lid", "sugar_cube")
            )
            and np.linalg.norm(
                self.object_pos("jar")[:2] - self.object_pos("jar_lid")[:2]
            )
            > 0.10
            and not self._sugar_in_jar()
        )

    def _sugar_and_jar_local_bbox_points(self):
        """Return both object bboxes in the jar's local coordinate frame."""
        jar = self.objects["jar"]
        jar_pos = np.asarray(self.sim.data.get_body_xpos(jar.root_body), dtype=float)
        jar_mat = np.asarray(
            self.sim.data.get_body_xmat(jar.root_body), dtype=float
        ).reshape(3, 3)
        sugar_local = (self.object_bbox_points("sugar_cube") - jar_pos) @ jar_mat
        jar_local = (self.object_bbox_points("jar") - jar_pos) @ jar_mat
        return sugar_local, jar_local

    def _sugar_in_jar(self):
        sugar_local, jar_local = self._sugar_and_jar_local_bbox_points()
        jar_lower = np.min(jar_local, axis=0)
        jar_upper = np.max(jar_local, axis=0)
        jar_center = (jar_lower + jar_upper) / 2.0
        jar_half = (jar_upper - jar_lower) / 2.0
        local = sugar_local - jar_center
        horizontal_inside = (
            np.all(np.abs(local[:, :2]) <= jar_half[:2] + self.JAR_BBOX_TOLERANCE)
            and np.all(
                np.abs(local[:, 0]) + np.abs(local[:, 1])
                <= np.sqrt(2.0) * min(jar_half[0], jar_half[1])
                + self.JAR_BBOX_TOLERANCE
            )
        )
        return bool(
            horizontal_inside
            and np.all(local[:, 2] >= -jar_half[2] - self.JAR_BBOX_TOLERANCE)
            and np.all(local[:, 2] <= jar_half[2] + self.JAR_BBOX_TOLERANCE)
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        valid = self._sugar_in_jar()
        self._success_hold = self._success_hold + 1 if valid else 0
        return ret

    @property
    def hold_steps(self):
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="pick up the enlarged sugar cube and place it inside the open jar",
            source_object="sugar_cube",
            source_model=self.SUGAR_ASSET,
            source_scale=self.SUGAR_SCALE,
            target_object="jar",
            target_model=self.JAR_ASSET,
            target_lid="jar_lid",
            success_definition=(
                "the sugar cube must remain fully inside the jar's outer octagonal "
                "boundary for the hold interval"
            ),
        )
        return meta


__all__ = ["HolodeckSugarCubeInJar"]
