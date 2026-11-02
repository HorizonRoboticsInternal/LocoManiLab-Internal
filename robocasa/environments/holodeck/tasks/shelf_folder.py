"""Move a folder from a ShelvingUnit and remove it from the shelf."""

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import robosuite.utils.transform_utils as T

import robocasa
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.environments.ithor_atomic.pick_place import resolve_task_support_region
from robocasa.models.arenas.molmospaces_arena import inline_defaults
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec
from robocasa.models.scenes.thor_semantics import SemanticRegion


class _HolodeckShelfFolderMixin:
    """Shared physical checks for a folder and the calibrated shelf region."""

    FOLDER_ASSET = "fixtures/articulated/green_folder001/model.xml"
    FOLDER_HALF_SIZE = np.array([0.16, 0.0355, 0.1305], dtype=float)
    FOLDER_SCALE_OVERRIDE = None
    # Override the source asset's green cover material in the task-local MJCF.
    # The metal clip keeps its original material.
    FOLDER_COVER_RGBA = "0.0862745 0.5372549 0.9176471 0.8"
    SHELF_BODY = "chestofdrawers_192fae93d3058f720017885d21c8146a_2_0_0"
    # This baked display island occupies the approach aisle to the shelf in
    # train_1018.  It is task-irrelevant and its structural collision mesh
    # overlaps the calibrated G1 spawn, so remove it only for this task family
    # by exact stable body ID.
    HIDDEN_APPROACH_BODIES = (
        "obja_61d33b4d0acc47ff8d9b87d400854f69_2_0_0",
    )
    HOLD_SECONDS = 1.0
    HOLD_STEPS = 5  # Fallback used by declarative composite stage metadata.
    RELEASE_DISTANCE = 0.18
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    NOISE_FREE_ROBOT_SPAWN = True
    # The exported shelf opening is slightly deeper than the old hand-tuned
    # region.  Keep the reset search within a small, deterministic calibration
    # envelope when the folder is rotated.
    FOLDER_RESET_OFFSETS = (
        (0.0, 0.0, 0.0),
        (0.0, 0.0, -0.02),
        (0.0, 0.0, 0.02),
        (0.0, 0.0, -0.04),
        (0.0, 0.0, -0.06),
        (0.0, 0.0, -0.08),
        (0.0, 0.0, 0.04),
        (-0.02, 0.0, 0.0),
        (0.02, 0.0, 0.0),
        (0.0, -0.02, 0.0),
        (0.0, 0.02, 0.0),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        required = [
            "shelf_body",
            "robot_pose",
            "folder_pose",
            "folder_scale",
            "region_center",
            "region_half_size",
        ]
        if self.MODE == "insert":
            required.extend(("source_table_body", "source_center", "source_half_size"))
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=required,
        )
        self._apply_folder_scale_override()
        self.shelf_body = self.task_spec["shelf_body"]
        self.shelf_fixture = self.get_fixture(id=self.shelf_body)
        self.source_table_fixture = None
        if self.MODE == "insert":
            self.source_table_fixture = self.get_fixture(
                id=self.task_spec["source_table_body"]
            )
        if self.shelf_fixture is not None:
            self.fixture_refs["shelf"] = self.shelf_fixture
            if self.source_table_fixture is not None:
                self.fixture_refs["source_desk"] = self.source_table_fixture
            pose = self.task_spec["robot_pose"]
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )

    def _apply_folder_scale_override(self):
        override = getattr(self, "FOLDER_SCALE_OVERRIDE", None)
        if override is None:
            return
        original = self.task_spec["folder_scale"]
        original = (
            np.full(3, float(original), dtype=float)
            if np.isscalar(original)
            else np.asarray(original, dtype=float)
        )
        override = np.asarray(override, dtype=float)
        if override.shape != (3,):
            raise ValueError("FOLDER_SCALE_OVERRIDE must be a 3-vector")
        self.task_spec["folder_scale"] = tuple(float(value) for value in override)

        # Preserve the original shelf-floor contact after shortening the
        # upright local X axis. Without this compensation, the folder starts
        # above the shelf and tips under gravity through its free joint.
        pose = self.task_spec["folder_pose"]
        pose["pos"] = list(pose["pos"])
        pose["pos"][2] -= self.FOLDER_HALF_SIZE[0] * (
            original[0] - override[0]
        )

    def _setup_baked_scene_fixtures(self):
        if self.scene_name != "train_1018":
            return super()._setup_baked_scene_fixtures()
        removed = self.remove_scene_bodies(self.HIDDEN_APPROACH_BODIES)
        missing = set(self.HIDDEN_APPROACH_BODIES) - removed
        if missing:
            raise RuntimeError(
                f"{self.scene_name}: folder task approach bodies are missing: "
                f"{sorted(missing)}"
            )
        super()._setup_baked_scene_fixtures()

    def _configured_source_desk_region(self):
        """Build the calibrated desktop region when the export has no sites."""
        center = np.asarray(self.task_spec["source_center"], dtype=float)
        half_size = np.asarray(self.task_spec["source_half_size"], dtype=float)
        return SemanticRegion(
            stable_id=f"{self.source_table_fixture.stable_id}:region:task_support",
            fixture_id=self.source_table_fixture.stable_id,
            kind="support",
            parent_body=None,
            local_center=center.copy(),
            half_size=half_size,
            _world_center=center.copy(),
            _world_yaw=0.0,
        )

    def _is_scene_valid(self):
        return bool(
            self.shelf_fixture is not None
            and self.find_scene_body(self.shelf_body) is not None
            and (
                self.MODE != "insert"
                or (
                    self.source_table_fixture is not None
                    and self.find_scene_body(self.task_spec["source_table_body"])
                    is not None
                )
            )
        )

    def _setup_scene(self):
        super()._setup_scene()
        # Keep the tall, thin folder from toppling under small grasp/contact
        # impulses.  Preserve its translational free-joint damping so the
        # placement motion remains physically responsive.
        qvel_addr = self.sim.model.get_joint_qvel_addr(self.objects["folder"].joints[0])
        if not isinstance(qvel_addr, tuple):
            raise RuntimeError("folder must have a six-DoF free joint")
        _, end = qvel_addr
        angular_start = end - 3
        self.sim.model.dof_damping[angular_start:end] = np.maximum(
            self.sim.model.dof_damping[angular_start:end], 0.05
        )
        self.sim.data.qvel[angular_start:end] = 0.0
        if self.MODE == "insert":
            support_regions = self.get_region(
                fixture=self.source_table_fixture, kind="support", return_all=True
            )
            if not support_regions:
                shallow_regions = self.get_region(
                    fixture=self.source_table_fixture,
                    kind="receptacle",
                    return_all=True,
                )
                support_regions = []
                for region in shallow_regions:
                    try:
                        support_regions.append(resolve_task_support_region(region))
                    except ValueError:
                        continue
            if not support_regions:
                self.source_desk_region = self._configured_source_desk_region()
            else:
                self.source_desk_region = max(
                    support_regions,
                    key=lambda region: (
                        float(np.prod(region.half_size[:2])),
                        region.stable_id,
                    ),
                )
            self.region_refs["source_desk"] = self.source_desk_region
        if not self._calibrate_folder_robot_spawn():
            raise RuntimeError(
                f"{self.scene_name}: cannot calibrate folder task robot height"
            )

    def _calibrate_folder_robot_spawn(self):
        """Find a collision-free spawn close to the calibrated task surface."""
        pose = self.task_spec["robot_pose"]
        base = np.asarray(pose["pos"], dtype=float)
        yaw = float(pose["yaw"])
        reference_region = (
            self.source_desk_region
            if self.MODE == "insert"
            else self.task_spec["region_center"]
        )
        reference_xy = np.asarray(
            reference_region.world_pose(self.sim)[0]
            if self.MODE == "insert"
            else reference_region,
            dtype=float,
        )[:2]
        away = base[:2] - reference_xy
        norm = float(np.linalg.norm(away))
        away = away / norm if norm > 1e-6 else np.array([-1.0, 0.0])
        lateral = np.array([-away[1], away[0]])

        # The source desk's collision mesh extends into the calibrated spawn.
        # Keep the search deterministic and bounded so the task remains near
        # its intended manipulation surface.
        candidates = [
            base[:2] + away * distance + lateral * side
            for distance in (0.0, 0.12, 0.24, 0.36, 0.48)
            for side in (0.0, 0.16, -0.16)
        ]
        for xy in candidates:
            if not self._set_robot_free_pose(xy, yaw):
                continue
            if not self._robot_footprint_is_in_room(radius=0.25):
                continue
            if (
                self._calibrate_robot_free_joint_height()
                and self.count_body_collisions() == 0
            ):
                self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
                self.robot_base_ori = np.array([0.0, 0.0, yaw])
                return True
        return False

    def _robot_footprint_is_in_room(self, radius=0.25):
        """Return whether the robot base and footprint samples are in a room."""
        base = self._robot_base_xy()
        samples = [base]
        samples.extend(
            base + float(radius) * np.array([np.cos(angle), np.sin(angle)])
            for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        )
        return all(
            self.room_of_point([point[0], point[1], 0.0]) is not None
            for point in samples
        )

    def _get_obj_cfgs(self):
        return []

    @classmethod
    def _make_folder_xml(cls):
        """Flatten the folder fixture and remove its optional hinge."""
        source = os.path.join(robocasa.models.assets_root, cls.FOLDER_ASSET)
        tree = ET.parse(source)
        root = tree.getroot()
        inline_defaults(root)
        cover_material = next(
            (
                material
                for material in root.iter("material")
                if material.get("name") == "green_mat"
            ),
            None,
        )
        if cover_material is None:
            raise ValueError(f"{source}: missing green_mat folder cover material")
        cover_material.set("rgba", cls.FOLDER_COVER_RGBA)
        # The checked-in fixture was exported on another machine. Resolve its
        # mesh paths relative to this asset so the task is portable.
        visual_dir = os.path.join(os.path.dirname(source), "visuals")
        for mesh in root.iter("mesh"):
            mesh_path = mesh.get("file")
            if mesh_path and not os.path.exists(mesh_path):
                candidate = os.path.join(visual_dir, os.path.basename(mesh_path))
                if os.path.exists(candidate):
                    mesh.set("file", candidate)
        for geom in root.iter("geom"):
            if geom.get("group") == "4":
                geom.set("group", "0")
            elif geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                geom.set("group", "1")
        worldbody = root.find("worldbody")
        body = worldbody.find("body")
        if body is None:
            raise ValueError(f"{source}: missing root folder body")
        worldbody.remove(body)
        nested_body = body.find("body[@name='object']")
        if nested_body is not None:
            body = nested_body
        for parent in body.iter():
            for child in list(parent):
                if child.tag == "joint":
                    parent.remove(child)
        body.set("name", "object")
        # Keep a stable local bbox after removing the optional folder hinge.
        ET.SubElement(
            body,
            "geom",
            name="reg_bbox",
            type="box",
            pos="0 0 0",
            size=" ".join(str(float(value)) for value in cls.FOLDER_HALF_SIZE),
            contype="0",
            conaffinity="0",
            group="1",
            rgba="0 0 0 0",
        )
        wrapper = ET.SubElement(worldbody, "body")
        wrapper.append(body)
        contact = root.find("contact")
        if contact is not None:
            root.remove(contact)
        actuator = root.find("actuator")
        if actuator is not None:
            root.remove(actuator)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".xml", dir=os.path.dirname(source), delete=False
        ) as f:
            tree.write(f, encoding="utf-8", xml_declaration=True)
            return f.name

    def _create_objects(self):
        super()._create_objects()
        xml_path = self._make_folder_xml()
        try:
            raw_scale = self.task_spec["folder_scale"]
            if np.isscalar(raw_scale):
                raw_scale = (float(raw_scale),) * 3
            if len(raw_scale) != 3:
                raise ValueError("folder_scale must be a scalar or a 3-vector")
            scale = np.asarray(raw_scale, dtype=float)
            folder = MJCFObject(
                name="folder",
                mjcf_path=xml_path,
                scale=tuple(float(value) for value in scale),
                density=180,
            )
            # MJCFObject's region parser expects prefixed ``reg_*`` names;
            # the nested folder asset can lose that prefix during subtree
            # extraction. Register the same scaled local bbox explicitly so
            # all standard object_bbox_points() callers remain valid.
            bbox_elem = ET.Element(
                "geom",
                {
                    "pos": "0 0 0",
                    "size": " ".join(
                        str(float(v)) for v in (scale * self.FOLDER_HALF_SIZE)
                    ),
                },
            )
            folder._regions["bbox"] = {"elem": bbox_elem}
        finally:
            if os.path.exists(xml_path):
                os.remove(xml_path)
        self.objects[folder.name] = folder
        self.model.merge_objects([folder])

    def _shelf_body_ids(self):
        return self._body_tree_ids([self.sim.model.body_name2id(self.shelf_body)])

    def _folder_contacts_shelf(self):
        folder_bodies = self._body_tree_ids([self.obj_body_id["folder"]])
        shelf_bodies = self._shelf_body_ids()
        return any(
            (
                int(self.sim.model.geom_bodyid[c.geom1]) in folder_bodies
                and int(self.sim.model.geom_bodyid[c.geom2]) in shelf_bodies
            )
            or (
                int(self.sim.model.geom_bodyid[c.geom2]) in folder_bodies
                and int(self.sim.model.geom_bodyid[c.geom1]) in shelf_bodies
            )
            for c in (self.sim.data.contact[i] for i in range(self.sim.data.ncon))
        )

    def _folder_has_deep_shelf_penetration(self, tolerance=0.002):
        """Reject reset poses that bury the folder in shelf structure."""
        folder_bodies = self._body_tree_ids([self.obj_body_id["folder"]])
        shelf_bodies = self._shelf_body_ids()
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            shelf_contact = (body1 in folder_bodies and body2 in shelf_bodies) or (
                body2 in folder_bodies and body1 in shelf_bodies
            )
            if shelf_contact and float(contact.dist) < -float(tolerance):
                return True
        return False

    def _folder_on_desk(self, require_released=True, require_bounds=True):
        if require_bounds:
            on_desk = self.object_on_region(
                "folder", self.source_desk_region, z_tol=0.06, require_contact=True
            )
        else:
            # Some workflows intentionally treat the complete desk footprint as
            # valid. In that mode, retain only physical contact with the desk;
            # do not reject edge placements using a semantic region or bbox.
            folder_bodies = self._body_tree_ids([self.obj_body_id["folder"]])
            fixture = self.get_fixture(id=self.source_desk_region.fixture_id)
            fixture_bodies = self._fixture_body_ids(fixture, include_siblings=True)
            on_desk = any(
                (
                    int(self.sim.model.geom_bodyid[contact.geom1]) in folder_bodies
                    and int(self.sim.model.geom_bodyid[contact.geom2])
                    in fixture_bodies
                )
                or (
                    int(self.sim.model.geom_bodyid[contact.geom2]) in folder_bodies
                    and int(self.sim.model.geom_bodyid[contact.geom1])
                    in fixture_bodies
                )
                for contact in (
                    self.sim.data.contact[i]
                    for i in range(self.sim.data.ncon)
                )
            )
        if not require_released:
            return bool(on_desk)
        return bool(
            on_desk and self.object_released("folder", th=self.RELEASE_DISTANCE)
        )

    def _folder_in_region(self):
        points = self._folder_bbox_points()
        center = np.asarray(self.task_spec["region_center"], dtype=float)
        half = np.asarray(self.task_spec["region_half_size"], dtype=float)
        return bool(
            np.all(points >= center - half) and np.all(points <= center + half)
        )

    def _folder_center_in_region(self):
        """Return whether the folder center is inside the shelf region."""
        folder_center = np.mean(self._folder_bbox_points(), axis=0)
        center = np.asarray(self.task_spec["region_center"], dtype=float)
        half = np.asarray(self.task_spec["region_half_size"], dtype=float)
        return bool(
            np.all(folder_center >= center - half)
            and np.all(folder_center <= center + half)
        )

    def _folder_outside_region(self):
        """Return whether the folder's full bounding box cleared the shelf region."""
        points = self._folder_bbox_points()
        center = np.asarray(self.task_spec["region_center"], dtype=float)
        half = np.asarray(self.task_spec["region_half_size"], dtype=float)
        lower = center - half
        upper = center + half
        eps = 1e-4
        return bool(
            np.any(np.max(points, axis=0) < lower - eps)
            or np.any(np.min(points, axis=0) > upper + eps)
        )

    def _folder_bbox_points(self):
        """Return folder bounds, with a fallback for assets lacking bbox regions."""
        try:
            return self.object_bbox_points("folder")
        except KeyError as exc:
            if str(exc) != "'bbox'":
                raise
        qpos = np.asarray(
            self.sim.data.get_joint_qpos(self.objects["folder"].joints[0]),
            dtype=float,
        )
        raw_scale = self.task_spec["folder_scale"]
        scale = (
            np.full(3, float(raw_scale), dtype=float)
            if np.isscalar(raw_scale)
            else np.asarray(raw_scale, dtype=float)
        )
        half = self.FOLDER_HALF_SIZE * scale
        rotation = T.quat2mat(T.convert_quat(qpos[3:7], to="xyzw"))
        signs = np.array(
            [
                [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
                [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
            ],
            dtype=float,
        )
        return qpos[:3] + (signs * half) @ rotation.T

    def _set_folder_pose(self):
        joint = self.objects["folder"].joints[0]
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        pose = self.task_spec["folder_pose"]
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.data.set_joint_qvel(joint, np.zeros(6))

    def _calibrate_folder_reset_pose(self):
        """Find a nearby pose that remains inside and contacts the shelf."""
        joint = self.objects["folder"].joints[0]
        base_qpos = self.sim.data.get_joint_qpos(joint).copy()
        base_pos = base_qpos[:3].copy()
        for dx, dy, dz in self.FOLDER_RESET_OFFSETS:
            qpos = base_qpos.copy()
            qpos[:3] = base_pos + np.array([dx, dy, dz], dtype=float)
            self.sim.data.set_joint_qpos(joint, qpos)
            self.sim.data.set_joint_qvel(joint, np.zeros(6))
            self.sim.forward()
            if (
                self._folder_in_region()
                and self._folder_contacts_shelf()
                and not self._folder_has_deep_shelf_penetration()
            ):
                return True
        self.sim.data.set_joint_qpos(joint, base_qpos)
        self.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.sim.forward()
        return False

    def _reset_internal(self):
        super()._reset_internal()
        # ``sim.reset()`` restores the XML free-joint height, so force the
        # shared height helper to re-evaluate it for every episode.
        if hasattr(self, "_robot_free_joint_z"):
            delattr(self, "_robot_free_joint_z")
        if not self._calibrate_robot_free_joint_height():
            raise RuntimeError(
                f"{self.scene_name}: cannot restore folder task robot height"
            )
        self._set_folder_pose()
        self.sim.forward()
        if self.MODE == "insert":
            # Insert must begin with the folder standing on the calibrated
            # source desk; otherwise the task starts in the transfer phase.
            valid_reset = self._folder_on_desk(require_released=False)
        else:
            valid_reset = self._calibrate_folder_reset_pose()
        if not valid_reset:
            raise RuntimeError(
                f"{self.scene_name}: folder reset is not supported by the calibrated shelf region"
            )
        self._grasped_at_start = False
        self._success_hold = 0

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self.MODE == "insert":
            success = (
                self._folder_center_in_region() and self._folder_contacts_shelf()
            )
        else:
            success = not self._folder_contacts_shelf()
        self._success_hold = self._success_hold + 1 if success else 0
        return ret

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    @property
    def hold_steps(self):
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        # Composite folder workflows load a task-specific calibration that
        # intentionally contains geometry only; their own metadata methods
        # provide the language instruction after this shared hook returns.
        if "lang" in self.task_spec:
            meta["lang"] = self.task_spec["lang"]
        meta["source_fixture"] = self.shelf_body
        meta["object_model"] = self.FOLDER_ASSET
        meta["object_scale"] = self.task_spec["folder_scale"]
        if self.MODE == "remove":
            meta["target_fixture"] = self.shelf_body
            meta["target_region_center"] = self.task_spec["region_center"]
            meta["success_definition"] = (
                "the folder must no longer contact the ShelvingUnit for one second"
            )
        else:
            meta["source_fixture"] = self.source_table_fixture.stable_id
            meta["source_region"] = self.source_desk_region.stable_id
            meta["target_fixture"] = self.shelf_body
            meta["target_region_center"] = self.task_spec["region_center"]
            meta["success_definition"] = (
                "the folder center must be inside the calibrated ShelvingUnit "
                "target region and contact the ShelvingUnit for one second"
            )
        return meta


class HolodeckInsertFolderIntoShelf(_HolodeckShelfFolderMixin, HolodeckEnv):
    """Insert a folder from the nearest desk into the middle shelf.

    The historical class name is retained for compatibility with existing
    launch commands and datasets.
    """

    ATOMIC_SKILL = "PickPlaceOnSurface"
    TASK_FAMILY = "pick_place"
    SUPPORTED_SCENE_IDS = (1018, 1197, 2025)
    SPEC_TASK = "insert_folder_into_shelf"
    MODE = "insert"

    def __init__(self, *args, scene_ids=None, **kwargs):
        super().__init__(
            *args,
            scene_ids=self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids,
            **kwargs,
        )


class HolodeckRemoveFolderFromShelf(_HolodeckShelfFolderMixin, HolodeckEnv):
    """Remove the folder from the middle open region of the ShelvingUnit."""

    ATOMIC_SKILL = "PickFromShelf"
    TASK_FAMILY = "pick"
    SUPPORTED_SCENE_IDS = (1018, 1197, 2025)
    SPEC_TASK = "remove_folder_from_shelf"
    MODE = "remove"

    def __init__(self, *args, scene_ids=None, **kwargs):
        super().__init__(
            *args,
            scene_ids=self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids,
            **kwargs,
        )


__all__ = ["HolodeckInsertFolderIntoShelf", "HolodeckRemoveFolderFromShelf"]
