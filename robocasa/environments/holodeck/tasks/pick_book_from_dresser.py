"""Pick up a book placed on a calibrated office surface."""

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

import robocasa
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.arenas.molmospaces_arena import inline_defaults
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckPickBookFromDresser(HolodeckEnv):
    """Grasp the calibrated book and lift it."""

    ATOMIC_SKILL = "PickObject"
    TASK_FAMILY = "pick"
    BOOK_ASSET = "objects/thor/Bedroom Objects/Book/Prefabs/Book_3/Book_3.xml"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "pick_book_from_dresser"
    HOLD_SECONDS = 1.0
    HOLD_STEPS = 5  # Fallback used by declarative composite metadata.
    MIN_LIFT_DELTA = 0.30
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    NOISE_FREE_ROBOT_SPAWN = True
    # The baked desk laptop obstructs the book workflow in train_1018.
    # Remove only this exact body for this task family.
    CALIBRATED_LAPTOP_BODIES = {
        "train_1018": (
            "laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",
        ),
    }
    SOURCE_DECOR_BODIES = {
        "train_117": (
            "obja_733250257e9d44458ef4d0785e822ed0_1_0_0",
        ),
    }

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "source_body",
                "robot_pose",
                "book_pose",
                "book_scale",
            ),
        )
        self.source_fixture = self.get_fixture(id=self.task_spec["source_body"])
        print(
            f"[holodeck] pick_book scene={self.scene_name} "
            f"robot_pose={self.task_spec['robot_pose']} "
            f"source_body={self.task_spec['source_body']}",
            flush=True,
        )
        if self.source_fixture is not None:
            self.fixture_refs["book_support"] = self.source_fixture
            pose = self.task_spec["robot_pose"]
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )

    def _is_scene_valid(self):
        return self.source_fixture is not None

    def _get_obj_cfgs(self):
        return []

    def _setup_baked_scene_fixtures(self):
        """Apply task-local book and source-surface calibration before parsing."""
        spec = get_holodeck_task_spec(self.scene_name, self.SPEC_TASK)
        laptop_bodies = self.CALIBRATED_LAPTOP_BODIES.get(self.scene_name, ())
        if laptop_bodies:
            removed = self.remove_scene_bodies(laptop_bodies)
            missing = set(laptop_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured task laptop bodies are missing: "
                    f"{sorted(missing)}"
                )
        decor_bodies = tuple(self.SOURCE_DECOR_BODIES.get(self.scene_name, ()))
        decor_bodies += tuple(spec.get("remove_scene_bodies", ()))
        if decor_bodies:
            removed = self.remove_scene_bodies(decor_bodies)
            missing = set(decor_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured source decor is missing: "
                    f"{sorted(missing)}"
                )
        self._calibrate_baked_book(spec)
        self._scale_book_source_table(spec)
        super()._setup_baked_scene_fixtures()

    def _robot_footprint_is_in_room(self, radius=0.25):
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

    def _calibrate_baked_book(self, spec):
        target_body_name = spec.get("target_body")
        if not target_body_name:
            return
        target_body = self.find_scene_body(target_body_name)
        if target_body is None:
            raise RuntimeError(
                f"{self.scene_name}: configured book body is missing: "
                f"{target_body_name}"
            )

        scale = np.asarray(spec.get("book_scale", 1.0), dtype=float).reshape(-1)
        if scale.size == 1:
            scale = np.repeat(scale, 3)
        if scale.shape != (3,) or np.any(scale <= 0):
            raise ValueError(
                f"{self.scene_name}: book_scale must be positive XYZ scale"
            )
        density = spec.get("book_density")
        asset = self.mujoco_arena.root.find("asset")
        mesh_names = {
            geom.get("mesh") for geom in target_body.iter("geom") if geom.get("mesh")
        }
        for mesh_name in mesh_names:
            mesh = (
                asset.find(f"./mesh[@name='{mesh_name}']")
                if asset is not None
                else None
            )
            if mesh is None:
                raise RuntimeError(
                    f"{self.scene_name}: book mesh is missing: {mesh_name}"
                )
            mesh_scale = np.asarray(
                [float(value) for value in mesh.get("scale", "1 1 1").split()],
                dtype=float,
            )
            mesh.set("scale", " ".join(str(value) for value in mesh_scale * scale))

        if density is not None:
            for geom in target_body.iter("geom"):
                # MolmoSpacesArena inlines and removes default classes before
                # task-local scene edits run. The cleaned collision geoms use
                # group 0, while the baked visual geoms use group 1.
                if geom.get("group") == "0":
                    geom.set("density", str(float(density)))

    def _scale_book_source_table(self, spec):
        """Uniformly enlarge the source table around its ground contact."""
        scale = spec.get("source_table_scale")
        if scale is None:
            return
        scale = float(scale)
        if scale <= 0:
            raise ValueError(
                f"{self.scene_name}: source_table_scale must be positive"
            )

        source_body = self.find_scene_body(spec["source_body"])
        if source_body is None:
            raise RuntimeError(
                f"{self.scene_name}: configured book source is missing: "
                f"{spec['source_body']}"
            )
        source_child = source_body.find("./body")
        if source_child is None:
            raise RuntimeError(
                f"{self.scene_name}: book source has no table body to scale"
            )

        def scale_attr(element, name):
            values = np.asarray(
                [float(value) for value in element.get(name, "").split()],
                dtype=float,
            )
            if not len(values):
                return
            element.set(name, " ".join(str(value) for value in values * scale))

        # The child body's origin is the table's foot plane. Keep its position
        # fixed so the scaled mesh and collision boxes remain grounded.
        for nested_body in source_child.iter("body"):
            if nested_body is not source_child:
                scale_attr(nested_body, "pos")
        for element in source_child.iter():
            if element.tag in ("geom", "site"):
                scale_attr(element, "pos")
                scale_attr(element, "size")

        visual_geom = next(
            (
                geom
                for geom in source_child.iter("geom")
                if geom.get("mesh") and geom.get("group") == "1"
            ),
            None,
        )
        asset = self.mujoco_arena.root.find("asset")
        if visual_geom is None or asset is None:
            raise RuntimeError(
                f"{self.scene_name}: source table has no visual mesh asset"
            )
        mesh_name = visual_geom.get("mesh")
        mesh = asset.find(f"./mesh[@name='{mesh_name}']")
        if mesh is None:
            raise RuntimeError(
                f"{self.scene_name}: source table mesh is missing: {mesh_name}"
            )
        scaled_mesh_name = f"{mesh_name}_holodeck_source_scaled"
        scaled_mesh = asset.find(f"./mesh[@name='{scaled_mesh_name}']")
        if scaled_mesh is None:
            scaled_mesh = ET.SubElement(asset, "mesh", dict(mesh.attrib))
            scaled_mesh.set("name", scaled_mesh_name)
        mesh_scale = np.asarray(
            [float(value) for value in scaled_mesh.get("scale", "1 1 1").split()],
            dtype=float,
        )
        scaled_mesh.set(
            "scale", " ".join(str(value) for value in mesh_scale * scale)
        )
        visual_geom.set("mesh", scaled_mesh_name)

    def _uses_baked_target(self):
        return "target_body" in self.task_spec

    @classmethod
    def _make_robocasa_book_xml(cls):
        """Convert the nested-default THOR book into RoboCasa object layout."""
        source_path = os.path.join(robocasa.models.assets_root, cls.BOOK_ASSET)
        tree = ET.parse(source_path)
        root = tree.getroot()
        inline_defaults(root)

        # RoboCasa uses group 1 for visuals and group 0 for collision. THOR's
        # nested defaults encode the same distinction as non-colliding vs.
        # structural geoms, but use groups 0 / 4 instead.
        for geom in root.iter("geom"):
            if geom.get("group") == "4":
                geom.set("group", "0")
            elif geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                geom.set("group", "1")

        worldbody = root.find("worldbody")
        book_body = worldbody.find("body")
        if book_body is None:
            raise ValueError(f"{source_path}: missing root book body")
        worldbody.remove(book_body)
        book_body.set("name", "object")

        # MJCFObject supplies one free joint for the entire book. The THOR
        # export's free / cover joints would create a second movable chain and
        # make the grasp target physically unstable.
        for parent in book_body.iter():
            for child in list(parent):
                if child.tag == "joint":
                    parent.remove(child)

        wrapper = ET.SubElement(worldbody, "body")
        wrapper.append(book_body)
        contact = root.find("contact")
        if contact is not None:
            root.remove(contact)

        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".xml", dir=os.path.dirname(source_path), delete=False
        ) as file:
            tree.write(file, encoding="utf-8", xml_declaration=True)
            return file.name

    def _create_objects(self):
        super()._create_objects()
        if self._uses_baked_target():
            return
        book_xml = self._make_robocasa_book_xml()
        try:
            book = MJCFObject(
                name="book",
                mjcf_path=book_xml,
                scale=self.task_spec["book_scale"],
                density=float(self.task_spec.get("book_density", 100.0)),
            )
        finally:
            if os.path.exists(book_xml):
                os.remove(book_xml)
        self.objects[book.name] = book
        self.model.merge_objects([book])

    def _target_body_id(self):
        body_name = self.task_spec.get("target_body")
        if body_name is None:
            body_name = self.objects["book"].root_body
        return self.sim.model.body_name2id(body_name)

    def _target_grasped(self):
        target_geoms = self._geom_names_for_bodies(
            self._body_tree_ids([self._target_body_id()])
        )
        for side in ("left", "right"):
            if self.g1_hand_closed(side) and self.check_contact(
                self._geom_names_for_bodies(self._body_ids_matching((f"{side}_hand",))),
                target_geoms,
            ):
                return True
        return False

    def _book_grasped(self):
        """Return whether the task book is grasped for either asset path."""
        if self._uses_baked_target():
            return self._target_grasped()
        return self.g1_hand_grasping_object("book")

    def _book_bbox_points(self):
        """Return world-space book bounds for dynamic or baked books."""
        if not self._uses_baked_target():
            try:
                return self.object_bbox_points("book")
            except KeyError as exc:
                # The THOR Book_3 asset has collision geometry but no
                # ``reg_bbox`` region, which is required by MJCFObject's
                # generic bbox helper. Use the compiled geometry bounds for
                # this asset, matching the baked-book path below.
                if str(exc) != "'bbox'":
                    raise
                points, _ = self._body_geom_points(
                    [self.objects["book"].root_body]
                )
                if not len(points):
                    raise RuntimeError("dynamic book has no geometry")
                return points
        points, _ = self._body_geom_points([self.task_spec["target_body"]])
        if not len(points):
            raise RuntimeError(f"{self.scene_name}: baked book has no geometry")
        return points

    def _book_on_region(
        self, region, z_tol=0.08, require_contact=True, require_bounds=True
    ):
        """Check placement for either the dynamic or baked book asset.

        ``require_bounds=False`` keeps only the fixture-contact predicate for
        workflows whose placement target intentionally allows partial overlap.
        """
        center, yaw = region.world_pose(self.sim)
        points = self._book_bbox_points()
        c, s = np.cos(-yaw), np.sin(-yaw)
        delta = points[:, :2] - center[:2]
        local = np.column_stack(
            (c * delta[:, 0] - s * delta[:, 1], s * delta[:, 0] + c * delta[:, 1])
        )
        if require_bounds and not (
            np.all(np.abs(local) <= region.half_size[:2] + 1e-4)
            and abs(float(points[:, 2].min()) - float(center[2])) <= z_tol
        ):
            return False
        if not require_contact:
            return True

        book_bodies = self._body_tree_ids([self._target_body_id()])
        fixture = self.get_fixture(id=region.fixture_id)
        fixture_bodies = self._fixture_body_ids(fixture, include_siblings=True)
        return any(
            (
                int(self.sim.model.geom_bodyid[contact.geom1]) in book_bodies
                and int(self.sim.model.geom_bodyid[contact.geom2]) in fixture_bodies
            )
            or (
                int(self.sim.model.geom_bodyid[contact.geom2]) in book_bodies
                and int(self.sim.model.geom_bodyid[contact.geom1]) in fixture_bodies
            )
            for contact in (
                self.sim.data.contact[i] for i in range(self.sim.data.ncon)
            )
        )

    def _book_released(self, th=0.25):
        """Check release distance for the task book on either asset path."""
        if not self._uses_baked_target():
            return self.object_released("book", th=th)
        pos = np.asarray(self.sim.data.body_xpos[self._target_body_id()])
        return all(np.linalg.norm(eef - pos) > th for eef in self.eef_positions())

    def _reset_internal(self):
        super()._reset_internal()
        if not self._robot_footprint_is_in_room(radius=0.25):
            raise RuntimeError(
                f"{self.scene_name}: book task robot starts off the room floor"
            )
        robot_root = self.robots[0].robot_model.root_body
        robot_pos = self.sim.data.get_body_xpos(robot_root)
        print(
            f"[holodeck] pick_book reset robot_root={robot_root} "
            f"robot_world_pos={np.asarray(robot_pos).round(3).tolist()} "
            f"book_pose={self.task_spec['book_pose']['pos']}",
            flush=True,
        )
        if self._uses_baked_target():
            body_id = self._target_body_id()
            joint_id = int(self.sim.model.body_jntadr[body_id])
            joint_name = self.sim.model.joint_id2name(joint_id)
        else:
            joint_name = self.objects["book"].joints[0]
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        pose = self.task_spec["book_pose"]
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(joint_name, qpos)
        self.sim.data.set_joint_qvel(joint_name, np.zeros(6))
        self.sim.forward()
        if self.count_body_collisions() != 0:
            raise RuntimeError(
                f"{self.scene_name}: book task reset has robot collision"
            )
        self._book_initial_z = self._book_root_z()
        self._success_hold = 0

    def _book_root_z(self):
        return float(self.sim.data.body_xpos[self._target_body_id()][2])

    def _post_action(self, action):
        ret = super()._post_action(action)
        book_z = self._book_root_z()
        lifted = book_z > self._book_initial_z + self.lift_delta
        self._success_hold = self._success_hold + 1 if lifted else 0
        return ret

    @property
    def lift_delta(self):
        return float(self.MIN_LIFT_DELTA)

    @property
    def hold_steps(self):
        control_freq = getattr(self, "control_freq", None)
        if control_freq is None:
            return max(1, int(self.HOLD_STEPS))
        return max(1, int(round(self.HOLD_SECONDS * control_freq)))

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        if self.scene_name == "train_117":
            meta["lang"] = (
                "pick up the closed book from the side table by the window and lift it"
            )
        elif self.scene_name in ("train_1197", "train_1664", "train_2025"):
            meta["lang"] = "pick up the book from the dresser and lift it"
        else:
            meta["lang"] = "pick up the book in front of you and lift it"
        meta["source_fixture"] = self.source_fixture.name
        meta["success_definition"] = (
            f"the book root rises more than {self.lift_delta:.2f} m above its "
            f"reset height for {self.HOLD_SECONDS:.1f} s"
        )
        return meta


__all__ = ["HolodeckPickBookFromDresser"]
