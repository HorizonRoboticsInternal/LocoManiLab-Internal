"""Pick up an empty bottle beside a Holodeck trash can and throw it in."""

import os
import xml.etree.ElementTree as ET

import numpy as np

import robocasa
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec
from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion


class HolodeckThrowBottleIntoTrashCan(HolodeckEnv):
    """Pick the calibrated bottle source, then place it on the bin floor."""

    ATOMIC_SKILL = "PickPlaceIntoReceptacle"
    TASK_FAMILY = "composite"
    BOTTLE_ASSET = "objects/aigen_objs/bottled_drink/bottled_drink_0/model.xml"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "throw_bottle_into_trash_can"
    NOISE_FREE_ROBOT_SPAWN = True
    SOURCE_LABEL = "side table"
    SIDE_TABLE_SPEC_TASK = "throw_bottle_into_trash_can"
    HOLD_SECONDS = 1.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_baked_scene_fixtures(self):
        if self.SPEC_TASK == self.SIDE_TABLE_SPEC_TASK:
            spec = get_holodeck_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "source_body",
                    "source_table_pose",
                    "source_surface_z",
                    "source_table_half_size",
                ),
            )
            self._replace_bottle_source_with_side_table(spec)
        super()._setup_baked_scene_fixtures()

    def _replace_bottle_source_with_side_table(self, spec):
        """Add a grounded, task-local side table at the bottle source."""
        replaced_body = spec.get("replaced_body")
        if replaced_body is not None:
            removed = self.remove_scene_bodies((replaced_body,))
            if replaced_body not in removed:
                raise RuntimeError(
                    f"{self.scene_name}: configured bottle source body is missing: "
                    f"{replaced_body}"
                )

        source_body = spec["source_body"]
        if self.find_scene_body(source_body) is not None:
            raise RuntimeError(
                f"{self.scene_name}: task-local bottle source already exists: "
                f"{source_body}"
            )

        pose = spec["source_table_pose"]
        position = np.asarray(pose["pos"], dtype=float)
        if position.shape != (3,):
            raise ValueError(
                f"{self.scene_name}: source_table_pose.pos must have 3 values"
            )
        yaw = float(pose.get("yaw", 0.0))
        half_size = np.asarray(spec["source_table_half_size"], dtype=float)
        if half_size.shape != (2,) or np.any(half_size <= 0):
            raise ValueError(
                f"{self.scene_name}: source_table_half_size must be positive XY"
            )
        surface_z = float(spec["source_surface_z"])
        tabletop_half_z = 0.025
        if surface_z <= 2.0 * tabletop_half_z:
            raise ValueError(
                f"{self.scene_name}: source_surface_z is too low for side table"
            )

        worldbody = self.mujoco_arena.worldbody
        table = ET.SubElement(
            worldbody,
            "body",
            name=source_body,
            pos=" ".join(str(float(value)) for value in position),
            quat=" ".join(
                str(value)
                for value in (
                    np.cos(yaw / 2.0),
                    0.0,
                    0.0,
                    np.sin(yaw / 2.0),
                )
            ),
        )

        def add_box(name, pos, size, visual):
            attrs = {
                "name": name,
                "type": "box",
                "pos": " ".join(str(float(value)) for value in pos),
                "size": " ".join(str(float(value)) for value in size),
                "friction": "0.9 0.9 0.001",
            }
            if visual:
                attrs.update(
                    material=(
                        "material_WornWood"
                        if self.scene_name != "train_1018"
                        else "material_RoboTHOR_Desk_Gustav_Wood_Mat"
                    ),
                    contype="0",
                    conaffinity="0",
                    group="1",
                    mass="1e-8",
                )
            else:
                attrs.update(
                    contype="1",
                    conaffinity="15",
                    density="120",
                )
            ET.SubElement(table, "geom", **attrs)

        top_center_z = surface_z - tabletop_half_z
        add_box(
            f"{source_body}_tabletop_visual",
            [0.0, 0.0, top_center_z],
            [half_size[0], half_size[1], tabletop_half_z],
            visual=True,
        )
        add_box(
            f"{source_body}_tabletop_collision",
            [0.0, 0.0, top_center_z],
            [half_size[0], half_size[1], tabletop_half_z],
            visual=False,
        )

        leg_half_xy = min(0.035, float(np.min(half_size) * 0.18))
        leg_height = (surface_z - 2.0 * tabletop_half_z) / 2.0
        leg_x = max(float(half_size[0]) - 0.07, leg_half_xy)
        leg_y = max(float(half_size[1]) - 0.07, leg_half_xy)
        for index, (x, y) in enumerate(
            ((-leg_x, -leg_y), (-leg_x, leg_y), (leg_x, -leg_y), (leg_x, leg_y))
        ):
            add_box(
                f"{source_body}_leg_{index}_visual",
                [x, y, leg_height],
                [leg_half_xy, leg_half_xy, leg_height],
                visual=True,
            )
            add_box(
                f"{source_body}_leg_{index}_collision",
                [x, y, leg_height],
                [leg_half_xy, leg_half_xy, leg_height],
                visual=False,
            )

    def _build_semantic_scene(self, scene_info):
        registry = super()._build_semantic_scene(scene_info)
        if self.SPEC_TASK != self.SIDE_TABLE_SPEC_TASK:
            return registry

        spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "source_body",
                "source_table_pose",
                "source_surface_z",
                "source_table_half_size",
            ),
        )
        source_body = spec["source_body"]
        position = np.asarray(spec["source_table_pose"]["pos"], dtype=float)
        surface_z = float(spec["source_surface_z"])
        half_size = np.asarray(spec["source_table_half_size"], dtype=float)
        table_center = position.copy()
        table_center[2] = surface_z / 2.0
        table_fixture = SemanticFixture(
            stable_id=f"{self.scene_name}:task:{source_body}",
            name=source_body,
            category="sidetable",
            source_category="SideTable",
            room_id=None,
            body_name=source_body,
            body_names=[source_body],
            site_names=[],
            asset_id="task_side_table",
            object_id=source_body,
            center=table_center,
            half_size=np.array([half_size[0], half_size[1], surface_z / 2.0]),
            yaw=float(spec["source_table_pose"].get("yaw", 0.0)),
            front=np.array([0.0, -1.0]),
            capabilities={"support"},
            source={"task_local": True},
            local_center=np.array([0.0, 0.0, surface_z / 2.0]),
            local_half_size=np.array(
                [half_size[0], half_size[1], surface_z / 2.0]
            ),
        )
        table_fixture.regions[
            f"{table_fixture.stable_id}:region:support"
        ] = SemanticRegion(
            stable_id=f"{table_fixture.stable_id}:region:support",
            fixture_id=table_fixture.stable_id,
            kind="support",
            parent_body=source_body,
            local_center=np.array([0.0, 0.0, surface_z]),
            half_size=np.array([half_size[0], half_size[1], 0.02]),
            _world_center=position + np.array([0.0, 0.0, surface_z]),
            _world_yaw=table_fixture.yaw,
        )
        registry.add_fixture(table_fixture)
        return registry

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "trash_can_body",
                "source_body",
                "robot_pose",
                "bottle_pose",
            ),
        )
        self.trash_can_body = self.task_spec["trash_can_body"]
        self.source_body = self.task_spec["source_body"]
        self.source_fixture = self.get_fixture(id=self.source_body)
        self.trash_can_fixture = next(
            (
                fixture
                for fixture in self.fixtures.values()
                if fixture.body_name == self.trash_can_body
            ),
            None,
        )
        self.trash_can_xml_body = self.find_scene_body(self.trash_can_body)
        self.source_xml_body = self.find_scene_body(self.source_body)
        if self.trash_can_fixture is not None:
            self.fixture_refs["trash_can"] = self.trash_can_fixture
        self.fixture_refs[f"bottle_source_{self.SOURCE_LABEL}"] = self.source_body
        self._configure_trash_can_collision()
        pose = self.task_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        return bool(
            self.trash_can_xml_body is not None
            and self.source_xml_body is not None
        )

    def _configure_trash_can_collision(self):
        """Replace the baked solid proxy with a hollow, collidable receptacle.

        The Holodeck export contains only a solid cylinder for this bin.  A
        task-local hollow proxy preserves robot and bottle contact while
        allowing the bottle to pass through the opening.
        """
        body = self.find_scene_body(self.trash_can_body)
        if body is None:
            return
        solid = body.find(
            f"./geom[@name='{self.trash_can_body}_collision']"
        )
        if solid is not None:
            body.remove(solid)
        prefix = self.trash_can_body
        geoms = (
            # The scaled bottle's lowest collision vertices are about 0.072 m
            # above its root at the calibrated target pose.  Raise the floor
            # top to 0.08 m so a settled bottle makes a real contact.
            (f"{prefix}_inner_floor", "0 0.04 0", "0.19 0.04 0.19"),
            (f"{prefix}_inner_wall_x0", "-0.19 0.32 0", "0.03 0.32 0.19"),
            (f"{prefix}_inner_wall_x1", "0.19 0.32 0", "0.03 0.32 0.19"),
            (f"{prefix}_inner_wall_z0", "0 0.32 -0.19", "0.19 0.32 0.03"),
            (f"{prefix}_inner_wall_z1", "0 0.32 0.19", "0.19 0.32 0.03"),
        )
        for name, pos, size in geoms:
            if body.find(f"./geom[@name='{name}']") is None:
                ET.SubElement(
                    body,
                    "geom",
                    name=name,
                    type="box",
                    pos=pos,
                    size=size,
                    contype="1",
                    conaffinity="15",
                    density="250",
                )

    def _setup_scene(self):
        super()._setup_scene()
        if not self._calibrate_robot_free_joint_height():
            raise RuntimeError(f"{self.scene_name}: cannot calibrate robot free joint")
        if not self._robot_footprint_is_in_room(radius=0.25):
            raise RuntimeError(f"{self.scene_name}: bottle task robot starts off floor")
        if self.count_body_collisions() != 0:
            raise RuntimeError(
                f"{self.scene_name}: bottle task robot has spawn collision"
            )

    def _robot_footprint_is_in_room(self, radius=0.25):
        """Use room polygons because Holodeck has no Floor fixture record."""
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

    def _create_objects(self):
        super()._create_objects()
        bottle = MJCFObject(
            name="bottle",
            mjcf_path=os.path.join(robocasa.models.assets_root, self.BOTTLE_ASSET),
            scale=float(self.task_spec.get("bottle_scale", 1.0)),
        )
        self.objects[bottle.name] = bottle
        self.model.merge_objects([bottle])

    def _reset_internal(self):
        super()._reset_internal()
        bottle = self.objects["bottle"]
        qpos = self.sim.data.get_joint_qpos(bottle.joints[0]).copy()
        pose = self.task_spec["bottle_pose"]
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(bottle.joints[0], qpos)
        self.sim.data.set_joint_qvel(bottle.joints[0], np.zeros(6))
        self.sim.forward()
        self._initial_bottle_pos = np.asarray(
            self.sim.data.get_body_xpos(bottle.root_body), dtype=float
        ).copy()
        if not self._bottle_near_source():
            raise RuntimeError(
                f"{self.scene_name}: bottle reset is not at the source "
                f"{self.SOURCE_LABEL}"
            )
        self._success_hold = 0

    def _bottle_near_source(self):
        position = self.object_pos("bottle")
        initial = self._initial_bottle_pos
        return bool(
            np.linalg.norm(position[:2] - initial[:2]) <= 0.15
            and position[2] <= initial[2] + 0.10
        )

    def _bottle_contacts_trash_can_floor(self):
        """Return whether any bottle collision geom touches the bin floor."""
        bottle_body = self.sim.model.body_name2id(
            self.objects["bottle"].root_body
        )
        bottle_bodies = self._body_tree_ids([bottle_body])
        floor_name = f"{self.trash_can_body}_inner_floor"
        try:
            floor_geom = self.sim.model.geom_name2id(floor_name)
        except (KeyError, ValueError):
            return False

        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            if contact.geom1 == floor_geom:
                other_geom = contact.geom2
            elif contact.geom2 == floor_geom:
                other_geom = contact.geom1
            else:
                continue
            if int(self.sim.model.geom_bodyid[other_geom]) in bottle_bodies:
                return True
        return False

    def _bottle_in_bin(self):
        return self._bottle_contacts_trash_can_floor()

    def _post_action(self, action):
        _, done, info = super()._post_action(action)
        valid = self._bottle_in_bin()
        self._success_hold = self._success_hold + 1 if valid else 0
        success = bool(self._check_success())
        info["task_success"] = success
        info["success"] = success
        info["success_hold"] = self._success_hold
        return float(success), done, info

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    @property
    def hold_steps(self):
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = (
            "pick up the empty bottle from the "
            f"{self.SOURCE_LABEL} and throw it into the trash can"
        )
        meta["source_object"] = "bottle"
        meta["source_fixture"] = self.source_body
        meta["target_fixture"] = self.trash_can_body
        meta["success_definition"] = (
            "the bottle must remain in contact with the trash-can bottom support "
            "surface for one second"
        )
        return meta


__all__ = ["HolodeckThrowBottleIntoTrashCan"]
