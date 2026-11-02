"""Ordered low-microwave door and apple manipulation tasks."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_atomic.pick_place import SupportContactPlacementMixin
from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
    get_ithor_kitchen_task_specs,
)
from robocasa.models.scenes.thor_semantics import SemanticRegion


class _LowMicrowaveMixin(
    IThorMultiStageCompositeMixin,
    RoomOpenArticulation,
):
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_kitchen_task_spec)
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    COMPOSITE_ORDERING = "sequential"
    INITIAL_POSE_JITTER = False
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Microwave",)
    TARGET_JOINT_TYPES = ("hinge",)
    TARGET_LANG = "microwave door"
    HOLD_SECONDS = 1.0
    HANDLE_SCALE = 1.0
    ATOMIC_TASKS = (
        IThorAtomicTask("open_microwave", "_raw_success", hold_steps=5),
        IThorAtomicTask("close_microwave", "_door_closed"),
    )

    def _setup_baked_scene_fixtures(self):
        super()._setup_baked_scene_fixtures()
        spec = self._get_task_spec()
        worldbody = self.mujoco_arena.worldbody
        microwave = worldbody.find(f"./body[@name='{spec['microwave_body']}']")
        if microwave is None:
            raise ValueError(f"{self.scene_name}: missing calibrated microwave")
        # Fix the native housing in this task's model; retain its hinged door.
        for joint in list(microwave):
            if joint.tag == "freejoint" or (
                joint.tag == "joint" and joint.get("type") == "free"
            ):
                microwave.remove(joint)
        microwave.set("pos", " ".join(map(str, spec["microwave_pos"])))
        microwave.set("quat", " ".join(map(str, spec["microwave_quat"])))

        self._add_door_handle(microwave, spec)

        x, y, height = spec["table_center"]
        half_x, half_y = spec["table_half_xy"]
        table = ET.SubElement(
            worldbody, "body", name="task_low_microwave_table", pos=f"{x} {y} 0"
        )
        # A low wooden bench: broad side panels run all the way to the floor.
        # Separate visual/collision copies keep the supports visible in viewers.
        parts = [
            ("top", (0, 0, height - 0.025), (half_x, half_y, 0.025)),
            (
                "left_support",
                (0, -half_y + 0.03, (height - 0.052) / 2),
                (half_x - 0.015, 0.03, (height - 0.048) / 2),
            ),
            (
                "right_support",
                (0, half_y - 0.03, (height - 0.052) / 2),
                (half_x - 0.015, 0.03, (height - 0.048) / 2),
            ),
        ]
        for name, pos, size in parts:
            for visual in (False, True):
                ET.SubElement(
                    table,
                    "geom",
                    name=f"task_low_microwave_table_{name}_{'visual' if visual else 'collision'}",
                    type="box",
                    pos=" ".join(map(str, pos)),
                    size=" ".join(map(str, size)),
                    rgba="0.55 0.35 0.18 1" if visual else "0 0 0 0",
                    group="1" if visual else "0",
                    contype="0" if visual else "1",
                    conaffinity="0" if visual else "15",
                    friction="1 0.005 0.0001",
                )

    def _add_door_handle(self, microwave, spec):
        door = next(
            body
            for body in microwave.iter("body")
            if body.find(f"./joint[@name='{spec['target_joint']}']") is not None
        )
        asset_dir = (
            Path(__file__).resolve().parents[4] / "models/assets/objects/manual/handle"
        )
        scale = self.HANDLE_SCALE
        ET.SubElement(
            self.mujoco_arena.asset,
            "mesh",
            name="task_microwave_handle_mesh",
            file=str(asset_dir / "d03ab85f0a58cfcb46bfb9d7d3491743.obj"),
            scale=" ".join(str(value * scale) for value in (0.065, 0.16143, 0.18)),
        )
        ET.SubElement(
            self.mujoco_arena.asset,
            "texture",
            name="task_microwave_handle_texture",
            type="2d",
            file=str(asset_dir / "texture_pbr_20250901.png"),
        )
        ET.SubElement(
            self.mujoco_arena.asset,
            "material",
            name="task_microwave_handle_material",
            texture="task_microwave_handle_texture",
            specular="0.8",
            shininess="0.65",
        )
        # Native door coordinates: +x goes away from the hinge, +y is up,
        # +z faces the operator. Scale about the midpoint of the mounting plane
        # so the enlarged handle remains centered and attached to the door.
        handle_pos = np.array([0.425, -0.10, 0.055]) + (1.0 - scale) * np.array(
            [0.0, 0.09, -0.025]
        )
        handle = ET.SubElement(
            door,
            "body",
            name="task_microwave_handle",
            pos=" ".join(map(str, handle_pos)),
        )
        ET.SubElement(
            handle,
            "geom",
            name="task_microwave_handle_visual",
            type="mesh",
            mesh="task_microwave_handle_mesh",
            material="task_microwave_handle_material",
            group="1",
            contype="0",
            conaffinity="0",
            mass="0",
        )
        # Do not use the mesh's convex hull: it fills the finger opening.
        for name, start, end in (
            ("grip", "0 .025 .044", "0 .155 .044"),
            ("lower_mount", "0 .025 -.025", "0 .025 .044"),
            ("upper_mount", "0 .155 -.025", "0 .155 .044"),
        ):
            endpoints = np.fromstring(f"{start} {end}", sep=" ") * scale
            ET.SubElement(
                handle,
                "geom",
                name=f"task_microwave_handle_{name}",
                type="capsule",
                fromto=" ".join(map(str, endpoints)),
                size=str(0.012 * scale),
                group="0",
                rgba="0 0 0 0",
                contype="1",
                conaffinity="15",
                friction="0.8 0.03 0.01",
                density="500",
            )

    def _get_obj_cfgs(self):
        return []

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualLowMicrowaveSampler")

    def _setup_scene(self):
        super()._setup_scene()
        self.SUCCESS_CLOSE_FRACTION = float(
            self._get_task_spec()["success_close_fraction"]
        )
        self.sim.data.set_joint_qvel(self.target_joint, 0.0)

    def _raw_success(self):
        return bool(
            self.target_semantic_joint.fraction(self.sim)
            >= self.SUCCESS_OPEN_FRACTION - 1e-8
        )

    def _door_closed(self):
        return bool(
            self.target_semantic_joint.fraction(self.sim)
            <= self.SUCCESS_CLOSE_FRACTION + 1e-8
        )

    def _required_stage_hold_steps(self, task):
        if task.hold_steps is not None:
            return int(task.hold_steps)
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=self.TASK_LANG,
            composite_stages=[task.name for task in self._atomic_tasks],
            composite_ordering=self.COMPOSITE_ORDERING,
            table_height=float(self._get_task_spec()["table_center"][2]),
            success_definition=(
                "open the fixed low microwave to at least 40% for 5 control steps, "
                "then keep its door at most 10% open for 1 second"
            ),
        )
        return meta


class IThorKitchenOpenCloseLowMicrowave(_LowMicrowaveMixin, IThorKitchenEnv):
    HANDLE_SCALE = 1.5
    SPEC_TASK = "open_close_low_microwave"
    ATOMIC_SKILL = "OpenAndCloseLowMicrowave"
    COMPOSITE_KIND = "open_close_low_microwave"
    TASK_LANG = "crouch down, open the microwave on the low table, and close its door"
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)


class _LowMicrowaveAppleMixin(SupportContactPlacementMixin):
    APPLE_XML = "objaverse/apple/apple_0/model.xml"
    APPLE_INITIAL_LOCATION = "top"
    GRASP_HEIGHT_ALLOWANCE = 0.10
    PICK_LIFT_DELTA = 0.05
    MIN_LIFT_DELTA = 0.30
    RELEASE_DISTANCE = 0.18
    INSIDE_TOLERANCE = 0.003
    APPLE_INSIDE_XY_KEY = None
    APPLE_INSIDE_SUPPORT_HALF_XY = (0.05, 0.05)

    def _get_obj_cfgs(self):
        return [
            dict(name="apple", info=dict(mjcf_path=self.APPLE_XML), object_scale=1.0)
        ]

    def _setup_scene(self):
        super()._setup_scene()
        site = self._get_task_spec()["microwave_receptacle_site"]
        self._microwave_receptacle_site_id = self.sim.model.site_name2id(site)
        if site not in self.target_fixture.site_names:
            raise ValueError(
                f"{self.scene_name}: receptacle must belong to the microwave"
            )

    def _before_object_placement(self):
        super()._before_object_placement()
        if self.APPLE_INITIAL_LOCATION == "top":
            self._place_apple_on_top()
        elif self.APPLE_INITIAL_LOCATION == "inside":
            self._place_apple_inside()
        else:
            raise ValueError(
                f"{type(self).__name__}: unsupported apple initial location "
                f"{self.APPLE_INITIAL_LOCATION!r}"
            )
        self.sim.forward()
        self._apple_initial_root_z = self._apple_root_z()
        self._grasped_at_source_height = False

    def _apple_top_pos(self):
        spec = self._get_task_spec()
        key = "apple_top_pos" if "apple_top_pos" in spec else "apple_pos"
        return np.asarray(spec[key], dtype=float)

    def _apple_top_region(self):
        pos = self._apple_top_pos()
        return SemanticRegion(
            stable_id=f"{self.target_fixture.stable_id}:task:apple_top_support",
            fixture_id=self.target_fixture.stable_id,
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.array([0.06, 0.06, 0.02]),
            _world_center=pos,
            _world_yaw=0.0,
        )

    def _place_apple_on_top(self):
        pos = self._apple_top_pos()
        set_free_joint_bottom_at(self, self.objects["apple"], pos, yaw=0.0)
        if not self.place_object_on_support_contact(
            "apple",
            self._apple_top_region(),
            self.target_fixture,
            xy=pos[:2],
            inward_fractions=(0.0,),
            probe_above=0.02,
            probe_below=0.015,
            probe_step=0.0005,
        ):
            raise RuntimeError(
                f"{self.scene_name}: apple has no stable microwave support"
            )

    def _microwave_receptacle_pose(self):
        site = self._microwave_receptacle_site_id
        return (
            np.asarray(self.sim.data.site_xpos[site], dtype=float),
            np.asarray(self.sim.model.site_size[site], dtype=float),
            np.asarray(self.sim.data.site_xmat[site], dtype=float).reshape(3, 3),
        )

    def _microwave_inside_support_region(self):
        center, half_size, rotation = self._microwave_receptacle_pose()
        lower_z = float(center[2] - np.abs(rotation[2]) @ half_size)
        return SemanticRegion(
            stable_id=f"{self.target_fixture.stable_id}:task:apple_inside_support",
            fixture_id=self.target_fixture.stable_id,
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.array([*self.APPLE_INSIDE_SUPPORT_HALF_XY, 0.02]),
            _world_center=np.array([center[0], center[1], lower_z]),
            _world_yaw=0.0,
        )

    def _apple_inside_xy(self):
        center, _, _ = self._microwave_receptacle_pose()
        key = self.APPLE_INSIDE_XY_KEY
        if key is None:
            return center[:2]
        return np.asarray(self._get_task_spec()[key], dtype=float)

    def _place_apple_inside(self):
        support = self._microwave_inside_support_region()
        if not self.place_object_on_support_contact(
            "apple",
            support,
            self.target_fixture,
            xy=self._apple_inside_xy(),
            inward_fractions=(0.0,),
            probe_above=0.08,
            probe_below=0.03,
            probe_step=0.0005,
        ):
            raise RuntimeError(
                f"{self.scene_name}: apple has no stable support inside microwave"
            )
        tolerance = float(
            self._get_task_spec().get(
                "initial_inside_tolerance", self.INSIDE_TOLERANCE
            )
        )
        if not self._apple_fully_in_microwave(tolerance=tolerance):
            raise RuntimeError(
                f"{self.scene_name}: initial apple is outside microwave receptacle"
            )

    def _apple_in_microwave(self):
        center = np.mean(self.object_bbox_points("apple"), axis=0)
        site_center, half_size, rotation = self._microwave_receptacle_pose()
        local = (center - site_center) @ rotation
        return bool(np.all(np.abs(local) <= half_size))

    def _apple_fully_in_microwave(self, tolerance=None):
        center, half_size, rotation = self._microwave_receptacle_pose()
        local_points = (self.object_bbox_points("apple") - center) @ rotation
        if tolerance is None:
            tolerance = self.INSIDE_TOLERANCE
        return bool(np.all(np.abs(local_points) <= half_size + float(tolerance)))

    def _apple_contacts_target_fixture(self):
        object_bodies = self._body_tree_ids([self.obj_body_id["apple"]])
        fixture_bodies = self._fixture_body_ids(
            self.target_fixture, include_siblings=True
        )
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body1 in object_bodies and body2 in fixture_bodies) or (
                body2 in object_bodies and body1 in fixture_bodies
            ):
                return True
        return False

    def _apple_inside_and_released(self):
        return bool(
            self._apple_fully_in_microwave()
            and self._apple_contacts_target_fixture()
            and self.object_released("apple", th=self.RELEASE_DISTANCE)
        )

    def _apple_on_top_and_released(self):
        return bool(
            self.object_on_region("apple", self._apple_top_region())
            and self.object_released("apple", th=self.RELEASE_DISTANCE)
        )

    def _apple_root_z(self):
        root = self.objects["apple"].root_body
        return float(self.sim.data.get_body_xpos(root)[2])

    def _apple_lifted_from_source(self, minimum_delta):
        height = self._apple_root_z()
        grasped = self.g1_hand_grasping_object("apple")
        if (
            grasped
            and height <= self._apple_initial_root_z + self.GRASP_HEIGHT_ALLOWANCE
        ):
            self._grasped_at_source_height = True
        return bool(
            self._raw_success()
            and self._grasped_at_source_height
            and grasped
            and height >= self._apple_initial_root_z + float(minimum_delta)
        )

    def _apple_picked_up_with_door_open(self):
        return self._apple_lifted_from_source(self.PICK_LIFT_DELTA)

    def _apple_lifted_30cm_with_door_open(self):
        return self._apple_lifted_from_source(self.MIN_LIFT_DELTA)

    def _apple_on_top_with_door_open(self):
        return self._raw_success() and self._apple_on_top_and_released()

    def _apple_inside_with_door_open(self):
        return self._raw_success() and self._apple_inside_and_released()


class IThorKitchenPutAppleInLowMicrowave(
    _LowMicrowaveAppleMixin,
    _LowMicrowaveMixin,
    IThorKitchenEnv,
):
    SPEC_TASK = "put_apple_in_low_microwave"
    ATOMIC_SKILL = "OpenLowMicrowavePutAppleAndClose"
    COMPOSITE_KIND = "open_low_microwave_put_apple_close"
    TASK_LANG = "take the apple from the top of the microwave, crouch down, open its door, put the apple inside, and close the door"
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)
    ATOMIC_TASKS = (
        IThorAtomicTask("open_microwave", "_raw_success", hold_steps=5),
        IThorAtomicTask(
            "put_apple_in_microwave", "_apple_placed_with_door_open", hold_steps=5
        ),
        IThorAtomicTask("close_microwave", "_apple_inside_and_door_closed"),
    )

    def _apple_placed_with_door_open(self):
        return self._raw_success() and self._apple_in_microwave()

    def _apple_inside_and_door_closed(self):
        return self._door_closed() and self._apple_in_microwave()

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            target_object="apple",
            microwave_receptacle_site=self._get_task_spec()[
                "microwave_receptacle_site"
            ],
            success_definition=(
                "open the fixed low microwave to at least 40% for 5 control steps, "
                "put the apple center inside its native receptacle with the door still "
                "at least 40% open for 5 control steps, then keep the apple inside "
                "and the door at most 10% open for 1 second"
            ),
        )
        return meta


class _ExtendedLowMicrowaveSceneMixin:
    """Defer pose collision validation for newly calibrated microwave scenes."""

    _EXTENDED_SCENE_NAMES = {
        "FloorPlan6",
        "FloorPlan7",
        "FloorPlan24",
        "FloorPlan25",
    }

    def _finalize_articulation_robot_pose(self, distance):
        if self.scene_name not in self._EXTENDED_SCENE_NAMES:
            return super()._finalize_articulation_robot_pose(distance)
        # The room reset clears spawn clutter only after all task objects are
        # placed. Validate collisions on that completed reset state instead.
        self.sim.forward()
        return True


class IThorKitchenOpenLowMicrowaveAndLiftApple(
    _LowMicrowaveAppleMixin,
    _ExtendedLowMicrowaveSceneMixin,
    _LowMicrowaveMixin,
    IThorKitchenEnv,
):
    SPEC_TASK = "open_low_microwave_and_lift_apple"
    APPLE_INSIDE_XY_KEY = "apple_inside_xy"
    APPLE_INSIDE_SUPPORT_HALF_XY = (0.10, 0.10)
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ATOMIC_SKILL = "OpenLowMicrowaveAndLiftApple"
    COMPOSITE_KIND = "open_low_microwave_lift_apple"
    TASK_LANG = "crouch down, open the low microwave, and take out the apple"
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)
    APPLE_INITIAL_LOCATION = "inside"
    ATOMIC_TASKS = (
        IThorAtomicTask("open_microwave", "_raw_success", hold_steps=5),
        IThorAtomicTask("lift_apple", "_apple_lifted_30cm_with_door_open"),
    )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            target_object="apple",
            lift_height=self.MIN_LIFT_DELTA,
            success_definition=(
                "open the fixed low microwave to at least 40% for 5 control "
                "steps, grasp the apple near its reset height inside the "
                "microwave, then keep it grasped at least 0.30 m above that "
                "height with the door open for 1 second"
            ),
        )
        return meta


class IThorKitchenTakeAppleFromOpenLowMicrowaveToTop(
    _LowMicrowaveAppleMixin,
    _ExtendedLowMicrowaveSceneMixin,
    _LowMicrowaveMixin,
    IThorKitchenEnv,
):
    SPEC_TASK = "take_apple_from_open_low_microwave_to_top"
    APPLE_INSIDE_XY_KEY = "apple_inside_xy"
    APPLE_INSIDE_SUPPORT_HALF_XY = (0.10, 0.10)
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ATOMIC_SKILL = "TakeAppleFromOpenLowMicrowaveToTop"
    COMPOSITE_KIND = "take_apple_from_open_low_microwave_to_top"
    TASK_LANG = "crouch down, take the apple from the open low microwave, and place it on top of the microwave"
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)
    APPLE_INITIAL_LOCATION = "inside"
    ATOMIC_TASKS = (
        IThorAtomicTask(
            "pick_up_apple", "_apple_picked_up_with_door_open", hold_steps=5
        ),
        IThorAtomicTask("place_apple_on_microwave", "_apple_on_top_with_door_open"),
    )

    def _setup_baked_scene_fixtures(self):
        super()._setup_baked_scene_fixtures()
        self._add_apple_top_collision_guard()

    def _add_apple_top_collision_guard(self):
        spec = self._get_task_spec()
        guard = spec.get("apple_top_collision_guard")
        if guard is None:
            return

        microwave = self.mujoco_arena.worldbody.find(
            f"./body[@name='{spec['microwave_body']}']"
        )
        if microwave is None:
            raise ValueError(f"{self.scene_name}: missing calibrated microwave")

        body_pos = np.asarray(spec["microwave_pos"], dtype=float)
        quat = np.asarray(spec["microwave_quat"], dtype=float)
        quat /= np.linalg.norm(quat)
        w, x, y, z = quat
        rotation = np.array(
            (
                (
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                ),
                (
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                ),
                (
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ),
            )
        )
        center = np.asarray(guard["center"], dtype=float)
        local_center = rotation.T @ (center - body_pos)
        ET.SubElement(
            microwave,
            "geom",
            name="task_apple_top_collision_guard",
            type="box",
            pos=" ".join(map(str, local_center)),
            quat=" ".join(map(str, (quat[0], -quat[1], -quat[2], -quat[3]))),
            size=" ".join(map(str, guard["half_size"])),
            rgba="0 0 0 0",
            contype="1",
            conaffinity="15",
            friction="1 0.005 0.0001",
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            target_object="apple",
            success_definition=(
                "with the door initially 80% open, grasp the apple near its "
                "reset height inside the microwave and lift it by at least "
                "0.05 m for 5 control steps, then place it on the microwave "
                "top, release it, and keep the door open for 1 second"
            ),
        )
        return meta


class IThorKitchenCloseThenOpenLowMicrowave(
    _ExtendedLowMicrowaveSceneMixin,
    _LowMicrowaveMixin,
    IThorKitchenEnv,
):
    HANDLE_SCALE = 1.5
    SPEC_TASK = "close_then_open_low_microwave"
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ATOMIC_SKILL = "CloseThenOpenLowMicrowave"
    COMPOSITE_KIND = "close_then_open_low_microwave"
    TASK_LANG = "crouch down, close the open low microwave, and open its door again"
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)
    ATOMIC_TASKS = (
        IThorAtomicTask("close_microwave", "_door_closed", hold_steps=5),
        IThorAtomicTask("open_microwave", "_raw_success"),
    )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["success_definition"] = (
            "close the initially 80% open door to at most 10% for 5 control "
            "steps, then reopen it to at least 40% for 1 second"
        )
        return meta


class IThorKitchenOpenLowMicrowaveAndPutAppleInside(
    _LowMicrowaveAppleMixin,
    _ExtendedLowMicrowaveSceneMixin,
    _LowMicrowaveMixin,
    IThorKitchenEnv,
):
    SPEC_TASK = "open_low_microwave_and_put_apple_inside"
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ATOMIC_SKILL = "OpenLowMicrowaveAndPutAppleInside"
    COMPOSITE_KIND = "open_low_microwave_pick_top_apple_put_inside"
    TASK_LANG = "crouch down, open the low microwave, pick up the apple from its top, and place the apple inside"
    SCENE_SPECS = get_ithor_kitchen_task_specs(SPEC_TASK)
    ATOMIC_TASKS = (
        IThorAtomicTask("open_microwave", "_raw_success", hold_steps=5),
        IThorAtomicTask(
            "pick_up_apple", "_apple_picked_up_with_door_open", hold_steps=5
        ),
        IThorAtomicTask("put_apple_in_microwave", "_apple_inside_with_door_open"),
    )

    def _apple_picked_up_with_door_open(self):
        return bool(
            self._raw_success()
            and self._apple_root_z()
            >= self._apple_initial_root_z + self.PICK_LIFT_DELTA
        )

    def _apple_inside_with_door_open(self):
        door_open = self._raw_success()
        site_center, half_size, rotation = self._microwave_receptacle_pose()
        bbox_points = self.object_bbox_points("apple")
        center_local = (np.mean(bbox_points, axis=0) - site_center) @ rotation
        center_inside = bool(np.all(np.abs(center_local) <= half_size))
        contact = self._apple_contacts_target_fixture()
        return bool(door_open and center_inside and contact)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            target_object="apple",
            success_definition=(
                "open the fixed low microwave to at least 40% for 5 control "
                "steps, move the apple from near its reset height on top by at "
                "least 0.05 m for 5 control steps, then place its center inside "
                "with physical microwave contact and keep the door open for 1 "
                "second"
            ),
        )
        return meta


__all__ = [
    "IThorKitchenOpenCloseLowMicrowave",
    "IThorKitchenPutAppleInLowMicrowave",
    "IThorKitchenOpenLowMicrowaveAndLiftApple",
    "IThorKitchenTakeAppleFromOpenLowMicrowaveToTop",
    "IThorKitchenCloseThenOpenLowMicrowave",
    "IThorKitchenOpenLowMicrowaveAndPutAppleInside",
]
