"""Pick up a cardboard box from the open Holodeck floor."""

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

import robocasa
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.arenas.molmospaces_arena import inline_defaults
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckPickBoxFromFloor(HolodeckEnv):
    """Lift the calibrated cardboard box clear of the room floor."""

    ATOMIC_SKILL = "PickObject"
    TASK_FAMILY = "pick"
    BOX_ASSET = (
        "objects/thor/Living Room and Bedroom Objects/Box_Full/Prefabs/"
        "Box_19/Box_19.xml"
    )
    BOX_SCALE = 2.0
    # Local half extents of the closed Box_19 body, including its fixed flaps.
    BOX_HALF_SIZE = (0.13, 0.28, 0.079)
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "pick_box_from_floor"
    HOLD_SECONDS = 1.0
    MIN_LIFT_DELTA = 0.30
    SUCCESS_DEBUG_INTERVAL_SECONDS = 2.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    NOISE_FREE_ROBOT_SPAWN = True

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=("robot_pose", "box_pose", "box_scale", "room_id"),
        )
        pose = self.task_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _is_scene_valid(self):
        return True

    def _get_obj_cfgs(self):
        return []

    def _create_objects(self):
        super()._create_objects()
        xml_path = self._make_box_xml()
        try:
            box = MJCFObject(
                name="box",
                mjcf_path=xml_path,
                scale=float(self.task_spec["box_scale"]),
            )
            # Register a scale-aware local bbox explicitly. THOR prefab region
            # geoms are not consistently scaled by every MJCF parser path.
            scale = float(self.task_spec["box_scale"])
            bbox_elem = ET.Element(
                "geom",
                {
                    "pos": "0 0 0",
                    "size": " ".join(
                        str(float(value) * scale) for value in self.BOX_HALF_SIZE
                    ),
                },
            )
            box._regions["bbox"] = {"elem": bbox_elem}
        finally:
            if os.path.exists(xml_path):
                os.remove(xml_path)
        self.objects[box.name] = box
        self.model.merge_objects([box])

    @classmethod
    def _make_box_xml(cls):
        """Flatten the THOR cardboard-box prefab into one free rigid body."""
        source = os.path.join(robocasa.models.assets_root, cls.BOX_ASSET)
        tree = ET.parse(source)
        root = tree.getroot()
        inline_defaults(root)
        worldbody = root.find("worldbody")
        body = worldbody.find("body") if worldbody is not None else None
        if body is None:
            raise ValueError(f"{source}: missing cardboard-box root body")
        worldbody.remove(body)
        for joint in list(body.iter("joint")):
            joint_parent = next(
                parent for parent in body.iter() if joint in list(parent)
            )
            joint_parent.remove(joint)
        body.set("name", "object")
        for geom in body.iter("geom"):
            if geom.get("group") == "4":
                geom.set("group", "0")
                # THOR exports collision meshes with debug colors (including
                # the red interior fill). They remain physical but invisible.
                geom.set("rgba", "0 0 0 0")
            elif geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                geom.set("group", "1")
        for site in body.iter("site"):
            # The prefab's red receptacle site is only a debug volume, not
            # part of this lift task's visible cardboard box.
            site.set("rgba", "0 0 0 0")
        ET.SubElement(
            body,
            "geom",
            name="reg_bbox",
            type="box",
            pos="0 0 0",
            size=" ".join(str(float(value)) for value in cls.BOX_HALF_SIZE),
            contype="0",
            conaffinity="0",
            group="1",
            rgba="0 0 0 0",
        )
        wrapper = ET.SubElement(worldbody, "body")
        wrapper.append(body)
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".xml", dir=os.path.dirname(source), delete=False
        ) as file:
            tree.write(file, encoding="utf-8", xml_declaration=True)
            return file.name

    def _setup_scene(self):
        super()._setup_scene()
        expected_room = self._normalized_room_id(self.task_spec["room_id"])
        robot_room = self.room_of_point([*self._robot_base_xy(), 0.0])
        if self._normalized_room_id(robot_room) != expected_room:
            raise RuntimeError(
                f"{self.scene_name}: robot is in room {robot_room}, "
                f"expected room {expected_room}"
            )
        if not self._robot_footprint_is_in_room(radius=0.25):
            raise RuntimeError(
                f"{self.scene_name}: box floor task robot starts off floor"
            )
        if self.count_body_collisions() != 0:
            raise RuntimeError(
                f"{self.scene_name}: box floor task robot has spawn collision"
            )

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

    def _reset_internal(self):
        super()._reset_internal()
        self._set_box_pose()
        self._success_hold = 0
        self._success_debug_next_log_time = float(
            self.SUCCESS_DEBUG_INTERVAL_SECONDS
        )
        self._success_debug_reported = False

    def _set_box_pose(self, pose=None, task_spec=None):
        """Place and calibrate the box for an atomic or composite reset."""
        box = self.objects["box"]
        qpos = self.sim.data.get_joint_qpos(box.joints[0]).copy()
        spec = self.task_spec if task_spec is None else task_spec
        pose = spec["box_pose"] if pose is None else pose
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(box.joints[0], qpos)
        self.sim.data.set_joint_qvel(box.joints[0], np.zeros(6))
        self.sim.forward()
        # Calibrate against the compiled geometry (including nested THOR
        # bodies) so the initial reset is truly grounded in every scene.
        geom_points, _ = self._body_geom_points([box.root_body])
        if not len(geom_points):
            raise RuntimeError(f"{self.scene_name}: box has no compiled geometry")
        qpos[:3] -= np.array([0.0, 0.0, np.min(geom_points[:, 2])])
        self.sim.data.set_joint_qpos(box.joints[0], qpos)
        self.sim.data.set_joint_qvel(box.joints[0], np.zeros(6))
        self.sim.forward()
        # Keep RoboCasa's semantic bbox aligned with the compiled geometry;
        # prefab region scaling differs between MuJoCo versions.
        geom_points, _ = self._body_geom_points([box.root_body])
        root_pos = np.asarray(self.object_pos("box"), dtype=float)
        local_min = np.min(geom_points, axis=0) - root_pos
        local_max = np.max(geom_points, axis=0) - root_pos
        bbox_elem = box._regions["bbox"]["elem"]
        bbox_elem.set(
            "pos",
            " ".join(str(float(v)) for v in (local_min + local_max) / 2.0),
        )
        bbox_elem.set(
            "size",
            " ".join(str(float(v)) for v in (local_max - local_min) / 2.0),
        )
        self._box_initial_z = self._box_root_z()
        box_room = self.room_of_point(self.object_pos("box"))
        expected_room = self._normalized_room_id(self.task_spec["room_id"])
        if self._normalized_room_id(box_room) != expected_room:
            raise RuntimeError(
                f"{self.scene_name}: box is in room {box_room}, expected "
                f"room {self.task_spec['room_id']}"
            )
        if float(np.min(geom_points[:, 2])) > 0.03:
            raise RuntimeError(
                f"{self.scene_name}: box floor pose is not grounded"
            )

    @staticmethod
    def _normalized_room_id(room_id):
        """Normalize numeric and ProcTHOR's ``roomN`` identifiers alike."""
        value = str(room_id)
        return value if value.startswith("room") else f"room{value}"

    def _box_root_z(self):
        return float(self.sim.data.get_body_xpos(self.objects["box"].root_body)[2])

    def _post_action(self, action):
        _, done, info = super()._post_action(action)
        box_z = self._box_root_z()
        lifted = box_z > self._box_initial_z + self.MIN_LIFT_DELTA
        self._success_hold = self._success_hold + 1 if lifted else 0
        success = bool(self._check_success())
        # Expose an explicit completion signal to teleop/data-collection
        # callers.  The base implementation evaluates reward before this
        # task's hold counter is updated, so recompute it after the predicate
        # update rather than returning the stale base reward.
        info["task_success"] = success
        info["success"] = success
        info["success_hold"] = self._success_hold
        self._log_success_status(box_z, lifted)
        return float(success), done, info

    def _log_success_status(self, box_z, lifted):
        """Print the lift predicate periodically and announce completion once."""
        # ``cur_time`` advances once per control step and remains reliable even
        # when a malformed controller state prevents MuJoCo's internal clock
        # from advancing. Fall back to the simulator clock for lightweight
        # test doubles.
        sim_time = getattr(self, "cur_time", None)
        if sim_time is None:
            sim = getattr(self, "sim", None)
            data = getattr(sim, "data", None)
            sim_time = getattr(data, "time", None)
        if sim_time is None:
            return
        sim_time = float(sim_time)
        task_success = bool(self._check_success())
        first_success = task_success and not getattr(
            self, "_success_debug_reported", False
        )
        next_log_time = getattr(
            self, "_success_debug_next_log_time", self.SUCCESS_DEBUG_INTERVAL_SECONDS
        )
        if sim_time < next_log_time and not first_success:
            return
        if sim_time >= next_log_time:
            while next_log_time <= sim_time:
                next_log_time += self.SUCCESS_DEBUG_INTERVAL_SECONDS
            self._success_debug_next_log_time = next_log_time
        if task_success:
            self._success_debug_reported = True

        threshold_z = self._box_initial_z + self.MIN_LIFT_DELTA
        print(
            f"[HolodeckPickBoxFromFloor][{self.scene_name}] "
            f"t={sim_time:.2f}s; 判定条件: 抬升>{self.MIN_LIFT_DELTA:.2f}m="
            f"{lifted}, 保持={self._success_hold}/{self.hold_steps}; "
            f"当前高度={box_z:.4f}m, 初始高度={self._box_initial_z:.4f}m, "
            f"抬升量={box_z - self._box_initial_z:.4f}m, "
            f"阈值高度={threshold_z:.4f}m; "
            f"任务={'成功' if task_success else '未成功'}",
            flush=True,
        )

    @property
    def lift_delta(self):
        return float(self.MIN_LIFT_DELTA)

    @property
    def hold_steps(self):
        control_freq = getattr(self, "control_freq", None)
        if control_freq is None:
            # Lightweight predicate tests may construct the task without the
            # full robosuite environment; one step is the safest fallback.
            return 1
        return max(1, int(round(self.HOLD_SECONDS * control_freq)))

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = "pick up the cardboard box from the floor and lift it"
        meta["source_object"] = "box"
        meta["success_definition"] = (
            f"the box root rises more than {self.lift_delta:.2f} m above "
            f"the floor reset height for {self.HOLD_SECONDS:.1f} s"
        )
        return meta


__all__ = ["HolodeckPickBoxFromFloor"]
