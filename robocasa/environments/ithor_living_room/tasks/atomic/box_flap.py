"""Open and close one explicitly calibrated flap of a scene box."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.environments.ithor_living_room.tasks.atomic.raised_work_surface import (
    TableSizedRaisedSurfaceMixin,
)
from robocasa.models.arenas.molmospaces_arena import inline_defaults, remap_geom_groups
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class _IThorLivingRoomBoxFlapTask(TableSizedRaisedSurfaceMixin):
    """Task-local implementation shared only by the open/close variants."""

    ATOMIC_SKILL = "Articulation"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "articulation"
    SUPPORTED_SCENE_IDS = (202, 207, 209, 210, 212, 226)
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    BEHAVIOR = None
    INITIAL_OPEN_FRACTION = None

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_baked_scene_fixtures(self):
        super()._setup_baked_scene_fixtures()
        if self.scene_name == "FloorPlan207":
            # The shared room no longer has a tabletop box. Restore the native
            # fixture only for its existing open/close tasks, before semantics.
            scene_path = Path(self.mujoco_arena.scene_xml)
            fixture = ET.parse(scene_path.parent / "FloorPlan207_box.xml").getroot()
            # The extracted body uses the source scene's named defaults.
            source = ET.parse(scene_path).getroot()
            fixture.insert(0, source.find("default"))
            inline_defaults(fixture)
            remap_geom_groups(fixture)
            self.mujoco_arena.worldbody.extend(fixture.find("worldbody"))
            self.mujoco_arena.contact.extend(fixture.find("contact"))

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "work_surface_support",
                    "work_surface_top_z",
                    "box_fixture",
                    "box_root_pos",
                    "box_root_quat",
                    "target_joint",
                    "target_closed_qpos",
                    "target_open_qpos",
                    "sibling_initial_qpos",
                    "robot_pose",
                    "success_fraction",
                    "hold_seconds",
                ),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _get_scene_surface_calibration(self):
        return None

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        if pos.shape != (3,) or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: robot_pose needs XYZ + yaw"
            )
        return pos, float(pose["yaw"])

    def _setup_task_fixtures(self):
        IThorLivingRoomEnv._setup_task_fixtures(self)
        spec = self._get_task_spec()
        self.WORK_SURFACE_TOP_Z = float(spec["work_surface_top_z"])
        self._setup_table_sized_work_surface(
            [spec["box_root_pos"]],
            fixture_id=spec["work_surface_support"]["fixture"],
        )
        self.box_fixture = self.get_fixture(id=spec["box_fixture"])
        if self.box_fixture is None or not self.box_fixture.body_name:
            raise ValueError(f"{self.scene_name}/{self.SPEC_TASK}: missing box")
        box_body = self.find_scene_body(self.box_fixture.body_name)
        root_pos = np.asarray(spec["box_root_pos"], dtype=float)
        root_quat = np.asarray(spec["box_root_quat"], dtype=float)
        if (
            box_body is None
            or root_pos.shape != (3,)
            or root_quat.shape != (4,)
            or not np.isclose(np.linalg.norm(root_quat), 1.0, atol=1e-4)
        ):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: invalid box root placement"
            )
        box_body.set("pos", " ".join(str(float(value)) for value in root_pos))
        box_body.set("quat", " ".join(str(float(value)) for value in root_quat))
        root_free_joints = [
            joint
            for joint in box_body.findall("joint")
            if joint.get("type", "hinge") == "free"
        ]
        if len(root_free_joints) != 1:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: box root must have exactly "
                "one free joint"
            )
        # Fix the box base to the world while keeping all child flap hinges movable.
        box_body.remove(root_free_joints[0])
        self.target_joint = str(spec["target_joint"])
        joint_names = {
            joint.get("name")
            for joint in box_body.iter("joint")
            if joint.get("name") and "flap" in joint.get("name").lower()
        }
        sibling_names = set(spec["sibling_initial_qpos"])
        if self.target_joint not in joint_names or sibling_names != joint_names - {
            self.target_joint
        }:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: every non-target flap joint "
                "must have an explicit initial qpos"
            )
        self.target_closed_qpos = float(spec["target_closed_qpos"])
        self.target_open_qpos = float(spec["target_open_qpos"])
        self.sibling_initial_qpos = {
            str(name): float(qpos)
            for name, qpos in spec["sibling_initial_qpos"].items()
        }
        self.SUCCESS_FRACTION = float(spec["success_fraction"])
        self.HOLD_SECONDS = float(spec["hold_seconds"])
        if self.INITIAL_OPEN_FRACTION is None:
            raise ValueError(f"{self.scene_name}/{self.SPEC_TASK}: invalid threshold")
        initial_target_fraction = (
            self.INITIAL_OPEN_FRACTION
            if self.BEHAVIOR == "open"
            else 1.0 - self.INITIAL_OPEN_FRACTION
        )
        if (
            not 0.0 <= self.INITIAL_OPEN_FRACTION <= 1.0
            or not 0.0 < self.SUCCESS_FRACTION <= 1.0
            or initial_target_fraction >= self.SUCCESS_FRACTION
            or self.HOLD_SECONDS <= 0.0
        ):
            raise ValueError(f"{self.scene_name}/{self.SPEC_TASK}: invalid threshold")
        self.set_robot_base_pose(*self._get_scene_robot_pose_calibration())

    def _is_scene_valid(self):
        return bool(
            IThorLivingRoomEnv._is_scene_valid(self)
            and getattr(self, "work_surface_region", None) is not None
        )

    def _set_joint_qpos(self, joint_name, qpos):
        self.sim.data.set_joint_qpos(joint_name, float(qpos))
        joint_id = self.sim.model.joint_name2id(joint_name)
        velocity_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[velocity_address] = 0.0

    def _reset_internal(self):
        IThorLivingRoomEnv._reset_internal(self)
        self._set_joint_qpos(self.target_joint, self._initial_target_qpos())
        for name, qpos in self.sibling_initial_qpos.items():
            self._set_joint_qpos(name, qpos)
        self.sim.forward()
        self._success_hold_steps = 0

    def _initial_target_qpos(self):
        return self.target_closed_qpos + self.INITIAL_OPEN_FRACTION * (
            self.target_open_qpos - self.target_closed_qpos
        )

    def _target_fraction(self):
        qpos = float(self.sim.data.get_joint_qpos(self.target_joint))
        denominator = self.target_open_qpos - self.target_closed_qpos
        if abs(denominator) < 1e-8:
            return 0.0
        open_fraction = float(
            np.clip((qpos - self.target_closed_qpos) / denominator, 0.0, 1.0)
        )
        return open_fraction if self.BEHAVIOR == "open" else 1.0 - open_fraction

    def _setup_scene(self):
        IThorLivingRoomEnv._setup_scene(self)
        self._success_hold_steps = 0
        self.HOLD_STEPS = max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def _post_action(self, action):
        ret = IThorLivingRoomEnv._post_action(self, action)
        self._success_hold_steps = (
            self._success_hold_steps + 1
            if self._target_fraction() >= self.SUCCESS_FRACTION
            else 0
        )
        return ret

    def _check_success(self):
        return self._success_hold_steps >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = IThorLivingRoomEnv.get_ep_meta(self)
        ep_meta.update(
            lang=f"{self.BEHAVIOR} the box flap",
            target_fixture=self.box_fixture.stable_id,
            target_joint=self.target_joint,
            box_base_fixed=True,
            initial_open_fraction=self.INITIAL_OPEN_FRACTION,
            success_open_fraction_threshold=(
                self.SUCCESS_FRACTION
                if self.BEHAVIOR == "open"
                else 1.0 - self.SUCCESS_FRACTION
            ),
        )
        return ep_meta


class IThorLivingRoomOpenBoxFlap(
    _IThorLivingRoomBoxFlapTask,
    IThorLivingRoomEnv,
):
    """Open one flap of the box on the pad."""

    SPEC_TASK = "open_box_flap"
    BEHAVIOR = "open"
    INITIAL_OPEN_FRACTION = 0.2


class IThorLivingRoomCloseBoxFlap(
    _IThorLivingRoomBoxFlapTask,
    IThorLivingRoomEnv,
):
    """Close one initially open flap of the box on the pad."""

    SPEC_TASK = "close_box_flap"
    BEHAVIOR = "close"
    INITIAL_OPEN_FRACTION = 0.8


__all__ = ["IThorLivingRoomOpenBoxFlap", "IThorLivingRoomCloseBoxFlap"]
