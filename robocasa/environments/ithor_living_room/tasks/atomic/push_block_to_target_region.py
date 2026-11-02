"""iTHOR living-room: push a small block into a visible target region."""

import xml.etree.ElementTree as ET

import numpy as np

from robocasa.environments.ithor_atomic.lift import RoomLiftMixin
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.environments.ithor_living_room.tasks.atomic.object_lift import (
    FixedObjectLiftMixin,
)
from robocasa.environments.ithor_living_room.tasks.atomic.raised_work_surface import (
    TableSizedRaisedSurfaceMixin,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomPushBlockToTargetRegion(
    TableSizedRaisedSurfaceMixin,
    FixedObjectLiftMixin,
    RoomLiftMixin,
    IThorLivingRoomEnv,
):
    """Push a small cube across the raised pad into the target region."""

    ATOMIC_SKILL = "PushObject"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "push"
    TARGET_NAME = "block"
    TARGET_XML = "models/assets/objects/aigen_objs/chocolate/chocolate_5/model.xml"
    SPEC_TASK = "push_block_to_target_region"
    SUPPORTED_SCENE_IDS = (202, 207, 209, 210, 212, 226)
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.04
    TARGET_VISUAL_HALF_HEIGHT = 0.0001

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "work_surface_support",
                    "work_surface_top_z",
                    "object_pos",
                    "object_yaw",
                    "target_region_pos",
                    "target_region_half_xy",
                    "target_region_yaw",
                    "robot_pose",
                    "max_lift_height",
                    "hold_seconds",
                    "source_category",
                    "source_wrapper_pos",
                    "source_wrapper_quat",
                    "source_bbox_pos",
                    "source_bbox_size",
                    "source_scale",
                ),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _get_scene_surface_calibration(self):
        return None

    def _setup_task_fixtures(self):
        IThorLivingRoomEnv._setup_task_fixtures(self)
        spec = self._get_task_spec()
        self.WORK_SURFACE_TOP_Z = float(spec["work_surface_top_z"])
        self._setup_table_sized_work_surface(
            [spec["object_pos"], spec["target_region_pos"]],
            fixture_id=spec["work_surface_support"]["fixture"],
        )
        if self.work_surface_region is None:
            return

        self.target_region_pos = np.asarray(spec["target_region_pos"], dtype=float)
        self.target_region_half_xy = np.asarray(
            spec["target_region_half_xy"], dtype=float
        )
        self.target_region_yaw = float(spec["target_region_yaw"])
        surface_center, _ = self.work_surface_region.world_pose()
        delta = self.target_region_pos[:2] - surface_center[:2]
        cosine = abs(np.cos(self.target_region_yaw))
        sine = abs(np.sin(self.target_region_yaw))
        target_world_half_xy = np.array(
            [
                cosine * self.target_region_half_xy[0]
                + sine * self.target_region_half_xy[1],
                sine * self.target_region_half_xy[0]
                + cosine * self.target_region_half_xy[1],
            ]
        )
        overlap = (
            self.work_surface_region.half_size[:2]
            + target_world_half_xy
            - np.abs(delta)
        )
        if (
            self.target_region_pos.shape != (3,)
            or self.target_region_half_xy.shape != (2,)
            or np.any(self.target_region_half_xy <= 0.0)
            or np.any(overlap <= 0.0)
            or not np.isclose(self.target_region_pos[2], surface_center[2], atol=1e-6)
        ):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: target region must overlap "
                "the raised work surface"
            )

        half_yaw = 0.5 * self.target_region_yaw
        target_body = ET.SubElement(
            self.mujoco_arena.worldbody,
            "body",
            name=f"target_region_{self.scene_name}",
            pos=" ".join(str(float(value)) for value in self.target_region_pos),
            quat=f"{np.cos(half_yaw)} 0 0 {np.sin(half_yaw)}",
        )
        ET.SubElement(
            target_body,
            "geom",
            name=f"target_region_{self.scene_name}_visual",
            type="box",
            size=(
                f"{self.target_region_half_xy[0]} "
                f"{self.target_region_half_xy[1]} "
                f"{self.TARGET_VISUAL_HALF_HEIGHT}"
            ),
            pos=f"0 0 {self.TARGET_VISUAL_HALF_HEIGHT}",
            rgba="0.15 0.65 0.25 0.8",
            contype="0",
            conaffinity="0",
            group="1",
        )
        self.MAX_LIFT_HEIGHT = float(spec["max_lift_height"])
        self.HOLD_SECONDS = float(spec["hold_seconds"])
        if min(self.MAX_LIFT_HEIGHT, self.HOLD_SECONDS) <= 0.0:
            raise ValueError(f"{self.scene_name}/{self.SPEC_TASK}: invalid threshold")
        self.set_robot_base_pose(*self._get_scene_robot_pose_calibration())

    def _is_scene_valid(self):
        return bool(
            IThorLivingRoomEnv._is_scene_valid(self)
            and getattr(self, "work_surface_region", None) is not None
        )

    def _before_object_placement(self):
        self._set_target_pose_from_spec()

    def _initial_jitter_allowed_contact_bodies(self, name):
        if name != self.TARGET_NAME:
            return set()
        return self._fixture_body_ids(self.work_surface_fixture)

    def _initial_jitter_candidate_is_valid(self):
        return (
            self.object_on_region(
                self.TARGET_NAME,
                self.work_surface_region,
                z_tol=0.05,
                require_contact=False,
            )
            and not self._block_is_in_target_region()
        )

    def _reset_internal(self):
        IThorLivingRoomEnv._reset_internal(self)
        self.sim.forward()
        self._initial_block_bottom = self.object_bottom_z(self.TARGET_NAME)
        self._block_was_touched = False
        self._block_was_lifted = False
        self._success_hold_steps = 0

    def _block_is_in_target_region(self):
        points = np.asarray(self.object_bbox_points(self.TARGET_NAME), dtype=float)
        delta = points[:, :2] - self.target_region_pos[:2]
        cosine = np.cos(-self.target_region_yaw)
        sine = np.sin(-self.target_region_yaw)
        local = np.column_stack(
            (
                cosine * delta[:, 0] - sine * delta[:, 1],
                sine * delta[:, 0] + cosine * delta[:, 1],
            )
        )
        return bool(np.all(np.abs(local) <= self.target_region_half_xy + 1e-4))

    def _raw_success(self):
        return bool(
            self._block_was_touched
            and not self._block_was_lifted
            and self._block_is_in_target_region()
            and self.object_on_region(
                self.TARGET_NAME,
                self.work_surface_region,
                z_tol=0.05,
                require_contact=False,
            )
        )

    def _setup_scene(self):
        IThorLivingRoomEnv._setup_scene(self)
        self._block_was_touched = False
        self._block_was_lifted = False
        self._success_hold_steps = 0
        self.HOLD_STEPS = max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def _post_action(self, action):
        ret = IThorLivingRoomEnv._post_action(self, action)
        self._block_was_touched = bool(
            self._block_was_touched
            or self.g1_hand_object_contact("left", self.TARGET_NAME)
            or self.g1_hand_object_contact("right", self.TARGET_NAME)
        )
        self._block_was_lifted = bool(
            self._block_was_lifted
            or self.object_bottom_z(self.TARGET_NAME)
            > self._initial_block_bottom + self.MAX_LIFT_HEIGHT
        )
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return ret

    def _check_success(self):
        return self._success_hold_steps >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = IThorLivingRoomEnv.get_ep_meta(self)
        ep_meta.update(
            lang="push the small block into the target region",
            target_object=self.TARGET_NAME,
            target_region="target_region",
        )
        return ep_meta


__all__ = ["IThorLivingRoomPushBlockToTargetRegion"]
