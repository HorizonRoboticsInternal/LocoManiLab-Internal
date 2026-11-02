"""TeleLab task: carry a shelf cup to the adjacent desk."""

import math
from pathlib import Path

import numpy as np

from robocasa.environments.telelab.telelab_mani import TeleLabMani
from robocasa.models.objects.objects import MJCFObject
from robosuite.utils.mjcf_utils import array_to_string, string_to_array


_ASSET_ROOT = Path(__file__).resolve().parents[2] / "models" / "assets"
_CUP_ASSET_DIR = _ASSET_ROOT / "objects" / "manual" / "cup"


class TeleLabCarryCupFromShelfToDesk(TeleLabMani):
    """Lift a cup from the shelf, approach the desk, and place it there."""

    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "pick_navigate_place"
    COMPOSITE_ORDERING = (
        "pick_cup_from_shelf",
        "walk_to_desk",
        "place_cup_on_desk",
    )
    SCENE_ID = 1
    SUPPORTED_SCENE_IDS = (SCENE_ID,)
    manual_episode_save = False

    CUP_NAME = "cup"
    CUP_XML = _CUP_ASSET_DIR / "cup.xml"

    # Board 2 supports the middle (second) shelf bay. The visual shelf mesh's
    # matching support is at about 0.86 m after the fixture scale is applied;
    # keep the task collision support at that same visible surface.
    SOURCE_SHELF_BOARD = "shelf_collision_board_2"
    SOURCE_SHELF_SURFACE_HEIGHT = 0.86
    SOURCE_SHELF_LOCAL_HALF_THICKNESS = 0.00543
    SOURCE_SHELF_LOCAL_CENTER_Z = (
        SOURCE_SHELF_SURFACE_HEIGHT - SOURCE_SHELF_LOCAL_HALF_THICKNESS
    )
    # Use the negative-Y end of the shelf, away from the desk at Y ~= 2.0.
    SOURCE_XY = (-0.65, 0.90)
    SOURCE_SUPPORT_CLEARANCE = 0.002

    TARGET_DESK_SURFACE = "desk_collision_top"
    CUP_BOTTOM_GEOM = "cup_base"
    PICK_LIFT_DELTA = 0.10
    DESK_APPROACH_DISTANCE = 0.35
    STAGE_HOLD_SECONDS = 1.0
    MAX_LINEAR_SPEED = 0.05
    MAX_ANGULAR_SPEED = 0.50
    DESK_SUPPORT_Z_TOLERANCE = 0.01
    SUCCESS_DEBUG_INTERVAL_SECONDS = 5.0

    # Start at the shelf-side operation position. The desk is intentionally
    # outside the approach radius.
    ROBOT_BASE_POS = (-0.15, 0.85, 0.0)
    ROBOT_BASE_YAW = math.atan2(
        SOURCE_XY[1] - ROBOT_BASE_POS[1],
        SOURCE_XY[0] - ROBOT_BASE_POS[0],
    )

    def __init__(self, *args, scene_id=None, scene_ids=None, **kwargs):
        requested_scene = self.SCENE_ID if scene_id is None else scene_id
        if scene_ids is not None:
            if isinstance(scene_ids, (list, tuple)):
                if len(scene_ids) != 1:
                    raise ValueError(
                        f"{type(self).__name__} supports exactly one scene id: "
                        f"{self.SCENE_ID}"
                    )
                requested_scene = scene_ids[0]
            else:
                requested_scene = scene_ids
        if int(requested_scene) != self.SCENE_ID:
            raise ValueError(
                f"{type(self).__name__} supports only scene_id={self.SCENE_ID}, "
                f"got {requested_scene!r}"
            )

        self.scene_id = self.SCENE_ID
        self.objects = {}
        self._stage = 0
        self._stage_hold = 0
        self._stage_hold_steps = 1
        self._cup_lifted_once = False
        self._success_debug_next_log_time = self.SUCCESS_DEBUG_INTERVAL_SECONDS
        kwargs.setdefault("robot_base_pos", self.ROBOT_BASE_POS)
        kwargs.setdefault("robot_base_ori", (0.0, 0.0, self.ROBOT_BASE_YAW))
        super().__init__(*args, **kwargs)

    def _load_model(self):
        super()._load_model()
        self._calibrate_source_shelf_collision()

        self.cup = MJCFObject(
            name=self.CUP_NAME,
            mjcf_path=str(self.CUP_XML),
            scale=1.0,
        )
        self.cup.set_pos((*self.SOURCE_XY, 0.0))
        self.objects[self.CUP_NAME] = self.cup
        self.model.merge_objects([self.cup])

    def _calibrate_source_shelf_collision(self):
        shelf = self.fixtures["shelf"]
        geom_name = f"{shelf.naming_prefix}{self.SOURCE_SHELF_BOARD}"
        geom = shelf.get_obj().find(f".//geom[@name='{geom_name}']")
        if geom is None:
            raise RuntimeError(
                f"{type(self).__name__}: missing source shelf geom {geom_name!r}"
            )

        position = string_to_array(geom.get("pos"))
        half_size = string_to_array(geom.get("size"))
        position[1] = self.SOURCE_SHELF_LOCAL_CENTER_Z
        half_size[1] = self.SOURCE_SHELF_LOCAL_HALF_THICKNESS
        geom.set("pos", array_to_string(position))
        geom.set("size", array_to_string(half_size))

    def _setup_references(self):
        super()._setup_references()
        self._cup_body_id = self.sim.model.body_name2id(self.cup.root_body)
        # MJCFObject preserves the hand-authored cup geom names when merged;
        # unlike fixture geoms, these are not prefixed with ``cup_``.
        self._cup_bottom_geom_name = self.CUP_BOTTOM_GEOM
        self._cup_bottom_geom_id = self.sim.model.geom_name2id(
            self._cup_bottom_geom_name
        )
        shelf_geom_name = (
            f"{self.fixtures['shelf'].naming_prefix}{self.SOURCE_SHELF_BOARD}"
        )
        self._source_shelf_geom_id = self.sim.model.geom_name2id(shelf_geom_name)
        self._target_desk_geom_name = (
            f"{self.fixtures['desk'].naming_prefix}{self.TARGET_DESK_SURFACE}"
        )
        self._target_desk_geom_id = self.sim.model.geom_name2id(
            self._target_desk_geom_name
        )

    def _reset_internal(self):
        super()._reset_internal()

        self._set_cup_pose(
            (
                *self.SOURCE_XY,
                self._geom_top_z(self._source_shelf_geom_id)
                + self.SOURCE_SUPPORT_CLEARANCE,
            )
        )
        self.sim.forward()

        self._source_cup_root_z = self._cup_root_z()
        self._stage = 0
        self._stage_hold = 0
        self._stage_hold_steps = max(
            1,
            int(math.ceil(self.STAGE_HOLD_SECONDS * self.control_freq)),
        )
        self._cup_lifted_once = False
        self._success_debug_next_log_time = self.SUCCESS_DEBUG_INTERVAL_SECONDS
        if self._final_stage_goal():
            raise RuntimeError(
                f"{type(self).__name__}: reset already satisfies the task goal"
            )

    def _set_cup_pose(self, position):
        joint_name = self.cup.joints[0]
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        qpos[:3] = np.asarray(position, dtype=float)
        qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.sim.data.set_joint_qpos(joint_name, qpos)

        qvel_addr = self.sim.model.get_joint_qvel_addr(joint_name)
        if isinstance(qvel_addr, tuple):
            self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = 0.0
        else:
            self.sim.data.qvel[qvel_addr : qvel_addr + 6] = 0.0

    def _geom_top_z(self, geom_id):
        rotation = np.asarray(
            self.sim.data.geom_xmat[geom_id], dtype=float
        ).reshape(3, 3)
        half_size = np.asarray(self.sim.model.geom_size[geom_id], dtype=float)
        return float(
            self.sim.data.geom_xpos[geom_id][2]
            + np.abs(rotation[2, :]) @ half_size
        )

    def _robot_base_xy(self):
        root_body = self.robots[0].robot_model.root_body
        return np.asarray(self.sim.data.get_body_xpos(root_body), dtype=float)[:2]

    def _cup_root_z(self):
        return float(self.sim.data.body_xpos[self._cup_body_id][2])

    def _cup_is_picked(self):
        reached_lift_height = (
            self._cup_root_z() >= self._source_cup_root_z + self.PICK_LIFT_DELTA
        )
        if reached_lift_height:
            self._cup_lifted_once = True
        return bool(getattr(self, "_cup_lifted_once", False))

    def _desk_approach_distance(self):
        """Shortest XY distance from the base to the desktop collision box."""
        geom_id = self._target_desk_geom_id
        rotation = np.asarray(
            self.sim.data.geom_xmat[geom_id], dtype=float
        ).reshape(3, 3)
        horizontal_axes = np.argsort(np.abs(rotation[2, :]))[:2]
        displacement = np.zeros(3, dtype=float)
        displacement[:2] = (
            self._robot_base_xy() - self.sim.data.geom_xpos[geom_id][:2]
        )
        local_displacement = rotation.T @ displacement
        half_size = np.asarray(self.sim.model.geom_size[geom_id], dtype=float)
        outside = np.maximum(
            np.abs(local_displacement[horizontal_axes])
            - half_size[horizontal_axes],
            0.0,
        )
        return float(np.linalg.norm(outside))

    def _cup_carried_to_desk(self):
        return bool(self._desk_approach_distance() <= self.DESK_APPROACH_DISTANCE)

    def _cup_speeds(self):
        velocity = np.asarray(
            self.sim.data.get_joint_qvel(self.cup.joints[0]),
            dtype=float,
        )
        return (
            float(np.linalg.norm(velocity[:3])),
            float(np.linalg.norm(velocity[3:])),
        )

    def _cup_is_stable(self):
        linear_speed, angular_speed = self._cup_speeds()
        return bool(
            linear_speed <= self.MAX_LINEAR_SPEED
            and angular_speed <= self.MAX_ANGULAR_SPEED
        )

    def _cup_bottom_supported_by_desk(self):
        """Detect tabletop support even when MuJoCo reports no contact pair."""
        within_desk, cup_bottom_z, desk_top_z = self._cup_desk_support_geometry()
        return bool(
            within_desk
            and abs(cup_bottom_z - desk_top_z) <= self.DESK_SUPPORT_Z_TOLERANCE
        )

    def _cup_desk_support_geometry(self):
        cup_geom_id = self._cup_bottom_geom_id
        desk_geom_id = self._target_desk_geom_id
        cup_rotation = np.asarray(
            self.sim.data.geom_xmat[cup_geom_id], dtype=float
        ).reshape(3, 3)
        cup_half_size = np.asarray(
            self.sim.model.geom_size[cup_geom_id], dtype=float
        )
        cup_bottom_z = float(
            self.sim.data.geom_xpos[cup_geom_id][2]
            - np.abs(cup_rotation[2, :]) @ cup_half_size
        )
        desk_rotation = np.asarray(
            self.sim.data.geom_xmat[desk_geom_id], dtype=float
        ).reshape(3, 3)
        displacement = np.zeros(3, dtype=float)
        displacement[:] = (
            self.sim.data.geom_xpos[cup_geom_id]
            - self.sim.data.geom_xpos[desk_geom_id]
        )
        local_displacement = desk_rotation.T @ displacement
        desk_half_size = np.asarray(
            self.sim.model.geom_size[desk_geom_id], dtype=float
        )
        horizontal_axes = np.argsort(np.abs(desk_rotation[2, :]))[:2]
        within_desk = bool(
            np.all(
                np.abs(local_displacement[horizontal_axes])
                <= desk_half_size[horizontal_axes]
            )
        )
        return within_desk, cup_bottom_z, self._geom_top_z(desk_geom_id)

    def _cup_on_desk(self):
        contact = self.check_contact(
            self._cup_bottom_geom_name,
            self._target_desk_geom_name,
        )
        return bool(
            (contact or self._cup_bottom_supported_by_desk())
            and self._cup_is_stable()
        )

    def _final_stage_goal(self):
        return bool(
            self._cup_on_desk()
            and self._desk_approach_distance() <= self.DESK_APPROACH_DISTANCE
        )

    def _advance_task_progress(self):
        if self._stage == 0:
            if self._cup_is_picked():
                self._stage = 1
            return

        if self._stage == 1:
            if self._cup_carried_to_desk():
                self._stage = 2
            return

        if self._stage == 2:
            goal_satisfied = self._final_stage_goal()
            self._stage_hold = self._stage_hold + 1 if goal_satisfied else 0
            if self._stage_hold >= self._stage_hold_steps:
                self._stage = 3
                self._stage_hold = 0
            return

        if not self._final_stage_goal():
            self._stage = 2
            self._stage_hold = 0

    def _log_success_conditions(self):
        """Print the success predicates once per five simulated seconds."""
        sim_time = float(self.sim.data.time)
        next_log_time = getattr(
            self,
            "_success_debug_next_log_time",
            self.SUCCESS_DEBUG_INTERVAL_SECONDS,
        )
        if sim_time < next_log_time:
            return

        interval = float(self.SUCCESS_DEBUG_INTERVAL_SECONDS)
        while next_log_time <= sim_time:
            next_log_time += interval
        self._success_debug_next_log_time = next_log_time

        cup_root_z = self._cup_root_z()
        lift_target_z = self._source_cup_root_z + self.PICK_LIFT_DELTA
        current_lift_ok = cup_root_z >= lift_target_z
        lifted_once = self._cup_is_picked()
        approach_distance = self._desk_approach_distance()
        approach_ok = approach_distance <= self.DESK_APPROACH_DISTANCE
        contact = bool(
            self.check_contact(
                self._cup_bottom_geom_name,
                self._target_desk_geom_name,
            )
        )
        within_desk, cup_bottom_z, desk_top_z = self._cup_desk_support_geometry()
        geometric_support = bool(
            within_desk
            and abs(cup_bottom_z - desk_top_z) <= self.DESK_SUPPORT_Z_TOLERANCE
        )
        linear_speed, angular_speed = self._cup_speeds()
        stable = bool(
            linear_speed <= self.MAX_LINEAR_SPEED
            and angular_speed <= self.MAX_ANGULAR_SPEED
        )
        desk_goal = bool((contact or geometric_support) and stable)
        final_goal = bool(desk_goal and approach_ok)
        active_stage = (
            self.COMPOSITE_ORDERING[self._stage]
            if self._stage < len(self.COMPOSITE_ORDERING)
            else None
        )
        print(
            f"[{type(self).__name__}][scene_id={self.scene_id}] "
            f"t={sim_time:.2f}s stage={self._stage}/3 "
            f"active={active_stage}; "
            f"lift(root_z={cup_root_z:.3f}m >= target={lift_target_z:.3f}m="
            f"{'OK' if current_lift_ok else 'MISSING'}, "
            f"completed_once={'OK' if lifted_once else 'MISSING'}); "
            f"approach(distance={approach_distance:.3f}m <= "
            f"{self.DESK_APPROACH_DISTANCE:.3f}m="
            f"{'OK' if approach_ok else 'MISSING'}); "
            f"desk(contact={'OK' if contact else 'MISSING'}, "
            f"geometric_support={'OK' if geometric_support else 'MISSING'}, "
            f"within_desk={'OK' if within_desk else 'MISSING'}, "
            f"bottom_z={cup_bottom_z:.3f}m, desk_top_z={desk_top_z:.3f}m, "
            f"z_gap={abs(cup_bottom_z - desk_top_z):.4f}m <= "
            f"{self.DESK_SUPPORT_Z_TOLERANCE:.3f}m, "
            f"linear_speed={linear_speed:.4f}m/s <= {self.MAX_LINEAR_SPEED:.2f}="
            f"{'OK' if linear_speed <= self.MAX_LINEAR_SPEED else 'MISSING'}, "
            f"angular_speed={angular_speed:.4f}rad/s <= "
            f"{self.MAX_ANGULAR_SPEED:.2f}="
            f"{'OK' if angular_speed <= self.MAX_ANGULAR_SPEED else 'MISSING'}, "
            f"goal={'OK' if desk_goal else 'MISSING'}); "
            f"hold={self._stage_hold}/{self._stage_hold_steps}; "
            f"final_goal={'OK' if final_goal else 'MISSING'}; "
            f"success={'OK' if self._check_success() else 'MISSING'}",
            flush=True,
        )

    def _post_action(self, action):
        _, done, info = super()._post_action(action)
        self._advance_task_progress()
        self._log_success_conditions()
        success = bool(self._check_success())
        info.update(
            task_success=success,
            success=success,
            stage_index=self._stage,
            stage_hold_steps=self._stage_hold,
            active_stage=(
                self.COMPOSITE_ORDERING[self._stage]
                if self._stage < len(self.COMPOSITE_ORDERING)
                else None
            ),
        )
        return float(success), done, info

    def _check_success(self):
        return bool(
            self._stage == len(self.COMPOSITE_ORDERING)
            and self._final_stage_goal()
        )

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            {
                "lang": (
                    "lift the cup from the second shelf, approach the desk, "
                    "and place the cup on the desk"
                ),
                "scene_id": self.scene_id,
                "task_family": self.TASK_FAMILY,
                "stage_order": list(self.COMPOSITE_ORDERING),
                "composite_stages": list(self.COMPOSITE_ORDERING),
                "source_fixture": "shelf",
                "source_level": "second shelf",
                "target_fixture": "desk",
                "target_object": self.CUP_NAME,
                "cup_asset": str(self.CUP_XML.relative_to(_ASSET_ROOT)),
                "pick_lift_delta_m": self.PICK_LIFT_DELTA,
                "desk_approach_distance_m": self.DESK_APPROACH_DISTANCE,
                "desk_support_z_tolerance_m": self.DESK_SUPPORT_Z_TOLERANCE,
                "stage_hold_seconds": self.STAGE_HOLD_SECONDS,
                "success_definition": (
                    "the cup must be lifted from the shelf, the robot must "
                    "approach the desk, and the cup bottom must be supported "
                    "by the desktop while stable for 1 second"
                ),
            }
        )
        return ep_meta


__all__ = ["TeleLabCarryCupFromShelfToDesk"]
