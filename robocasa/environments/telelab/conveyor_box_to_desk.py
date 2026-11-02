"""TeleLab conveyor task: move a parcel from the belt onto the adjacent desk."""

import math

import numpy as np

from robocasa.environments.telelab.telelab_conveyor import TeleLabConveyor


_CONVEYOR_SCALE = (2.5 / 4.3, 0.7 / 1.2, 0.7 / 0.788)
_Z_AXIS_90_QUAT = (0.70710678, 0.0, 0.0, 0.70710678)
_NARROW_FACE_UP_QUAT = (0.70710678, 0.70710678, 0.0, 0.0)
_SHORT_EDGE_FRONT_QUAT = (0.5, 0.5, 0.5, 0.5)


def _quat_multiply(first, second):
    """Multiply MuJoCo-order (w, x, y, z) quaternions."""
    w1, x1, y1, z1 = first
    w2, x2, y2, z2 = second
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


class TeleLabConveyorPlaceBoxOnDesk(TeleLabConveyor):
    """Pick the three leftmost conveyor boxes and place them on the desk.

    The three payloads at the most negative X positions are retained as a
    small, fixed scene; all three must be moved onto the desk while the belt
    runs at 0.1 m/s. The first two are smaller parcels, and all three use the
    same light mass.
    """

    TASK_FAMILY = "pick_place"
    ATOMIC_SKILL = "PickPlace"
    SCENE_ID = 1
    SUPPORTED_SCENE_IDS = (SCENE_ID,)
    manual_episode_save = False
    CONVEYOR_SCALE = _CONVEYOR_SCALE
    CONVEYOR_SPEED = 0.05
    # Keep the belt stopped until the operator arms the task with the same
    # keyboard fallback as the PICO A+B+X+Y start chord.
    KEY_CONVEYOR_START = ord("E")
    DESK_SCALE = (1.2 / 1.59, 0.65 / 0.798, 0.6 / 0.844)
    # This task uses a tighter workstation layout than the task-free scene:
    # bring the robot toward the belt and shift the desk toward the robot.
    ROBOT_INNER_CLEARANCE_Y = 0.20
    L_CORNER_GAP_Y = 0.02
    DESK_SHIFT_X = 0.05
    BOX_SCALE = 0.5
    BOX_SCALE_OVERRIDES = {"box_1": 0.45, "box_2": 0.45}
    # MuJoCo masses are kilograms: 0.15 kg = 150 g per box.
    BOX_MASS_OVERRIDES = {"box_1": 0.15, "box_2": 0.15, "box_3": 0.08}
    BOX_MASS_LIMIT_KG = 0.2
    # Apply a 90-degree world-Z rotation to each retained box's existing
    # support-plane orientation.
    BOX_QUAT_OVERRIDES = {
        "box_1": _quat_multiply(_Z_AXIS_90_QUAT, _SHORT_EDGE_FRONT_QUAT),
        "box_2": _quat_multiply(_Z_AXIS_90_QUAT, _NARROW_FACE_UP_QUAT),
        "box_3": _quat_multiply(_Z_AXIS_90_QUAT, _SHORT_EDGE_FRONT_QUAT),
    }
    BOX_SPECS = tuple(
        (name, local_x * _CONVEYOR_SCALE[0])
        for name, local_x in TeleLabConveyor.BOX_SPECS[:3]
    )
    # Shift the three-box row to the far-left end of the belt while retaining
    # its authored spacing, and move it toward the robot on world -Y.
    BOX_POSITION_OFFSET = (-0.18, -0.25, 0.0)
    # The leftmost and rightmost boxes are another 5 cm closer than the
    # shared row offset.
    BOX_POSITION_OFFSET_OVERRIDES = {
        "box_1": (-0.18, -0.28, 0.0),
        "box_3": (-0.18, -0.28, 0.0),
    }
    SOURCE_BOX_NAMES = tuple(name for name, _ in BOX_SPECS)
    DESK_TOP_GEOM = "green_desk_collision_top"
    HOLD_SECONDS = 1.0
    DESK_EDGE_MARGIN = 0.015
    MAX_LINEAR_SPEED = 0.05
    MAX_ANGULAR_SPEED = 0.50
    SUCCESS_DEBUG_INTERVAL_SECONDS = 5.0

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
        self._success_hold = 0
        self._success_hold_steps = 1
        self._success_debug_next_log_time = self.SUCCESS_DEBUG_INTERVAL_SECONDS
        self._conveyor_started = False
        kwargs["conveyor_speed"] = 0.0
        kwargs["conveyor_scale"] = self.CONVEYOR_SCALE
        kwargs["desk_scale"] = self.DESK_SCALE
        super().__init__(*args, **kwargs)

    def _setup_references(self):
        super()._setup_references()
        try:
            self._desk_top_geom_id = self.sim.model.geom_name2id(self.DESK_TOP_GEOM)
        except (KeyError, ValueError):
            desk_geom_name = next(
                name
                for name in self.sim.model.geom_names
                if name and name.endswith(self.DESK_TOP_GEOM)
            )
            self._desk_top_geom_id = self.sim.model.geom_name2id(desk_geom_name)
        self._source_body_ids = {}
        self._source_bbox_geom_ids = {}
        for box_name in self.SOURCE_BOX_NAMES:
            source = self.objects[box_name]
            self._source_body_ids[box_name] = self.sim.model.body_name2id(
                source.root_body
            )
            bbox_name = next(
                name
                for name in self.sim.model.geom_names
                if name
                and name.endswith("reg_bbox")
                and source.naming_prefix in name
            )
            self._source_bbox_geom_ids[box_name] = self.sim.model.geom_name2id(
                bbox_name
            )

    def _reset_internal(self):
        super()._reset_internal()
        # The task belt is deliberately stationary at episode start.  The
        # viewer dispatches E to this hook before forwarding it to the G1
        # controller, so E retains its normal A+B+X+Y arming behavior.
        self._conveyor_started = False
        self.set_conveyor_speed(0.0)
        self._success_hold = 0
        self._success_hold_steps = max(
            1, int(math.ceil(self.HOLD_SECONDS * self.control_freq))
        )
        self._success_debug_next_log_time = self.SUCCESS_DEBUG_INTERVAL_SECONDS
        self.sim.forward()
        if any(self._box_on_desk(name) for name in self.SOURCE_BOX_NAMES):
            raise RuntimeError(
                f"{type(self).__name__}: a source box starts on the desk"
            )

    def handle_scene_keypress(self, keycode):
        """Start the belt on E while preserving the G1 E-mode toggle."""
        normalized = ord(chr(keycode).upper()) if 0 <= keycode < 256 else keycode
        if normalized == self.KEY_CONVEYOR_START:
            self.set_conveyor_speed(self.CONVEYOR_SPEED)
            self._conveyor_started = True
            print(f"[telelab] conveyor speed: {self.conveyor_speed:+.2f} m/s")
            # Return False so the viewer forwards E to the controller too;
            # E is the keyboard equivalent of the PICO A+B+X+Y chord.
            return False
        return super().handle_scene_keypress(keycode)

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        # A physical PICO A+B+X+Y chord changes the G1 mode manager directly,
        # so it does not pass through ``handle_scene_keypress``.  Mirror the
        # keyboard-E path once the controller reports an engaged mode.
        robots = getattr(self, "robots", ())
        if not robots:
            return
        composite = getattr(robots[0], "composite_controller", None)
        manager = getattr(composite, "_mode_manager", None)
        mode = getattr(manager, "current", None)
        if (
            mode is not None
            and getattr(mode, "name", str(mode)) != "OFF"
            and not self._conveyor_started
        ):
            self.set_conveyor_speed(self.CONVEYOR_SPEED)
            self._conveyor_started = True

    def _box_bbox_points(self, box_name):
        geom_id = self._source_bbox_geom_ids[box_name]
        center = np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float)
        rotation = np.asarray(
            self.sim.data.geom_xmat[geom_id], dtype=float
        ).reshape(3, 3)
        half_size = np.asarray(self.sim.model.geom_size[geom_id], dtype=float)
        signs = np.asarray(
            [
                (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
            ],
            dtype=float,
        )
        return center + (signs * half_size) @ rotation.T

    def _box_contacts_desk(self, box_name):
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self._desk_top_geom_id in pair:
                other = next(iter(pair - {self._desk_top_geom_id}))
                body_id = int(self.sim.model.geom_bodyid[other])
                if body_id == self._source_body_ids[box_name]:
                    return True
        return False

    def _box_inside_desk(self, box_name):
        points = self._box_bbox_points(box_name)
        center = np.asarray(
            self.sim.data.geom_xpos[self._desk_top_geom_id], dtype=float
        )
        rotation = np.asarray(
            self.sim.data.geom_xmat[self._desk_top_geom_id], dtype=float
        ).reshape(3, 3)
        local = (points - center) @ rotation
        half_size = np.asarray(
            self.sim.model.geom_size[self._desk_top_geom_id], dtype=float
        )
        margin = self.DESK_EDGE_MARGIN
        box_center = np.mean(local, axis=0)
        return bool(
            abs(box_center[0]) <= half_size[0] - margin
            and abs(box_center[2]) <= half_size[2] - margin
        )

    def _box_is_released(self, box_name):
        return not self._check_grasp(
            self.robots[0].gripper, self.objects[box_name]
        )

    def _box_is_stable(self, box_name):
        linear_speed, angular_speed = self._box_speeds(box_name)
        return bool(
            linear_speed <= self.MAX_LINEAR_SPEED
            and angular_speed <= self.MAX_ANGULAR_SPEED
        )

    def _box_speeds(self, box_name):
        velocity = np.asarray(
            self.sim.data.get_joint_qvel(self.objects[box_name].joints[0]),
            dtype=float,
        )
        return (
            float(np.linalg.norm(velocity[:3])),
            float(np.linalg.norm(velocity[3:])),
        )

    def _box_on_desk(self, box_name):
        return bool(
            self._box_contacts_desk(box_name)
            and self._box_inside_desk(box_name)
            and self._box_is_released(box_name)
            and self._box_is_stable(box_name)
        )

    def _all_boxes_on_desk(self):
        return all(self._box_on_desk(name) for name in self.SOURCE_BOX_NAMES)

    def _post_action(self, action):
        _, done, info = super()._post_action(action)
        self._success_hold = (
            self._success_hold + 1 if self._all_boxes_on_desk() else 0
        )
        # self._log_success_conditions()
        success = bool(self._check_success())
        info.update(
            task_success=success,
            success=success,
            success_hold_steps=self._success_hold,
            required_hold_steps=self._success_hold_steps,
        )
        return float(success), done, info

    def _log_success_conditions(self):
        """Print every success predicate once per five seconds of sim time."""
        sim_time = float(self.sim.data.time)
        if sim_time < self._success_debug_next_log_time:
            return
        interval = float(self.SUCCESS_DEBUG_INTERVAL_SECONDS)
        while self._success_debug_next_log_time <= sim_time:
            self._success_debug_next_log_time += interval

        box_parts = []
        for box_name in self.SOURCE_BOX_NAMES:
            contact = self._box_contacts_desk(box_name)
            inside = self._box_inside_desk(box_name)
            released = self._box_is_released(box_name)
            linear_speed, angular_speed = self._box_speeds(box_name)
            stable = bool(
                linear_speed <= self.MAX_LINEAR_SPEED
                and angular_speed <= self.MAX_ANGULAR_SPEED
            )
            complete = contact and inside and released and stable
            box_parts.append(
                f"{box_name}(contact={'OK' if contact else 'MISSING'}, "
                f"inside={'OK' if inside else 'MISSING'}, "
                f"released={'OK' if released else 'MISSING'}, "
                f"linear_speed={linear_speed:.4f}<={self.MAX_LINEAR_SPEED:.2f}="
                f"{'OK' if linear_speed <= self.MAX_LINEAR_SPEED else 'MISSING'}, "
                f"angular_speed={angular_speed:.4f}<={self.MAX_ANGULAR_SPEED:.2f}="
                f"{'OK' if angular_speed <= self.MAX_ANGULAR_SPEED else 'MISSING'}, "
                f"complete={'OK' if complete else 'MISSING'})"
            )

        print(
            f"[{type(self).__name__}][scene_id={self.scene_id}] "
            f"t={sim_time:.2f}s; "
            + "; ".join(box_parts)
            + f"; all_boxes={'OK' if self._all_boxes_on_desk() else 'MISSING'}, "
            f"hold={self._success_hold}/{self._success_hold_steps}, "
            f"success={'OK' if self._check_success() else 'MISSING'}",
            flush=True,
        )

    def _check_success(self):
        return self._success_hold >= self._success_hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            {
                "lang": "pick up the boxes from the conveyor belt and place them on the desk",
                "scene_id": self.scene_id,
                "source_objects": list(self.SOURCE_BOX_NAMES),
                "target_object": self.DESK_TOP_GEOM,
                "conveyor_speed": self.CONVEYOR_SPEED,
                "conveyor_dimensions": [2.5, 0.7, 0.7],
                "desk_dimensions": [1.2, 0.6, 0.65],
                "box_scale": self.BOX_SCALE,
                "box_scale_overrides": dict(self.BOX_SCALE_OVERRIDES),
                "box_mass_kg": dict(self.BOX_MASS_OVERRIDES),
                "retained_boxes": [name for name, _ in self.BOX_SPECS],
                "robot_base_yaw": self.ROBOT_BASE_YAW,
                "success_definition": (
                    "all three released boxes must contact the desk top, have "
                    "their centers inside its surface, and remain stable together for "
                    f"{self.HOLD_SECONDS:.1f} s"
                ),
            }
        )
        return meta


# Descriptive alias used by callers that name the source and destination.
TeleLabConveyorBoxToDesk = TeleLabConveyorPlaceBoxOnDesk

__all__ = ["TeleLabConveyorPlaceBoxOnDesk", "TeleLabConveyorBoxToDesk"]
