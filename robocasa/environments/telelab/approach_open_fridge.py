"""TeleLab task: approach the fridge, then open its main door."""

import math

import numpy as np

from robocasa.environments.telelab.telelab_mani import TeleLabMani


class TeleLabApproachOpenFridge(TeleLabMani):
    """Walk to the TeleLab fridge and open the main fridge door.

    The task is deliberately a two-stage state machine.  Opening the door
    before the approach stage is latched does not count; if the door is
    already open when the robot arrives, it must first be closed and then
    opened again. The door starts 20% open, and reaching 80% open after the
    approach stage completes the task.
    """

    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "navigate_then_manipulate"
    COMPOSITE_ORDERING = ("approach_fridge", "open_fridge_door")
    # TeleLab scenes are authored rather than downloaded FloorPlans. Keep a
    # stable numeric ID for trajectory metadata and diagnostics.
    SCENE_ID = 1
    SUPPORTED_SCENE_IDS = (SCENE_ID,)
    SUCCESS_DEBUG_INTERVAL_SECONDS = 5.0

    APPROACH_DISTANCE = 1.0
    APPROACH_HOLD_STEPS = 5
    INITIAL_OPEN_FRACTION = 0.20
    OPEN_FRACTION = 0.80
    REARM_CLOSED_FRACTION = INITIAL_OPEN_FRACTION
    OPEN_HOLD_SECONDS = 0.0

    # The authored manual handle is offset from the fixture body origin. The
    # values below include TeleLabMani's anisotropic fixture scale and locate
    # the handle center in the world XY plane for the formal scene.
    ROBOT_BASE_POS = (1, 1, 0.0)
    FRIDGE_HANDLE_OFFSET_XY = (-0.149665289449, -0.253734222665)
    FRIDGE_POS_XY = (1.676210710551, 1.757265777335)
    # Spawn facing the handle center, which is also the navigation reference.
    ROBOT_BASE_YAW = math.atan2(
        FRIDGE_POS_XY[1] - ROBOT_BASE_POS[1],
        FRIDGE_POS_XY[0] - ROBOT_BASE_POS[0],
    )

    def __init__(self, *args, scene_id=None, scene_ids=None, **kwargs):
        # Accept both the direct constructor spelling and the repository-wide
        # ``--scene-id`` forwarding spelling (``scene_ids``).
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
        kwargs.setdefault("robot_base_pos", self.ROBOT_BASE_POS)
        kwargs.setdefault("robot_base_ori", (0.0, 0.0, self.ROBOT_BASE_YAW))
        self._approach_hold = 0
        self._open_hold = 0
        self._stage = 0
        self._door_rearmed = False
        self._open_hold_steps = 1
        self._success_debug_next_log_time = self.SUCCESS_DEBUG_INTERVAL_SECONDS
        super().__init__(*args, **kwargs)

    def _setup_references(self):
        super()._setup_references()
        self.fridge = self.fixtures["fridge"]
        self._main_door_joint_names = tuple(self.fridge._fridge_door_joint_names)
        if not self._main_door_joint_names:
            raise RuntimeError("TeleLab fridge has no main fridge door joint")

    def _reset_internal(self):
        super()._reset_internal()

        # The manually authored hinge has friction, but explicitly setting the
        # 20%-open initial state avoids reset drift after a model rebuild.
        self.fridge.open_door(
            self,
            min=self.INITIAL_OPEN_FRACTION,
            max=self.INITIAL_OPEN_FRACTION,
        )
        for joint_name in self._main_door_joint_names:
            qvel_addr = self.sim.model.get_joint_qvel_addr(joint_name)
            self.sim.data.qvel[qvel_addr] = 0.0
        self.sim.forward()

        self._approach_hold = 0
        self._open_hold = 0
        self._stage = 0
        self._door_rearmed = False
        self._open_hold_steps = 1
        self._success_debug_next_log_time = self.SUCCESS_DEBUG_INTERVAL_SECONDS

        # Door friction is the only dynamic articulation in this task.  A few
        # noslip iterations keep the hinge from drifting while it is held open.
        self.sim.model.opt.noslip_iterations = max(
            int(self.sim.model.opt.noslip_iterations), 10
        )

    def _robot_base_xy(self):
        root_body = self.robots[0].robot_model.root_body
        return np.asarray(self.sim.data.get_body_xpos(root_body), dtype=float)[:2]

    def _fridge_operation_xy(self):
        """Return the scaled manual-handle center for navigation."""
        angle = float(self.fridge.rot)
        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)
        offset_x, offset_y = self.FRIDGE_HANDLE_OFFSET_XY
        rotated_offset = np.array(
            (
                cos_angle * offset_x - sin_angle * offset_y,
                sin_angle * offset_x + cos_angle * offset_y,
            ),
            dtype=float,
        )
        return np.asarray(self.fridge.pos, dtype=float)[:2] + rotated_offset

    def approach_distance(self):
        return float(np.linalg.norm(self._robot_base_xy() - self._fridge_operation_xy()))

    def _door_open_fraction(self):
        states = self.fridge.get_joint_state(self, self._main_door_joint_names)
        return min(float(value) for value in states.values())

    def _post_action(self, action):
        ret = super()._post_action(action)

        if self._stage == 0:
            if self.approach_distance() <= self.APPROACH_DISTANCE:
                self._approach_hold += 1
            else:
                self._approach_hold = 0

            if self._approach_hold >= self.APPROACH_HOLD_STEPS:
                self._stage = 1
                self._open_hold = 0
                # Opening before arrival is not credited.  Require a clear
                # closed state before accepting the post-arrival opening.
                self._door_rearmed = (
                    self._door_open_fraction() <= self.REARM_CLOSED_FRACTION
                )
            # self._log_success_conditions()  # periodic debug output disabled
            return ret

        door_fraction = self._door_open_fraction()
        if door_fraction <= self.REARM_CLOSED_FRACTION:
            self._door_rearmed = True

        if (
            self._door_rearmed
            and self.approach_distance() <= self.APPROACH_DISTANCE
            and door_fraction >= self.OPEN_FRACTION
        ):
            self._open_hold += 1
        else:
            self._open_hold = 0
        # self._log_success_conditions()  # periodic debug output disabled
        return ret

    def _log_success_conditions(self):
        """Print all success predicates every five simulated seconds."""
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

        approach_distance = self.approach_distance()
        approach_ok = approach_distance <= self.APPROACH_DISTANCE
        door_fraction = self._door_open_fraction()
        door_open_ok = door_fraction >= self.OPEN_FRACTION
        hold_ok = self._open_hold >= self._open_hold_steps
        task_success = bool(self._check_success())
        print(
            f"[{type(self).__name__}][scene_id={self.scene_id}] "
            f"t={sim_time:.2f}s stage={self._stage}/1; "
            f"approach(distance={approach_distance:.3f}m "
            f"<={self.APPROACH_DISTANCE:.3f}m="
            f"{'OK' if approach_ok else 'MISSING'}, "
            f"hold={self._approach_hold}/{self.APPROACH_HOLD_STEPS}); "
            f"door(open_fraction={door_fraction:.3f} "
            f">={self.OPEN_FRACTION:.2f}="
            f"{'OK' if door_open_ok else 'MISSING'}, "
            f"rearmed={'OK' if self._door_rearmed else 'MISSING'}, "
            f"hold={self._open_hold}/{self._open_hold_steps}="
            f"{'OK' if hold_ok else 'MISSING'}); "
            f"success={'OK' if task_success else 'MISSING'}",
            flush=True,
        )

    def _check_success(self):
        return bool(
            self._stage == 1
            and self._door_rearmed
            and self._open_hold >= self._open_hold_steps
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            {
                "lang": (
                    "walk to the fridge, then open the fridge door"
                ),
                "stage_order": ["approach_fridge", "open_fridge_door"],
                "composite_stages": ["approach_fridge", "open_fridge_door"],
                "scene_id": self.scene_id,
                "robot_base_yaw": self.ROBOT_BASE_YAW,
                "approach_distance_m": self.APPROACH_DISTANCE,
                "door_open_fraction": self.OPEN_FRACTION,
                "door_open_hold_seconds": self.OPEN_HOLD_SECONDS,
            }
        )
        return ep_meta


__all__ = ["TeleLabApproachOpenFridge"]
