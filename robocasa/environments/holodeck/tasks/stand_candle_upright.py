"""Stand up the outer native candle on scene 2025's far side table."""

import math

import mujoco
import numpy as np

from robocasa.environments.holodeck.holodeck import (
    HolodeckEnv,
    normalize_holodeck_scene_name,
)
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckStandCandleUpright(HolodeckEnv):
    """Keep the designated candle upright on its original tabletop for one second."""

    ATOMIC_SKILL = "PlaceObject"
    TASK_FAMILY = "place"
    SUPPORTED_SCENE_IDS = (2025,)
    SPEC_TASK = "stand_candle_upright"
    TARGET_BODY = "candle_c6e5d947796037dc5ccc452b1320f772_1_0_0"
    HOLD_SECONDS = 1.0
    UPRIGHT_TOLERANCE_DEGREES = 10.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    USE_DEFAULT_ASSET_LAYOUT = False
    # Remove the intersecting desk laptop from this task's loaded scene only.
    HIDDEN_SCENE_BODIES = {
        "train_2025": ("laptop_63fbcd680e0db93dfb0454951f2114f3_1_0_0",),
    }
    NOISE_FREE_ROBOT_SPAWN = True
    LANG = (
        "Stand up the fallen candle on the side table, "
        "and keep it upright on that tabletop."
    )

    def __init__(self, *args, scene_ids=None, **kwargs):
        ids = self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids
        ids = [ids] if isinstance(ids, (int, str)) else list(ids)
        if not ids or any(
            normalize_holodeck_scene_name(i) != "train_2025" for i in ids
        ):
            raise ValueError("HolodeckStandCandleUpright only supports scene 2025")
        self._success_hold = 0
        super().__init__(*args, scene_ids=ids, **kwargs)

    def _setup_baked_scene_fixtures(self):
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=("table_body", "tabletop_geom", "candle_pose", "robot_pose"),
        )
        candle = self.find_scene_body(self.TARGET_BODY)
        if candle is None or self.find_scene_body(self.task_spec["table_body"]) is None:
            raise RuntimeError(f"{self.scene_name}: missing candle or its side table")
        # Preserve the native model, mesh scale and physics. Only this task's
        # loaded XML copy starts the specified candle lying parallel to the edge.
        pose = self.task_spec["candle_pose"]
        candle.set("pos", " ".join(map(str, pose["pos"])))
        candle.set("quat", " ".join(map(str, pose["quat"])))
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        pose = self.task_spec["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        robot_model = self.robots[0].robot_model
        if type(robot_model).__name__ == "G1":
            pos += robot_model.bottom_offset
        self.set_robot_base_pose(pos, pose["yaw"])

    def _setup_references(self):
        super()._setup_references()
        model = self.sim.model
        self._candle_body_id = model.body_name2id(self.TARGET_BODY)
        candle_ids = self._body_tree_ids([self._candle_body_id])
        self._candle_geoms = self._geom_names_for_bodies(candle_ids)
        joint_id = int(model.body_jntadr[self._candle_body_id])
        if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise RuntimeError("target candle must have a free joint")
        self._candle_joint = model.joint_id2name(joint_id)
        self._tabletop_geom_id = model.geom_name2id(self.task_spec["tabletop_geom"])

    def _reset_internal(self):
        self._success_hold = 0
        super()._reset_internal()
        pose = self.task_spec["candle_pose"]
        self.sim.data.set_joint_qpos(
            self._candle_joint, np.array([*pose["pos"], *pose["quat"]])
        )
        self.sim.data.set_joint_qvel(self._candle_joint, np.zeros(6))
        self.sim.forward()
        base_xy = self._robot_base_xy()
        samples = [base_xy]
        samples.extend(
            base_xy + 0.25 * np.array([np.cos(a), np.sin(a)])
            for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)
        )
        if any(self.room_of_point([*xy, 0.0]) is None for xy in samples):
            raise RuntimeError(
                f"{self.scene_name}: candle task robot starts off the floor"
            )
        if self.count_body_collisions():
            raise RuntimeError(
                f"{self.scene_name}: candle task has a robot spawn collision"
            )

    def _goal_satisfied(self):
        # The native candle's +Y axis points from its base towards its top.
        rotation = np.asarray(self.sim.data.body_xmat[self._candle_body_id]).reshape(
            3, 3
        )
        upright = rotation[2, 1] >= math.cos(
            math.radians(self.UPRIGHT_TOLERANCE_DEGREES)
        )
        return bool(
            upright
            and self.check_contact(
                [self.task_spec["tabletop_geom"]], self._candle_geoms
            )
        )

    @property
    def hold_steps(self):
        return max(1, int(math.ceil(self.HOLD_SECONDS * self.control_freq)))

    def _post_action(self, action):
        self._success_hold = self._success_hold + 1 if self._goal_satisfied() else 0
        return super()._post_action(action)

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=self.LANG,
            target_body=self.TARGET_BODY,
            table_body=self.task_spec["table_body"],
            tabletop_geom=self.task_spec["tabletop_geom"],
            object_model="RoboTHOR_candle_glittrig_3_v",
            object_scale=1.0,
            # Retain this native candle's 9.90 cm short-axis grasp span so
            # its identity and shared scene asset remain unchanged.
            grasp_span=0.0990028,
            upright_tolerance_degrees=self.UPRIGHT_TOLERANCE_DEGREES,
            hold_seconds=self.HOLD_SECONDS,
            success_definition=(
                "the designated candle's +Y axis stays within 10 degrees of world +Z "
                "while contacting the original side tabletop continuously for 1 second"
            ),
        )
        return meta


__all__ = ["HolodeckStandCandleUpright"]
