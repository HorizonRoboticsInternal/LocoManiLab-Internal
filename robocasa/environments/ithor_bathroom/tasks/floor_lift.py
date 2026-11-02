"""Lift a graspable bathroom candle from a fixed floor-standing stool."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robocasa.environments.ithor_atomic.pick_place import (
    SupportContactPlacementMixin,
)
from robocasa.environments.ithor_atomic.task_assets import (
    CANDLE_MODEL,
    CANDLE_SCALE,
    MAX_GRASP_SPAN,
    MIN_GRASP_SPAN,
    asset_xml_path,
)
from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.models.fixtures import Fixture
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_specs,
    get_supported_scene_ids,
)
from robocasa.models.scenes.thor_semantics import SemanticFixture

from .object_manipulation import _NativeBathroomFixtureMixin


class _BathroomCandleStoolTask(
    _NativeBathroomFixtureMixin,
    SupportContactPlacementMixin,
    SceneSpecMixin,
):
    """Grasp a pillar candle from a low bathroom stool and lift it 30 cm."""

    TARGET_NAME = "candle"
    TARGET_XML = asset_xml_path(CANDLE_MODEL)
    TARGET_SCALE = CANDLE_SCALE
    STOOL_NAME = "bathroom_stool"
    STOOL_MODEL = "/objects/lightwheel/stool/Stool006/model.xml"
    STOOL_SCALE = 0.55
    STOOL_BASE_WELD_NAME = "task_bathroom_candle_stool_fixed_base"
    HOLD_STEPS = 5
    MIN_LIFT_DELTA = 0.30
    GRASP_HEIGHT_ALLOWANCE = 0.10
    SUPPORT_PROBE_STEP = 0.0005
    SUPPORT_PENETRATION_TOLERANCE = 0.0025
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0

    def _get_obj_cfgs(self):
        return [
            {
                "name": self.TARGET_NAME,
                "obj_groups": self.TARGET_XML,
                "object_scale": self.TARGET_SCALE,
                "placement": {"rotation": 0.0},
            }
        ]

    def _validated_stool_layout(self):
        spec = self.scene_task_spec()
        source_xy = np.asarray(spec["source_xy"], dtype=float)
        source_quat = np.asarray(spec["source_quat"], dtype=float)
        stool_xy = np.asarray(spec["stool_xy"], dtype=float)
        stool_yaw = float(spec["stool_yaw"])
        stool_top_half_xy = np.asarray(spec["stool_top_half_xy"], dtype=float)
        robot_pose = spec["robot_pose"]
        source_grasp_span = float(spec["source_grasp_span"])
        if (
            source_xy.shape != (2,)
            or source_quat.shape != (4,)
            or stool_xy.shape != (2,)
            or stool_top_half_xy.shape != (2,)
            or not np.all(np.isfinite(source_xy))
            or not np.all(np.isfinite(source_quat))
            or not np.all(np.isfinite(stool_xy))
            or not np.isfinite(stool_yaw)
            or not np.all(np.isfinite(stool_top_half_xy))
            or np.any(stool_top_half_xy <= 0.0)
            or not np.isclose(np.linalg.norm(source_quat), 1.0, atol=1e-6)
            or not MIN_GRASP_SPAN < source_grasp_span <= MAX_GRASP_SPAN
            or spec["stool_model"] != self.STOOL_MODEL
            or not np.isclose(float(spec["stool_scale"]), self.STOOL_SCALE)
            or not np.isfinite(float(spec["stool_floor_probe_above"]))
            or not np.isfinite(float(spec["stool_floor_probe_below"]))
            or float(spec["stool_floor_probe_above"]) < 0.0
            or float(spec["stool_floor_probe_below"]) < 0.0
            or not isinstance(robot_pose, dict)
            or set(robot_pose) != {"xy", "yaw"}
        ):
            raise ValueError(f"{self.scene_name}: invalid candle-stool layout")
        robot_xy = np.asarray(robot_pose["xy"], dtype=float)
        robot_yaw = float(robot_pose["yaw"])
        if (
            robot_xy.shape != (2,)
            or not np.all(np.isfinite(robot_xy))
            or not np.isfinite(robot_yaw)
        ):
            raise ValueError(f"{self.scene_name}: invalid candle-stool robot pose")
        return (
            source_xy,
            source_quat,
            stool_xy,
            stool_yaw,
            stool_top_half_xy,
            robot_xy,
            robot_yaw,
        )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        (
            _,
            _,
            stool_xy,
            stool_yaw,
            _,
            robot_xy,
            robot_yaw,
        ) = self._validated_stool_layout()
        spec = self.scene_task_spec()
        self._removed_fixtures = self._remove_exact_task_fixtures(
            spec["remove_fixtures"]
        )
        self._removed_fixture_ids = tuple(spec["remove_fixtures"])
        self.task_stool = Fixture(
            xml=asset_xml_path(self.STOOL_MODEL),
            name=self.STOOL_NAME,
            duplicate_collision_geoms=False,
            scale=self.STOOL_SCALE,
            rng=self.rng,
            joints=[dict(type="free", damping="0.001")],
        )
        initial_z = -float(self.task_stool.bottom_offset[2])
        self.task_stool.set_pos([float(stool_xy[0]), float(stool_xy[1]), initial_z])
        self.task_stool.set_euler([0.0, 0.0, stool_yaw])
        self.furniture[self.STOOL_NAME] = self.task_stool
        bbox = self.task_stool._regions["bbox"]
        lower = np.asarray(bbox["p0"], dtype=float)
        upper = np.array(
            [bbox["px"][0], bbox["py"][1], bbox["pz"][2]],
            dtype=float,
        )
        local_center = 0.5 * (lower + upper)
        local_half_size = 0.5 * (upper - lower)
        rotation = np.array(
            [
                [np.cos(stool_yaw), -np.sin(stool_yaw), 0.0],
                [np.sin(stool_yaw), np.cos(stool_yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.task_stool_fixture = SemanticFixture(
            stable_id=f"{self.scene_name}:spawned:{self.STOOL_NAME}",
            name=self.STOOL_NAME,
            category="stool",
            source_category="Stool006",
            room_id=None,
            body_name=self.task_stool.root_body,
            body_names=[
                body.get("name")
                for body in self.task_stool.worldbody.iter("body")
                if body.get("name")
            ],
            site_names=[],
            asset_id="Stool006",
            object_id=self.STOOL_NAME,
            center=np.r_[stool_xy, initial_z] + rotation @ local_center,
            half_size=local_half_size,
            yaw=stool_yaw,
            front=rotation[:2, 0],
            local_center=local_center,
            local_half_size=local_half_size,
        )
        ET.SubElement(
            self.mujoco_arena.equality,
            "weld",
            name=self.STOOL_BASE_WELD_NAME,
            body1=self.task_stool.root_body,
            active="false",
            solref="0.002 1",
            solimp="0.9999 0.9999 0.001 0.5 2",
        )
        self.set_robot_base_pose(
            [float(robot_xy[0]), float(robot_xy[1]), 0.0],
            robot_yaw,
        )

    def _setup_scene(self):
        spec = self.scene_task_spec()
        self._set_stool_base_weld_active(False)
        if not self._place_fixture_on_floor(
            self.task_stool_fixture,
            spec["stool_xy"],
            contract_name="bathroom_stool_floor_contact",
            probe_above=float(spec["stool_floor_probe_above"]),
            probe_below=float(spec["stool_floor_probe_below"]),
            probe_step=self.SUPPORT_PROBE_STEP,
        ):
            raise RuntimeError(
                f"{self.scene_name}: bathroom stool could not contact Floor"
            )
        self._lock_stool_base_at_current_pose()
        super()._setup_scene()
        self._set_exact_robot_pose(
            spec["robot_pose"],
            "Bathroom candle-stool task",
        )

    def _stool_base_weld_id(self):
        equality_id = mujoco.mj_name2id(
            self.sim.model._model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            self.STOOL_BASE_WELD_NAME,
        )
        if equality_id < 0:
            raise RuntimeError(f"{self.scene_name}: missing bathroom stool weld")
        return equality_id

    def _set_stool_base_weld_active(self, active):
        self.sim.data.eq_active[self._stool_base_weld_id()] = int(bool(active))
        self.sim.forward()

    def _lock_stool_base_at_current_pose(self):
        equality_id = self._stool_base_weld_id()
        body_id = self.sim.model.body_name2id(self.task_stool.root_body)
        position = np.asarray(self.sim.data.body_xpos[body_id], dtype=float).copy()
        rotation = np.asarray(self.sim.data.body_xmat[body_id], dtype=float).reshape(
            3, 3
        )
        quaternion = np.asarray(self.sim.data.body_xquat[body_id], dtype=float).copy()
        self.sim.model.eq_data[equality_id, 3:6] = -rotation.T @ position
        self.sim.model.eq_data[equality_id, 6:10] = np.array(
            [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
            dtype=float,
        )
        self._zero_fixture_velocity(self.task_stool_fixture)
        self.sim.data.eq_active[equality_id] = 1
        self.sim.forward()
        self._fixed_stool_base_position = position

    def _candle_footprint_is_on_stool(self, stool_top_half_xy):
        body_id = self.sim.model.body_name2id(self.task_stool.root_body)
        center = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
        rotation = np.asarray(self.sim.data.body_xmat[body_id], dtype=float).reshape(
            3, 3
        )
        local = (self.object_bbox_points(self.TARGET_NAME) - center) @ rotation
        return bool(
            np.all(
                np.abs(local[:, :2])
                <= np.asarray(stool_top_half_xy, dtype=float) + 1e-6
            )
        )

    def _place_candle_on_stool(
        self,
        source_xy,
        source_quat,
        stool_top_half_xy,
    ):
        obj = self.objects[self.TARGET_NAME]
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[3:7] = source_quat
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.forward()
        stool_bodies = self._fixture_body_ids(
            self.task_stool_fixture,
            include_siblings=True,
        )
        stool_top_z = float(
            np.max(self._fixture_bbox_points(self.task_stool_fixture)[:, 2])
        )
        for bottom_z in np.arange(
            stool_top_z + 0.03,
            stool_top_z - 0.02 - 0.5 * self.SUPPORT_PROBE_STEP,
            -self.SUPPORT_PROBE_STEP,
        ):
            self._set_task_object_bottom_at(
                self.TARGET_NAME,
                source_xy,
                bottom_z,
            )
            if not self._candle_footprint_is_on_stool(stool_top_half_xy):
                continue
            if not self._object_fixture_contact_is_shallow(
                self.TARGET_NAME,
                self.task_stool_fixture,
                self.SUPPORT_PENETRATION_TOLERANCE,
            ):
                continue
            if self.illegal_contact_pairs_for_bodies(
                [self.obj_body_id[self.TARGET_NAME]],
                tuple(stool_bodies),
                tolerance=self.SUPPORT_PENETRATION_TOLERANCE,
            ):
                continue
            self.register_task_body_contact_contract(
                "bathroom_stool_candle_contact",
                self.obj_body_id[self.TARGET_NAME],
                tuple(stool_bodies),
            )
            return True
        return False

    def _source_root_z(self):
        root = self.objects[self.TARGET_NAME].root_body
        return float(self.sim.data.get_body_xpos(root)[2])

    def _robot_facing_target_xy(self, source_xy):
        """Return the point the reset pose must face; atoms face the source."""
        return source_xy

    def _reset_internal(self):
        super()._reset_internal()
        (
            source_xy,
            source_quat,
            _,
            _,
            stool_top_half_xy,
            robot_xy,
            robot_yaw,
        ) = self._validated_stool_layout()
        if not self._place_candle_on_stool(
            source_xy,
            source_quat,
            stool_top_half_xy,
        ):
            raise RuntimeError(
                f"{self.scene_name}: no stable stool-contact candle pose"
            )
        if self.count_body_collisions() != 0:
            raise RuntimeError(
                f"{self.scene_name}: candle-stool staging collides with G1"
            )

        root_xy = np.asarray(
            self.sim.data.get_body_xpos(self.objects[self.TARGET_NAME].root_body)[:2],
            dtype=float,
        )
        self._source_horizontal_distance = float(np.linalg.norm(root_xy - robot_xy))
        expected_distance = float(self.scene_task_spec()["source_horizontal_distance"])
        if not np.isclose(
            self._source_horizontal_distance,
            expected_distance,
            atol=1e-4,
        ):
            raise RuntimeError(
                f"{self.scene_name}: candle distance is "
                f"{self._source_horizontal_distance:.4f} m, expected "
                f"{expected_distance:.4f} m"
            )
        facing_xy = np.asarray(self._robot_facing_target_xy(root_xy), dtype=float)
        desired_yaw = float(
            np.arctan2(facing_xy[1] - robot_xy[1], facing_xy[0] - robot_xy[0])
        )
        heading_error = float(
            np.arctan2(
                np.sin(robot_yaw - desired_yaw),
                np.cos(robot_yaw - desired_yaw),
            )
        )
        if abs(np.rad2deg(heading_error)) > 1.0:
            raise RuntimeError(
                f"{self.scene_name}: robot does not face the stool candle"
            )

        horizontal_spans = np.ptp(
            self.object_bbox_points(self.TARGET_NAME)[:, :2],
            axis=0,
        )
        self._source_grasp_span = float(np.min(horizontal_spans))
        expected_span = float(self.scene_task_spec()["source_grasp_span"])
        if not np.isclose(self._source_grasp_span, expected_span, atol=1e-4):
            raise RuntimeError(
                f"{self.scene_name}: candle grasp span is "
                f"{self._source_grasp_span:.4f} m, expected "
                f"{expected_span:.4f} m"
            )

        self._initial_source_z = self._source_root_z()
        self._grasped_near_source = False
        self._success_hold = 0

    def _raw_success(self):
        grasped = self.g1_hand_grasping_object(self.TARGET_NAME)
        height = self._source_root_z()
        if grasped and height <= self._initial_source_z + self.GRASP_HEIGHT_ALLOWANCE:
            self._grasped_near_source = True
        return bool(
            self._grasped_near_source
            and grasped
            and height >= self._initial_source_z + self.MIN_LIFT_DELTA
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold = self._success_hold + 1 if self._raw_success() else 0
        return ret

    def _check_success(self):
        return self._success_hold >= self.HOLD_STEPS

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        spec = self.scene_task_spec()
        ep_meta.update(
            {
                "lang": spec["lang"],
                "source_surface": "fixed floor-standing bathroom stool",
                "source_model": CANDLE_MODEL,
                "source_scale": float(self.TARGET_SCALE),
                "stool_model": self.STOOL_MODEL,
                "stool_scale": self.STOOL_SCALE,
                "stool_xy": list(spec["stool_xy"]),
                "source_grasp_span": float(self._source_grasp_span),
                "source_horizontal_distance": float(self._source_horizontal_distance),
                "removed_fixtures": list(self._removed_fixture_ids),
                "success_definition": (
                    "the pillar candle must first be grasped at its reset "
                    "stool height, then remain grasped at least "
                    f"{self.MIN_LIFT_DELTA:.2f} m higher for "
                    "1.0 s"
                ),
            }
        )
        return ep_meta


class IThorBathroomLiftCandleFromFloor(_BathroomCandleStoolTask, IThorBathroomEnv):
    """Grasp a pillar candle from a low bathroom stool and lift it 30 cm."""

    ATOMIC_SKILL = "LiftObjectFromFloor"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "lift"
    SPEC_TASK = "lift_candle_from_floor"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK,
        required=(
            "source_xy",
            "source_quat",
            "source_grasp_span",
            "stool_model",
            "stool_scale",
            "stool_xy",
            "stool_yaw",
            "stool_top_half_xy",
            "stool_floor_probe_above",
            "stool_floor_probe_below",
            "remove_fixtures",
            "robot_pose",
            "source_horizontal_distance",
            "lang",
        ),
    )


__all__ = ["IThorBathroomLiftCandleFromFloor"]
