"""Lift a basket from a calibrated Bedroom support or low stool."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robocasa.environments.ithor_atomic.free_fixture import (
    FreeFixturePlacementMixin,
)
from robocasa.environments.ithor_atomic.lift import RoomLiftMixin
from robocasa.environments.ithor_atomic.pick_place import (
    SupportContactPlacementMixin,
    resolve_task_support_region,
)
from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.models.fixtures import Fixture
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_spec,
    get_ithor_bedroom_task_specs,
    get_supported_scene_ids,
)
from robocasa.models.scenes.thor_semantics import SemanticFixture

from .assets import (
    LIFT_BASKET_MODEL,
    LIFT_BASKET_SCALE,
    asset_xml_path,
)
from .surface_placement import (
    align_object_long_axis_parallel_to_surface_edge,
)


def _reset_basket_lift_progress(task):
    task._lift_initial_z = task._target_root_z()
    task._grasped_at_source_height = False
    task._bedroom_lift_hold = 0


def _advance_basket_lift_progress(task):
    height = task._target_root_z()
    grasped = task.g1_hand_grasping_object(task.TARGET_NAME)
    if grasped and height <= task._lift_initial_z + task.GRASP_HEIGHT_ALLOWANCE:
        task._grasped_at_source_height = True
    valid = bool(
        task._grasped_at_source_height
        and grasped
        and height >= task._lift_initial_z + task.MIN_LIFT_DELTA
    )
    task._bedroom_lift_hold = task._bedroom_lift_hold + 1 if valid else 0


def _basket_lift_success_definition(task):
    return (
        "basket must first be grasped near its source height, then remain "
        f"grasped at least {task.MIN_LIFT_DELTA:.2f} m higher for "
        f"{task.HOLD_STEPS} control steps"
    )


class IThorBedroomLiftBasket(
    SupportContactPlacementMixin,
    SceneSpecMixin,
    RoomLiftMixin,
    IThorBedroomEnv,
):
    """Grasp the named basket at table height, then lift it while held."""

    SPEC_TASK = "lift_basket"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK, required=("fixture", "region", "lang")
    )
    TARGET_NAME = "basket"
    TARGET_XML = asset_xml_path(LIFT_BASKET_MODEL)
    TARGET_SCALE = LIFT_BASKET_SCALE
    MIN_LIFT_DELTA = 0.30
    PLACEMENT_MARGIN = 0.01
    HOLD_STEPS = 5
    GRASP_HEIGHT_ALLOWANCE = 0.10

    def _get_obj_cfgs(self):
        return [
            {
                "name": self.TARGET_NAME,
                "obj_groups": self.TARGET_XML,
                "object_scale": self.TARGET_SCALE,
                "placement": {
                    "region": getattr(self, "lift_region", None),
                    "margin": self.PLACEMENT_MARGIN,
                    "rotation": 0.0,
                    "strategy": "edge",
                },
            }
        ]

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = get_ithor_bedroom_task_spec(
            self.scene_name, self.SPEC_TASK, required=("fixture", "region")
        )
        self.lift_fixture = self.get_fixture(id=spec["fixture"])
        regions = []
        if self.lift_fixture is not None:
            regions.extend(
                self.get_region(
                    fixture=self.lift_fixture,
                    kind="support",
                    return_all=True,
                )
            )
            regions.extend(
                self.get_region(
                    fixture=self.lift_fixture,
                    kind="receptacle",
                    return_all=True,
                )
            )
        raw_region = next(
            (region for region in regions if region.stable_id == spec["region"]),
            None,
        )
        self.lift_region = (
            resolve_task_support_region(raw_region)
            if raw_region is not None
            else None
        )
        removal_ids = tuple(spec.get("remove_fixtures", ()))
        removals = [self.get_fixture(id=stable_id) for stable_id in removal_ids]
        missing = [
            stable_id
            for stable_id, fixture in zip(removal_ids, removals)
            if fixture is None
        ]
        if missing:
            raise ValueError(
                f"{self.scene_name}: unknown exact lift obstructions {missing}"
            )
        self.removed_lift_obstruction_ids = list(removal_ids)
        if removals:
            self.remove_semantic_fixtures(removals)
        if self.lift_region is None:
            return
        self.fixture_refs["lift_surface"] = self.lift_fixture
        self.region_refs["lift_surface"] = self.lift_region

    def _reset_internal(self):
        super()._reset_internal()
        align_object_long_axis_parallel_to_surface_edge(
            self, self.TARGET_NAME, self.lift_region
        )
        self.sim.forward()
        basket_xy, _ = self._lift_target_pose()
        if not self.place_object_on_support_contact(
            self.TARGET_NAME,
            self.lift_region,
            fixture=self.lift_fixture,
            xy=basket_xy,
        ):
            raise RuntimeError(
                f"{self.scene_name}: no contacted basket pose on the named "
                "lift surface"
            )
        self._apply_calibrated_source_pose()
        _reset_basket_lift_progress(self)

    def _record_initial_manipulation_layout(self, calibrated_distance=None):
        source_xy = np.asarray(
            self.sim.data.get_body_xpos(
                self.objects[self.TARGET_NAME].root_body
            )[:2],
            dtype=float,
        )
        base_xy = np.asarray(self._robot_base_xy(), dtype=float)
        actual = float(np.linalg.norm(base_xy - source_xy))
        clearance = float(
            self.robot_fixture_collision_distance(
                self.lift_fixture,
                max_distance=2.0,
            )
        )
        self._source_horizontal_distance = actual
        self._robot_lift_surface_clearance = clearance
        if calibrated_distance is None:
            return

        calibrated_distance = float(calibrated_distance)
        failures = []
        if not self.robot_footprint_is_on_floor(base_xy, radius=0.25):
            failures.append("robot footprint leaves Floor")
        collisions = self.count_body_collisions()
        if collisions:
            failures.append(f"robot has {collisions} scene collisions")
        if not np.isfinite(clearance) or clearance <= 0.0:
            failures.append("robot touches the named lift surface")
        yaw = self._robot_base_yaw()
        forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
        if float(np.dot(source_xy - base_xy, forward)) <= 0.0:
            failures.append("basket is not in front of the robot")
        if failures:
            raise RuntimeError(
                f"{self.scene_name}: invalid fixed-base basket layout: "
                + "; ".join(failures)
            )
        self._calibrated_source_horizontal_distance = calibrated_distance
        self._fixed_base_static_contract_verified = True

    def _apply_calibrated_source_pose(self):
        spec = self.scene_task_spec()
        desired = spec.get("source_horizontal_distance")
        if desired is None:
            self._record_initial_manipulation_layout()
            return
        desired = float(desired)
        source_xy = np.asarray(
            self.sim.data.get_body_xpos(
                self.objects[self.TARGET_NAME].root_body
            )[:2],
            dtype=float,
        )
        current_xy = np.asarray(self._robot_base_xy(), dtype=float)
        outward = current_xy - source_xy
        norm = float(np.linalg.norm(outward))
        if norm <= 1e-8:
            raise RuntimeError(
                f"{self.scene_name}: robot overlaps {self.TARGET_NAME}"
            )
        target_xy = source_xy + outward * (desired / norm)
        yaw = float(
            np.arctan2(
                source_xy[1] - target_xy[1],
                source_xy[0] - target_xy[0],
            )
        )
        if hasattr(self, "_robot_free_joint_z"):
            delattr(self, "_robot_free_joint_z")
        valid = bool(
            self.robot_footprint_is_on_floor(target_xy, radius=0.25)
            and self._set_robot_free_pose(target_xy, yaw)
            and self._calibrate_robot_free_joint_height()
        )
        self.sim.forward()
        if not valid:
            raise RuntimeError(
                f"{self.scene_name}: cannot apply the scene-local "
                f"{desired:.2f} m basket calibration"
            )
        self.robot_base_pos = np.array([target_xy[0], target_xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        self._record_initial_manipulation_layout(
            calibrated_distance=desired
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        _advance_basket_lift_progress(self)
        return ret

    def _check_success(self):
        return self._bedroom_lift_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.scene_task_spec()["lang"]
        ep_meta["source_surface"] = self.lift_region.stable_id
        ep_meta["removed_lift_obstructions"] = list(
            self.removed_lift_obstruction_ids
        )
        if hasattr(self, "_source_horizontal_distance"):
            ep_meta["source_horizontal_distance"] = float(
                self._source_horizontal_distance
            )
        if hasattr(self, "_calibrated_source_horizontal_distance"):
            ep_meta["calibrated_source_horizontal_distance"] = float(
                self._calibrated_source_horizontal_distance
            )
        if hasattr(self, "_robot_lift_surface_clearance"):
            ep_meta["robot_work_surface_clearance"] = float(
                self._robot_lift_surface_clearance
            )
        if hasattr(self, "_fixed_base_static_contract_verified"):
            ep_meta["static_contract_verified"] = bool(
                self._fixed_base_static_contract_verified
            )
        ep_meta["success_definition"] = _basket_lift_success_definition(self)
        return ep_meta


class IThorBedroomLiftBasketFromFloor(
    FreeFixturePlacementMixin,
    SupportContactPlacementMixin,
    SceneSpecMixin,
    IThorBedroomEnv,
):
    """Bend down and lift Basket039 from a fixed low bedroom stool."""

    ATOMIC_SKILL = "LiftObjectFromFloor"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "lift"
    SPEC_TASK = "lift_basket_from_floor"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK,
        required=(
            "basket_xy",
            "basket_quat",
            "stool_model",
            "stool_scale",
            "stool_xy",
            "stool_yaw",
            "stool_top_half_xy",
            "stool_floor_probe_below",
            "robot_pose",
            "staging_robot_pose",
            "source_horizontal_distance",
            "remove_fixtures",
            "lang",
        ),
    )
    TARGET_NAME = "basket"
    TARGET_XML = asset_xml_path(LIFT_BASKET_MODEL)
    TARGET_SCALE = LIFT_BASKET_SCALE
    STOOL_NAME = "bedroom_basket_stool"
    STOOL_MODEL = "/objects/lightwheel/stool/Stool005/model.xml"
    STOOL_SCALE = (1.15, 0.90, 0.30)
    STOOL_BASE_WELD_NAME = "task_bedroom_basket_stool_fixed_base"
    MIN_LIFT_DELTA = 0.30
    HOLD_STEPS = 5
    GRASP_HEIGHT_ALLOWANCE = 0.10
    SUPPORT_PROBE_ABOVE = 0.02
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
        basket_xy = np.asarray(spec["basket_xy"], dtype=float)
        basket_quat = np.asarray(spec["basket_quat"], dtype=float)
        stool_xy = np.asarray(spec["stool_xy"], dtype=float)
        stool_yaw = float(spec["stool_yaw"])
        stool_top_half_xy = np.asarray(spec["stool_top_half_xy"], dtype=float)
        robot_pose = spec["robot_pose"]
        staging_robot_pose = spec["staging_robot_pose"]
        source_horizontal_distance = float(spec["source_horizontal_distance"])
        if (
            basket_xy.shape != (2,)
            or basket_quat.shape != (4,)
            or stool_xy.shape != (2,)
            or stool_top_half_xy.shape != (2,)
            or not np.all(np.isfinite(basket_xy))
            or not np.all(np.isfinite(basket_quat))
            or not np.all(np.isfinite(stool_xy))
            or not np.isfinite(stool_yaw)
            or not np.all(np.isfinite(stool_top_half_xy))
            or np.any(stool_top_half_xy <= 0.0)
            or not np.isclose(np.linalg.norm(basket_quat), 1.0, atol=1e-6)
            or spec["stool_model"] != self.STOOL_MODEL
            or tuple(float(value) for value in spec["stool_scale"])
            != self.STOOL_SCALE
            or float(spec["stool_floor_probe_below"]) < 0.0
            or not np.isfinite(source_horizontal_distance)
            or source_horizontal_distance <= 0.0
            or not isinstance(robot_pose, dict)
            or set(robot_pose) != {"xy", "yaw"}
            or not isinstance(staging_robot_pose, dict)
            or set(staging_robot_pose) != {"xy", "yaw"}
        ):
            raise ValueError(f"{self.scene_name}: invalid basket-stool layout")
        robot_xy = np.asarray(robot_pose["xy"], dtype=float)
        robot_yaw = float(robot_pose["yaw"])
        staging_robot_xy = np.asarray(staging_robot_pose["xy"], dtype=float)
        staging_robot_yaw = float(staging_robot_pose["yaw"])
        if (
            robot_xy.shape != (2,)
            or not np.all(np.isfinite(robot_xy))
            or not np.isfinite(robot_yaw)
            or staging_robot_xy.shape != (2,)
            or not np.all(np.isfinite(staging_robot_xy))
            or not np.isfinite(staging_robot_yaw)
        ):
            raise ValueError(
                f"{self.scene_name}: invalid basket-stool robot pose"
            )
        removal_ids = spec["remove_fixtures"]
        if not isinstance(removal_ids, (list, tuple)) or isinstance(
            removal_ids, str
        ):
            raise ValueError(f"{self.scene_name}: remove_fixtures must be a list")
        removal_ids = tuple(removal_ids)
        if len(removal_ids) != len(set(removal_ids)):
            raise ValueError(
                f"{self.scene_name}: duplicate basket-stool removal"
            )
        return (
            basket_xy,
            basket_quat,
            stool_xy,
            stool_yaw,
            stool_top_half_xy,
            robot_xy,
            robot_yaw,
            staging_robot_xy,
            staging_robot_yaw,
            removal_ids,
        )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            staging_robot_xy,
            staging_robot_yaw,
            removal_ids,
        ) = self._validated_stool_layout()
        removals = []
        for stable_id in removal_ids:
            fixture = self.get_fixture(id=stable_id)
            if fixture is None or fixture.stable_id != stable_id:
                raise LookupError(
                    f"{self.scene_name}: missing floor basket obstruction "
                    f"{stable_id}"
                )
            removals.append(fixture)
        if removals:
            self.remove_semantic_fixtures(removals)
        self.removed_floor_obstruction_ids = list(removal_ids)
        spec = self.scene_task_spec()
        stool_xy = np.asarray(spec["stool_xy"], dtype=float)
        stool_yaw = float(spec["stool_yaw"])
        self.task_stool = Fixture(
            xml=asset_xml_path(self.STOOL_MODEL),
            name=self.STOOL_NAME,
            duplicate_collision_geoms=False,
            scale=self.STOOL_SCALE,
            rng=self.rng,
            joints=[dict(type="free", damping="0.001")],
        )
        initial_z = -float(self.task_stool.bottom_offset[2])
        self.task_stool.set_pos(
            [float(stool_xy[0]), float(stool_xy[1]), initial_z]
        )
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
            ],
            dtype=float,
        )
        self.task_stool_fixture = SemanticFixture(
            stable_id=f"{self.scene_name}:spawned:{self.STOOL_NAME}",
            name=self.STOOL_NAME,
            category="stool",
            source_category="Stool005",
            room_id=None,
            body_name=self.task_stool.root_body,
            body_names=[
                body.get("name")
                for body in self.task_stool.worldbody.iter("body")
                if body.get("name")
            ],
            site_names=[],
            asset_id="Stool005",
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
            [staging_robot_xy[0], staging_robot_xy[1], 0.0],
            staging_robot_yaw,
        )

    def _apply_floor_basket_robot_pose(self, robot_xy, robot_yaw, label):
        if not self.robot_footprint_is_on_floor(robot_xy, radius=0.25):
            raise RuntimeError(
                f"{self.scene_name}: {label} robot footprint leaves Floor"
            )
        if hasattr(self, "_robot_free_joint_z"):
            delattr(self, "_robot_free_joint_z")
        valid = bool(
            self._set_robot_free_pose(robot_xy, robot_yaw)
            and self._calibrate_robot_free_joint_height()
        )
        self.sim.forward()
        if not valid or self.count_body_collisions():
            raise RuntimeError(f"{self.scene_name}: {label} robot pose collides")
        self.robot_base_pos = np.array([robot_xy[0], robot_xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, robot_yaw])

    def _setup_scene(self):
        self._set_stool_base_weld_active(False)
        if not self._place_fixture_on_floor(
            self.task_stool_fixture,
            self.scene_task_spec()["stool_xy"],
            contract_name="bedroom_stool_floor_contact",
            probe_above=self.SUPPORT_PROBE_ABOVE,
            probe_below=float(
                self.scene_task_spec()["stool_floor_probe_below"]
            ),
            probe_step=self.SUPPORT_PROBE_STEP,
        ):
            raise RuntimeError(
                f"{self.scene_name}: bedroom stool could not contact Floor"
            )
        self._lock_stool_base_at_current_pose()
        super()._setup_scene()
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            staging_robot_xy,
            staging_robot_yaw,
            _,
        ) = self._validated_stool_layout()
        self._apply_floor_basket_robot_pose(
            staging_robot_xy,
            staging_robot_yaw,
            "basket-stool staging",
        )

    def _stool_base_weld_id(self):
        equality_id = mujoco.mj_name2id(
            self.sim.model._model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            self.STOOL_BASE_WELD_NAME,
        )
        if equality_id < 0:
            raise RuntimeError(f"{self.scene_name}: missing bedroom stool weld")
        return equality_id

    def _set_stool_base_weld_active(self, active):
        self.sim.data.eq_active[self._stool_base_weld_id()] = int(bool(active))
        self.sim.forward()

    def _lock_stool_base_at_current_pose(self):
        equality_id = self._stool_base_weld_id()
        body_id = self.sim.model.body_name2id(self.task_stool.root_body)
        position = np.asarray(self.sim.data.body_xpos[body_id], dtype=float).copy()
        rotation = np.asarray(
            self.sim.data.body_xmat[body_id], dtype=float
        ).reshape(3, 3)
        quaternion = np.asarray(
            self.sim.data.body_xquat[body_id], dtype=float
        ).copy()
        self.sim.model.eq_data[equality_id, 3:6] = -rotation.T @ position
        self.sim.model.eq_data[equality_id, 6:10] = np.array(
            [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
            dtype=float,
        )
        self._zero_fixture_velocity(self.task_stool_fixture)
        self.sim.data.eq_active[equality_id] = 1
        self.sim.forward()
        self._fixed_stool_base_position = position

    def _basket_footprint_is_on_stool(self, stool_top_half_xy):
        body_id = self.sim.model.body_name2id(self.task_stool.root_body)
        center = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
        rotation = np.asarray(
            self.sim.data.body_xmat[body_id], dtype=float
        ).reshape(3, 3)
        local = (self.object_bbox_points(self.TARGET_NAME) - center) @ rotation
        return bool(
            np.all(
                np.abs(local[:, :2])
                <= np.asarray(stool_top_half_xy, dtype=float) + 1e-6
            )
        )

    def _place_basket_on_stool(
        self,
        basket_xy,
        basket_quat,
        stool_top_half_xy,
    ):
        obj = self.objects[self.TARGET_NAME]
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[3:7] = basket_quat
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
            stool_top_z - 0.03 - 0.5 * self.SUPPORT_PROBE_STEP,
            -self.SUPPORT_PROBE_STEP,
        ):
            self._set_task_object_bottom_at(
                self.TARGET_NAME, basket_xy, bottom_z
            )
            if not self._basket_footprint_is_on_stool(stool_top_half_xy):
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
                "bedroom_stool_basket_contact",
                self.obj_body_id[self.TARGET_NAME],
                tuple(stool_bodies),
            )
            return True
        return False

    def _target_root_z(self):
        root = self.objects[self.TARGET_NAME].root_body
        return float(self.sim.data.get_body_xpos(root)[2])

    def _reset_internal(self):
        super()._reset_internal()
        (
            basket_xy,
            basket_quat,
            _,
            _,
            stool_top_half_xy,
            robot_xy,
            robot_yaw,
            _,
            _,
            _,
        ) = self._validated_stool_layout()
        if not self._place_basket_on_stool(
            basket_xy,
            basket_quat,
            stool_top_half_xy,
        ):
            raise RuntimeError(
                f"{self.scene_name}: no stable stool-contact basket pose"
            )
        self._apply_floor_basket_robot_pose(
            robot_xy,
            robot_yaw,
            "basket-stool operation",
        )
        source_xy = np.asarray(
            self.sim.data.get_body_xpos(
                self.objects[self.TARGET_NAME].root_body
            )[:2],
            dtype=float,
        )
        self._source_horizontal_distance = float(
            np.linalg.norm(source_xy - robot_xy)
        )
        desired_yaw = float(
            np.arctan2(
                source_xy[1] - robot_xy[1],
                source_xy[0] - robot_xy[0],
            )
        )
        heading_error = float(
            np.arctan2(
                np.sin(robot_yaw - desired_yaw),
                np.cos(robot_yaw - desired_yaw),
            )
        )
        if abs(np.rad2deg(heading_error)) > 1.0:
            raise RuntimeError(
                f"{self.scene_name}: robot does not face the floor basket"
            )
        _reset_basket_lift_progress(self)

    def _post_action(self, action):
        ret = super()._post_action(action)
        _advance_basket_lift_progress(self)
        return ret

    def _check_success(self):
        return self._bedroom_lift_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            {
                "lang": self.scene_task_spec()["lang"],
                "source_surface": self.task_stool_fixture.stable_id,
                "support_model": self.STOOL_MODEL,
                "support_scale": list(self.STOOL_SCALE),
                "fixed_support": True,
                "removed_floor_obstructions": list(
                    self.removed_floor_obstruction_ids
                ),
                "source_horizontal_distance": float(
                    self._source_horizontal_distance
                ),
                "success_definition": _basket_lift_success_definition(self),
            }
        )
        return ep_meta


__all__ = ["IThorBedroomLiftBasket", "IThorBedroomLiftBasketFromFloor"]
