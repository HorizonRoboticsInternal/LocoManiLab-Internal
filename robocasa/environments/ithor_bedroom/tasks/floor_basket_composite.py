"""Carry a basket from the calibrated low stool to the bedroom desk."""

from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np

from robocasa.environments.ithor_atomic.free_fixture import FreeFixturePlacementMixin
from robocasa.models.fixtures import Fixture
from robocasa.models.scenes.thor_semantics import SemanticFixture
from robocasa.utils import env_utils as EnvUtils

from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.environments.ithor_composite.progress import CompositeStageReporterMixin

from .assets import (
    BOXED_FOOD_MODEL,
    BOXED_FOOD_SCALE,
    LIFT_BASKET_MODEL,
    LIFT_BASKET_SCALE,
    asset_xml_path,
)
from .lift_basket import IThorBedroomLiftBasketFromFloor as _FloorBasketAtomic
from .object_in_receptacle import (
    BedroomNearSurfaceStartMixin,
    BedroomPnPMixin,
    BedroomRightHandSourceLayoutMixin,
    BedroomSimpleReceptaclePlacementMixin,
)


class IThorBedroomLiftBasketFromFloorThenPlaceOnDeskAndPutBoxedFoodInBasket(
    CompositeStageReporterMixin,
    FreeFixturePlacementMixin,
    BedroomSimpleReceptaclePlacementMixin,
    BedroomNearSurfaceStartMixin,
    BedroomRightHandSourceLayoutMixin,
    BedroomPnPMixin,
    IThorBedroomEnv,
):
    """Lift the basket beside the desk, then turn to place and fill it."""

    TASK_FAMILY = "composite"
    ATOMIC_SKILL = "LiftThenPickPlace"
    DIFFICULTY_LEVEL = "L2"
    SUPPORTED_SCENE_IDS = (329,)
    SCENE_SPECS = {
        "FloorPlan329": {
            **deepcopy(_FloorBasketAtomic.SCENE_SPECS["FloorPlan329"]),
            # Rotate the atomic layout clockwise 90 degrees beside the desk.
            # G1 faces the basket and can turn right 90 degrees toward the desk.
            "basket_xy": [-0.7265, 0.2504],
            "basket_quat": [np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)],
            "stool_xy": [-0.7265, 0.2504],
            "stool_yaw": -np.pi / 2,
            "robot_pose": {"xy": [-0.2765, 0.2504], "yaw": np.pi},
            "staging_robot_pose": {"xy": [-0.0265, 0.2504], "yaw": np.pi},
            "lang": (
                "pick up the basket from the low stool in front of you, place it "
                "on the desk, then put the boxed food in the basket"
            ),
        }
    }
    # Reuse the atomic support geometry, contact probes and locked stool base.
    STOOL_NAME = _FloorBasketAtomic.STOOL_NAME
    STOOL_MODEL = _FloorBasketAtomic.STOOL_MODEL
    STOOL_SCALE = _FloorBasketAtomic.STOOL_SCALE
    STOOL_BASE_WELD_NAME = _FloorBasketAtomic.STOOL_BASE_WELD_NAME
    SUPPORT_PROBE_ABOVE = _FloorBasketAtomic.SUPPORT_PROBE_ABOVE
    SUPPORT_PROBE_STEP = _FloorBasketAtomic.SUPPORT_PROBE_STEP
    SUPPORT_PENETRATION_TOLERANCE = _FloorBasketAtomic.SUPPORT_PENETRATION_TOLERANCE
    _validated_stool_layout = _FloorBasketAtomic._validated_stool_layout
    _stool_base_weld_id = _FloorBasketAtomic._stool_base_weld_id
    _set_stool_base_weld_active = _FloorBasketAtomic._set_stool_base_weld_active
    _lock_stool_base_at_current_pose = (
        _FloorBasketAtomic._lock_stool_base_at_current_pose
    )
    _basket_footprint_is_on_stool = _FloorBasketAtomic._basket_footprint_is_on_stool
    _place_basket_on_stool = _FloorBasketAtomic._place_basket_on_stool
    _apply_floor_basket_robot_pose = _FloorBasketAtomic._apply_floor_basket_robot_pose
    SOURCE_NAME = "boxed_food"
    RECEPTACLE_NAME = TARGET_NAME = "basket"
    SOURCE_GROUP, RECEPTACLE_GROUP = "boxed_food", "basket"
    SOURCE_MJCF_PATH = asset_xml_path(BOXED_FOOD_MODEL)
    SOURCE_SCALE = BOXED_FOOD_SCALE
    RECEPTACLE_MJCF_PATH = asset_xml_path(LIFT_BASKET_MODEL)
    RECEPTACLE_SCALE = LIFT_BASKET_SCALE
    # Reserve room for the canonical lift basket, wider than the food atomic basket.
    RECEPTACLE_LATERAL_OFFSET, SOURCE_LATERAL_OFFSET = -0.18, 0.18
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    ROBOT_PRE_SNAP_FREE_JOINT_Z = 0.0
    MIN_LIFT_DELTA = 0.30
    INTERMEDIATE_HOLD_STEPS = 5
    COMPOSITE_ORDERING = (
        "lift_basket_from_stool",
        "place_basket_on_desk",
        "put_boxed_food_in_basket",
    )

    def _apply_calibrated_operation_pose(self, *, include_composites=True):
        # Reuse the atomic food task's near-desk pose for this composite only.
        super()._apply_calibrated_operation_pose(include_composites=True)

    def _before_object_placement(self):
        # The inherited sampler first calibrates the two objects on the desk.
        # Restore its recipe on soft resets, before rebuilding the sampler.
        for cfg in self.object_cfgs:
            if cfg["name"] == self.RECEPTACLE_NAME:
                cfg["placement"]["region"] = self.work_surface_region
        getattr(self, "_task_body_contact_contracts", {}).pop(
            "bedroom_stool_basket_contact", None
        )
        super()._before_object_placement()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self._validated_stool_layout()
        spec = self.scene_task_spec()
        stool_xy = np.asarray(
            _FloorBasketAtomic.SCENE_SPECS[self.scene_name]["stool_xy"], dtype=float
        )
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

    def _seat_task_stool(self, xy):
        self._set_stool_base_weld_active(False)
        if not self._place_fixture_on_floor(
            self.task_stool_fixture,
            xy,
            contract_name="bedroom_stool_floor_contact",
            probe_above=self.SUPPORT_PROBE_ABOVE,
            probe_below=float(self.scene_task_spec()["stool_floor_probe_below"]),
            probe_step=self.SUPPORT_PROBE_STEP,
        ):
            raise RuntimeError(
                f"{self.scene_name}: bedroom stool could not contact Floor"
            )
        self._lock_stool_base_at_current_pose()

    def _setup_scene(self):
        # Keep the stool at the atomic staging position while the inherited
        # sampler and robot settling establish the near-desk operation pose.
        self._seat_task_stool(
            _FloorBasketAtomic.SCENE_SPECS[self.scene_name]["stool_xy"]
        )
        super()._setup_scene()

    def _reset_internal(self):
        super()._reset_internal()
        basket = self.objects[self.RECEPTACLE_NAME]
        self._basket_desk_target_xy = np.asarray(
            self.sim.data.get_body_xpos(basket.root_body)[:2]
        ).copy()
        spec = self.scene_task_spec()
        self._seat_task_stool(spec["stool_xy"])
        if not self._place_basket_on_stool(
            spec["basket_xy"], spec["basket_quat"], spec["stool_top_half_xy"]
        ):
            raise RuntimeError(
                f"{self.scene_name}: no stable stool-contact basket pose"
            )
        robot_xy = np.asarray(spec["robot_pose"]["xy"], dtype=float)
        basket_xy = np.asarray(spec["basket_xy"], dtype=float)
        direction = basket_xy - robot_xy
        yaw = float(np.arctan2(direction[1], direction[0]))
        # Restore G1's nominal pre-snap height instead of reusing the surface
        # pose search's cache, which can place the hands into the low stool.
        robot_joint = self._get_robot_free_joint()
        self._robot_default_free_joint_z = self.ROBOT_PRE_SNAP_FREE_JOINT_Z
        self._apply_floor_basket_robot_pose(robot_xy, yaw, "basket-stool operation")
        EnvUtils.settle_robot_spawn(self)
        self._robot_free_joint_z = float(self.sim.data.get_joint_qpos(robot_joint)[2])
        # The desk placement was only the inherited sampler's staging recipe.
        self._task_support_contracts.pop(self.RECEPTACLE_NAME, None)
        for cfg in self.object_cfgs:
            if cfg["name"] == self.RECEPTACLE_NAME:
                cfg["placement"].pop("region", None)
        self._basket_initial_position = np.asarray(
            self.sim.data.get_body_xpos(basket.root_body)
        ).copy()
        self._record_initial_manipulation_layout()
        self._reset_task_progress()
        self._lift_initial_z = self._target_root_z()

    def _target_root_z(self):
        root = self.objects[self.TARGET_NAME].root_body
        return float(self.sim.data.get_body_xpos(root)[2])

    def _reset_task_progress(self):
        super()._reset_task_progress()
        self._completed_stage = 0
        self._bedroom_lift_hold = 0
        self._desk_hold = 0
        self._final_stage_reported = False

    def _basket_is_on_desk(self):
        # Keep the shared support checks, but ignore hand state in this task.
        basket = self.objects[self.RECEPTACLE_NAME]
        matrix = np.asarray(
            self.sim.data.get_body_xmat(basket.root_body), dtype=float
        ).reshape(3, 3)
        upright_cosine = self.RECEPTACLE_UPRIGHT_COSINE
        support_tolerance = self.RECEPTACLE_SUPPORT_PENETRATION_TOLERANCE
        return bool(
            (upright_cosine is None or float(matrix[2, 2]) >= upright_cosine)
            and self.object_on_region(
                self.RECEPTACLE_NAME,
                self.work_surface_region,
                z_tol=self.RECEPTACLE_SUPPORT_Z_TOLERANCE,
                require_contact=True,
            )
            and self._object_fixture_contact_is_shallow(
                self.RECEPTACLE_NAME, self.work_surface_fixture, support_tolerance
            )
            and not self._object_has_illegal_penetration(
                self.RECEPTACLE_NAME,
                self.work_surface_region,
                tolerance=support_tolerance,
            )
        )

    def _food_waits_on_desk(self):
        return bool(
            self.object_on_region(
                self.SOURCE_NAME, self.work_surface_region, require_contact=True
            )
            and not self._source_bbox_inside_receptacle()
        )

    def _raw_place_success(self):
        source = self.objects[self.SOURCE_NAME]
        basket = self.objects[self.RECEPTACLE_NAME]
        return bool(
            self._basket_is_on_desk()
            and self.check_contact(source, basket)
            and self._source_bbox_inside_receptacle()
        )

    def _advance_composite_progress(self):
        if self._completed_stage == 0:
            lifted = self._target_root_z() >= self._lift_initial_z + self.MIN_LIFT_DELTA
            self._bedroom_lift_hold = self._bedroom_lift_hold + 1 if lifted else 0
            if self._bedroom_lift_hold >= self.INTERMEDIATE_HOLD_STEPS:
                self._completed_stage = 1
                self._report_stage_complete(1, 3, next_stage=self.COMPOSITE_ORDERING[1])
        elif self._completed_stage == 1:
            valid = self._basket_is_on_desk() and self._food_waits_on_desk()
            self._desk_hold = self._desk_hold + 1 if valid else 0
            if self._desk_hold >= self.INTERMEDIATE_HOLD_STEPS:
                self._completed_stage = 2
                self._report_stage_complete(2, 3, next_stage=self.COMPOSITE_ORDERING[2])
        else:
            self._success_hold = (
                self._success_hold + 1 if self._raw_place_success() else 0
            )
            if self._check_success() and not self._final_stage_reported:
                self._final_stage_reported = True
                self._report_stage_complete(3, 3)

    def _post_action(self, action):
        # Avoid the atomic PnP counter: it would allow skipping the lift stages.
        self._advance_composite_progress()
        return IThorBedroomEnv._post_action(self, action)

    def _check_success(self):
        return self._completed_stage == 2 and self._success_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["composite_stages"] = list(self.COMPOSITE_ORDERING)
        meta["composite_ordering"] = list(self.COMPOSITE_ORDERING)
        meta["completed_stages"] = self._completed_stage + int(self._check_success())
        meta["workspace_layout"] = (
            "G1 starts beside the desk facing the basket on the low stool; "
            "the atomic basket-lift layout is rotated and translated as a unit, "
            "so G1 can turn right 90 degrees in place to face the desk"
        )
        meta["basket_initial_position"] = self._basket_initial_position.tolist()
        meta["basket_source_surface"] = self.task_stool_fixture.stable_id
        meta["support_model"] = self.STOOL_MODEL
        meta["support_scale"] = list(self.STOOL_SCALE)
        meta["fixed_support"] = True
        meta["basket_horizontal_distance"] = float(
            np.linalg.norm(self._basket_initial_position[:2] - self.robot_base_pos[:2])
        )
        meta["basket_desk_target_xy"] = self._basket_desk_target_xy.tolist()
        meta["success_definition"] = (
            "first lift the basket at least 0.30 m above its initial stool height "
            f"for {self.INTERMEDIATE_HOLD_STEPS} control steps; then keep it upright "
            "in contact with the desk while the food remains outside on the desk "
            f"for {self.INTERMEDIATE_HOLD_STEPS} control steps; finally keep the boxed "
            "food inside and contacting the basket while the basket stays upright "
            f"on the desk for {self.HOLD_STEPS} control steps; "
            "grasp and release state are not required in any stage"
        )
        return meta
