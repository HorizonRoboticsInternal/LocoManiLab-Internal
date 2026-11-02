"""Scene 13: place a canned drink from the counter into the stove pan."""

import math
import os

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.pick_place import (
    SupportContactPlacementMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.environments.ithor_kitchen.tasks.atomic.stand_canned_drink_upright import (
    IThorKitchenStandCannedDrinkUpright,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import get_ithor_kitchen_task_spec
from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion


class IThorKitchenCannedDrinkIntoPan(SupportContactPlacementMixin, IThorKitchenEnv):
    SUPPORTED_SCENE_IDS = (13,)
    SPEC_TASK = "canned_drink_into_pan"
    ATOMIC_SKILL = "PickPlaceCannedDrink"
    TASK_FAMILY = "pick_place"
    DIFFICULTY_LEVEL = "L2"
    INITIAL_POSE_JITTER = False

    PAN_BODY = "cookingpan_4da4a422c19c7482d3513dc63425448b_1_0_0"
    PAN_JOINT = PAN_BODY + "_jntfree_0"
    PAN_SITES = (
        "cookingpan_4da4a422c19c7482d3513dc63425448b_1_1_0_pan13ReceptacleCollider0_0",
        "cookingpan_4da4a422c19c7482d3513dc63425448b_1_1_0_pan13ReceptacleCollider1_1",
    )

    def __init__(self, *args, scene_ids=None, **kwargs):
        scene_ids = self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids
        kwargs.setdefault("initialization_noise", None)
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        return get_ithor_kitchen_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "robot_pose",
                "support_region",
                "can_pos",
                "can_quat",
                "hold_seconds",
                "containment_tolerance",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()["support_region"]
        center = np.asarray(spec["center"], dtype=float)
        half_xy = np.asarray(spec["half_xy"], dtype=float)
        yaw = float(spec.get("yaw", 0.0))
        body_name = spec["body_name"]
        if center.shape != (3,) or half_xy.shape != (2,) or np.any(half_xy <= 0):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: invalid support_region"
            )

        fixture_id = f"{self.scene_name}:task:{self.SPEC_TASK}:support"
        self.work_surface_region = SemanticRegion(
            stable_id=f"{fixture_id}:region:support",
            fixture_id=fixture_id,
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.array([half_xy[0], half_xy[1], 0.02]),
            _world_center=center,
            _world_yaw=yaw,
        )
        self.work_surface_fixture = SemanticFixture(
            stable_id=fixture_id,
            name=f"{self.SPEC_TASK}_support",
            category="counter",
            source_category="TaskCalibratedSupport",
            room_id=None,
            body_name=body_name,
            body_names=[body_name],
            site_names=[],
            asset_id=None,
            object_id=f"{self.SPEC_TASK}_support",
            center=center,
            half_size=np.array([half_xy[0], half_xy[1], 0.02]),
            yaw=yaw,
            front=np.array([np.cos(yaw), np.sin(yaw)]),
            capabilities={"support", "task_calibrated"},
            regions={self.work_surface_region.stable_id: self.work_surface_region},
        )
        self.semantic_scene.add_fixture(self.work_surface_fixture)
        self.fixture_refs["work_surface"] = self.work_surface_fixture
        self.region_refs["work_surface"] = self.work_surface_region

    def _get_obj_cfgs(self):
        path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../../../..",
                IThorKitchenStandCannedDrinkUpright.CAN_XML,
            )
        )
        return [dict(name="canned_drink", obj_groups=path)]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualCannedDrinkPanSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _spawn_clearable_fixture_roots(self):
        roots = super()._spawn_clearable_fixture_roots()
        return roots - {self.sim.model.body_name2id(self.PAN_BODY)}

    def _setup_references(self):
        super()._setup_references()
        self._pan_site_ids = [
            self.sim.model.site_name2id(name) for name in self.PAN_SITES
        ]

    def _place_canned_drink(self):
        spec = self._get_task_spec()
        pos = np.asarray(spec["can_pos"], dtype=float)
        quat = np.asarray(spec["can_quat"], dtype=float)
        if (
            pos.shape != (3,)
            or quat.shape != (4,)
            or not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-5)
        ):
            raise ValueError(f"{self.scene_name}/{self.SPEC_TASK}: invalid can pose")

        obj = self.objects["canned_drink"]
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[:3] = pos
        qpos[3:7] = quat
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.forward()
        if not self.place_object_on_support_contact(
            "canned_drink",
            self.work_surface_region,
            self.work_surface_fixture,
            xy=pos[:2],
        ):
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: no stable canned-drink "
                "contact on the counter"
            )
        contacted_bottom = self.object_bottom_z("canned_drink")
        self._set_task_object_bottom_at(
            "canned_drink",
            self.object_pos("canned_drink")[:2],
            contacted_bottom - 0.001,
        )

    def _before_object_placement(self):
        super()._before_object_placement()
        self._place_canned_drink()
        start, end = self.sim.model.get_joint_qvel_addr(self.PAN_JOINT)
        self.sim.data.qvel[start:end] = 0.0
        self.sim.forward()

    @staticmethod
    def _point_inside_region(point, center, rotation, half_size, tolerance):
        local = (
            np.asarray(point, dtype=float) - np.asarray(center, dtype=float)
        ) @ np.asarray(rotation, dtype=float)
        return bool(
            np.all(
                np.abs(local) <= np.asarray(half_size, dtype=float) + float(tolerance)
            )
        )

    def _canned_drink_in_pan(self):
        points = self.object_bbox_points("canned_drink")
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        tolerance = float(self._get_task_spec()["containment_tolerance"])
        for site_id in self._pan_site_ids:
            if self._point_inside_region(
                center,
                self.sim.data.site_xpos[site_id],
                self.sim.data.site_xmat[site_id].reshape(3, 3),
                self.sim.model.site_size[site_id],
                tolerance,
            ):
                return True
        return False

    def _reset_internal(self):
        super()._reset_internal()
        self._success_hold_steps = 0

    def _post_action(self, action):
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._canned_drink_in_pan() else 0
        )
        return super()._post_action(action)

    def _check_success(self):
        required = math.ceil(
            float(self._get_task_spec()["hold_seconds"]) * self.control_freq
        )
        return self._success_hold_steps >= required

    def _success_region_bounds(self):
        return []

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="place the canned drink from the counter into the pan on the stove",
            task_id=type(self).__name__,
            target_entity="canned_drink",
            target_receptacle="FloorPlan13:Pan_13_29",
            target_pan_sites=list(self.PAN_SITES),
            success_definition=(
                "the canned drink center remains inside the native pan "
                "receptacle continuously for 1 second"
            ),
        )
        return meta


__all__ = ["IThorKitchenCannedDrinkIntoPan"]
