"""iTHOR kitchen: swap an apple and banana between a plate and table."""

import numpy as np
from robosuite.utils.mjcf_utils import new_site
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_kitchen.tasks.atomic.apple_on_plate import (
    IThorKitchenAppleOnPlate,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)
from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion
from robocasa.utils import object_utils as OU


class IThorKitchenAppleBananaPlateSwap(IThorKitchenAppleOnPlate):
    ATOMIC_SKILL = "AppleBananaPlateSwap"
    TASK_FAMILY = "composite"
    COMPOSITE_ORDERING = "any"
    SPEC_TASK = "apple_banana_plate_swap"

    BANANA_ON_PLATE_XY_TOL = 0.13
    TARGET_REGION_SITE_NAME = "apple_banana_plate_swap_target_region"

    def _get_task_spec(self):
        if getattr(self, "_swap_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_kitchen_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "robot_pose",
                    "support_region",
                    "plate_pos",
                    "banana_pos",
                    "hold_seconds",
                ),
            )
            self._swap_spec_scene = self.scene_name
        return self.task_spec

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()["support_region"]
        self.support_fixture, self.support_region = self._make_support_region(spec)
        self.fixture_refs["support"] = self.support_fixture
        self.region_refs["support"] = self.support_region
        if bool(self._get_task_spec().get("show_target_region", False)):
            self._add_target_region_site()

    def _add_target_region_site(self):
        center, yaw = self.support_region.world_pose()
        half_size = np.asarray(self.support_region.half_size, dtype=float).copy()
        half_size[2] = 0.006
        center = np.asarray(center, dtype=float).copy()
        center[2] += 0.012
        self.mujoco_arena.worldbody.append(
            new_site(
                name=self.TARGET_REGION_SITE_NAME,
                type="box",
                pos=[float(value) for value in center],
                size=[float(value) for value in half_size],
                quat=[np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
                rgba=[1.0, 0.8, 0.05, 0.30],
                group=0,
            )
        )

    def _make_support_region(self, spec):
        center = np.asarray(spec.get("center"), dtype=float)
        half_xy = np.asarray(spec.get("half_xy"), dtype=float)
        z_tolerance = float(spec["z_tolerance"])
        yaw = float(spec.get("yaw", 0.0))
        body_name = spec.get("body_name")
        if (
            center.shape != (3,)
            or half_xy.shape != (2,)
            or np.any(half_xy <= 0.0)
            or z_tolerance < 0.0
        ):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}/support_region: "
                "expected center XYZ and positive half_xy"
            )
        if not isinstance(body_name, str) or not body_name:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}/support_region: "
                "body_name is required"
            )

        fixture_id = f"{self.scene_name}:task:{self.SPEC_TASK}:table_support"
        self.apple_region_z_tolerance = z_tolerance
        region = SemanticRegion(
            stable_id=f"{fixture_id}:region:support",
            fixture_id=fixture_id,
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.array([half_xy[0], half_xy[1], 0.02]),
            _world_center=center,
            _world_yaw=yaw,
        )
        fixture = SemanticFixture(
            stable_id=fixture_id,
            name=f"{self.SPEC_TASK}_table_support",
            category="table",
            source_category="TaskCalibratedSupport",
            room_id=None,
            body_name=body_name,
            body_names=[body_name],
            site_names=[],
            asset_id=None,
            object_id=f"{self.SPEC_TASK}_table_support",
            center=center,
            half_size=np.array([half_xy[0], half_xy[1], 0.02]),
            yaw=yaw,
            front=np.array([np.cos(yaw), np.sin(yaw)]),
            capabilities={"support", "task_calibrated"},
            regions={region.stable_id: region},
        )
        self.semantic_scene.add_fixture(fixture)
        return fixture, region

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.support_region is not None)

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        if not cfgs:
            return []
        return [
            cfgs[0],
            dict(
                name="banana",
                obj_groups="banana",
            ),
            cfgs[1],
        ]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualAppleBananaPlateSwapSampler")

    def _place_task_objects(self):
        spec = self._get_task_spec()
        plate_pos = np.asarray(spec["plate_pos"], dtype=float)
        banana_pos = np.asarray(spec["banana_pos"], dtype=float)
        if plate_pos.shape != (3,) or banana_pos.shape != (3,):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: plate_pos and banana_pos "
                "must contain XYZ"
            )

        self._set_free_joint_bottom_at(
            self.objects["plate"], plate_pos[:2], plate_pos[2]
        )
        self._set_free_joint_bottom_at(
            self.objects["banana"], banana_pos[:2], banana_pos[2]
        )
        self.sim.forward()

        plate_top_z = float(np.max(self.object_bbox_points("plate")[:, 2]))
        self._set_free_joint_bottom_at(
            self.objects["apple"], plate_pos[:2], plate_top_z
        )
        self.sim.forward()

    def _apple_on_table(self):
        return bool(
            self.object_on_region(
                "apple",
                self.support_region,
                z_tol=self.apple_region_z_tolerance,
                require_contact=True,
            )
        )

    def _success_region_bounds(self):
        if not bool(self._get_task_spec().get("show_target_region", False)):
            return []
        center, yaw = self.support_region.world_pose(self.sim)
        half_size = np.asarray(self.support_region.half_size, dtype=float).copy()
        half_size[2] = 0.006
        center = np.asarray(center, dtype=float).copy()
        center[2] += 0.012
        return [(center, half_size, yaw)]

    def _banana_on_plate(self):
        return bool(
            OU.check_obj_in_receptacle(
                self, "banana", "plate", th=self.BANANA_ON_PLATE_XY_TOL
            )
        )

    def _reset_task_progress(self):
        self._success_hold_steps = 0

    def _raw_success(self):
        return bool(
            self._apple_on_table()
            # FloorPlan13 keeps the evaluation predicate without an off-plate check.
            and (self.scene_name == "FloorPlan13" or not super()._raw_success())
            and self._banana_on_plate()
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "pick the apple from the plate, put the apple on the table, and put "
            "the banana on the plate"
        )
        ep_meta["composite_ordering"] = self.COMPOSITE_ORDERING
        return ep_meta


__all__ = ["IThorKitchenAppleBananaPlateSwap"]
