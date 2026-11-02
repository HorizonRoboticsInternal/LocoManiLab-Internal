"""Scene 13: open a drawer, then put a canned drink inside."""

import os

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_atomic.pick_place import (
    SupportContactPlacementMixin,
)
from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.environments.ithor_kitchen.tasks.atomic.stand_canned_drink_upright import (
    IThorKitchenStandCannedDrinkUpright,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)
from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion


class IThorKitchenOpenDrawerThenPutCannedDrinkInside(
    SupportContactPlacementMixin,
    CompositeStageReporterMixin,
    RoomOpenArticulation,
    IThorKitchenEnv,
):
    """Open the drawer in front of G1, then put the counter can inside."""

    ATOMIC_SKILL = "OpenDrawerThenPutCannedDrinkInside"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_STAGES = (
        "open_drawer",
        "put_canned_drink_in_drawer",
        "close_drawer",
    )
    SUPPORTED_SCENE_IDS = (13,)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_kitchen_task_spec)
    SPEC_TASK = "canned_drink_drawer_cycle"
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Drawer",)
    TARGET_LANG = "drawer"
    TARGET_JOINT_ROLES = ("drawer",)
    TARGET_JOINT_TYPES = ("slide",)
    OPEN_STAND_DISTANCE = 0.40
    OBJECT_ACCEPTANCE_XY_TOL = 0.015
    OBJECT_ACCEPTANCE_Z_TOL = 0.025

    def __init__(self, *args, scene_ids=None, **kwargs):
        scene_ids = self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids
        kwargs.setdefault("initialization_noise", None)
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

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
        return SequentialCompositeSampler(name="ManualCannedDrinkDrawerSampler")

    def _semantic_placements_are_valid(self):
        return True

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

    def _setup_scene(self):
        super()._setup_scene()
        spec = self._get_task_spec()
        self.success_hold_seconds = float(spec["hold_seconds"])
        region_id = spec["drawer_receptacle_region"]
        self.drawer_receptacle_region = next(
            (
                region
                for region in self.semantic_scene.get_region(return_all=True)
                if region.stable_id == region_id
            ),
            None,
        )
        if self.drawer_receptacle_region is None:
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: unknown "
                f"drawer_receptacle_region {region_id!r}"
            )
        self._bind_drawer_receptacle_site()
        self._reset_task_progress()

    def _bind_drawer_receptacle_site(self):
        model = self.sim.model
        drawer_body = int(model.jnt_bodyid[model.joint_name2id(self.target_joint)])
        candidates = [
            model.site_name2id(name)
            for name in self.target_fixture.site_names
            if "receptacle" in name.lower()
            and int(model.site_bodyid[model.site_name2id(name)]) == drawer_body
        ]
        if len(
            candidates
        ) != 1 or self.drawer_receptacle_region.parent_body != model.body_id2name(
            drawer_body
        ):
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: expected one receptacle "
                "site on the target drawer's moving body"
            )
        self._drawer_receptacle_site_id = candidates[0]
        self._drawer_contact_geoms = self._geom_names_for_bodies(
            self._body_tree_ids([drawer_body])
        )

    def _before_object_placement(self):
        super()._before_object_placement()
        self._place_canned_drink_from_spec()

    def _place_canned_drink_from_spec(self):
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

    def _reset_task_progress(self):
        self._success_hold = 0
        self._task_success_hold_steps = 0
        self._drawer_opened = False
        self._canned_drink_placed = False
        self._final_stage_reported = False

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_task_progress()

    def _post_action(self, action):
        if not self._drawer_opened and (
            self.target_semantic_joint.fraction(self.sim) >= self.SUCCESS_OPEN_FRACTION
        ):
            self._drawer_opened = True
            self._report_stage_complete(
                1,
                len(self.COMPOSITE_STAGES),
                next_stage=self.COMPOSITE_STAGES[1],
            )
        elif (
            self._drawer_opened
            and not self._canned_drink_placed
            and self._canned_drink_inside_drawer()
        ):
            self._canned_drink_placed = True
            self._report_stage_complete(
                2,
                len(self.COMPOSITE_STAGES),
                next_stage=self.COMPOSITE_STAGES[2],
            )

        self._task_success_hold_steps = (
            self._task_success_hold_steps + 1 if self._task_success_condition() else 0
        )
        if self._check_success() and not self._final_stage_reported:
            self._final_stage_reported = True
            self._report_stage_complete(
                len(self.COMPOSITE_STAGES), len(self.COMPOSITE_STAGES)
            )
        return super()._post_action(action)

    def _drawer_receptacle_pose(self):
        site_id = self._drawer_receptacle_site_id
        return (
            np.asarray(self.sim.data.site_xpos[site_id], dtype=float).copy(),
            np.asarray(self.sim.model.site_size[site_id], dtype=float).copy(),
            np.asarray(self.sim.data.site_xmat[site_id], dtype=float).reshape(3, 3),
        )

    def _canned_drink_inside_drawer(self):
        center, half_size, rotation = self._drawer_receptacle_pose()
        local_points = (self.object_bbox_points("canned_drink") - center) @ rotation
        world_tolerance = np.array(
            [
                self.OBJECT_ACCEPTANCE_XY_TOL,
                self.OBJECT_ACCEPTANCE_XY_TOL,
                self.OBJECT_ACCEPTANCE_Z_TOL,
            ]
        )
        local_tolerance = np.abs(rotation).T @ world_tolerance
        return bool(
            np.all(np.abs(local_points) <= half_size + local_tolerance)
            and self.check_contact(
                self.objects["canned_drink"], self._drawer_contact_geoms
            )
            and not any(
                self.g1_hand_object_contact(side, "canned_drink")
                for side in ("left", "right")
            )
        )

    def _task_success_condition(self):
        return bool(
            self._drawer_opened
            and self._canned_drink_placed
            and self._canned_drink_inside_drawer()
            and self.target_semantic_joint.fraction(self.sim)
            <= float(self._get_task_spec()["success_close_fraction"])
        )

    @property
    def required_success_hold_steps(self):
        return max(1, int(round(self.success_hold_seconds * self.control_freq)))

    def _check_success(self):
        return self._task_success_hold_steps >= self.required_success_hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def _drawer_receptacle_bounds(self):
        return None

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            lang=(
                "open the drawer, put the canned drink from the counter inside, "
                "then close the drawer"
            ),
            target_entity="canned_drink",
            target_state="released_inside_closed_drawer",
            success_hold_seconds=self.success_hold_seconds,
        )
        return ep_meta


__all__ = ["IThorKitchenOpenDrawerThenPutCannedDrinkInside"]
