"""iTHOR kitchen: carry a banana basket from one desk to another."""

import os

import numpy as np
from robosuite.utils.mjcf_utils import new_site
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.pick_place import (
    SupportContactPlacementMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)
from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion
from robocasa.utils import object_utils as OU


class IThorKitchenBananaBasketToTable(SupportContactPlacementMixin, IThorKitchenEnv):
    """Move a handled basket with a banana in it to a nearby desk."""

    ATOMIC_SKILL = "CarryLoadedReceptacle"
    DIFFICULTY_LEVEL = "L3"
    TASK_FAMILY = "pick_place"
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    SPEC_TASK = "banana_basket_to_table"
    TARGET_REGION_SITE_NAME = "banana_basket_target_region"

    BASKET_XML = "models/assets/objects/lightwheel/basket/Basket027/model.xml"
    TARGET_Z_TOL = 0.035
    BANANA_IN_BASKET_XY_TOL = 0.14
    RELEASE_DISTANCE = 0.18

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_kitchen_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "robot_pose",
                    "target_region",
                    "basket_pos",
                    "banana_pos",
                ),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        if pos.shape != (3,) or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: robot_pose needs XYZ + yaw"
            )
        return pos, float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        self.target_fixture, self.target_region = self._make_task_support_region(
            "target", spec["target_region"]
        )

        self.fixture_refs["target_desk"] = self.target_fixture
        self.region_refs["target_desk"] = self.target_region
        if bool(spec.get("show_target_region", False)):
            center, yaw = self.target_region.world_pose()
            self.mujoco_arena.worldbody.append(
                new_site(
                    name=self.TARGET_REGION_SITE_NAME,
                    type="box",
                    pos=[float(center[0]), float(center[1]), float(center[2]) + 0.012],
                    size=[
                        float(self.target_region.half_size[0]),
                        float(self.target_region.half_size[1]),
                        0.01,
                    ],
                    quat=[np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
                    rgba=[0.1, 0.9, 0.2, 0.35],
                    group=0,
                )
            )

    def _make_task_support_region(self, label, spec):
        center = np.asarray(spec.get("center"), dtype=float)
        half_xy = np.asarray(spec.get("half_xy"), dtype=float)
        yaw = float(spec.get("yaw", 0.0))
        body_name = spec.get("body_name")
        if center.shape != (3,) or half_xy.shape != (2,) or np.any(half_xy <= 0.0):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}/{label}_region: "
                "expected center XYZ and positive half_xy"
            )
        if not isinstance(body_name, str) or not body_name:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}/{label}_region: "
                "body_name is required"
            )

        fixture_id = f"{self.scene_name}:task:{self.SPEC_TASK}:{label}_support"
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
            name=f"{self.SPEC_TASK}_{label}_support",
            category="table",
            source_category="TaskCalibratedSupport",
            room_id=None,
            body_name=body_name,
            body_names=[body_name],
            site_names=[],
            asset_id=None,
            object_id=f"{self.SPEC_TASK}_{label}_support",
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
        return bool(super()._is_scene_valid() and self.target_region is not None)

    def _get_obj_cfgs(self):
        return [
            dict(
                name="basket",
                obj_groups=self._basket_xml_path(),
            ),
            dict(
                name="banana",
                obj_groups="banana",
            ),
        ]

    def _semantic_placements_are_valid(self):
        return True

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="BananaBasketToTableSampler")

    def _basket_xml_path(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", self.BASKET_XML)
        )

    def _reset_internal(self):
        super()._reset_internal()
        self.target_hold_seconds = float(self._get_task_spec()["hold_seconds"])
        self._target_hold_steps = 0
        self._place_task_objects()

    def _place_task_objects(self):
        spec = self._get_task_spec()
        for name in ("basket", "banana"):
            pos = np.asarray(spec[f"{name}_pos"], dtype=float)
            if pos.shape != (3,) or not np.all(np.isfinite(pos)):
                raise ValueError(
                    f"{self.scene_name}/{self.SPEC_TASK}: "
                    f"{name}_pos must be finite [x, y, bottom_z]"
                )
            self._set_task_object_bottom_at(name, pos[:2], pos[2])

    def _basket_on_target_desk(self):
        return self.object_on_region(
            "basket",
            self.target_region,
            z_tol=self.TARGET_Z_TOL,
            require_contact=True,
        )

    def _basket_placed_and_stable(self):
        return bool(
            self._basket_on_target_desk()
            and self.object_released("basket", th=self.RELEASE_DISTANCE)
        )

    def _banana_in_basket(self):
        return bool(
            OU.check_obj_in_receptacle(
                self, "banana", "basket", th=self.BANANA_IN_BASKET_XY_TOL
            )
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        settled = self._basket_placed_and_stable() and self._banana_in_basket()
        self._target_hold_steps = self._target_hold_steps + 1 if settled else 0
        return ret

    @property
    def target_hold_steps(self):
        return max(1, int(round(self.target_hold_seconds * self.control_freq)))

    def _check_success(self):
        return self._target_hold_steps >= self.target_hold_steps

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        target_center, _ = self.target_region.world_pose(self.sim)
        ep_meta.update(
            target_surface=[float(v) for v in target_center],
            lang=(
                "pick up the handled basket with the banana in it, carry it to the "
                "closest desk, and put the basket down while keeping the banana inside"
            ),
        )
        return ep_meta


__all__ = ["IThorKitchenBananaBasketToTable"]
