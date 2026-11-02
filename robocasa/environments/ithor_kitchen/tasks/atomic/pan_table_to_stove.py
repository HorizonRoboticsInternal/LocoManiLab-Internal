"""iTHOR kitchen: move a frying pan from the side counter to a stove burner."""

import os

import numpy as np
from robosuite.utils.mjcf_utils import new_site
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.pick_place import (
    SupportContactPlacementMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import get_ithor_kitchen_task_spec
from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion


class IThorKitchenPanTableToStove(SupportContactPlacementMixin, IThorKitchenEnv):
    """Place a released, upright frying pan on one named stove burner."""

    ATOMIC_SKILL = "PickPlaceOnStove"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "pick_place"
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    SPEC_TASK = "pan_table_to_stove"
    PAN_XML = "models/assets/objects/objaverse/pan/pan_0/model.xml"
    TARGET_SITE_NAME = "pan_stove_success_region"

    def __init__(self, *args, scene_ids=None, **kwargs):
        super().__init__(
            *args,
            scene_ids=self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids,
            **kwargs,
        )

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_kitchen_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "robot_pose",
                    "target_region",
                    "pan_pos",
                    "pan_quat",
                    "show_target_region",
                    "support_z_tolerance",
                    "pan_base_local_xy",
                    "pan_base_radius",
                    "hold_seconds",
                ),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _make_region(self, label, spec, category):
        center = np.asarray(spec["center"], dtype=float)
        half_xy = np.asarray(spec["half_xy"], dtype=float)
        yaw, body_name = float(spec.get("yaw", 0.0)), spec["body_name"]
        if center.shape != (3,) or half_xy.shape != (2,) or np.any(half_xy <= 0):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: invalid {label}_region"
            )
        fixture_id = f"{self.scene_name}:task:{self.SPEC_TASK}:{label}"
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
            name=f"{self.SPEC_TASK}_{label}",
            category=category,
            source_category="TaskCalibratedSupport",
            room_id=None,
            body_name=body_name,
            body_names=[body_name],
            site_names=[],
            asset_id=None,
            object_id=f"{self.SPEC_TASK}_{label}",
            center=center,
            half_size=np.array([half_xy[0], half_xy[1], 0.02]),
            yaw=yaw,
            front=np.array([np.cos(yaw), np.sin(yaw)]),
            capabilities={"support", "task_calibrated"},
            regions={region.stable_id: region},
        )
        self.semantic_scene.add_fixture(fixture)
        return fixture, region

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        self.target_fixture, self.target_region = self._make_region(
            "target", spec["target_region"], "stove_burner"
        )
        self.fixture_refs["stove"] = self.target_fixture
        self.region_refs["stove"] = self.target_region
        if bool(spec["show_target_region"]):
            center, yaw = self.target_region.world_pose()
            self.mujoco_arena.worldbody.append(
                new_site(
                    name=self.TARGET_SITE_NAME,
                    type="box",
                    pos=[float(center[0]), float(center[1]), float(center[2]) + 0.012],
                    size=[
                        float(self.target_region.half_size[0]),
                        float(self.target_region.half_size[1]),
                        0.01,
                    ],
                    quat=[float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))],
                    rgba=[0.1, 0.9, 0.2, 0.35],
                    group=0,
                )
            )

    def _get_obj_cfgs(self):
        return [
            dict(
                name="pan",
                obj_groups=os.path.abspath(
                    os.path.join(
                        os.path.dirname(__file__), "../../../..", self.PAN_XML
                    )
                ),
            )
        ]

    def _semantic_placements_are_valid(self):
        return True

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualPanTableToStoveSampler")

    def _reset_internal(self):
        super()._reset_internal()
        spec = self._get_task_spec()
        self.support_z_tolerance = float(spec["support_z_tolerance"])
        self.pan_base_local_xy = np.asarray(spec["pan_base_local_xy"], dtype=float)
        self.pan_base_radius = float(spec["pan_base_radius"])
        if self.pan_base_local_xy.shape != (2,) or self.pan_base_radius <= 0.0:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: invalid pan base geometry"
            )
        self.success_hold_seconds = float(spec["hold_seconds"])
        self._success_hold_steps = 0
        pos, quat = np.asarray(spec["pan_pos"], float), np.asarray(
            spec["pan_quat"], float
        )
        obj = self.objects["pan"]
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[:3], qpos[3:7] = pos, quat
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.forward()
        self._set_task_object_bottom_at("pan", pos[:2], pos[2])

    def _pan_base_in_target_region(self):
        obj = self.objects["pan"]
        body_pos = np.asarray(self.sim.data.get_body_xpos(obj.root_body), dtype=float)
        body_mat = np.asarray(
            self.sim.data.get_body_xmat(obj.root_body), dtype=float
        ).reshape(3, 3)
        base_local = np.array(
            [self.pan_base_local_xy[0], self.pan_base_local_xy[1], 0.0]
        )
        base_world = body_pos + body_mat @ base_local

        center, yaw = self.target_region.world_pose(self.sim)
        delta = base_world[:2] - center[:2]
        c, s = np.cos(-yaw), np.sin(-yaw)
        region_xy = np.array([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]])
        base_inside_xy = np.all(
            np.abs(region_xy) + self.pan_base_radius
            <= self.target_region.half_size[:2] + 1e-4
        )
        base_on_target_height = (
            abs(self.object_bottom_z("pan") - float(center[2]))
            <= self.support_z_tolerance
        )
        return bool(base_inside_xy and base_on_target_height)

    def _raw_success(self):
        return bool(
            self._pan_base_in_target_region()
            and not self.g1_hand_grasping_object("pan")
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return ret

    def _check_success(self):
        required = max(1, int(round(self.success_hold_seconds * self.control_freq)))
        return self._success_hold_steps >= required

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="pick up the frying pan from the table beside the stove and place it on the stove",
            target_entity="stove_target_region",
            target_state="released_pan_base_inside_target_region",
        )
        return meta


__all__ = ["IThorKitchenPanTableToStove"]
