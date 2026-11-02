"""iTHOR kitchen: stand a canned drink upright on its work surface."""

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


class IThorKitchenStandCannedDrinkUpright(
    SupportContactPlacementMixin, IThorKitchenEnv
):
    """Reorient a canned drink from lying sideways to standing bottom-down."""

    ATOMIC_SKILL = "ReorientObjectUpright"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "pick_place"
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    SPEC_TASK = "stand_canned_drink_upright"
    SUPPORT_SITE_NAME = "canned_drink_upright_success_region"

    CAN_XML = "models/assets/objects/objaverse/can/can_0/model.xml"

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
                    "support_region",
                    "can_pos",
                    "can_quat",
                    "show_support",
                    "upright_tolerance_degrees",
                    "support_z_tolerance",
                    "release_distance",
                    "hold_seconds",
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
        spec = self._get_task_spec()["support_region"]
        center = np.asarray(spec.get("center"), dtype=float)
        half_xy = np.asarray(spec.get("half_xy"), dtype=float)
        yaw = float(spec.get("yaw", 0.0))
        body_name = spec.get("body_name")
        if (
            center.shape != (3,)
            or half_xy.shape != (2,)
            or np.any(half_xy <= 0.0)
            or not isinstance(body_name, str)
            or not body_name
        ):
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
            category="table",
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
        if bool(self._get_task_spec()["show_support"]):
            self.mujoco_arena.worldbody.append(
                new_site(
                    name=self.SUPPORT_SITE_NAME,
                    type="box",
                    pos=[
                        float(center[0]),
                        float(center[1]),
                        float(center[2]) + 0.012,
                    ],
                    size=[float(half_xy[0]), float(half_xy[1]), 0.01],
                    quat=[
                        float(np.cos(yaw / 2.0)),
                        0.0,
                        0.0,
                        float(np.sin(yaw / 2.0)),
                    ],
                    rgba=[0.1, 0.9, 0.2, 0.35],
                    group=0,
                )
            )

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.work_surface_fixture is not None
            and self.work_surface_region is not None
        )

    def _get_obj_cfgs(self):
        if self.work_surface_region is None:
            return []
        return [
            dict(
                name="canned_drink",
                obj_groups=self._can_xml_path(),
            )
        ]

    def _can_xml_path(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", self.CAN_XML)
        )

    def _semantic_placements_are_valid(self):
        return True

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualCannedDrinkUprightSampler")

    def _reset_internal(self):
        super()._reset_internal()
        spec = self._get_task_spec()
        self.upright_cosine = float(
            np.cos(np.deg2rad(float(spec["upright_tolerance_degrees"])))
        )
        self.support_z_tolerance = float(spec["support_z_tolerance"])
        self.release_distance = float(spec["release_distance"])
        self.success_hold_seconds = float(spec["hold_seconds"])
        if not 0.0 < self.upright_cosine <= 1.0:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: invalid upright tolerance"
            )
        self._success_hold_steps = 0
        self._place_canned_drink()

    def _place_canned_drink(self):
        spec = self._get_task_spec()
        pos = np.asarray(spec["can_pos"], dtype=float)
        quat = np.asarray(spec["can_quat"], dtype=float)
        if (
            pos.shape != (3,)
            or quat.shape != (4,)
            or not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-5)
        ):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: can_pos needs XYZ and "
                "can_quat needs a normalized WXYZ quaternion"
            )

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
                f"{self.scene_name}/{self.SPEC_TASK}: no non-penetrating "
                "canned-drink contact on the named work surface"
            )
        # Keep a small, bounded preload so MuJoCo retains a contact constraint
        # on thin exported tabletops instead of dropping an exactly tangent pair.
        contacted_bottom = self.object_bottom_z("canned_drink")
        self._set_task_object_bottom_at(
            "canned_drink",
            self.object_pos("canned_drink")[:2],
            contacted_bottom - 0.001,
        )

    def _raw_success(self):
        obj = self.objects["canned_drink"]
        matrix = np.asarray(
            self.sim.data.get_body_xmat(obj.root_body), dtype=float
        ).reshape(3, 3)
        bottom_down = float(matrix[2, 2]) >= self.upright_cosine
        return bool(
            bottom_down
            and self.object_on_region(
                "canned_drink",
                self.work_surface_region,
                z_tol=self.support_z_tolerance,
                require_contact=True,
            )
            and self.object_released("canned_drink", th=self.release_distance)
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return ret

    @property
    def required_success_hold_steps(self):
        return max(1, int(round(self.success_hold_seconds * self.control_freq)))

    def _check_success(self):
        return self._success_hold_steps >= self.required_success_hold_steps

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            lang="stand the canned drink that is lying on its side upright on the table",
            target_entity="canned_drink",
            target_state="upright_bottom_down_on_work_surface_and_released",
            success_hold_seconds=self.success_hold_seconds,
        )
        return ep_meta


__all__ = ["IThorKitchenStandCannedDrinkUpright"]
