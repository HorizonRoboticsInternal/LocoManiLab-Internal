"""iTHOR kitchen lift tasks with independent YAML scene calibrations."""

import os

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.pick_place import (
    SupportContactPlacementMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)
from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion


class KitchenLiftObjectMixin:
    """Lift one object using a task-specific, scene-calibrated pose."""

    ATOMIC_SKILL = "LiftObject"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "lift"

    TARGET_NAME = None
    TARGET_GROUP = None
    TARGET_XML = None
    TARGET_LANG = None
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    SPEC_TASK = None

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
                    "object_pos",
                    "object_quat",
                    "lift_height",
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
        fixture_id = f"{self.scene_name}:task:{self.SPEC_TASK}:lift_support"
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
            name=f"{self.TARGET_NAME}_lift_support",
            category="table",
            source_category="TaskCalibratedSupport",
            room_id=None,
            body_name=body_name,
            body_names=[body_name],
            site_names=[],
            asset_id=None,
            object_id=f"{self.TARGET_NAME}_lift_support",
            center=center,
            half_size=self.work_surface_region.half_size,
            yaw=yaw,
            front=np.array([np.cos(yaw), np.sin(yaw)]),
            capabilities={"support", "task_calibrated"},
            regions={self.work_surface_region.stable_id: self.work_surface_region},
        )
        self.semantic_scene.add_fixture(self.work_surface_fixture)
        self.fixture_refs["work_surface"] = self.work_surface_fixture
        self.region_refs["work_surface"] = self.work_surface_region

    def _is_scene_valid(self):
        return bool(
            IThorKitchenEnv._is_scene_valid(self)
            and self.work_surface_fixture is not None
            and self.work_surface_region is not None
        )

    def _get_obj_cfgs(self):
        obj_group = self.TARGET_GROUP
        if self.TARGET_XML is not None:
            obj_group = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../../../..", self.TARGET_XML)
            )
        return [dict(name=self.TARGET_NAME, obj_groups=obj_group)]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name=f"Manual{self.TARGET_NAME}LiftSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _place_lift_target(self):
        """Place the lift target using its independent YAML pose."""
        spec = self._get_task_spec()
        pos = np.asarray(spec["object_pos"], dtype=float)
        quat = np.asarray(spec["object_quat"], dtype=float)
        if (
            pos.shape != (3,)
            or quat.shape != (4,)
            or not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-5)
        ):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: object_pos needs XYZ and "
                "object_quat must be normalized WXYZ"
            )

        obj = self.objects[self.TARGET_NAME]
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[:3] = pos
        qpos[3:7] = quat
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.forward()
        if not self.place_object_on_support_contact(
            self.TARGET_NAME,
            self.work_surface_region,
            self.work_surface_fixture,
            xy=pos[:2],
        ):
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: no non-penetrating "
                f"{self.TARGET_NAME} contact on the named work surface"
            )
        contacted_bottom = self.object_bottom_z(self.TARGET_NAME)
        self._set_task_object_bottom_at(
            self.TARGET_NAME,
            self.object_pos(self.TARGET_NAME)[:2],
            contacted_bottom - 0.001,
        )

    def _target_root_z(self):
        root_body = self.objects[self.TARGET_NAME].root_body
        return float(self.sim.data.get_body_xpos(root_body)[2])

    def _reset_internal(self):
        super()._reset_internal()
        spec = self._get_task_spec()
        self.min_lift_delta = float(spec["lift_height"])
        self.lift_hold_seconds = float(spec["hold_seconds"])
        if self.min_lift_delta <= 0.0 or self.lift_hold_seconds <= 0.0:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: lift_height and "
                "hold_seconds must be positive"
            )
        self._place_lift_target()
        self._lift_initial_z = self._target_root_z()
        self._lift_hold_steps = 0

    def _post_action(self, action):
        ret = super()._post_action(action)
        height = self._target_root_z()
        valid = bool(height >= self._lift_initial_z + self.min_lift_delta)
        self._lift_hold_steps = self._lift_hold_steps + 1 if valid else 0
        return ret

    def _raw_success(self):
        return bool(self._target_root_z() >= self._lift_initial_z + self.min_lift_delta)

    @property
    def required_lift_hold_steps(self):
        return max(1, int(round(self.lift_hold_seconds * self.control_freq)))

    def _check_success(self):
        return self._lift_hold_steps >= self.required_lift_hold_steps

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        seconds_unit = "second" if self.lift_hold_seconds == 1.0 else "seconds"
        ep_meta.update(
            lang=(
                f"lift the {self.TARGET_LANG} from the table by "
                f"{self.min_lift_delta:g} meters"
            ),
            target_entity=self.TARGET_NAME,
            target_state="lifted_above_initial_height",
            lift_height=self.min_lift_delta,
            success_hold_seconds=self.lift_hold_seconds,
            success_definition=(
                f"the {self.TARGET_LANG} must remain at least "
                f"{self.min_lift_delta:g} meters above its reset height for "
                f"{self.lift_hold_seconds:g} {seconds_unit}; grasping is not required"
            ),
        )
        return ep_meta


class IThorKitchenLiftEggplant(
    KitchenLiftObjectMixin, SupportContactPlacementMixin, IThorKitchenEnv
):
    TARGET_NAME = "eggplant"
    TARGET_GROUP = "eggplant"
    TARGET_LANG = "eggplant"
    SPEC_TASK = "lift_eggplant"


class IThorKitchenLiftPlate(
    KitchenLiftObjectMixin, SupportContactPlacementMixin, IThorKitchenEnv
):
    TARGET_NAME = "plate"
    TARGET_GROUP = "plate"
    TARGET_LANG = "plate"
    SPEC_TASK = "lift_plate"


class IThorKitchenLiftSaltShaker(
    KitchenLiftObjectMixin, SupportContactPlacementMixin, IThorKitchenEnv
):
    TARGET_NAME = "salt_shaker"
    TARGET_LANG = "salt shaker"
    TARGET_XML = (
        "models/assets/objects/lightwheel/salt_and_pepper_shaker/"
        "SaltShaker002/model.xml"
    )
    SPEC_TASK = "lift_salt_shaker"


__all__ = [
    "IThorKitchenLiftEggplant",
    "IThorKitchenLiftPlate",
    "IThorKitchenLiftSaltShaker",
]
