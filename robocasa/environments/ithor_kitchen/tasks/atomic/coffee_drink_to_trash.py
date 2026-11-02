"""iTHOR kitchen: throw a coffee drink into the trash can."""

import numpy as np
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


class IThorKitchenCoffeeDrinkToTrash(IThorKitchenEnv):
    """Pick up a coffee drink and drop it into the calibrated trash can."""

    ATOMIC_SKILL = "PlaceIntoReceptacle"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "receptacle"

    SOURCE_NAME = "coffee_drink"
    SOURCE_GROUP = "coffee_cup"
    SUPPORTED_SCENE_IDS = (7, 13, 24, 25)
    SPEC_TASK = "coffee_drink_to_trash"
    TRASH_XY_TOL = 0.04
    TRASH_Z_TOL = 0.04

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
                    "source_pos",
                    "trash_fixture",
                    "trash_region",
                    "trash_body_pos",
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
        self.trash_fixture = self.get_fixture(id=spec["trash_fixture"])
        if self.trash_fixture is None:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: missing configured trash fixture"
            )
        self.trash_region = next(
            (
                region
                for region in self.trash_fixture.regions.values()
                if region.stable_id == spec["trash_region"]
            ),
            None,
        )
        if self.trash_region is None:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: missing configured trash region"
            )
        self._set_trash_body_pos(spec["trash_body_pos"])
        self.fixture_refs["trash_can"] = self.trash_fixture
        self.region_refs["trash_receptacle"] = self.trash_region

    def _set_trash_body_pos(self, configured_pos):
        if self.trash_fixture is None or self.trash_fixture.body_name is None:
            raise RuntimeError(
                f"{self.scene_name}: selected trash can has no root body"
            )
        body = self.find_scene_body(self.trash_fixture.body_name)
        if body is None:
            raise RuntimeError(
                f"{self.scene_name}: missing trash body {self.trash_fixture.body_name!r}"
            )

        old_pos = string_to_array(body.get("pos", "0 0 0")).astype(float)
        target_pos = np.asarray(configured_pos, dtype=float)
        if target_pos.shape != (3,):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: trash_body_pos needs XYZ"
            )
        body.set("pos", array_to_string(target_pos))

        delta = target_pos - old_pos
        self.trash_fixture.center = (
            np.asarray(self.trash_fixture.center, dtype=float) + delta
        )
        for region in self.trash_fixture.regions.values():
            region._world_center = np.asarray(region._world_center, dtype=float) + delta

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.trash_region is not None)

    def _get_obj_cfgs(self):
        return [
            dict(
                name=self.SOURCE_NAME,
                obj_groups=self.SOURCE_GROUP,
            )
        ]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualCoffeeDrinkToTrashSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _reset_internal(self):
        self._success_hold_steps = 0
        super()._reset_internal()
        self.success_hold_seconds = float(self._get_task_spec()["hold_seconds"])
        if self.SOURCE_NAME in self.objects:
            self._place_source_object()

    def _place_source_object(self):
        source_pos = np.asarray(self._get_task_spec()["source_pos"], dtype=float)
        if source_pos.shape != (3,):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: source_pos needs XYZ"
            )
        obj = self.objects[self.SOURCE_NAME]
        joint_name = obj.joints[0]
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        qpos[:3] = source_pos
        self.sim.data.set_joint_qpos(joint_name, qpos)
        joint_id = self.sim.model.joint_name2id(joint_name)
        qvel_addr = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[qvel_addr : qvel_addr + 6] = 0.0
        self.sim.forward()

    def _raw_success(self):
        return bool(
            self.object_in_region(
                self.SOURCE_NAME,
                self.trash_region,
                rim_tol=self.TRASH_Z_TOL,
                xy_tol=self.TRASH_XY_TOL,
            )
            and self.object_released(self.SOURCE_NAME)
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return ret

    def _check_success(self):
        required_steps = max(
            1, int(round(self.success_hold_seconds * self.control_freq))
        )
        return self._success_hold_steps >= required_steps

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        trash_center, _ = self.trash_region.world_pose(self.sim)
        ep_meta.update(
            source_pos=[float(v) for v in self._get_task_spec()["source_pos"]],
            trash_receptacle=[float(v) for v in trash_center],
            lang="pick up the coffee drink and throw it into the trash can",
        )
        return ep_meta


__all__ = ["IThorKitchenCoffeeDrinkToTrash"]
