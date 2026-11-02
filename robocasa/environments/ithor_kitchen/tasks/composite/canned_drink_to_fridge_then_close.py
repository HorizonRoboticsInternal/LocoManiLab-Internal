"""Scene 8: carry a canned drink into the open lower fridge, then close it."""

import math
import os

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.pick_place import SupportContactPlacementMixin
from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.environments.ithor_kitchen.tasks.atomic.stand_canned_drink_upright import (
    IThorKitchenStandCannedDrinkUpright,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


class IThorKitchenCannedDrinkIntoFridgeThenCloseDoor(
    SupportContactPlacementMixin,
    IThorMultiStageCompositeMixin,
    IThorKitchenEnv,
):
    """Carry a canned drink into the open lower freezer drawer and close it."""

    SUPPORTED_SCENE_IDS = (8,)
    SPEC_TASK = "canned_drink_into_fridge_then_close_door"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    INITIAL_POSE_JITTER = False
    CAN_NAME = "canned_drink"

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
                "hold_seconds",
                "target_fixture",
                "target_joint",
                "fridge_receptacle_regions",
                "init_open_fraction",
                "success_open_fraction",
                "success_close_fraction",
                "can_xy",
                "can_support_z",
                "can_region_center_tolerance",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _get_obj_cfgs(self):
        path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../../../..",
                IThorKitchenStandCannedDrinkUpright.CAN_XML,
            )
        )
        return [dict(name=self.CAN_NAME, obj_groups=path)]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualCannedDrinkFridgeSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                name="place_canned_drink_in_fridge",
                success="_fresh_canned_drink_placement",
                task_cls="IThorKitchenCannedDrinkInFridge",
                hold_steps=1,
                on_enter="_arm_canned_drink_placement",
            ),
            IThorAtomicTask(
                name="close_lower_fridge_door",
                success="_closed_with_canned_drink_retained",
                task_cls="IThorKitchenCloseFridge",
                on_enter="_arm_door_close",
            ),
        )

    def _required_stage_hold_steps(self, task):
        if task is self._atomic_tasks[-1]:
            return math.ceil(
                float(self._get_task_spec()["hold_seconds"]) * self.control_freq
            )
        return super()._required_stage_hold_steps(task)

    def _setup_task_fixtures(self):
        spec = self._get_task_spec()
        self.target_fixture = self.get_fixture(id=spec["target_fixture"])
        if self.target_fixture is None:
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: missing target fridge"
            )
        regions_by_id = {
            region.stable_id: region for region in self.get_region(return_all=True)
        }
        self.fridge_receptacle_regions = tuple(
            regions_by_id[region_id]
            for region_id in spec["fridge_receptacle_regions"]
            if region_id in regions_by_id
        )
        if len(self.fridge_receptacle_regions) != len(
            spec["fridge_receptacle_regions"]
        ):
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: missing lower-fridge region"
            )
        if any(
            region.fixture_id != self.target_fixture.stable_id
            for region in self.fridge_receptacle_regions
        ):
            raise ValueError("fridge receptacle region belongs to another fixture")
        self.fixture_refs["target_fridge"] = self.target_fixture
        for index, region in enumerate(self.fridge_receptacle_regions):
            self.region_refs[f"fridge_receptacle_{index}"] = region

    def _setup_references(self):
        super()._setup_references()
        spec = self._get_task_spec()
        self._can_body_id = self.obj_body_id[self.CAN_NAME]
        can_bodies = self._body_tree_ids([self._can_body_id])
        self._can_geom_ids = self._collidable_geoms(can_bodies)
        self._fridge_body_ids = self._fixture_body_ids(
            self.target_fixture,
            include_siblings=True,
        )
        self._target_joint_name = spec["target_joint"]
        self._target_joint_id = self.sim.model.joint_name2id(self._target_joint_name)

    @staticmethod
    def _box_signs():
        return np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )

    def _geom_points(self, geom_ids):
        model, data = self.sim.model._model, self.sim.data._data
        points = []
        for geom_id in geom_ids:
            if int(model.geom_type[geom_id]) == 7:
                mesh_id = int(model.geom_dataid[geom_id])
                start = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                local = model.mesh_vert[start : start + count]
            else:
                bounds = model.geom_aabb[geom_id]
                local = bounds[:3] + self._box_signs() * bounds[3:]
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            points.append(local @ rotation.T + data.geom_xpos[geom_id])
        return np.concatenate(points)

    def _set_door_fraction(self, fraction):
        joint_range = np.asarray(
            self.sim.model.jnt_range[self._target_joint_id],
            dtype=float,
        )
        qpos = float(joint_range[0] + fraction * (joint_range[1] - joint_range[0]))
        self.sim.data.set_joint_qpos(self._target_joint_name, qpos)
        dof = int(self.sim.model.jnt_dofadr[self._target_joint_id])
        self.sim.data.qvel[dof] = 0.0

    def _door_fraction(self):
        joint_range = np.asarray(
            self.sim.model.jnt_range[self._target_joint_id],
            dtype=float,
        )
        qpos = float(self.sim.data.get_joint_qpos(self._target_joint_name))
        span = float(joint_range[1] - joint_range[0])
        return float(np.clip((qpos - joint_range[0]) / span, 0.0, 1.0))

    def _place_canned_drink(self):
        spec = self._get_task_spec()
        obj = self.objects[self.CAN_NAME]
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.forward()
        self._set_task_object_bottom_at(
            self.CAN_NAME,
            spec["can_xy"],
            float(spec["can_support_z"]),
        )

    def _before_object_placement(self):
        super()._before_object_placement()
        spec = self._get_task_spec()
        self._place_canned_drink()
        self._set_door_fraction(float(spec["init_open_fraction"]))
        self.sim.forward()

    def _reset_task_progress(self):
        super()._reset_task_progress()
        self._canned_drink_placement_armed = False
        self._door_close_armed = False

    def _canned_drink_released(self):
        return self.object_released(self.CAN_NAME, th=0.20)

    def _canned_drink_contacts_fridge(self):
        can_bodies = self._body_tree_ids([self._can_body_id])
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body1 in can_bodies and body2 in self._fridge_body_ids) or (
                body2 in can_bodies and body1 in self._fridge_body_ids
            ):
                return True
        return False

    def _canned_drink_center_in_lower_fridge(self):
        points = self._geom_points(self._can_geom_ids)
        can_center = (points.min(axis=0) + points.max(axis=0)) / 2
        tolerance = np.asarray(
            self._get_task_spec()["can_region_center_tolerance"], dtype=float
        )
        if tolerance.shape != (3,) or np.any(tolerance < 0):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: "
                "can_region_center_tolerance must contain non-negative XYZ"
            )
        for region in self.fridge_receptacle_regions:
            center, yaw = region.world_pose(self.sim)
            delta = can_center - np.asarray(center, dtype=float)
            cosine, sine = np.cos(-yaw), np.sin(-yaw)
            local = delta.copy()
            local[0] = cosine * delta[0] - sine * delta[1]
            local[1] = sine * delta[0] + cosine * delta[1]
            if np.all(np.abs(local) <= np.asarray(region.half_size) + tolerance):
                return True
        return False

    def _canned_drink_in_fridge(self):
        return bool(
            self._canned_drink_center_in_lower_fridge()
            and self._canned_drink_contacts_fridge()
            and self._canned_drink_released()
        )

    def _door_open(self):
        return self._door_fraction() >= float(
            self._get_task_spec()["success_open_fraction"]
        )

    def _door_closed(self):
        return self._door_fraction() <= float(
            self._get_task_spec()["success_close_fraction"]
        )

    def _arm_canned_drink_placement(self):
        self._canned_drink_placement_armed = False

    def _fresh_canned_drink_placement(self):
        placed = self._canned_drink_in_fridge()
        if not self._canned_drink_placement_armed:
            if not placed:
                self._canned_drink_placement_armed = True
            return False
        return bool(placed and self._door_open())

    def _arm_door_close(self):
        self._door_close_armed = False

    def _closed_with_canned_drink_retained(self):
        closed = self._door_closed()
        if not self._door_close_armed:
            if not closed:
                self._door_close_armed = True
            return False
        return bool(closed and self._canned_drink_in_fridge())

    def _success_region_bounds(self):
        return []

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "put the canned drink in the open lower freezer drawer, then close it"
            ),
            target_entity="task canned drink",
            target_receptacles=[
                region.stable_id for region in self.fridge_receptacle_regions
            ],
            target_joint=self._target_joint_name,
            task_id=type(self).__name__,
            success_definition=(
                "release the canned drink with its center inside the expanded lower freezer "
                "drawer, then close that drawer to at most 10 percent while "
                "retaining the canned drink for 1 second; robot position is not scored"
            ),
        )
        return meta


__all__ = ["IThorKitchenCannedDrinkIntoFridgeThenCloseDoor"]
