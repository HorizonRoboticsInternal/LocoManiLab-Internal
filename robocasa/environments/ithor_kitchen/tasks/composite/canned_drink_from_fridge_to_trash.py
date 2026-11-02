"""Scene 25: take a canned drink from an open fridge and discard it."""

import os

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_atomic.pick_place import (
    SupportContactPlacementMixin,
)
from robocasa.environments.ithor_composite.base import IThorAtomicTask
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.environments.ithor_kitchen.tasks.composite.multi_step_manipulation import (
    _OrderedArticulationWorkflowMixin,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.stand_canned_drink_upright import (
    IThorKitchenStandCannedDrinkUpright,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)
from robocasa.models.scenes.thor_semantics import SemanticRegion


class IThorKitchenCannedDrinkFromFridgeToTrash(
    _OrderedArticulationWorkflowMixin,
    SupportContactPlacementMixin,
    RoomOpenArticulation,
    IThorKitchenEnv,
):
    """Take a canned drink from an open fridge and put it in the trash can."""

    ATOMIC_SKILL = "CannedDrinkFromFridgeToTrash"
    COMPOSITE_KIND = "take_discard"
    SPEC_TASK = "canned_drink_from_fridge_to_trash"
    TASK_LANG = (
        "take the canned drink from the open fridge, then walk to the trash can "
        "and put the canned drink inside"
    )
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    SUPPORTED_SCENE_IDS = (25,)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_kitchen_task_spec)
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Fridge",)
    TARGET_LANG = "fridge door"
    TARGET_JOINT_ROLES = ("door",)
    INITIAL_POSE_JITTER = False
    TRASH_CENTER_XY_TOL = 0.04
    TRASH_CENTER_Z_TOL = 0.05

    STOOL_BODY = "stool_ddf34c98948ee78cfc554a575e4ac59e_1_0_0"
    STOOL_JOINT = STOOL_BODY + "_jntfree_0"
    STOOL_SEAT_GEOM = "stool_ddf34c98948ee78cfc554a575e4ac59e_1_1_0_collision_1"
    CAN_NAME = "canned_drink"
    CAN_XML = IThorKitchenStandCannedDrinkUpright.CAN_XML

    REQUIRED_SPEC_FIELDS = (
        "hold_seconds",
        "robot_pose",
        "target_fixture",
        "target_joint",
        "init_open_fraction",
        "success_open_fraction",
        "fridge_receptacle_region",
        "can_xy",
        "trash_fixture",
        "trash_region",
        "stool_xy",
    )

    def __init__(self, *args, scene_ids=None, **kwargs):
        scene_ids = self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids
        kwargs.setdefault("initialization_noise", None)
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                "take_canned_drink_from_fridge",
                "_canned_drink_removed_active_stage",
                task_cls="PickCannedDrinkFromFridge",
                hold_steps=self.INTERMEDIATE_HOLD_STEPS,
                on_enter="_arm_active_stage_condition",
            ),
            IThorAtomicTask(
                "put_canned_drink_in_trash",
                "_canned_drink_in_trash",
                task_cls="IThorKitchenCannedDrinkInTrash",
            ),
        )

    def _get_obj_cfgs(self):
        path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", self.CAN_XML)
        )
        return [dict(name=self.CAN_NAME, obj_groups=path)]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualCannedDrinkFridgeTrashSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        regions = {
            region.stable_id: region for region in self.get_region(return_all=True)
        }
        self.fridge_receptacle_region = regions.get(spec["fridge_receptacle_region"])
        self.trash_fixture = self.get_fixture(id=spec["trash_fixture"])
        self.trash_region = regions.get(spec["trash_region"])
        if self.fridge_receptacle_region is None:
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: missing fridge region"
            )
        if self.fridge_receptacle_region.fixture_id != self.target_fixture.stable_id:
            raise ValueError("fridge region belongs to another fixture")
        if self.trash_fixture is None or self.trash_region is None:
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: missing trash receptacle"
            )
        if self.trash_region.fixture_id != self.trash_fixture.stable_id:
            raise ValueError("trash region belongs to another fixture")
        self.fixture_refs["trash_can"] = self.trash_fixture
        self.region_refs["fridge_receptacle"] = self.fridge_receptacle_region
        self.region_refs["trash_receptacle"] = self.trash_region

    def _setup_references(self):
        super()._setup_references()
        self._stool_seat_geom_id = self.sim.model.geom_name2id(self.STOOL_SEAT_GEOM)

    def _spawn_clearable_fixture_roots(self):
        roots = super()._spawn_clearable_fixture_roots()
        return roots - {self.sim.model.body_name2id(self.STOOL_BODY)}

    def _before_object_placement(self):
        super()._before_object_placement()
        self._move_stool_out_of_trash_path()
        self._place_canned_drink_in_fridge()

    def _move_stool_out_of_trash_path(self):
        target_xy = np.asarray(self._get_task_spec()["stool_xy"], dtype=float)
        if target_xy.shape != (2,):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: stool_xy must contain XY"
            )
        qpos = self.sim.data.get_joint_qpos(self.STOOL_JOINT).copy()
        seat_xy = np.asarray(
            self.sim.data.geom_xpos[self._stool_seat_geom_id], dtype=float
        )[:2]
        qpos[:2] += target_xy - seat_xy
        self.sim.data.set_joint_qpos(self.STOOL_JOINT, qpos)
        start, end = self.sim.model.get_joint_qvel_addr(self.STOOL_JOINT)
        self.sim.data.qvel[start:end] = 0.0
        self.sim.forward()

    def _fridge_support_region(self):
        center, yaw = self.fridge_receptacle_region.world_pose(self.sim)
        center = np.asarray(center, dtype=float).copy()
        center[2] -= float(self.fridge_receptacle_region.half_size[2])
        can_xy = np.asarray(self._get_task_spec()["can_xy"], dtype=float)
        if can_xy.shape != (2,):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: can_xy must contain XY"
            )
        center[:2] = can_xy
        half_xy = np.minimum(
            np.asarray(self.fridge_receptacle_region.half_size[:2], dtype=float),
            np.array([0.12, 0.18]),
        )
        return SemanticRegion(
            stable_id=f"{self.scene_name}:task:{self.SPEC_TASK}:canned_drink_support",
            fixture_id=self.target_fixture.stable_id,
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.array([half_xy[0], half_xy[1], 0.02]),
            _world_center=center,
            _world_yaw=float(yaw),
        )

    def _place_canned_drink_in_fridge(self):
        support = self._fridge_support_region()
        center, _ = support.world_pose(self.sim)
        obj = self.objects[self.CAN_NAME]
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.forward()
        if not self.place_object_on_support_contact(
            self.CAN_NAME,
            support,
            self.target_fixture,
            xy=center[:2],
            inward_fractions=(0.0,),
            probe_above=0.12,
            probe_below=0.08,
        ):
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: canned drink has no stable "
                "support inside the fridge"
            )
        contacted_bottom = self.object_bottom_z(self.CAN_NAME)
        self._set_task_object_bottom_at(
            self.CAN_NAME,
            self.object_pos(self.CAN_NAME)[:2],
            contacted_bottom - 0.001,
        )

    def _canned_drink_in_fridge(self):
        # Center containment tolerates small shifts on the fridge shelf.
        center, yaw = self.fridge_receptacle_region.world_pose(self.sim)
        center = np.asarray(center, dtype=float)
        half = np.asarray(self.fridge_receptacle_region.half_size, dtype=float)
        position = self.object_pos(self.CAN_NAME)
        delta = position - center
        cosine, sine = np.cos(-yaw), np.sin(-yaw)
        local = delta.copy()
        local[0] = cosine * delta[0] - sine * delta[1]
        local[1] = sine * delta[0] + cosine * delta[1]
        return bool(np.all(np.abs(local) <= half + 0.04))

    def _canned_drink_removed_active_stage(self):
        return self._active_stage_condition(not self._canned_drink_in_fridge())

    def _canned_drink_in_trash(self):
        center, yaw = self.trash_region.world_pose(self.sim)
        center = np.asarray(center, dtype=float)
        half = np.asarray(self.trash_region.half_size, dtype=float)
        pos = self.object_pos(self.CAN_NAME)
        c, s = np.cos(-yaw), np.sin(-yaw)
        delta = pos[:2] - center[:2]
        local_xy = np.array([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]])
        center_inside = bool(
            np.all(np.abs(local_xy) <= half[:2] + self.TRASH_CENTER_XY_TOL)
            and pos[2] >= center[2] - half[2] - self.TRASH_CENTER_Z_TOL
            and pos[2] <= center[2] + half[2] + self.TRASH_CENTER_Z_TOL
        )
        return bool(center_inside and not self.g1_hand_grasping_object(self.CAN_NAME))

    def _success_region_bounds(self):
        return []

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            task_id=type(self).__name__,
            initial_canned_drink_location="near_front_of_open_fridge",
            initial_stool_state="moved_away_from_trash_can",
            trash_fixture=self.trash_fixture.stable_id,
            success_definition=(
                "after removing the canned drink from the open fridge, its released "
                "center must remain inside the trash can for 1 second"
            ),
        )
        return meta


__all__ = ["IThorKitchenCannedDrinkFromFridgeToTrash"]
