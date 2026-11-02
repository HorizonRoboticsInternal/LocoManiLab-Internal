"""iTHOR kitchen: put an apple into an initially open cabinet, then close it."""

import numpy as np
from robosuite.utils.mjcf_utils import find_elements, new_site
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


class IThorKitchenAppleCabinetPnP(
    CompositeStageReporterMixin,
    RoomOpenArticulation,
    IThorKitchenEnv,
):
    ATOMIC_SKILL = "AppleCabinetPnP"
    DIFFICULTY_LEVEL = "L3"
    TASK_FAMILY = "composite"
    COMPOSITE_STAGES = ("pick_apple", "put_apple_in_cabinet", "close_cabinet")
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_kitchen_task_spec)
    SPEC_TASK = "apple_cabinet_pnp"
    BEHAVIOR = "close"
    TARGET_CATEGORIES = ("Cabinet",)
    TARGET_LANG = "cabinet door"
    TARGET_JOINT_ROLES = ("door",)
    SUCCESS_JOINT_SELECTION = "matching"
    INITIAL_OBJECT_JITTER_RADIUS = 0.02
    TARGET_REGION_SITE_NAME = "apple_cabinet_pnp_target_region"

    APPLE_ACCEPTANCE_XY_TOL = 0.03
    APPLE_ACCEPTANCE_Z_TOL = 0.03

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        if self.target_fixture is not None:
            self.fixture_refs["target_cabinet"] = self.target_fixture
        self.cabinet_receptacle_region = self._require_cabinet_receptacle_region()
        if self.cabinet_receptacle_region is None:
            stable_id = self._get_task_spec()["cabinet_receptacle_region"]
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: unknown "
                f"cabinet_receptacle_region {stable_id!r}"
            )
        self.region_refs["cabinet_receptacle"] = self.cabinet_receptacle_region
        if bool(self._get_task_spec()["show_target_region"]):
            self._add_cabinet_receptacle_site(self.cabinet_receptacle_region)

    def _add_cabinet_receptacle_site(self, region):
        parent = None
        if region.parent_body is not None:
            parent = find_elements(
                self.mujoco_arena.worldbody,
                tags="body",
                attribs={"name": region.parent_body},
                return_first=True,
            )
        if parent is None:
            parent = self.mujoco_arena.worldbody
            center, yaw = region.world_pose()
        else:
            center = region.local_center
            yaw = region.local_yaw
        parent.append(
            new_site(
                name=self.TARGET_REGION_SITE_NAME,
                type="box",
                pos=[float(value) for value in center],
                size=[float(value) for value in region.half_size],
                quat=[np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
                rgba=[0.1, 0.9, 0.2, 0.35],
                group=0,
            )
        )

    def _get_obj_cfgs(self):
        return [dict(name="apple", obj_groups="apple")]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualAppleCabinetPnPSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _setup_scene(self):
        super()._setup_scene()
        self.success_hold_seconds = float(
            self._get_task_spec()["hold_seconds"]
        )
        self.cabinet_receptacle_region = self.region_refs["cabinet_receptacle"]
        self._reset_task_progress()

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_task_progress()

    def _reset_task_progress(self):
        self._success_hold = 0
        self._reported_stages = set()
        self._task_success_hold_steps = 0

    def _require_cabinet_receptacle_region(self):
        stable_id = self._get_task_spec()["cabinet_receptacle_region"]
        region = next(
            (
                region
                for region in self.get_region(return_all=True)
                if region.stable_id == stable_id
            ),
            None,
        )
        if region is not None and region.kind != "receptacle":
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: "
                "cabinet_receptacle_region must reference a receptacle, got "
                f"{region.kind!r} from {stable_id!r}"
            )
        return region

    def _before_object_placement(self):
        super()._before_object_placement()
        spec = self._get_task_spec()
        pos = np.asarray(spec["apple_pos"], dtype=float)
        if pos.shape != (3,):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: apple_pos must contain XYZ"
            )
        obj = self.objects["apple"]
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[:2] = pos[:2]
        qpos[2] = float(pos[2] - obj.bottom_offset[2])
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        qvel_addr = self.sim.model.get_joint_qvel_addr(obj.joints[0])
        if isinstance(qvel_addr, tuple):
            self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = 0.0
        else:
            self.sim.data.qvel[qvel_addr] = 0.0
        self.sim.forward()

    def _apple_inside_cabinet(self):
        return bool(
            self.object_in_region(
                "apple",
                self.cabinet_receptacle_region,
                rim_tol=self.APPLE_ACCEPTANCE_Z_TOL,
                xy_tol=self.APPLE_ACCEPTANCE_XY_TOL,
            )
        )

    def _task_success_condition(self):
        return bool(
            self._apple_inside_cabinet()
            and self.object_released("apple")
            and all(
                joint.fraction(self.sim) <= self.SUCCESS_CLOSE_FRACTION
                for joint in self._success_joints()
            )
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        if 1 not in self._reported_stages and self.g1_hand_grasping_object("apple"):
            self._reported_stages.add(1)
            self._report_stage_complete(1, 3, next_stage=self.COMPOSITE_STAGES[1])
        apple_placed = bool(
            self._apple_inside_cabinet() and self.object_released("apple")
        )
        if 2 not in self._reported_stages and apple_placed:
            self._reported_stages.add(2)
            self._report_stage_complete(2, 3, next_stage=self.COMPOSITE_STAGES[2])
        task_success = self._task_success_condition()
        if 3 not in self._reported_stages and task_success:
            self._reported_stages.add(3)
            self._report_stage_complete(3, 3)
        self._task_success_hold_steps = (
            self._task_success_hold_steps + 1 if task_success else 0
        )
        return ret

    def _check_success(self):
        required_steps = max(
            1, int(round(self.success_hold_seconds * self.control_freq))
        )
        return self._task_success_hold_steps >= required_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "take the apple from the table, put it inside the open cabinet, "
            "then close the cabinet"
        )
        ep_meta["composite_stages"] = list(self.COMPOSITE_STAGES)
        return ep_meta


__all__ = ["IThorKitchenAppleCabinetPnP"]
