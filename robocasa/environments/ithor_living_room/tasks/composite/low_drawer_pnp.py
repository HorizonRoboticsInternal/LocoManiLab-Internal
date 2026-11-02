"""Take an apple from the furniture top, put it in its low drawer, and close it."""

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
)
from robocasa.environments.ithor_living_room.tasks.atomic.articulation import (
    IThorLivingRoomOpenDrawer,
)


class IThorLivingRoomLowDrawerPnP(
    IThorMultiStageCompositeMixin, IThorLivingRoomOpenDrawer,
):
    SUPPORTED_SCENE_IDS = (209, 210, 226)
    SPEC_TASK = "low_drawer_pnp"
    ATOMIC_SKILL = "LowDrawerPnP"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    COMPOSITE_KIND = "open_low_drawer_place_apple_and_close"
    COMPOSITE_ORDERING = "sequential"
    APPLE_XML = "objaverse/apple/apple_0/model.xml"
    INITIAL_POSE_JITTER = False
    ATOMIC_TASKS = (
        IThorAtomicTask("open_low_drawer", "_raw_success", hold_steps=5),
        IThorAtomicTask("put_apple_in_drawer_and_close", "_apple_inside_and_drawer_closed"),
    )

    def _get_obj_cfgs(self):
        return [dict(name="apple", info=dict(mjcf_path=self.APPLE_XML), object_scale=1.0)]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualLowDrawerAppleSampler")

    def _setup_scene(self):
        super()._setup_scene()
        spec = self._get_task_spec()
        model = self.sim.model
        self._drawer_receptacle_site_id = model.site_name2id(spec["drawer_receptacle_site"])
        drawer_body = int(model.jnt_bodyid[model.joint_name2id(self.target_joint)])
        if int(model.site_bodyid[self._drawer_receptacle_site_id]) != drawer_body:
            raise ValueError(f"{self.scene_name}: receptacle must belong to the target drawer")
        self._apple_support_geom_id = model.geom_name2id(spec["apple_support_geom"])
        self.SUCCESS_CLOSE_FRACTION = float(spec["success_close_fraction"])
        self.sim.data.set_joint_qvel(self.target_joint, 0.0)

    def _before_object_placement(self):
        super()._before_object_placement()
        geom_id = self._apple_support_geom_id
        rotation = self.sim.data.geom_xmat[geom_id].reshape(3, 3)
        top_z = float(
            self.sim.data.geom_xpos[geom_id][2]
            + np.abs(rotation[2]) @ self.sim.model.geom_size[geom_id]
        )
        # The fixed upright apple's bbox matches its mesh bottom. Rotated
        # per-mesh AABBs are looser and can leave this asset above the surface.
        set_free_joint_bottom_at(
            self, self.objects["apple"],
            [*self._get_task_spec()["apple_xy"], top_z - 0.001], yaw=0.0,
        )
        self.sim.forward()

    def _raw_success(self):
        return bool(
            self.target_semantic_joint.fraction(self.sim)
            >= self.SUCCESS_OPEN_FRACTION - 1e-8
        )

    def _apple_inside_drawer(self):
        site_id = self._drawer_receptacle_site_id
        center = np.mean(self.object_bbox_points("apple"), axis=0)
        rotation = self.sim.data.site_xmat[site_id].reshape(3, 3)
        local = (center - self.sim.data.site_xpos[site_id]) @ rotation
        return bool(np.all(np.abs(local) <= self.sim.model.site_size[site_id]))

    def _apple_inside_and_drawer_closed(self):
        return bool(
            self.target_semantic_joint.fraction(self.sim)
            <= self.SUCCESS_CLOSE_FRACTION + 1e-8
            and self._apple_inside_drawer()
        )

    def _required_stage_hold_steps(self, task):
        if task.hold_steps is not None:
            return int(task.hold_steps)
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        if self.scene_name == "FloorPlan226":
            lang = (
                "pick up the apple from the coffee table, open the second drawer "
                "from the top on its front side, put the apple inside, and close the drawer"
            )
        elif self.scene_name == "FloorPlan210":
            lang = (
                "pick up the apple from the top of the dresser, open its lower-right "
                "drawer, put the apple inside, and close the drawer"
            )
        else:
            lang = (
                "pick up the apple from the top of the dresser, open its lower-left "
                "drawer, put the apple inside, and close the drawer"
            )
        meta.update(
            lang=lang,
            composite_stages=[task.name for task in self._atomic_tasks],
            composite_ordering=self.COMPOSITE_ORDERING,
            target_object="apple",
            drawer_receptacle_site=self._get_task_spec()["drawer_receptacle_site"],
            init_open_fraction=self.OPEN_INIT_FRACTION,
            success_open_fraction=self.SUCCESS_OPEN_FRACTION,
            success_close_fraction=self.SUCCESS_CLOSE_FRACTION,
            success_definition=(
                "open the named low drawer to at least 80% for 5 control steps, "
                "then keep the apple center in its moving receptacle and the drawer "
                "at most 20% open for 1 second"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomLowDrawerPnP"]
