"""iTHOR kitchen: open a drawer, take an apple out, and place it on a table."""

import numpy as np
from robosuite.utils.mjcf_utils import new_site

from robocasa.environments.ithor_kitchen.tasks.composite.drawer_pnp import (
    IThorKitchenDrawerPnP,
)
from robocasa.models.scenes.thor_semantics import SemanticRegion


class _CalibrateCollidingExactPoseMixin:
    def _finalize_articulation_robot_pose(self, distance):
        if (
            self.scene_name == "FloorPlan25"
            and self._get_articulation_task_spec() is not None
        ):
            self.sim.forward()
            if self.count_body_collisions() == 0:
                return True
            return self._calibrate_robot_free_joint_height()
        return super()._finalize_articulation_robot_pose(distance)


class IThorKitchenOpenDrawerTakeApple(
    _CalibrateCollidingExactPoseMixin, IThorKitchenDrawerPnP
):
    ATOMIC_SKILL = "OpenDrawerTakeApple"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    SPEC_TASK = "open_drawer_take_apple"
    COMPOSITE_STAGES = ("open_drawer", "put_apple_on_table")
    TARGET_REGION_SITE_NAME = "open_drawer_take_apple_target_region"

    APPLE_XML = "objaverse/apple/apple_0/model.xml"

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()["target_region"]
        center = np.asarray(spec["center"], dtype=float)
        half_xy = np.asarray(spec["half_xy"], dtype=float)
        z_tolerance = float(spec["z_tolerance"])
        yaw = float(spec.get("yaw", 0.0))
        if (
            center.shape != (3,)
            or half_xy.shape != (2,)
            or np.any(half_xy <= 0.0)
            or z_tolerance < 0.0
        ):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}/target_region: expected "
                "center XYZ, positive half_xy, and nonnegative z_tolerance"
            )
        self.apple_region = SemanticRegion(
            stable_id=f"{self.scene_name}:task:{self.SPEC_TASK}:region:target",
            fixture_id=self.target_fixture.stable_id,
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.array([half_xy[0], half_xy[1], z_tolerance]),
            _world_center=center,
            _world_yaw=yaw,
        )
        self.region_refs["apple_surface"] = self.apple_region
        if bool(self._get_task_spec()["show_target_region"]):
            self.mujoco_arena.worldbody.append(
                new_site(
                    name=self.TARGET_REGION_SITE_NAME,
                    type="box",
                    pos=[float(center[0]), float(center[1]), float(center[2]) + 0.012],
                    size=[float(half_xy[0]), float(half_xy[1]), 0.01],
                    quat=[np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
                    rgba=[0.1, 0.9, 0.2, 0.35],
                    group=0,
                )
            )

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.apple_region is not None)

    def _get_obj_cfgs(self):
        return [
            dict(
                name="apple",
                info=dict(mjcf_path=self.APPLE_XML),
            )
        ]

    def _setup_scene(self):
        super()._setup_scene()
        self.apple_region = self.region_refs["apple_surface"]

    def _apple_on_table(self):
        center, local, bottom, _ = self._object_region_bounds(
            "apple", self.apple_region
        )
        return bool(
            np.all(np.abs(local) <= self.apple_region.half_size[:2])
            and abs(bottom - float(center[2])) <= self.apple_region.half_size[2]
        )

    def _drawer_receptacle_bounds(self):
        return None

    def _update_placement_progress(self):
        # Placing on the table is this task's final stage; there is no insert stage.
        pass

    def _task_success_condition(self):
        return bool(
            self._drawer_opened
            and self.target_semantic_joint.fraction(self.sim)
            >= self.SUCCESS_OPEN_FRACTION
            and self._apple_on_table()
            and (
                self.object_released("apple")
                if self.scene_name == "FloorPlan13"
                else not any(
                    self.g1_hand_object_contact(side, "apple")
                    for side in ("left", "right")
                )
            )
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "open the drawer, take out the apple, and place it on the table"
        return ep_meta


__all__ = ["IThorKitchenOpenDrawerTakeApple"]
