"""iTHOR kitchen: open a drawer, put an apple inside, then close it."""

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


class IThorKitchenDrawerPnP(
    CompositeStageReporterMixin,
    RoomOpenArticulation,
    IThorKitchenEnv,
):
    ATOMIC_SKILL = "DrawerPnP"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_STAGES = ("open_drawer", "put_apple_in_drawer", "close_drawer")
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_kitchen_task_spec)
    SPEC_TASK = "drawer_pnp"
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Drawer",)
    TARGET_LANG = "drawer"
    TARGET_JOINT_ROLES = ("drawer",)
    TARGET_JOINT_TYPES = ("slide",)
    OPEN_STAND_DISTANCE = 0.40
    APPLE_ACCEPTANCE_XY_TOL = 0.005
    APPLE_ACCEPTANCE_Z_TOL = 0.02
    INITIAL_OBJECT_JITTER_RADIUS = 0.01

    def _get_obj_cfgs(self):
        return [dict(name="apple", obj_groups="apple")]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name=f"Manual{self.ATOMIC_SKILL}Sampler")

    def _semantic_placements_are_valid(self):
        return True

    def _setup_scene(self):
        super()._setup_scene()
        self.success_hold_seconds = float(
            self._get_task_spec()["hold_seconds"]
        )
        region_id = self._get_task_spec()["drawer_receptacle_region"]
        self.drawer_receptacle_region = next(
            (
                region
                for region in self.semantic_scene.get_region(return_all=True)
                if region.stable_id == region_id
            ),
            None,
        )
        if self.drawer_receptacle_region is None:
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: unknown "
                f"drawer_receptacle_region {region_id!r}"
            )
        self._bind_drawer_receptacle_site()
        self._reset_task_progress()

    def _bind_drawer_receptacle_site(self):
        model = self.sim.model
        drawer_body = model.jnt_bodyid[model.joint_name2id(self.target_joint)]
        candidates = [
            model.site_name2id(name)
            for name in self.target_fixture.site_names
            if "receptacle" in name.lower()
            and model.site_bodyid[model.site_name2id(name)] == drawer_body
        ]
        if (
            len(candidates) != 1
            or self.drawer_receptacle_region.parent_body
            != model.body_id2name(drawer_body)
        ):
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: expected one receptacle "
                "site on the target drawer's moving body"
            )
        self._drawer_receptacle_site_id = candidates[0]
        self._drawer_contact_geoms = self._geom_names_for_bodies(
            self._body_tree_ids([int(drawer_body)])
        )

    def _before_object_placement(self):
        super()._before_object_placement()
        self._place_apple_from_spec()

    def _place_apple_from_spec(self):
        spec = self._get_task_spec()
        pos = np.asarray(spec["apple_pos"], dtype=float)
        if pos.shape != (3,):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: apple_pos must contain XYZ"
            )
        self._set_free_joint_bottom_at(self.objects["apple"], pos[:2], pos[2])
        self.sim.forward()

    def _set_free_joint_bottom_at(self, obj, xy, bottom_z):
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[:2] = np.asarray(xy, dtype=float)[:2]
        qpos[2] = float(bottom_z - obj.bottom_offset[2])
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        qvel_addr = self.sim.model.get_joint_qvel_addr(obj.joints[0])
        if isinstance(qvel_addr, tuple):
            self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = 0.0
        else:
            self.sim.data.qvel[qvel_addr] = 0.0

    def _reset_task_progress(self):
        self._success_hold = 0
        self._task_success_hold_steps = 0
        self._drawer_opened = False
        self._apple_placed = False
        self._final_stage_reported = False

    def _post_action(self, action):
        if not self._drawer_opened and (
            self.target_semantic_joint.fraction(self.sim)
            >= self.SUCCESS_OPEN_FRACTION
        ):
            self._drawer_opened = True
            self._report_stage_complete(
                1,
                len(self.COMPOSITE_STAGES),
                next_stage=self.COMPOSITE_STAGES[1],
            )
        elif self._drawer_opened:
            self._update_placement_progress()
        self._task_success_hold_steps = (
            self._task_success_hold_steps + 1 if self._task_success_condition() else 0
        )
        if self._check_success() and not self._final_stage_reported:
            self._final_stage_reported = True
            self._report_stage_complete(
                len(self.COMPOSITE_STAGES), len(self.COMPOSITE_STAGES)
            )
        return super()._post_action(action)

    def _update_placement_progress(self):
        if (
            not self._apple_placed
            and self.target_semantic_joint.fraction(self.sim)
            > float(self._get_task_spec()["success_close_fraction"])
            and self._apple_inside_drawer()
        ):
            self._apple_placed = True
            self._report_stage_complete(
                2, len(self.COMPOSITE_STAGES), next_stage=self.COMPOSITE_STAGES[2]
            )

    def _drawer_receptacle_pose(self):
        site_id = self._drawer_receptacle_site_id
        return (
            np.asarray(self.sim.data.site_xpos[site_id], dtype=float).copy(),
            np.asarray(self.sim.model.site_size[site_id], dtype=float).copy(),
            np.asarray(self.sim.data.site_xmat[site_id], dtype=float).reshape(3, 3),
        )

    def _apple_inside_drawer(self):
        center, half_size, rotation = self._drawer_receptacle_pose()
        # THOR drawer bodies use local Y as vertical. Transform all three
        # axes, including the site's own rotation, instead of using yaw only.
        local_points = (self.object_bbox_points("apple") - center) @ rotation
        world_tolerance = np.array(
            [
                self.APPLE_ACCEPTANCE_XY_TOL,
                self.APPLE_ACCEPTANCE_XY_TOL,
                self.APPLE_ACCEPTANCE_Z_TOL,
            ]
        )
        local_tolerance = np.abs(rotation).T @ world_tolerance
        return bool(
            np.all(np.abs(local_points) <= half_size + local_tolerance)
            and self.check_contact(self.objects["apple"], self._drawer_contact_geoms)
            and not any(
                self.g1_hand_object_contact(side, "apple")
                for side in ("left", "right")
            )
        )

    def _drawer_receptacle_bounds(self):
        if not bool(
            self._get_task_spec().get("show_target_region", False)
        ):
            return None
        if not hasattr(self, "_drawer_receptacle_site_id"):
            return None
        center, half_size, rotation = self._drawer_receptacle_pose()
        # Express the native site in horizontal world axes for the viewer.
        horizontal_axis = int(np.argmin(np.abs(rotation[2])))
        yaw = float(
            np.arctan2(rotation[1, horizontal_axis], rotation[0, horizontal_axis])
        )
        c, s = np.cos(yaw), np.sin(yaw)
        world_rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        half_size = np.abs(world_rotation.T @ rotation) @ half_size
        return center, half_size, yaw

    def _task_success_condition(self):
        return bool(
            self._drawer_opened
            and self._apple_placed
            and self._apple_inside_drawer()
            and self.target_semantic_joint.fraction(self.sim)
            <= float(self._get_task_spec()["success_close_fraction"])
        )

    def _check_success(self):
        required_steps = max(
            1, int(round(self.success_hold_seconds * self.control_freq))
        )
        return self._task_success_hold_steps >= required_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta[
            "lang"
        ] = "open the drawer in front of you, put the apple inside, then close the drawer"
        return ep_meta


__all__ = ["IThorKitchenDrawerPnP"]
