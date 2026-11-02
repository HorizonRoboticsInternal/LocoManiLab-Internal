"""Two exact native-object atomic tasks for the toilet-side upper 410 drawer."""

from dataclasses import replace

import numpy as np

from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_specs,
    get_supported_scene_ids,
)

from .object_manipulation import _BathroomWorkspaceMixin


class _BathroomDrawerTask(_BathroomWorkspaceMixin):
    """Stage a native object beside or inside one exact, initially open drawer."""

    DIFFICULTY_LEVEL = "L1"
    HOLD_STEPS = 5
    SOURCE_IN_DRAWER = False
    BOUNDS_TOLERANCE = 0.005
    REQUIRED_SPEC = (
        "source_fixture",
        "source_xy",
        "source_quat",
        "surface_fixture",
        "surface_region",
        "drawer_fixture",
        "drawer_joint",
        "drawer_region",
        "drawer_region_z_offset",
        "robot_pose",
        "lang",
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.source_fixture = self._exact_fixture(spec["source_fixture"])
        self.drawer_fixture = self._exact_fixture(spec["drawer_fixture"])
        self.drawer_joint = None
        self.drawer_region = None
        if self.drawer_fixture is not None:
            self.drawer_joint = next(
                (
                    joint
                    for joint in self.drawer_fixture.joints.values()
                    if joint.name == spec["drawer_joint"]
                ),
                None,
            )
            self.drawer_region = next(
                (
                    region
                    for region in self.drawer_fixture.regions.values()
                    if region.stable_id == spec["drawer_region"]
                ),
                None,
            )
            self.fixture_refs["target_drawer"] = self.drawer_fixture
        if self.source_fixture is not None:
            self.fixture_refs["source_object"] = self.source_fixture
        if self.drawer_region is not None:
            # Align this task-local volume with the drawer's physical bottom.
            offset = np.asarray(self.drawer_region._world_offset).copy()
            offset[2] += float(spec["drawer_region_z_offset"])
            self.drawer_region = replace(self.drawer_region, _world_offset=offset)
            # A drawer is a deep receptacle: its lower face is the source support.
            offset = offset.copy()
            offset[2] -= self.drawer_region.half_size[2]
            self.drawer_support_region = replace(
                self.drawer_region,
                kind="support",
                _world_offset=offset,
            )
            self.region_refs["target_drawer"] = self.drawer_region

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.source_fixture is not None
            and self.drawer_joint is not None
            and self.drawer_region is not None
        )

    def _get_obj_cfgs(self):
        return []

    def _setup_scene(self):
        self.sim.data.set_joint_qpos(
            self.drawer_joint.name,
            self.drawer_joint.open_qpos,
        )
        self.sim.data.set_joint_qvel(self.drawer_joint.name, 0.0)
        self.sim.forward()
        super()._setup_scene()

    def _reset_internal(self):
        super()._reset_internal()
        spec = self.scene_task_spec()
        self._set_fixture_quaternion(self.source_fixture, spec["source_quat"])
        support = (
            self.drawer_support_region
            if self.SOURCE_IN_DRAWER
            else self.work_surface_region
        )
        fixture = (
            self.drawer_fixture if self.SOURCE_IN_DRAWER else self.work_surface_fixture
        )
        if not self._place_fixture_on_support(
            self.source_fixture,
            support,
            fixture,
            spec["source_xy"],
            contract_name="drawer_task_source_support",
        ):
            raise RuntimeError(
                f"{self.scene_name}: no stable source pose for {self.source_fixture.stable_id}"
            )
        if self.count_body_collisions():
            raise RuntimeError(
                f"{self.scene_name}: drawer task staging collides with G1"
            )
        self._record_source_horizontal_distance(self.source_fixture)
        self._success_hold = 0

    def _source_bounds_in_drawer_frame(self):
        center, yaw = self.drawer_region.world_pose(self.sim)
        points = self._fixture_bbox_points(self.source_fixture) - center
        c, s = np.cos(yaw), np.sin(yaw)
        points[:, :2] = points[:, :2] @ np.array([[c, -s], [s, c]])
        return points.min(axis=0), points.max(axis=0)

    def _source_inside_drawer(self):
        lower, upper = self._source_bounds_in_drawer_frame()
        half = self.drawer_region.half_size
        return bool(
            np.all(lower >= -half - self.BOUNDS_TOLERANCE)
            and np.all(upper <= half + self.BOUNDS_TOLERANCE)
        )

    def _source_clear_of_drawer(self):
        lower, upper = self._source_bounds_in_drawer_frame()
        half = self.drawer_region.half_size
        return bool(np.any(lower > half + 0.01) or np.any(upper < -half - 0.01))

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold = self._success_hold + 1 if self._raw_success() else 0
        return ret

    def _check_success(self):
        return self._success_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            {
                "lang": self.scene_task_spec()["lang"],
                "source_fixture": self.source_fixture.stable_id,
                "target_fixture": self.drawer_fixture.stable_id,
                "target_region": self.drawer_region.stable_id,
                "source_horizontal_distance": self._source_horizontal_distance,
                "success_definition": f"{self.SUCCESS_DESCRIPTION} for {self.HOLD_STEPS} control steps",
            }
        )
        return meta


class IThorBathroomToiletPaperRollIntoDrawer(_BathroomDrawerTask, IThorBathroomEnv):
    """Rest the native roll on the toilet-side upper drawer bottom in any pose."""

    ATOMIC_SKILL = "PickPlace"
    TASK_FAMILY = "pick_place"
    SPEC_TASK = "toilet_paper_roll_into_drawer"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK,
        required=_BathroomDrawerTask.REQUIRED_SPEC,
    )
    SUCCESS_DESCRIPTION = (
        "the roll contacts the target drawer bottom support surface "
        "in any orientation, regardless of grasp state"
    )

    def _source_contacts_drawer_bottom(self):
        source_bodies = self._fixture_body_ids(
            self.source_fixture,
            include_siblings=True,
        )
        drawer_bodies = self._fixture_body_ids(self.drawer_fixture)
        bottom_z = float(self.drawer_support_region.world_pose(self.sim)[0][2])
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if body1 in drawer_bodies and body2 in source_bodies:
                normal_z = float(contact.frame[2])
            elif body2 in drawer_bodies and body1 in source_bodies:
                normal_z = -float(contact.frame[2])
            else:
                continue
            # Use the contact itself: a tilted roll may extend above the rim.
            # The bbox support plane is 12 mm below the actual bottom contacts.
            if (
                -self.CONTACT_PENETRATION_TOLERANCE <= contact.dist <= 0.001
                and normal_z >= 0.5
                and abs(float(contact.pos[2]) - bottom_z) <= 0.02
                and self.drawer_support_region.contains_xy(contact.pos, self.sim)
            ):
                return True
        return False

    def _raw_success(self):
        return self._source_contacts_drawer_bottom()


class IThorBathroomTakeSoapBottleFromDrawer(_BathroomDrawerTask, IThorBathroomEnv):
    """Lift the native soap bottle 0.30 m from the drawer, regardless of grasp."""

    ATOMIC_SKILL = "LiftObject"
    TASK_FAMILY = "lift"
    SPEC_TASK = "take_soap_bottle_from_drawer"
    SOURCE_IN_DRAWER = True
    MIN_LIFT_DELTA = 0.30
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK,
        required=_BathroomDrawerTask.REQUIRED_SPEC,
    )
    SUCCESS_DESCRIPTION = (
        f"the soap bottle is at least {MIN_LIFT_DELTA:.2f} m above its reset height, "
        "regardless of grasp or hand contact"
    )

    def _source_root_z(self):
        body_id = self.sim.model.body_name2id(self.source_fixture.body_name)
        return float(self.sim.data.body_xpos[body_id, 2])

    def _reset_internal(self):
        super()._reset_internal()
        self._initial_source_z = self._source_root_z()

    def _raw_success(self):
        return self._source_root_z() >= self._initial_source_z + self.MIN_LIFT_DELTA


__all__ = [
    "IThorBathroomToiletPaperRollIntoDrawer",
    "IThorBathroomTakeSoapBottleFromDrawer",
]
