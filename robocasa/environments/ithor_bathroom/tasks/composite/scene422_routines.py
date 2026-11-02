"""Two ordered bathroom routines whose layouts are isolated to scene 422."""

from copy import deepcopy
from time import monotonic

import mujoco
import numpy as np

from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv

from ..floor_lift import IThorBathroomLiftCandleFromFloor, _BathroomCandleStoolTask
from ..storage_articulation import (
    IThorBathroomOpenUpperCabinetDoor,
    _BathroomUpperCabinetArticulation,
)
from .drawer_routines import _OrderedDrawerRoutineMixin


class IThorBathroomOpenThenCloseUpperCabinetDoorAndApproachSink(
    _OrderedDrawerRoutineMixin, _BathroomUpperCabinetArticulation, IThorBathroomEnv
):
    """Cycle the upper cabinet door, then walk to the separate sink."""

    BEHAVIOR = "open"
    TARGET_JOINT_TYPES = ("hinge",)
    SUPPORTED_SCENE_IDS = (422,)
    SPEC_TASK = "open_then_close_upper_cabinet_door_and_approach_sink"
    COMPOSITE_KIND = "upper_cabinet_then_sink"
    COMPOSITE_ORDERING = (
        "open_upper_cabinet_door",
        "close_upper_cabinet_door",
        "approach_sink",
    )
    SUCCESS_DISTANCE = 0.35
    MIN_NAVIGATION_DISPLACEMENT = 0.30
    SCENE_SPECS = {
        "FloorPlan422": deepcopy(
            IThorBathroomOpenUpperCabinetDoor.SCENE_SPECS["FloorPlan422"]
        )
    }
    SCENE_SPECS["FloorPlan422"].update(
        sink_fixture="FloorPlan422:Sink_d76d2cbd_23",
        lang=(
            "Open the upper bathroom cabinet door, close the same door, "
            "then walk to the sink."
        ),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        stable_id = self.scene_task_spec()["sink_fixture"]
        self.sink_fixture = self.get_fixture(id=stable_id)
        if self.sink_fixture is None or self.sink_fixture.stable_id != stable_id:
            raise LookupError(f"{self.scene_name}: missing exact navigation sink")
        self.fixture_refs["navigation_sink"] = self.sink_fixture

    def _reset_task_progress(self):
        self._navigation_departure_xy = self._robot_base_xy().copy()
        super()._reset_task_progress()

    def _advance_drawer_routine(self):
        previous_stage = self._routine_stage
        super()._advance_drawer_routine()
        if previous_stage == 1 and self._routine_stage == 2:
            self._navigation_departure_xy = self._robot_base_xy().copy()
            self._routine_armed = True

    def _stage_goal(self, index):
        fraction = self.target_semantic_joint.fraction(self.sim)
        if index == 0:
            return fraction >= self.SUCCESS_OPEN_FRACTION
        if index == 1:
            return fraction <= self.SUCCESS_CLOSE_FRACTION
        return self._raw_success()

    def _raw_success(self):
        distance = self.robot_fixture_collision_distance(
            self.sink_fixture, max_distance=1.0
        )
        return bool(
            self.target_semantic_joint.fraction(self.sim) <= self.SUCCESS_CLOSE_FRACTION
            and np.isfinite(distance)
            and distance <= self.SUCCESS_DISTANCE
            and np.linalg.norm(self._robot_base_xy() - self._navigation_departure_xy)
            >= self.MIN_NAVIGATION_DISPLACEMENT
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            navigation_target_fixture=self.sink_fixture.stable_id,
            navigation_success_distance=self.SUCCESS_DISTANCE,
            minimum_navigation_displacement=self.MIN_NAVIGATION_DISPLACEMENT,
            success_definition=(
                "in order: open the exact upper door to at least 80%; close "
                "the same door to at most 20%; move at least 0.30 m after "
                "closing and reach within 0.35 m of the sink collision geometry, "
                "keeping the door closed for the final 1.0 s"
            ),
        )
        return meta


class IThorBathroomOpenDrawerThenLiftCandleFromStoolAndPutInDrawer(
    _OrderedDrawerRoutineMixin, _BathroomCandleStoolTask, IThorBathroomEnv
):
    """Open the cabinet drawer, lift its nearby low candle, and store it."""

    SUPPORTED_SCENE_IDS = (422,)
    SPEC_TASK = "open_drawer_then_lift_candle_from_stool_and_put_in_drawer"
    COMPOSITE_KIND = "drawer_candle_routine"
    COMPOSITE_ORDERING = (
        "open_drawer",
        "bend_down_and_lift_candle_from_stool",
        "put_candle_in_drawer",
    )
    SUCCESS_OPEN_FRACTION = 0.80
    INITIAL_DRAWER_FRACTION = 0.20
    BOUNDS_TOLERANCE = 0.005
    SCENE_SPECS = {
        "FloorPlan422": deepcopy(
            IThorBathroomLiftCandleFromFloor.SCENE_SPECS["FloorPlan401"]
        )
    }
    SCENE_SPECS["FloorPlan422"].update(
        source_xy=[-0.87953, 1.00641],
        stool_xy=[-0.87953, 1.00641],
        remove_fixtures=["FloorPlan422:Candle_5_11"],
        robot_pose={"xy": [-1.19715, 0.81885], "yaw": 0.0},
        source_horizontal_distance=float(np.hypot(0.31762, 0.18756)),
        drawer_fixture="FloorPlan422:Drawer_452cc903_0",
        drawer_joint="drawer_e69d502ac6388ef5b2b2cb0a94a4962b_1_1_0_joint_0",
        drawer_region="FloorPlan422:Drawer_452cc903_0:region:receptacle:0",
        drawer_bottom_geom="drawer_e69d502ac6388ef5b2b2cb0a94a4962b_1_1_0_collision_3",
        lang=(
            "Open the drawer, bend down to pick "
            "up the candle from the nearby low stool, then put the candle "
            "inside the drawer."
        ),
    )

    def _robot_facing_target_xy(self, source_xy):
        # Face along +X, 90 degrees counterclockwise from the -Y stance.
        robot_y = self.scene_task_spec()["robot_pose"]["xy"][1]
        return np.array([self.drawer_fixture.center[0], robot_y], dtype=float)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.drawer_fixture = self._exact_fixture(spec["drawer_fixture"])
        if self.drawer_fixture is None:
            raise LookupError(f"{self.scene_name}: missing exact candle drawer")
        self.drawer_joint = next(
            joint
            for joint in self.drawer_fixture.joints.values()
            if joint.name == spec["drawer_joint"]
        )
        self.drawer_region = self.drawer_fixture.regions[spec["drawer_region"]]
        self.fixture_refs["target_drawer"] = self.drawer_fixture
        self.region_refs["target_drawer"] = self.drawer_region

    def _setup_scene(self):
        qpos = self.drawer_joint.closed_qpos + self.INITIAL_DRAWER_FRACTION * (
            self.drawer_joint.open_qpos - self.drawer_joint.closed_qpos
        )
        self.sim.data.set_joint_qpos(self.drawer_joint.name, qpos)
        self.sim.data.set_joint_qvel(self.drawer_joint.name, 0.0)
        self.sim.forward()
        super()._setup_scene()
        self._drawer_bottom_geom_id = self.sim.model.geom_name2id(
            self.scene_task_spec()["drawer_bottom_geom"]
        )

        model = self.sim.model
        bodies = self._body_tree_ids([self.obj_body_id[self.TARGET_NAME]])
        self._candle_collision_meshes = []
        for geom in self._geom_ids_for_bodies(bodies):
            if not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
                raise RuntimeError("expected candle collision mesh geometry")
            mesh = int(model.geom_dataid[geom])
            start = int(model.mesh_vertadr[mesh])
            count = int(model.mesh_vertnum[mesh])
            self._candle_collision_meshes.append(
                (geom, np.asarray(model.mesh_vert[start : start + count]))
            )
        if not self._candle_collision_meshes:
            raise RuntimeError("candle has no active collision meshes")

    def _reset_task_progress(self):
        self._initial_source_z = self._source_root_z()
        super()._reset_task_progress()
        self._last_success_diagnostic_time = monotonic()

    def _advance_drawer_routine(self):
        previous_stage = self._routine_stage
        super()._advance_drawer_routine()
        if previous_stage == 1 and self._routine_stage == 2:
            # Lifting and placement can finish together; count the final hold
            # from this transition without requiring removal and reinsertion.
            self._routine_armed = True
        # self._print_success_diagnostic()

    def _print_success_diagnostic(self):
        now = monotonic()
        if now - self._last_success_diagnostic_time < 5.0:
            return
        self._last_success_diagnostic_time = now
        fraction = self.drawer_joint.fraction(self.sim)
        lift_delta = self._source_root_z() - self._initial_source_z
        illegal_pairs = self.illegal_contact_pairs_for_bodies(
            [self.obj_body_id[self.TARGET_NAME]],
            tuple(self._fixture_body_ids(self.drawer_fixture)),
            tolerance=self.SUPPORT_PENETRATION_TOLERANCE,
        )
        conditions = {
            "drawer_open": fraction >= self.SUCCESS_OPEN_FRACTION,
            "inside_drawer": self._candle_inside_drawer(),
            "bottom_contact": self._candle_contacts_drawer_bottom(),
            "no_illegal_penetration": not illegal_pairs,
        }
        pair_names = [
            tuple(self.sim.model.geom_id2name(geom) for geom in pair)
            for pair in sorted(illegal_pairs)
        ]
        required = (
            self.HOLD_STEPS
            if self._routine_stage == 2
            else self.INTERMEDIATE_HOLD_STEPS
        )
        details = " ".join(f"{name}={value}" for name, value in conditions.items())
        center, lower, upper, half = self._candle_drawer_bounds()
        outside = np.maximum(np.maximum(-half - lower, upper - half), 0.0)
        print(
            f"[Scene422 candle success] stage={self._routine_stage + 1}/3 "
            f"armed={self._routine_armed} stage_goal={self._stage_goal(self._routine_stage)} "
            f"hold={self._routine_hold}/{required}steps "
            f"({self._routine_hold / self.control_freq:.2f}/{required / self.control_freq:.2f}s) "
            f"{details} drawer_fraction={fraction:.4f}/{self.SUCCESS_OPEN_FRACTION:.4f} "
            f"lift_delta={lift_delta:.4f}/{self.MIN_LIFT_DELTA:.4f}m "
            f"placement_ok={all(conditions.values())} success={self._check_success()} "
            f"illegal_pairs={pair_names} "
            f"target_drawer={self.drawer_fixture.stable_id} "
            f"candle_xyz={np.round(self.object_pos(self.TARGET_NAME), 4).tolist()} "
            f"drawer_xyz={np.round(center, 4).tolist()} "
            f"bounds_source=collision_mesh "
            f"bbox_local_min={np.round(lower, 4).tolist()} "
            f"bbox_local_max={np.round(upper, 4).tolist()} "
            f"allowed_half={np.round(half, 4).tolist()} "
            f"outside_xyz_m={np.round(outside, 4).tolist()} "
            f"target_bottom={self.sim.model.geom_id2name(self._drawer_bottom_geom_id)} "
            f"candle_contacts={self._candle_contact_summary()}",
            flush=True,
        )

    def _candle_released(self):
        return not any(
            self.g1_hand_object_contact(side, self.TARGET_NAME)
            for side in ("left", "right")
        )

    def _candle_is_on_stool(self):
        return bool(
            abs(self._source_root_z() - self._initial_source_z) <= 0.02
            and self._candle_footprint_is_on_stool(
                self.scene_task_spec()["stool_top_half_xy"]
            )
            and self._object_fixture_contact_is_shallow(
                self.TARGET_NAME,
                self.task_stool_fixture,
                self.SUPPORT_PENETRATION_TOLERANCE,
            )
            and self._candle_released()
        )

    def _candle_collision_points(self):
        data = self.sim.data
        return np.concatenate(
            [
                vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                for geom, vertices in self._candle_collision_meshes
            ]
        )

    def _candle_drawer_bounds(self):
        center, yaw = self.drawer_region.world_pose(self.sim)
        # A tilted candle's box corners extend beyond its actual shape and can
        # cross the floor or side walls even while its collision meshes fit.
        points = self._candle_collision_points() - center
        c, s = np.cos(yaw), np.sin(yaw)
        points[:, :2] = points[:, :2] @ np.array([[c, -s], [s, c]])
        half = self.drawer_region.half_size + self.BOUNDS_TOLERANCE
        return center, points.min(axis=0), points.max(axis=0), half

    def _candle_inside_drawer(self):
        _, lower, upper, half = self._candle_drawer_bounds()
        return bool(np.all(lower >= -half) and np.all(upper <= half))

    def _candle_contact_summary(self):
        # Group contact points by the other geom to keep the five-second log short.
        bodies = self._body_tree_ids([self.obj_body_id[self.TARGET_NAME]])
        contacts = {}
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if body1 in bodies and body2 not in bodies:
                other, normal_z = contact.geom2, -float(contact.frame[2])
            elif body2 in bodies and body1 not in bodies:
                other, normal_z = contact.geom1, float(contact.frame[2])
            else:
                continue
            name = self.sim.model.geom_id2name(other)
            distance = float(contact.dist)
            bounds = contacts.setdefault(name, [distance, distance, normal_z])
            bounds[0] = min(bounds[0], distance)
            bounds[1] = max(bounds[1], distance)
            bounds[2] = max(bounds[2], normal_z)
        return {
            name: dict(
                dist_min_m=round(values[0], 5),
                dist_max_m=round(values[1], 5),
                up_normal_z=round(values[2], 4),
            )
            for name, values in contacts.items()
        }

    def _candle_contacts_drawer_bottom(self):
        candle_bodies = self._body_tree_ids([self.obj_body_id[self.TARGET_NAME]])
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            if (
                contact.geom1 == self._drawer_bottom_geom_id
                and int(self.sim.model.geom_bodyid[contact.geom2]) in candle_bodies
            ):
                normal_z = float(contact.frame[2])
            elif (
                contact.geom2 == self._drawer_bottom_geom_id
                and int(self.sim.model.geom_bodyid[contact.geom1]) in candle_bodies
            ):
                normal_z = -float(contact.frame[2])
            else:
                continue
            if (
                -self.SUPPORT_PENETRATION_TOLERANCE <= contact.dist <= 0.001
                and normal_z >= 0.5
            ):
                return True
        return False

    def _stage_goal(self, index):
        opened = self.drawer_joint.fraction(self.sim) >= self.SUCCESS_OPEN_FRACTION
        if index == 0:
            return bool(opened and self._candle_is_on_stool())
        if index == 1:
            return bool(
                opened
                and self._source_root_z()
                >= self._initial_source_z + self.MIN_LIFT_DELTA
            )
        return self._raw_success()

    def _raw_success(self):
        return bool(
            self.drawer_joint.fraction(self.sim) >= self.SUCCESS_OPEN_FRACTION
            and self._candle_inside_drawer()
            and self._candle_contacts_drawer_bottom()
            and not self.illegal_contact_pairs_for_bodies(
                [self.obj_body_id[self.TARGET_NAME]],
                tuple(self._fixture_body_ids(self.drawer_fixture)),
                tolerance=self.SUPPORT_PENETRATION_TOLERANCE,
            )
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            target_fixture=self.drawer_fixture.stable_id,
            target_joint=self.drawer_joint.name,
            target_region=self.drawer_region.stable_id,
            initial_drawer_open_fraction=self.INITIAL_DRAWER_FRACTION,
            drawer_success_open_fraction=self.SUCCESS_OPEN_FRACTION,
            success_definition=(
                "in order: open the exact sink-side cabinet drawer to at least "
                "80% while the candle remains released on its low stool; lift "
                "the candle at least 0.30 m above its initial height without "
                "requiring current or prior grasping; "
                "place the whole candle inside the same open drawer with shallow "
                "bottom contact and no illegal penetration for the final 1.0 s, "
                "without requiring grasping, release or absence of hand contact"
            ),
        )
        return meta


TASK_CLASSES = (
    IThorBathroomOpenThenCloseUpperCabinetDoorAndApproachSink,
    IThorBathroomOpenDrawerThenLiftCandleFromStoolAndPutInDrawer,
)

__all__ = [cls.__name__ for cls in TASK_CLASSES]
