"""Laptop-lid articulation tasks and calibrated workspace placement."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robocasa.environments.ithor_atomic.pick_place import (
    resolve_task_support_region,
)
from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_spec,
    get_ithor_bedroom_task_specs,
    get_supported_scene_ids,
)

from .articulation_base import (
    BedroomArticulationMixin,
    BedroomPartiallyOpenStartMixin,
)
from .assets import asset_xml_path


_LAPTOP_SPEC_FIELDS = {
    "existing": ("fixture", "joint"),
    "spawn_asset": (
        "asset",
        "asset_id",
        "spawn_name",
        "spawn_root_pos",
        "spawn_quat",
        "spawn_scale",
        "surface_fixture",
        "surface_region",
        "joint",
    ),
}


def load_bedroom_laptop_scene_specs(task_key, required=()):
    """Load and validate explicit existing/spawned laptop calibrations."""

    specs = get_ithor_bedroom_task_specs(
        task_key,
        required=("fixture_mode", "target", "lang", *tuple(required)),
    )
    for scene_name, spec in specs.items():
        mode = spec["fixture_mode"]
        if mode not in _LAPTOP_SPEC_FIELDS:
            raise ValueError(
                f"{scene_name}/{task_key}: unsupported fixture_mode {mode!r}"
            )
        missing = [key for key in _LAPTOP_SPEC_FIELDS[mode] if key not in spec]
        if missing:
            raise ValueError(
                f"{scene_name}/{task_key}: {mode} laptop is missing {missing}"
            )
        if mode == "spawn_asset":
            if float(spec["spawn_scale"]) != 1.0:
                raise ValueError(
                    f"{scene_name}/{task_key}: spawned laptop scale must be 1"
                )
            if np.asarray(spec["spawn_root_pos"], dtype=float).shape != (3,):
                raise ValueError(
                    f"{scene_name}/{task_key}: spawn_root_pos must be XYZ"
                )
            if np.asarray(spec["spawn_quat"], dtype=float).shape != (4,):
                raise ValueError(
                    f"{scene_name}/{task_key}: spawn_quat must be WXYZ"
                )
        behaviors = spec.get("behaviors")
        if behaviors is not None and (
            not isinstance(behaviors, list)
            or not behaviors
            or any(value not in ("open", "close") for value in behaviors)
            or len(set(behaviors)) != len(behaviors)
        ):
            raise ValueError(
                f"{scene_name}/{task_key}: invalid laptop behaviors"
            )
    return specs


_LAPTOP_SCENE_SPECS = load_bedroom_laptop_scene_specs(
    "laptop", required=("behaviors",)
)
_OPEN_LAPTOP_SCENE_SPECS = {
    scene_name: spec
    for scene_name, spec in _LAPTOP_SCENE_SPECS.items()
    if "open" in spec["behaviors"]
}


class BedroomLaptopMixin(
    BedroomPartiallyOpenStartMixin,
    BedroomArticulationMixin,
):
    """Stage calibrated laptops side-on at a reachable, hand-safe edge."""

    # Preserve the calibrated stance across resets. Even a small spawn jitter
    # can move a neutral gripper back over the desk edge in tight scenes.
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    LAPTOP_EDGE_CLEARANCE = 0.006
    LAPTOP_SUPPORT_PROBE_ABOVE = 0.08
    LAPTOP_SUPPORT_PROBE_BELOW = 0.15
    LAPTOP_SUPPORT_PROBE_STEP = 0.002
    # Start as close as the measured hand clearance allows. The explicit
    # hand-edge check below rejects close candidates in scenes where a neutral
    # hand would overhang the table, while keeping deep desks reachable.
    LAPTOP_STAND_EDGE_DISTANCES = (0.28, 0.35, 0.42, 0.50, 0.60, 0.72)
    LAPTOP_STAND_LATERAL_OFFSETS = (0.0, 0.12, -0.12, 0.24, -0.24)
    MIN_HAND_EDGE_CLEARANCE = 0.16
    HAND_GEOMETRY_ROOTS = (
        "robot0_left_wrist_yaw_link",
        "robot0_right_wrist_yaw_link",
    )
    LAPTOP_BASE_WELD_NAME = "task_open_laptop_fixed_base"

    def _fixture_mode(self):
        return self.scene_task_spec()["fixture_mode"]

    def _articulation_fixture_query(self):
        if self._fixture_mode() == "spawn_asset":
            return None
        return super()._articulation_fixture_query()

    def _semantic_candidates(self):
        if self._fixture_mode() == "spawn_asset":
            return []
        return super()._semantic_candidates()

    def _articulation_joint_name(self):
        return self.scene_task_spec()["joint"]

    def _workspace_spec(self):
        workspace = get_ithor_bedroom_task_spec(
            self.scene_name, "workspace", required=("fixture", "region")
        )
        task_spec = self.scene_task_spec()
        surface_fixture = task_spec.get("surface_fixture")
        surface_region = task_spec.get("surface_region")
        if (surface_fixture is None) != (surface_region is None):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: exact laptop surface "
                "needs both fixture and region"
            )
        if surface_fixture is not None:
            workspace["fixture"] = surface_fixture
            workspace["region"] = surface_region
        return workspace

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        workspace = self._workspace_spec()
        surface_id = workspace["fixture"]
        self.laptop_work_surface_fixture = self.get_fixture(id=surface_id)
        exact_region_id = workspace["region"]
        raw_region = (
            next(
                (
                    region
                    for region in self.laptop_work_surface_fixture.regions.values()
                    if region.stable_id == exact_region_id
                ),
                None,
            )
            if self.laptop_work_surface_fixture is not None
            else None
        )
        region = (
            resolve_task_support_region(raw_region)
            if raw_region is not None
            else None
        )
        self.laptop_work_surface_region = (
            region
            if region is not None
            and 0.35 <= float(region.world_pose()[0][2]) <= 1.20
            else None
        )
        target_id = (
            self.target_fixture.stable_id
            if self.target_fixture is not None
            else None
        )
        seating_ids = tuple(
            stable_id
            for stable_id in workspace.get("laptop_seating", ())
            if stable_id != target_id
        )
        seating = [self.get_fixture(id=stable_id) for stable_id in seating_ids]
        missing = [
            stable_id
            for stable_id, fixture in zip(seating_ids, seating)
            if fixture is None
        ]
        if missing:
            raise ValueError(
                f"{self.scene_name}: unknown exact laptop seating {missing}"
            )
        self.removed_laptop_seating_stable_ids = sorted(
            fixture.stable_id for fixture in seating
        )
        if seating:
            self.remove_semantic_fixtures(seating)
        obstruction_ids = tuple(
            dict.fromkeys(
                (
                    *workspace.get("laptop_obstructions", ()),
                    *workspace.get("remove_fixtures", ()),
                )
            )
        )
        obstruction_ids = tuple(
            stable_id for stable_id in obstruction_ids if stable_id != target_id
        )
        obstructions = [
            self.get_fixture(id=stable_id) for stable_id in obstruction_ids
        ]
        missing = [
            stable_id
            for stable_id, fixture in zip(obstruction_ids, obstructions)
            if fixture is None
        ]
        if missing:
            raise ValueError(
                f"{self.scene_name}: unknown exact laptop obstructions {missing}"
            )
        self.removed_laptop_obstruction_stable_ids = sorted(
            fixture.stable_id for fixture in obstructions
        )
        if obstructions:
            self.remove_semantic_fixtures(obstructions)
        if (
            self._fixture_mode() == "spawn_asset"
            and self.laptop_work_surface_region is not None
        ):
            self._create_spec_laptop_fixture()
        if self.laptop_work_surface_region is not None:
            self.fixture_refs["laptop_surface"] = self.laptop_work_surface_fixture
            self.region_refs["laptop_surface"] = self.laptop_work_surface_region
        if self.BEHAVIOR == "open":
            ET.SubElement(
                self.mujoco_arena.equality,
                "weld",
                name=self.LAPTOP_BASE_WELD_NAME,
                body1=self.target_fixture.body_name,
                active="false",
                solref="0.002 1",
                solimp="0.9999 0.9999 0.001 0.5 2",
            )

    def _create_spec_laptop_fixture(self):
        spec = self.scene_task_spec()
        quat = np.asarray(spec["spawn_quat"], dtype=float)
        quat /= max(float(np.linalg.norm(quat)), 1e-8)
        self.target_fixture = self.create_scene_fixture(
            xml_path=asset_xml_path(spec["asset"]),
            pos=np.asarray(spec["spawn_root_pos"], dtype=float),
            quat=quat,
            name=spec["spawn_name"],
            category="Laptop",
            source_category="Laptop",
            asset_id=spec["asset_id"],
            front=(0.0, -1.0),
        )
        self.fixture_refs["articulation_target"] = self.target_fixture

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.laptop_work_surface_fixture is not None
            and self.laptop_work_surface_region is not None
        )

    def _setup_scene(self):
        if self.BEHAVIOR == "open":
            self._set_laptop_base_weld_active(False)
        super()._setup_scene()
        self.register_task_body_contact_contract(
            "laptop_work_surface",
            tuple(self.target_fixture.body_names),
            tuple(self.laptop_work_surface_fixture.body_names),
        )

    def _laptop_base_weld_id(self):
        model = self.sim.model
        for equality_id in range(model.neq):
            name = mujoco.mj_id2name(
                model._model,
                mujoco.mjtObj.mjOBJ_EQUALITY,
                equality_id,
            )
            if name == self.LAPTOP_BASE_WELD_NAME:
                return equality_id
        raise RuntimeError(
            f"{self.scene_name}: missing laptop base weld constraint"
        )

    def _set_laptop_base_weld_active(self, active):
        equality_id = self._laptop_base_weld_id()
        self.sim.data.eq_active[equality_id] = int(bool(active))
        self.sim.forward()

    def _lock_laptop_base_at_current_pose(self):
        """Weld the staged laptop root to world while leaving its lid free."""

        equality_id = self._laptop_base_weld_id()
        body_id = self.sim.model.body_name2id(self.target_fixture.body_name)
        position = np.asarray(self.sim.data.body_xpos[body_id], dtype=float).copy()
        rotation = np.asarray(
            self.sim.data.body_xmat[body_id], dtype=float
        ).reshape(3, 3)
        quaternion = np.asarray(
            self.sim.data.body_xquat[body_id], dtype=float
        ).copy()
        # A body-to-world weld stores the world frame in body-local
        # coordinates: inverse translation followed by inverse quaternion.
        self.sim.model.eq_data[equality_id, 3:6] = -rotation.T @ position
        self.sim.model.eq_data[equality_id, 6:10] = np.array(
            [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
            dtype=float,
        )
        root_joint = self._laptop_free_joint()
        joint_id = self.sim.model.joint_name2id(root_joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[dof_address : dof_address + 6] = 0.0
        self.sim.data.eq_active[equality_id] = 1
        self.sim.forward()
        self._fixed_laptop_base_position = position

    def _configure_articulation_joint(self, joint):
        super()._configure_articulation_joint(joint)
        # The exported laptop assets use q=0 for a vertical, physically open
        # lid and q=pi/2 for a horizontal, physically closed lid. The generic
        # semantic loader assumes the reverse for every positive hinge range.
        lo, hi = (float(value) for value in joint.range)
        joint.closed_qpos = hi
        joint.open_qpos = lo

    def _laptop_free_joint(self):
        body_id = self.sim.model.body_name2id(self.target_fixture.body_name)
        joint_id = int(self.sim.model.body_jntadr[body_id])
        if joint_id < 0 or int(self.sim.model.jnt_type[joint_id]) != 0:
            raise RuntimeError(f"{self.scene_name}: laptop has no free joint")
        return self.sim.model.joint_names[joint_id]

    def _laptop_support_contact_is_valid(self):
        laptop_bodies = self._fixture_body_ids(self.target_fixture)
        support_bodies = self._fixture_body_ids(
            self.laptop_work_surface_fixture, include_siblings=True
        )
        supported = False
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body_1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body_2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if not (
                (body_1 in laptop_bodies and body_2 in support_bodies)
                or (body_2 in laptop_bodies and body_1 in support_bodies)
            ):
                continue
            if float(contact.dist) < -0.002:
                return False
            supported = True
        illegal = self.illegal_contact_pairs_for_bodies(
            laptop_bodies,
            allowed_body_roots=support_bodies,
            tolerance=0.002,
        )
        return bool(supported and not illegal)

    def _probe_laptop_support_contact(self, root_joint, qpos, surface_z):
        points, _ = self._body_geom_points(self.target_fixture.body_names)
        if not len(points):
            return None
        contact_qpos = qpos.copy()
        contact_qpos[2] += float(surface_z) - float(np.min(points[:, 2]))
        offsets = np.arange(
            self.LAPTOP_SUPPORT_PROBE_ABOVE,
            -self.LAPTOP_SUPPORT_PROBE_BELOW
            - 0.5 * self.LAPTOP_SUPPORT_PROBE_STEP,
            -self.LAPTOP_SUPPORT_PROBE_STEP,
        )
        for offset in offsets:
            candidate = contact_qpos.copy()
            candidate[2] += offset
            self.sim.data.set_joint_qpos(root_joint, candidate)
            self.sim.forward()
            if self._laptop_support_contact_is_valid():
                joint_id = self.sim.model.joint_name2id(root_joint)
                dof_address = int(self.sim.model.jnt_dofadr[joint_id])
                self.sim.data.qvel[dof_address : dof_address + 6] = 0.0
                return candidate
        return None

    @staticmethod
    def _prepend_yaw(quat, yaw):
        c, s = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        w, x, y, z = np.asarray(quat, dtype=float)
        result = np.array(
            [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w],
            dtype=float,
        )
        return result / max(float(np.linalg.norm(result)), 1e-8)

    def _laptop_approach_directions(self):
        center, yaw = self.laptop_work_surface_region.world_pose(self.sim)
        axes = np.array(
            [[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]],
            dtype=float,
        )
        # Prefer the surface's long axis, but retain the short-axis sides for
        # desks whose long ends face a wall. The laptop fit check below rejects
        # any side that cannot support its complete footprint.
        long_axis = int(np.argmax(self.laptop_work_surface_region.half_size[:2]))
        short_axis = 1 - long_axis
        long_edge_hint = self._workspace_spec().get(
            "laptop_long_edge_approach_hint"
        )
        if long_edge_hint is not None:
            # A rectangle's long edge is parallel to its long axis, so its
            # approach normal is the short axis. Select the hinted side and
            # deliberately omit both short-edge fallback directions.
            direction = axes[short_axis].copy()
            if float(np.dot(direction, long_edge_hint)) < 0.0:
                direction *= -1.0
            return (direction,)
        preferred = np.asarray(self.room_center[:2], dtype=float) - center[:2]
        return sorted(
            (axes[long_axis], -axes[long_axis], axes[short_axis], -axes[short_axis]),
            key=lambda candidate: -float(np.dot(candidate, preferred)),
        )

    def _stage_laptop_at_edge(self, root_joint, original_qpos, direction):
        direction = np.asarray(direction, dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        lateral = np.array([-direction[1], direction[0]], dtype=float)

        qpos = original_qpos.copy()
        self.sim.data.set_joint_qpos(root_joint, qpos)
        self.sim.forward()
        joint_id = self.sim.model.joint_name2id(self.target_joint)
        lid_body_id = int(self.sim.model.jnt_bodyid[joint_id])
        hinge_axis = np.asarray(
            self.sim.data.body_xmat[lid_body_id], dtype=float
        ).reshape(3, 3)[:2, 0]
        hinge_axis /= max(float(np.linalg.norm(hinge_axis)), 1e-8)
        yaw_delta = float(
            np.arctan2(
                hinge_axis[0] * direction[1] - hinge_axis[1] * direction[0],
                np.dot(hinge_axis, direction),
            )
        )
        qpos[3:7] = self._prepend_yaw(qpos[3:7], yaw_delta)
        self.sim.data.set_joint_qpos(root_joint, qpos)
        self.sim.forward()

        points, _ = self._body_geom_points(self.target_fixture.body_names)
        if not len(points):
            return False
        root_xy = qpos[:2].copy()
        relative = points[:, :2] - root_xy
        front_extent = float(np.max(relative @ direction))
        lateral_projection = relative @ lateral
        lateral_center = float(
            0.5 * (np.min(lateral_projection) + np.max(lateral_projection))
        )
        surface_center, _ = self.laptop_work_surface_region.world_pose(self.sim)
        surface_extent = self.support_region_radial_extent(
            self.laptop_work_surface_region,
            direction,
            sim=self.sim,
        )
        qpos[:2] = (
            surface_center[:2]
            + direction
            * (surface_extent - self.LAPTOP_EDGE_CLEARANCE - front_extent)
            - lateral * lateral_center
        )
        self.sim.data.set_joint_qpos(root_joint, qpos)
        self.sim.forward()

        qpos = self._probe_laptop_support_contact(
            root_joint, qpos, surface_center[2]
        )
        if qpos is None:
            return False

        # Reject directions for which the rotated laptop does not fit on the
        # complete calibrated work surface.
        points, _ = self._body_geom_points(self.target_fixture.body_names)
        _, surface_yaw = self.laptop_work_surface_region.world_pose(self.sim)
        c, s = np.cos(-surface_yaw), np.sin(-surface_yaw)
        delta = points[:, :2] - surface_center[:2]
        local = np.column_stack(
            (c * delta[:, 0] - s * delta[:, 1], s * delta[:, 0] + c * delta[:, 1])
        )
        return bool(
            np.all(
                np.abs(local)
                <= np.asarray(self.laptop_work_surface_region.half_size[:2])
                + 0.008
            )
        )

    def _hands_clear_of_laptop_surface(self, direction):
        center, _ = self.laptop_work_surface_region.world_pose(self.sim)
        extent = self.support_region_radial_extent(
            self.laptop_work_surface_region,
            direction,
            sim=self.sim,
        )
        return all(
            float(np.dot(eef[:2] - center[:2], direction) - extent)
            >= self.MIN_HAND_EDGE_CLEARANCE
            for eef in self.eef_positions()
        )

    def _hand_geometries_clear_of_laptop_surface(self, direction):
        required = float(
            self._workspace_spec().get("laptop_hand_geometry_clearance", 0.0)
        )
        if required <= 0.0:
            return True
        center, _ = self.laptop_work_surface_region.world_pose(self.sim)
        extent = self.support_region_radial_extent(
            self.laptop_work_surface_region,
            direction,
            sim=self.sim,
        )
        for body_name in self.HAND_GEOMETRY_ROOTS:
            points, _ = self._body_geom_points((body_name,))
            if not len(points):
                return False
            clearance = np.min(
                (points[:, :2] - center[:2]) @ direction - extent
            )
            if float(clearance) < required:
                return False
        return True

    def _finalize_articulation_robot_pose(self, distance):
        del distance
        root_joint = self._laptop_free_joint()
        original_qpos = self.sim.data.get_joint_qpos(root_joint).copy()
        surface_center, _ = self.laptop_work_surface_region.world_pose(self.sim)
        for direction in self._laptop_approach_directions():
            direction = np.asarray(direction, dtype=float)
            if not self._stage_laptop_at_edge(
                root_joint, original_qpos, direction
            ):
                continue
            self.target_fixture_xy = self.target_fixture.world_center(self.sim)[:2].copy()
            self.target_joint_pos = self._joint_interaction_pos(
                self.target_semantic_joint
            )
            self.target_face_pos = self._joint_target_pos(self.target_semantic_joint)
            target = np.asarray(self.target_joint_pos, dtype=float)
            surface_extent = self.support_region_radial_extent(
                self.laptop_work_surface_region,
                direction,
                sim=self.sim,
            )
            lateral = np.array([-direction[1], direction[0]], dtype=float)
            for edge_distance in self.LAPTOP_STAND_EDGE_DISTANCES:
                for lateral_offset in self.LAPTOP_STAND_LATERAL_OFFSETS:
                    xy = (
                        surface_center[:2]
                        + direction * (surface_extent + edge_distance)
                        + lateral * lateral_offset
                    )
                    if not self.robot_footprint_is_on_floor(xy, radius=0.25):
                        continue
                    yaw = float(
                        np.arctan2(target[1] - xy[1], target[0] - xy[0])
                    )
                    if not (
                        self._set_robot_free_pose(xy, yaw)
                        and self._calibrate_robot_free_joint_height()
                        and self.count_body_collisions() == 0
                        and self._hands_clear_of_laptop_surface(direction)
                        and self._hand_geometries_clear_of_laptop_surface(
                            direction
                        )
                    ):
                        continue
                    self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
                    self.robot_base_ori = np.array([0.0, 0.0, yaw])
                    self._laptop_operator_direction = direction.copy()
                    if not self._laptop_support_contact_is_valid():
                        continue
                    if self.BEHAVIOR == "open":
                        self._lock_laptop_base_at_current_pose()
                    return True
        self.sim.data.set_joint_qpos(root_joint, original_qpos)
        self.sim.forward()
        return False

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["removed_laptop_seating"] = list(
            self.removed_laptop_seating_stable_ids
        )
        ep_meta["removed_laptop_obstructions"] = list(
            self.removed_laptop_obstruction_stable_ids
        )
        ep_meta["laptop_surface"] = self.laptop_work_surface_region.stable_id
        ep_meta["laptop_stance"] = (
            "closest reachable pose outside the table edge with "
            f"{self.MIN_HAND_EDGE_CLEARANCE:.2f} m hand clearance and the full "
            "0.25 m foot support on floor"
        )
        ep_meta["laptop_hand_geometry_clearance"] = float(
            self._workspace_spec().get("laptop_hand_geometry_clearance", 0.0)
        )
        ep_meta["laptop_orientation"] = (
            "hinge axis aligned with the operator approach (side-on)"
        )
        ep_meta["laptop_base_fixed"] = self.BEHAVIOR == "open"
        return ep_meta


class IThorBedroomOpenLaptop(BedroomLaptopMixin, IThorBedroomEnv):
    ATOMIC_SKILL = "OpenLid"
    SPEC_TASK = "laptop"
    SUPPORTED_SCENE_IDS = tuple(
        sorted(
            int(scene_name.removeprefix("FloorPlan"))
            for scene_name in _OPEN_LAPTOP_SCENE_SPECS
        )
    )
    SCENE_SPECS = _OPEN_LAPTOP_SCENE_SPECS
    BEHAVIOR = "open"


class IThorBedroomCloseLaptop(BedroomLaptopMixin, IThorBedroomEnv):
    ATOMIC_SKILL = "CloseLid"
    SPEC_TASK = "laptop"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = _LAPTOP_SCENE_SPECS
    BEHAVIOR = "close"

__all__ = ["IThorBedroomOpenLaptop", "IThorBedroomCloseLaptop"]
