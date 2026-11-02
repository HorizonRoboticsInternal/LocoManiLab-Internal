"""Reusable physical layout helpers for Bedroom composite tasks."""

import numpy as np

from robocasa.environments.ithor_atomic.pick_place import (
    SemanticSurfaceTask,
    SupportContactPlacementMixin,
)
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from .surface_placement import BedroomSurfacePlacementMixin


class BedroomCompositeObjectMixin(SupportContactPlacementMixin):
    """Deterministic, contact-checked object layout for staged tasks."""

    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    OBJECT_EDGE_MARGIN = 0.035
    RECEPTACLE_INTERIOR_SCALE = 0.82
    RECEPTACLE_RIM_CLEARANCE = 0.005
    RECEPTACLE_BOTTOM_TOLERANCE = 0.005
    RECEPTACLE_UPRIGHT_COSINE = float(np.cos(np.deg2rad(20.0)))
    SUPPORT_CONTACT_PENETRATION_TOLERANCE = 0.003
    # Some Bedroom basket assets report decorative bbox feet 0.111 m below
    # their collidable floor. Named shallow contact still proves support; this
    # tolerance only reconciles that visual-bbox offset with the semantic plane.
    RECEPTACLE_SUPPORT_Z_TOLERANCE = 0.12

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(
            name=f"Manual{type(self).__name__}Sampler"
        )

    def _semantic_placements_are_valid(self):
        return True

    @staticmethod
    def _surface_plane_z(region, sim):
        if region.kind != "support":
            raise ValueError(
                f"{region.stable_id}: composite placement requires a support"
            )
        return float(region.world_pose(sim)[0][2])

    @staticmethod
    def _prepend_yaw(quat, yaw):
        c, s = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
        w, x, y, z = np.asarray(quat, dtype=float)
        result = np.array(
            [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]
        )
        return result / max(float(np.linalg.norm(result)), 1e-8)

    def _zero_free_joint_velocity(self, obj):
        joint_id = self.sim.model.joint_name2id(obj.joints[0])
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[dof_address : dof_address + 6] = 0.0

    def _set_object_bottom_at(self, name, xy, bottom_z):
        # Asset roots are not guaranteed to coincide with the reported object
        # body or its bbox origin. Reuse the neutral geometry-aware setter
        # instead of reconstructing a free-joint Z from ``bottom_offset``.
        self._set_task_object_bottom_at(name, xy, bottom_z)

    def _object_projected_half_extent(self, name, direction):
        half_xy = np.asarray(self._object_half_xy(name), dtype=float)
        matrix = self.sim.data.get_body_xmat(
            self.objects[name].root_body
        ).reshape(3, 3)
        direction = np.asarray(direction, dtype=float)[:2]
        return float(
            abs(np.dot(matrix[:2, 0], direction)) * half_xy[0]
            + abs(np.dot(matrix[:2, 1], direction)) * half_xy[1]
        )

    def _surface_layout_frame(self, region, operator_xy=None):
        center, _ = region.world_pose(self.sim)
        if operator_xy is None:
            operator_xy = self._robot_base_xy()
        direction = np.asarray(
            self.support_region_long_edge_approach_directions(
                region,
                toward_xy=operator_xy,
                sim=self.sim,
            )[0],
            dtype=float,
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        lateral = np.array([-direction[1], direction[0]], dtype=float)
        return center, direction, lateral

    def _near_edge_object_xy(self, region, name, lateral_offset):
        center, direction, lateral = self._surface_layout_frame(region)
        surface_depth = self.support_region_radial_extent(
            region, direction, sim=self.sim
        )
        surface_lateral = self.support_region_radial_extent(
            region, lateral, sim=self.sim
        )
        object_depth = self._object_projected_half_extent(name, direction)
        object_lateral = self._object_projected_half_extent(name, lateral)
        max_lateral = max(
            0.0,
            float(surface_lateral - object_lateral - self.OBJECT_EDGE_MARGIN),
        )
        lateral_offset = float(
            np.clip(lateral_offset, -max_lateral, max_lateral)
        )
        depth = float(surface_depth - object_depth - self.OBJECT_EDGE_MARGIN)
        return center[:2] + direction * depth + lateral * lateral_offset

    def _place_object_on_surface(
        self,
        name,
        region,
        lateral_offset=0.0,
        orient_crosswise=False,
        inward_fraction=0.0,
    ):
        xy = self._near_edge_object_xy(region, name, lateral_offset)
        self._set_object_bottom_at(
            name, xy, self._surface_plane_z(region, self.sim)
        )
        self.sim.forward()
        if orient_crosswise:
            self._orient_object_long_axis_crosswise(name, region)
            self.sim.forward()
            xy = self._near_edge_object_xy(region, name, lateral_offset)
        if not self._place_object_on_region_contact(
            name,
            region,
            xy,
            inward_fraction=inward_fraction,
        ):
            raise RuntimeError(
                f"{self.scene_name}: no contacted support pose for {name} on "
                f"{region.stable_id}"
            )

    def _place_object_on_region_contact(
        self, name, region, xy=None, inward_fraction=0.0
    ):
        fractions = tuple(
            value
            for value in (inward_fraction, 0.35, 0.50, 0.75, 1.0)
            if value >= inward_fraction
        )
        return self.place_object_on_support_contact(
            name,
            region,
            fixture=self.get_fixture(id=region.fixture_id),
            xy=xy,
            inward_fractions=fractions,
            probe_below=0.08,
            probe_step=0.001,
            penetration_tolerance=0.002,
        )

    def _orient_object_long_axis_crosswise(self, name, region):
        half_xy = np.asarray(self._object_half_xy(name), dtype=float)
        if half_xy.shape != (2,) or float(np.max(half_xy)) <= 1e-6:
            return
        _, _, lateral = self._surface_layout_frame(region)
        obj = self.objects[name]
        matrix = self.sim.data.get_body_xmat(obj.root_body).reshape(3, 3)
        current = matrix[:2, int(np.argmax(half_xy))]
        current /= max(float(np.linalg.norm(current)), 1e-8)
        delta = float(
            np.arctan2(
                current[0] * lateral[1] - current[1] * lateral[0],
                np.dot(current, lateral),
            )
        )
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[3:7] = self._prepend_yaw(qpos[3:7], delta)
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self._zero_free_joint_velocity(obj)

    def _orient_object_long_axis_to_receptacle(self, name, receptacle_name):
        """Align an elongated item with the receptacle's widest local axis."""

        half_xy = np.asarray(self._object_half_xy(name), dtype=float)
        if half_xy.shape != (2,) or float(np.max(half_xy)) <= 1e-6:
            return
        receptacle = self.objects[receptacle_name]
        receptacle_matrix = np.asarray(
            self.sim.data.get_body_xmat(receptacle.root_body), dtype=float
        ).reshape(3, 3)
        receptacle_origin = np.asarray(
            self.sim.data.get_body_xpos(receptacle.root_body), dtype=float
        )
        local_points = (
            self.object_bbox_points(receptacle_name) - receptacle_origin
        ) @ receptacle_matrix
        local_half_xy = 0.5 * np.ptp(local_points[:, :2], axis=0)
        desired = receptacle_matrix[:2, int(np.argmax(local_half_xy))]
        desired /= max(float(np.linalg.norm(desired)), 1e-8)

        obj = self.objects[name]
        object_matrix = np.asarray(
            self.sim.data.get_body_xmat(obj.root_body), dtype=float
        ).reshape(3, 3)
        current = object_matrix[:2, int(np.argmax(half_xy))]
        current /= max(float(np.linalg.norm(current)), 1e-8)
        delta = float(
            np.arctan2(
                current[0] * desired[1] - current[1] * desired[0],
                np.dot(current, desired),
            )
        )
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[3:7] = self._prepend_yaw(qpos[3:7], delta)
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self._zero_free_joint_velocity(obj)
        self.sim.forward()

    def _object_in_named_receptacle(
        self,
        name,
        receptacle_name,
        xy_tol=0.14,
        *,
        containment="bbox",
        bottom_tolerance=None,
    ):
        del xy_tol
        source = self.objects[name]
        receptacle = self.objects[receptacle_name]
        origin = np.asarray(
            self.sim.data.get_body_xpos(receptacle.root_body), dtype=float
        )
        matrix = np.asarray(
            self.sim.data.get_body_xmat(receptacle.root_body), dtype=float
        ).reshape(3, 3)
        receptacle_local = (
            self.object_bbox_points(receptacle_name) - origin
        ) @ matrix
        source_local = (self.object_bbox_points(name) - origin) @ matrix
        lower = np.min(receptacle_local, axis=0)
        upper = np.max(receptacle_local, axis=0)
        center_xy = 0.5 * (lower[:2] + upper[:2])
        scale = np.asarray(self.RECEPTACLE_INTERIOR_SCALE, dtype=float)
        if scale.ndim == 0:
            scale = np.repeat(scale, 2)
        if scale.shape != (2,) or np.any(scale <= 0.0) or np.any(scale > 1.0):
            raise ValueError(
                "RECEPTACLE_INTERIOR_SCALE must be one or two values in (0, 1]"
            )
        interior_half = np.maximum(
            0.5 * (upper[:2] - lower[:2]) * scale,
            1e-6,
        )
        if containment == "bbox":
            source_xy = source_local[:, :2]
        elif containment == "center":
            source_xy = 0.5 * (
                np.min(source_local[:, :2], axis=0)
                + np.max(source_local[:, :2], axis=0)
            )[None, :]
        else:
            raise ValueError("containment must be 'bbox' or 'center'")
        normalized = (source_xy - center_xy) / interior_half
        fully_inside_xy = bool(
            np.all(np.sum(normalized * normalized, axis=1) <= 1.0)
        )
        source_bottom = float(np.min(source_local[:, 2]))
        bottom_tolerance = (
            self.RECEPTACLE_BOTTOM_TOLERANCE
            if bottom_tolerance is None
            else float(bottom_tolerance)
        )
        if bottom_tolerance < 0.0:
            raise ValueError("bottom_tolerance must be nonnegative")
        return bool(
            self._objects_contact_shallow(name, receptacle_name)
            and fully_inside_xy
            and source_bottom
            >= float(lower[2]) - bottom_tolerance
            and source_bottom
            <= float(upper[2]) - self.RECEPTACLE_RIM_CLEARANCE
        )

    def _object_object_contact_distances(self, first_name, second_name):
        first_bodies = self._body_tree_ids(
            [self.obj_body_id[first_name]]
        )
        second_bodies = self._body_tree_ids(
            [self.obj_body_id[second_name]]
        )
        distances = []
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body_1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body_2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body_1 in first_bodies and body_2 in second_bodies) or (
                body_2 in first_bodies and body_1 in second_bodies
            ):
                distances.append(float(contact.dist))
        return distances

    def _objects_contact_shallow(
        self, first_name, second_name, penetration_tolerance=0.003
    ):
        distances = self._object_object_contact_distances(
            first_name, second_name
        )
        return bool(
            distances
            and min(distances) >= -float(penetration_tolerance)
        )

    def _composite_object_released(self, name):
        return self.object_released(name, th=0.18)

    def _receptacle_upright_released_on_region(self, name, region):
        matrix = np.asarray(
            self.sim.data.get_body_xmat(self.objects[name].root_body),
            dtype=float,
        ).reshape(3, 3)
        fixture = self.get_fixture(id=region.fixture_id)
        tolerance = self.SUPPORT_CONTACT_PENETRATION_TOLERANCE
        return bool(
            float(matrix[2, 2]) >= self.RECEPTACLE_UPRIGHT_COSINE
            and self.object_on_region(
                name,
                region,
                z_tol=self.RECEPTACLE_SUPPORT_Z_TOLERANCE,
                require_contact=True,
            )
            and self._object_contacts_fixture(
                name, fixture, penetration_tolerance=tolerance
            )
            and not self._object_has_illegal_penetration(
                name, region, tolerance=tolerance
            )
            and self._composite_object_released(name)
        )

    def _place_object_in_receptacle_contact(
        self,
        name,
        receptacle_name,
        lateral_offset=0.0,
        xy_tol=0.14,
        *,
        containment="bbox",
        bottom_tolerance=None,
    ):
        self._orient_object_long_axis_to_receptacle(name, receptacle_name)
        receptacle = self.objects[receptacle_name]
        position = self.sim.data.get_body_xpos(receptacle.root_body).copy()
        matrix = self.sim.data.get_body_xmat(receptacle.root_body).reshape(3, 3)
        lateral = matrix[:2, 0]
        lateral /= max(float(np.linalg.norm(lateral)), 1e-8)
        requested = float(lateral_offset)
        receptacle_bottom = float(
            np.min(self.object_bbox_points(receptacle_name)[:, 2])
        )
        # Search down from clear interior space to the first shallow physical
        # basket contact. The external bbox lower face is only an anchor: the
        # collidable interior floor may sit above it, and placing an item at
        # the outer bottom can make it penetrate and falsely contact the table.
        bottom_offsets = np.arange(0.08, -0.006, -0.001)
        for offset in (
            requested,
            requested - 0.01,
            requested + 0.01,
            requested - 0.02,
            requested + 0.02,
            0.0,
        ):
            for floor_offset in bottom_offsets:
                self._set_object_bottom_at(
                    name,
                    position[:2] + lateral * offset,
                    receptacle_bottom + float(floor_offset),
                )
                if not self._object_in_named_receptacle(
                    name,
                    receptacle_name,
                    xy_tol=xy_tol,
                    containment=containment,
                    bottom_tolerance=bottom_tolerance,
                ):
                    continue
                work_surface = getattr(self, "work_surface_fixture", None)
                if work_surface is not None and self._object_fixture_contact_distances(
                    name, work_surface
                ):
                    continue
                return True
        return False

    def _object_fixture_contact_distances(self, name, fixture):
        object_bodies = self._body_tree_ids([self.obj_body_id[name]])
        fixture_bodies = self._fixture_body_ids(
            fixture, include_siblings=True
        )
        distances = []
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body_1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body_2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body_1 in object_bodies and body_2 in fixture_bodies) or (
                body_2 in object_bodies and body_1 in fixture_bodies
            ):
                distances.append(float(contact.dist))
        return distances

    def _object_contacts_fixture(
        self, name, fixture, penetration_tolerance=0.003
    ):
        distances = self._object_fixture_contact_distances(name, fixture)
        return bool(
            distances
            and min(distances) >= -float(penetration_tolerance)
        )

    def _place_object_in_named_region_contact(
        self,
        name,
        region,
        fixture,
        *,
        xy=None,
        probe_above=0.08,
        probe_below=0.04,
        probe_step=0.002,
        penetration_tolerance=0.003,
    ):
        """Lower an object onto a deep receptacle's physical bottom."""

        if region.kind != "receptacle":
            raise ValueError("interior contact placement needs a receptacle")
        center, _ = region.world_pose(self.sim)
        target_xy = center[:2] if xy is None else np.asarray(xy)[:2]
        lower_face_z = float(center[2] - region.half_size[2])
        bottom_values = np.arange(
            lower_face_z + float(probe_above),
            lower_face_z - float(probe_below) - 0.5 * float(probe_step),
            -float(probe_step),
        )
        for bottom_z in bottom_values:
            self._set_object_bottom_at(name, target_xy, bottom_z)
            self.sim.forward()
            if not self.object_in_region(
                name, region, rim_tol=0.005, xy_tol=0.0
            ):
                continue
            if not self._object_contacts_fixture(
                name, fixture, penetration_tolerance
            ):
                continue
            if self._object_has_illegal_penetration(
                name, region, tolerance=penetration_tolerance
            ):
                continue
            self._zero_free_joint_velocity(self.objects[name])
            return True
        return False


class BedroomCompositeSurfaceTaskMixin(
    BedroomCompositeObjectMixin,
    BedroomSurfacePlacementMixin,
    SemanticSurfaceTask,
):
    """Shared near-surface behavior; concrete tasks add the room base."""

    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    MIN_SURFACE_HALF_XY = (0.16, 0.15)
    STAND_LATERAL_OFFSETS = (0.0, 0.12, -0.12, 0.24, -0.24)
    MIN_EEF_EDGE_CLEARANCE = 0.12
    MIN_HAND_GEOMETRY_CLEARANCE = 0.04
    HAND_GEOMETRY_ROOTS = (
        "robot0_left_wrist_yaw_link",
        "robot0_right_wrist_yaw_link",
    )

    def _surface_spawn_is_hand_safe(self, direction):
        center, _ = self.work_surface_region.world_pose(self.sim)
        extent = self.support_region_radial_extent(
            self.work_surface_region, direction, sim=self.sim
        )
        if any(
            float(np.dot(eef[:2] - center[:2], direction) - extent)
            < self.MIN_EEF_EDGE_CLEARANCE
            for eef in self.eef_positions()
        ):
            return False
        for body_name in self.HAND_GEOMETRY_ROOTS:
            points, _ = self._body_geom_points((body_name,))
            if not len(points):
                return False
            clearance = np.min(
                (points[:, :2] - center[:2]) @ direction - extent
            )
            if float(clearance) < self.MIN_HAND_GEOMETRY_CLEARANCE:
                return False
        return True

    def _hands_clear_of_work_surface(self, direction):
        return self._surface_spawn_is_hand_safe(direction)

    def _park_task_objects_for_layout(self):
        """Keep not-yet-staged objects out of each bounded contact probe."""

        center, _ = self.work_surface_region.world_pose(self.sim)
        for index, name in enumerate(self.objects):
            self._set_task_object_bottom_at(
                name,
                center[:2],
                2.5 + 0.20 * index,
            )

    def _reset_internal(self):
        super()._reset_internal()
        self._park_task_objects_for_layout()
        self._place_task_objects()
        self.sim.forward()
        self._reset_composite_progress()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            lang=self.scene_task_spec()["lang"],
            composite_stages=list(self.COMPOSITE_STAGES),
            source_surface=self.work_surface_region.stable_id,
            removed_workspace_fixtures=list(
                self.removed_workspace_fixture_ids
            ),
        )
        ordering = getattr(self, "COMPOSITE_ORDERING", None)
        if ordering is not None:
            ep_meta["composite_ordering"] = ordering
        return ep_meta


__all__ = [
    "BedroomCompositeObjectMixin",
    "BedroomCompositeSurfaceTaskMixin",
]
