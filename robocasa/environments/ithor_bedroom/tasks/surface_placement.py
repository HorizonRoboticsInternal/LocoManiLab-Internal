"""Shared calibrated surface placement for bedroom object tasks."""

import numpy as np

from robocasa.environments.ithor_atomic.pick_place import (
    resolve_task_support_region,
)
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_spec,
    get_ithor_bedroom_task_specs,
)


_WORKSPACE_SPECS = get_ithor_bedroom_task_specs(
    "grasp_workspace",
    required=(
        "fixture",
        "region",
        "approach_direction",
        "operation_normal_distance",
    ),
)


def _square_operation_pose(
    source_xy,
    receptacle_xy,
    outward_direction,
    normal_distance,
):
    """Return a table-square base pose centered on a source/receptacle pair."""

    source_xy = np.asarray(source_xy, dtype=float)
    receptacle_xy = np.asarray(receptacle_xy, dtype=float)
    outward = np.asarray(outward_direction, dtype=float)
    if source_xy.shape != (2,) or receptacle_xy.shape != (2,):
        raise ValueError("operation object positions must be two-dimensional")
    if outward.shape != (2,) or not np.all(np.isfinite(outward)):
        raise ValueError("approach_direction must contain two finite values")
    norm = float(np.linalg.norm(outward))
    if norm <= 1e-8:
        raise ValueError("approach_direction must be nonzero")
    distance = float(normal_distance)
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("operation_normal_distance must be positive")

    outward /= norm
    midpoint = 0.5 * (source_xy + receptacle_xy)
    base_xy = midpoint + outward * distance
    inward = -outward
    yaw = float(np.arctan2(inward[1], inward[0]))
    return base_xy, yaw


def align_object_long_axis_parallel_to_surface_edge(
    task, name, region, minimum_aspect_ratio=1.15
):
    """Rotate an elongated free object parallel to a surface's long edge."""

    if name not in getattr(task, "objects", {}):
        return False
    half_xy = np.asarray(task._object_half_xy(name), dtype=float)
    if (
        half_xy.shape != (2,)
        or float(np.min(half_xy)) <= 1e-6
        or float(np.max(half_xy) / np.min(half_xy))
        < float(minimum_aspect_ratio)
    ):
        return False
    direction = np.asarray(
        task.support_region_long_edge_approach_directions(
            region, sim=task.sim
        )[0],
        dtype=float,
    )
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    edge_axis = np.array([-direction[1], direction[0]], dtype=float)

    obj = task.objects[name]
    matrix = task.sim.data.get_body_xmat(obj.root_body).reshape(3, 3)
    current = matrix[:2, int(np.argmax(half_xy))]
    current /= max(float(np.linalg.norm(current)), 1e-8)
    delta = float(
        np.arctan2(
            current[0] * edge_axis[1] - current[1] * edge_axis[0],
            np.dot(current, edge_axis),
        )
    )
    qpos = task.sim.data.get_joint_qpos(obj.joints[0]).copy()
    c, s = np.cos(delta / 2.0), np.sin(delta / 2.0)
    w, x, y, z = qpos[3:7]
    qpos[3:7] = np.array(
        [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w],
        dtype=float,
    )
    qpos[3:7] /= max(float(np.linalg.norm(qpos[3:7])), 1e-8)
    task.sim.data.set_joint_qpos(obj.joints[0], qpos)
    joint_id = task.sim.model.joint_name2id(obj.joints[0])
    dof_address = int(task.sim.model.jnt_dofadr[joint_id])
    task.sim.data.qvel[dof_address : dof_address + 6] = 0.0
    return True


class BedroomSurfacePlacementMixin:
    """Shared physical containment and calibrated surface layout."""

    ENFORCE_EDGE_PARALLEL_LAYOUT = True
    WORKSPACE_SPEC_TASK = "grasp_workspace"
    WORK_SURFACE_FIXTURES = {
        scene_name: spec["fixture"]
        for scene_name, spec in _WORKSPACE_SPECS.items()
    }
    WORK_SURFACE_REGIONS = {
        scene_name: spec["region"]
        for scene_name, spec in _WORKSPACE_SPECS.items()
    }
    STAND_DISTANCES = (0.48, 0.58, 0.68, 0.80)
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    MIN_HAND_EDGE_CLEARANCE = 0.12
    MIN_SURFACE_HALF_XY = (0.16, 0.16)
    SOURCE_SCALE, RECEPTACLE_SCALE = 0.65, 0.75
    SOURCE_ON_OPERATOR_RIGHT = False
    MIN_SOURCE_RIGHT_SEPARATION = 0.05
    # Start both objects near the operator-facing edge. The final placement
    # pass below puts their centers on one edge-parallel row exactly.
    RECEPTACLE_EDGE_FRACTION = SOURCE_EDGE_FRACTION = 0.88
    RECEPTACLE_LATERAL_OFFSET, SOURCE_LATERAL_OFFSET = -0.12, 0.12
    WORKSPACE_EDGE_CLEARANCE = 0.025
    NEAR_EDGE_ROW_TOLERANCE = 0.01
    OPERATION_DISTANCE_OFFSETS = (0.0, 0.01, 0.02, 0.03, 0.04, 0.05)
    SOURCE_MJCF_PATH = RECEPTACLE_MJCF_PATH = None

    def _workspace_spec(self):
        return get_ithor_bedroom_task_spec(
            self.scene_name,
            self.WORKSPACE_SPEC_TASK,
            required=(
                "fixture",
                "region",
                "approach_direction",
                "operation_normal_distance",
            ),
        )

    def _workspace_outward_direction(self):
        preferred = np.asarray(
            self._workspace_spec()["approach_direction"], dtype=float
        )
        if preferred.shape != (2,) or not np.all(np.isfinite(preferred)):
            raise ValueError("approach_direction must contain two finite values")
        norm = float(np.linalg.norm(preferred))
        if norm <= 1e-8:
            raise ValueError("approach_direction must be nonzero")
        preferred /= norm
        direction = max(
            self.support_region_long_edge_approach_directions(
                self.work_surface_region,
                sim=self.sim if getattr(self, "sim", None) is not None else None,
            ),
            key=lambda candidate: float(np.dot(candidate, preferred)),
        )
        direction = np.asarray(direction, dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        return direction

    def _operation_pose(self, normal_distance):
        source_xy = np.asarray(
            self.sim.data.get_body_xpos(
                self.objects[self.SOURCE_NAME].root_body
            )[:2],
            dtype=float,
        )
        receptacle_xy = np.asarray(
            self.sim.data.get_body_xpos(
                self.objects[self.RECEPTACLE_NAME].root_body
            )[:2],
            dtype=float,
        )
        return _square_operation_pose(
            source_xy,
            receptacle_xy,
            self._workspace_outward_direction(),
            normal_distance,
        )

    def _hands_clear_of_work_surface(self, direction):
        direction = np.asarray(direction, dtype=float)
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        center, _ = self.work_surface_region.world_pose(self.sim)
        extent = self.support_region_radial_extent(
            self.work_surface_region,
            direction,
            sim=self.sim,
        )
        return all(
            float(np.dot(eef[:2] - center[:2], direction) - extent)
            >= self.MIN_HAND_EDGE_CLEARANCE
            for eef in self.eef_positions()
        )

    def _place_robot_at_surface(self):
        center, _ = self.work_surface_region.world_pose(self.sim)
        for direction in self._approach_directions():
            direction = np.asarray(direction, dtype=float)
            direction /= max(float(np.linalg.norm(direction)), 1e-8)
            extent = self.support_region_radial_extent(
                self.work_surface_region,
                direction,
                sim=self.sim,
            )
            lateral = np.array([-direction[1], direction[0]], dtype=float)
            for distance in self.STAND_DISTANCES:
                for offset in self.STAND_LATERAL_OFFSETS:
                    xy = center[:2] + direction * (extent + distance)
                    xy = xy + lateral * offset
                    yaw = float(
                        np.arctan2(center[1] - xy[1], center[0] - xy[0])
                    )
                    if not (
                        self.robot_footprint_is_on_floor(xy, radius=0.25)
                        and self._set_robot_free_pose(xy, yaw)
                        and self._calibrate_robot_free_joint_height()
                        and self.count_body_collisions() == 0
                        and self._hands_clear_of_work_surface(direction)
                    ):
                        continue
                    self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
                    self.robot_base_ori = np.array([0.0, 0.0, yaw])
                    return True
        return False

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        if self.work_surface_region is None:
            return
        self.work_surface_region = resolve_task_support_region(
            self.work_surface_region
        )
        self.region_refs["task_surface"] = self.work_surface_region
        workspace = self._workspace_spec()
        remove_ids = tuple(
            dict.fromkeys(
                (*workspace.get("seating", ()), *workspace.get("remove_fixtures", ()))
            )
        )
        seating = [self.get_fixture(id=stable_id) for stable_id in remove_ids]
        missing = [
            stable_id
            for stable_id, fixture in zip(remove_ids, seating)
            if fixture is None
        ]
        if missing:
            raise ValueError(
                f"{self.scene_name}: unknown exact workspace removals {missing}"
            )
        self.removed_workspace_fixture_ids = sorted(
            fixture.stable_id for fixture in seating
        )
        if seating:
            self.remove_semantic_fixtures(seating)

    def _operator_reference_xy(self):
        if getattr(self, "TASK_FAMILY", None) != "composite":
            return self._robot_base_xy()
        center, _ = self.work_surface_region.world_pose(
            self.sim if getattr(self, "sim", None) is not None else None
        )
        sim = self.sim if getattr(self, "sim", None) is not None else None
        preferred = self._workspace_spec().get("approach_direction")
        if preferred is None:
            toward_xy = (
                self._robot_base_xy()
                if sim is not None
                else np.asarray(self.robot_base_pos[:2], dtype=float)
            )
            direction = self.support_region_long_edge_approach_directions(
                self.work_surface_region,
                toward_xy=toward_xy,
                sim=sim,
            )[0]
        else:
            preferred = np.asarray(preferred, dtype=float)
            direction = max(
                self.support_region_long_edge_approach_directions(
                    self.work_surface_region,
                    sim=sim,
                ),
                key=lambda candidate: float(np.dot(candidate, preferred)),
            )
        extent = self.support_region_radial_extent(
            self.work_surface_region,
            direction,
            sim=sim,
        )
        return center[:2] + np.asarray(direction, dtype=float) * (
            extent + float(self.STAND_DISTANCES[0])
        )

    def _source_crosswise_rotation(self):
        source_name = getattr(self, "SOURCE_NAME", None)
        if source_name is None:
            return None
        half_xy = np.asarray(self._object_half_xy(source_name), dtype=float)
        if (
            half_xy.shape != (2,)
            or float(np.min(half_xy)) <= 1e-6
            or float(np.max(half_xy) / np.min(half_xy)) < 1.15
        ):
            return None
        cfg = next(
            item for item in self.object_cfgs
            if item.get("name") == source_name
        )
        placement = cfg["placement"]
        operator_xy = self._operator_reference_xy()
        target_xy, _ = self.support_region_edge_pose(
            self.work_surface_region,
            half_xy,
            operator_xy,
            margin=float(placement.get("margin", 0.04)),
            edge_fraction=placement.get("edge_fraction", 1.0),
            lateral_offset=placement.get("edge_lateral_offset"),
            sim=self.sim,
        )
        view = np.asarray(target_xy, dtype=float)[:2] - operator_xy
        if float(np.linalg.norm(view)) <= 1e-6:
            return None
        view_yaw = float(np.arctan2(view[1], view[0]))
        local_long_yaw = 0.0 if int(np.argmax(half_xy)) == 0 else np.pi / 2.0
        rotation = view_yaw + np.pi / 2.0 - local_long_yaw
        base_margin = float(placement.setdefault("crosswise_base_margin", 0.04))
        placement["margin"] = base_margin + float(np.max(half_xy) - np.min(half_xy))
        return float(np.arctan2(np.sin(rotation), np.cos(rotation)))

    def _build_semantic_placement_initializer(self):
        if (
            getattr(self, "TASK_FAMILY", None) != "composite"
            or getattr(self, "work_surface_region", None) is None
        ):
            return super()._build_semantic_placement_initializer()
        # Composite tasks can keep a task-specific robot pose that differs
        # from their manipulation point. Build edge sampling against the
        # verified operation point, then restore the task pose.
        saved_pos = np.asarray(self.robot_base_pos, dtype=float).copy()
        operator_xy = self._operator_reference_xy()
        self.robot_base_pos = saved_pos.copy()
        self.robot_base_pos[:2] = operator_xy
        try:
            return super()._build_semantic_placement_initializer()
        finally:
            self.robot_base_pos = saved_pos

    def _before_object_placement(self):
        rotation = self._source_crosswise_rotation()
        if rotation is not None:
            cfg = next(
                item for item in self.object_cfgs
                if item.get("name") == self.SOURCE_NAME
            )
            cfg["placement"]["rotation"] = rotation
        return super()._before_object_placement()

    def _apply_source_crosswise_yaw(self):
        source_name = getattr(self, "SOURCE_NAME", None)
        if source_name is not None:
            align_object_long_axis_parallel_to_surface_edge(
                self, source_name, self.work_surface_region
            )

    def _apply_receptacle_crosswise_yaw(self):
        receptacle_name = getattr(self, "RECEPTACLE_NAME", None)
        if receptacle_name is not None:
            align_object_long_axis_parallel_to_surface_edge(
                self, receptacle_name, self.work_surface_region
            )

    def _object_half_extent_along(self, name, direction):
        obj = self.objects[name]
        half_xy = np.asarray(self._object_half_xy(name), dtype=float)
        matrix = self.sim.data.get_body_xmat(obj.root_body).reshape(3, 3)
        direction = np.asarray(direction, dtype=float)[:2]
        return float(
            abs(np.dot(matrix[:2, 0], direction)) * half_xy[0]
            + abs(np.dot(matrix[:2, 1], direction)) * half_xy[1]
        )

    def _align_objects_on_near_edge_row(self):
        names = (
            getattr(self, "SOURCE_NAME", None),
            getattr(self, "RECEPTACLE_NAME", None),
        )
        if None in names:
            return
        if any(name not in getattr(self, "objects", {}) for name in names):
            return
        operator_xy = self._operator_reference_xy()
        center, _ = self.work_surface_region.world_pose(self.sim)
        direction = np.asarray(
            self.support_region_long_edge_approach_directions(
                self.work_surface_region,
                toward_xy=operator_xy,
                sim=self.sim,
            )[0],
            dtype=float,
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        lateral = np.array([-direction[1], direction[0]], dtype=float)
        surface_extent = self.support_region_radial_extent(
            self.work_surface_region,
            direction,
            sim=self.sim,
        )
        object_extents = [
            self._object_half_extent_along(name, direction) for name in names
        ]
        # Use the larger projected half-extent for both objects. Their body
        # centers therefore have identical depth, while the larger object is
        # still fully supported with a small visible edge clearance.
        target_depth = (
            float(surface_extent)
            - max(object_extents)
            - float(self.WORKSPACE_EDGE_CLEARANCE)
        )
        for name in names:
            obj = self.objects[name]
            body_xy = self.sim.data.get_body_xpos(obj.root_body)[:2]
            current_depth = float(np.dot(body_xy - center[:2], direction))
            qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
            qpos[:2] += direction * (target_depth - current_depth)
            self.sim.data.set_joint_qpos(obj.joints[0], qpos)

        # A few scene-local desks have a collidable lower frame that reaches
        # into the otherwise valid square operation pose. Keep the source and
        # receptacle side-by-side, but allow a pick/place-only lateral shift
        # along the desk edge so the calibrated normal distance remains usable
        # without changing other task families.
        lateral_offset = self._workspace_spec().get("workspace_lateral_offset")
        if (
            lateral_offset is not None
            and getattr(self, "TASK_FAMILY", None) == "pick_place"
        ):
            lateral_offset = float(lateral_offset)
            current_offsets = [
                float(
                    np.dot(
                        self.sim.data.get_body_xpos(
                            self.objects[name].root_body
                        )[:2]
                        - center[:2],
                        lateral,
                    )
                )
                for name in names
            ]
            delta = lateral_offset - float(np.mean(current_offsets))
            if abs(delta) > 1e-9:
                for name in names:
                    qpos = self.sim.data.get_joint_qpos(
                        self.objects[name].joints[0]
                    ).copy()
                    qpos[:2] += lateral * delta
                    self.sim.data.set_joint_qpos(
                        self.objects[name].joints[0], qpos
                    )

    def _place_source_on_operator_right(self):
        """Swap the two object positions when the source is left of the operator."""

        if not self.SOURCE_ON_OPERATOR_RIGHT:
            return
        source_name = getattr(self, "SOURCE_NAME", None)
        receptacle_name = getattr(self, "RECEPTACLE_NAME", None)
        if (
            source_name not in getattr(self, "objects", {})
            or receptacle_name not in getattr(self, "objects", {})
        ):
            return
        operator_xy = np.asarray(self._operator_reference_xy(), dtype=float)
        center, _ = self.work_surface_region.world_pose(self.sim)
        forward = np.asarray(center[:2], dtype=float) - operator_xy
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm <= 1e-8:
            raise RuntimeError(
                f"{self.scene_name}: operator coincides with work-surface center"
            )
        forward /= forward_norm
        operator_right = np.array([forward[1], -forward[0]], dtype=float)

        source = self.objects[source_name]
        receptacle = self.objects[receptacle_name]
        source_xy = np.asarray(
            self.sim.data.get_body_xpos(source.root_body)[:2], dtype=float
        ).copy()
        receptacle_xy = np.asarray(
            self.sim.data.get_body_xpos(receptacle.root_body)[:2], dtype=float
        ).copy()
        separation = float(np.dot(source_xy - receptacle_xy, operator_right))
        if separation <= 0.0:
            source_qpos = self.sim.data.get_joint_qpos(source.joints[0]).copy()
            receptacle_qpos = self.sim.data.get_joint_qpos(
                receptacle.joints[0]
            ).copy()
            source_qpos[:2] += receptacle_xy - source_xy
            receptacle_qpos[:2] += source_xy - receptacle_xy
            self.sim.data.set_joint_qpos(source.joints[0], source_qpos)
            self.sim.data.set_joint_qpos(receptacle.joints[0], receptacle_qpos)
            self.sim.forward()
            source_xy = np.asarray(
                self.sim.data.get_body_xpos(source.root_body)[:2], dtype=float
            )
            receptacle_xy = np.asarray(
                self.sim.data.get_body_xpos(receptacle.root_body)[:2], dtype=float
            )
            separation = float(
                np.dot(source_xy - receptacle_xy, operator_right)
            )
        if separation < float(self.MIN_SOURCE_RIGHT_SEPARATION):
            raise RuntimeError(
                f"{self.scene_name}: source is not sufficiently to the "
                f"operator's right ({separation:.3f} m)"
            )

    def _finalize_near_edge_layout(self):
        # The source yaw changes its projected footprint. Two short passes make
        # both the crosswise yaw and the shared near-edge depth exact after
        # placement sampling and after independent pose jitter.
        for _ in range(2):
            self._apply_source_crosswise_yaw()
            self._apply_receptacle_crosswise_yaw()
            self.sim.forward()
            self._align_objects_on_near_edge_row()
            self.sim.forward()
        self._place_source_on_operator_right()

    def _object_horizontal_distance_from_robot(self, name):
        position = np.asarray(
            self.sim.data.get_body_xpos(self.objects[name].root_body),
            dtype=float,
        )
        return float(np.linalg.norm(position[:2] - self._robot_base_xy()))

    def _record_initial_manipulation_layout(
        self,
        calibrated_distance=None,
        operation_normal_distance=None,
    ):
        """Record the measured layout and enforce cheap static preconditions.

        This deliberately does not claim kinematic reachability. Fixed-base
        completion is a teleoperation acceptance check; reset-time validation
        only guarantees a collision-free base and a near-edge, forward-facing
        manipulation layout.
        """

        source_name = getattr(self, "SOURCE_NAME", None)
        if source_name not in getattr(self, "objects", {}):
            return
        base_xy = np.asarray(self._robot_base_xy(), dtype=float)
        source_distance = self._object_horizontal_distance_from_robot(
            source_name
        )
        surface_clearance = float(
            self.robot_fixture_collision_distance(
                self.work_surface_fixture,
                max_distance=2.0,
            )
        )
        layout = {
            "source_horizontal_distance": source_distance,
            "robot_work_surface_clearance": surface_clearance,
        }
        receptacle_name = getattr(self, "RECEPTACLE_NAME", None)
        if receptacle_name in getattr(self, "objects", {}):
            layout["receptacle_horizontal_distance"] = (
                self._object_horizontal_distance_from_robot(receptacle_name)
            )
        if (
            operation_normal_distance is not None
            and receptacle_name in getattr(self, "objects", {})
        ):
            outward = self._workspace_outward_direction()
            inward = -outward
            right = np.array([inward[1], -inward[0]], dtype=float)
            source_xy = np.asarray(
                self.sim.data.get_body_xpos(
                    self.objects[source_name].root_body
                )[:2],
                dtype=float,
            )
            receptacle_xy = np.asarray(
                self.sim.data.get_body_xpos(
                    self.objects[receptacle_name].root_body
                )[:2],
                dtype=float,
            )
            midpoint = 0.5 * (source_xy + receptacle_xy)
            normal_distance = float(np.dot(midpoint - base_xy, inward))
            expected_yaw = float(np.arctan2(inward[1], inward[0]))
            yaw_error = float(
                np.arctan2(
                    np.sin(self._robot_base_yaw() - expected_yaw),
                    np.cos(self._robot_base_yaw() - expected_yaw),
                )
            )
            layout.update(
                {
                    "operation_normal_distance": normal_distance,
                    "requested_operation_normal_distance": float(
                        operation_normal_distance
                    ),
                    "source_lateral_offset": float(
                        np.dot(source_xy - base_xy, right)
                    ),
                    "receptacle_lateral_offset": float(
                        np.dot(receptacle_xy - base_xy, right)
                    ),
                    "operation_yaw_error_degrees": float(
                        np.degrees(yaw_error)
                    ),
                }
            )
        self._initial_manipulation_layout = layout

        if calibrated_distance is None and operation_normal_distance is None:
            return
        failures = []
        if not self.robot_footprint_is_on_floor(base_xy, radius=0.25):
            failures.append("robot footprint leaves Floor")
        collisions = self.count_body_collisions()
        if collisions:
            failures.append(f"robot has {collisions} scene collisions")
        if not np.isfinite(surface_clearance) or surface_clearance <= 0.0:
            failures.append("robot touches the named work surface")
        if operation_normal_distance is not None:
            if abs(layout["operation_yaw_error_degrees"]) > 1e-6:
                failures.append("robot is not square to the work-surface edge")
            if layout["source_lateral_offset"] <= 0.0:
                failures.append("source is not on the robot's right")
            if layout["receptacle_lateral_offset"] >= 0.0:
                failures.append("receptacle is not on the robot's left")

        yaw = self._robot_base_yaw()
        forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
        object_names = [source_name]
        if receptacle_name in getattr(self, "objects", {}):
            object_names.append(receptacle_name)
        for name in object_names:
            position = np.asarray(
                self.sim.data.get_body_xpos(self.objects[name].root_body),
                dtype=float,
            )[:2]
            if float(np.dot(position - base_xy, forward)) <= 0.0:
                failures.append(f"{name} is not in front of the robot")

        if len(object_names) == 2:
            center, _ = self.work_surface_region.world_pose(self.sim)
            direction = np.asarray(
                self.support_region_long_edge_approach_directions(
                    self.work_surface_region,
                    toward_xy=base_xy,
                    sim=self.sim,
                )[0],
                dtype=float,
            )
            direction /= max(float(np.linalg.norm(direction)), 1e-8)
            depths = [
                float(
                    np.dot(
                        self.sim.data.get_body_xpos(
                            self.objects[name].root_body
                        )[:2]
                        - center[:2],
                        direction,
                    )
                )
                for name in object_names
            ]
            if abs(depths[0] - depths[1]) > self.NEAR_EDGE_ROW_TOLERANCE:
                failures.append("source and receptacle are not on one edge row")

        if failures:
            raise RuntimeError(
                f"{self.scene_name}: invalid fixed-base manipulation layout: "
                + "; ".join(failures)
            )
        if calibrated_distance is not None:
            layout["calibrated_source_horizontal_distance"] = float(
                calibrated_distance
            )
        layout["static_contract_verified"] = True

    def _record_operation_reference_layout(self):
        """Record near-workspace geometry without moving a distant-start robot."""

        source_name = getattr(self, "SOURCE_NAME", None)
        if source_name not in getattr(self, "objects", {}):
            return
        source_xy = np.asarray(
            self.sim.data.get_body_xpos(
                self.objects[source_name].root_body
            )[:2],
            dtype=float,
        )
        receptacle_name = getattr(self, "RECEPTACLE_NAME", None)
        if receptacle_name not in getattr(self, "objects", {}):
            return
        receptacle_xy = np.asarray(
            self.sim.data.get_body_xpos(
                self.objects[receptacle_name].root_body
            )[:2],
            dtype=float,
        )
        requested_distance = float(
            self._workspace_spec()["operation_normal_distance"]
        )
        reference_xy, reference_yaw = self._operation_pose(requested_distance)
        inward = -self._workspace_outward_direction()
        right = np.array([inward[1], -inward[0]], dtype=float)
        layout = {
            "operation_reference_xy": np.asarray(reference_xy).tolist(),
            "operation_reference_yaw": float(reference_yaw),
            "operation_normal_distance": requested_distance,
            "operation_source_horizontal_distance": float(
                np.linalg.norm(source_xy - reference_xy)
            ),
            "operation_source_lateral_offset": float(
                np.dot(source_xy - reference_xy, right)
            ),
            "operation_receptacle_horizontal_distance": float(
                np.linalg.norm(receptacle_xy - reference_xy)
            ),
            "operation_receptacle_lateral_offset": float(
                np.dot(receptacle_xy - reference_xy, right)
            ),
        }
        self._initial_manipulation_layout = layout

    def _ensure_surface_object_contacts(self):
        names = (
            getattr(self, "SOURCE_NAME", None),
            getattr(self, "RECEPTACLE_NAME", None),
        )
        if (
            None in names
            or self.work_surface_region is None
            or any(name not in getattr(self, "objects", {}) for name in names)
        ):
            return
        center, _ = self.work_surface_region.world_pose(self.sim)
        direction = np.asarray(
            self.support_region_long_edge_approach_directions(
                self.work_surface_region,
                toward_xy=self._operator_reference_xy(),
                sim=self.sim,
            )[0],
            dtype=float,
        )
        direction /= max(float(np.linalg.norm(direction)), 1e-8)
        lateral = np.array([-direction[1], direction[0]], dtype=float)
        positions = {
            name: self.sim.data.get_body_xpos(
                self.objects[name].root_body
            )[:2].copy()
            for name in names
        }
        requested_depth = min(
            float(np.dot(position - center[:2], direction))
            for position in positions.values()
        )
        lateral_offsets = {
            name: float(np.dot(position - center[:2], lateral))
            for name, position in positions.items()
        }
        for inward_fraction in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0):
            depth = requested_depth * (1.0 - inward_fraction)
            placed = True
            for name in names:
                xy = (
                    center[:2]
                    + direction * depth
                    + lateral * lateral_offsets[name]
                )
                placed = bool(
                    self.place_object_on_support_contact(
                        name,
                        self.work_surface_region,
                        fixture=self.work_surface_fixture,
                        xy=xy,
                        inward_fractions=(0.0,),
                        probe_below=0.08,
                        probe_step=0.001,
                        penetration_tolerance=0.002,
                    )
                    and placed
                )
            if placed:
                self.sim.forward()
                return
        raise RuntimeError(
            f"{self.scene_name}: no shared contacted edge row on "
            f"{self.work_surface_region.stable_id}"
        )

    def _apply_object_placements(self, placements):
        super()._apply_object_placements(placements)
        self.sim.forward()
        self._finalize_near_edge_layout()
        self._ensure_surface_object_contacts()
        self._apply_calibrated_operation_pose()

    def _apply_calibrated_operation_pose(self, *, include_composites=False):
        """Place the robot square to the object row at the calibrated distance."""

        if (
            getattr(self, "TASK_FAMILY", None) == "composite"
            and not include_composites
        ):
            self._record_operation_reference_layout()
            return
        desired = self._workspace_spec().get("operation_normal_distance")
        source_name = getattr(self, "SOURCE_NAME", None)
        receptacle_name = getattr(self, "RECEPTACLE_NAME", None)
        if (
            source_name not in getattr(self, "objects", {})
            or receptacle_name not in getattr(self, "objects", {})
        ):
            return
        if desired is None:
            self._record_initial_manipulation_layout()
            return
        desired = float(desired)
        if hasattr(self, "_robot_free_joint_z"):
            delattr(self, "_robot_free_joint_z")
        selected = None
        for offset in self.OPERATION_DISTANCE_OFFSETS:
            distance = desired + float(offset)
            target_xy, yaw = self._operation_pose(distance)
            valid = bool(
                self.robot_footprint_is_on_floor(target_xy, radius=0.25)
                and self._set_robot_free_pose(target_xy, yaw)
                and self._calibrate_robot_free_joint_height()
            )
            self.sim.forward()
            clearance = float(
                self.robot_fixture_collision_distance(
                    self.work_surface_fixture,
                    max_distance=2.0,
                )
            )
            if (
                valid
                and self.count_body_collisions() == 0
                and np.isfinite(clearance)
                and clearance > 0.0
            ):
                selected = (target_xy, yaw, distance)
                break
        if selected is None:
            raise RuntimeError(
                f"{self.scene_name}: no collision-free table-square pose "
                f"for G1 from {desired:.3f} m"
            )
        target_xy, yaw, actual_normal_distance = selected
        self.robot_base_pos = np.array([target_xy[0], target_xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        self._record_initial_manipulation_layout(
            operation_normal_distance=desired
        )
        self._initial_manipulation_layout[
            "selected_operation_normal_distance"
        ] = float(actual_normal_distance)

    def _constrain_initial_jitter_candidate(self):
        ret = super()._constrain_initial_jitter_candidate()
        self._finalize_near_edge_layout()
        self._ensure_surface_object_contacts()
        self._apply_calibrated_operation_pose()
        return ret

    def _rebuild_surface_placement_on_reset(self):
        return True

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["removed_workspace_fixtures"] = list(
            self.removed_workspace_fixture_ids
        )
        ep_meta["source_orientation"] = (
            "long axis is perpendicular to the near operator-to-object direction"
        )
        ep_meta["workspace_layout"] = (
            "source and receptacle are side by side at the same operator-facing "
            f"edge depth with {self.WORKSPACE_EDGE_CLEARANCE:.3f} m minimum clearance"
        )
        if self.SOURCE_ON_OPERATOR_RIGHT:
            ep_meta["workspace_layout"] += "; source is on the operator's right"
        layout = getattr(self, "_initial_manipulation_layout", {})
        ep_meta.update(layout)
        return ep_meta


__all__ = [
    "BedroomSurfacePlacementMixin",
    "align_object_long_axis_parallel_to_surface_edge",
]
