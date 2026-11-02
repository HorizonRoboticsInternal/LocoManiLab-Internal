"""iTHOR pick-and-place atomic tasks."""

from dataclasses import replace

import numpy as np

from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.models.scenes.thor_semantics import SemanticRegion


SHALLOW_RECEPTACLE_MAX_HALF_HEIGHT = 0.08


def resolve_task_support_region(region):
    """Return a task-local physical support view of a semantic region.

    Some iTHOR tabletops are exported as shallow receptacle volumes. Object
    placement must use their lower face, where the collidable tabletop lives,
    rather than the receptacle center. True support regions are copied without
    changing their geometry. The semantic registry is never mutated.
    """

    if not isinstance(region, SemanticRegion):
        raise TypeError("task support region must be a SemanticRegion")

    local_center = np.asarray(region.local_center, dtype=float).copy()
    world_center = np.asarray(region._world_center, dtype=float).copy()
    world_offset = np.asarray(region._world_offset, dtype=float).copy()
    half_size = np.asarray(region.half_size, dtype=float).copy()
    if region.kind == "support":
        return replace(
            region,
            local_center=local_center,
            half_size=half_size,
            _world_center=world_center,
            _world_offset=world_offset,
        )
    if region.kind != "receptacle":
        raise ValueError(
            f"{region.stable_id}: {region.kind!r} is not a placement support"
        )
    if float(half_size[2]) > SHALLOW_RECEPTACLE_MAX_HALF_HEIGHT:
        raise ValueError(
            f"{region.stable_id}: receptacle is too deep to be a support"
        )

    # Region half-sizes are world-axis projections. Applying this displacement
    # to ``local_center[2]`` is incorrect for iTHOR bodies whose local Z axis is
    # rotated into world X/Y: it moves an edge placement sideways instead of
    # lowering it. Keep the parent-relative center unchanged and express the
    # shallow volume's lower face as an explicit world-vertical offset.
    world_offset[2] -= float(half_size[2])
    return replace(
        region,
        kind="support",
        local_center=local_center,
        half_size=half_size,
        _world_center=world_center,
        _world_offset=world_offset,
    )


class RoomMultiLevelFixturePnP:
    """Move an object from the highest support of one fixture to its lowest."""

    ATOMIC_SKILL = "PickPlaceOnSurface"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "pick_place"

    TARGET_CATEGORY = None
    TARGET_ASSET_ID = None
    SOURCE_NAME = "object"
    SOURCE_GROUP = None
    SOURCE_LANG = "object"
    SOURCE_SCALE = 1.0
    SOURCE_MAX_SIZE = (None, None, None)
    SOURCE_ROTATION = None
    SOURCE_PLACEMENT_MARGIN = 0.03
    # ``center`` gives a fixed source pose; ``uniform`` samples the usable
    # semantic region uniformly in XY.
    SOURCE_PLACEMENT_STRATEGY = "center"
    # Task-local multiplier for both horizontal semantic support regions.
    # The source / target values below override this common value when set.
    REGION_XY_SCALE = 1.0
    SOURCE_REGION_XY_SCALE = None
    TARGET_REGION_XY_SCALE = None
    SOURCE_PLACEMENT_NEAR_ROBOT_HALF = False
    SOURCE_REGION_INDEX = None
    TARGET_REGION_INDEX = None
    HOLD_STEPS = 5
    RELEASE_DISTANCE = 0.18

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        fixtures = self.get_fixture(
            category=self.TARGET_CATEGORY,
            return_all=True,
        )
        self.multilevel_fixture = next(
            (
                fixture
                for fixture in fixtures
                if self.TARGET_ASSET_ID is None
                or fixture.asset_id == self.TARGET_ASSET_ID
            ),
            None,
        )
        self.source_region = None
        self.target_region = None
        if self.multilevel_fixture is None:
            return

        support_regions = self.get_region(
            fixture=self.multilevel_fixture,
            kind="support",
            return_all=True,
        )
        if len(support_regions) < 2:
            return
        if self.SOURCE_REGION_INDEX is None and self.TARGET_REGION_INDEX is None:
            support_regions = sorted(
                support_regions,
                key=lambda region: (
                    float(region.world_pose()[0][2]),
                    region.stable_id,
                ),
            )
            self.target_region = support_regions[0]
            self.source_region = support_regions[-1]
            if (
                self.source_region.world_pose()[0][2]
                - self.target_region.world_pose()[0][2]
                < 0.12
            ):
                self.source_region = None
                self.target_region = None
                return
        else:
            # Explicit indices refer to the fixture export order. For shelf
            # assets this is the numbered receptacle order (0, 1, 2, ...),
            # which is independent of each receptacle's physical height.
            support_regions = [
                region
                for region in self.multilevel_fixture.regions.values()
                if region.kind == "support"
            ]
            source_index = (
                len(support_regions) - 1
                if self.SOURCE_REGION_INDEX is None
                else self.SOURCE_REGION_INDEX
            )
            target_index = (
                0 if self.TARGET_REGION_INDEX is None else self.TARGET_REGION_INDEX
            )
            if not (
                0 <= source_index < len(support_regions)
                and 0 <= target_index < len(support_regions)
                and source_index != target_index
            ):
                return
            self.source_region = support_regions[source_index]
            self.target_region = support_regions[target_index]
        source_scale = (
            self.REGION_XY_SCALE
            if self.SOURCE_REGION_XY_SCALE is None
            else self.SOURCE_REGION_XY_SCALE
        )
        target_scale = (
            self.REGION_XY_SCALE
            if self.TARGET_REGION_XY_SCALE is None
            else self.TARGET_REGION_XY_SCALE
        )
        if source_scale <= 0 or target_scale <= 0:
            raise ValueError("semantic region XY scales must be positive")

        def _scaled_region(region, scale):
            if scale == 1.0:
                return region
            half_size = np.asarray(region.half_size, dtype=float).copy()
            half_size[:2] *= scale
            return replace(region, half_size=half_size)

        # Keep these overrides task-local: the semantic registry itself is
        # shared by fixture lookup and must retain its physical extents.
        self.source_region = _scaled_region(self.source_region, source_scale)
        self.target_region = _scaled_region(self.target_region, target_scale)
        self.fixture_refs["multilevel_fixture"] = self.multilevel_fixture
        self.region_refs["source_shelf"] = self.source_region
        self.region_refs["target_shelf"] = self.target_region

    def _near_robot_half_ranges(self, region, half):
        center, yaw = region.world_pose()
        delta = np.asarray(self.robot_base_pos[:2], dtype=float) - center[:2]
        c, s = np.cos(-yaw), np.sin(-yaw)
        local_robot = np.array(
            [c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]]
        )
        axis = int(np.argmax(np.abs(local_robot)))
        sign = 1.0 if local_robot[axis] >= 0.0 else -1.0
        ranges = [[-float(half[0]), float(half[0])], [-float(half[1]), float(half[1])]]
        ranges[axis] = sorted([0.0, sign * float(half[axis])])
        return ranges[0], ranges[1]

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.multilevel_fixture is not None
            and self.source_region is not None
            and self.target_region is not None
        )

    def _get_obj_cfgs(self):
        cfg = dict(
            name=self.SOURCE_NAME,
            obj_groups=self.SOURCE_GROUP,
            object_scale=self.SOURCE_SCALE,
            placement=dict(
                region=self.source_region,
                margin=self.SOURCE_PLACEMENT_MARGIN,
                rotation=self.SOURCE_ROTATION,
                strategy=self.SOURCE_PLACEMENT_STRATEGY,
                near_robot_half=self.SOURCE_PLACEMENT_NEAR_ROBOT_HALF,
            ),
        )
        if any(size is not None for size in self.SOURCE_MAX_SIZE):
            cfg["max_size"] = self.SOURCE_MAX_SIZE
        return [cfg]

    def _setup_scene(self):
        super()._setup_scene()
        self._success_hold_steps = 0

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self._object_is_on_target_shelf_and_released():
            self._success_hold_steps += 1
        else:
            self._success_hold_steps = 0
        return ret

    def _object_is_on_target_shelf_and_released(self):
        # Exported shelf receptacles are volumetric sites, whereas ordinary
        # tabletops are thin support patches.  Use the corresponding spatial
        # predicate so both kinds of semantic region can share this task.
        if self.target_region.half_size[2] > 0.05:
            placed = self.object_in_region(self.SOURCE_NAME, self.target_region)
        else:
            placed = self.object_on_region(self.SOURCE_NAME, self.target_region)
        return bool(
            placed and self.object_released(self.SOURCE_NAME, th=self.RELEASE_DISTANCE)
        )

    def _check_success(self):
        return self._success_hold_steps >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            f"pick up the {self.SOURCE_LANG} from the upper shelf and place it "
            "on the lower shelf"
        )
        return ep_meta


class SupportContactPlacementMixin:
    """Place free-joint task objects on named supports by real contact.

    This mixin deliberately has no environment base class and no setup hooks.
    It only depends on the iTHOR environment geometry helpers used by its
    methods, so exact-scene and semantic-surface tasks can share the same
    bounded contact probe without changing their initialization lifecycle.
    """

    def _set_task_object_bottom_at(self, name, xy, bottom_z):
        obj = self.objects[name]
        self.sim.forward()
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        body_xy = np.asarray(
            self.sim.data.get_body_xpos(obj.root_body)[:2], dtype=float
        )
        qpos[:2] += np.asarray(xy, dtype=float)[:2] - body_xy
        qpos[2] += float(bottom_z) - self.object_bottom_z(name)
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        joint_id = self.sim.model.joint_name2id(obj.joints[0])
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[dof_address : dof_address + 6] = 0.0
        self.sim.forward()

    def _object_fixture_contact_is_shallow(
        self, name, fixture, penetration_tolerance
    ):
        object_bodies = self._body_tree_ids([self.obj_body_id[name]])
        fixture_bodies = self._fixture_body_ids(fixture, include_siblings=True)
        distances = []
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body1 in object_bodies and body2 in fixture_bodies) or (
                body2 in object_bodies and body1 in fixture_bodies
            ):
                distances.append(float(contact.dist))
        return bool(
            distances and min(distances) >= -float(penetration_tolerance)
        )

    def place_object_on_support_contact(
        self,
        name,
        region,
        fixture=None,
        *,
        xy=None,
        inward_fractions=(0.0, 0.10, 0.20, 0.35, 0.50, 0.75, 1.0),
        probe_above=0.08,
        probe_below=0.15,
        probe_step=0.002,
        penetration_tolerance=0.0025,
    ):
        """Place an object at the first real, non-penetrating support contact.

        Semantic support heights are only search anchors: exported shallow
        receptacles can remain centimetres above the collidable tabletop even
        after lower-face conversion. For each bounded inward XY candidate this
        method scans down from free space, accepts the first named-fixture
        contact, rejects penetration and unrelated collisions, and clears the
        object's free-joint velocity.
        """

        if not isinstance(region, SemanticRegion) or region.kind != "support":
            raise ValueError("contact placement requires a semantic support")
        if name not in self.objects:
            raise KeyError(f"unknown task object {name!r}")
        fixture = fixture or self.get_fixture(id=region.fixture_id)
        if fixture is None or fixture.stable_id != region.fixture_id:
            raise ValueError(
                f"{region.stable_id}: support fixture does not match region"
            )
        if probe_above < 0.0 or probe_below < 0.0 or probe_step <= 0.0:
            raise ValueError("support probe bounds and step must be positive")
        contracts = getattr(self, "_task_support_contracts", None)
        if not isinstance(contracts, dict):
            contracts = {}
            self._task_support_contracts = contracts
        # Do not let a failed probe on a later reset inherit an old contract.
        contracts.pop(name, None)

        center = region.world_pose(self.sim)[0]
        requested_xy = (
            center[:2] if xy is None else np.asarray(xy, dtype=float)[:2]
        )
        anchor_z = float(center[2])
        z_values = np.arange(
            anchor_z + float(probe_above),
            anchor_z - float(probe_below) - 0.5 * float(probe_step),
            -float(probe_step),
        )
        z_tolerance = float(probe_above + probe_below + probe_step)
        for inward_fraction in inward_fractions:
            fraction = float(inward_fraction)
            if fraction < 0.0 or fraction > 1.0:
                raise ValueError("inward fractions must be in [0, 1]")
            candidate_xy = (
                (1.0 - fraction) * requested_xy + fraction * center[:2]
            )
            for bottom_z in z_values:
                self._set_task_object_bottom_at(name, candidate_xy, bottom_z)
                if not self.object_on_region(
                    name,
                    region,
                    z_tol=z_tolerance,
                    require_contact=True,
                ):
                    continue
                if not self._object_fixture_contact_is_shallow(
                    name, fixture, penetration_tolerance
                ):
                    continue
                if self._object_has_illegal_penetration(
                    name, region, tolerance=penetration_tolerance
                ):
                    continue
                self._set_task_object_bottom_at(
                    name, candidate_xy, self.object_bottom_z(name)
                )
                contracts[name] = region
                return True
        return False


class SemanticSurfaceTask(SupportContactPlacementMixin, SceneSpecMixin):
    """Base for calibrated tasks performed on a semantic work surface."""

    SEMANTIC_PLACEMENT_ATTEMPTS = 30
    MIN_SURFACE_HEIGHT = 0.35
    MAX_SURFACE_HEIGHT = 1.20
    MIN_SURFACE_HALF_XY = (0.18, 0.16)
    STAND_DISTANCES = (0.30, 0.40, 0.50, 0.65, 0.80)
    STAND_LATERAL_OFFSETS = (0.0, 0.15, -0.15, 0.30, -0.30)
    INITIAL_ROBOT_JITTER_RADIUS = 0.04
    INITIAL_OBJECT_JITTER_RADIUS = 0.04
    WORK_SURFACE_FIXTURES = None
    WORK_SURFACE_REGIONS = None

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.work_surface_region = None
        self.work_surface_fixture = None
        self._select_work_surface()
        if self.work_surface_region is None:
            return
        self.fixture_refs["task_surface"] = self.work_surface_fixture
        self.region_refs["task_surface"] = self.work_surface_region
        self._set_planned_surface_pose()

    def _select_work_surface(self):
        fixture_ids = self.WORK_SURFACE_FIXTURES
        region_ids = self.WORK_SURFACE_REGIONS
        if (
            not isinstance(fixture_ids, dict)
            or not isinstance(region_ids, dict)
            or self.scene_name not in fixture_ids
            or self.scene_name not in region_ids
        ):
            return
        expected_fixture_id = fixture_ids[self.scene_name]
        fixture = self.get_fixture(id=expected_fixture_id)
        if fixture is None or fixture.stable_id != expected_fixture_id:
            return
        raw_region = next(
            (
                region
                for region in self.get_region(
                    fixture=fixture,
                    return_all=True,
                )
                if region.stable_id == region_ids[self.scene_name]
            ),
            None,
        )
        if raw_region is None or raw_region.fixture_id != fixture.stable_id:
            return
        try:
            region = resolve_task_support_region(raw_region)
        except ValueError:
            return
        requirements = {
            "height_range": (self.MIN_SURFACE_HEIGHT, self.MAX_SURFACE_HEIGHT),
            "min_half_xy": self.MIN_SURFACE_HALF_XY,
        }
        if not self._support_matches_requirements(
            region, fixture, requirements
        ):
            return
        self.work_surface_region = region
        self.work_surface_fixture = fixture

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.work_surface_region is not None
            and self.work_surface_fixture is not None
        )

    def _approach_directions(self):
        return self.support_region_long_edge_approach_directions(
            self.work_surface_region,
            toward_xy=self.robot_base_pos[:2],
        )

    def _set_planned_surface_pose(self):
        center, _ = self.work_surface_region.world_pose()
        direction = self._approach_directions()[0]
        distance = self.support_region_radial_extent(
            self.work_surface_region, direction
        ) + float(self.STAND_DISTANCES[0])
        xy = center[:2] + direction * distance
        yaw = float(np.arctan2(center[1] - xy[1], center[0] - xy[0]))
        self.set_robot_base_pose([xy[0], xy[1], 0.0], yaw)

    def _setup_scene(self):
        super()._setup_scene()
        if not self._place_robot_at_surface():
            raise RuntimeError(
                f"{self.scene_name}: no collision-free pose at task surface"
            )

    def _place_robot_at_surface(self):
        center, _ = self.work_surface_region.world_pose(self.sim)
        for direction in self._approach_directions():
            direction = np.asarray(direction, dtype=float)
            extent = self.support_region_radial_extent(
                self.work_surface_region, direction
            )
            lateral = np.array([-direction[1], direction[0]], dtype=float)
            for distance in self.STAND_DISTANCES:
                for offset in self.STAND_LATERAL_OFFSETS:
                    xy = center[:2] + direction * (extent + distance)
                    xy = xy + lateral * offset
                    yaw = float(
                        np.arctan2(center[1] - xy[1], center[0] - xy[0])
                    )
                    if (
                        self.robot_footprint_is_on_floor(xy, radius=0.25)
                        and self._set_robot_free_pose(xy, yaw)
                        and self._calibrate_robot_free_joint_height()
                        and self.count_body_collisions() == 0
                    ):
                        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
                        self.robot_base_ori = np.array([0.0, 0.0, yaw])
                        return True
        return False

    def _before_object_placement(self):
        if self.object_cfgs and self._rebuild_surface_placement_on_reset():
            self.placement_initializer = self._build_placement_initializer()
            self.object_placements = self.placement_initializer.sample()

    def _rebuild_surface_placement_on_reset(self):
        return True


class SurfaceObjectInReceptacle(SemanticSurfaceTask):
    """Place one spawned object into one spawned receptacle."""

    ATOMIC_SKILL = "PickPlace"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "pick_place"
    SOURCE_NAME = "item"
    SOURCE_GROUP = "apple"
    SOURCE_MJCF_PATH = None
    SOURCE_SCALE = 0.65
    RECEPTACLE_NAME = "receptacle"
    RECEPTACLE_GROUP = "bowl"
    RECEPTACLE_MJCF_PATH = None
    RECEPTACLE_SCALE = 0.75
    RECEPTACLE_EDGE_FRACTION = -0.40
    RECEPTACLE_LATERAL_OFFSET = None
    SOURCE_EDGE_FRACTION = 0.55
    SOURCE_LATERAL_OFFSET = None
    HOLD_STEPS = 5
    RELEASE_DISTANCE = 0.18
    RECEPTACLE_INTERIOR_SCALE = 0.82
    RECEPTACLE_XY_CONTAINMENT = "bbox"
    RECEPTACLE_UPRIGHT_COSINE = float(np.cos(np.deg2rad(20.0)))
    RECEPTACLE_SUPPORT_Z_TOLERANCE = 0.05
    RECEPTACLE_SUPPORT_PENETRATION_TOLERANCE = 0.003
    RECEPTACLE_RIM_CLEARANCE = 0.005
    RECEPTACLE_BOTTOM_TOLERANCE = 0.005

    def _get_obj_cfgs(self):
        if self.work_surface_region is None:
            return []
        receptacle = {
            "name": self.RECEPTACLE_NAME,
            "obj_groups": self.RECEPTACLE_GROUP,
            "object_scale": self.RECEPTACLE_SCALE,
            "placement": {
                "region": self.work_surface_region,
                "margin": 0.03,
                "strategy": "edge",
                "edge_fraction": self.RECEPTACLE_EDGE_FRACTION,
                "edge_lateral_offset": self.RECEPTACLE_LATERAL_OFFSET,
                "rotation": 0.0,
            },
        }
        source = {
            "name": self.SOURCE_NAME,
            "obj_groups": self.SOURCE_GROUP,
            "object_scale": self.SOURCE_SCALE,
            "placement": {
                "region": self.work_surface_region,
                "margin": 0.04,
                "strategy": "edge",
                "edge_fraction": self.SOURCE_EDGE_FRACTION,
                "edge_lateral_offset": self.SOURCE_LATERAL_OFFSET,
                "rotation": 0.0,
            },
        }
        if self.RECEPTACLE_MJCF_PATH:
            receptacle["info"] = {"mjcf_path": self.RECEPTACLE_MJCF_PATH}
        if self.SOURCE_MJCF_PATH:
            source["info"] = {"mjcf_path": self.SOURCE_MJCF_PATH}
        return [receptacle, source]

    def _setup_scene(self):
        super()._setup_scene()
        self._reset_task_progress()

    def _reset_task_progress(self):
        parent = getattr(super(), "_reset_task_progress", None)
        if callable(parent):
            parent()
        self._success_hold = 0

    def _receptacle_is_upright_on_support(self):
        receptacle = self.objects[self.RECEPTACLE_NAME]
        matrix = np.asarray(
            self.sim.data.get_body_xmat(receptacle.root_body), dtype=float
        ).reshape(3, 3)
        upright_cosine = self.RECEPTACLE_UPRIGHT_COSINE
        upright = (
            True
            if upright_cosine is None
            else float(matrix[2, 2]) >= float(upright_cosine)
        )
        support_tolerance = self.RECEPTACLE_SUPPORT_PENETRATION_TOLERANCE
        return bool(
            upright
            and self.object_on_region(
                self.RECEPTACLE_NAME,
                self.work_surface_region,
                z_tol=self.RECEPTACLE_SUPPORT_Z_TOLERANCE,
                require_contact=True,
            )
            and self._object_fixture_contact_is_shallow(
                self.RECEPTACLE_NAME,
                self.work_surface_fixture,
                support_tolerance,
            )
            and not self._object_has_illegal_penetration(
                self.RECEPTACLE_NAME,
                self.work_surface_region,
                tolerance=support_tolerance,
            )
            and self.object_released(
                self.RECEPTACLE_NAME, th=self.RELEASE_DISTANCE
            )
        )

    def _source_bbox_inside_receptacle(self):
        """Check configured horizontal containment and bottom insertion."""

        source_points = self.object_bbox_points(self.SOURCE_NAME)
        receptacle_points = self.object_bbox_points(self.RECEPTACLE_NAME)
        receptacle = self.objects[self.RECEPTACLE_NAME]
        origin = np.asarray(
            self.sim.data.get_body_xpos(receptacle.root_body), dtype=float
        )
        matrix = np.asarray(
            self.sim.data.get_body_xmat(receptacle.root_body), dtype=float
        ).reshape(3, 3)
        receptacle_local = (receptacle_points - origin) @ matrix
        source_local = (source_points - origin) @ matrix
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
        containment = self.RECEPTACLE_XY_CONTAINMENT
        if containment == "bbox":
            source_xy = source_local[:, :2]
        elif containment == "center":
            source_xy = 0.5 * (
                np.min(source_local[:, :2], axis=0)
                + np.max(source_local[:, :2], axis=0)
            )[None, :]
        else:
            raise ValueError(
                "RECEPTACLE_XY_CONTAINMENT must be 'bbox' or 'center'"
            )
        normalized = (source_xy - center_xy) / interior_half
        inside_xy = bool(
            np.all(np.sum(normalized * normalized, axis=1) <= 1.0)
        )
        source_bottom = float(np.min(source_local[:, 2]))
        below_rim = source_bottom <= (
            float(upper[2]) - float(self.RECEPTACLE_RIM_CLEARANCE)
        )
        above_bottom = source_bottom >= (
            float(lower[2]) - float(self.RECEPTACLE_BOTTOM_TOLERANCE)
        )
        return bool(inside_xy and below_rim and above_bottom)

    def _raw_place_success(self):
        source = self.objects[self.SOURCE_NAME]
        receptacle = self.objects[self.RECEPTACLE_NAME]
        return bool(
            self.check_contact(source, receptacle)
            and self._source_bbox_inside_receptacle()
            and self.object_released(
                self.SOURCE_NAME, th=self.RELEASE_DISTANCE
            )
            and self._receptacle_is_upright_on_support()
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold = (
            self._success_hold + 1 if self._raw_place_success() else 0
        )
        return ret

    def _check_success(self):
        return self._success_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.scene_task_spec()["lang"]
        ep_meta["surface_fixture"] = self.work_surface_fixture.stable_id
        ep_meta["success_definition"] = (
            "the released source bounding box must fit inside and contact the "
            "upright receptacle while the released receptacle remains in "
            "contact with the named work surface"
        )
        return ep_meta


__all__ = [
    "RoomMultiLevelFixturePnP",
    "SemanticSurfaceTask",
    "SupportContactPlacementMixin",
    "SurfaceObjectInReceptacle",
    "resolve_task_support_region",
]
