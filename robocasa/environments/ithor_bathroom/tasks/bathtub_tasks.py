"""Low-occlusion bathtub and garbage-can tasks for bathroom FloorPlan401."""

import numpy as np

from robocasa.environments.ithor_atomic.pick_place import (
    SupportContactPlacementMixin,
)
from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_atomic.task_assets import (
    SPONGE_MODEL,
    SPONGE_SCALE,
    asset_xml_path,
)
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_specs,
    get_supported_scene_ids,
)


class _BathroomNorthRimTask(SupportContactPlacementMixin, SceneSpecMixin):
    """Stage one small object on the open north rim of FloorPlan401's bathtub."""

    RIM_ANCHOR_TOLERANCE = 0.02
    ROBOT_STAND_DISTANCES = (0.65, 0.75, 0.85, 1.00)
    ROBOT_LATERAL_OFFSETS = (0.0, 0.10, -0.10, 0.20, -0.20)
    HOLD_STEPS = 5
    RELEASE_DISTANCE = 0.18

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.rim_anchor_xy = np.asarray(spec["rim_anchor_xy"], dtype=float)
        self.bathtub_fixture = self.get_fixture(id=spec["bathtub_fixture"])
        self.rim_fixture = self.get_fixture(id=spec["rim_fixture"])
        if (
            self.bathtub_fixture is not None
            and self.bathtub_fixture.stable_id != spec["bathtub_fixture"]
        ):
            self.bathtub_fixture = None
        if (
            self.rim_fixture is not None
            and self.rim_fixture.stable_id != spec["rim_fixture"]
        ):
            self.rim_fixture = None
        self.rim_clutter_fixture_ids = tuple(spec["remove_fixtures"])
        self.rim_clutter_fixtures = tuple(
            fixture
            for stable_id in self.rim_clutter_fixture_ids
            for fixture in (self.get_fixture(id=stable_id),)
            if fixture is not None and fixture.stable_id == stable_id
        )
        self._missing_rim_clutter_ids = tuple(
            stable_id
            for stable_id in self.rim_clutter_fixture_ids
            if all(
                fixture.stable_id != stable_id for fixture in self.rim_clutter_fixtures
            )
        )
        self.rim_region = (
            next(
                (
                    region
                    for region in self.get_region(
                        fixture=self.rim_fixture,
                        kind="support",
                        return_all=True,
                    )
                    if region.stable_id == spec["rim_region"]
                ),
                None,
            )
            if self.rim_fixture is not None
            else None
        )
        if self.bathtub_fixture is not None:
            self.fixture_refs["bathtub"] = self.bathtub_fixture
        if self.rim_fixture is not None and self.rim_region is not None:
            self.fixture_refs["bathtub_north_rim"] = self.rim_fixture
            self.region_refs["bathtub_north_rim"] = self.rim_region
            self._remove_exact_rim_clutter()

    def _is_scene_valid(self):
        if not super()._is_scene_valid():
            return False
        if self.bathtub_fixture is None:
            self._scene_invalid_reason = (
                f"{self.scene_name}: bathtub fixture is missing"
            )
            return False
        if self.rim_fixture is None or self.rim_region is None:
            self._scene_invalid_reason = (
                f"{self.scene_name}: bathtub north-rim support is missing"
            )
            return False
        if self._missing_rim_clutter_ids:
            self._scene_invalid_reason = (
                f"{self.scene_name}: exact rim clutter fixtures are missing: "
                f"{self._missing_rim_clutter_ids}"
            )
            return False
        if not self._rim_clutter_removal_complete:
            self._scene_invalid_reason = (
                f"{self.scene_name}: exact rim clutter could not be removed"
            )
            return False
        center, _ = self.rim_region.world_pose()
        if (
            float(np.linalg.norm(center[:2] - self.rim_anchor_xy))
            > self.RIM_ANCHOR_TOLERANCE
        ):
            self._scene_invalid_reason = (
                f"{self.scene_name}: exact bathtub rim disagrees with its "
                "calibrated anchor"
            )
            return False
        return True

    def _remove_exact_rim_clutter(self):
        """Remove only scene fixtures explicitly named by the task spec."""

        removed_bodies = self.remove_semantic_fixtures(self.rim_clutter_fixtures)
        expected_bodies = {
            fixture.body_name
            for fixture in self.rim_clutter_fixtures
            if fixture.body_name is not None
        }
        self._removed_rim_clutter_body_names = tuple(sorted(removed_bodies))
        self._rim_clutter_removal_complete = bool(
            not self._missing_rim_clutter_ids
            and expected_bodies.issubset(removed_bodies)
        )

    def _setup_scene(self):
        super()._setup_scene()
        self._reset_task_progress()
        if not self._place_robot_at_north_rim():
            raise RuntimeError(
                f"{self.scene_name}: no collision-free G1 pose at bathtub north rim"
            )

    def _reset_task_progress(self):
        parent_reset = getattr(super(), "_reset_task_progress", None)
        if callable(parent_reset):
            parent_reset()
        self._success_hold_steps = 0

    def _place_robot_at_north_rim(self):
        """Search after MuJoCo compilation and accept only zero-collision poses."""

        target, _ = self.rim_region.world_pose(self.sim)
        primary = np.asarray(self.room_center[:2], dtype=float) - target[:2]
        primary /= max(float(np.linalg.norm(primary)), 1e-8)
        base_angle = float(np.arctan2(primary[1], primary[0]))
        directions = [
            np.array([np.cos(base_angle + offset), np.sin(base_angle + offset)])
            for offset in np.deg2rad((0, 20, -20, 40, -40, 70, -70))
        ]
        yaw_offsets = np.deg2rad((0, 10, -10, 20, -20))

        for direction in directions:
            right = np.array([-direction[1], direction[0]], dtype=float)
            for distance in self.ROBOT_STAND_DISTANCES:
                for lateral in self.ROBOT_LATERAL_OFFSETS:
                    xy = (
                        target[:2]
                        + direction * float(distance)
                        + right * float(lateral)
                    )
                    if not self.robot_footprint_is_on_floor(xy, radius=0.25):
                        continue
                    for yaw_offset in yaw_offsets:
                        yaw = float(
                            np.arctan2(
                                target[1] - xy[1],
                                target[0] - xy[0],
                            )
                        ) + float(yaw_offset)
                        self._clear_robot_height_cache()
                        if not self._set_robot_free_pose(xy, yaw):
                            continue
                        if not self._calibrate_robot_free_joint_height():
                            continue
                        if self.count_body_collisions() != 0:
                            continue
                        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
                        self.robot_base_ori = np.array([0.0, 0.0, yaw])
                        return True
        return False

    def _clear_robot_height_cache(self):
        if hasattr(self, "_robot_free_joint_z"):
            delattr(self, "_robot_free_joint_z")

    def _source_contacts_fixture(self, fixture):
        object_bodies = self._body_tree_ids([self.obj_body_id[self.SOURCE_NAME]])
        fixture_bodies = self._fixture_body_ids(fixture, include_siblings=True)
        object_geoms = self._geom_names_for_bodies(object_bodies)
        fixture_geoms = self._geom_names_for_bodies(fixture_bodies)
        return bool(
            object_geoms
            and fixture_geoms
            and self.check_contact(object_geoms, fixture_geoms)
        )

    def _reset_internal(self):
        super()._reset_internal()
        spec = self.scene_task_spec()
        initial_xy = None
        if spec["initial_rim_strategy"] == "edge":
            initial_xy, _ = self.support_region_edge_pose(
                self.rim_region,
                self._object_half_xy(self.SOURCE_NAME),
                self.robot_base_pos[:2],
                margin=0.01,
                edge_fraction=float(spec["initial_rim_edge_fraction"]),
                sim=self.sim,
            )
        placed = self.place_object_on_support_contact(
            self.SOURCE_NAME,
            self.rim_region,
            fixture=self.rim_fixture,
            xy=initial_xy,
        )
        source_on_rim = self.object_on_region(
            self.SOURCE_NAME,
            self.rim_region,
            z_tol=0.12,
            require_contact=False,
        )
        if not (
            placed
            and source_on_rim
            and self._source_contacts_fixture(self.bathtub_fixture)
        ):
            raise RuntimeError(
                f"{self.scene_name}: source object is not physically on the "
                "named bathtub rim after reset"
            )
        self._reset_task_progress()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["removed_scene_fixtures"] = list(self.rim_clutter_fixture_ids)
        return ep_meta


class _BathroomSpongeInBathtubTask(_BathroomNorthRimTask):
    """Pick up a bathroom sponge from the rim and release it in the tub."""

    ATOMIC_SKILL = "PickPlaceIntoBathtub"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "pick_place"
    SPEC_TASK = "put_sponge_in_bathtub"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK,
        required=(
            "bathtub_fixture",
            "rim_fixture",
            "rim_region",
            "remove_fixtures",
            "rim_anchor_xy",
            "basin_center_xy",
            "basin_half_xy",
            "basin_z_range",
            "robot_pose",
            "source_horizontal_distance",
            "initial_rim_z_offset",
            "initial_rim_strategy",
            "initial_rim_edge_fraction",
            "lang",
        ),
    )
    SOURCE_NAME = "sponge"
    SOURCE_SCALE = SPONGE_SCALE
    INITIAL_POSE_JITTER = False
    BASIN_XY_TOLERANCE = 0.03
    BASIN_Z_TOLERANCE = 0.03
    PENETRATION_TOLERANCE = 0.004
    # FloorPlan405 contains two identical bathtub collision sets. Its basin
    # bottom also sits below the scene floor, so a dropped sponge contacts the
    # floor instead of the named bathtub. Keep this correction task-local.
    _405_DUPLICATE_BODY = "bathtubmesh_c13235ec5be7c4a3ad184fbc5209539d_1_1_0"
    _405_BASIN_FLOOR_GEOM = "bathtub_130810c2273407e7e112a6cb5304788c_1_1_0_collision_5"
    # FloorPlan407's porcelain-clad side partition has an oversized collider
    # that blocks the calibrated bathtub-rim stance despite being visually open.
    # Keep the correction local to this task and Scene.
    _407_BATHTUB_SIDE_WALL_GEOM = (
        "fp407:polysurface346_f461c4359192d24fd7e1c6d239af7575_1_1_0_collision_9"
    )

    def _setup_model(self):
        super()._setup_model()
        world = self.mujoco_arena.worldbody
        if self.scene_name == "FloorPlan407":
            side_wall = world.find(
                f".//geom[@name='{self._407_BATHTUB_SIDE_WALL_GEOM}']"
            )
            if side_wall is None:
                raise RuntimeError(
                    "FloorPlan407: bathtub-side collision calibration is missing"
                )
            side_wall.set("contype", "0")
            side_wall.set("conaffinity", "0")
            return
        if self.scene_name != "FloorPlan405":
            return
        duplicate = world.find(f".//body[@name='{self._405_DUPLICATE_BODY}']")
        floor = world.find(f".//geom[@name='{self._405_BASIN_FLOOR_GEOM}']")
        if duplicate is None or floor is None:
            raise RuntimeError("FloorPlan405: bathtub collision calibration is missing")
        for geom in duplicate.iter("geom"):
            if "collision" in geom.get("name", ""):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
        pos = np.fromstring(floor.get("pos"), sep=" ")
        # Local Y is world Z. Raise the bottom above the global floor before
        # compilation so MuJoCo also updates its broad-phase collision AABB.
        pos[1] += 0.05
        floor.set("pos", " ".join(str(value) for value in pos))

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.basin_center_xy = np.asarray(spec["basin_center_xy"], dtype=float)
        self.basin_half_xy = np.asarray(spec["basin_half_xy"], dtype=float)
        self.basin_z_range = np.asarray(spec["basin_z_range"], dtype=float)
        if (
            self.basin_center_xy.shape != (2,)
            or self.basin_half_xy.shape != (2,)
            or self.basin_z_range.shape != (2,)
            or np.any(self.basin_half_xy <= 0.0)
            or not self.basin_z_range[0] < self.basin_z_range[1]
        ):
            raise ValueError(f"{self.scene_name}: invalid bathtub basin bounds")
        pose = spec["robot_pose"]
        self.set_robot_base_pose(
            [float(pose["xy"][0]), float(pose["xy"][1]), 0.0],
            float(pose["yaw"]),
        )

    def _get_obj_cfgs(self):
        if self.rim_region is None:
            return []
        spec = self.scene_task_spec()
        return [
            dict(
                name=self.SOURCE_NAME,
                obj_groups=asset_xml_path(SPONGE_MODEL),
                object_scale=self.SOURCE_SCALE,
                placement=dict(
                    region=self.rim_region,
                    margin=0.01,
                    rotation=0.0,
                    strategy=spec["initial_rim_strategy"],
                    edge_fraction=float(spec["initial_rim_edge_fraction"]),
                    z_offset=float(spec["initial_rim_z_offset"]),
                ),
            )
        ]

    def _place_robot_at_north_rim(self):
        pose = self.scene_task_spec()["robot_pose"]
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        self._clear_robot_height_cache()
        if not (
            xy.shape == (2,)
            and self.robot_footprint_is_on_floor(xy, radius=0.25)
            and self._set_robot_free_pose(xy, yaw)
            and self._calibrate_robot_free_joint_height()
            and self.count_body_collisions() == 0
        ):
            return False
        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        return True

    def _source_is_released(self):
        return not any(
            self.g1_hand_object_contact(side, self.SOURCE_NAME)
            for side in ("left", "right")
        )

    def _source_has_illegal_penetration(self):
        source_root = int(self.obj_body_id[self.SOURCE_NAME])
        bathtub_roots = self._fixture_body_ids(
            self.bathtub_fixture,
            include_siblings=True,
        )
        return bool(
            self.illegal_contact_pairs_for_bodies(
                (source_root,),
                tuple(bathtub_roots),
                tolerance=self.PENETRATION_TOLERANCE,
            )
        )

    def _source_has_shallow_bathtub_contact(self):
        source_bodies = self._body_tree_ids([int(self.obj_body_id[self.SOURCE_NAME])])
        bathtub_bodies = self._fixture_body_ids(
            self.bathtub_fixture,
            include_siblings=True,
        )
        distances = []
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body_1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body_2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body_1 in source_bodies and body_2 in bathtub_bodies) or (
                body_2 in source_bodies and body_1 in bathtub_bodies
            ):
                distances.append(float(contact.dist))
        return bool(distances and min(distances) >= -self.PENETRATION_TOLERANCE)

    def _source_is_in_bathtub_and_released(self):
        points = self.object_bbox_points(self.SOURCE_NAME)
        bbox_center_xy = 0.5 * (
            np.min(points[:, :2], axis=0) + np.max(points[:, :2], axis=0)
        )
        # 405's narrow near edge must contain the *whole* sponge, not just
        # its center, or a sponge hung over the rim would count as inside.
        xy = (
            points[:, :2]
            if getattr(self, "scene_name", None) == "FloorPlan405"
            else bbox_center_xy
        )
        center_inside = np.all(
            np.abs(xy - self.basin_center_xy)
            <= self.basin_half_xy + self.BASIN_XY_TOLERANCE
        )
        vertically_inside = bool(
            float(np.min(points[:, 2]))
            >= self.basin_z_range[0] - self.BASIN_Z_TOLERANCE
            and float(np.max(points[:, 2]))
            <= self.basin_z_range[1] + self.BASIN_Z_TOLERANCE
        )
        return bool(
            center_inside
            and vertically_inside
            and self._source_has_shallow_bathtub_contact()
            and not self._source_has_illegal_penetration()
            and self._source_is_released()
        )

    def _raw_success(self):
        return self._source_is_in_bathtub_and_released()

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return ret

    def _check_success(self):
        return self._success_hold_steps >= self.HOLD_STEPS

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        spec = self.scene_task_spec()
        ep_meta.update(
            {
                "lang": spec["lang"],
                "source_grasp_span": 0.075,
                "source_horizontal_distance": float(spec["source_horizontal_distance"]),
                "success_definition": (
                    "the sponge bounding-box center must be inside the "
                    "calibrated bathtub basin, below the rim, in physical "
                    "contact with the bathtub, free of unrelated penetration, "
                    f"and released for {self.HOLD_STEPS} control steps"
                ),
            }
        )
        return ep_meta


class IThorBathroomPutSpongeInBathtub(
    _BathroomSpongeInBathtubTask,
    IThorBathroomEnv,
):
    """Pick up a bathroom sponge from the rim and release it in the tub."""


__all__ = [
    "IThorBathroomPutSpongeInBathtub",
]
