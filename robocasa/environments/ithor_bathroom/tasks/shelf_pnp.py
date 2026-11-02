"""Move a canonical bar of soap anywhere on the lowest bathroom wall shelf."""

import numpy as np

from robocasa.environments.ithor_atomic.pick_place import (
    RoomMultiLevelFixturePnP,
    SupportContactPlacementMixin,
)
from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_atomic.task_assets import (
    SOAP_MODEL,
    SOAP_SCALE,
    asset_xml_path,
)
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_specs,
    get_supported_scene_ids,
)


class _BathroomShelfPnPMixin(
    SceneSpecMixin,
    SupportContactPlacementMixin,
    RoomMultiLevelFixturePnP,
):
    """Physical-level shelf transfer logic shared by final Bathroom tasks."""

    TARGET_CATEGORY = "WallShelf_81fbd644"
    TARGET_ASSET_ID = "WallShelf_81fbd644"
    STAND_DISTANCE = 0.22
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    SOURCE_PLACEMENT_MARGIN = 0.005
    SOURCE_PLACEMENT_STRATEGY = "center"
    # Keep placement and success checks inside the physical receptacles.
    SOURCE_REGION_XY_SCALE = 1.0
    TARGET_REGION_XY_SCALE = 1.0
    SOURCE_PLACEMENT_NEAR_ROBOT_HALF = False
    TARGET_LEVEL_Z_TOLERANCE = 0.08

    def _get_obj_cfgs(self):
        cfgs = super()._get_obj_cfgs()
        cfgs[0]["placement"]["z_offset"] = 0.0
        return cfgs

    def _setup_task_fixtures(self):
        spec = self.scene_task_spec()
        # Skip RoomMultiLevelFixturePnP's category-first fixture discovery:
        # Bathroom task specs bind all three semantic entities exactly.
        super(RoomMultiLevelFixturePnP, self)._setup_task_fixtures()
        fixture = self.get_fixture(id=spec["fixture"])
        self.multilevel_fixture = (
            fixture
            if fixture is not None and fixture.stable_id == spec["fixture"]
            else None
        )
        self.source_region = None
        self.target_region = None
        if self.multilevel_fixture is None:
            return
        regions = {
            region.stable_id: region
            for region in self.multilevel_fixture.regions.values()
            if region.kind == "support"
        }
        self.source_region = regions.get(spec["source_region"])
        self.target_region = regions.get(spec["target_region"])
        if self.source_region is None or self.target_region is None:
            self.source_region = None
            self.target_region = None
            return
        self.fixture_refs["multilevel_fixture"] = self.multilevel_fixture
        self.region_refs["source_shelf"] = self.source_region
        self.region_refs["target_shelf"] = self.target_region
        center, yaw = self.source_region.world_pose()
        # Stand perpendicular to the shelf's long edge. The generic fixture
        # front comes from the full diagonal mesh, not the shelf levels.
        if self.source_region.half_size[0] >= self.source_region.half_size[1]:
            front = np.array([-np.sin(yaw), np.cos(yaw)])
        else:
            front = np.array([np.cos(yaw), np.sin(yaw)])
        if np.dot(front, self.multilevel_fixture.front) < 0.0:
            front = -front
        distance = self.multilevel_fixture.radial_extent(front) + self.STAND_DISTANCE
        pos = np.array(
            [center[0] + front[0] * distance, center[1] + front[1] * distance, 0.0]
        )
        robot_yaw = np.arctan2(center[1] - pos[1], center[0] - pos[0])
        self.set_robot_base_pose(pos, robot_yaw)

    def _object_contacts_shelf(self):
        object_bodies = self._body_tree_ids([self.obj_body_id[self.SOURCE_NAME]])
        shelf_bodies = self._fixture_body_ids(
            self.multilevel_fixture, include_siblings=True
        )
        object_geoms = self._geom_names_for_bodies(object_bodies)
        shelf_geoms = self._geom_names_for_bodies(shelf_bodies)
        return bool(
            object_geoms
            and shelf_geoms
            and self.check_contact(object_geoms, shelf_geoms)
        )

    def _object_is_contained_by(self, region):
        if float(region.half_size[2]) > 0.05:
            return self.object_in_region(self.SOURCE_NAME, region)
        return self.object_on_region(
            self.SOURCE_NAME, region, require_contact=False
        )

    def _object_center_is_on_target_level(self):
        """Accept the complete physical shelf level, including edge poses.

        The exported support patch is slightly smaller than the collidable
        lowest shelf. Requiring every object bounding-box corner to remain in
        that patch rejects stable placements near an edge. Use the bounding-box
        center for XY membership, with the audited per-axis physical overhang,
        while retaining a tight bottom-height check for this particular level.
        """

        center, local, bottom, _ = self._object_region_bounds(
            self.SOURCE_NAME, self.target_region
        )
        xy_tolerance = np.asarray(
            self.scene_task_spec()["target_center_xy_tolerance"], dtype=float
        )
        if xy_tolerance.shape != (2,) or np.any(xy_tolerance < 0.0):
            raise ValueError(
                "target_center_xy_tolerance must contain two non-negative values"
            )
        local_center = np.mean(local, axis=0)
        return bool(
            np.all(
                np.abs(local_center)
                <= self.target_region.half_size[:2] + xy_tolerance + 1e-4
            )
            and abs(bottom - float(center[2])) <= self.TARGET_LEVEL_Z_TOLERANCE
        )

    def _reset_internal(self):
        super()._reset_internal()
        placed = self.place_object_on_support_contact(
            self.SOURCE_NAME,
            self.source_region,
            fixture=self.multilevel_fixture,
        )
        if not (
            placed
            and self._object_is_contained_by(self.source_region)
            and self._object_contacts_shelf()
        ):
            raise RuntimeError(
                f"{self.scene_name}: source object is not physically on the "
                "named source shelf after reset"
            )
        self._success_hold_steps = 0

    def _object_is_on_target_shelf_and_released(self):
        return bool(
            self._object_center_is_on_target_level()
            and self._object_contacts_shelf()
            and self.object_released(
                self.SOURCE_NAME, th=self.RELEASE_DISTANCE
            )
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.scene_task_spec()["lang"].format(
            source=self.SOURCE_LANG
        )
        ep_meta["success_definition"] = (
            "the released object's bounding-box center and bottom height must "
            "belong to the complete physical lowest shelf, including its "
            f"audited edge tolerance, and remain in physical contact for "
            f"{self.HOLD_STEPS} control steps"
        )
        return ep_meta


class IThorBathroomSoapToLowerShelf(_BathroomShelfPnPMixin, IThorBathroomEnv):
    """Place a bar of soap from the second-lowest shelf to the lowest."""

    SPEC_TASK = "soap_to_lower_shelf"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK,
        required=(
            "fixture",
            "source_region",
            "target_region",
            "target_center_xy_tolerance",
            "lang",
        ),
    )
    SOURCE_NAME = "bar_soap"
    SOURCE_GROUP = asset_xml_path(SOAP_MODEL)
    SOURCE_LANG = "bar of soap"
    SOURCE_SCALE = SOAP_SCALE
    SOURCE_ROTATION = 0.0


__all__ = ["IThorBathroomSoapToLowerShelf"]
