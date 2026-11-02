"""Scene-405 cabinet-to-counter routine with an isolated atomic layout."""

from copy import deepcopy
import xml.etree.ElementTree as ET

import numpy as np

from robocasa.environments.ithor_atomic.pick_place import resolve_task_support_region
from robocasa.environments.ithor_atomic.task_assets import asset_xml_path
from robocasa.environments.ithor_composite.progress import CompositeStageReporterMixin
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv

from ..cabinet_lift import (
    IThorBathroomLiftSoapBottleFromCabinet,
    _BathroomCabinetSoapTask,
)


class IThorBathroomOpenCabinetThenPlaceSoapBottleOnCounterAndCloseCabinet(
    CompositeStageReporterMixin, _BathroomCabinetSoapTask, IThorBathroomEnv
):
    """Open the low cabinet, lift its bottle onto the counter, then close it."""

    SUPPORTED_SCENE_IDS = (405,)
    ATOMIC_SKILL = "OrderedRoutine"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    COMPOSITE_KIND = "cabinet_soap_routine"
    COMPOSITE_ORDERING = (
        "open_cabinet_door",
        "bend_down_and_lift_soap_bottle",
        "place_soap_bottle_on_counter_and_release",
        "close_cabinet_door",
    )
    INTERMEDIATE_HOLD_STEPS = 5
    SUCCESS_OPEN_FRACTION = 0.8
    SUCCESS_CLOSE_FRACTION = 0.1
    SUPPORT_PLANE_TOLERANCE = 0.03
    SCENE_SPECS = {
        "FloorPlan405": deepcopy(
            IThorBathroomLiftSoapBottleFromCabinet.SCENE_SPECS["FloorPlan405"]
        )
    }
    SCENE_SPECS["FloorPlan405"].update(
        cabinet_open_fraction=0.2,
        # The atomic starts fully open. This stance also clears the door sweep.
        robot_pose={"xy": [-2.30, 0.30], "yaw": -2.50},
        counter_fixture="FloorPlan405:CabinetBodyMesh_45407f37_25",
        counter_region="FloorPlan405:CabinetBodyMesh_45407f37_25:region:receptacle:7",
        counter_top_geom=(
            "cabinetbodymesh_a4049078edb3f5805892102ffab9a4bf_1_1_0_collision_11"
        ),
        counter_underlay_geom=(
            "cabinetbodymesh_a4049078edb3f5805892102ffab9a4bf_1_1_0_collision_8"
        ),
        counter_collision_model="/objects/manual/bathroom_405_countertop_collision.xml",
        lang=(
            "Open the bathroom cabinet door, pick up the soap "
            "bottle inside, place it on the vanity top and release it, "
            "then close the cabinet door."
        ),
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.counter_fixture = self._exact_fixture(spec["counter_fixture"])
        if self.counter_fixture is None:
            raise LookupError(f"{self.scene_name}: missing soap placement counter")
        self.counter_region = resolve_task_support_region(
            self.counter_fixture.regions[spec["counter_region"]]
        )
        self.fixture_refs["target_counter"] = self.counter_fixture
        self.region_refs["target_counter"] = self.counter_region
        self._replace_countertop_collision()

    def _replace_countertop_collision(self):
        # Both original boxes bridge the sink opening, even below the top slab.
        # Replace only this environment's XML with prisms of the visible top.
        spec = self.scene_task_spec()
        body = self.find_scene_body(
            spec["counter_top_geom"].rsplit("_collision_", 1)[0]
        )
        for key in ("counter_top_geom", "counter_underlay_geom"):
            geom = body.find(f"./geom[@name='{spec[key]}']")
            body.remove(geom)
        collision = ET.parse(asset_xml_path(spec["counter_collision_model"])).getroot()
        self.mujoco_arena.asset.extend(collision.find("asset"))
        geoms = list(collision.find("worldbody/body"))
        self.counter_top_geom_names = tuple(geom.get("name") for geom in geoms)
        body.extend(geoms)

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_task_progress()

    def _reset_task_progress(self):
        # Fixed-scene collection invokes this after restoring initial physics.
        self._initial_source_z = self._source_root_z()
        self._success_hold = 0
        self._routine_stage = 0
        self._routine_hold = 0
        self._routine_armed = not self._stage_goal(0)
        self._routine_reported_success = False

    def _bottle_is_on_surface(self, fixture, region):
        support_bodies = self._fixture_body_ids(fixture, include_siblings=True)
        distances = self._fixture_contact_distances(self.source_fixture, support_bodies)
        support_z = float(region.world_pose(self.sim)[0][2])
        bottom_z = float(np.min(self._fixture_bbox_points(self.source_fixture)[:, 2]))
        return bool(
            self._fixture_bbox_inside_region_xy(self.source_fixture, region)
            and abs(bottom_z - support_z) <= self.SUPPORT_PLANE_TOLERANCE
            and distances
            and min(distances) >= -self.CONTACT_PENETRATION_TOLERANCE
            and not self._fixture_has_illegal_penetration(
                self.source_fixture, fixture.body_names
            )
            and self._fixture_released(self.source_fixture)
        )

    def _bottle_is_on_counter(self):
        if not self._fixture_released(self.source_fixture):
            return False
        # The countertop prisms include narrow strips and leave the basin open.
        # Classify its top contacts without restricting the bottle's footprint.
        top_geoms = {
            self.sim.model.geom_name2id(name) for name in self.counter_top_geom_names
        }
        source_bodies = self._fixture_body_ids(
            self.source_fixture, include_siblings=True
        )
        support_z = float(self.counter_region.world_pose(self.sim)[0][2])
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            if int(contact.geom1) in top_geoms:
                other_geom = int(contact.geom2)
            elif int(contact.geom2) in top_geoms:
                other_geom = int(contact.geom1)
            else:
                continue
            if (
                int(self.sim.model.geom_bodyid[other_geom]) in source_bodies
                and abs(float(contact.pos[2]) - support_z)
                <= self.SUPPORT_PLANE_TOLERANCE
            ):
                return True
        return False

    def _stage_goal(self, index):
        fraction = self.cabinet_joint.fraction(self.sim)
        if index == 0:
            return bool(
                fraction >= self.SUCCESS_OPEN_FRACTION
                and self._bottle_is_on_surface(
                    self.work_surface_fixture, self.work_surface_region
                )
            )
        if index == 1:
            return bool(
                fraction >= self.SUCCESS_OPEN_FRACTION
                and _BathroomCabinetSoapTask._raw_success(self)
                and not self._fixture_has_illegal_penetration(self.source_fixture, ())
            )
        if index == 2:
            return self._bottle_is_on_counter()
        return self._raw_success()

    def _raw_success(self):
        return bool(
            self.cabinet_joint.fraction(self.sim) <= self.SUCCESS_CLOSE_FRACTION
            and self._bottle_is_on_counter()
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._advance_cabinet_routine()
        return ret

    def _advance_cabinet_routine(self):
        goal = self._stage_goal(self._routine_stage)
        if not goal:
            self._routine_armed = True
        self._routine_hold = (
            self._routine_hold + 1 if goal and self._routine_armed else 0
        )
        final_stage = len(self.COMPOSITE_ORDERING) - 1
        required = (
            self.HOLD_STEPS
            if self._routine_stage in (2, final_stage)
            else self.INTERMEDIATE_HOLD_STEPS
        )
        if self._routine_hold < required:
            return
        if self._routine_stage < final_stage:
            self._routine_stage += 1
            self._routine_hold = 0
            self._routine_armed = self._routine_stage == 2 or not self._stage_goal(
                self._routine_stage
            )
            self._report_stage_complete(
                self._routine_stage,
                len(self.COMPOSITE_ORDERING),
                next_stage=self.COMPOSITE_ORDERING[self._routine_stage],
            )
        elif not self._routine_reported_success:
            self._routine_reported_success = True
            self._report_stage_complete(
                len(self.COMPOSITE_ORDERING), len(self.COMPOSITE_ORDERING)
            )

    def _check_success(self):
        return bool(
            self._routine_stage == len(self.COMPOSITE_ORDERING) - 1
            and self._routine_hold >= self.HOLD_STEPS
            and self._raw_success()
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            stage_order=list(self.COMPOSITE_ORDERING),
            intermediate_hold_steps=self.INTERMEDIATE_HOLD_STEPS,
            placement_hold_steps=self.HOLD_STEPS,
            placement_hold_seconds=1.0,
            final_hold_steps=self.HOLD_STEPS,
            placement_surface=self.counter_fixture.stable_id,
            placement_contact_geoms=list(self.counter_top_geom_names),
            cabinet_success_open_fraction=self.SUCCESS_OPEN_FRACTION,
            cabinet_success_close_fraction=self.SUCCESS_CLOSE_FRACTION,
            success_definition=(
                "in order: open the cabinet to at least 80% while the bottle "
                "remains on its low support; lift the bottle at least 0.30 m "
                "above its reset height, independent of grasp or hand state; place "
                "the bottle in contact with the vanity top, including its narrow "
                "strips beside the sink, with neither hand touching it for 1.0 s; close "
                "the same door to at most 10% and keep the released bottle "
                "in contact with the counter for 1.0 s, excluding the sink basin"
            ),
        )
        return meta


TASK_CLASSES = (IThorBathroomOpenCabinetThenPlaceSoapBottleOnCounterAndCloseCabinet,)

__all__ = [cls.__name__ for cls in TASK_CLASSES]
