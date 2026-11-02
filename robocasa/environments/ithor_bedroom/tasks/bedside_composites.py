"""Ordered, task-local routines for the native objects in FloorPlan301."""

from copy import deepcopy
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.utils.mjcf_utils import get_elements, scale_mjcf_model

from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.environments.ithor_composite.progress import CompositeStageReporterMixin
from robocasa.models.scenes.thor_semantics import SemanticJoint

from .assets import asset_xml_path
from .bedside_objects import (
    _BedsideObjectStateMixin,
    _HeadboardPillowMixin,
    _SofaStatueMixin,
    _fixed_support,
    IThorBedroomLiftStatueFromSofa as StatueTask,
    IThorBedroomStandPillowAtHeadboard as PillowTask,
)


class _BedsideSequenceMixin(CompositeStageReporterMixin):
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L2"
    INTERMEDIATE_HOLD_STEPS = 5

    def _reset_internal(self):
        super()._reset_internal()
        if self.count_body_collisions() or not self.robot_footprint_is_on_floor(
            self._robot_base_xy(), radius=0.25
        ):
            raise RuntimeError(f"{self.scene_name}: unsafe composite start")
        self._reset_sequence()

    def _reset_sequence(self):
        self._sequence_stage = 0
        self._sequence_hold = 0
        self._sequence_success = False
        self._stage_armed = True

    def _post_action(self, action):
        valid = self._sequence_goal(self._sequence_stage)
        if not valid:
            self._stage_armed = True
        self._sequence_hold = (
            self._sequence_hold + 1 if valid and self._stage_armed else 0
        )
        final = self._sequence_stage == len(self.COMPOSITE_ORDERING) - 1
        required = self.HOLD_STEPS if final else self.INTERMEDIATE_HOLD_STEPS
        was_success = self._sequence_success
        self._sequence_success = bool(final and self._sequence_hold >= required)
        if self._sequence_success and not was_success:
            self._report_stage_complete(3, 3)
        elif not final and self._sequence_hold >= required:
            self._sequence_stage += 1
            self._sequence_hold = 0
            # A target already completed before its stage must be undone first.
            self._stage_armed = not self._sequence_goal(self._sequence_stage)
            self._report_stage_complete(
                self._sequence_stage,
                3,
                self.COMPOSITE_ORDERING[self._sequence_stage],
            )
        # The reused native-object initializer has an atomic hold counter; only
        # this sequence updates progress, before reward/done queries success.
        return IThorBedroomEnv._post_action(self, action)

    def _check_success(self):
        return bool(getattr(self, "_sequence_success", False))

    def _near_fixture(self, fixture):
        distance = self.robot_fixture_collision_distance(fixture, max_distance=2.0)
        return bool(np.isfinite(distance) and distance <= 0.35)

    def _hands_touch_fixture(self, fixture):
        roots = [
            self.sim.model.body_name2id(f"robot0_{side}_wrist_yaw_link")
            for side in ("left", "right")
        ]
        return bool(
            self.check_contact(
                self._geom_names_for_bodies(self._body_tree_ids(roots)),
                self._geom_names_for_bodies(self._fixture_body_ids(fixture)),
            )
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            composite_stages=list(self.COMPOSITE_ORDERING),
            composite_ordering=list(self.COMPOSITE_ORDERING),
            composite_stage=getattr(self, "_sequence_stage", 0),
            intermediate_hold_steps=self.INTERMEDIATE_HOLD_STEPS,
            success_hold_seconds=1.0,
        )
        return meta


class IThorBedroomApproachBedStandPillowThenDesk(
    _BedsideSequenceMixin,
    _HeadboardPillowMixin,
    _BedsideObjectStateMixin,
    IThorBedroomEnv,
):
    """Approach the bed, stand its pillow at the headboard, then reach the desk."""

    TARGET_STABLE_ID = PillowTask.TARGET_STABLE_ID
    BED_STABLE_ID = PillowTask.BED_STABLE_ID
    BED_BODY = PillowTask.BED_BODY
    FACE_NORMAL_AXIS = PillowTask.FACE_NORMAL_AXIS
    MAX_LEAN_DEGREES = PillowTask.MAX_LEAN_DEGREES
    DESK_STABLE_ID = "FloorPlan301:Desk_301_1_21"
    COMPOSITE_ORDERING = ("approach_bed", "stand_pillow_at_headboard", "approach_desk")
    SCENE_SPECS = deepcopy(PillowTask.SCENE_SPECS)
    SCENE_SPECS["FloorPlan301"].update(
        robot_pose={"xy": [1.3, 0.9], "yaw": np.pi},
        minimum_start_distance=0.50,
        desk=DESK_STABLE_ID,
        stage_order=COMPOSITE_ORDERING,
        lang=(
            "Walk to the bed, stand the flat pillow upright against the "
            "headboard, then walk to the desk."
        ),
    )
    _target_state_reached = PillowTask._target_state_reached

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.source_fixture = self.get_fixture(id=self.BED_STABLE_ID)
        self.desk_fixture = self.get_fixture(id=self.DESK_STABLE_ID)
        if self.source_fixture is None or self.desk_fixture is None:
            raise LookupError(f"{self.scene_name}: missing bed or desk")
        self.source_region = _fixed_support(
            self.source_fixture,
            "mattress",
            [-0.63502, 0.87339, 0.56464],
            [0.7019, 1.0164],
        )
        self.fixture_refs.update(bed=self.source_fixture, desk=self.desk_fixture)
        self.region_refs["source_surface"] = self.source_region

    def _reset_internal(self):
        # These IDs are needed by the sequence after the native-object reset.
        super()._reset_internal()
        self._headboard_geom = self.sim.model.geom_name2id(
            f"{self.BED_BODY}_collision_1"
        )
        self._mattress_geom = self.sim.model.geom_name2id(
            f"{self.BED_BODY}_collision_2"
        )
        self._pillow_bodies = self._fixture_body_ids(self.target_fixture)
        self._bed_arrival_xy = None
        self._bed_start_distance = self.robot_fixture_collision_distance(
            self.source_fixture, max_distance=5.0
        )
        if self._bed_start_distance < self.scene_task_spec()["minimum_start_distance"]:
            raise RuntimeError(f"{self.scene_name}: composite starts too near bed")

    def _sequence_goal(self, stage):
        upright = self._target_state_reached()
        if stage == 0:
            valid = self._near_fixture(self.source_fixture) and not upright
            if valid:
                self._bed_arrival_xy = self._robot_base_xy().copy()
            return valid
        if stage == 1:
            return self._near_fixture(self.source_fixture) and upright
        return bool(
            upright
            and self._bed_arrival_xy is not None
            and np.linalg.norm(self._robot_base_xy() - self._bed_arrival_xy) >= 0.60
            and self._near_fixture(self.desk_fixture)
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            pillow_scale=list(self.PILLOW_SCALE),
            desk_fixture_id=self.DESK_STABLE_ID,
            start_distance=self._bed_start_distance,
            success_distance=0.35,
            minimum_stage_travel=0.60,
            success_definition=(
                "Reach bed before reorienting pillow; pillow remains "
                "upright touching mattress and headboard while "
                "robot stays by desk for 1.0 s."
            ),
        )
        return meta


class IThorBedroomOpenBoxThenStatueFromSofaIntoBox(
    _BedsideSequenceMixin,
    _SofaStatueMixin,
    _BedsideObjectStateMixin,
    IThorBedroomEnv,
):
    """Open the floor box before lifting the sofa's statue and placing it inside."""

    TARGET_STABLE_ID = StatueTask.TARGET_STABLE_ID
    SOFA_STABLE_ID = StatueTask.SOFA_STABLE_ID
    MIN_LIFT_DELTA = StatueTask.MIN_LIFT_DELTA
    BOX_MODEL = "/objects/thor/Living Room and Bedroom Objects/Box_Full/Prefabs/Box_10/Box_10.xml"
    BOX_SCALE = 1.6
    BOX_NAME = "task_statue_box"
    COMPOSITE_ORDERING = (
        "open_box",
        "lift_statue_from_sofa",
        "put_statue_in_box",
    )
    SCENE_SPECS = deepcopy(StatueTask.SCENE_SPECS)
    SCENE_SPECS["FloorPlan301"].update(
        robot_pose={"xy": [1.2, -0.3], "yaw": np.arctan2(-0.20, 0.92)},
        box="FloorPlan301:spawned:task_statue_box",
        box_xy=[1.85, -0.50],
        box_scale=BOX_SCALE,
        stage_order=COMPOSITE_ORDERING,
        lang=(
            "Open the cardboard box, pick up the statue from the sofa, "
            "and put the statue inside the box."
        ),
    )
    _target_state_reached = StatueTask._target_state_reached

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.source_fixture = self.get_fixture(id=self.SOFA_STABLE_ID)
        if self.source_fixture is None:
            raise LookupError(f"{self.scene_name}: missing sofa")
        self.source_region = _fixed_support(
            self.source_fixture, "seat", [0.93, -0.85, 0.46379], [0.37, 0.34]
        )
        self.fixture_refs["sofa"] = self.source_fixture
        self.region_refs["source_surface"] = self.source_region
        original = self.get_fixture(id="FloorPlan301:Box_1_1")
        if original is None:
            raise LookupError(f"{self.scene_name}: missing original small box")
        self.remove_semantic_fixtures([original])
        self._create_statue_box()

    def _create_statue_box(self):
        # A distinct large-box identity; source MJCF and all old tasks stay intact.
        source = Path(asset_xml_path(self.BOX_MODEL))
        root = ET.parse(source).getroot()
        asset = root.find("asset")
        for element in asset:
            if element.get("file"):
                element.set(
                    "file", str((source.parent / element.get("file")).resolve())
                )
        worldbody = root.find("worldbody")
        scale_mjcf_model(worldbody, asset, self.BOX_SCALE, get_elements, worldbody)
        body = worldbody.find("body")
        body.remove(body.find("joint"))  # Fixed floor box, movable flaps.
        for joint in body.iter("joint"):
            joint.set("frictionloss", "0.8")
            joint.set("damping", "0.15")
        for site in body.iter("site"):
            site.set("group", "5")
            site.set("rgba", "0 0 0 0")
        with tempfile.TemporaryDirectory(prefix="bedroom_301_statue_box_") as directory:
            path = Path(directory) / "box.xml"
            ET.ElementTree(root).write(path)
            self.box_fixture = self.create_scene_fixture(
                xml_path=str(path),
                name=self.BOX_NAME,
                pos=[*self.scene_task_spec()["box_xy"], 0.31],
                quat=[np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
                category="Box",
                source_category="Box",
                asset_id="StatueBox_1p6",
            )
        self.box_fixture.source = {"xml_path": str(source), "scale": self.BOX_SCALE}
        box_root = self.find_scene_body(self.box_fixture.body_name)
        self.box_flap_joint = f"{self.BOX_NAME}_Box_10_box_10_flap_outer_1_joint_0"
        self.box_floor_geom = f"{self.BOX_NAME}_Box_10_box_10_PrimitiveCollider_0"
        self.box_site_name = f"{self.BOX_NAME}_Box_10_box_10_ReceptacleCollider_0"
        self.box_fixture.joints.clear()
        for joint in box_root.iter("joint"):
            name = joint.get("name")
            low, high = (float(value) for value in joint.get("range").split())
            self.box_fixture.joints[name] = SemanticJoint(
                stable_id=f"{self.box_fixture.stable_id}:joint:{name}",
                name=name,
                role="flap",
                range=(low, high),
                closed_qpos=low if abs(low) > abs(high) else high,
                open_qpos=0.0,
            )
        if self.box_flap_joint not in self.box_fixture.joints:
            raise LookupError("missing calibrated box flap")
        self.fixture_refs["box"] = self.box_fixture

    def _setup_scene(self):
        super()._setup_scene()
        statue_bodies = self._fixture_body_ids(self.target_fixture)
        self._statue_geom_ids = [
            geom
            for geom in range(self.sim.model.ngeom)
            if int(self.sim.model.geom_bodyid[geom]) in statue_bodies
            and self.sim.model.geom_contype[geom]
        ]
        self._box_floor_id = self.sim.model.geom_name2id(self.box_floor_geom)
        self._box_site_id = self.sim.model.site_name2id(self.box_site_name)
        floor = self._box_floor_id
        rotation = self.sim.data.geom_xmat[floor].reshape(3, 3)
        bottom = self.sim.data.geom_xpos[floor, 2] - np.dot(
            np.abs(rotation[2]), self.sim.model.geom_size[floor]
        )
        root_id = self.sim.model.body_name2id(self.box_fixture.body_name)
        self.sim.model.body_pos[root_id, 2] -= bottom
        # Other flaps start open. The named flap starts upright, like the
        # existing bedroom opening primitive; friction prevents gravity credit.
        for joint in self.box_fixture.joints.values():
            self.sim.data.set_joint_qpos(joint.name, 0.0)
        self.sim.data.set_joint_qpos(self.box_flap_joint, -1.3843677664)
        self.sim.forward()

    def _box_open(self):
        return float(self.sim.data.get_joint_qpos(self.box_flap_joint)) >= -0.30

    def _statue_on_sofa(self):
        return bool(
            self._fixture_bbox_inside_region_xy(self.target_fixture, self.source_region)
            and self._fixture_contacts_fixture(self.target_fixture, self.source_fixture)
            and not self._target_state_reached()
        )

    def _statue_collision_bounds(self):
        # Stone_Statue_4 uses seven box colliders; transform every corner so
        # containment remains exact when the statue is tilted inside the box.
        signs = np.asarray(
            [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
        )
        points = []
        for geom in self._statue_geom_ids:
            rotation = self.sim.data.geom_xmat[geom].reshape(3, 3)
            corners = signs * self.sim.model.geom_size[geom]
            points.extend(self.sim.data.geom_xpos[geom] + corners @ rotation.T)
        return np.asarray(points)

    def _statue_inside_box(self):
        # Use the actual box site's frame and full statue bounds, not center-only.
        center = self.sim.data.site_xpos[self._box_site_id]
        rotation = self.sim.data.site_xmat[self._box_site_id].reshape(3, 3)
        world_points = self._statue_collision_bounds()
        points = (world_points - center) @ rotation
        half = self.sim.model.site_size[self._box_site_id]
        vertical_axis = int(np.argmax(np.abs(rotation[2])))
        horizontal_axes = [axis for axis in range(3) if axis != vertical_axis]
        if not np.all(
            np.abs(points[:, horizontal_axes]) <= half[horizontal_axes] - 0.005
        ):
            return False
        floor = self._box_floor_id
        floor_rotation = self.sim.data.geom_xmat[floor].reshape(3, 3)
        floor_top = self.sim.data.geom_xpos[floor, 2] + np.dot(
            np.abs(floor_rotation[2]), self.sim.model.geom_size[floor]
        )
        rim_z = center[2] + np.dot(np.abs(rotation[2]), half)
        if (
            world_points[:, 2].min() < floor_top - 0.003
            or world_points[:, 2].max() > rim_z - 0.005
        ):
            return False
        if self._hands_touch_fixture(self.target_fixture):
            return False
        if self._fixture_has_illegal_penetration(
            self.target_fixture, tuple(self.box_fixture.body_names)
        ):
            return False
        statue_bodies = self._fixture_body_ids(self.target_fixture)
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            for subject, support in (
                (contact.geom1, contact.geom2),
                (contact.geom2, contact.geom1),
            ):
                if (
                    int(support) == self._box_floor_id
                    and int(self.sim.model.geom_bodyid[subject]) in statue_bodies
                    and -self.CONTACT_PENETRATION_TOLERANCE <= contact.dist <= 0.003
                ):
                    return True
        return False

    def _sequence_goal(self, stage):
        if stage == 0:
            return self._box_open() and self._statue_on_sofa()
        if stage == 1:
            return self._box_open() and self._target_state_reached()
        return self._box_open() and self._statue_inside_box()

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            box_fixture_id=self.box_fixture.stable_id,
            box_model=self.BOX_MODEL,
            box_scale=self.BOX_SCALE,
            box_base_fixed=True,
            requires_release=True,
            box_flap_joint=self.box_flap_joint,
            removed_fixtures=[self.REMOVED_BASKETBALL_ID, "FloorPlan301:Box_1_1"],
            minimum_lift_delta=self.MIN_LIFT_DELTA,
            success_definition=(
                "Open the named flap while statue remains on sofa; "
                "then lift statue by 0.30 m; finally release it "
                "fully inside box, resting on its bottom for 1.0 s."
            ),
        )
        return meta


__all__ = [
    "IThorBedroomApproachBedStandPillowThenDesk",
    "IThorBedroomOpenBoxThenStatueFromSofaIntoBox",
]
