"""Scene 301 object tasks with state-only, one-second success."""

from copy import deepcopy
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.utils.mjcf_utils import scale_body_element, scale_mesh_element

from robocasa.environments.ithor_atomic.free_fixture import FreeFixturePlacementMixin
from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.environments.ithor_bedroom.tasks.assets import asset_xml_path
from robocasa.environments.ithor_kitchen.tasks.atomic.objects_into_box import (
    LivingRoomStatueObject,
)
from robocasa.models.scenes.ithor_scene_defaults import BEDROOM_301_SOFA_ID
from robocasa.models.scenes.thor_semantics import SemanticRegion


class _BedsideObjectStateMixin(FreeFixturePlacementMixin, SceneSpecMixin):
    """Place a native object on an exact support; count control steps once."""

    SUPPORTED_SCENE_IDS = (301,)
    INITIAL_POSE_JITTER = False
    DIFFICULTY_LEVEL = "L1"

    def _get_scene_robot_pose_calibration(self):
        pose = self.scene_task_spec()["robot_pose"]
        return np.array([*pose["xy"], 0.0]), float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.target_fixture = self.get_fixture(id=self.TARGET_STABLE_ID)
        if self.target_fixture is None:
            raise LookupError(f"{self.scene_name}: missing {self.TARGET_STABLE_ID}")
        self.fixture_refs["target_object"] = self.target_fixture

    def _setup_scene(self):
        super()._setup_scene()
        spec = self.scene_task_spec()
        self._set_fixture_quaternion(self.target_fixture, spec["source_quat"])
        if not self._place_fixture_on_support(
            self.target_fixture,
            self.source_region,
            self.source_fixture,
            spec["source_xy"],
            contract_name="bedside_source",
            probe_step=0.001,
        ):
            raise RuntimeError(f"{self.scene_name}: target has no valid source contact")

    def _reset_internal(self):
        super()._reset_internal()
        self._state_hold_steps = 0
        center = self.target_fixture.world_center(self.sim)
        self._initial_target_z = float(center[2])
        self._source_horizontal_distance = float(
            np.linalg.norm(center[:2] - self._robot_base_xy())
        )

    def _post_action(self, action):
        self._state_hold_steps = (
            self._state_hold_steps + 1 if self._target_state_reached() else 0
        )
        return super()._post_action(action)

    def _check_success(self):
        return self._state_hold_steps >= self.HOLD_STEPS

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=self.scene_task_spec()["lang"],
            target_fixture_id=self.TARGET_STABLE_ID,
            source_surface=self.source_region.stable_id,
            requires_grasp=False,
            source_horizontal_distance=self._source_horizontal_distance,
        )
        return meta


def _fixed_support(fixture, name, center, half_xy):
    """Task-local region calibrated to a named, static collision surface."""
    return SemanticRegion(
        stable_id=f"{fixture.stable_id}:support:{name}",
        fixture_id=fixture.stable_id,
        kind="support",
        parent_body=None,
        local_center=np.zeros(3),
        half_size=np.array([*half_xy, 0.0]),
        _world_center=np.asarray(center, dtype=float),
    )


class _HeadboardPillowMixin:
    """Share the scaled native pillow in the two Scene 301 pillow tasks."""

    PILLOW_BODY = "pillow_87d454cec75cae7f8d7c8b2a9579f0b7_1_0_0"
    # Uniformly shrink the previously thinned pillow to 80%: 0.457 x 0.060 x 0.283 m.
    PILLOW_SCALE = (0.8, 0.432, 0.8)

    def _setup_baked_scene_fixtures(self):
        super()._setup_baked_scene_fixtures()
        # Scale both task copies before semantic bounds and inertias compile.
        body = self.mujoco_arena.worldbody.find(f"body[@name='{self.PILLOW_BODY}']")
        scale = np.asarray(self.PILLOW_SCALE)
        for child in body.iter("body"):
            if child is not body:
                scale_body_element(child, scale)
        mesh_names = {geom.get("mesh") for geom in body.iter("geom")}
        for mesh in self.mujoco_arena.asset.findall("mesh"):
            if mesh.get("name") in mesh_names:
                scale_mesh_element(mesh, scale)


class IThorBedroomStandPillowAtHeadboard(
    _HeadboardPillowMixin, _BedsideObjectStateMixin, IThorBedroomEnv
):
    """Stand the flat pillow on the mattress against the actual headboard."""

    TASK_FAMILY = "reorient"
    ATOMIC_SKILL = "ReorientObject"
    TARGET_STABLE_ID = "FloorPlan301:pillow_1_5"
    BED_STABLE_ID = "FloorPlan301:Bed_1_3"
    BED_BODY = "bed_21ab223cdf4f1daf96f5ee52e8696e00_1_1_0"
    FACE_NORMAL_AXIS = 1
    MAX_LEAN_DEGREES = 20.0
    SCENE_SPECS = {
        "FloorPlan301": {
            "fixture": TARGET_STABLE_ID,
            "support_fixture": BED_STABLE_ID,
            # Preserve headboard clearance after the uniform shrink.
            "source_xy": [-0.25, 1.6774],
            "source_quat": [0.7071067812, 0.7071067812, 0.0, 0.0],
            "robot_pose": {"xy": [0.3, 1.48], "yaw": np.pi},
            "lang": "Stand the flat pillow upright against the headboard.",
        },
    }

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.source_fixture = self.get_fixture(id=self.BED_STABLE_ID)
        if self.source_fixture is None:
            raise LookupError(f"{self.scene_name}: missing bed")
        # Bed collision_2 is the mattress support; collision_1 is the headboard.
        self.source_region = _fixed_support(
            self.source_fixture,
            "mattress",
            [-0.63502, 0.87339, 0.56464],
            [0.7019, 1.0164],
        )
        self.fixture_refs["bed"] = self.source_fixture
        self.region_refs["source_surface"] = self.source_region

    def _reset_internal(self):
        super()._reset_internal()
        self._headboard_geom = self.sim.model.geom_name2id(
            f"{self.BED_BODY}_collision_1"
        )
        self._mattress_geom = self.sim.model.geom_name2id(
            f"{self.BED_BODY}_collision_2"
        )
        self._pillow_bodies = self._fixture_body_ids(self.target_fixture)

    def _target_state_reached(self):
        body_id = self.sim.model.body_name2id(self.target_fixture.body_name)
        normal = self.sim.data.body_xmat[body_id].reshape(3, 3)[
            :, self.FACE_NORMAL_AXIS
        ]
        # Face parallel to the headboard, within 20 degrees in either direction.
        headboard_normal = self.sim.data.geom_xmat[self._headboard_geom].reshape(3, 3)[
            :, 2
        ]
        alignment = abs(float(np.dot(normal, headboard_normal)))
        if alignment < np.cos(np.deg2rad(self.MAX_LEAN_DEGREES)):
            return False
        touched = set()
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            if contact.dist < -self.CONTACT_PENETRATION_TOLERANCE:
                if any(
                    int(self.sim.model.geom_bodyid[g]) in self._pillow_bodies
                    for g in (contact.geom1, contact.geom2)
                ):
                    return False
            if contact.dist > 0.003:
                continue
            for subject, support in (
                (contact.geom1, contact.geom2),
                (contact.geom2, contact.geom1),
            ):
                if int(self.sim.model.geom_bodyid[subject]) in self._pillow_bodies:
                    touched.add(int(support))
        return self._headboard_geom in touched and self._mattress_geom in touched

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            pillow_scale=list(self.PILLOW_SCALE),
            headboard_geom=f"{self.BED_BODY}_collision_1",
            maximum_lean_degrees=self.MAX_LEAN_DEGREES,
            success_definition=(
                "pillow upright, touching both mattress and headboard "
                "for 1.0 s; grasp independent"
            ),
        )
        return meta


class _SofaStatueMixin:
    """Stage the living-room upright statue only in the two Scene 301 tasks."""

    STATUE_MODEL = (
        "/objects/thor/Miscellaneous Objects/Statue/Prefabs/"
        "Stone_Statue_4/Stone_Statue_4.xml"
    )
    STATUE_SCALE = 0.5
    STATUE_NAME = "task_sofa_statue"
    TARGET_STABLE_ID = "FloorPlan301:spawned:task_sofa_statue"
    REMOVED_BASKETBALL_ID = "FloorPlan301:Basketball_1_7"

    def _setup_task_fixtures(self):
        self._create_sofa_statue()
        super()._setup_task_fixtures()
        basketball = self.get_fixture(id=self.REMOVED_BASKETBALL_ID)
        if basketball is None:
            raise LookupError(
                f"{self.scene_name}: missing {self.REMOVED_BASKETBALL_ID}"
            )
        self.remove_semantic_fixtures([basketball])

    def _create_sofa_statue(self):
        source = Path(asset_xml_path(self.STATUE_MODEL))
        statue = LivingRoomStatueObject(
            name="statue", source_xml=str(source), scale=self.STATUE_SCALE
        )
        # Serialize the same prepared object used by the living-room atomic task.
        root = ET.Element("mujoco", model=self.STATUE_NAME)
        root.append(deepcopy(statue.asset))
        body = deepcopy(statue.get_obj())
        # Placement controls the free root pose; compute bounds in that frame.
        body.set("pos", "0 0 0")
        body.set("quat", "1 0 0 0")
        # The adapter's invisible registration box describes a different asset;
        # derive semantic bounds from this statue's actual box colliders.
        for geom in list(body.findall("geom")):
            if geom.get("name", "").endswith("reg_bbox"):
                body.remove(geom)
        ET.SubElement(root, "worldbody").append(body)
        with tempfile.TemporaryDirectory(
            prefix="bedroom_301_sofa_statue_"
        ) as directory:
            path = Path(directory) / "statue.xml"
            ET.ElementTree(root).write(path)
            fixture = self.create_scene_fixture(
                xml_path=str(path),
                name=self.STATUE_NAME,
                pos=[0.0, 0.0, 0.0],
                category="Statue",
                source_category="Statue",
                asset_id="Stone_Statue_4",
                ensure_inertials=False,
            )
        fixture.source = {"xml_path": str(source), "scale": self.STATUE_SCALE}

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            removed_fixtures=[self.REMOVED_BASKETBALL_ID],
            target_asset="Stone_Statue_4",
            target_scale=self.STATUE_SCALE,
        )
        return meta


class IThorBedroomLiftStatueFromSofa(
    _SofaStatueMixin, _BedsideObjectStateMixin, IThorBedroomEnv
):
    """Lift the upright stone statue from the default sofa beside the desk."""

    TASK_FAMILY = "lift"
    ATOMIC_SKILL = "LiftObject"
    TARGET_STABLE_ID = _SofaStatueMixin.TARGET_STABLE_ID
    MIN_LIFT_DELTA = 0.30
    SOFA_STABLE_ID = BEDROOM_301_SOFA_ID
    SCENE_SPECS = {
        "FloorPlan301": {
            "fixture": TARGET_STABLE_ID,
            "sofa": SOFA_STABLE_ID,
            "source_xy": [0.93, -0.75],
            "source_quat": [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
            "robot_pose": {"xy": [0.93, -0.3], "yaw": -np.pi / 2},
            "lang": "Bend down and lift the statue from the sofa.",
        },
    }

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.source_fixture = self.get_fixture(id=self.SOFA_STABLE_ID)
        if self.source_fixture is None:
            raise LookupError(f"{self.scene_name}: missing {self.SOFA_STABLE_ID}")
        # The sofa base's flat top, between the armrests and ahead of the back.
        self.source_region = _fixed_support(
            self.source_fixture, "seat", [0.93, -0.85, 0.46379], [0.37, 0.34]
        )
        self.fixture_refs["sofa"] = self.source_fixture
        self.region_refs["source_surface"] = self.source_region

    def _target_state_reached(self):
        return float(self.target_fixture.world_center(self.sim)[2]) >= (
            self._initial_target_z + self.MIN_LIFT_DELTA
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            minimum_lift_delta=self.MIN_LIFT_DELTA,
            initial_target_z=self._initial_target_z,
            success_definition=(
                "statue center at least 0.30 m above reset height "
                "for 1.0 s; grasp independent"
            ),
        )
        return meta


__all__ = ["IThorBedroomStandPillowAtHeadboard", "IThorBedroomLiftStatueFromSofa"]
