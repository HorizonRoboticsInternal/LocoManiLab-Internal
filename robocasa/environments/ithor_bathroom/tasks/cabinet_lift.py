"""Lift the native 405 soap bottle from an already-open cabinet."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robocasa.environments.ithor_atomic.task_assets import asset_xml_path
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.models.fixtures import Fixture
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_specs,
    get_supported_scene_ids,
)
from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion

from .lift import _BathroomNativeLiftTask


class _BathroomCabinetSoapTask(_BathroomNativeLiftTask):
    """Lift the bottle 30 cm for one second, independent of hand state."""

    SPEC_TASK = "lift_soap_bottle_from_cabinet"
    SOURCE_LABEL = "soap bottle"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK,
        required=(
            "source_fixture",
            "surface_fixture",
            "surface_region",
            "source_xy",
            "remove_fixtures",
            "auxiliary_placements",
            "robot_pose",
            "lang",
            "cabinet_fixture",
            "cabinet_joint",
            "cabinet_open_fraction",
            "cabinet_region",
            "table_xy",
            "table_model",
            "table_base_z",
        ),
    )
    TABLE_NAME = "soap_cabinet_table"
    TABLE_WELD_NAME = "task_soap_cabinet_table_fixed_base"
    INITIAL_POSE_JITTER = False

    def _setup_task_fixtures(self):
        spec = self.scene_task_spec()
        xy = np.asarray(spec["table_xy"], dtype=float)
        self.task_table = Fixture(
            xml=asset_xml_path(spec["table_model"]),
            name=self.TABLE_NAME,
            duplicate_collision_geoms=True,
            joints=[dict(type="free", damping="0.001")],
            rng=self.rng,
        )
        self.task_table.set_pos([*xy, float(spec["table_base_z"])])
        self.furniture[self.TABLE_NAME] = self.task_table
        self.table_fixture = SemanticFixture(
            stable_id=spec["surface_fixture"],
            name=self.TABLE_NAME,
            category="table",
            source_category="bathroom_cabinet_table",
            room_id=None,
            body_name=self.task_table.root_body,
            body_names=[self.task_table.root_body],
            site_names=[],
            asset_id="bathroom_cabinet_table",
            object_id=self.TABLE_NAME,
            center=np.r_[xy, float(spec["table_base_z"]) + 0.175],
            half_size=np.array([0.14, 0.14, 0.175]),
            yaw=0.0,
            front=np.array([1.0, 0.0]),
            local_center=np.array([0.0, 0.0, 0.175]),
            local_half_size=np.array([0.14, 0.14, 0.175]),
        )
        region = SemanticRegion(
            stable_id=spec["surface_region"],
            fixture_id=self.table_fixture.stable_id,
            kind="support",
            parent_body=self.task_table.root_body,
            local_center=np.array([0.0, 0.0, 0.35]),
            half_size=np.array([0.14, 0.14, 0.01]),
        )
        self.table_fixture.regions[region.stable_id] = region
        self.semantic_scene.add_fixture(self.table_fixture)
        ET.SubElement(
            self.mujoco_arena.equality,
            "weld",
            name=self.TABLE_WELD_NAME,
            body1=self.task_table.root_body,
            active="false",
            solref="0.002 1",
            solimp="0.9999 0.9999 0.001 0.5 2",
        )
        super()._setup_task_fixtures()
        self.cabinet_fixture = self._exact_fixture(spec["cabinet_fixture"])
        self.cabinet_body_fixture = self._exact_fixture(
            spec["cabinet_region"].split(":region:")[0]
        )
        if self.cabinet_fixture is None or self.cabinet_body_fixture is None:
            raise LookupError(f"{self.scene_name}: missing soap cabinet")
        self.cabinet_joint = next(
            joint
            for joint in self.cabinet_fixture.joints.values()
            if joint.name == spec["cabinet_joint"]
        )
        self.cabinet_region = self.cabinet_body_fixture.regions[spec["cabinet_region"]]
        # This deep receptacle's lower face is the cabinet's real bottom shelf.
        center, _ = self.cabinet_region.world_pose()
        self.table_bottom_region = SemanticRegion(
            stable_id=f"{self.cabinet_body_fixture.stable_id}:region:soap_table_support",
            fixture_id=self.cabinet_body_fixture.stable_id,
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.r_[self.cabinet_region.half_size[:2], 0.01],
            _world_center=center - np.r_[0.0, 0.0, self.cabinet_region.half_size[2]],
        )
        self.fixture_refs["source_cabinet"] = self.cabinet_fixture

    def _setup_scene(self):
        spec = self.scene_task_spec()
        joint = self.cabinet_joint
        self.sim.data.set_joint_qpos(
            joint.name,
            joint.closed_qpos
            + float(spec["cabinet_open_fraction"])
            * (joint.open_qpos - joint.closed_qpos),
        )
        joint_id = self.sim.model.joint_name2id(joint.name)
        self.sim.data.qvel[self.sim.model.jnt_dofadr[joint_id]] = 0.0
        equality_id = mujoco.mj_name2id(
            self.sim.model._model, mujoco.mjtObj.mjOBJ_EQUALITY, self.TABLE_WELD_NAME
        )
        self.sim.data.eq_active[equality_id] = 0
        self.sim.forward()
        if not self._place_fixture_on_support(
            self.table_fixture,
            self.table_bottom_region,
            self.cabinet_body_fixture,
            spec["table_xy"],
            contract_name="soap_cabinet_table_shelf_contact",
            probe_above=0.01,
            probe_below=0.01,
            probe_step=0.0005,
        ):
            raise RuntimeError(f"{self.scene_name}: cabinet table has no shelf contact")
        body_id = self.sim.model.body_name2id(self.task_table.root_body)
        rotation = self.sim.data.body_xmat[body_id].reshape(3, 3)
        position = self.sim.data.body_xpos[body_id].copy()
        quaternion = self.sim.data.body_xquat[body_id].copy()
        self.sim.model.eq_data[equality_id, 3:6] = -rotation.T @ position
        self.sim.model.eq_data[equality_id, 6:10] = quaternion * [1.0, -1.0, -1.0, -1.0]
        self.sim.data.eq_active[equality_id] = 1
        self.sim.forward()
        super()._setup_scene()

    def _fixture_bbox_points(self, fixture):
        if fixture is not self.source_fixture:
            return super()._fixture_bbox_points(fixture)
        # Rotated mesh AABBs inflate this bottle's footprint. Use its actual
        # mesh vertices to check full support near the robot-facing table edge.
        model, data = self.sim.model, self.sim.data
        body_ids = self._fixture_body_ids(fixture)
        points = []
        for geom_id in np.flatnonzero(np.isin(model.geom_bodyid, list(body_ids))):
            if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
                raise RuntimeError("expected native soap bottle mesh geometry")
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            vertices = model.mesh_vert[start : start + count]
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            points.append(vertices @ rotation.T + data.geom_xpos[geom_id])
        return np.concatenate(points)

    def _raw_success(self):
        # Height alone determines success, independent of grasp or hand contact.
        return self._source_root_z() >= self._initial_source_z + self.MIN_LIFT_DELTA

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            {
                "source_cabinet": self.cabinet_fixture.stable_id,
                "cabinet_initial_open_fraction": self.scene_task_spec()[
                    "cabinet_open_fraction"
                ],
                "source_initial_height": float(self._initial_source_z),
                "source_support_height": float(
                    self.work_surface_region.world_pose(self.sim)[0][2]
                ),
                "success_definition": (
                    "the soap bottle must remain at least 0.30 m above its "
                    "reset height for 1.0 s, independent of grasp or hand contact"
                ),
            }
        )
        return ep_meta


class IThorBathroomLiftSoapBottleFromCabinet(
    _BathroomCabinetSoapTask, IThorBathroomEnv
):
    """Lift the bottle 30 cm for one second, independent of hand state."""


__all__ = ["IThorBathroomLiftSoapBottleFromCabinet"]
