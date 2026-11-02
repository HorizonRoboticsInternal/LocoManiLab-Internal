"""Optional fixed stool support for the native bathroom scrub brush."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robocasa.environments.ithor_atomic.task_assets import asset_xml_path
from robocasa.models.fixtures import Fixture
from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion


class _BathroomScrubBrushSupportMixin:
    """Stage a native scrub brush on a fixed, floor-contacting stool."""

    STOOL_NAME = "brush_stool"
    STOOL_BASE_WELD_NAME = "task_bathroom_brush_stool_fixed_base"
    SOURCE_STOOL_FIELD = "source_stool"

    def _setup_task_fixtures(self):
        spec = self.scene_task_spec()
        layout = spec.get(self.SOURCE_STOOL_FIELD)
        if layout is not None:
            # Register before the workspace mixin resolves the exact support ID.
            xy = np.asarray(layout["xy"], dtype=float)
            yaw = float(layout["yaw"])
            self.task_stool = Fixture(
                xml=asset_xml_path(layout["model"]),
                name=self.STOOL_NAME,
                duplicate_collision_geoms=False,
                scale=layout["scale"],
                rng=self.rng,
                joints=[dict(type="free", damping="0.001")],
            )
            initial_z = -float(self.task_stool.bottom_offset[2])
            self.task_stool.set_pos([*xy, initial_z])
            self.task_stool.set_euler([0.0, 0.0, yaw])
            self.furniture[self.STOOL_NAME] = self.task_stool
            bbox = self.task_stool._regions["bbox"]
            lower = np.asarray(bbox["p0"], dtype=float)
            upper = np.array([bbox["px"][0], bbox["py"][1], bbox["pz"][2]])
            local_center = 0.5 * (lower + upper)
            half_size = 0.5 * (upper - lower)
            rotation = np.array(
                [
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw), np.cos(yaw), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            self.task_stool_fixture = SemanticFixture(
                stable_id=spec["surface_fixture"],
                name=self.STOOL_NAME,
                category="stool",
                source_category="Stool005",
                room_id=None,
                body_name=self.task_stool.root_body,
                body_names=[
                    body.get("name")
                    for body in self.task_stool.worldbody.iter("body")
                    if body.get("name")
                ],
                site_names=[],
                asset_id="Stool005",
                object_id=self.STOOL_NAME,
                center=np.r_[xy, initial_z] + rotation @ local_center,
                half_size=half_size,
                yaw=yaw,
                front=rotation[:2, 0],
                local_center=local_center,
                local_half_size=half_size,
            )
            region = SemanticRegion(
                stable_id=spec["surface_region"],
                fixture_id=self.task_stool_fixture.stable_id,
                kind="support",
                parent_body=self.task_stool.root_body,
                local_center=np.r_[local_center[:2], upper[2]],
                half_size=np.r_[layout["top_half_xy"], 0.01],
            )
            self.task_stool_fixture.regions[region.stable_id] = region
            self.semantic_scene.add_fixture(self.task_stool_fixture)
            ET.SubElement(
                self.mujoco_arena.equality,
                "weld",
                name=self.STOOL_BASE_WELD_NAME,
                body1=self.task_stool.root_body,
                active="false",
                solref="0.002 1",
                solimp="0.9999 0.9999 0.001 0.5 2",
            )
        super()._setup_task_fixtures()

    def _stool_base_weld_id(self):
        equality_id = mujoco.mj_name2id(
            self.sim.model._model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            self.STOOL_BASE_WELD_NAME,
        )
        if equality_id < 0:
            raise RuntimeError(f"{self.scene_name}: missing brush stool weld")
        return equality_id

    def _setup_scene(self):
        layout = self.scene_task_spec().get(self.SOURCE_STOOL_FIELD)
        if layout is not None:
            equality_id = self._stool_base_weld_id()
            self.sim.data.eq_active[equality_id] = 0
            self.sim.forward()
            # Probe the real Floor collision surface before fixing the base;
            # never suspend the stool in air.
            if not self._place_fixture_on_floor(
                self.task_stool_fixture,
                layout["xy"],
                contract_name="bathroom_brush_stool_floor_contact",
                probe_above=float(layout["floor_probe_above"]),
                probe_below=float(layout["floor_probe_below"]),
                probe_step=float(layout["floor_probe_step"]),
            ):
                raise RuntimeError(
                    f"{self.scene_name}: no stable floor pose for brush stool"
                )
            self.sim.data.eq_active[equality_id] = 1
            self.sim.forward()
        super()._setup_scene()


__all__ = ["_BathroomScrubBrushSupportMixin"]
