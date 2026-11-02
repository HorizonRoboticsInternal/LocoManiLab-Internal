"""Shared table-sized raised work surface for living-room tasks."""

import xml.etree.ElementTree as ET

import numpy as np

from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion


class TableSizedRaisedSurfaceMixin:
    """Add an axis-aligned riser matching an existing support region."""

    WORK_SURFACE_RGBA = "0.42 0.30 0.20 1"
    WORK_SURFACE_TOP_Z = 0.75

    def _select_work_surface_support(self, source_positions, fixture_id=None):
        calibrated = self._get_task_spec().get("work_surface_support")
        if calibrated is not None:
            fixture = self.get_fixture(id=calibrated["fixture"])
            center = np.asarray(calibrated["center"], dtype=float)
            half_xy = np.asarray(calibrated["half_xy"], dtype=float)
            if fixture is None or center.shape != (3,) or half_xy.shape != (2,):
                raise ValueError(
                    f"{self.scene_name}/{self.SPEC_TASK}: invalid "
                    "work_surface_support"
                )
            region = SemanticRegion(
                stable_id=(f"{fixture.stable_id}:task:{self.SPEC_TASK}:base_support"),
                fixture_id=fixture.stable_id,
                kind="support",
                parent_body=None,
                local_center=np.zeros(3),
                half_size=np.array([half_xy[0], half_xy[1], 0.02]),
                _world_center=center,
                _world_yaw=float(calibrated.get("yaw", 0.0)),
            )
            return region, fixture

        positions = np.asarray(source_positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3 or len(positions) == 0:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: source positions need XYZ"
            )

        candidates = []
        for region in self.get_region(kind="support", return_all=True):
            fixture = self.get_fixture(id=region.fixture_id)
            if fixture is None or (fixture_id and fixture.stable_id != fixture_id):
                continue
            center, yaw = region.world_pose()
            cosine, sine = np.cos(-yaw), np.sin(-yaw)
            delta = positions[:, :2] - center[:2]
            local = np.column_stack(
                (
                    cosine * delta[:, 0] - sine * delta[:, 1],
                    sine * delta[:, 0] + cosine * delta[:, 1],
                )
            )
            if np.any(np.abs(local) > region.half_size[:2] - 0.005):
                continue
            z_error = float(np.mean(np.abs(positions[:, 2] - center[2])))
            candidates.append((z_error, region.stable_id, region, fixture))

        if not candidates:
            return None, None
        _, _, region, fixture = min(candidates)
        return region, fixture

    def _setup_table_sized_work_surface(self, source_positions, fixture_id=None):
        base_region, base_fixture = self._select_work_surface_support(
            source_positions, fixture_id=fixture_id
        )
        if base_region is None:
            self._scene_invalid_reason = (
                f"{self.scene_name}/{self.SPEC_TASK}: no support under source objects"
            )
            return None, None
        self.base_support_region = base_region
        self.base_support_fixture = base_fixture
        region, fixture = self._add_table_sized_work_surface(
            base_region,
            top_z=float(self.WORK_SURFACE_TOP_Z),
        )
        self.work_surface_region = region
        self.work_surface_fixture = fixture
        return region, fixture

    def _add_table_sized_work_surface(self, base_region, top_z):
        base_center, base_yaw = base_region.world_pose()
        half_xy = np.asarray(base_region.half_size[:2], dtype=float)
        height = float(top_z) - float(base_center[2])
        if half_xy.shape != (2,) or np.any(half_xy <= 0.0):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: invalid base support size"
            )
        if not np.isclose(np.sin(base_yaw), 0.0, atol=1e-6):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: rotated support cannot use "
                "an axis-aligned work surface"
            )
        if height <= 0.02:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: work surface must be above "
                "the original support"
            )

        body_name = f"raised_work_surface_{self.SPEC_TASK}_{self.scene_name}"
        body = ET.SubElement(
            self.mujoco_arena.worldbody,
            "body",
            name=body_name,
            pos=(f"{base_center[0]:.8g} {base_center[1]:.8g} " f"{base_center[2]:.8g}"),
        )
        common = dict(
            type="box",
            pos=f"0 0 {height / 2.0:.8g}",
            size=f"{half_xy[0]:.8g} {half_xy[1]:.8g} {height / 2.0:.8g}",
            rgba=self.WORK_SURFACE_RGBA,
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{body_name}_geom",
            friction="1 0.005 0.0001",
            contype="1",
            conaffinity="1",
            group="0",
            **common,
        )
        ET.SubElement(
            body,
            "geom",
            name=f"{body_name}_visual",
            contype="0",
            conaffinity="0",
            group="1",
            **common,
        )

        fixture_id = f"{self.scene_name}:task:{self.SPEC_TASK}:work_surface"
        center = np.array([base_center[0], base_center[1], top_z], dtype=float)
        region = SemanticRegion(
            stable_id=f"{fixture_id}:region:support",
            fixture_id=fixture_id,
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.array([half_xy[0], half_xy[1], 0.02]),
            _world_center=center,
            _world_yaw=0.0,
        )
        fixture = SemanticFixture(
            stable_id=fixture_id,
            name=f"{self.SPEC_TASK}_work_surface",
            category="table_riser",
            source_category="TaskRaisedWorkSurface",
            room_id=None,
            body_name=body_name,
            body_names=[body_name],
            site_names=[],
            asset_id=None,
            object_id=body_name,
            center=np.array(
                [base_center[0], base_center[1], base_center[2] + height / 2.0]
            ),
            half_size=np.array([half_xy[0], half_xy[1], height / 2.0]),
            yaw=0.0,
            front=np.array([1.0, 0.0]),
            capabilities={"support", "task_calibrated"},
            regions={region.stable_id: region},
        )
        self.semantic_scene.add_fixture(fixture)
        return region, fixture

    def _work_surface_bottom_position(self, position):
        position = np.asarray(position, dtype=float)
        if position.shape != (3,) or self.work_surface_region is None:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: invalid work-surface placement"
            )
        center, _ = self.work_surface_region.world_pose(self.sim)
        return np.array([position[0], position[1], center[2]], dtype=float)


__all__ = ["TableSizedRaisedSurfaceMixin"]
