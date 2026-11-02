"""Bend to plate an apple on a low table in an open Kitchen area."""

import xml.etree.ElementTree as ET

import numpy as np

from robocasa.environments.ithor_atomic.pick_place import SupportContactPlacementMixin
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.environments.ithor_kitchen.tasks.atomic.apple_on_plate import (
    IThorKitchenAppleOnPlate,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
    get_ithor_kitchen_task_specs,
)
from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion


class IThorKitchenBendAndPutAppleOnPlate(
    SupportContactPlacementMixin, IThorKitchenAppleOnPlate
):
    """Low-table manipulation; only the final apple-on-plate relation is scored."""

    ATOMIC_SKILL = "BendAndPickPlace"
    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "bend_then_pick_place"
    SPEC_TASK = "bend_and_put_apple_on_plate"
    TASK_LANG = "bend down and put the apple on the plate on the low table"
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    SCENE_SPECS = get_ithor_kitchen_task_specs(
        SPEC_TASK,
        required=(
            "robot_pose",
            "table_center",
            "table_half_xy",
            "plate_pos",
            "apple_pos",
            "hold_seconds",
        ),
    )
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0

    def _get_task_spec(self):
        if getattr(self, "_low_table_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_kitchen_task_spec(
                self.scene_name, self.SPEC_TASK
            )
            self._low_table_spec_scene = self.scene_name
        return self.task_spec

    def _get_obj_cfgs(self):
        return [
            dict(
                name="plate", info=dict(mjcf_path="objaverse/plate/plate_0/model.xml")
            ),
            dict(
                name="apple", info=dict(mjcf_path="objaverse/apple/apple_0/model.xml")
            ),
        ]

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        center = np.asarray(spec["table_center"], dtype=float)
        half_xy = np.asarray(spec["table_half_xy"], dtype=float)
        # A fixed four-legged table, with its feet on the scene floor (z=0).
        body_name = "task_kitchen_low_table"
        body = ET.SubElement(self.mujoco_arena.worldbody, "body", name=body_name)
        body.set("pos", " ".join(map(str, [*center[:2], 0.0])))
        thickness = 0.025
        ET.SubElement(
            body,
            "geom",
            name=f"{body_name}_top",
            type="box",
            group="1",
            pos=f"0 0 {center[2] - thickness}",
            size=f"{half_xy[0]} {half_xy[1]} {thickness}",
            rgba="0.55 0.35 0.18 1",
            friction="1 0.005 0.0001",
        )
        leg_half_height = (center[2] - 2 * thickness) / 2
        for index, (x_sign, y_sign) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            x = x_sign * (half_xy[0] - 0.035)
            y = y_sign * (half_xy[1] - 0.035)
            ET.SubElement(
                body,
                "geom",
                name=f"{body_name}_leg_{index}",
                type="box",
                group="1",
                pos=f"{x} {y} {leg_half_height}",
                size=f"0.025 0.025 {leg_half_height}",
                rgba="0.35 0.22 0.12 1",
            )
        stable_id = f"{self.scene_name}:spawned:{body_name}"
        self.low_table_region = SemanticRegion(
            stable_id=f"{stable_id}:region:support",
            fixture_id=stable_id,
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.r_[half_xy, 0.02],
            _world_center=center,
            _world_yaw=0.0,
        )
        self.low_table_fixture = SemanticFixture(
            stable_id=stable_id,
            name=body_name,
            category="table",
            source_category="TaskLowTable",
            room_id=None,
            body_name=body_name,
            body_names=[body_name],
            site_names=[],
            asset_id=None,
            object_id=body_name,
            center=np.r_[center[:2], center[2] / 2],
            half_size=np.r_[half_xy, center[2] / 2],
            yaw=0.0,
            front=np.array([1.0, 0.0]),
            capabilities={"support"},
            regions={self.low_table_region.stable_id: self.low_table_region},
        )
        self.semantic_scene.add_fixture(self.low_table_fixture)
        self.fixture_refs["low_table"] = self.low_table_fixture
        self.region_refs["low_table"] = self.low_table_region

    def _place_task_objects(self):
        super()._place_task_objects()
        for name, pos in zip(("plate", "apple"), self._configured_object_positions()):
            if not self.place_object_on_support_contact(
                name,
                self.low_table_region,
                self.low_table_fixture,
                xy=pos[:2],
                inward_fractions=(0.0,),
                probe_step=0.0005,
            ):
                raise RuntimeError(
                    f"{self.scene_name}: {name} cannot rest on the low table"
                )

    def _setup_scene(self):
        super()._setup_scene()
        self._place_task_objects()
        pose = self._get_task_spec()["robot_pose"]
        xy = np.asarray(pose["pos"][:2], dtype=float)
        if hasattr(self, "_robot_free_joint_z"):
            del self._robot_free_joint_z
        if not (
            self.robot_footprint_is_on_floor(xy, radius=0.25)
            and self._set_robot_free_pose(xy, float(pose["yaw"]))
            and self._calibrate_robot_free_joint_height()
        ):
            raise RuntimeError(f"{self.scene_name}: invalid low-table standing pose")

    def _reset_internal(self):
        IThorKitchenEnv._reset_internal(self)
        self._place_task_objects()
        self.success_hold_seconds = float(self._get_task_spec()["hold_seconds"])
        self._reset_task_progress()
        if self.count_body_collisions() or self._raw_success():
            raise RuntimeError(f"{self.scene_name}: invalid low-table initial state")

    def _reset_task_progress(self):
        self._success_hold_steps = 0

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=self.TASK_LANG,
            target_entity="apple",
            target_receptacle="plate",
            support_fixture=self.low_table_fixture.stable_id,
            table_height=float(self._get_task_spec()["table_center"][2]),
            success_hold_seconds=self.success_hold_seconds,
            success_definition=(
                "the apple contacts the plate near its center continuously for "
                f"{self.success_hold_seconds:g} seconds"
            ),
        )
        return meta


__all__ = ["IThorKitchenBendAndPutAppleOnPlate"]
