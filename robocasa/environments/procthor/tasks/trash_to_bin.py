"""ProcTHOR task: move trash from a table in one room into a bin in another."""

from copy import deepcopy

import numpy as np

from robocasa.environments.procthor.tasks.place_to_place import (
    ProcTHORPlaceToPlaceTask,
    room_label,
)
from robocasa.models.scenes.procthor_task_fixtures import (
    add_task_fixture,
    load_task_fixture_configs,
)
from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion


class ProcTHORTrashToBin(ProcTHORPlaceToPlaceTask):
    ATOMIC_SKILL = "PlaceIntoReceptacle"
    DIFFICULTY_LEVEL = "L3"
    TASK_FAMILY = "receptacle"

    SOURCE_NAME = "trash"
    SOURCE_LANG = "empty can"
    TABLE_NAME = "procthor_task_trash_table"
    BIN_NAME = "procthor_semantic_trash_can"

    BIN_SAFE_HALF = 0.05
    BIN_RADIUS = 0.24
    BIN_BOTTOM_Z = 0.03
    BIN_RIM_Z = 0.62

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.source_room, self.target_room = self._pick_room_pair()
        if self.source_room is None:
            self._scene_invalid_reason = f"{self.scene_name}: need two rooms"
            return
        self._add_table(self.TABLE_NAME, "source", self.source_room)
        self._add_bin_body()

    def _is_scene_valid(self):
        return super()._is_scene_valid() and getattr(self, "source_room", None)

    def _get_obj_cfgs(self):
        return [dict(name=self.SOURCE_NAME, obj_groups="can")]

    def _setup_scene(self):
        super()._setup_scene()
        self._success_hold = 0
        ignore = (self.TABLE_NAME, self.BIN_NAME)
        table_xy = self._free_point(
            self.source_room,
            margin=0.65,
            require_robot_space=True,
            ignore_body_names=ignore,
        )
        bin_xy = self._free_point(
            self.target_room,
            margin=0.40,
            safe_half=self.BIN_SAFE_HALF,
            ignore_body_names=ignore,
        )
        if table_xy is None or bin_xy is None:
            raise RuntimeError(f"{self.scene_name}: no free table/bin floor points")

        self._move_body(self.TABLE_NAME, table_xy)
        self._move_body(self.BIN_NAME, bin_xy, 0.271)
        self._update_table_semantics(
            self.source_fixture, self.source_region, table_xy, self.source_room
        )
        self.target_bin_fixture.center = np.array([bin_xy[0], bin_xy[1], 0.271])
        self.target_bin_region._world_center = self.target_bin_fixture.center.copy()
        if not self.place_robot_at_fixture(self.source_fixture):
            raise RuntimeError(f"{self.scene_name}: cannot place G1 at trash table")
        self.target_bin_pos, _ = self.target_bin_region.world_pose(self.sim)

    def _reset_internal(self):
        super()._reset_internal()
        self._place_robot_by_fixture(self.source_fixture)

    def _apply_object_placements(self, placements):
        super()._apply_object_placements(placements)
        self._place_object_on_region(self.source_region)

    def _add_bin_body(self):
        cfg = deepcopy(
            next(
                c
                for c in load_task_fixture_configs("trash_to_bin_scene18")
                if c["fixture_type"] == "trash_can"
            )
        )
        cfg["name"] = self.BIN_NAME
        cfg["body"]["name"] = self.BIN_NAME
        add_task_fixture(self, cfg)

        stable = f"{self.scene_name}:task:trash_can"
        self.target_bin_region = SemanticRegion(
            stable_id=f"{stable}:region:receptacle",
            fixture_id=stable,
            kind="receptacle",
            parent_body=self.BIN_NAME,
            local_center=np.zeros(3),
            half_size=np.array([0.20, 0.20, 0.32]),
        )
        self.target_bin_fixture = SemanticFixture(
            stable_id=stable,
            name=self.BIN_NAME,
            category="trash_can",
            source_category="TrashCan",
            room_id=self.target_room,
            body_name=self.BIN_NAME,
            body_names=[self.BIN_NAME],
            site_names=[],
            asset_id=cfg.get("asset_id"),
            object_id=self.BIN_NAME,
            center=np.zeros(3),
            half_size=np.array([0.23, 0.23, 0.37]),
            yaw=0.0,
            front=np.array([1.0, 0.0]),
            capabilities={"receptacle"},
            regions={self.target_bin_region.stable_id: self.target_bin_region},
        )
        self.semantic_scene.add_fixture(self.target_bin_fixture)

    def _raw_success(self):
        bbox = self.object_bbox_points(self.SOURCE_NAME)
        center = np.mean(bbox, axis=0)
        bottom = float(np.min(bbox[:, 2]))
        bin_center, _ = self.target_bin_region.world_pose(self.sim)
        return bool(
            np.linalg.norm(center[:2] - bin_center[:2]) <= self.BIN_RADIUS
            and self.BIN_BOTTOM_Z <= float(center[2]) <= self.BIN_RIM_Z
            and bottom <= self.BIN_RIM_Z
            and self.object_released(self.SOURCE_NAME, th=self.RELEASE_DIST)
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        rooms, _ = self.get_room_layout()
        ep_meta.update(
            source_body=self.SOURCE_NAME,
            source_room=self.source_room,
            target_room=self.target_room,
            source_pos=[float(v) for v in self.object_pos(self.SOURCE_NAME)],
            target_bin_pos=[float(v) for v in self.target_bin_pos],
            lang=(
                f"pick up the {self.SOURCE_LANG} from the table in the "
                f"{room_label(rooms[self.source_room]['type'])}, navigate to the "
                f"trash can in the {room_label(rooms[self.target_room]['type'])}, "
                "and drop it inside"
            ),
        )
        return ep_meta


__all__ = ["ProcTHORTrashToBin"]
