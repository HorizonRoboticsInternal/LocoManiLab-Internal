"""ProcTHOR task: move an object from one generated table to another."""

from robocasa.environments.procthor.tasks.place_to_place import (
    ProcTHORPlaceToPlaceTask,
    room_label,
)


class ProcTHORTableToTable(ProcTHORPlaceToPlaceTask):
    ATOMIC_SKILL = "PickPlaceOnSurface"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "pick_place"

    SOURCE_NAME = "item"
    SOURCE_LANG = "empty can"
    SOURCE_TABLE_NAME = "procthor_task_source_table"
    TARGET_TABLE_NAME = "procthor_task_target_table"

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.source_room, self.target_room = self._pick_room_pair()
        if self.source_room is None:
            self._scene_invalid_reason = f"{self.scene_name}: need two rooms"
            return
        self._add_table(self.SOURCE_TABLE_NAME, "source", self.source_room)
        self._add_table(self.TARGET_TABLE_NAME, "target", self.target_room)

    def _is_scene_valid(self):
        return super()._is_scene_valid() and getattr(self, "source_room", None)

    def _get_obj_cfgs(self):
        return [dict(name=self.SOURCE_NAME, obj_groups="can")]

    def _setup_scene(self):
        super()._setup_scene()
        self._success_hold = 0
        ignore = (self.SOURCE_TABLE_NAME, self.TARGET_TABLE_NAME)
        source_xy = self._free_point(
            self.source_room,
            margin=0.65,
            require_robot_space=True,
            require_table_space=True,
            ignore_body_names=ignore,
        )
        target_xy = self._free_point(
            self.target_room,
            margin=0.65,
            require_table_space=True,
            ignore_body_names=ignore,
        )
        if source_xy is None or target_xy is None:
            raise RuntimeError(f"{self.scene_name}: no free table floor points")

        self._move_body(self.SOURCE_TABLE_NAME, source_xy)
        self._move_body(self.TARGET_TABLE_NAME, target_xy)
        self._update_table_semantics(
            self.source_fixture, self.source_region, source_xy, self.source_room
        )
        self._update_table_semantics(
            self.target_fixture, self.target_region, target_xy, self.target_room
        )
        if not self.place_robot_at_fixture(self.source_fixture):
            raise RuntimeError(f"{self.scene_name}: cannot place G1 at source table")

        source_center, _ = self.source_region.world_pose(self.sim)
        target_center, _ = self.target_region.world_pose(self.sim)
        self.source_surface = (source_center, self.source_region.half_size[:2])
        self.target_surface = (target_center, self.target_region.half_size[:2])

    def _reset_internal(self):
        super()._reset_internal()
        self._place_robot_by_fixture(self.source_fixture)

    def _apply_object_placements(self, placements):
        super()._apply_object_placements(placements)
        self._place_object_on_region(self.source_region, near_robot=True)

    def _raw_success(self):
        return self.object_on_target_table()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        rooms, _ = self.get_room_layout()
        ep_meta.update(
            source_object=self.SOURCE_NAME,
            source_room=self.source_room,
            target_room=self.target_room,
            source_surface=[float(v) for v in self.source_surface[0]],
            target_surface=[float(v) for v in self.target_surface[0]],
            lang=(
                f"pick up the {self.SOURCE_LANG} from the table in the "
                f"{room_label(rooms[self.source_room]['type'])}, navigate to the "
                f"other table in the {room_label(rooms[self.target_room]['type'])}, "
                "and place it on top"
            ),
        )
        return ep_meta


__all__ = ["ProcTHORTableToTable"]
