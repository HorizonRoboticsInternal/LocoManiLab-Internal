"""Scene 209: carry a table-side statue to the trash bin."""

import math
import os

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.objects_into_box import (
    LivingRoomStatueObject,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_oriented_bottom_at,
)
from robocasa.environments.ithor_living_room.tasks.atomic.lift_statue import (
    IThorLivingRoomLiftStatue,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomStatueFromTableToTrashBin(
    IThorMultiStageCompositeMixin,
    IThorLivingRoomEnv,
):
    """Take the statue from the table and drop it into the trash bin."""

    SUPPORTED_SCENE_IDS = (209,)
    SPEC_TASK = "statue_from_table_to_trash_bin"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    INITIAL_POSE_JITTER = False

    STATUE_NAME = "statue"
    STATUE_XML = IThorLivingRoomLiftStatue.TARGET_XML
    STATUE_SCALE = IThorLivingRoomLiftStatue.TARGET_SCALE
    NATIVE_STATUE_BODY = "statue_ad73194bc098e0c7a3246019c7443373_1_0_0"
    TABLE_BODY = "coffeetable_b3f143252f435a2cbc346ca9563926cf_1_0_0"
    BIN_BODY = "ashcan_fcdad2926d886fb599ddce839f895b0d_1_0_0"
    BIN_SITE = "ashcan_fcdad2926d886fb599ddce839f895b0d_1_1_0_" "ReceptacleCollider0_0"

    def __init__(self, *args, scene_ids=None, **kwargs):
        scene_ids = self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids
        kwargs.setdefault("initialization_noise", None)
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        return get_ithor_livingroom_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "robot_pose",
                "statue_xy",
                "statue_yaw",
                "bin_xy_half_tolerance",
                "bin_above_rim_tolerance",
                "bin_below_floor_tolerance",
                "hold_seconds",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _statue_xml_path(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", self.STATUE_XML)
        )

    def _get_obj_cfgs(self):
        return [
            dict(
                name=self.STATUE_NAME,
                obj_groups=self._statue_xml_path(),
                object_scale=self.STATUE_SCALE,
            )
        ]

    def _create_objects(self):
        self.objects = {}
        self.object_cfgs = self._get_obj_cfgs()
        statue = LivingRoomStatueObject(
            name=self.STATUE_NAME,
            source_xml=self._statue_xml_path(),
            scale=self.STATUE_SCALE,
        )
        self.objects[statue.name] = statue
        self.model.merge_objects([statue])

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualStatueToTrashBinSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _setup_baked_scene_fixtures(self):
        super()._setup_baked_scene_fixtures()
        removed = self.remove_scene_bodies((self.NATIVE_STATUE_BODY,))
        if self.NATIVE_STATUE_BODY not in removed:
            raise RuntimeError(f"{self.scene_name}: could not replace native statue")

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                name="take_statue_from_table",
                success="_statue_removed_from_table",
                hold_steps=5,
            ),
            IThorAtomicTask(
                name="put_statue_in_trash_bin",
                success="_statue_in_relaxed_bin_region",
            ),
        )

    def _required_stage_hold_steps(self, task):
        if task is self._atomic_tasks[-1]:
            return math.ceil(
                float(self._get_task_spec()["hold_seconds"]) * self.control_freq
            )
        return super()._required_stage_hold_steps(task)

    def _spawn_clearable_fixture_roots(self):
        roots = super()._spawn_clearable_fixture_roots()
        protected = {
            self.sim.model.body_name2id(self.TABLE_BODY),
            self.sim.model.body_name2id(self.BIN_BODY),
        }
        return roots - protected

    def _setup_references(self):
        super()._setup_references()
        statue = self.objects[self.STATUE_NAME]
        statue_root = self.sim.model.body_name2id(statue.root_body)
        table_root = self.sim.model.body_name2id(self.TABLE_BODY)
        self._statue_geom_ids = self._collidable_geoms(
            self._body_tree_ids([statue_root])
        )
        self._table_geom_ids = self._collidable_geoms(self._body_tree_ids([table_root]))
        self._bin_site_id = self.sim.model.site_name2id(self.BIN_SITE)

    @staticmethod
    def _box_signs():
        return np.asarray(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )

    def _geom_points(self, geom_ids):
        model, data = self.sim.model._model, self.sim.data._data
        points = []
        for geom_id in geom_ids:
            if int(model.geom_type[geom_id]) == 7:
                mesh_id = int(model.geom_dataid[geom_id])
                start = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                local = model.mesh_vert[start : start + count]
            else:
                bounds = model.geom_aabb[geom_id]
                local = bounds[:3] + self._box_signs() * bounds[3:]
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            points.append(local @ rotation.T + data.geom_xpos[geom_id])
        return np.concatenate(points)

    def _place_statue_on_table(self):
        spec = self._get_task_spec()
        table_top = float(self._geom_points(self._table_geom_ids)[:, 2].max())
        position = [*np.asarray(spec["statue_xy"], dtype=float), table_top]
        set_free_joint_oriented_bottom_at(
            self,
            self.objects[self.STATUE_NAME],
            position,
            yaw=float(spec["statue_yaw"]),
            pitch=np.pi / 2.0,
        )
        self.sim.forward()

    def _before_object_placement(self):
        super()._before_object_placement()
        self._place_statue_on_table()

    def _constrain_initial_jitter_candidate(self):
        self._place_statue_on_table()

    def _statue_on_table(self):
        statue = self._geom_points(self._statue_geom_ids)
        table = self._geom_points(self._table_geom_ids)
        center = (statue.min(axis=0) + statue.max(axis=0)) / 2
        return bool(
            np.all(center[:2] >= table.min(axis=0)[:2])
            and np.all(center[:2] <= table.max(axis=0)[:2])
            and abs(float(statue[:, 2].min()) - float(table[:, 2].max())) <= 0.05
        )

    def _statue_removed_from_table(self):
        return not self._statue_on_table()

    def _bin_world_bounds(self):
        model, data = self.sim.model._model, self.sim.data._data
        size = np.asarray(model.site_size[self._bin_site_id], dtype=float)
        rotation = np.asarray(data.site_xmat[self._bin_site_id], dtype=float).reshape(
            3, 3
        )
        points = (
            self._box_signs() * size @ rotation.T + data.site_xpos[self._bin_site_id]
        )
        return points.min(axis=0), points.max(axis=0)

    def _statue_in_relaxed_bin_region(self):
        statue = self._geom_points(self._statue_geom_ids)
        center = (statue.min(axis=0) + statue.max(axis=0)) / 2
        bin_low, bin_high = self._bin_world_bounds()
        spec = self._get_task_spec()
        xy_tolerance = np.asarray(spec["bin_xy_half_tolerance"], dtype=float)
        center_inside = bool(
            np.all(center[:2] >= bin_low[:2] - xy_tolerance)
            and np.all(center[:2] <= bin_high[:2] + xy_tolerance)
            and center[2] >= bin_low[2] - float(spec["bin_below_floor_tolerance"])
            and center[2] <= bin_high[2] + float(spec["bin_above_rim_tolerance"])
        )
        return bool(center_inside and self.object_released(self.STATUE_NAME))

    def _success_region_bounds(self):
        return []

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "take the statue from the side of the table nearest the trash bin, "
                "walk to the trash bin, and drop the statue into it"
            ),
            task_id=type(self).__name__,
            target_entity="Stone_Statue_4",
            source_support="FloorPlan209:Coffee_Table_209_1_1",
            target_receptacle="FloorPlan209:bin_9_12",
            replaced_scene_entity="FloorPlan209:Towel_Statue_1_22",
            success_definition=(
                "after the statue leaves the table, release it with its center "
                "inside an expanded volume around the trash-bin opening "
                "continuously for 1 second; the statue may protrude above or rest "
                "across the rim"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomStatueFromTableToTrashBin"]
