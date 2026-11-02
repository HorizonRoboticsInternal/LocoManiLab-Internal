"""Scene 201: carry a coffee-table mug to an open box on the sofa."""

import math
import os

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_atomic.task_assets import (
    MUG_MODEL,
    MUG_SCALE as CANONICAL_MUG_SCALE,
    asset_xml_path,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomMugFromCoffeeTableToBoxOnSofa(
    IThorMultiStageCompositeMixin,
    IThorLivingRoomEnv,
):
    """Take the coffee-table mug and place it in an open box on the sofa."""

    SUPPORTED_SCENE_IDS = (201,)
    SPEC_TASK = "mug_from_coffee_table_to_box_on_sofa"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    INITIAL_POSE_JITTER = False

    BOX_XML = "models/assets/objects/manual/open_cardboard_box/model.xml"
    BOX_SCALE = 1.2
    MUG_NAME = "mug"
    MUG_XML = MUG_MODEL
    MUG_SCALE = CANONICAL_MUG_SCALE
    NATIVE_BOWL_BODY = "bowl_6befd62f08fd322391939c2b44d3f839_1_0_0"
    COFFEE_TABLE_BODY = "coffeetable_115d8cc298b8c8d1896da0ead948f727_1_0_0"
    SOFA_BODY = "sofa_3d3e882f3ea660fd8789fd4c17c8805c_1_0_0"

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
                "mug_xy",
                "mug_yaw",
                "box_xy",
                "box_yaw",
                "sofa_surface_z",
                "box_inner_half_xy",
                "box_floor_z",
                "box_rim_z",
                "containment_tolerance",
                "hold_seconds",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _box_xml_path(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", self.BOX_XML)
        )

    def _mug_xml_path(self):
        return asset_xml_path(self.MUG_XML)

    def _get_obj_cfgs(self):
        return [
            dict(
                name="cardboard_box",
                obj_groups=self._box_xml_path(),
                object_scale=self.BOX_SCALE,
            ),
            dict(
                name=self.MUG_NAME,
                obj_groups=self._mug_xml_path(),
                object_scale=self.MUG_SCALE,
            ),
        ]

    def sample_object(self, groups, **kwargs):
        if groups == self._box_xml_path():
            scale = kwargs.get("object_scale") or 1.0
            return (
                dict(mjcf_path=groups, scale=scale),
                dict(
                    mjcf_path=groups,
                    cat="cardboard_box",
                    groups=["cardboard_box"],
                    groups_containing_sampled_obj=["cardboard_box"],
                    split=None,
                ),
            )
        return super().sample_object(groups, **kwargs)

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualMugToSofaBoxSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                name="take_mug_from_coffee_table",
                success="_mug_removed_from_coffee_table",
                hold_steps=5,
            ),
            IThorAtomicTask(
                name="put_mug_in_box_on_sofa",
                success="_mug_in_box",
                task_cls="IThorLivingRoomMugIntoBox",
            ),
        )

    def _required_stage_hold_steps(self, task):
        if task is self._atomic_tasks[-1]:
            return math.ceil(
                float(self._get_task_spec()["hold_seconds"]) * self.control_freq
            )
        return super()._required_stage_hold_steps(task)

    def _setup_baked_scene_fixtures(self):
        super()._setup_baked_scene_fixtures()
        removed = self.remove_scene_bodies((self.NATIVE_BOWL_BODY,))
        if self.NATIVE_BOWL_BODY not in removed:
            raise RuntimeError(
                f"{self.scene_name}: could not remove the LC01 native bowl"
            )

    def _spawn_clearable_fixture_roots(self):
        roots = super()._spawn_clearable_fixture_roots()
        protected = {
            self.sim.model.body_name2id(self.COFFEE_TABLE_BODY),
            self.sim.model.body_name2id(self.SOFA_BODY),
        }
        return roots - protected

    def _setup_references(self):
        super()._setup_references()
        mug = self.objects[self.MUG_NAME]
        mug_root = self.sim.model.body_name2id(mug.root_body)
        table_root = self.sim.model.body_name2id(self.COFFEE_TABLE_BODY)
        self._mug_geom_ids = self._collidable_geoms(self._body_tree_ids([mug_root]))
        self._coffee_table_geom_ids = self._collidable_geoms(
            self._body_tree_ids([table_root])
        )
        self._mug_joint = mug.joints[0]
        box = self.objects["cardboard_box"]
        self._box_joint = box.joints[0]
        self._box_body_id = self.sim.model.body_name2id(box.root_body)

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

    def _place_box_on_sofa(self):
        spec = self._get_task_spec()
        yaw = float(spec["box_yaw"])
        qpos = self.sim.data.get_joint_qpos(self._box_joint).copy()
        qpos[:2] = np.asarray(spec["box_xy"], dtype=float)
        qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.sim.data.set_joint_qpos(self._box_joint, qpos)
        self.sim.forward()
        qpos[2] += (
            float(spec["sofa_surface_z"])
            - self.object_bottom_z("cardboard_box")
            - 0.001
        )
        self.sim.data.set_joint_qpos(self._box_joint, qpos)
        self.sim.data.set_joint_qvel(self._box_joint, np.zeros(6))
        self.sim.forward()

    def _place_mug_on_coffee_table(self):
        spec = self._get_task_spec()
        yaw = float(spec["mug_yaw"])
        qpos = self.sim.data.get_joint_qpos(self._mug_joint).copy()
        qpos[:2] = np.asarray(spec["mug_xy"], dtype=float)
        qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.sim.data.set_joint_qpos(self._mug_joint, qpos)
        self.sim.forward()
        table_top = float(self._geom_points(self._coffee_table_geom_ids)[:, 2].max())
        mug_bottom = float(self._geom_points(self._mug_geom_ids)[:, 2].min())
        qpos[2] += table_top - mug_bottom - 0.001
        self.sim.data.set_joint_qpos(self._mug_joint, qpos)
        self.sim.data.set_joint_qvel(self._mug_joint, np.zeros(6))
        self.sim.forward()

    def _set_task_layout(self):
        self._place_box_on_sofa()
        self._place_mug_on_coffee_table()

    def _before_object_placement(self):
        super()._before_object_placement()
        self._set_task_layout()

    def _constrain_initial_jitter_candidate(self):
        self._set_task_layout()

    def _mug_on_coffee_table(self):
        mug_points = self._geom_points(self._mug_geom_ids)
        table_points = self._geom_points(self._coffee_table_geom_ids)
        mug_center = (mug_points.min(axis=0) + mug_points.max(axis=0)) / 2
        table_low = table_points.min(axis=0)
        table_high = table_points.max(axis=0)
        return bool(
            np.all(mug_center[:2] >= table_low[:2])
            and np.all(mug_center[:2] <= table_high[:2])
            and abs(float(mug_points[:, 2].min()) - float(table_high[2])) <= 0.05
        )

    def _mug_removed_from_coffee_table(self):
        return not self._mug_on_coffee_table()

    def _mug_points_in_box_frame(self):
        points = self._geom_points(self._mug_geom_ids)
        box_pos = np.asarray(self.sim.data.body_xpos[self._box_body_id])
        box_rot = np.asarray(self.sim.data.body_xmat[self._box_body_id]).reshape(3, 3)
        return (points - box_pos) @ box_rot

    def _mug_in_box(self):
        points = self._mug_points_in_box_frame()
        spec = self._get_task_spec()
        half_xy = np.asarray(spec["box_inner_half_xy"], dtype=float)
        tolerance = float(spec["containment_tolerance"])
        return bool(
            np.all(points[:, :2] >= -half_xy - tolerance)
            and np.all(points[:, :2] <= half_xy + tolerance)
            and float(points[:, 2].min()) >= float(spec["box_floor_z"]) - tolerance
            and float(points[:, 2].max()) <= float(spec["box_rim_z"]) + tolerance
            and self.object_released(self.MUG_NAME)
        )

    def _success_region_bounds(self):
        return []

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "take the mug from the coffee table, walk to the sofa, and put "
                "the mug into the open cardboard box on the sofa"
            ),
            task_id=type(self).__name__,
            target_entity=self.MUG_NAME,
            source_support="FloorPlan201:Coffee_Table_201_1_0",
            target_receptacle="cardboard_box",
            target_support="FloorPlan201:Sofa_201_1_6",
            removed_scene_bodies=[self.NATIVE_BOWL_BODY],
            success_definition=(
                "after the mug leaves the coffee table, release the whole mug "
                "inside the open box on the sofa continuously for 1 second"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomMugFromCoffeeTableToBoxOnSofa"]
