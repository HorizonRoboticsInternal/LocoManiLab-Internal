"""Scene 201: carry a canonical mug from the sofa to the coffee table."""

import math

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.task_assets import (
    MUG_MODEL,
    MUG_SCALE as CANONICAL_MUG_SCALE,
    asset_xml_path,
)
from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomMugFromSofaToCoffeeTable(
    IThorMultiStageCompositeMixin,
    IThorLivingRoomEnv,
):
    """Take the mug from the sofa and set it down on the coffee table."""

    SUPPORTED_SCENE_IDS = (201,)
    SPEC_TASK = "mug_from_sofa_to_coffee_table"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    INITIAL_POSE_JITTER = False

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
                "sofa_surface_z",
                "max_linear_speed",
                "max_angular_speed",
                "hold_seconds",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _get_obj_cfgs(self):
        return [
            dict(
                name=self.MUG_NAME,
                obj_groups=asset_xml_path(self.MUG_XML),
                object_scale=self.MUG_SCALE,
            )
        ]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualMugFromSofaToTableSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                name="take_mug_from_sofa",
                success="_mug_removed_from_sofa",
                hold_steps=5,
            ),
            IThorAtomicTask(
                name="put_mug_on_coffee_table",
                success="_mug_set_down_on_coffee_table",
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
                f"{self.scene_name}: could not remove the LC02 native bowl"
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
        sofa_root = self.sim.model.body_name2id(self.SOFA_BODY)
        self._mug_joint = mug.joints[0]
        self._mug_body_ids = self._body_tree_ids([mug_root])
        self._sofa_body_ids = self._body_tree_ids([sofa_root])
        self._mug_geom_ids = self._collidable_geoms(self._mug_body_ids)
        self._coffee_table_geom_ids = self._collidable_geoms(
            self._body_tree_ids([table_root])
        )

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

    def _place_mug_on_sofa(self):
        spec = self._get_task_spec()
        yaw = float(spec["mug_yaw"])
        qpos = self.sim.data.get_joint_qpos(self._mug_joint).copy()
        qpos[:2] = np.asarray(spec["mug_xy"], dtype=float)
        qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.sim.data.set_joint_qpos(self._mug_joint, qpos)
        self.sim.forward()
        qpos[2] += (
            float(spec["sofa_surface_z"])
            - float(self._geom_points(self._mug_geom_ids)[:, 2].min())
            - 0.001
        )
        self.sim.data.set_joint_qpos(self._mug_joint, qpos)
        self.sim.data.set_joint_qvel(self._mug_joint, np.zeros(6))
        self.sim.forward()

    def _before_object_placement(self):
        super()._before_object_placement()
        self._place_mug_on_sofa()

    def _constrain_initial_jitter_candidate(self):
        self._place_mug_on_sofa()

    def _mug_contacts_sofa(self):
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body_1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body_2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body_1 in self._mug_body_ids and body_2 in self._sofa_body_ids) or (
                body_2 in self._mug_body_ids and body_1 in self._sofa_body_ids
            ):
                return True
        return False

    def _mug_removed_from_sofa(self):
        return not self._mug_contacts_sofa()

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

    def _mug_is_stable(self):
        start, end = self.sim.model.get_joint_qvel_addr(self._mug_joint)
        velocity = np.asarray(self.sim.data.qvel[start:end])
        spec = self._get_task_spec()
        return bool(
            np.linalg.norm(velocity[:3]) <= float(spec["max_linear_speed"])
            and np.linalg.norm(velocity[3:]) <= float(spec["max_angular_speed"])
        )

    def _mug_set_down_on_coffee_table(self):
        return bool(
            self._mug_on_coffee_table()
            and self._mug_is_stable()
            and self.object_released(self.MUG_NAME)
        )

    def _success_region_bounds(self):
        return []

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "take the mug from the sofa, walk to the coffee table, and set "
                "the mug down on the coffee table"
            ),
            task_id=type(self).__name__,
            target_entity=self.MUG_NAME,
            source_support="FloorPlan201:Sofa_201_1_6",
            target_support="FloorPlan201:Coffee_Table_201_1_0",
            removed_scene_bodies=[self.NATIVE_BOWL_BODY],
            success_definition=(
                "after the mug leaves the sofa, release it stably on the coffee "
                "table continuously for 1 second"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomMugFromSofaToCoffeeTable"]
