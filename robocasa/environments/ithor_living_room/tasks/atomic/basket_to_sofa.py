"""iTHOR living-room: move a fixed basket asset onto the sofa."""

import os

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomBasketToSofa(IThorLivingRoomEnv):
    ATOMIC_SKILL = "PickPlaceOnSurface"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "pick_place"

    SOURCE_NAME = "basket"
    SOURCE_XML = "models/assets/objects/lightwheel/basket/Basket039/model.xml"
    SOURCE_SCALE = 1.0
    SOURCE_HALF_XY = np.array([0.18, 0.18])
    SOURCE_MARGIN = 0.00
    STAND_DISTANCE = 0.35
    TABLE_TOKENS = ("table", "desk")
    HOLD_SECONDS = 1.0
    RELEASE_DISTANCE = 0.25
    SUCCESS_MARGIN = 0.05
    SUPPORTED_SCENE_IDS = (202, 207, 209, 210, 212, 226)
    SPEC_TASK = "basket_to_sofa"

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "sofa_fixture",
                    "robot_pose",
                    "basket_pos",
                ),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _get_scene_surface_calibration(self):
        return None

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        self.sofa_fixture = self.register_fixture_ref(
            "sofa", dict(id=spec["sofa_fixture"])
        )
        self.HOLD_SECONDS = float(spec.get("hold_seconds", 1.0))
        self.set_robot_base_pose(*self._get_scene_robot_pose_calibration())

    def _get_obj_cfgs(self):
        return [
            dict(
                name=self.SOURCE_NAME,
                obj_groups=self._source_xml_path(),
                object_scale=self.SOURCE_SCALE,
            )
        ]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualLivingRoomBasketSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _source_xml_path(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", self.SOURCE_XML)
        )

    def _reset_internal(self):
        super()._reset_internal()
        if self.SOURCE_NAME in self.objects:
            set_free_joint_bottom_at(
                self,
                self.objects[self.SOURCE_NAME],
                self._get_task_spec()["basket_pos"],
            )
            self.sim.forward()

    def _setup_scene(self):
        super()._setup_scene()
        self._success_hold_steps = 0
        self.HOLD_STEPS = int(round(self.HOLD_SECONDS * self.control_freq))

    def _post_action(self, action):
        ret = super()._post_action(action)
        placed = self._source_is_on_sofa_collision()
        released = self.object_released(self.SOURCE_NAME, th=self.RELEASE_DISTANCE)
        if placed and released:
            self._success_hold_steps += 1
        else:
            self._success_hold_steps = 0
        return ret

    def _check_success(self):
        return self._success_hold_steps >= self.HOLD_STEPS

    def _source_is_on_sofa_collision(self):
        if not self._object_contacts_sofa():
            return False
        center = np.mean(self.object_bbox_points(self.SOURCE_NAME), axis=0)
        for lo, hi in self._success_aabbs():
            if np.all(center[:2] >= lo[:2] - 0.03) and np.all(
                center[:2] <= hi[:2] + 0.03
            ):
                return True
        return False

    def _object_contacts_sofa(self):
        sofa_bodies = self._fixture_body_ids(self.sofa_fixture, include_siblings=True)
        object_bodies = self._body_tree_ids([self.obj_body_id[self.SOURCE_NAME]])
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body1 in object_bodies and body2 in sofa_bodies) or (
                body2 in object_bodies and body1 in sofa_bodies
            ):
                return True
        return False

    def _sofa_geom_ids(self):
        bodies = self._fixture_body_ids(self.sofa_fixture, include_siblings=True)
        return self._collidable_geoms(bodies)

    def _sofa_collision_aabbs(self):
        return [self._geom_world_aabb(geom_id) for geom_id in self._sofa_geom_ids()]

    def _geom_world_aabb(self, geom_id):
        model, data = self.sim.model, self.sim.data
        half = np.asarray(model.geom_size[geom_id], dtype=float)
        local = np.array(np.meshgrid(*[[-v, v] for v in half[:3]])).T.reshape(-1, 3)
        mat = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        points = data.geom_xpos[geom_id] + local @ mat.T
        return points.min(axis=0), points.max(axis=0)

    def _success_aabbs(self):
        margin = np.array([self.SUCCESS_MARGIN, self.SUCCESS_MARGIN, 0.0])
        return [
            (lo - margin, hi + margin)
            for lo, hi in self._sofa_collision_aabbs()
            if hi[2] >= 0.20
        ]

    def _success_region_bounds(self):
        return [
            ((lo + hi) / 2, np.maximum((hi - lo) / 2, 0.01), 0.0)
            for lo, hi in self._success_aabbs()
        ]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick up the basket and place it on the sofa"
        return ep_meta


__all__ = ["IThorLivingRoomBasketToSofa"]
