"""Pack two explicitly placed living-room objects into one basket."""

import numpy as np
import robocasa.utils.object_utils as OU
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_kitchen.tasks.atomic.objects_into_box import (
    LivingRoomStatueObject,
)
from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
    set_free_joint_oriented_bottom_at,
)
from robocasa.environments.ithor_living_room.tasks.composite.explicit_spec import (
    ExplicitCompositeSpecMixin,
)
from robocasa.environments.ithor_living_room.tasks.atomic.raised_work_surface import (
    TableSizedRaisedSurfaceMixin,
)


class IThorLivingRoomPackCandleStatueIntoBasket(
    CompositeStageReporterMixin,
    TableSizedRaisedSurfaceMixin,
    ExplicitCompositeSpecMixin,
    IThorLivingRoomEnv,
):
    """Place a candle and a statue into one basket in any order."""

    ATOMIC_SKILL = "PackTwoObjects"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "pack_candle_statue"
    COMPOSITE_ORDERING = "any"
    SPEC_TASK = "pack_candle_statue_into_basket"
    SUPPORTED_SCENE_IDS = (202, 207, 209, 212, 226)
    REQUIRED_SPEC_FIELDS = (
        "work_surface_support",
        "robot_pose",
        "basket_pos",
        "basket_yaw",
        "basket_scale",
        "object_poses",
        "candle_scale",
        "statue_scale",
        "success_xy_tolerance",
        "hold_seconds",
    )
    ITEM_NAMES = ("candle", "statue")
    BASKET_XML = "models/assets/objects/lightwheel/basket/Basket039/model.xml"
    CANDLE_XML = "models/assets/objects/objaverse/candle/candle_3/model.xml"
    STATUE_XML = (
        "models/assets/objects/thor/Miscellaneous Objects/Statue/Prefabs/"
        "Stone_Statue_4/Stone_Statue_4.xml"
    )
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.04

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        self.WORK_SURFACE_TOP_Z = float(spec.get("work_surface_top_z", 0.75))
        self._setup_table_sized_work_surface(
            [
                spec["basket_pos"],
                *(pose["pos"] for pose in spec["object_poses"].values()),
            ]
        )
        self.success_xy_tolerance = float(spec["success_xy_tolerance"])
        self.HOLD_SECONDS = float(spec["hold_seconds"])

    def _get_obj_cfgs(self):
        spec = self._get_task_spec()
        return [
            dict(
                name="basket",
                obj_groups=self._asset_path(self.BASKET_XML),
                object_scale=float(spec["basket_scale"]),
            ),
            dict(
                name="candle",
                obj_groups=self._asset_path(self.CANDLE_XML),
                object_scale=float(spec["candle_scale"]),
            ),
        ]

    def _create_objects(self):
        super()._create_objects()
        statue = LivingRoomStatueObject(
            name="statue",
            source_xml=self._asset_path(self.STATUE_XML),
            scale=float(self._get_task_spec()["statue_scale"]),
        )
        self.objects[statue.name] = statue
        self.model.merge_objects([statue])

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualLivingRoomPackObjectsSampler")

    def _is_scene_valid(self):
        return bool(
            IThorLivingRoomEnv._is_scene_valid(self)
            and getattr(self, "work_surface_region", None) is not None
        )

    def _before_object_placement(self):
        super()._before_object_placement()
        spec = self._get_task_spec()
        set_free_joint_bottom_at(
            self,
            self.objects["basket"],
            self._work_surface_bottom_position(spec["basket_pos"]),
            yaw=spec["basket_yaw"],
        )
        for name, pose in spec["object_poses"].items():
            position = self._work_surface_bottom_position(pose["pos"])
            if name == "statue":
                set_free_joint_oriented_bottom_at(
                    self,
                    self.objects[name],
                    position,
                    yaw=pose["yaw"],
                    pitch=np.pi / 2.0,
                )
            else:
                set_free_joint_bottom_at(
                    self, self.objects[name], position, yaw=pose["yaw"]
                )
        self.sim.forward()

    def _constrain_initial_jitter_candidate(self):
        return None

    def _initial_jitter_allowed_contact_bodies(self, name):
        del name
        return set(self._fixture_body_ids(self.work_surface_fixture))

    def _initial_jitter_candidate_is_valid(self):
        return all(
            self.object_on_region(name, self.work_surface_region, require_contact=False)
            for name in ("basket", *self.ITEM_NAMES)
        )

    def _setup_scene(self):
        super()._setup_scene()
        self.HOLD_STEPS = max(1, int(round(self.HOLD_SECONDS * self.control_freq)))
        self._success_hold = 0
        self._packed_items = set()

    def _reset_internal(self):
        super()._reset_internal()
        self._success_hold = 0
        self._packed_items = set()

    def _item_is_packed(self, name):
        if name == "candle" and self.scene_name != "FloorPlan209":
            # FloorPlan209 keeps the evaluation contact-and-distance predicate.
            # A candle inside may rest on the statue and lose basket contact.
            # Use the actual basket volume instead of a root-distance cutoff.
            basket_id = self.obj_body_id["basket"]
            basket_pos = self.sim.data.body_xpos[basket_id]
            basket_rot = self.sim.data.body_xmat[basket_id].reshape(3, 3)
            center = np.mean(self.object_bbox_points(name), axis=0)
            local_center = (center - basket_pos) @ basket_rot
            bounds = np.asarray(self.objects["basket"].get_bbox_points())
            return bool(
                np.all(local_center >= bounds.min(axis=0))
                and np.all(local_center <= bounds.max(axis=0))
            )
        return OU.check_obj_in_receptacle(
            self, name, "basket", th=self.success_xy_tolerance
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        packed = {name for name in self.ITEM_NAMES if self._item_is_packed(name)}
        for name in self.ITEM_NAMES:
            if name not in packed or name in self._packed_items:
                continue
            self._packed_items.add(name)
            completed = len(self._packed_items)
            remaining = [
                item for item in self.ITEM_NAMES if item not in self._packed_items
            ]
            next_stage = f"put_{remaining[0]}_in_basket" if remaining else None
            self._report_stage_complete(completed, len(self.ITEM_NAMES), next_stage)
        valid = len(packed) == len(self.ITEM_NAMES)
        self._success_hold = self._success_hold + 1 if valid else 0
        return ret

    def _check_success(self):
        return self._success_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            lang="put the candle and statue into the basket",
            composite_stages=[
                "put the candle in the basket",
                "put the statue in the same basket",
            ],
            composite_ordering="any",
        )
        return ep_meta


__all__ = ["IThorLivingRoomPackCandleStatueIntoBasket"]
