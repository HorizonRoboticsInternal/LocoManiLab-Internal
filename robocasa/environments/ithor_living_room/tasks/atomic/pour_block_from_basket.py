"""Tip a basket to pour out the same block used by the pushing task."""

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.pick_place import (
    SupportContactPlacementMixin,
)
from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
)
from robocasa.environments.ithor_living_room.tasks.atomic.object_lift import (
    FixedObjectLiftMixin,
)
from robocasa.environments.ithor_living_room.tasks.atomic.push_block_to_target_region import (
    IThorLivingRoomPushBlockToTargetRegion,
)
from robocasa.environments.ithor_living_room.tasks.atomic.raised_work_surface import (
    TableSizedRaisedSurfaceMixin,
)
from robocasa.environments.ithor_living_room.tasks.atomic.remote_to_basket import (
    IThorLivingRoomRemoteToBasket,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomPourBlockFromBasket(
    SupportContactPlacementMixin,
    TableSizedRaisedSurfaceMixin,
    FixedObjectLiftMixin,
    IThorLivingRoomEnv,
):
    """Pour one block out by tilting the basket, then keep it outside for 1 s."""

    ATOMIC_SKILL = "PourContents"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "reorientation"
    SPEC_TASK = "pour_block_from_basket"
    TARGET_NAME = "block"
    TARGET_XML = IThorLivingRoomPushBlockToTargetRegion.TARGET_XML
    BASKET_NAME = "basket"
    BASKET_XML = IThorLivingRoomRemoteToBasket.BASKET_XML
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    OUTSIDE_MARGIN = 0.003

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "robot_pose", "work_surface_support", "work_surface_top_z",
                    "basket_pos", "basket_yaw", "hold_seconds",
                ),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _get_object_adapter(self):
        # Reuse the pushing task's complete adapter, including scale and bbox.
        return get_ithor_livingroom_task_spec(
            self.scene_name, IThorLivingRoomPushBlockToTargetRegion.SPEC_TASK
        )

    def _get_obj_cfgs(self):
        return [
            dict(name=self.BASKET_NAME, obj_groups=self._asset_path(self.BASKET_XML)),
            *super()._get_obj_cfgs(),
        ]

    def _setup_task_fixtures(self):
        IThorLivingRoomEnv._setup_task_fixtures(self)
        spec = self._get_task_spec()
        self.WORK_SURFACE_TOP_Z = float(spec["work_surface_top_z"])
        self.HOLD_SECONDS = float(spec["hold_seconds"])
        if self.HOLD_SECONDS <= 0.0:
            raise ValueError(f"{self.scene_name}/{self.SPEC_TASK}: invalid hold time")
        self._setup_table_sized_work_surface([spec["basket_pos"]])
        self.fixture_refs["basket_support"] = self.work_surface_fixture
        self.region_refs["basket_support"] = self.work_surface_region

    def _is_scene_valid(self):
        return bool(
            IThorLivingRoomEnv._is_scene_valid(self)
            and getattr(self, "work_surface_region", None) is not None
        )

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualPourBlockFromBasketSampler")

    def _block_basket_contact_distances(self):
        block_bodies = self._body_tree_ids([self.obj_body_id[self.TARGET_NAME]])
        basket_bodies = self._body_tree_ids([self.obj_body_id[self.BASKET_NAME]])
        distances = []
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            a = int(self.sim.model.geom_bodyid[contact.geom1])
            b = int(self.sim.model.geom_bodyid[contact.geom2])
            if (
                (a in block_bodies and b in basket_bodies)
                or (b in block_bodies and a in basket_bodies)
            ):
                distances.append(float(contact.dist))
        return distances

    def _before_object_placement(self):
        spec = self._get_task_spec()
        set_free_joint_bottom_at(
            self, self.objects[self.BASKET_NAME], spec["basket_pos"],
            yaw=spec["basket_yaw"],
        )
        self.sim.forward()
        if not self.place_object_on_support_contact(
            self.BASKET_NAME, self.work_surface_region, self.work_surface_fixture,
            xy=spec["basket_pos"][:2], inward_fractions=(0.0,),
        ):
            raise RuntimeError(f"{self.scene_name}: basket has no shallow table contact")

        basket_pos = self.object_pos(self.BASKET_NAME)
        # Probe the real basket floor, not its enclosing bbox or a settle delay.
        for bottom_z in np.arange(
            basket_pos[2] + 0.06, basket_pos[2] - 0.005, -0.001
        ):
            set_free_joint_bottom_at(
                self, self.objects[self.TARGET_NAME], [*basket_pos[:2], bottom_z],
                yaw=spec["basket_yaw"],
            )
            self.sim.forward()
            distances = self._block_basket_contact_distances()
            if distances:
                if min(distances) < -0.0025 or not self._block_fully_inside_basket():
                    raise RuntimeError(
                        f"{self.scene_name}: block penetrates or exceeds basket"
                    )
                break
        else:
            raise RuntimeError(f"{self.scene_name}: block has no basket-floor contact")

    def _constrain_initial_jitter_candidate(self):
        return None

    def _reset_internal(self):
        IThorLivingRoomEnv._reset_internal(self)

    def _setup_scene(self):
        IThorLivingRoomEnv._setup_scene(self)
        self._success_hold_steps = 0
        self._hold_steps_required = max(
            1, int(round(self.HOLD_SECONDS * self.control_freq))
        )

    def _block_bounds_in_basket(self):
        body_id = self.obj_body_id[self.BASKET_NAME]
        center = np.asarray(self.sim.data.body_xpos[body_id])
        rotation = np.asarray(self.sim.data.body_xmat[body_id]).reshape(3, 3)
        block = (self.object_bbox_points(self.TARGET_NAME) - center) @ rotation
        basket = (self.object_bbox_points(self.BASKET_NAME) - center) @ rotation
        return (
            block.min(axis=0), block.max(axis=0),
            basket.min(axis=0), basket.max(axis=0),
        )

    def _block_fully_inside_basket(self):
        lower, upper, basket_lower, basket_upper = self._block_bounds_in_basket()
        return bool(np.all(lower >= basket_lower) and np.all(upper <= basket_upper))

    def _block_fully_outside_basket(self):
        lower, upper, basket_lower, basket_upper = self._block_bounds_in_basket()
        # A separating axis rejects partial overlap, including enclosing the rim.
        return bool(
            np.any(upper < basket_lower - self.OUTSIDE_MARGIN)
            or np.any(lower > basket_upper + self.OUTSIDE_MARGIN)
        )

    def _raw_success(self):
        return self._block_fully_outside_basket()

    def _post_action(self, action):
        ret = IThorLivingRoomEnv._post_action(self, action)
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return ret

    def _check_success(self):
        return self._success_hold_steps >= self._hold_steps_required

    def get_ep_meta(self):
        meta = IThorLivingRoomEnv.get_ep_meta(self)
        meta.update(
            lang="pick up the basket and tilt it to pour the block out",
            target_object=self.BASKET_NAME,
            contained_object=self.TARGET_NAME,
            target_asset=self.BASKET_XML,
            contained_asset=self.TARGET_XML,
            contained_object_scale=float(self._get_object_adapter()["source_scale"]),
            success_hold_seconds=self.HOLD_SECONDS,
            success_definition=(
                "the entire block must remain outside the basket "
                f"for {self.HOLD_SECONDS:g} seconds"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomPourBlockFromBasket"]
