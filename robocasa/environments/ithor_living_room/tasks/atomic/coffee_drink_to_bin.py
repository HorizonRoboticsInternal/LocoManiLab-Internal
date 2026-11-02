"""iTHOR living-room FloorPlan209: throw a coffee drink into the bin."""

import numpy as np
from robosuite.utils.mjcf_utils import array_to_string, string_to_array
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomCoffeeDrinkToBin(IThorLivingRoomEnv):
    """Pick up a coffee drink and drop it into the FloorPlan209 bin."""

    ATOMIC_SKILL = "PlaceIntoReceptacle"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "receptacle"

    SOURCE_NAME = "coffee_drink"
    SOURCE_GROUP = "coffee_cup"
    SUPPORTED_SCENE_IDS = (202, 207, 209, 210, 212, 226)
    SPEC_TASK = "coffee_drink_to_bin"

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        self.bin_fixture = None
        self.bin_region = None

        self.bin_fixture = self.get_fixture(id=spec["bin_fixture"])
        if self.bin_fixture is None:
            self._scene_invalid_reason = f"{self.scene_name}: no bin"
            return
        self.bin_region = self.bin_fixture.get_region("receptacle", required=False)
        if self.bin_region is None:
            self._scene_invalid_reason = f"{self.scene_name}: bin has no receptacle"
            return
        self.bin_target_xy = np.asarray(spec["bin_target_xy"], dtype=float)
        if self.bin_target_xy.shape != (2,):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: bin_target_xy needs XY"
            )
        self.bin_xy_tolerance = float(spec["bin_xy_tolerance"])
        self.bin_z_tolerance = float(spec["bin_z_tolerance"])
        self.HOLD_SECONDS = float(spec["hold_seconds"])
        if min(self.bin_xy_tolerance, self.bin_z_tolerance, self.HOLD_SECONDS) <= 0:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: tolerances and hold_seconds "
                "must be positive"
            )
        self._move_bin_to_target_xy(self.bin_target_xy)

        self.fixture_refs["bin"] = self.bin_fixture
        self.region_refs["bin_receptacle"] = self.bin_region
        pose = spec["robot_pose"]
        self.set_robot_base_pose(pose["pos"], pose["yaw"])

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "bin_fixture",
                    "coffee_pos",
                    "coffee_yaw",
                    "bin_target_xy",
                    "bin_xy_tolerance",
                    "bin_z_tolerance",
                    "robot_pose",
                    "hold_seconds",
                ),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _setup_scene(self):
        super()._setup_scene()
        self._success_hold_steps = 0
        self._hold_steps_required = int(round(self.HOLD_SECONDS * self.control_freq))

    def _move_bin_to_target_xy(self, target_xy):
        if self.bin_fixture is None or self.bin_fixture.body_name is None:
            raise RuntimeError(f"{self.scene_name}: selected bin has no root body")
        body = self.find_scene_body(self.bin_fixture.body_name)
        if body is None:
            raise RuntimeError(
                f"{self.scene_name}: missing bin body {self.bin_fixture.body_name!r}"
            )

        old_pos = string_to_array(body.get("pos", "0 0 0")).astype(float)
        target_pos = old_pos.copy()
        target_pos[:2] = target_xy
        body.set("pos", array_to_string(target_pos))

        delta = target_pos - old_pos
        self.bin_fixture.center = (
            np.asarray(self.bin_fixture.center, dtype=float) + delta
        )
        for region in self.bin_fixture.regions.values():
            region._world_center = np.asarray(region._world_center, dtype=float) + delta

    def _is_scene_valid(self):
        return bool(super()._is_scene_valid() and self.bin_region is not None)

    def _get_obj_cfgs(self):
        return [
            dict(
                name=self.SOURCE_NAME,
                obj_groups=self.SOURCE_GROUP,
            )
        ]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualCoffeeDrinkToBinSampler")

    def _reset_internal(self):
        super()._reset_internal()
        if self.SOURCE_NAME in self.objects:
            self._place_source_object()

    def _place_source_object(self):
        set_free_joint_bottom_at(
            self,
            self.objects[self.SOURCE_NAME],
            self._get_task_spec()["coffee_pos"],
            yaw=self._get_task_spec()["coffee_yaw"],
        )
        self.sim.forward()

    def _raw_success(self):
        return bool(
            self.object_in_region(
                self.SOURCE_NAME,
                self.bin_region,
                rim_tol=self.bin_z_tolerance,
                xy_tol=self.bin_xy_tolerance,
            )
            and self.object_released(self.SOURCE_NAME)
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return ret

    def _check_success(self):
        return self._success_hold_steps >= self._hold_steps_required

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        bin_center, _ = self.bin_region.world_pose(self.sim)
        ep_meta.update(
            bin_receptacle=[float(v) for v in bin_center],
            lang="pick up the coffee drink, approach the bin, and throw it into the bin",
        )
        return ep_meta


__all__ = ["IThorLivingRoomCoffeeDrinkToBin"]
