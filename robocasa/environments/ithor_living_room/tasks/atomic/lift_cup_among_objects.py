"""Lift a target cup from a five-object living-room table layout."""

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.lift import RoomLiftMixin
from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
)
from robocasa.environments.ithor_living_room.tasks.atomic.raised_work_surface import (
    TableSizedRaisedSurfaceMixin,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomLiftCupAmongObjects(
    TableSizedRaisedSurfaceMixin, RoomLiftMixin, IThorLivingRoomEnv
):
    """Lift the cup while four other objects act only as distractors."""

    TARGET_NAME = "target_cup"
    DISTRACTOR_GROUPS = {
        "distractor_apple": "apple",
        "distractor_can": "can",
        "distractor_egg": "egg",
        "distractor_candle": "candle",
    }
    SUPPORTED_SCENE_IDS = (202, 207, 209, 210, 212, 226)
    SPEC_TASK = "lift_cup_among_objects"
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.04

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
                    "robot_pose",
                    "object_poses",
                    "lift_height",
                    "hold_seconds",
                ),
            )
            expected = {self.TARGET_NAME, *self.DISTRACTOR_GROUPS}
            actual = set(self.task_spec["object_poses"])
            if actual != expected:
                raise ValueError(
                    f"{self.scene_name}/{self.SPEC_TASK}: object_poses must contain "
                    f"exactly {sorted(expected)}, got {sorted(actual)}"
                )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _get_scene_surface_calibration(self):
        return None

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        if pos.shape != (3,) or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: robot_pose needs XYZ + yaw"
            )
        return pos, float(pose["yaw"])

    def _place_robot_at_target(self):
        return None

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        self.MIN_LIFT_DELTA = float(spec["lift_height"])
        self.HOLD_SECONDS = float(spec["hold_seconds"])
        self.INITIAL_OBJECT_JITTER_RADIUS = float(
            spec.get("object_jitter_radius", self.INITIAL_OBJECT_JITTER_RADIUS)
        )
        if min(self.MIN_LIFT_DELTA, self.HOLD_SECONDS) <= 0.0:
            raise ValueError(f"{self.scene_name}/{self.SPEC_TASK}: invalid threshold")
        if self.INITIAL_OBJECT_JITTER_RADIUS < 0.0:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: invalid object jitter"
            )
        self._setup_table_sized_work_surface(
            [spec["object_poses"][self.TARGET_NAME]["pos"]]
        )

    def _get_obj_cfgs(self):
        configs = [dict(name=self.TARGET_NAME, obj_groups="coffee_cup")]
        configs.extend(
            dict(name=name, obj_groups=group)
            for name, group in self.DISTRACTOR_GROUPS.items()
        )
        return configs

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualLiftCupAmongObjectsSampler")

    def _is_scene_valid(self):
        return bool(
            IThorLivingRoomEnv._is_scene_valid(self)
            and getattr(self, "work_surface_region", None) is not None
        )

    def _place_objects_from_spec(self):
        for name, pose in self._get_task_spec()["object_poses"].items():
            set_free_joint_bottom_at(
                self,
                self.objects[name],
                self._work_surface_bottom_position(pose["pos"]),
                yaw=pose["yaw"],
            )
        self.sim.forward()

    def _before_object_placement(self):
        super()._before_object_placement()
        self._place_objects_from_spec()

    def _constrain_initial_jitter_candidate(self):
        # The generic jitter validator rejects collisions and penetrations.
        # Keep the accepted XY perturbation instead of restoring the YAML pose.
        return None

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        support = "support surface" if self.scene_name == "FloorPlan210" else "table"
        ep_meta.update(
            lang=f"lift the cup from the {support} with five objects",
            target_object=self.TARGET_NAME,
            distractor_objects=list(self.DISTRACTOR_GROUPS),
        )
        return ep_meta


__all__ = ["IThorLivingRoomLiftCupAmongObjects"]
