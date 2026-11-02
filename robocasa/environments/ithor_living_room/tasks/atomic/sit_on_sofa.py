"""iTHOR living-room: turn around and sit on a sofa."""

import numpy as np
from robosuite.utils.mjcf_utils import new_site

from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)
from robocasa.models.scenes.thor_semantics import SemanticRegion


class IThorLivingRoomSitOnSofa(IThorLivingRoomEnv):
    """Turn away from the sofa and sit against its calibrated seat surface."""

    ATOMIC_SKILL = "Sit"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "locomotion"
    SUPPORTED_SCENE_IDS = (202, 207, 209, 210, 212, 226)
    SPEC_TASK = "sit_on_sofa"
    INITIAL_POSE_JITTER = False
    SUCCESS_REGION_SITE_NAME = "sit_on_sofa_success_region"

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
                    "target_fixture",
                    "robot_pose",
                    "seat_center_xy",
                    "seat_yaw",
                    "seat_half_xy",
                    "seat_surface_z",
                    "contact_z_tolerance",
                    "hold_seconds",
                    "show_success_region",
                ),
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

    def _setup_task_fixtures(self):
        spec = self._get_task_spec()
        self.sofa = self.get_fixture(id=spec["target_fixture"])
        self.fixture_refs["sofa"] = self.sofa
        self.seat_center_xy = np.asarray(spec["seat_center_xy"], dtype=float)
        self.seat_yaw = float(spec["seat_yaw"])
        self.seat_half_xy = np.asarray(spec["seat_half_xy"], dtype=float)
        self.seat_surface_z = float(spec["seat_surface_z"])
        self.contact_z_tolerance = float(spec["contact_z_tolerance"])
        hold_seconds = float(spec["hold_seconds"])
        self.HOLD_STEPS = int(round(hold_seconds * self.control_freq))
        if (
            self.seat_center_xy.shape != (2,)
            or self.seat_half_xy.shape != (2,)
            or np.any(self.seat_half_xy <= 0.0)
            or self.seat_surface_z <= 0.0
            or self.contact_z_tolerance <= 0.0
            or hold_seconds <= 0.0
            or self.HOLD_STEPS < 1
        ):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: invalid sitting calibration"
            )
        self.seat_region = SemanticRegion(
            stable_id=f"{self.sofa.stable_id}:region:sit_surface",
            fixture_id=self.sofa.stable_id,
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.array(
                [*self.seat_half_xy, self.contact_z_tolerance], dtype=float
            ),
            _world_center=np.array(
                [*self.seat_center_xy, self.seat_surface_z], dtype=float
            ),
            _world_yaw=self.seat_yaw,
        )
        self.sofa.regions[self.seat_region.stable_id] = self.seat_region
        self.region_refs["sofa_seat"] = self.seat_region
        if bool(spec["show_success_region"]):
            center, yaw = self.seat_region.world_pose()
            self.mujoco_arena.worldbody.append(
                new_site(
                    name=self.SUCCESS_REGION_SITE_NAME,
                    type="box",
                    pos=center,
                    size=self.seat_region.half_size,
                    quat=[np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
                    rgba=[0.1, 0.9, 0.2, 0.25],
                    group=0,
                )
            )
        super()._setup_task_fixtures()

    def _setup_scene(self):
        super()._setup_scene()
        self._sit_hold_steps = 0

    def _reset_internal(self):
        super()._reset_internal()
        self._sit_hold_steps = 0

    def _seat_contact_active(self):
        sitting_bodies = self._body_ids_matching(("pelvis", "hip"))
        sofa_bodies = self._fixture_body_ids(self.sofa, include_siblings=True)
        center, _ = self.seat_region.world_pose(self.sim)
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if not (
                (body1 in sitting_bodies and body2 in sofa_bodies)
                or (body2 in sitting_bodies and body1 in sofa_bodies)
            ):
                continue
            point = np.asarray(contact.pos, dtype=float)
            if (
                self.seat_region.contains_xy(point, self.sim)
                and abs(float(point[2]) - float(center[2])) <= self.contact_z_tolerance
            ):
                return True
        return False

    def _raw_success(self):
        return self._seat_contact_active()

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._sit_hold_steps = self._sit_hold_steps + 1 if self._raw_success() else 0
        return ret

    def _check_success(self):
        return self._sit_hold_steps >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "turn around from facing the sofa, then sit down on the sofa"
        ep_meta["success"] = {
            "seat_center_xy": self.seat_center_xy.tolist(),
            "seat_yaw": self.seat_yaw,
            "seat_half_xy": self.seat_half_xy.tolist(),
            "seat_surface_z": self.seat_surface_z,
            "contact_z_tolerance": self.contact_z_tolerance,
            "accepted_robot_bodies": ["pelvis", "hip"],
            "hold_steps": self.HOLD_STEPS,
            "requires_sofa_contact": True,
        }
        return ep_meta


__all__ = ["IThorLivingRoomSitOnSofa"]
