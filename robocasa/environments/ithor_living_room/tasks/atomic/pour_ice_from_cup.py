"""Pour one ice cube out of a cup on a raised living-room work surface."""

import os
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.environments.ithor_living_room.tasks.atomic.raised_work_surface import (
    TableSizedRaisedSurfaceMixin,
)
from robocasa.models.scenes.thor_semantics import SemanticRegion
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomPourIceFromCup(TableSizedRaisedSurfaceMixin, IThorLivingRoomEnv):
    """Pick up a cup and pour its single ice cube onto the surrounding surface."""

    ATOMIC_SKILL = "PourContents"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "reorientation"
    SPEC_TASK = "pour_ice_from_cup"
    SUPPORTED_SCENE_IDS = (202, 207, 209, 210, 212, 226)
    CUP_NAME = "open_tapered_cup"
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    ICE_NAME = "ice_cube"
    INITIAL_OBJECT_JITTER_RADIUS = 0.04

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _asset_path(self, relative_path):
        path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", relative_path)
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing pour-ice asset: {path}")
        return path

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

    def _select_support_region(self):
        calibrated = self._get_task_spec().get("success_region")
        if calibrated is not None:
            fixture = self.get_fixture(id=self._get_task_spec()["support_fixture"])
            center = np.asarray(calibrated["center"], dtype=float)
            half_xy = np.asarray(calibrated["half_xy"], dtype=float)
            if fixture is None or center.shape != (3,) or half_xy.shape != (2,):
                raise ValueError(
                    f"{self.scene_name}/{self.SPEC_TASK}: invalid success_region"
                )
            region = SemanticRegion(
                stable_id=f"{fixture.stable_id}:task:{self.SPEC_TASK}:success",
                fixture_id=fixture.stable_id,
                kind="support",
                parent_body=None,
                local_center=np.zeros(3),
                half_size=np.array([half_xy[0], half_xy[1], 0.02]),
                _world_center=center,
                _world_yaw=float(calibrated.get("yaw", 0.0)),
            )
            fixture.regions[region.stable_id] = region
            return region, fixture

        cup_pos = np.asarray(self._get_task_spec()["cup_pos"], dtype=float)
        requested_fixture = self._get_task_spec().get("support_fixture")
        candidates = []
        for region in self.get_region(kind="support", return_all=True):
            fixture = self.get_fixture(id=region.fixture_id)
            if fixture is None:
                continue
            if requested_fixture and fixture.stable_id != requested_fixture:
                continue
            center, yaw = region.world_pose()
            c, s = np.cos(-yaw), np.sin(-yaw)
            delta = cup_pos[:2] - center[:2]
            local = np.array([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]])
            if np.any(np.abs(local) > region.half_size[:2] - 0.06):
                continue
            z_error = abs(float(center[2]) - float(cup_pos[2]))
            if z_error > 0.08:
                continue
            candidates.append((z_error, region.stable_id, region, fixture))
        if not candidates:
            return None, None
        _, _, region, fixture = min(candidates)
        return region, fixture

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and getattr(self, "support_region", None) is not None
        )

    def _check_success(self):
        return self._success_hold_steps >= self._hold_steps_required

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "robot_pose",
                    "support_fixture",
                    "cup_mjcf",
                    "cup_scale",
                    "cup_pos",
                    "cup_yaw",
                    "ice_mjcf",
                    "ice_scale",
                    "ice_local_pos",
                    "work_surface_top_z",
                    "ice_region_margin",
                    "hold_seconds",
                ),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _setup_task_fixtures(self):
        spec = self._get_task_spec()
        self.support_region, self.support_fixture = self._select_support_region()
        if self.support_region is None:
            self._scene_invalid_reason = (
                f"{self.scene_name}/{self.SPEC_TASK}: no support under cup"
            )
            return
        self.base_support_region = self.support_region
        self.base_support_fixture = self.support_fixture
        self.support_region, self.support_fixture = self._add_table_sized_work_surface(
            self.base_support_region,
            top_z=float(spec["work_surface_top_z"]),
        )
        self.work_surface_region = self.support_region
        self.work_surface_fixture = self.support_fixture
        self.region_refs["cup_support"] = self.support_region
        self.fixture_refs["cup_support"] = self.support_fixture

        self.ICE_REGION_MARGIN = float(spec["ice_region_margin"])
        self.HOLD_SECONDS = float(spec["hold_seconds"])
        if self.ICE_REGION_MARGIN < 0.0 or self.HOLD_SECONDS <= 0.0:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: invalid success threshold"
            )

    def _get_obj_cfgs(self):
        spec = self._get_task_spec()
        return [
            dict(
                name=self.CUP_NAME,
                info=dict(mjcf_path=self._asset_path(spec["cup_mjcf"])),
                object_scale=float(spec["cup_scale"]),
            ),
            dict(
                name=self.ICE_NAME,
                info=dict(mjcf_path=self._asset_path(spec["ice_mjcf"])),
                object_scale=float(spec["ice_scale"]),
            ),
        ]

    def _create_objects(self):
        super()._create_objects()
        cup = self.objects[self.CUP_NAME]
        interior = cup._regions["int"]["elem"]
        interior_pos = np.fromstring(interior.get("pos"), sep=" ")
        interior_size = np.fromstring(interior.get("size"), sep=" ")
        floor_half_height = min(0.005, 0.15 * float(interior_size[1]))
        floor_center_z = (
            float(interior_pos[2]) - float(interior_size[1]) + floor_half_height
        )
        ET.SubElement(
            cup.get_obj(),
            "geom",
            name=f"{cup.naming_prefix}inner_floor",
            type="cylinder",
            pos=f"0 0 {floor_center_z:.8g}",
            size=f"{0.90 * interior_size[0]:.8g} {floor_half_height:.8g}",
            rgba="0 0 0 0",
            density="100",
            friction="0.95 0.3 0.1",
            contype="1",
            conaffinity="1",
            group="0",
        )

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualLivingRoomPourIceSampler")

    def _place_cup_and_ice_from_spec(self):
        spec = self._get_task_spec()
        support_center, _ = self.support_region.world_pose(self.sim)
        set_free_joint_bottom_at(
            self,
            self.objects[self.CUP_NAME],
            [spec["cup_pos"][0], spec["cup_pos"][1], support_center[2]],
            yaw=spec["cup_yaw"],
        )
        self.sim.forward()

        self._place_ice_in_cup()
        for obj in (self.objects[self.CUP_NAME], self.objects[self.ICE_NAME]):
            qvel_addr = self.sim.model.get_joint_qvel_addr(obj.joints[0])
            if isinstance(qvel_addr, tuple):
                self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = 0.0

    def _place_ice_in_cup(self):
        spec = self._get_task_spec()

        cup_body_id = self.obj_body_id[self.CUP_NAME]
        cup_pos = np.asarray(self.sim.data.body_xpos[cup_body_id], dtype=float)
        cup_rot = np.asarray(self.sim.data.body_xmat[cup_body_id], dtype=float).reshape(
            3, 3
        )
        ice_local_pos = np.asarray(spec["ice_local_pos"], dtype=float)
        if ice_local_pos.shape != (3,):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: ice_local_pos needs XYZ"
            )
        ice = self.objects[self.ICE_NAME]
        ice_qpos = self.sim.data.get_joint_qpos(ice.joints[0]).copy()
        ice_qpos[:3] = cup_pos + cup_rot @ ice_local_pos
        ice_qpos[3:7] = np.asarray(self.sim.data.body_xquat[cup_body_id], dtype=float)
        self.sim.data.set_joint_qpos(ice.joints[0], ice_qpos)
        self.sim.forward()

    def _before_object_placement(self):
        self._place_cup_and_ice_from_spec()

    def _constrain_initial_jitter_candidate(self):
        """Keep the ice at its calibrated cup-local pose while the cup jitters."""
        self._place_ice_in_cup()

    def _initial_jitter_allowed_contact_bodies(self, name):
        paired_name = {
            self.CUP_NAME: self.ICE_NAME,
            self.ICE_NAME: self.CUP_NAME,
        }.get(name)
        if paired_name is None:
            return set()
        return self._body_tree_ids([self.obj_body_id[paired_name]])

    def _setup_scene(self):
        self._success_hold_steps = 0
        self._hold_steps_required = max(
            1, int(round(self.HOLD_SECONDS * self.control_freq))
        )

    def _cup_interior_geom_id(self):
        prefix = self.objects[self.CUP_NAME].naming_prefix
        candidates = [
            geom_id
            for geom_id in range(self.sim.model.ngeom)
            if (self.sim.model.geom_id2name(geom_id) or "").startswith(prefix)
            and (self.sim.model.geom_id2name(geom_id) or "").endswith("reg_int")
        ]
        if len(candidates) != 1:
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: expected one cup interior, "
                f"found {len(candidates)}"
            )
        return candidates[0]

    def _ice_fully_outside_cup(self):
        geom_id = self._cup_interior_geom_id()
        center = np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float)
        rotation = np.asarray(self.sim.data.geom_xmat[geom_id], dtype=float).reshape(
            3, 3
        )
        size = np.asarray(self.sim.model.geom_size[geom_id], dtype=float)
        local_points = (self.object_bbox_points(self.ICE_NAME) - center) @ rotation
        margin = self.ICE_REGION_MARGIN
        points_inside = np.logical_and(
            np.linalg.norm(local_points[:, :2], axis=1) <= size[0] + margin,
            np.abs(local_points[:, 2]) <= size[1] + margin,
        )
        return not bool(np.any(points_inside))

    def _raw_success(self):
        return self._ice_fully_outside_cup()

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return ret

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            lang="pick up the cup and pour its ice cube out onto the surface",
            target_object=self.CUP_NAME,
            contained_object=self.ICE_NAME,
            target_asset=self._get_task_spec()["cup_mjcf"],
            success_definition=(
                "the entire ice cube must remain outside the cup for 2 seconds"
            ),
        )
        return ep_meta


__all__ = ["IThorLivingRoomPourIceFromCup"]
