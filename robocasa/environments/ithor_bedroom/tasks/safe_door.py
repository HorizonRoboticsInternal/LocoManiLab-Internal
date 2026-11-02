"""Fixed-base safe-door tasks for calibrated bedroom scenes."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_specs,
    get_supported_scene_ids,
)

from .articulation_base import (
    BedroomArticulationMixin,
    BedroomSupportedFixtureMixin,
    BedroomTwentyEightyArticulationMixin,
)


class _FixedSafeMixin(BedroomSupportedFixtureMixin):
    """Make the safe movable for seating, then weld its cabinet to world."""

    SAFE_BASE_WELD_NAME = "task_safe_fixed_base"
    SAFE_PRE_SNAP_FREE_JOINT_Z = 0.0
    SAFE_SPAWN_RETREAT_OFFSETS = (
        0.025,
        0.05,
        0.075,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        fixture = self.target_fixture
        safe_body = self.find_scene_body(fixture.body_name)
        if safe_body is None:
            raise RuntimeError(
                f"{self.scene_name}: missing safe root body {fixture.body_name}"
            )
        self._ensure_fixture_free_joint(
            fixture, f"task_safe_free_joint_{self.scene_name}"
        )
        ET.SubElement(
            self.mujoco_arena.equality,
            "weld",
            name=self.SAFE_BASE_WELD_NAME,
            body1=fixture.body_name,
            active="false",
            solref="0.002 1",
            solimp="0.9999 0.9999 0.001 0.5 2",
        )

        safe_pos = np.fromstring(safe_body.get("pos", "0 0 0"), sep=" ")
        if safe_pos.size != 3:
            raise RuntimeError(
                f"{self.scene_name}: invalid safe body position {safe_body.get('pos')}"
            )
        desired_xy = np.asarray(spec["safe_xy"], dtype=float)
        if desired_xy.shape != (2,):
            raise RuntimeError(
                f"{self.scene_name}: invalid calibrated safe XY {desired_xy}"
            )
        safe_pos[:2] += desired_xy - np.asarray(fixture.center[:2], dtype=float)
        safe_body.set("pos", " ".join(f"{value:.8g}" for value in safe_pos))
        yaw_delta = float(spec.get("safe_yaw_delta", 0.0))
        safe_quat = np.fromstring(safe_body.get("quat", "1 0 0 0"), sep=" ")
        if safe_quat.size != 4 or not np.isfinite(yaw_delta):
            raise RuntimeError(
                f"{self.scene_name}: invalid calibrated safe orientation"
            )
        yaw_quat = np.array(
            [np.cos(yaw_delta / 2.0), 0.0, 0.0, np.sin(yaw_delta / 2.0)],
            dtype=float,
        )
        w1, x1, y1, z1 = yaw_quat
        w2, x2, y2, z2 = safe_quat
        rotated_quat = np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )
        rotated_quat /= np.linalg.norm(rotated_quat)
        safe_body.set(
            "quat", " ".join(f"{value:.8g}" for value in rotated_quat)
        )
        self._safe_xy = desired_xy

    def _setup_scene(self):
        self._set_safe_base_weld_active(False)
        super()._setup_scene()

    def _safe_base_weld_id(self):
        equality_id = mujoco.mj_name2id(
            self.sim.model._model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            self.SAFE_BASE_WELD_NAME,
        )
        if equality_id < 0:
            raise RuntimeError(
                f"{self.scene_name}: missing safe base weld constraint"
            )
        return equality_id

    def _set_safe_base_weld_active(self, active):
        self.sim.data.eq_active[self._safe_base_weld_id()] = int(bool(active))
        self.sim.forward()

    def _lock_safe_base_at_current_pose(self):
        """Weld the safe body to world while leaving its door hinge movable."""

        equality_id = self._safe_base_weld_id()
        body_id = self.sim.model.body_name2id(self.target_fixture.body_name)
        position = np.asarray(self.sim.data.body_xpos[body_id], dtype=float).copy()
        rotation = np.asarray(
            self.sim.data.body_xmat[body_id], dtype=float
        ).reshape(3, 3)
        quaternion = np.asarray(
            self.sim.data.body_xquat[body_id], dtype=float
        ).copy()
        self.sim.model.eq_data[equality_id, 3:6] = -rotation.T @ position
        self.sim.model.eq_data[equality_id, 6:10] = np.array(
            [quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3]],
            dtype=float,
        )
        root_joint = self._fixture_free_joint_name(self.target_fixture)
        joint_id = self.sim.model.joint_name2id(root_joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[dof_address : dof_address + 6] = 0.0
        self.sim.data.eq_active[equality_id] = 1
        self.sim.forward()
        self._fixed_safe_base_position = position

    def _safe_robot_clearance_is_valid(self, minimum):
        clearance = self.robot_fixture_collision_distance(
            self.target_fixture,
            max_distance=float(minimum) + 0.25,
        )
        valid = bool(np.isfinite(clearance) and clearance >= float(minimum))
        if valid:
            self._safe_robot_initial_clearance = float(clearance)
        return valid

    def _finalize_supported_articulation_robot_pose(self, distance):
        """Restore G1's nominal pre-snap height after XY pose selection."""

        # The articulation pose search caches its collision-free free-joint
        # height. The final ground snap intentionally replaces that height,
        # so carrying the cache into a later reset can reject every candidate.
        joint = self._get_robot_free_joint()
        if joint is None:
            return False
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[2] = self.SAFE_PRE_SNAP_FREE_JOINT_Z
        self.sim.data.set_joint_qpos(joint, qpos)
        self._robot_default_free_joint_z = self.SAFE_PRE_SNAP_FREE_JOINT_Z
        if hasattr(self, "_robot_free_joint_z"):
            del self._robot_free_joint_z
        self.sim.forward()

        spec = self.scene_task_spec()
        min_clearance = float(spec["safe_robot_min_clearance"])
        if min_clearance <= 0.0:
            raise RuntimeError(
                f"{self.scene_name}: invalid safe robot clearance "
                f"{min_clearance}"
            )
        calibrated_xy = np.asarray(spec["safe_robot_xy"], dtype=float)
        if calibrated_xy.shape != (2,):
            raise RuntimeError(
                f"{self.scene_name}: invalid calibrated safe robot XY "
                f"{calibrated_xy}"
            )
        face_xy = np.asarray(self.target_face_pos[:2], dtype=float)
        yaw = float(
            np.arctan2(
                face_xy[1] - calibrated_xy[1],
                face_xy[0] - calibrated_xy[0],
            )
        )
        target_xy = np.asarray(self.target_joint_pos[:2], dtype=float)
        away = calibrated_xy - target_xy
        norm = float(np.linalg.norm(away))
        if norm <= 1e-8:
            return False
        away /= norm
        placed = False
        for offset in (0.0,) + self.SAFE_SPAWN_RETREAT_OFFSETS:
            candidate_xy = calibrated_xy + away * float(offset)
            candidate_qpos = self.sim.data.get_joint_qpos(joint).copy()
            candidate_qpos[:2] = candidate_xy
            candidate_qpos[2] = self.SAFE_PRE_SNAP_FREE_JOINT_Z
            candidate_qpos[3:7] = [
                np.cos(yaw / 2.0),
                0.0,
                0.0,
                np.sin(yaw / 2.0),
            ]
            self.sim.data.set_joint_qpos(joint, candidate_qpos)
            self._robot_default_free_joint_z = self.SAFE_PRE_SNAP_FREE_JOINT_Z
            if hasattr(self, "_robot_free_joint_z"):
                del self._robot_free_joint_z
            self.sim.forward()
            calibrated = self._calibrate_robot_free_joint_height()
            if calibrated and self._safe_robot_clearance_is_valid(min_clearance):
                self.robot_base_pos = np.array(
                    [candidate_xy[0], candidate_xy[1], 0.0]
                )
                self.robot_base_ori = np.array([0.0, 0.0, yaw])
                placed = True
                break
        if not placed:
            return False
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[2] = self.SAFE_PRE_SNAP_FREE_JOINT_Z
        target_xy = np.asarray(self.target_joint_pos[:2], dtype=float)
        away = np.asarray(qpos[:2], dtype=float) - target_xy
        norm = float(np.linalg.norm(away))
        if norm > 1e-8:
            away /= norm
        self.sim.data.set_joint_qpos(joint, qpos)
        joint_id = self.sim.model.joint_name2id(joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[dof_address : dof_address + 6] = 0.0
        self.sim.forward()
        self._robot_free_joint_z = float(qpos[2])
        if (
            self.count_body_collisions() == 0
            and self._safe_robot_clearance_is_valid(min_clearance)
        ):
            return True

        if norm <= 1e-8:
            return False
        original_xy = np.asarray(qpos[:2], dtype=float).copy()
        for offset in self.SAFE_SPAWN_RETREAT_OFFSETS:
            qpos[:2] = original_xy + away * float(offset)
            self.sim.data.set_joint_qpos(joint, qpos)
            self.sim.forward()
            if (
                self.count_body_collisions() == 0
                and self._safe_robot_clearance_is_valid(min_clearance)
            ):
                self.robot_base_pos = np.array([qpos[0], qpos[1], 0.0])
                return True
        return False

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["safe_robot_initial_clearance"] = float(
            self._safe_robot_initial_clearance
        )
        ep_meta["safe_base_fixed"] = True
        ep_meta["fixed_safe_base_position"] = (
            self._fixed_safe_base_position.tolist()
        )
        return ep_meta


class _RaisedSafeMixin(_FixedSafeMixin):
    """Place the task safe on a collidable support at a reachable height."""

    SAFE_PEDESTAL_RGBA = "0.34 0.27 0.20 1"

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        z_offset = float(spec["safe_z_offset"])
        safe_body = self.find_scene_body(self.target_fixture.body_name)
        if safe_body is None:
            raise RuntimeError(
                f"{self.scene_name}: missing safe root body "
                f"{self.target_fixture.body_name}"
            )
        safe_pos = np.fromstring(safe_body.get("pos", "0 0 0"), sep=" ")
        if safe_pos.size != 3:
            raise RuntimeError(
                f"{self.scene_name}: invalid safe body position "
                f"{safe_body.get('pos')}"
            )
        safe_pos[2] += z_offset
        safe_body.set("pos", " ".join(f"{value:.8g}" for value in safe_pos))

        # The source safes have a small non-yaw rotation. Use the calibrated
        # world AABB rather than the local Z bounds so the pedestal top touches
        # the actual lower extent after translation.
        support_height = float(
            self.target_fixture.center[2]
            + z_offset
            - self.target_fixture.half_size[2]
        )
        if support_height <= 0.04:
            raise RuntimeError(
                f"{self.scene_name}: calibrated safe support is too low "
                f"({support_height:.3f} m)"
            )

        pedestal_name = f"task_safe_pedestal_{self.scene_name}"
        if self.find_scene_body(pedestal_name) is None:
            pedestal = ET.SubElement(
                self.mujoco_arena.worldbody,
                "body",
                name=pedestal_name,
                pos=(
                    f"{self._safe_xy[0]:.8g} "
                    f"{self._safe_xy[1]:.8g} "
                    f"{support_height / 2.0:.8g}"
                ),
            )
            pedestal_half_xy = np.asarray(
                spec["safe_pedestal_half_xy"],
                dtype=float,
            )
            if pedestal_half_xy.shape != (2,) or np.any(pedestal_half_xy <= 0.0):
                raise RuntimeError(
                    f"{self.scene_name}: invalid safe pedestal footprint "
                    f"{pedestal_half_xy}"
                )
            ET.SubElement(
                pedestal,
                "geom",
                name=f"{pedestal_name}_geom",
                type="box",
                size=(
                    f"{pedestal_half_xy[0]:.8g} "
                    f"{pedestal_half_xy[1]:.8g} "
                    f"{support_height / 2.0:.8g}"
                ),
                rgba=self.SAFE_PEDESTAL_RGBA,
                friction="1 0.005 0.0001",
                contype="1",
                conaffinity="1",
            )

        self._safe_pedestal_name = pedestal_name
        self._safe_pedestal_height = support_height
        self._safe_pedestal_xy = self._safe_xy

    def _place_supported_articulation_fixture(self):
        if not self._place_fixture_on_support_contact(
            self.target_fixture,
            (self._safe_pedestal_name,),
            self._safe_pedestal_height,
            probe_below=float(self.scene_task_spec()["safe_support_probe_below"]),
        ):
            raise RuntimeError(
                f"{self.scene_name}: safe could not contact its pedestal"
            )
        self._lock_safe_base_at_current_pose()

    def _supported_fixture_contact_contract(self):
        return (
            "safe_pedestal",
            tuple(self.target_fixture.body_names),
            (self._safe_pedestal_name,),
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["safe_pedestal"] = self._safe_pedestal_name
        ep_meta["safe_pedestal_height"] = float(self._safe_pedestal_height)
        ep_meta["safe_pedestal_xy"] = self._safe_pedestal_xy.tolist()
        return ep_meta


class _BedsideSafeMixin(_FixedSafeMixin):
    """Seat the safe on an exact bedside table before fixing its base."""

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        support_id = spec["safe_support_fixture"]
        support = self.get_fixture(id=support_id)
        if support is None or support.stable_id != support_id:
            raise LookupError(
                f"{self.scene_name}: missing safe support {support_id}"
            )
        if support.stable_id == self.target_fixture.stable_id:
            raise ValueError(
                f"{self.scene_name}: safe cannot support itself"
            )
        region_id = spec["safe_support_region"]
        regions = [
            region
            for region in support.regions.values()
            if region.stable_id == region_id and region.kind == "support"
        ]
        if len(regions) != 1:
            raise LookupError(
                f"{self.scene_name}: missing exact safe support region "
                f"{region_id}"
            )
        self._safe_support_fixture = support
        self._safe_support_region = regions[0]

        removal_ids = spec["remove_fixtures"]
        if not isinstance(removal_ids, list) or len(removal_ids) != len(
            set(removal_ids)
        ):
            raise ValueError(
                f"{self.scene_name}: invalid safe obstruction list"
            )
        removals = []
        for stable_id in removal_ids:
            fixture = self.get_fixture(id=stable_id)
            if fixture is None or fixture.stable_id != stable_id:
                raise LookupError(
                    f"{self.scene_name}: missing safe obstruction {stable_id}"
                )
            if fixture.stable_id == self.target_fixture.stable_id:
                raise ValueError(
                    f"{self.scene_name}: safe target cannot be an obstruction"
                )
            if fixture.stable_id == support.stable_id:
                raise ValueError(
                    f"{self.scene_name}: safe support cannot be an obstruction"
                )
            removals.append(fixture)
        self.remove_semantic_fixtures(removals)
        self.removed_safe_obstruction_ids = list(removal_ids)

    def _place_supported_articulation_fixture(self):
        root_joint = self._fixture_free_joint_name(self.target_fixture)
        qpos = self.sim.data.get_joint_qpos(root_joint).copy()
        actual_xy = np.asarray(
            self.target_fixture.world_center(self.sim)[:2], dtype=float
        )
        qpos[:2] += self._safe_xy - actual_xy
        self.sim.data.set_joint_qpos(root_joint, qpos)
        self.sim.forward()
        support_top_z = float(self._safe_support_region.world_pose(self.sim)[0][2])
        if not self._place_fixture_on_support_contact(
            self.target_fixture,
            (self._safe_support_fixture.body_name,),
            support_top_z,
            probe_below=float(
                self.scene_task_spec()["safe_support_probe_below"]
            ),
        ):
            raise RuntimeError(
                f"{self.scene_name}: safe could not contact its bedside table"
            )
        self._lock_safe_base_at_current_pose()

    def _supported_fixture_contact_contract(self):
        return (
            "safe_bedside_table",
            tuple(self.target_fixture.body_names),
            tuple(self._safe_support_fixture.body_names),
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["safe_support_fixture"] = self._safe_support_fixture.stable_id
        ep_meta["safe_support_region"] = self._safe_support_region.stable_id
        ep_meta["removed_safe_obstructions"] = list(
            self.removed_safe_obstruction_ids
        )
        return ep_meta


class IThorBedroomOpenSafeDoor(
    BedroomTwentyEightyArticulationMixin,
    _BedsideSafeMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    ATOMIC_SKILL = "OpenDoor"
    SPEC_TASK = "open_safe_door"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK,
        required=(
            "fixture",
            "joint",
            "target",
            "lang",
            "safe_xy",
            "safe_yaw_delta",
            "safe_support_fixture",
            "safe_support_region",
            "safe_support_probe_below",
            "remove_fixtures",
            "safe_robot_xy",
            "safe_robot_min_clearance",
        ),
    )
    BEHAVIOR = "open"


class IThorBedroomCloseSafeDoor(
    BedroomTwentyEightyArticulationMixin,
    _RaisedSafeMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    ATOMIC_SKILL = "CloseDoor"
    SPEC_TASK = "safe_door"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK,
        required=(
            "fixture",
            "joint",
            "target",
            "lang",
            "safe_z_offset",
            "safe_xy",
            "safe_pedestal_half_xy",
            "safe_support_probe_below",
            "safe_robot_xy",
            "safe_robot_min_clearance",
        ),
    )
    BEHAVIOR = "close"

__all__ = ["IThorBedroomOpenSafeDoor", "IThorBedroomCloseSafeDoor"]
