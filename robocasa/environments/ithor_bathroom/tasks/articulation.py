"""Canonical toilet-lid articulation tasks for maintained bathrooms."""

import numpy as np

from robocasa.environments.ithor_atomic.articulation import ExactSceneArticulation
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_specs,
    get_supported_scene_ids,
)


_TOILET_LID_SPECS = get_ithor_bathroom_task_specs(
    "toilet_lid",
    required=("fixture", "joint", "close_robot_pose", "target", "lang"),
)


class _BathroomToiletLid(ExactSceneArticulation):
    """Resolve one named lid joint and format the behavior-specific language."""

    SUPPORTED_SCENE_IDS = get_supported_scene_ids("toilet_lid")
    SCENE_SPECS = _TOILET_LID_SPECS
    OPEN_START_FRACTION = 0.20
    CLOSE_START_FRACTION = 0.80
    SUCCESS_OPEN_FRACTION = 0.80
    SUCCESS_CLOSE_FRACTION = 0.20

    def _initial_joint_qpos(self, joint):
        fraction = (
            self.CLOSE_START_FRACTION
            if self.BEHAVIOR == "close"
            else self.OPEN_START_FRACTION
        )
        return float(
            joint.closed_qpos + fraction * (joint.open_qpos - joint.closed_qpos)
        )

    def _configure_closed_direction_stop(self):
        """Keep the lid from falling below its reachable 20% opening."""

        joint = self.target_semantic_joint
        stop_qpos = float(
            joint.closed_qpos
            + self.OPEN_START_FRACTION * (joint.open_qpos - joint.closed_qpos)
        )
        joint_id = self.sim.model.joint_name2id(self.target_joint)
        lo, hi = (float(value) for value in self.sim.model.jnt_range[joint_id])
        if joint.open_qpos < joint.closed_qpos:
            hi = min(hi, stop_qpos)
        else:
            lo = max(lo, stop_qpos)
        if not lo < hi:
            raise RuntimeError(
                f"{self.scene_name}: invalid 20% toilet-lid stop [{lo}, {hi}]"
            )
        self.sim.model.jnt_range[joint_id] = (lo, hi)
        self.sim.model.jnt_limited[joint_id] = 1

    def _reset_toilet_lid_pose(self):
        self.sim.data.set_joint_qpos(
            self.target_joint,
            self._initial_joint_qpos(self.target_semantic_joint),
        )
        joint_id = self.sim.model.joint_name2id(self.target_joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[dof_address] = 0.0
        self.sim.forward()

    def _setup_scene(self):
        super()._setup_scene()
        self._configure_closed_direction_stop()
        self._reset_toilet_lid_pose()

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_toilet_lid_pose()

    def scene_task_spec(self):
        spec = super().scene_task_spec()
        spec["lang"] = spec["lang"].format(verb=self.BEHAVIOR)
        return spec

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["initial_open_fraction"] = (
            self.CLOSE_START_FRACTION
            if self.BEHAVIOR == "close"
            else self.OPEN_START_FRACTION
        )
        ep_meta["minimum_open_fraction"] = self.OPEN_START_FRACTION
        ep_meta["success_fraction_threshold"] = (
            self.SUCCESS_CLOSE_FRACTION
            if self.BEHAVIOR == "close"
            else self.SUCCESS_OPEN_FRACTION
        )
        return ep_meta


class _StableCloseToiletLid:
    """Stabilize the lid and keep G1 to the side of its closing sweep."""

    TOILET_LID_DAMPING = 0.5
    TOILET_LID_FRICTIONLOSS = 0.6
    CLOSE_START_FRACTION_BY_SCENE = {"FloorPlan405": 0.85}
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    LID_SWEEP_SAMPLES = 13
    CLOSE_LID_SEARCH_RADII = (
        0.38,
        0.45,
        0.52,
        0.60,
        0.70,
        0.85,
        1.00,
        1.20,
    )
    CLOSE_LID_SEARCH_YAW_OFFSETS = (
        0.0,
        np.deg2rad(15),
        -np.deg2rad(15),
        np.deg2rad(30),
        -np.deg2rad(30),
    )

    def _close_start_fraction(self):
        return self.CLOSE_START_FRACTION_BY_SCENE.get(
            getattr(self, "scene_name", None), self.CLOSE_START_FRACTION
        )

    def _initial_joint_qpos(self, joint):
        if self.BEHAVIOR != "close":
            return super()._initial_joint_qpos(joint)
        return float(
            joint.closed_qpos
            + self._close_start_fraction() * (joint.open_qpos - joint.closed_qpos)
        )

    def _close_robot_pose(self):
        pose = self.scene_task_spec()["close_robot_pose"]
        if not isinstance(pose, dict) or set(pose) != {"xy", "yaw"}:
            raise ValueError(
                f"{self.scene_name}: close_robot_pose must contain exactly xy and yaw"
            )
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        if xy.shape != (2,) or not np.all(np.isfinite(xy)) or not np.isfinite(yaw):
            raise ValueError(f"{self.scene_name}: invalid close_robot_pose")
        return xy, yaw

    def _robot_clears_lid_sweep(self):
        joint = self.target_semantic_joint
        joint_id = self.sim.model.joint_name2id(self.target_joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        original_qpos = float(self.sim.data.get_joint_qpos(self.target_joint))
        original_qvel = float(self.sim.data.qvel[dof_address])
        try:
            for fraction in np.linspace(
                self._close_start_fraction(),
                self.SUCCESS_CLOSE_FRACTION,
                self.LID_SWEEP_SAMPLES,
            ):
                qpos = float(
                    joint.closed_qpos + fraction * (joint.open_qpos - joint.closed_qpos)
                )
                self.sim.data.set_joint_qpos(self.target_joint, qpos)
                self.sim.data.qvel[dof_address] = 0.0
                self.sim.forward()
                if self.count_body_collisions() != 0:
                    return False
            return True
        finally:
            self.sim.data.set_joint_qpos(self.target_joint, original_qpos)
            self.sim.data.qvel[dof_address] = original_qvel
            self.sim.forward()

    def _close_robot_pose_is_statically_safe(self, xy, yaw):
        if not self.robot_footprint_is_on_floor(xy, radius=0.25):
            return False
        return bool(
            self._set_robot_free_pose(xy, yaw)
            and self._calibrate_robot_free_joint_height()
            and self.count_body_collisions() == 0
        )

    def _close_robot_pose_has_current_height_clearance(self, xy, yaw):
        """Quickly screen candidates at the current free-joint height.

        Height calibration tries 25 vertical offsets, so using it for every
        rejected XY/yaw candidate makes a failed spawn search unnecessarily
        slow. The normal G1 spawn height is already collision-free for most
        candidates. A full calibrated pass remains below for layouts that
        genuinely require a raised base.
        """

        return bool(
            self.robot_footprint_is_on_floor(xy, radius=0.25)
            and self._set_robot_free_pose(xy, yaw)
            and self.count_body_collisions() == 0
        )

    def _close_robot_pose_is_safe(self, xy, yaw):
        return bool(
            self._close_robot_pose_is_statically_safe(xy, yaw)
            and self._robot_clears_lid_sweep()
        )

    def _close_lid_pose_candidates(self, radius, yaw_offsets=None):
        """Yield one full ring of candidate poses around the lid hinge.

        Bathroom layouts can leave the calibrated collection pose obstructed by
        a different G1 configuration.  Search a full ring around the actual
        lid hinge rather than assuming the fixture-front direction is usable.
        Yaw offsets account for the G1's non-circular upper-body envelope.
        """

        target = np.asarray(self.target_joint_pos, dtype=float)
        yaw_offsets = (
            self.CLOSE_LID_SEARCH_YAW_OFFSETS if yaw_offsets is None else yaw_offsets
        )
        for angle in np.linspace(-np.pi, np.pi, 24, endpoint=False):
            xy = target[:2] + radius * np.array([np.cos(angle), np.sin(angle)])
            face_yaw = float(np.arctan2(target[1] - xy[1], target[0] - xy[0]))
            for yaw_offset in yaw_offsets:
                yield xy, face_yaw + yaw_offset

    def _finalize_articulation_robot_pose(self, distance):
        del distance
        xy, yaw = self._close_robot_pose()
        if not self._close_robot_pose_is_safe(xy, yaw):
            # The close calibration is deliberately fixed for collection, but
            # use the same joint-centered search as ExactSceneArticulation if
            # scene geometry makes that exact pose invalid. Every fallback is
            # still floor-contained, collision-free, and clear of the lid's
            # complete closing sweep.
            target = np.asarray(self.target_joint_pos, dtype=float)
            for radius in self.CLOSE_LID_SEARCH_RADII:
                current_height_clear = []
                for candidate_xy, candidate_yaw in self._close_lid_pose_candidates(
                    radius
                ):
                    if not self._close_robot_pose_has_current_height_clearance(
                        candidate_xy, candidate_yaw
                    ):
                        continue
                    reach = min(
                        float(np.linalg.norm(target - eef))
                        for eef in self.eef_positions()
                    )
                    current_height_clear.append(
                        (reach, candidate_xy.copy(), candidate_yaw)
                    )
                for _, candidate_xy, candidate_yaw in sorted(
                    current_height_clear, key=lambda candidate: candidate[0]
                ):
                    if self._close_robot_pose_is_safe(candidate_xy, candidate_yaw):
                        xy, yaw = candidate_xy, candidate_yaw
                        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
                        self.robot_base_ori = np.array([0.0, 0.0, yaw])
                        return True

            # A few unusual layouts require lifting G1 to clear a fixture.
            # Try the fixture-facing yaw before the more expensive all-yaw
            # calibrated fallback; that is the intended collection posture.
            for radius in self.CLOSE_LID_SEARCH_RADII:
                for candidate_xy, candidate_yaw in self._close_lid_pose_candidates(
                    radius, yaw_offsets=(0.0,)
                ):
                    if self._close_robot_pose_is_safe(candidate_xy, candidate_yaw):
                        xy, yaw = candidate_xy, candidate_yaw
                        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
                        self.robot_base_ori = np.array([0.0, 0.0, yaw])
                        return True

            # Preserve an exhaustive calibrated search for layouts where the
            # G1 upper body needs a nonzero yaw offset to clear the fixture.
            for radius in self.CLOSE_LID_SEARCH_RADII:
                for candidate_xy, candidate_yaw in self._close_lid_pose_candidates(
                    radius
                ):
                    if self._close_robot_pose_is_safe(candidate_xy, candidate_yaw):
                        xy, yaw = candidate_xy, candidate_yaw
                        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
                        self.robot_base_ori = np.array([0.0, 0.0, yaw])
                        return True
            return False
        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        return True

    def _setup_scene(self):
        super()._setup_scene()
        joint_id = self.sim.model.joint_name2id(self.target_joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.model.dof_damping[dof_address] = max(
            float(self.sim.model.dof_damping[dof_address]),
            self.TOILET_LID_DAMPING,
        )
        self.sim.model.dof_frictionloss[dof_address] = max(
            float(self.sim.model.dof_frictionloss[dof_address]),
            self.TOILET_LID_FRICTIONLOSS,
        )
        self.sim.data.qvel[dof_address] = 0.0
        self.sim.forward()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["initial_open_fraction"] = self._close_start_fraction()
        ep_meta["close_robot_pose"] = {
            "xy": [float(value) for value in self.robot_base_pos[:2]],
            "yaw": float(self.robot_base_ori[2]),
        }
        ep_meta["lid_sweep_collision_free"] = True
        return ep_meta


class IThorBathroomOpenToiletLid(_BathroomToiletLid, IThorBathroomEnv):
    """Open the named toilet lid in any calibrated bathroom."""

    ATOMIC_SKILL = "OpenArticulation"
    BEHAVIOR = "open"


class IThorBathroomCloseToiletLid(
    _StableCloseToiletLid,
    _BathroomToiletLid,
    IThorBathroomEnv,
):
    """Close the named toilet lid in any calibrated bathroom."""

    ATOMIC_SKILL = "CloseArticulation"
    BEHAVIOR = "close"


__all__ = [
    "IThorBathroomCloseToiletLid",
    "IThorBathroomOpenToiletLid",
]
