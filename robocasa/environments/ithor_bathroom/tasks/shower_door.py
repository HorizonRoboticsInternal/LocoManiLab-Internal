"""Exact shower-door tasks for calibrated iTHOR bathrooms."""

import numpy as np

from robocasa.environments.ithor_atomic.articulation import ExactSceneArticulation
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_specs,
    get_supported_scene_ids,
)


_COMMON_REQUIRED_FIELDS = (
    "fixture",
    "joint",
    "joint_type",
    "joint_range",
    "closed_qpos",
    "open_qpos",
    "init_open_fraction",
    "success_open_fraction",
    "success_close_fraction",
    "target_open_angle_degrees",
    "closed_tolerance_degrees",
    "target",
    "lang",
)
def _load_shower_door_specs(task_key):
    return get_ithor_bathroom_task_specs(
        task_key,
        required=_COMMON_REQUIRED_FIELDS,
    )


class _BathroomShowerDoor(ExactSceneArticulation):
    """Bind one exact shower door and its real articulation endpoints."""

    TARGET_JOINT_TYPES = ("hinge",)
    MINIMUM_JOINT_SPAN = 0.50
    TARGET_OPEN_ANGLE_DEGREES = 30.0
    CLOSED_ANGLE_TOLERANCE_DEGREES = 2.0
    SUCCESS_OPEN_FRACTION = 1.0 / 3.0
    SUCCESS_CLOSE_FRACTION = 2.0 / 90.0

    def _configure_articulation_task(self):
        super()._configure_articulation_task()
        spec = self.scene_task_spec()
        self.OPEN_INIT_FRACTION = float(spec["init_open_fraction"])
        self.SUCCESS_OPEN_FRACTION = float(spec["success_open_fraction"])
        self.SUCCESS_CLOSE_FRACTION = float(spec["success_close_fraction"])

        fractions = (
            self.OPEN_INIT_FRACTION,
            self.SUCCESS_OPEN_FRACTION,
            self.SUCCESS_CLOSE_FRACTION,
        )
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: fractions must be in [0, 1]"
            )
        if self.SUCCESS_CLOSE_FRACTION >= self.SUCCESS_OPEN_FRACTION:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: open/close thresholds overlap"
            )
        if self.BEHAVIOR == "open":
            valid_initial = self.OPEN_INIT_FRACTION <= self.SUCCESS_CLOSE_FRACTION
        else:
            valid_initial = self.OPEN_INIT_FRACTION >= self.SUCCESS_OPEN_FRACTION
        if not valid_initial:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: initial state is not opposite "
                f"the {self.BEHAVIOR} goal"
            )

    def _configure_articulation_joint(self, joint):
        super()._configure_articulation_joint(joint)
        spec = self.scene_task_spec()
        expected_type = str(spec["joint_type"]).lower()
        actual_type = self._joint_type_name(joint)
        if expected_type != "hinge" or actual_type != expected_type:
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: expected hinge joint, "
                f"found {actual_type!r}"
            )

        joint_id = self.sim.model.joint_name2id(joint.name)
        actual_range = np.asarray(self.sim.model.jnt_range[joint_id], dtype=float)
        expected_range = np.asarray(spec["joint_range"], dtype=float)
        if expected_range.shape != (2,) or not np.allclose(
            actual_range, expected_range, atol=1e-6, rtol=0.0
        ):
            raise RuntimeError(
                f"{self.scene_name}/{self.SPEC_TASK}: joint range changed "
                f"from {expected_range.tolist()} to {actual_range.tolist()}"
            )

        closed_qpos = float(spec["closed_qpos"])
        open_qpos = float(spec["open_qpos"])
        lo, hi = (float(value) for value in actual_range)
        if not (lo <= closed_qpos <= hi and lo <= open_qpos <= hi):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: endpoints leave joint range"
            )
        if abs(open_qpos - closed_qpos) < self.MINIMUM_JOINT_SPAN:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: door motion is too small"
            )
        travel = abs(open_qpos - closed_qpos)
        target_degrees = float(spec["target_open_angle_degrees"])
        closed_tolerance_degrees = float(spec["closed_tolerance_degrees"])
        expected_open_fraction = float(np.deg2rad(target_degrees) / travel)
        expected_close_fraction = float(
            np.deg2rad(closed_tolerance_degrees) / travel
        )
        expected_initial_fraction = (
            0.0 if self.BEHAVIOR == "open" else expected_open_fraction
        )
        if not (
            np.isclose(
                target_degrees,
                self.TARGET_OPEN_ANGLE_DEGREES,
                atol=1e-8,
                rtol=0.0,
            )
            and np.isclose(
                closed_tolerance_degrees,
                self.CLOSED_ANGLE_TOLERANCE_DEGREES,
                atol=1e-8,
                rtol=0.0,
            )
            and np.isclose(
                self.SUCCESS_OPEN_FRACTION,
                expected_open_fraction,
                atol=2e-6,
                rtol=0.0,
            )
            and np.isclose(
                self.SUCCESS_CLOSE_FRACTION,
                expected_close_fraction,
                atol=2e-6,
                rtol=0.0,
            )
            and np.isclose(
                self.OPEN_INIT_FRACTION,
                expected_initial_fraction,
                atol=2e-6,
                rtol=0.0,
            )
        ):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: shower door must use "
                "0 to 30 degrees, or the reverse, with a 2-degree closed "
                "tolerance"
            )
        joint.closed_qpos = closed_qpos
        joint.open_qpos = open_qpos

    def _setup_scene(self):
        super()._setup_scene()
        joint_id = self.sim.model.joint_name2id(self.target_joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[dof_address] = 0.0
        self.sim.forward()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        success_threshold = (
            self.SUCCESS_CLOSE_FRACTION
            if self.BEHAVIOR == "close"
            else self.SUCCESS_OPEN_FRACTION
        )
        relation = "at or below" if self.BEHAVIOR == "close" else "at or above"
        ep_meta["mechanism"] = "shower door"
        ep_meta["joint_type"] = "hinge"
        ep_meta["closed_qpos"] = float(self.target_semantic_joint.closed_qpos)
        ep_meta["open_qpos"] = float(self.target_semantic_joint.open_qpos)
        ep_meta["initial_open_fraction"] = float(self.OPEN_INIT_FRACTION)
        ep_meta["target_open_angle_degrees"] = float(
            self.TARGET_OPEN_ANGLE_DEGREES
        )
        ep_meta["closed_angle_tolerance_degrees"] = float(
            self.CLOSED_ANGLE_TOLERANCE_DEGREES
        )
        ep_meta["target_state"] = self.BEHAVIOR
        ep_meta["success_fraction_threshold"] = float(success_threshold)
        ep_meta["success_definition"] = (
            f"the exact shower-door open fraction must remain {relation} "
            f"{success_threshold:.2f} for "
            f"{self.HOLD_STEPS} consecutive control steps"
        )
        return ep_meta


class IThorBathroomOpenShowerDoor(_BathroomShowerDoor, IThorBathroomEnv):
    """Open the exact shower door from zero to thirty degrees."""

    ATOMIC_SKILL = "OpenArticulation"
    SPEC_TASK = "open_shower_door"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = _load_shower_door_specs(SPEC_TASK)
    BEHAVIOR = "open"


class IThorBathroomCloseShowerDoor(_BathroomShowerDoor, IThorBathroomEnv):
    """Close the exact shower door from a thirty-degree start."""

    ATOMIC_SKILL = "CloseArticulation"
    SPEC_TASK = "close_shower_door"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = _load_shower_door_specs(SPEC_TASK)
    BEHAVIOR = "close"


__all__ = [
    "IThorBathroomCloseShowerDoor",
    "IThorBathroomOpenShowerDoor",
]
