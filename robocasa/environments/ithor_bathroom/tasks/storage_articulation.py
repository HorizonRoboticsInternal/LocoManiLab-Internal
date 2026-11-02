"""Exact, short-stroke Bathroom cabinet-door and lower-drawer tasks."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
from robosuite.utils.mjcf_utils import get_elements, scale_mjcf_model

from robocasa.environments.ithor_atomic.articulation import ExactSceneArticulation
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_specs,
    get_supported_scene_ids,
)


_REQUIRED_STORAGE_SPEC = (
    "fixture",
    "joint",
    "joint_type",
    "robot_pose",
    "init_open_fraction",
    "success_open_fraction",
    "success_close_fraction",
    "anchor_fixture",
    "target",
    "lang",
)


class _BathroomStorageArticulation(ExactSceneArticulation):
    """Bind one exact joint, exact stand pose, and collision-free short sweep."""

    HOLD_STEPS = 5
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    SWEEP_SAMPLES = 13
    JOINT_DAMPING = 0.5
    JOINT_FRICTIONLOSS = 0.2

    def _setup_task_fixtures(self):
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
            raise ValueError(f"{self.scene_name}: articulation fractions need [0, 1]")
        super()._setup_task_fixtures()
        if self.target_fixture is not None and bool(spec["anchor_fixture"]):
            self._anchor_target_fixture_root()

    def _anchor_target_fixture_root(self):
        root = self.find_scene_body(self.target_fixture.body_name)
        if root is None:
            raise RuntimeError(
                f"{self.scene_name}: missing fixture root "
                f"{self.target_fixture.body_name}"
            )
        free_joints = [
            joint
            for joint in root.findall("./joint")
            if joint.get("type") == "free" or "jntfree" in joint.get("name", "")
        ]
        if len(free_joints) != 1:
            raise RuntimeError(
                f"{self.scene_name}: anchored fixture must have one free joint, "
                f"found {len(free_joints)}"
            )
        root.remove(free_joints[0])
        self._fixture_was_anchored = True

    def _articulation_robot_pose(self):
        pose = self.scene_task_spec()["robot_pose"]
        xy = np.asarray(pose["xy"], dtype=float)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError(f"{self.scene_name}: invalid articulation robot XY")
        yaw = float(pose["yaw"])
        if not np.isfinite(yaw):
            raise ValueError(f"{self.scene_name}: invalid articulation robot yaw")
        return np.array([xy[0], xy[1], 0.0]), yaw

    def _initial_joint_qpos(self, joint):
        return float(
            joint.closed_qpos
            + self.OPEN_INIT_FRACTION * (joint.open_qpos - joint.closed_qpos)
        )

    def _success_fraction(self):
        return (
            self.SUCCESS_CLOSE_FRACTION
            if self.BEHAVIOR == "close"
            else self.SUCCESS_OPEN_FRACTION
        )

    def _robot_clears_joint_sweep(self):
        joint = self.target_semantic_joint
        joint_id = self.sim.model.joint_name2id(self.target_joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        original_qpos = float(self.sim.data.get_joint_qpos(self.target_joint))
        original_qvel = float(self.sim.data.qvel[dof_address])
        try:
            for fraction in np.linspace(
                self.OPEN_INIT_FRACTION,
                self._success_fraction(),
                self.SWEEP_SAMPLES,
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

    def _finalize_articulation_robot_pose(self, distance):
        del distance
        pose = self.scene_task_spec()["robot_pose"]
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        if hasattr(self, "_robot_free_joint_z"):
            delattr(self, "_robot_free_joint_z")
        if not (
            self.robot_footprint_is_on_floor(xy, radius=0.25)
            and self._set_robot_free_pose(xy, yaw)
            and self._calibrate_robot_free_joint_height()
            and self.count_body_collisions() == 0
            and self._robot_clears_joint_sweep()
        ):
            return False
        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        return True

    def _reset_target_joint_pose(self):
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
        expected_type = self.scene_task_spec()["joint_type"]
        actual_type = self._joint_type_name(self.target_semantic_joint)
        if actual_type != expected_type:
            raise RuntimeError(
                f"{self.scene_name}: expected {expected_type} joint, "
                f"found {actual_type}"
            )
        joint_id = self.sim.model.joint_name2id(self.target_joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.model.dof_damping[dof_address] = max(
            float(self.sim.model.dof_damping[dof_address]),
            self.JOINT_DAMPING,
        )
        self.sim.model.dof_frictionloss[dof_address] = max(
            float(self.sim.model.dof_frictionloss[dof_address]),
            self.JOINT_FRICTIONLOSS,
        )
        self._reset_target_joint_pose()

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_target_joint_pose()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        spec = self.scene_task_spec()
        ep_meta.update(
            {
                "lang": spec["lang"],
                "initial_open_fraction": float(self.OPEN_INIT_FRACTION),
                "success_fraction_threshold": float(self._success_fraction()),
                "fixture_anchored": bool(getattr(self, "_fixture_was_anchored", False)),
                "joint_sweep_collision_free": True,
                "success_definition": (
                    f"the exact {spec['joint_type']} joint must reach the "
                    f"{self._success_fraction():.2f} open-fraction threshold "
                    f"for {self.HOLD_STEPS} control steps"
                ),
            }
        )
        return ep_meta


class _BathroomUpperCabinetArticulation(_BathroomStorageArticulation):
    """Shrink the complete freestanding cabinet before building its semantics."""

    def _setup_baked_scene_fixtures(self):
        super()._setup_baked_scene_fixtures()
        spec = self.scene_task_spec()
        scale = float(spec["cabinet_scale"])
        origin = np.asarray(spec["cabinet_scale_origin"], dtype=float)
        # The carcass, doors, drawer and shelf contents are separate roots.
        # One world-space transform keeps their meshes and collisions aligned.
        cabinet = ET.Element("worldbody")
        for name in spec["cabinet_part_bodies"]:
            body = self.find_scene_body(name)
            if body is None:
                raise RuntimeError(f"{self.scene_name}: missing cabinet part {name}")
            cabinet.append(body)
        mesh_names = {geom.get("mesh") for geom in cabinet.iter("geom")}
        assets = ET.Element("asset")
        for mesh in self.mujoco_arena.asset.findall("mesh"):
            if mesh.get("name") in mesh_names:
                assets.append(mesh)
        scale_mjcf_model(cabinet, assets, scale, get_elements, cabinet)
        for body in cabinet:
            pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
            pos += (1.0 - scale) * origin
            body.set("pos", " ".join(f"{value:.9g}" for value in pos))


class IThorBathroomOpenCabinetDoor(
    _BathroomStorageArticulation,
    IThorBathroomEnv,
):
    """Open the exact low bathroom cabinet door to its scene target."""

    ATOMIC_SKILL = "OpenArticulation"
    BEHAVIOR = "open"
    SPEC_TASK = "open_cabinet_door"
    TARGET_JOINT_TYPES = ("hinge",)
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK, required=_REQUIRED_STORAGE_SPEC
    )


class IThorBathroomOpenUpperCabinetDoor(
    _BathroomUpperCabinetArticulation,
    IThorBathroomEnv,
):
    """Open the upper wall cabinet in scene 422 from 20% to 80%."""

    ATOMIC_SKILL = "OpenArticulation"
    BEHAVIOR = "open"
    SPEC_TASK = "open_upper_cabinet_door"
    TARGET_JOINT_TYPES = ("hinge",)
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK, required=_REQUIRED_STORAGE_SPEC
    )


class IThorBathroomCloseCabinetDoor(
    _BathroomStorageArticulation,
    IThorBathroomEnv,
):
    """Close the exact low bathroom cabinet door to its scene target."""

    ATOMIC_SKILL = "CloseArticulation"
    BEHAVIOR = "close"
    SPEC_TASK = "close_cabinet_door"
    TARGET_JOINT_TYPES = ("hinge",)
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK, required=_REQUIRED_STORAGE_SPEC
    )


class IThorBathroomCloseUpperCabinetDoor(
    _BathroomUpperCabinetArticulation,
    IThorBathroomEnv,
):
    """Close the upper wall cabinet in scene 422 from 80% to 20%."""

    ATOMIC_SKILL = "CloseArticulation"
    BEHAVIOR = "close"
    SPEC_TASK = "close_upper_cabinet_door"
    TARGET_JOINT_TYPES = ("hinge",)
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK, required=_REQUIRED_STORAGE_SPEC
    )


class IThorBathroomOpenLowerDrawer(
    _BathroomStorageArticulation,
    IThorBathroomEnv,
):
    """Crouch slightly and open the exact lower drawer from 20% to 80%."""

    ATOMIC_SKILL = "OpenDrawer"
    BEHAVIOR = "open"
    SPEC_TASK = "open_lower_drawer"
    TARGET_JOINT_TYPES = ("slide",)
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK, required=_REQUIRED_STORAGE_SPEC
    )


class IThorBathroomCloseLowerDrawer(
    _BathroomStorageArticulation,
    IThorBathroomEnv,
):
    """Crouch slightly and close the exact lower drawer from 80% to 20%."""

    ATOMIC_SKILL = "CloseDrawer"
    BEHAVIOR = "close"
    SPEC_TASK = "close_lower_drawer"
    TARGET_JOINT_TYPES = ("slide",)
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK, required=_REQUIRED_STORAGE_SPEC
    )


__all__ = [
    "IThorBathroomCloseCabinetDoor",
    "IThorBathroomCloseLowerDrawer",
    "IThorBathroomCloseUpperCabinetDoor",
    "IThorBathroomOpenCabinetDoor",
    "IThorBathroomOpenLowerDrawer",
    "IThorBathroomOpenUpperCabinetDoor",
]
