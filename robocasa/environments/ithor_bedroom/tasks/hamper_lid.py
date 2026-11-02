"""Laundry-hamper lid articulation tasks."""

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
    BedroomPartiallyOpenStartMixin,
    BedroomSupportedFixtureMixin,
)
from .assets import asset_xml_path


class _FloorHamperMixin(
    BedroomPartiallyOpenStartMixin,
    BedroomSupportedFixtureMixin,
):
    """Seat and fix the hamper on the physical floor."""

    HAMPER_BASE_WELD_NAME = "task_hamper_fixed_base"

    def _fixture_mode(self):
        return self.scene_task_spec()["fixture_mode"]

    def _create_spawned_hamper_fixture(self):
        """Add a task-private hamper in rooms without a native hamper."""

        spec = self.scene_task_spec()
        if self._fixture_mode() != "spawn_asset":
            return
        quat = np.asarray(spec["spawn_quat"], dtype=float)
        quat /= max(float(np.linalg.norm(quat)), 1e-8)
        fixture = self.create_scene_fixture(
            xml_path=asset_xml_path(spec["asset"]),
            pos=np.asarray(spec["spawn_root_pos"], dtype=float),
            quat=quat,
            name=spec["spawn_name"],
            category="LaundryHamper",
            source_category="LaundryHamper",
            asset_id=spec["asset_id"],
        )
        if fixture.stable_id != spec["fixture"]:
            raise RuntimeError(
                f"{self.scene_name}: spawned hamper id {fixture.stable_id!r} "
                f"does not match {spec['fixture']!r}"
            )

        # The source asset's lid hangs below the basket at the close-task
        # starting aperture.  Add a task-private low support pad so both the
        # open and close initial states remain clear of the physical floor.
        root = self.find_scene_body(fixture.body_name)
        if root is None:
            raise RuntimeError(
                f"{self.scene_name}: spawned hamper XML body is missing"
            )
        # The source asset includes red sites for receptacle visualization.
        # They are useful during asset authoring but look like a red filler
        # inside the hamper in the task scene.  Hide them only on this
        # task-private spawned copy; native room assets remain unchanged.
        for site in root.iter("site"):
            site.set("rgba", "0 0 0 0")
        base_bodies = [
            body
            for body in root.findall("./body")
            if "lid" not in body.get("name", "").lower()
        ]
        if len(base_bodies) != 1:
            raise RuntimeError(
                f"{self.scene_name}: expected one spawned hamper base body"
            )
        robot_y = float(spec["hamper_robot_pose"]["xy"][1])
        root_y = float(spec["spawn_root_pos"][1])
        support_local_y = 0.05 if robot_y > root_y else 0.50
        ET.SubElement(
            base_bodies[0],
            "geom",
            name=f"{spec['spawn_name']}_floor_support",
            type="box",
            size="0.06 0.06 0.04",
            pos=f"0 {support_local_y:.6f} -0.21",
            contype="1",
            conaffinity="15",
            friction="0.9 0.9 0.001",
            solref="0.025",
            solimp="0.998 0.998",
            group="4",
        )

    def _setup_task_fixtures(self):
        self._create_spawned_hamper_fixture()
        super()._setup_task_fixtures()
        fixture = self.target_fixture
        root = self.find_scene_body(fixture.body_name)
        if root is None:
            raise RuntimeError(
                f"{self.scene_name}: missing hamper root {fixture.body_name}"
            )
        self._ensure_fixture_free_joint(
            fixture,
            f"task_hamper_free_joint_{self.scene_name}",
        )
        ET.SubElement(
            self.mujoco_arena.equality,
            "weld",
            name=self.HAMPER_BASE_WELD_NAME,
            body1=fixture.body_name,
            active="false",
            solref="0.002 1",
            solimp="0.9999 0.9999 0.001 0.5 2",
        )

    def _setup_scene(self):
        self._set_hamper_base_weld_active(False)
        super()._setup_scene()

    def _place_supported_articulation_fixture(self):
        # The spawned asset's lid reaches the floor at its fully-open qpos,
        # while the basket body itself is still elevated.  Temporarily disable
        # only the lid collision geoms so the floor probe seats the basket, then
        # restore them before restoring the task's intended initial aperture.
        initial_qpos = None
        saved_lid_collision = {}
        if self._fixture_mode() == "spawn_asset":
            initial_qpos = float(
                self.sim.data.get_joint_qpos(self.target_joint)
            )
            self.sim.data.set_joint_qpos(
                self.target_joint,
                float(self.target_semantic_joint.open_qpos),
            )
            joint_id = self.sim.model.joint_name2id(self.target_joint)
            dof_address = int(self.sim.model.jnt_dofadr[joint_id])
            self.sim.data.qvel[dof_address] = 0.0
            self.sim.forward()
            target_joint_id = self.sim.model.joint_name2id(self.target_joint)
            lid_body_id = int(self.sim.model.jnt_bodyid[target_joint_id])
            lid_body_ids = self._body_tree_ids([lid_body_id])
            for geom_id in range(self.sim.model.ngeom):
                if int(self.sim.model.geom_bodyid[geom_id]) not in lid_body_ids:
                    continue
                saved_lid_collision[geom_id] = (
                    int(self.sim.model.geom_contype[geom_id]),
                    int(self.sim.model.geom_conaffinity[geom_id]),
                )
                self.sim.model.geom_contype[geom_id] = 0
                self.sim.model.geom_conaffinity[geom_id] = 0
            self.sim.forward()

        try:
            placed = self._place_fixture_on_support_contact(
                self.target_fixture,
                (0,),
                0.0,
                probe_below=float(
                    self.scene_task_spec()["hamper_floor_probe_below"]
                ),
            )
        finally:
            for geom_id, (contype, conaffinity) in saved_lid_collision.items():
                self.sim.model.geom_contype[geom_id] = contype
                self.sim.model.geom_conaffinity[geom_id] = conaffinity
            self.sim.forward()

        if not placed:
            raise RuntimeError(
                f"{self.scene_name}: hamper could not contact the floor"
            )
        self._lock_hamper_base_at_current_pose()
        if initial_qpos is not None:
            self.sim.data.set_joint_qpos(self.target_joint, initial_qpos)
            joint_id = self.sim.model.joint_name2id(self.target_joint)
            dof_address = int(self.sim.model.jnt_dofadr[joint_id])
            self.sim.data.qvel[dof_address] = 0.0
            self.sim.forward()

    def _hamper_base_weld_id(self):
        equality_id = mujoco.mj_name2id(
            self.sim.model._model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            self.HAMPER_BASE_WELD_NAME,
        )
        if equality_id < 0:
            raise RuntimeError(
                f"{self.scene_name}: missing hamper base weld constraint"
            )
        return equality_id

    def _set_hamper_base_weld_active(self, active):
        self.sim.data.eq_active[self._hamper_base_weld_id()] = int(bool(active))
        self.sim.forward()

    def _lock_hamper_base_at_current_pose(self):
        """Fix the hamper root while keeping its lid hinge movable."""

        equality_id = self._hamper_base_weld_id()
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
        self._fixed_hamper_base_position = position

    def _supported_fixture_contact_contract(self):
        return (
            "hamper_floor",
            tuple(self.target_fixture.body_names),
            (0,),
        )

    def _finalize_supported_articulation_robot_pose(self, distance):
        """Use one calibrated pose with clearance from nearby furniture."""

        if self._fixture_mode() == "spawn_asset":
            self._hamper_robot_clearances = {}
            spawn_pose = self.scene_task_spec().get("spawn_robot_pose")
            if spawn_pose is not None:
                xy = np.asarray(spawn_pose["xy"], dtype=float)
                yaw = float(spawn_pose["yaw"])
                if xy.shape != (2,):
                    raise ValueError(
                        "spawn_robot_pose.xy must contain two values"
                    )
                if not self.robot_footprint_is_on_floor(xy, radius=0.25):
                    return False
                if not (
                    self._set_robot_free_pose(xy, yaw)
                    and self._calibrate_robot_free_joint_height()
                ):
                    return False

                joint = self._get_robot_free_joint()
                if joint is None:
                    return False
                joint_id = self.sim.model.joint_name2id(joint)
                dof_address = int(self.sim.model.jnt_dofadr[joint_id])
                self.sim.data.qvel[dof_address : dof_address + 6] = 0.0
                self.sim.forward()
                if self.count_body_collisions() != 0:
                    return False

                self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
                self.robot_base_ori = np.array([0.0, 0.0, yaw])
                return True
            return super()._finalize_supported_articulation_robot_pose(
                distance
            )
        del distance
        spec = self.scene_task_spec()
        pose = spec["hamper_robot_pose"]
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        if xy.shape != (2,):
            raise ValueError("hamper_robot_pose.xy must contain two values")
        if not self.robot_footprint_is_on_floor(xy, radius=0.25):
            return False
        if not (
            self._set_robot_free_pose(xy, yaw)
            and self._calibrate_robot_free_joint_height()
        ):
            return False

        joint = self._get_robot_free_joint()
        if joint is None:
            return False
        joint_id = self.sim.model.joint_name2id(joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[dof_address : dof_address + 6] = 0.0
        self.sim.forward()
        if self.count_body_collisions() != 0:
            return False

        clearances = {}
        for stable_id, minimum in spec["hamper_robot_clearances"].items():
            fixture = self.get_fixture(id=stable_id)
            if fixture is None or fixture.stable_id != stable_id:
                raise LookupError(
                    f"{self.scene_name}: missing clearance fixture {stable_id}"
                )
            minimum = float(minimum)
            if minimum <= 0.0:
                raise ValueError(
                    f"{self.scene_name}: invalid clearance for {stable_id}"
                )
            clearance = self.robot_fixture_collision_distance(
                fixture,
                max_distance=minimum + 0.25,
            )
            if not np.isfinite(clearance) or clearance < minimum:
                return False
            clearances[stable_id] = float(clearance)

        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        self._hamper_robot_clearances = clearances
        return True

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["hamper_robot_clearances"] = dict(
            getattr(self, "_hamper_robot_clearances", {})
        )
        ep_meta["hamper_base_fixed"] = True
        ep_meta["hamper_on_floor"] = True
        return ep_meta


class IThorBedroomOpenHamperLid(
    _FloorHamperMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    ATOMIC_SKILL = "OpenLid"
    SPEC_TASK = "hamper_lid"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK,
        required=(
            "fixture",
            "joint",
            "target",
            "lang",
            "hamper_floor_probe_below",
            "hamper_robot_pose",
            "hamper_robot_clearances",
        ),
    )
    BEHAVIOR = "open"


class IThorBedroomCloseHamperLid(
    _FloorHamperMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    ATOMIC_SKILL = "CloseLid"
    SPEC_TASK = "hamper_lid"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK,
        required=(
            "fixture",
            "joint",
            "target",
            "lang",
            "hamper_floor_probe_below",
            "hamper_robot_pose",
            "hamper_robot_clearances",
        ),
    )
    BEHAVIOR = "close"


__all__ = ["IThorBedroomOpenHamperLid", "IThorBedroomCloseHamperLid"]
