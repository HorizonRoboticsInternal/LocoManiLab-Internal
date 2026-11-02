"""Box-flap articulation tasks for calibrated bedroom scenes."""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_specs,
    get_supported_scene_ids,
)
from robocasa.models.scenes.thor_semantics import SemanticJoint

from .articulation_base import (
    BedroomArticulationMixin,
    BedroomSupportedFixtureMixin,
)
from .assets import asset_xml_path


class BedroomBoxFlapMixin(
    BedroomSupportedFixtureMixin, BedroomArticulationMixin
):
    """Stage one uniquely identifiable flap on a waist-high supported box."""

    SPEC_TASK = "box_flap"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bedroom_task_specs(
        SPEC_TASK,
        required=(
            "fixture",
            "joint",
            "closed_qpos",
            "open_qpos",
            "open_start_qpos",
            "target",
            "lang",
            "box_floor_xy",
            "box_floor_probe_below",
            "remove_fixtures",
            "robot_outward_direction",
            "robot_center_distance",
            "minimum_target_right_offset",
        ),
    )
    BOX_BASE_WELD_NAME = "task_box_flap_fixed_base"

    def _fixture_mode(self):
        return self.scene_task_spec()["fixture_mode"]

    def _create_spawned_box_fixture(self):
        """Add the task-private Box asset when the source room has none."""

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
            category="Box",
            source_category="Box",
            asset_id=spec["asset_id"],
            front=spec["robot_outward_direction"],
        )
        if fixture.stable_id != spec["fixture"]:
            raise RuntimeError(
                f"{self.scene_name}: spawned box id {fixture.stable_id!r} "
                f"does not match {spec['fixture']!r}"
            )

        root = self.find_scene_body(fixture.body_name)
        # Native room boxes put the red receptacle marker in hidden group 5.
        for site in root.iter("site"):
            if "ReceptacleCollider" in site.get("name", ""):
                site.set("group", "5")

        # Keep all four hinges so sibling flaps can be staged like native boxes.
        fixture.joints.clear()
        for hinge in root.iter("joint"):
            if hinge.get("type") == "free":
                continue
            name = hinge.get("name")
            low, high = (float(value) for value in hinge.get("range").split())
            fixture.joints[name] = SemanticJoint(
                stable_id=f"{fixture.stable_id}:joint:{name}",
                name=name,
                role="flap",
                range=(low, high),
                closed_qpos=0.0,
                open_qpos=high if abs(high) >= abs(low) else low,
            )

        # Retain the explicitly calibrated flap as an addressable joint.
        fixture.joints["target_flap"] = SemanticJoint(
            stable_id=f"{fixture.stable_id}:joint:target_flap",
            name=spec["joint"],
            role="flap",
            range=(float(spec["closed_qpos"]), float(spec["open_qpos"])),
            closed_qpos=float(spec["closed_qpos"]),
            open_qpos=float(spec["open_qpos"]),
        )

    def scene_task_spec(self):
        spec = dict(super().scene_task_spec())
        initial_state = "upright" if self.BEHAVIOR == "open" else "open"
        spec["target"] = f"only {initial_state} box flap"
        spec["lang"] = (
            f"{self.BEHAVIOR} the only {initial_state} flap of the floor box"
        )
        return spec

    def _setup_task_fixtures(self):
        self._create_spawned_box_fixture()
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        floor_xy = np.asarray(spec["box_floor_xy"], dtype=float)
        outward = np.asarray(spec["robot_outward_direction"], dtype=float)
        center_distance = float(spec["robot_center_distance"])
        if (
            floor_xy.shape != (2,)
            or outward.shape != (2,)
            or not np.isclose(np.linalg.norm(outward), 1.0, atol=1e-6)
            or center_distance <= 0.0
            or float(spec["minimum_target_right_offset"]) <= 0.0
        ):
            raise ValueError(f"{self.scene_name}: invalid floor box calibration")

        removal_ids = tuple(spec["remove_fixtures"])
        if len(removal_ids) != len(set(removal_ids)):
            raise ValueError(
                f"{self.scene_name}: invalid box obstruction list"
            )
        removals = []
        for stable_id in removal_ids:
            fixture = self.get_fixture(id=stable_id)
            if fixture is None or fixture.stable_id != stable_id:
                raise LookupError(
                    f"{self.scene_name}: missing box obstruction {stable_id}"
                )
            if fixture.stable_id == self.target_fixture.stable_id:
                raise ValueError(
                    f"{self.scene_name}: box target cannot be an obstruction"
                )
            removals.append(fixture)
        self.remove_semantic_fixtures(removals)
        self.removed_box_obstruction_ids = list(removal_ids)

        root = self.find_scene_body(self.target_fixture.body_name)
        if root is None:
            raise RuntimeError(f"{self.scene_name}: missing box root body")

        # The original box is loose on the floor beside the room boundary.
        # Keep it dynamic during seating, move it to the calibrated open floor
        # area, and weld only its base after real floor contact is established.
        self._ensure_fixture_free_joint(
            self.target_fixture, f"task_box_free_joint_{self.scene_name}"
        )
        ET.SubElement(
            self.mujoco_arena.equality,
            "weld",
            name=self.BOX_BASE_WELD_NAME,
            body1=self.target_fixture.body_name,
            active="false",
            solref="0.002 1",
            solimp="0.9999 0.9999 0.001 0.5 2",
        )
        root_pos = np.fromstring(root.get("pos", "0 0 0"), sep=" ")
        if root_pos.shape != (3,):
            raise RuntimeError(f"{self.scene_name}: invalid box root position")
        root_pos[:2] = floor_xy
        root.set("pos", " ".join(str(value) for value in root_pos))

        # ExactSceneArticulation performs the final collision-aware placement
        # from the moved hinge after the compiled simulation is available.
        robot_xy = floor_xy + outward * center_distance
        robot_yaw = float(np.arctan2(-outward[1], -outward[0]))
        self.set_robot_base_pose(
            (robot_xy[0], robot_xy[1], 0.0),
            robot_yaw,
        )

    def _setup_scene(self):
        self._set_box_base_weld_active(False)
        super()._setup_scene()

    def _configure_articulation_joint(self, joint):
        """Use the Box asset's visual flap endpoints, not the generic guess."""

        super()._configure_articulation_joint(joint)
        spec = self.scene_task_spec()
        closed_qpos = float(spec["closed_qpos"])
        open_qpos = float(spec["open_qpos"])
        low, high = (float(value) for value in joint.range)
        if (
            not low <= closed_qpos <= high
            or not low <= open_qpos <= high
            or np.isclose(closed_qpos, open_qpos)
        ):
            raise ValueError(
                f"{self.scene_name}: invalid visual Box flap endpoints"
            )
        joint.closed_qpos = closed_qpos
        joint.open_qpos = open_qpos
        if self.BEHAVIOR == "open":
            start_qpos = self._open_start_qpos(joint)
            joint_id = self.sim.model.joint_name2id(joint.name)
            self.sim.model.jnt_limited[joint_id] = 1
            if open_qpos > closed_qpos:
                self.sim.model.jnt_range[joint_id, 0] = start_qpos
            else:
                self.sim.model.jnt_range[joint_id, 1] = start_qpos

    def _open_start_qpos(self, joint):
        start_qpos = float(self.scene_task_spec()["open_start_qpos"])
        low, high = sorted((float(joint.closed_qpos), float(joint.open_qpos)))
        if not low < start_qpos < high:
            raise ValueError(
                f"{self.scene_name}: upright Box flap qpos is outside travel"
            )
        fraction = (start_qpos - float(joint.closed_qpos)) / (
            float(joint.open_qpos) - float(joint.closed_qpos)
        )
        if not 0.0 < fraction < self.SUCCESS_OPEN_FRACTION:
            raise ValueError(
                f"{self.scene_name}: upright Box flap start must precede success"
            )
        return start_qpos

    def _initial_joint_qpos(self, joint):
        if self.BEHAVIOR == "open":
            return self._open_start_qpos(joint)
        return super()._initial_joint_qpos(joint)

    def _place_supported_articulation_fixture(self):
        if not self._place_fixture_on_support_contact(
            self.target_fixture,
            (0,),
            0.0,
            probe_below=float(
                self.scene_task_spec()["box_floor_probe_below"]
            ),
        ):
            raise RuntimeError(
                f"{self.scene_name}: box could not contact the floor"
            )
        self._lock_box_base_at_current_pose()

    def _box_base_weld_id(self):
        equality_id = mujoco.mj_name2id(
            self.sim.model._model,
            mujoco.mjtObj.mjOBJ_EQUALITY,
            self.BOX_BASE_WELD_NAME,
        )
        if equality_id < 0:
            raise RuntimeError(
                f"{self.scene_name}: missing box base weld constraint"
            )
        return equality_id

    def _set_box_base_weld_active(self, active):
        self.sim.data.eq_active[self._box_base_weld_id()] = int(bool(active))
        self.sim.forward()

    def _lock_box_base_at_current_pose(self):
        """Weld the box body to world while leaving its flap hinges movable."""

        equality_id = self._box_base_weld_id()
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
        self._fixed_box_base_position = position

    def _supported_fixture_contact_contract(self):
        return (
            "box_floor",
            tuple(self.target_fixture.body_names),
            (0,),
        )

    def _stage_non_target_flaps(self):
        # For these Box assets, zero is visually open and the nonzero range
        # endpoint is visually closed, opposite the generic joint inference.
        sibling_fraction = (
            self.OPEN_INIT_FRACTION if self.BEHAVIOR == "open" else 0.0
        )
        for joint in self.target_fixture.joints.values():
            if joint.name == self.target_joint:
                continue
            visual_closed_qpos = joint.open_qpos
            visual_open_qpos = joint.closed_qpos
            qpos = visual_closed_qpos + sibling_fraction * (
                visual_open_qpos - visual_closed_qpos
            )
            self.sim.data.set_joint_qpos(joint.name, qpos)
        self.sim.forward()

    def _finalize_supported_articulation_robot_pose(self, distance):
        if self._fixture_mode() == "spawn_asset":
            self._stage_non_target_flaps()
            if not super()._finalize_supported_articulation_robot_pose(
                distance
            ):
                return False
            box_xy = np.asarray(
                self.target_fixture.world_center(self.sim)[:2], dtype=float
            )
            metrics = self._box_robot_pose_metrics(
                self.robot_base_pos[:2], self.robot_base_ori[2], box_xy
            )
            if metrics is None:
                return False
            (
                self._box_robot_center_distance,
                self._box_robot_facing_error,
                self._box_target_right_offset,
            ) = metrics
            return True
        del distance
        self._stage_non_target_flaps()
        spec = self.scene_task_spec()
        box_xy = np.asarray(
            self.target_fixture.world_center(self.sim)[:2], dtype=float
        )
        outward = np.asarray(spec["robot_outward_direction"], dtype=float)
        outward /= np.linalg.norm(outward)
        xy = box_xy + outward * float(spec["robot_center_distance"])
        yaw = float(np.arctan2(-outward[1], -outward[0]))
        if not self.robot_footprint_is_on_floor(xy, radius=0.25):
            return False
        if not (
            self._set_robot_free_pose(xy, yaw)
            and self._calibrate_robot_free_joint_height()
        ):
            return False
        self.sim.forward()
        if self.count_body_collisions() != 0:
            return False

        metrics = self._box_robot_pose_metrics(xy, yaw, box_xy)
        if metrics is None:
            return False
        center_distance, facing_error, target_right_offset = metrics
        if (
            facing_error > np.deg2rad(1.0)
            or target_right_offset
            < float(spec["minimum_target_right_offset"])
        ):
            return False

        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        self._box_robot_center_distance = center_distance
        self._box_robot_facing_error = facing_error
        self._box_target_right_offset = target_right_offset
        return True

    def _box_robot_pose_metrics(self, xy, yaw, box_xy):
        forward = box_xy - np.asarray(xy, dtype=float)
        center_distance = float(np.linalg.norm(forward))
        if center_distance <= 1e-8:
            return None
        forward /= center_distance
        robot_forward = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
        facing_error = float(
            np.arccos(np.clip(np.dot(robot_forward, forward), -1.0, 1.0))
        )
        operator_right = np.array([forward[1], -forward[0]], dtype=float)
        target_right_offset = float(
            np.dot(
                np.asarray(self.target_joint_pos[:2], dtype=float) - box_xy,
                operator_right,
            )
        )
        return center_distance, facing_error, target_right_offset

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["box_base_fixed"] = True
        ep_meta["box_on_floor"] = True
        ep_meta["removed_box_obstructions"] = list(
            self.removed_box_obstruction_ids
        )
        ep_meta["box_robot_center_distance"] = float(
            self._box_robot_center_distance
        )
        ep_meta["box_robot_facing_error_degrees"] = float(
            np.rad2deg(self._box_robot_facing_error)
        )
        ep_meta["box_target_right_offset"] = float(
            self._box_target_right_offset
        )
        if self.BEHAVIOR == "open":
            start_qpos = float(self.scene_task_spec()["open_start_qpos"])
            ep_meta["box_open_start_qpos"] = start_qpos
            ep_meta["box_open_start_fraction"] = float(
                (start_qpos - self.target_semantic_joint.closed_qpos)
                / (
                    self.target_semantic_joint.open_qpos
                    - self.target_semantic_joint.closed_qpos
                )
            )
        ep_meta["fixed_box_base_position"] = (
            self._fixed_box_base_position.tolist()
        )
        return ep_meta


class IThorBedroomOpenBoxFlap(BedroomBoxFlapMixin, IThorBedroomEnv):
    ATOMIC_SKILL = "OpenFlap"
    BEHAVIOR = "open"


class IThorBedroomCloseBoxFlap(BedroomBoxFlapMixin, IThorBedroomEnv):
    ATOMIC_SKILL = "CloseFlap"
    BEHAVIOR = "close"

__all__ = [
    "IThorBedroomOpenBoxFlap",
    "IThorBedroomCloseBoxFlap",
]
