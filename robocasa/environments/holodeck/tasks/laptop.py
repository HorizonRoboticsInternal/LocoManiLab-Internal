"""Open and close calibrated laptops in Holodeck scenes."""

import os

import numpy as np

import robocasa
from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class _HolodeckLaptopMixin:
    """Bind one calibrated laptop body and joint."""

    TASK_FAMILY = "articulation"
    HOLD_SECONDS = 1.0
    TARGET_CATEGORIES = ("Laptop",)
    TARGET_LANG = "laptop lid"
    JOINT_TOKENS = ("lid",)
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    USE_DEFAULT_ASSET_LAYOUT = False
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    NOISE_FREE_ROBOT_SPAWN = True
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    LAPTOP_ASSET = (
        "objects/thor/Living Room and Bedroom Objects/Laptop_Full/"
        "Prefabs/Laptop_26/Laptop_26.xml"
    )
    LAPTOP_ASSET_ID = "Laptop_26"

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_baked_scene_fixtures(self):
        # The downloaded Holodeck XML contains scene-specific laptop
        # placements that are either embedded in the desk or intersect nearby
        # decor. Apply the task-local calibration before semantic parsing and
        # MuJoCo compilation.
        spec = get_holodeck_task_spec(self.scene_name, self.SPEC_TASK)
        remove_scene_bodies = tuple(spec.get("remove_scene_bodies", ()))
        source_body = spec.get("source_body")
        if source_body:
            remove_scene_bodies += (source_body,)
        if remove_scene_bodies:
            removed = self.remove_scene_bodies(remove_scene_bodies)
            missing = set(remove_scene_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured scene bodies are missing: "
                    f"{sorted(missing)}"
                )
        pose = spec.get("laptop_pose")
        if spec.get("fixture_mode") != "spawn_asset" and pose is not None:
            body = self.find_scene_body(spec["target_body"])
            if body is None:
                raise RuntimeError(
                    f"{self.scene_name}: configured laptop body is missing: "
                    f"{spec['target_body']}"
                )
            body.set(
                "pos",
                " ".join(f"{float(value):.9f}" for value in pose["pos"]),
            )
            body.set(
                "quat",
                " ".join(f"{float(value):.9f}" for value in pose["quat"]),
            )
        super()._setup_baked_scene_fixtures()

    def _set_calibrated_laptop_pose(self):
        pose = self.task_spec.get("laptop_pose")
        if pose is None or self.sim is None:
            return
        body_id = self.sim.model.body_name2id(self.task_spec["target_body"])
        joint_id = int(self.sim.model.body_jntadr[body_id])
        if joint_id < 0 or int(self.sim.model.jnt_type[joint_id]) != 0:
            # Some composite workflows deliberately remove the laptop's root
            # free joint.  Its XML root pose is then the fixed base pose, while
            # the lid hinge remains the only actuated laptop joint.
            if getattr(self, "_laptop_base_fixed", False):
                return
            raise RuntimeError(
                f"{self.scene_name}: calibrated laptop body has no free joint"
            )
        joint_name = self.sim.model.joint_names[joint_id]
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(joint_name, qpos)
        self.sim.data.set_joint_qvel(joint_name, np.zeros(6, dtype=float))
        self.sim.forward()

    def _setup_task_fixtures(self):
        spec = get_holodeck_task_spec(self.scene_name, self.SPEC_TASK)
        required = [
            "target_body",
            "target_joint",
            "robot_pose",
            "closed_qpos",
            "open_qpos",
        ]
        if self.scene_name in ("train_1664", "train_2025"):
            required.append("laptop_pose")
        if spec.get("fixture_mode") == "spawn_asset":
            required.extend(("desk_body", "laptop_pose", "laptop_support_geom"))
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=required,
        )
        if self.task_spec.get("fixture_mode") == "spawn_asset":
            self._spawn_laptop()
        super()._setup_task_fixtures()
        candidates = self._semantic_candidates()
        self.target_fixture = next(
            (
                fixture
                for fixture in candidates
                if fixture.body_name == self.task_spec["target_body"]
            ),
            None,
        )
        if self.target_fixture is None:
            return
        self.fixture_refs["articulation_target"] = self.target_fixture
        pose = self.task_spec["robot_pose"]
        self.set_robot_base_pose(
            np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
        )

    def _spawn_laptop(self):
        """Add the calibrated Laptop_26 before the scene is compiled."""
        support = self.task_spec.get("laptop_support_geom")
        if support is not None:
            self.add_scene_body_geom(
                self.task_spec["desk_body"],
                name=f"{self.scene_name}_laptop_support",
                type="box",
                pos=" ".join(str(float(value)) for value in support["pos"]),
                size=" ".join(str(float(value)) for value in support["size"]),
                contype="8",
                conaffinity="15",
                group="0",
                friction="0.9 0.9 0.001",
            )
        pose = self.task_spec["laptop_pose"]
        asset_path = os.path.join(robocasa.models.assets_root, self.LAPTOP_ASSET)
        self.target_fixture = self.create_scene_fixture(
            xml_path=asset_path,
            pos=np.asarray(pose["pos"], dtype=float),
            quat=np.asarray(pose["quat"], dtype=float),
            name="spawned_laptop",
            category="Laptop",
            source_category="Laptop",
            asset_id=self.LAPTOP_ASSET_ID,
            front=np.asarray(self.task_spec.get("laptop_front", [0.0, -1.0])),
        )

    def _fix_laptop_base(self, body_name=None):
        """Make this task's laptop base static while retaining its lid hinge."""
        body_name = body_name or self.task_spec["target_body"]
        body = self.find_scene_body(body_name)
        if body is None:
            raise RuntimeError(
                f"{self.scene_name}: configured laptop body is missing: "
                f"{body_name}"
            )
        free_joint = body.find("./joint[@type='free']")
        if free_joint is not None:
            body.remove(free_joint)
        self._laptop_base_fixed = True

    def _is_scene_valid(self):
        return super()._is_scene_valid()

    def _configure_articulation_task(self):
        super()._configure_articulation_task()
        self.HOLD_STEPS = max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def _setup_scene(self):
        self._set_calibrated_laptop_pose()
        super()._setup_scene()
        if self.target_joint != self.task_spec["target_joint"]:
            raise RuntimeError(
                f"{self.scene_name}: calibrated laptop joint "
                f"{self.task_spec['target_joint']!r} does not match "
                f"{self.target_joint!r}"
            )
        joint = self.target_semantic_joint
        joint.closed_qpos = float(self.task_spec["closed_qpos"])
        joint.open_qpos = float(self.task_spec["open_qpos"])
        self.CLOSED_QPOS = joint.closed_qpos
        initial_qpos = (
            joint.closed_qpos if self.BEHAVIOR == "open" else joint.open_qpos
        )
        self.sim.data.set_joint_qpos(self.target_joint, initial_qpos)
        self.sim.forward()

    def _reset_internal(self):
        super()._reset_internal()
        self._set_calibrated_laptop_pose()
        # ``sim.reset()`` restores the compiled XML qpos, not the task-specific
        # qpos applied by ``_setup_scene()``. Reapply the calibrated laptop
        # state so subsequent episodes match the first one.
        initial_qpos = (
            self.target_semantic_joint.closed_qpos
            if self.BEHAVIOR == "open"
            else self.target_semantic_joint.open_qpos
        )
        self.sim.data.set_joint_qpos(self.target_joint, initial_qpos)
        self.sim.data.set_joint_qvel(self.target_joint, 0.0)
        self.sim.forward()

    def _place_robot_for_articulation(self, distance):
        """Apply the manually calibrated G1 pose from the scene YAML."""
        del distance
        pose = self.task_spec["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        yaw = float(pose["yaw"])
        if not self._set_robot_free_pose(pos[:2], yaw):
            # The external robosuite_models G1FloatingBody variant used by
            # this workspace has a fixed lower body and no free joint. Its
            # calibrated pose is applied to the root body before the MuJoCo
            # model is compiled, so there is nothing to update in qpos here.
            self.sim.forward()
            root_body = self.robots[0].robot_model.root_body
            root_id = self.sim.model.body_name2id(root_body)
            actual_pos = np.asarray(self.sim.data.body_xpos[root_id], dtype=float)
            if not np.allclose(actual_pos[:2], pos[:2], atol=1e-6):
                return False
        # The calibrated pose already uses the scene's valid floor height.
        # Trying to raise the robot first can put its torso into the desk even
        # when the configured pose itself is collision-free.
        if self.count_body_collisions() != 0 and not self._calibrate_robot_free_joint_height():
            return False
        if self.count_body_collisions() == 0:
            self.robot_base_pos = pos.copy()
            self.robot_base_ori = np.array([0.0, 0.0, yaw])
            return True
        return False


class HolodeckOpenLaptop(_HolodeckLaptopMixin, RoomOpenArticulation, HolodeckEnv):
    """Open the existing laptop without spawning or replacing scene objects."""

    ATOMIC_SKILL = "OpenLid"
    BEHAVIOR = "open"
    SPEC_TASK = "open_laptop"
    OPEN_INIT_FRACTION = 0.0
    OPEN_STAND_DISTANCE = 0.30
    SUCCESS_OPEN_FRACTION = 0.95


class HolodeckCloseLaptop(_HolodeckLaptopMixin, RoomOpenArticulation, HolodeckEnv):
    """Close the existing laptop without spawning or replacing scene objects."""

    ATOMIC_SKILL = "CloseLaptop"
    BEHAVIOR = "close"
    SPEC_TASK = "close_laptop"
    OPEN_INIT_FRACTION = 1.0
    CLOSE_STAND_DISTANCE = 0.30
    SUCCESS_CLOSE_FRACTION = 0.05


__all__ = ["HolodeckOpenLaptop", "HolodeckCloseLaptop"]
