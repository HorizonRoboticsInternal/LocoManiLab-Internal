"""Semantic iTHOR articulation atomic task implementations."""

import os
from pathlib import Path

import mujoco
import numpy as np

from robocasa.utils.mjcf_xml_utils import (
    mjcf_xml_bounds,
    quat_multiply,
    rotated_bounds,
)
from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin


_JOINT_TYPE_NAMES = {
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
}


class RoomOpenArticulation:
    """Open or close a semantic articulated fixture."""

    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "articulation"
    BEHAVIOR = None

    TARGET_CATEGORIES = ()
    TARGET_LANG = "object"
    JOINT_TOKENS = ()
    TARGET_JOINT_ROLES = ()
    TARGET_JOINT_TYPES = ()
    EXCLUDE_JOINT_TOKENS = ("free", "jntfree", "handle")
    CLOSED_QPOS = 0.0
    OPEN_INIT_FRACTION = 0.65
    SUCCESS_OPEN_FRACTION = 0.90
    SUCCESS_CLOSE_FRACTION = 0.05
    HOLD_STEPS = 10
    HOLD_SECONDS = None
    OPEN_STAND_DISTANCE = 0.40
    CLOSE_STAND_DISTANCE = 0.40
    DESCRIBE_TARGET_JOINT_LEVEL = False
    TARGET_JOINT_LEVELS = ()
    TARGET_JOINT_SELECTION = "first"
    SUCCESS_JOINT_SELECTION = "target"
    RANDOMIZE_TARGET_JOINT = False
    ARTICULATION_STAND_POSE_JITTER = False
    STAND_POSE_JITTER_MIN = 0.01
    STAND_POSE_JITTER_MAX = 0.1
    STAND_POSE_JITTER_ATTEMPTS = 16
    ARTICULATION_TASK_SPEC_LOADER = None
    SPEC_TASK = None

    def __init__(self, *args, scene_ids=None, **kwargs):
        supported = getattr(self, "SUPPORTED_SCENE_IDS", None)
        if self.SPEC_TASK is not None and scene_ids is None and supported is not None:
            scene_ids = supported
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        """Return the active scene's calibrated articulation parameters."""
        if self.SPEC_TASK is None or self.ARTICULATION_TASK_SPEC_LOADER is None:
            return None
        key = (self.scene_name, self.SPEC_TASK)
        if getattr(self, "_articulation_task_spec_key", None) != key:
            self.articulation_task_spec = self.ARTICULATION_TASK_SPEC_LOADER(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "target_fixture",
                    "target_joint",
                    "robot_pose",
                    "init_open_fraction",
                    (
                        "success_close_fraction"
                        if self.BEHAVIOR == "close"
                        else "success_open_fraction"
                    ),
                ),
            )
            self._articulation_task_spec_key = key
        return self.articulation_task_spec

    def _get_articulation_task_spec(self):
        """Compatibility hook for existing articulation consumers in other rooms."""
        return self._get_task_spec()

    def _get_scene_surface_calibration(self):
        if self._get_articulation_task_spec() is not None:
            return None
        return super()._get_scene_surface_calibration()

    def _get_scene_robot_pose_calibration(self):
        spec = self._get_articulation_task_spec()
        if spec is None:
            return super()._get_scene_robot_pose_calibration()
        pose = spec["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        if pos.shape != (3,) or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: robot_pose needs XYZ + yaw"
            )
        return pos, float(pose["yaw"])

    def _get_scene_robot_scale_calibration(self):
        spec = self._get_articulation_task_spec()
        if spec is None:
            return super()._get_scene_robot_scale_calibration()
        scale = float(spec.get("robot_scale", 1.0))
        if scale <= 0.0:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: robot_scale must be positive"
            )
        return scale

    def _articulation_fixture_query(self):
        """Return an exact fixture query, or ``None`` for semantic selection."""
        spec = self._get_articulation_task_spec()
        return None if spec is None else {"id": spec["target_fixture"]}

    def _articulation_robot_pose(self):
        """Return a pre-compile ``(pos, yaw)`` pose, or ``None`` to derive it."""
        spec = self._get_articulation_task_spec()
        if spec is None:
            return None
        pose = spec["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _articulation_joint_name(self):
        """Return an exact joint name, or ``None`` for semantic selection."""
        spec = self._get_articulation_task_spec()
        return None if spec is None else spec["target_joint"]

    def _configure_articulation_task(self):
        """Apply task-specific thresholds before the model is compiled."""
        spec = self._get_articulation_task_spec()
        if spec is None:
            if self.HOLD_SECONDS is not None:
                self.HOLD_STEPS = int(
                    round(float(self.HOLD_SECONDS) * self.control_freq)
                )
            return
        if "hold_seconds" in spec:
            self.HOLD_STEPS = int(
                round(float(spec["hold_seconds"]) * self.control_freq)
            )
        else:
            self.HOLD_STEPS = int(spec.get("hold_steps", type(self).HOLD_STEPS))
        self.OPEN_INIT_FRACTION = float(spec["init_open_fraction"])
        if self.BEHAVIOR == "close":
            self.SUCCESS_CLOSE_FRACTION = float(spec["success_close_fraction"])
            fractions = (self.OPEN_INIT_FRACTION, self.SUCCESS_CLOSE_FRACTION)
        else:
            self.SUCCESS_OPEN_FRACTION = float(spec["success_open_fraction"])
            fractions = (self.OPEN_INIT_FRACTION, self.SUCCESS_OPEN_FRACTION)
        if any(value < 0.0 or value > 1.0 for value in fractions):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: open fractions must be in [0, 1]"
            )
        if self.HOLD_STEPS < 1:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: invalid task calibration"
            )

    def _configure_articulation_joint(self, joint):
        """Apply task-specific semantic joint endpoints before initialization."""
        spec = self._get_articulation_task_spec()
        if spec is not None:
            if "closed_qpos" in spec:
                joint.closed_qpos = float(spec["closed_qpos"])
            if "open_qpos" in spec:
                joint.open_qpos = float(spec["open_qpos"])

    def _finalize_articulation_robot_pose(self, distance):
        """Validate or derive the compiled robot pose."""
        if self._get_articulation_task_spec() is not None:
            self.sim.forward()
            return self.count_body_collisions() == 0
        return self._place_robot_for_articulation(distance)

    def _joint_type_name(self, joint):
        if getattr(self, "sim", None) is None:
            return None
        try:
            joint_id = self.sim.model.joint_name2id(joint.name)
            joint_type = int(self.sim.model.jnt_type[joint_id])
        except Exception:
            return None
        return _JOINT_TYPE_NAMES.get(joint_type, str(joint_type))

    def _joint_type_allowed(self, joint):
        allowed = {str(value).lower() for value in self.TARGET_JOINT_TYPES}
        if not allowed:
            return True
        joint_type = self._joint_type_name(joint)
        # During fixture preselection the compiled sim may not be available yet.
        # Do not reject early there; the final selection in _setup_scene runs
        # with sim bound and applies this filter strictly.
        return joint_type is None or joint_type in allowed

    def _joint_body_ids(self, joint):
        if getattr(self, "sim", None) is None:
            return []
        model = self.sim.model
        try:
            joint_id = model.joint_name2id(joint.name)
            root_id = int(model.jnt_bodyid[joint_id])
        except Exception:
            return []
        return self._body_tree_ids([root_id])

    def _joint_target_pos(self, joint):
        body_ids = self._joint_body_ids(joint)
        if not body_ids:
            return self.target_fixture.world_center(self.sim).copy()
        model, data = self.sim.model, self.sim.data
        points = []
        points.extend(
            np.asarray(data.body_xpos[body_id], dtype=float) for body_id in body_ids
        )
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) in body_ids:
                points.append(np.asarray(data.geom_xpos[geom_id], dtype=float))
        for site_id in range(model.nsite):
            if int(model.site_bodyid[site_id]) in body_ids:
                points.append(np.asarray(data.site_xpos[site_id], dtype=float))
        return (
            np.mean(np.asarray(points), axis=0)
            if points
            else self.target_fixture.world_center(self.sim).copy()
        )

    def _geom_mesh_name(self, geom_id):
        model = self.sim.model
        try:
            mesh_id = int(model.geom_dataid[geom_id])
            if mesh_id >= 0:
                return model.mesh_id2name(mesh_id) or ""
        except Exception:
            return ""
        return ""

    def _joint_interaction_pos(self, joint):
        body_ids = set(self._joint_body_ids(joint))
        if not body_ids:
            return self._joint_target_pos(joint)
        model, data = self.sim.model, self.sim.data
        handle_points = []
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) not in body_ids:
                continue
            geom_name = model.geom_id2name(geom_id) or ""
            mesh_name = self._geom_mesh_name(geom_id)
            if "handle" in f"{geom_name} {mesh_name}".lower():
                handle_points.append(np.asarray(data.geom_xpos[geom_id], dtype=float))
        if handle_points:
            return np.mean(np.asarray(handle_points), axis=0)
        return self._joint_target_pos(joint)

    def _joint_text(self, joint):
        return f"{joint.name} {joint.role}".lower()

    def _matching_joints(self, fixture):
        includes = tuple(str(token).lower() for token in self.JOINT_TOKENS)
        roles = {str(role).lower() for role in self.TARGET_JOINT_ROLES}
        excludes = tuple(str(token).lower() for token in self.EXCLUDE_JOINT_TOKENS)
        matches = []
        for joint in fixture.joints.values():
            if roles and str(joint.role).lower() not in roles:
                continue
            text = self._joint_text(joint)
            if includes and not all(token in text for token in includes):
                continue
            if not self._joint_type_allowed(joint):
                continue
            if excludes and any(token in text for token in excludes):
                continue
            matches.append(joint)
        return matches

    def _semantic_candidates(self):
        categories = [str(category) for category in self.TARGET_CATEGORIES]
        fixtures = self.get_fixture(
            category=categories,
            capabilities=("openable",),
            return_all=True,
        )
        return [fixture for fixture in fixtures if self._matching_joints(fixture)]

    def _select_semantic_candidate(self, candidates):
        candidates = sorted(
            candidates,
            key=lambda fixture: (
                np.linalg.norm(fixture.center[:2] - self.room_center[:2]),
                fixture.name,
            ),
        )
        serialized = (
            self._ep_meta.get("semantic_refs", {})
            .get("fixtures", {})
            .get("articulation_target")
        )
        if serialized is not None:
            replay = self.get_fixture(id=serialized)
            if replay in candidates:
                return replay
        return candidates[0] if candidates else None

    def _stand_distance(self):
        return (
            self.OPEN_STAND_DISTANCE
            if self.BEHAVIOR == "open"
            else self.CLOSE_STAND_DISTANCE
        )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        query = self._articulation_fixture_query()
        self.target_fixture = (
            self.get_fixture(**query)
            if query is not None
            else self._select_semantic_candidate(self._semantic_candidates())
        )
        if self.target_fixture is None:
            return
        self.fixture_refs["articulation_target"] = self.target_fixture
        self._configure_articulation_task()
        pose = self._articulation_robot_pose()
        if pose is None:
            self.set_robot_base_from_fixture(
                self.target_fixture, distance=self._stand_distance()
            )
        else:
            self.set_robot_base_pose(*pose)

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and getattr(self, "target_fixture", None) is not None
        )

    def _setup_scene(self):
        super()._setup_scene()
        self._success_hold = 0
        if self.target_fixture is None:
            raise RuntimeError(f"{self.scene_name}: no semantic articulation target")

        joint = self._select_target_joint(self.target_fixture)
        if joint is None or joint.name not in self.sim.model.joint_names:
            raise RuntimeError(
                f"{self.scene_name}: semantic joint is missing from compiled model"
            )
        self._configure_articulation_joint(joint)
        self.target_semantic_joint = joint
        self.target_joint = joint.name
        self.target_object_name = self.target_fixture.name
        self.target_category = self.target_fixture.source_category
        self.target_asset_id = self.target_fixture.asset_id
        self.target_joint_alias = joint.role
        fixture_pos = self.target_fixture.world_center(self.sim)
        self.target_fixture_xy = fixture_pos[:2].copy()
        self.CLOSED_QPOS = float(joint.closed_qpos)
        self.target_joint_level = self._target_joint_level_descriptor(
            self.target_fixture, joint
        )
        self.target_lang = self._target_lang()

        self.sim.data.set_joint_qpos(self.target_joint, self._initial_joint_qpos(joint))
        self.sim.forward()
        self.target_joint_pos = self._joint_interaction_pos(joint)
        self.target_face_pos = self._joint_target_pos(joint)

        if not self._finalize_articulation_robot_pose(self._stand_distance()):
            raise RuntimeError(
                f"{self.scene_name}: no collision-free G1 pose for "
                f"{self.target_fixture.name}"
            )

    def _reset_internal(self):
        super()._reset_internal()
        self._success_hold = 0

    def _initial_joint_qpos(self, joint):
        if self._get_articulation_task_spec() is not None:
            return float(
                self.CLOSED_QPOS
                + self.OPEN_INIT_FRACTION * (joint.open_qpos - self.CLOSED_QPOS)
            )
        if self.BEHAVIOR == "close":
            return float(
                self.CLOSED_QPOS
                + self.OPEN_INIT_FRACTION * (joint.open_qpos - self.CLOSED_QPOS)
            )
        return float(self.CLOSED_QPOS)

    def _place_robot_for_articulation(self, distance):
        target = np.asarray(self.target_joint_pos[:2], dtype=float)
        face_target = np.asarray(self.target_face_pos[:2], dtype=float)
        distances = tuple(distance + offset for offset in (0.0, 0.15, 0.30, 0.50, 0.80))
        directions = [np.asarray(self.target_fixture.front, dtype=float)]
        base_angle = float(np.arctan2(directions[0][1], directions[0][0]))
        directions.extend(
            np.array(
                [
                    np.cos(base_angle + np.deg2rad(offset)),
                    np.sin(base_angle + np.deg2rad(offset)),
                ]
            )
            for offset in (45, -45, 90, -90, 135, -135, 180)
        )
        yaw_offsets = [0.0] + [
            np.deg2rad(deg) * sign for deg in range(10, 31, 10) for sign in (1, -1)
        ]
        for direction in directions:
            norm = np.linalg.norm(direction)
            if norm < 1e-8:
                continue
            direction = direction / norm
            right = np.array([-direction[1], direction[0]])
            for stand_distance in distances:
                for lateral in (0.0, 0.25, -0.25):
                    xy = target + direction * float(stand_distance) + right * lateral
                    for yaw_offset in yaw_offsets:
                        yaw = (
                            float(
                                np.arctan2(
                                    face_target[1] - xy[1], face_target[0] - xy[0]
                                )
                            )
                            + yaw_offset
                        )
                        if (
                            self._set_robot_free_pose(xy, yaw)
                            and self._calibrate_robot_free_joint_height()
                        ):
                            self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
                            self.robot_base_ori = np.array([0.0, 0.0, yaw])
                            self._jitter_articulation_stand_pose()
                            return True
        return False

    def _jitter_articulation_stand_pose(self):
        """Optionally perturb the final articulation stand pose in XY.

        The normal fixture-facing pose remains the fallback.  This is useful for
        demo diversity on fixed-scene articulation tasks, while keeping all
        collision handling in one shared place instead of per-task copies.
        """
        if not self.ARTICULATION_STAND_POSE_JITTER or self.deterministic_reset:
            return False
        joint = self._get_robot_free_joint()
        if joint is None:
            return False

        original_qpos = self.sim.data.get_joint_qpos(joint).copy()
        original_pos = self.robot_base_pos.copy()
        original_ori = self.robot_base_ori.copy()
        yaw = float(original_ori[2])

        for _ in range(self.STAND_POSE_JITTER_ATTEMPTS):
            angle = self.rng.uniform(0.0, 2.0 * np.pi)
            radius = self.rng.uniform(
                self.STAND_POSE_JITTER_MIN, self.STAND_POSE_JITTER_MAX
            )
            xy = original_qpos[:2] + radius * np.array([np.cos(angle), np.sin(angle)])
            if self._set_robot_free_pose(xy, yaw) and self.count_body_collisions() == 0:
                self.robot_base_pos = np.array([xy[0], xy[1], original_pos[2]])
                self.robot_base_ori = original_ori.copy()
                return True

        self.sim.data.set_joint_qpos(joint, original_qpos)
        self.robot_base_pos = original_pos
        self.robot_base_ori = original_ori
        self.sim.forward()
        return False

    def _same_role_joints(self, fixture, joint):
        return [
            candidate
            for candidate in self._matching_joints(fixture)
            if candidate.role == joint.role
        ]

    def _joint_world_z(self, joint):
        body_ids = self._joint_body_ids(joint)
        if not body_ids:
            return float(self.target_fixture.world_center(self.sim)[2])
        model, data = self.sim.model, self.sim.data
        z_values = []
        z_values.extend(float(data.body_xpos[body_id][2]) for body_id in body_ids)
        for geom_id in range(model.ngeom):
            if int(model.geom_bodyid[geom_id]) in body_ids:
                z_values.append(float(data.geom_xpos[geom_id][2]))
        for site_id in range(model.nsite):
            if int(model.site_bodyid[site_id]) in body_ids:
                z_values.append(float(data.site_xpos[site_id][2]))
        return (
            float(np.mean(z_values))
            if z_values
            else float(self.target_fixture.world_center(self.sim)[2])
        )

    def _g1_target_z(self):
        body_id = self.sim.model.body_name2id("robot0_torso_link")
        return float(self.sim.data.body_xpos[body_id][2])

    def _joint_level_descriptors(self, fixture, joints, min_spread=0.05):
        ranked = sorted(
            joints,
            key=lambda candidate: (self._joint_world_z(candidate), candidate.name),
        )
        if len(ranked) <= 1:
            return {}
        zs = [self._joint_world_z(candidate) for candidate in ranked]
        if max(zs) - min(zs) < min_spread:
            return {}
        descriptors = {}
        for index, candidate in enumerate(ranked):
            if index == 0:
                descriptors[candidate.name] = "bottom"
            elif index == len(ranked) - 1:
                descriptors[candidate.name] = "top"
            else:
                descriptors[candidate.name] = "middle"
        return descriptors

    def _target_joint_level_descriptor(self, fixture, joint):
        if not self.DESCRIBE_TARGET_JOINT_LEVEL and not self.TARGET_JOINT_LEVELS:
            return None
        levels = self._joint_level_descriptors(
            fixture, self._same_role_joints(fixture, joint)
        )
        return levels.get(joint.name)

    def _select_target_joint(self, fixture):
        candidates = self._matching_joints(fixture)
        if not candidates:
            return None

        exact_name = self._articulation_joint_name()
        if exact_name is not None:
            return next(
                (joint for joint in candidates if joint.name == exact_name),
                None,
            )

        level_filtered = False
        allowed_levels = {
            str(level).lower() for level in (self.TARGET_JOINT_LEVELS or ())
        }
        if allowed_levels:
            by_role = {}
            for joint in candidates:
                by_role.setdefault(joint.role, []).append(joint)
            filtered = []
            for joints in by_role.values():
                levels = self._joint_level_descriptors(fixture, joints)
                filtered.extend(
                    joint
                    for joint in joints
                    if levels.get(joint.name) in allowed_levels
                )
            if filtered:
                candidates = filtered
                level_filtered = True

        if str(self.TARGET_JOINT_SELECTION).lower() == "nearest" or (
            allowed_levels and not level_filtered
        ):
            ref = np.asarray(self.robot_base_pos[:2], dtype=float)
            return min(
                candidates,
                key=lambda joint: (
                    float(np.linalg.norm(self._joint_target_pos(joint)[:2] - ref)),
                    joint.name,
                ),
            )

        if self.RANDOMIZE_TARGET_JOINT and len(candidates) > 1:
            if hasattr(self.rng, "integers"):
                index = int(self.rng.integers(len(candidates)))
            else:
                index = int(self.rng.randint(len(candidates)))
            return candidates[index]
        ref_z = self._g1_target_z()
        return min(
            candidates,
            key=lambda joint: (abs(self._joint_world_z(joint) - ref_z), joint.name),
        )

    def _target_lang(self):
        level = getattr(self, "target_joint_level", None)
        if level:
            return f"{level} {self.TARGET_LANG}"
        return self.TARGET_LANG

    def _success_joints(self):
        if self._get_articulation_task_spec() is not None:
            return [self.target_semantic_joint]
        mode = str(self.SUCCESS_JOINT_SELECTION).lower()
        if mode == "matching":
            joints = self._matching_joints(self.target_fixture)
            return joints if joints else [self.target_semantic_joint]
        if mode == "same_role":
            joints = self._same_role_joints(
                self.target_fixture, self.target_semantic_joint
            )
            return joints if joints else [self.target_semantic_joint]
        return [self.target_semantic_joint]

    def _raw_success(self):
        fractions = [joint.fraction(self.sim) for joint in self._success_joints()]
        if self.BEHAVIOR == "close":
            return all(
                fraction <= self.SUCCESS_CLOSE_FRACTION for fraction in fractions
            )
        return all(fraction >= self.SUCCESS_OPEN_FRACTION for fraction in fractions)

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold = self._success_hold + 1 if self._raw_success() else 0
        return ret

    def reward(self, action=None):
        return float(self._check_success())

    def _check_success(self):
        return self._success_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"{self.BEHAVIOR} the {self.target_lang} in front of you"
        ep_meta["target_object"] = self.target_object_name
        ep_meta["target_category"] = self.target_category
        ep_meta["target_asset_id"] = self.target_asset_id
        ep_meta["target_joint"] = self.target_joint
        ep_meta["target_joint_alias"] = self.target_joint_alias
        ep_meta["target_joint_pos"] = [float(v) for v in self.target_joint_pos]
        if self.target_joint_level is not None:
            ep_meta["target_joint_level"] = self.target_joint_level
        ep_meta["target_fixture_xy"] = [float(v) for v in self.target_fixture_xy]
        ep_meta["robot_after_xy"] = [float(v) for v in self._robot_base_xy()]
        ep_meta["robot_yaw"] = self._robot_base_yaw()
        return ep_meta


class RoomCloseLaptop(RoomOpenArticulation):
    """Close a laptop staged on a table in front of G1."""

    ATOMIC_SKILL = "CloseLaptop"
    BEHAVIOR = "close"
    TARGET_CATEGORIES = ("Laptop",)
    TARGET_LANG = "laptop lid"
    JOINT_TOKENS = ("lid",)
    OPEN_INIT_FRACTION = 1.0
    MIN_LAPTOP_SURFACE_HEIGHT = 0.35
    MAX_LAPTOP_SURFACE_HEIGHT = 1.20
    TABLE_TOKENS = ("table", "desk")
    SEAT_TOKENS = ("stool", "chair")
    SEAT_CLEAR_RADIUS = 1.25
    LAPTOP_TABLE_MARGIN = 0.06
    LAPTOP_STAND_DISTANCE = 0.3
    LAPTOP_CENTER = (-2.78, -2.502, 0.65)
    LAPTOP_XML = (
        "objects/thor/Living Room and Bedroom Objects/Laptop_Full/"
        "Prefabs/Laptop_13/Laptop_13.xml"
    )
    LEGACY_LAPTOP_XML = (
        "objects/thor/20251117/Living Room and Bedroom Objects/Laptop_Full/"
        "Prefabs/Laptop_13/Laptop_13.xml"
    )
    LAPTOP_EDGE_FRACTION = 0.70
    LAPTOP_TOWARD_ROBOT_OFFSET = 0.0
    LAPTOP_SURFACE_CLEARANCE = 0.006
    APPROACH_LONG_EDGE_ONLY = False

    def _setup_task_fixtures(self):
        spawn_pos = self.robot_base_pos.copy()
        self._source_robot_xy = spawn_pos[:2].copy()
        self._laptop_xml_path = self._laptop_asset_xml_path()
        (
            self._laptop_local_center,
            self._laptop_half_size,
            self._laptop_local_bottom,
        ) = mjcf_xml_bounds(self._laptop_xml_path)

        (
            self.work_surface_region,
            self.work_surface_fixture,
        ) = self._select_laptop_surface()
        if self.work_surface_region is None:
            self.target_fixture = None
            return

        self.fixture_refs["laptop_surface"] = self.work_surface_fixture
        self.region_refs["laptop_surface"] = self.work_surface_region
        self._set_planned_robot_pose()

        self._clear_work_surface()
        self._remove_seats_near_table_approaches()
        self._remove_seats_near(self._laptop_xy_on_surface())
        self._create_laptop_on_surface()
        self.fixture_refs["articulation_target"] = self.target_fixture

    def _laptop_asset_xml_path(self):
        package_assets = Path(__file__).resolve().parents[2] / "models" / "assets"
        candidates = [package_assets / self.LAPTOP_XML]
        cache_root = os.environ.get("MLSPACES_CACHE_DIR")
        if cache_root:
            candidates.extend(
                [
                    Path(cache_root) / self.LAPTOP_XML,
                    Path(cache_root) / self.LEGACY_LAPTOP_XML,
                ]
            )
        for path in candidates:
            if path.is_file():
                return str(path)
        checked = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"missing laptop asset; checked: {checked}")

    def _laptop_planar_half_size(self):
        if self.LAPTOP_CENTER is not None:
            return np.asarray(self._laptop_half_size[:2], dtype=float)
        # The laptop is rotated upright about X. Its horizontal extents then
        # come from local X and Z; use a square bound so either table edge works.
        extent = float(max(self._laptop_half_size[0], self._laptop_half_size[2]))
        return np.full(2, extent, dtype=float)

    def _select_laptop_surface(self):
        requirements = dict(
            height_range=(
                self.MIN_LAPTOP_SURFACE_HEIGHT,
                self.MAX_LAPTOP_SURFACE_HEIGHT,
            ),
            min_half_xy=(self._laptop_planar_half_size() + self.LAPTOP_TABLE_MARGIN),
            body_keywords=self.TABLE_TOKENS,
        )
        regions = sorted(
            self.get_region(kind="support", return_all=True),
            key=lambda region: region.stable_id,
        )
        for region in regions:
            fixture = self.get_fixture(id=region.fixture_id)
            if fixture is not None and self._support_matches_requirements(
                region, fixture, requirements
            ):
                return region, fixture
        return None, None

    def _table_approach_directions(self):
        if self.APPROACH_LONG_EDGE_ONLY:
            return list(
                self.support_region_long_edge_approach_directions(
                    self.work_surface_region,
                    toward_xy=self._source_robot_xy,
                )
            )
        center, yaw = self.work_surface_region.world_pose()
        axes = np.array(
            [
                [np.cos(yaw), np.sin(yaw)],
                [-np.sin(yaw), np.cos(yaw)],
            ],
            dtype=float,
        )
        directions = np.vstack((axes, -axes))
        toward_spawn = np.asarray(self._source_robot_xy, dtype=float) - center[:2]
        if np.linalg.norm(toward_spawn) < 1e-6:
            toward_spawn = np.asarray(self.work_surface_fixture.front, dtype=float)
        return sorted(directions, key=lambda d: -float(np.dot(d, toward_spawn)))

    def _set_planned_robot_pose(self):
        center, _ = self.work_surface_region.world_pose()
        direction = np.asarray(self._table_approach_directions()[0], dtype=float)
        distance = (
            self.support_region_radial_extent(self.work_surface_region, direction)
            + self.LAPTOP_STAND_DISTANCE
        )
        xy = center[:2] + direction * distance
        yaw = float(np.arctan2(center[1] - xy[1], center[0] - xy[0]))
        self.set_robot_base_pose([xy[0], xy[1], 0.0], yaw)

    def _laptop_surface_pose(self):
        if self.LAPTOP_CENTER is None:
            toward_xy = np.asarray(self.robot_base_pos[:2], dtype=float)
            edge_direction = None
            laptop_half = self._laptop_planar_half_size()
            lateral_fraction = 0.0
            if self.APPROACH_LONG_EDGE_ONLY:
                center, _ = self.work_surface_region.world_pose(
                    getattr(self, "sim", None)
                )
                edge_direction = self.support_region_long_edge_approach_directions(
                    self.work_surface_region,
                    toward_xy=toward_xy,
                    sim=getattr(self, "sim", None),
                )[0]
                lateral = np.array(
                    [-edge_direction[1], edge_direction[0]],
                    dtype=float,
                )
                lateral_offset = float(np.dot(toward_xy - center[:2], lateral))
                toward_xy = center[:2] + edge_direction + lateral * lateral_offset

                long_axis = int(
                    np.argmax(
                        np.asarray(
                            self.work_surface_region.half_size[:2],
                            dtype=float,
                        )
                    )
                )
                short_axis = 1 - long_axis
                laptop_half = np.zeros(2, dtype=float)
                laptop_half[long_axis] = self._laptop_half_size[0]
                laptop_half[short_axis] = self._laptop_half_size[2]
                lateral_fraction = 1.0
            center, desired_front = self.support_region_edge_pose(
                self.work_surface_region,
                laptop_half,
                toward_xy,
                margin=self.LAPTOP_TABLE_MARGIN,
                edge_fraction=self.LAPTOP_EDGE_FRACTION,
                sim=getattr(self, "sim", None),
                lateral_fraction=lateral_fraction,
                edge_direction=edge_direction,
            )
            center = self._offset_laptop_surface_pose_toward_robot(
                center, desired_front
            )
            return center, desired_front

        center = np.asarray(self.LAPTOP_CENTER, dtype=float)
        desired_front = np.asarray(self.robot_base_pos[:2], dtype=float) - center[:2]
        if np.linalg.norm(desired_front) < 1e-6:
            desired_front = np.asarray(self.work_surface_fixture.front, dtype=float)
        desired_front = desired_front / (np.linalg.norm(desired_front) + 1e-9)
        return center[:2], desired_front

    def _offset_laptop_surface_pose_toward_robot(self, center_xy, desired_front):
        offset = float(self.LAPTOP_TOWARD_ROBOT_OFFSET)
        if offset <= 0.0:
            return center_xy
        desired_front = np.asarray(desired_front, dtype=float)[:2]
        norm = float(np.linalg.norm(desired_front))
        if norm < 1e-8:
            return center_xy
        return np.asarray(center_xy, dtype=float) + desired_front / norm * offset

    def _laptop_xy_on_surface(self):
        return self._laptop_surface_pose()[0]

    def _laptop_pose_on_surface(self):
        center_xy, desired_front = self._laptop_surface_pose()
        source_front = np.array([0.0, -1.0])
        delta = float(
            np.arctan2(desired_front[1], desired_front[0])
            - np.arctan2(source_front[1], source_front[0])
        )
        yaw_quat = np.array([np.cos(delta / 2), 0.0, 0.0, np.sin(delta / 2)])
        upright_quat = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
        quat = quat_multiply(yaw_quat, upright_quat)
        local_center, rotated_half, _ = rotated_bounds(
            self._laptop_local_center, self._laptop_half_size, quat
        )
        if self.LAPTOP_CENTER is None:
            surface_center, _ = self.work_surface_region.world_pose(
                getattr(self, "sim", None)
            )
            center = np.array(
                [
                    center_xy[0],
                    center_xy[1],
                    surface_center[2] + rotated_half[2] + self.LAPTOP_SURFACE_CLEARANCE,
                ],
                dtype=float,
            )
        else:
            center = np.asarray(self.LAPTOP_CENTER, dtype=float)
        target_root_pos = center - local_center
        return target_root_pos, quat, desired_front

    def _create_laptop_on_surface(self):
        target_root_pos, quat, front = self._laptop_pose_on_surface()
        self.target_fixture = self.create_scene_fixture(
            xml_path=self._laptop_xml_path,
            pos=target_root_pos,
            quat=quat,
            name="spawned_laptop",
            category="Laptop",
            source_category="Laptop",
            asset_id="Laptop_13",
            front=front,
            half_size=self._laptop_half_size,
        )

    def _clear_work_surface(self):
        self.clear_support_region_clutter(
            self.work_surface_region,
            self.work_surface_fixture,
            protected_fixtures=(self.work_surface_fixture,),
            z_below=0.18,
            z_above=0.80,
        )

    def _remove_seats_near(self, xy):
        self.remove_semantic_fixtures_near(self.SEAT_TOKENS, xy, self.SEAT_CLEAR_RADIUS)

    def _remove_seats_near_table_approaches(self):
        center, _ = self.work_surface_region.world_pose()
        for direction in self._table_approach_directions():
            xy = center[:2] + direction * (
                self.support_region_radial_extent(self.work_surface_region, direction)
                + self.LAPTOP_STAND_DISTANCE
            )
            self._remove_seats_near(xy)

    def _is_scene_valid(self):
        if not super()._is_scene_valid():
            self._scene_invalid_reason = f"{self.scene_name}: no laptop target"
            return False
        if getattr(self, "work_surface_fixture", None) is None:
            self._scene_invalid_reason = f"{self.scene_name}: no laptop support surface"
            return False
        return True

    def _laptop_free_joint(self):
        body_id = self.sim.model.body_name2id(self.target_fixture.body_name)
        joint_id = int(self.sim.model.body_jntadr[body_id])
        if joint_id < 0 or int(self.sim.model.jnt_type[joint_id]) != 0:
            raise RuntimeError(f"{self.scene_name}: laptop has no free joint")
        return self.sim.model.joint_names[joint_id]

    def _initial_jitter_extra_joints(self):
        joints = list(super()._initial_jitter_extra_joints())
        if getattr(self, "sim", None) is not None and self.target_fixture is not None:
            joints.append(self._laptop_free_joint())
        return tuple(joints)

    def _move_laptop_on_surface(self):
        root_joint = self._laptop_free_joint()
        qpos = self.sim.data.get_joint_qpos(root_joint).copy()
        target_root_pos, quat, _ = self._laptop_pose_on_surface()
        qpos[:3] = target_root_pos
        qpos[3:7] = quat
        self.sim.data.set_joint_qpos(root_joint, qpos)
        self.sim.forward()

    def _set_laptop_lid_limits(self):
        joint_id = self.sim.model.joint_name2id(self.target_joint)
        lo, hi = (float(v) for v in self.sim.model.jnt_range[joint_id])
        self.target_semantic_joint.closed_qpos = hi
        self.target_semantic_joint.open_qpos = lo
        self.CLOSED_QPOS = hi

    def _setup_scene(self):
        super()._setup_scene()
        self._move_laptop_on_surface()
        self.clear_support_region_clutter(
            self.work_surface_region,
            self.work_surface_fixture,
            protected_body_roots=[
                self.sim.model.body_name2id(self.target_fixture.body_name)
            ],
            z_below=0.18,
            z_above=0.80,
        )
        self._set_laptop_lid_limits()
        self.target_fixture_xy = self.target_fixture.world_center(self.sim)[:2].copy()
        self.target_joint_pos = self._joint_target_pos(self.target_semantic_joint)
        self.sim.data.set_joint_qpos(
            self.target_joint, self._initial_joint_qpos(self.target_semantic_joint)
        )
        self.sim.forward()

    def _place_robot_for_articulation(self, distance):
        del distance
        return self.place_robot_at_fixture(
            self.work_surface_fixture,
            distances=(self.LAPTOP_STAND_DISTANCE, 0.7, 0.85),
            lateral_offsets=(0.0, 0.2, -0.2, 0.35, -0.35),
            max_yaw_offset_deg=20,
            approach_directions=self._table_approach_directions(),
        )


class ExactSceneArticulation(SceneSpecMixin, RoomOpenArticulation):
    """Open or close one exact fixture joint selected per scene."""

    TARGET_CATEGORIES = ()
    TARGET_LANG = "articulation"
    OPEN_STAND_DISTANCE = 0.45
    CLOSE_STAND_DISTANCE = 0.45
    SUCCESS_OPEN_FRACTION = 0.80
    SUCCESS_CLOSE_FRACTION = 0.12
    OPEN_INIT_FRACTION = 0.85
    HOLD_STEPS = 5

    def _articulation_fixture_query(self):
        return {"id": self.scene_task_spec()["fixture"]}

    def _articulation_joint_name(self):
        return self.scene_task_spec()["joint"]

    def _target_lang(self):
        return self.scene_task_spec()["target"]

    def _finalize_articulation_robot_pose(self, distance):
        """Place G1 from the joint position, not the often-huge fixture AABB."""

        del distance
        target = np.asarray(self.target_joint_pos, dtype=float)
        best = None
        for radius in (0.38, 0.45, 0.52, 0.60, 0.70):
            for angle in np.linspace(-np.pi, np.pi, 16, endpoint=False):
                xy = target[:2] + radius * np.array(
                    [np.cos(angle), np.sin(angle)]
                )
                yaw = float(np.arctan2(target[1] - xy[1], target[0] - xy[0]))
                if not (
                    self._set_robot_free_pose(xy, yaw)
                    and self._calibrate_robot_free_joint_height()
                    and self.count_body_collisions() == 0
                ):
                    continue
                reach = min(
                    float(np.linalg.norm(target - eef))
                    for eef in self.eef_positions()
                )
                candidate = (reach, radius, xy.copy(), yaw)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
        if best is None:
            return False
        _, _, xy, yaw = best
        self._set_robot_free_pose(xy, yaw)
        self._calibrate_robot_free_joint_height()
        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        return self.count_body_collisions() == 0

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = self.scene_task_spec()["lang"]
        return ep_meta


__all__ = ["ExactSceneArticulation", "RoomCloseLaptop", "RoomOpenArticulation"]
