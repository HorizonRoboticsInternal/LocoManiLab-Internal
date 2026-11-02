"""Bedroom-specific behavior layered on shared articulation primitives."""

import xml.etree.ElementTree as ET

import numpy as np

from robocasa.environments.ithor_atomic.articulation import ExactSceneArticulation


class BedroomArticulationMixin(ExactSceneArticulation):
    """Format the exact per-scene language after behavior is selected."""

    def scene_task_spec(self):
        spec = dict(super().scene_task_spec())
        spec["lang"] = spec["lang"].format(verb=self.BEHAVIOR)
        return spec


class BedroomTwentyEightyArticulationMixin:
    """Use a short 20%--80% stroke for drawers and small doors."""

    OPEN_START_FRACTION = 0.20
    CLOSE_START_FRACTION = 0.80
    SUCCESS_OPEN_FRACTION = 0.80
    SUCCESS_CLOSE_FRACTION = 0.20

    def _initial_joint_qpos(self, joint):
        if self.BEHAVIOR == "open":
            fraction = self.OPEN_START_FRACTION
        elif self.BEHAVIOR == "close":
            fraction = self.CLOSE_START_FRACTION
        else:
            raise ValueError(f"unsupported articulation behavior: {self.BEHAVIOR!r}")
        return float(
            joint.closed_qpos
            + fraction * (joint.open_qpos - joint.closed_qpos)
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["initial_open_fraction"] = float(
            self.OPEN_START_FRACTION
            if self.BEHAVIOR == "open"
            else self.CLOSE_START_FRACTION
        )
        ep_meta["success_fraction_threshold"] = float(
            self.SUCCESS_OPEN_FRACTION
            if self.BEHAVIOR == "open"
            else self.SUCCESS_CLOSE_FRACTION
        )
        return ep_meta


class BedroomPartiallyOpenStartMixin:
    """Start an opening task with a small, physically specified gap."""

    INITIAL_OPEN_ANGLE_DEGREES = 30.0

    def _initial_joint_qpos(self, joint):
        if self.BEHAVIOR != "open":
            return super()._initial_joint_qpos(joint)
        travel = float(joint.open_qpos - joint.closed_qpos)
        if abs(travel) <= 1e-8:
            raise RuntimeError(f"{self.scene_name}: articulation has no travel")
        opening = min(np.deg2rad(self.INITIAL_OPEN_ANGLE_DEGREES), abs(travel))
        return float(joint.closed_qpos + np.sign(travel) * opening)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        if self.BEHAVIOR == "open":
            ep_meta["initial_open_angle_degrees"] = float(
                self.INITIAL_OPEN_ANGLE_DEGREES
            )
        return ep_meta


class BedroomSupportedFixtureMixin:
    """Dynamically seat a task fixture on a named collidable support."""

    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    SUPPORT_PROBE_ABOVE = 0.03
    SUPPORT_PROBE_BELOW = 0.03
    SUPPORT_PROBE_STEP = 0.0005
    SUPPORT_PENETRATION_TOLERANCE = 0.0025

    def _setup_scene(self):
        """Seat the fixture before deriving its final interaction pose."""

        self._defer_supported_articulation_robot_pose = True
        try:
            super()._setup_scene()
        finally:
            self._defer_supported_articulation_robot_pose = False

        try:
            self._place_supported_articulation_fixture()
        finally:
            self._restore_supported_articulation_robot()

        self._refresh_supported_articulation_targets()
        if not self._finalize_articulation_robot_pose(self._stand_distance()):
            raise RuntimeError(
                f"{self.scene_name}: no collision-free G1 pose for "
                f"{self.target_fixture.name}"
            )
        contract_name, subject_roots, support_roots = (
            self._supported_fixture_contact_contract()
        )
        self.register_task_body_contact_contract(
            contract_name,
            subject_roots,
            support_roots,
        )

    def _place_supported_articulation_fixture(self):
        raise NotImplementedError

    def _supported_fixture_contact_contract(self):
        raise NotImplementedError

    def _park_supported_articulation_robot(self):
        joint = self._get_robot_free_joint()
        if joint is None:
            return False
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        joint_id = self.sim.model.joint_name2id(joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        qvel = self.sim.data.qvel[dof_address : dof_address + 6].copy()
        self._supported_articulation_robot_state = (joint, qpos, qvel)
        parked = qpos.copy()
        parked[2] += 5.0
        self.sim.data.set_joint_qpos(joint, parked)
        self.sim.data.qvel[dof_address : dof_address + 6] = 0.0
        self.sim.forward()
        return True

    def _restore_supported_articulation_robot(self):
        state = getattr(self, "_supported_articulation_robot_state", None)
        if state is None:
            return
        joint, qpos, qvel = state
        self.sim.data.set_joint_qpos(joint, qpos)
        joint_id = self.sim.model.joint_name2id(joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[dof_address : dof_address + 6] = qvel
        del self._supported_articulation_robot_state
        self.sim.forward()

    def _refresh_supported_articulation_targets(self):
        fixture_pos = self.target_fixture.world_center(self.sim)
        self.target_fixture_xy = fixture_pos[:2].copy()
        self.target_joint_level = self._target_joint_level_descriptor(
            self.target_fixture,
            self.target_semantic_joint,
        )
        self.target_joint_pos = self._joint_interaction_pos(
            self.target_semantic_joint
        )
        self.target_face_pos = self._joint_target_pos(self.target_semantic_joint)

    def _finalize_articulation_robot_pose(self, distance):
        if getattr(self, "_defer_supported_articulation_robot_pose", False):
            return self._park_supported_articulation_robot()
        return self._finalize_supported_articulation_robot_pose(distance)

    def _finalize_supported_articulation_robot_pose(self, distance):
        return super()._finalize_articulation_robot_pose(distance)

    def _ensure_fixture_free_joint(self, fixture, joint_name):
        root = self.find_scene_body(fixture.body_name)
        if root is None:
            raise RuntimeError(
                f"{self.scene_name}: missing fixture root {fixture.body_name}"
            )
        free_joints = [
            joint
            for joint in root.findall("./joint")
            if joint.get("type") == "free" or "jntfree" in joint.get("name", "")
        ]
        if len(free_joints) > 1:
            raise RuntimeError(
                f"{self.scene_name}: fixture has multiple free joints"
            )
        if not free_joints:
            root.insert(0, ET.Element("joint", name=joint_name, type="free"))

    def _fixture_free_joint_name(self, fixture):
        body_id = self.sim.model.body_name2id(fixture.body_name)
        start = int(self.sim.model.body_jntadr[body_id])
        count = int(self.sim.model.body_jntnum[body_id])
        free = [
            joint_id
            for joint_id in range(start, start + count)
            if int(self.sim.model.jnt_type[joint_id]) == 0
        ]
        if len(free) != 1:
            raise RuntimeError(
                f"{self.scene_name}: supported fixture needs one free joint"
            )
        return self.sim.model.joint_names[free[0]]

    def _fixture_support_contact_is_valid(
        self, fixture, support_body_roots, penetration_tolerance
    ):
        subject_root = self.sim.model.body_name2id(fixture.body_name)
        subject_bodies = self._body_tree_ids([subject_root])
        support_roots = [
            int(root)
            if isinstance(root, (int, np.integer))
            else self.sim.model.body_name2id(root)
            for root in support_body_roots
        ]
        support_bodies = self._body_tree_ids(
            [root for root in support_roots if root != 0]
        )
        if 0 in support_roots:
            # MuJoCo's floor plane belongs to the world body. Expanding body
            # zero as a normal subtree would incorrectly include the scene.
            support_bodies.add(0)
        distances = []
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body_1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body_2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body_1 in subject_bodies and body_2 in support_bodies) or (
                body_2 in subject_bodies and body_1 in support_bodies
            ):
                distances.append(float(contact.dist))
        illegal = self.illegal_contact_pairs_for_bodies(
            (subject_root,),
            tuple(support_roots),
            tolerance=penetration_tolerance,
        )
        return bool(
            distances
            and min(distances) >= -float(penetration_tolerance)
            and not illegal
        )

    def _collidable_fixture_geom_points(self, fixture):
        """Return AABB points only for collision-enabled fixture geometry."""

        root_id = self.sim.model.body_name2id(fixture.body_name)
        body_ids = self._body_tree_ids([root_id])
        signs = np.array(
            [
                [x, y, z]
                for x in (-1.0, 1.0)
                for y in (-1.0, 1.0)
                for z in (-1.0, 1.0)
            ],
            dtype=float,
        )
        points = []
        for geom_id in range(self.sim.model.ngeom):
            if int(self.sim.model.geom_bodyid[geom_id]) not in body_ids:
                continue
            if not (
                int(self.sim.model.geom_contype[geom_id])
                or int(self.sim.model.geom_conaffinity[geom_id])
            ):
                continue
            aabb = np.asarray(self.sim.model.geom_aabb[geom_id], dtype=float)
            rotation = np.asarray(
                self.sim.data.geom_xmat[geom_id], dtype=float
            ).reshape(3, 3)
            center = np.asarray(
                self.sim.data.geom_xpos[geom_id], dtype=float
            )
            points.extend(
                center + rotation @ (aabb[:3] + aabb[3:] * sign)
                for sign in signs
            )
        if not points:
            raise RuntimeError(
                f"{self.scene_name}: supported fixture has no collision geometry"
            )
        return np.asarray(points, dtype=float)

    def _place_fixture_on_support_contact(
        self,
        fixture,
        support_body_roots,
        support_top_z,
        *,
        probe_below=None,
    ):
        root_joint = self._fixture_free_joint_name(fixture)
        original = self.sim.data.get_joint_qpos(root_joint).copy()
        probe_below = (
            self.SUPPORT_PROBE_BELOW
            if probe_below is None
            else float(probe_below)
        )
        if probe_below < 0.0:
            raise ValueError("fixture support probe_below must be nonnegative")
        z_values = np.arange(
            float(support_top_z) + self.SUPPORT_PROBE_ABOVE,
            float(support_top_z) - probe_below
            - 0.5 * self.SUPPORT_PROBE_STEP,
            -self.SUPPORT_PROBE_STEP,
        )
        for bottom_z in z_values:
            self.sim.data.set_joint_qpos(root_joint, original)
            self.sim.forward()
            points = self._collidable_fixture_geom_points(fixture)
            candidate = original.copy()
            candidate[2] += float(bottom_z) - float(np.min(points[:, 2]))
            self.sim.data.set_joint_qpos(root_joint, candidate)
            self.sim.forward()
            if 0 in support_body_roots:
                points = self._collidable_fixture_geom_points(fixture)
                if not all(self.xy_is_on_floor(point[:2]) for point in points):
                    continue
            if not self._fixture_support_contact_is_valid(
                fixture,
                support_body_roots,
                self.SUPPORT_PENETRATION_TOLERANCE,
            ):
                continue
            joint_id = self.sim.model.joint_name2id(root_joint)
            dof_address = int(self.sim.model.jnt_dofadr[joint_id])
            self.sim.data.qvel[dof_address : dof_address + 6] = 0.0
            self.sim.forward()
            return True
        self.sim.data.set_joint_qpos(root_joint, original)
        self.sim.forward()
        return False


__all__ = [
    "BedroomArticulationMixin",
    "BedroomPartiallyOpenStartMixin",
    "BedroomSupportedFixtureMixin",
    "BedroomTwentyEightyArticulationMixin",
]
