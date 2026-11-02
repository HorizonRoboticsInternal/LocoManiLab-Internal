"""Geometry helpers for exact free-joint task fixtures.

The mixin has no environment base class or lifecycle hooks. Room tasks can use
it to seat a spawned or native fixture on a named physical support without
duplicating MuJoCo free-joint and contact-probe logic.
"""

from __future__ import annotations

import itertools

import mujoco
import numpy as np

from robocasa.models.scenes.thor_semantics import SemanticRegion


class FreeFixturePlacementMixin:
    """Place one-free-joint fixtures using bounded, physical contact probes."""

    CONTACT_PENETRATION_TOLERANCE = 0.003

    def _fixture_free_joint_name(self, fixture):
        joints = set()
        for body_id in self._fixture_body_ids(fixture):
            start = int(self.sim.model.body_jntadr[body_id])
            for joint_id in range(
                start,
                start + int(self.sim.model.body_jntnum[body_id]),
            ):
                if int(self.sim.model.jnt_type[joint_id]) == int(
                    mujoco.mjtJoint.mjJNT_FREE
                ):
                    joints.add(self.sim.model.joint_id2name(joint_id))
        if len(joints) != 1:
            raise RuntimeError(
                f"{fixture.stable_id}: expected one free joint, found "
                f"{sorted(joints)}"
            )
        return next(iter(joints))

    def _zero_fixture_velocity(self, fixture):
        joint = self._fixture_free_joint_name(fixture)
        joint_id = self.sim.model.joint_name2id(joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.data.qvel[dof_address : dof_address + 6] = 0.0

    def _fixture_bbox_points(self, fixture):
        if fixture.local_center is None or fixture.local_half_size is None:
            raise RuntimeError(
                f"{fixture.stable_id}: semantic local bounds are unavailable"
            )
        body_id = self.sim.model.body_name2id(fixture.body_name)
        body_pos = np.asarray(self.sim.data.body_xpos[body_id], dtype=float)
        body_mat = np.asarray(
            self.sim.data.body_xmat[body_id], dtype=float
        ).reshape(3, 3)
        center = np.asarray(fixture.local_center, dtype=float)
        half = np.asarray(fixture.local_half_size, dtype=float)
        return np.asarray(
            [
                body_pos + body_mat @ (center + half * np.asarray(signs))
                for signs in itertools.product((-1.0, 1.0), repeat=3)
            ],
            dtype=float,
        )

    def _set_fixture_center_xy_and_bbox_bottom(
        self,
        fixture,
        center_xy,
        bottom_z,
    ):
        joint = self._fixture_free_joint_name(fixture)
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        current_center = np.asarray(fixture.world_center(self.sim), dtype=float)
        current_bottom = float(np.min(self._fixture_bbox_points(fixture)[:, 2]))
        qpos[:2] += np.asarray(center_xy, dtype=float)[:2] - current_center[:2]
        qpos[2] += float(bottom_z) - current_bottom
        self.sim.data.set_joint_qpos(joint, qpos)
        self._zero_fixture_velocity(fixture)
        self.sim.forward()

    def _set_fixture_quaternion(self, fixture, quaternion):
        quaternion = np.asarray(quaternion, dtype=float)
        norm = float(np.linalg.norm(quaternion))
        if quaternion.shape != (4,) or norm <= 1e-8:
            raise ValueError(
                "native fixture quaternion must be four nonzero values"
            )
        joint = self._fixture_free_joint_name(fixture)
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[3:7] = quaternion / norm
        self.sim.data.set_joint_qpos(joint, qpos)
        self._zero_fixture_velocity(fixture)
        self.sim.forward()

    def _fixture_contact_distances(self, subject, support_body_ids):
        subject_bodies = self._fixture_body_ids(subject, include_siblings=True)
        support_bodies = {int(body_id) for body_id in support_body_ids}
        distances = []
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body1 in subject_bodies and body2 in support_bodies) or (
                body2 in subject_bodies and body1 in support_bodies
            ):
                distances.append(float(contact.dist))
        return distances

    def _fixture_contacts_fixture(self, subject, target):
        return bool(
            self._fixture_contact_distances(
                subject,
                self._fixture_body_ids(target, include_siblings=True),
            )
        )

    def _fixture_has_illegal_penetration(
        self,
        subject,
        allowed_body_roots,
        tolerance=None,
    ):
        tolerance = (
            self.CONTACT_PENETRATION_TOLERANCE
            if tolerance is None
            else float(tolerance)
        )
        root_id = self.sim.model.body_name2id(subject.body_name)
        allowed_ids = tuple(
            int(root)
            if isinstance(root, (int, np.integer))
            else self.sim.model.body_name2id(root)
            for root in allowed_body_roots
        )
        return bool(
            self.illegal_contact_pairs_for_bodies(
                [root_id],
                allowed_ids,
                tolerance=tolerance,
            )
        )

    def _fixture_bbox_inside_region_xy(self, fixture, region, margin=0.0):
        return all(
            region.contains_xy(point, self.sim, margin=margin)
            for point in self._fixture_bbox_points(fixture)
        )

    def _place_fixture_on_support(
        self,
        fixture,
        region,
        support_fixture,
        center_xy,
        *,
        contract_name,
        probe_above=0.08,
        probe_below=0.30,
        probe_step=0.002,
    ):
        if not isinstance(region, SemanticRegion) or region.kind != "support":
            raise ValueError("native fixture placement requires a support region")
        if support_fixture.stable_id != region.fixture_id:
            raise ValueError("support fixture does not own the placement region")
        support_roots = tuple(support_fixture.body_names)
        support_bodies = self._fixture_body_ids(
            support_fixture, include_siblings=True
        )
        anchor_z = float(region.world_pose(self.sim)[0][2])
        for bottom_z in np.arange(
            anchor_z + float(probe_above),
            anchor_z - float(probe_below) - 0.5 * float(probe_step),
            -float(probe_step),
        ):
            self._set_fixture_center_xy_and_bbox_bottom(
                fixture,
                center_xy,
                bottom_z,
            )
            if not self._fixture_bbox_inside_region_xy(fixture, region):
                continue
            distances = self._fixture_contact_distances(fixture, support_bodies)
            if not distances or min(distances) < -self.CONTACT_PENETRATION_TOLERANCE:
                continue
            if self._fixture_has_illegal_penetration(fixture, support_roots):
                continue
            self._zero_fixture_velocity(fixture)
            self.register_task_body_contact_contract(
                contract_name,
                fixture.body_name,
                support_roots,
            )
            return True
        return False

    def _place_fixture_on_floor(
        self,
        fixture,
        center_xy,
        *,
        contract_name,
        probe_above=0.05,
        probe_below=0.30,
        probe_step=0.002,
    ):
        for bottom_z in np.arange(
            float(probe_above),
            -float(probe_below) - 0.5 * float(probe_step),
            -float(probe_step),
        ):
            self._set_fixture_center_xy_and_bbox_bottom(
                fixture,
                center_xy,
                bottom_z,
            )
            points = self._fixture_bbox_points(fixture)
            if not all(self.xy_is_on_floor(point[:2]) for point in points):
                continue
            distances = self._fixture_contact_distances(fixture, {0})
            if not distances or min(distances) < -self.CONTACT_PENETRATION_TOLERANCE:
                continue
            if self._fixture_has_illegal_penetration(fixture, (0,)):
                continue
            self._zero_fixture_velocity(fixture)
            self.register_task_body_contact_contract(
                contract_name,
                fixture.body_name,
                (0,),
            )
            return True
        return False


__all__ = ["FreeFixturePlacementMixin"]
