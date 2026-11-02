"""Helpers for explicit living-room task calibrations."""

import numpy as np

from robocasa.models.scenes.thor_semantics import SemanticFixture, SemanticRegion


def make_task_support_region(env, task_name, label, spec):
    """Create a support region exclusively from an explicit YAML calibration."""
    center = np.asarray(spec.get("center"), dtype=float)
    half_xy = np.asarray(spec.get("half_xy"), dtype=float)
    yaw = float(spec.get("yaw", 0.0))
    body_name = spec.get("body_name")
    if center.shape != (3,) or half_xy.shape != (2,) or np.any(half_xy <= 0.0):
        raise ValueError(
            f"{env.scene_name}/{task_name}/{label}_region: "
            "expected center XYZ and positive half_xy"
        )
    if not isinstance(body_name, str) or not body_name:
        raise ValueError(
            f"{env.scene_name}/{task_name}/{label}_region: body_name is required"
        )

    fixture_id = f"{env.scene_name}:task:{task_name}:{label}_support"
    half_size = np.array([half_xy[0], half_xy[1], 0.02])
    region = SemanticRegion(
        stable_id=f"{fixture_id}:region:support",
        fixture_id=fixture_id,
        kind="support",
        parent_body=None,
        local_center=np.zeros(3),
        half_size=half_size,
        _world_center=center,
        _world_yaw=yaw,
    )
    fixture = SemanticFixture(
        stable_id=fixture_id,
        name=f"{task_name}_{label}_support",
        category="table",
        source_category="TaskCalibratedSupport",
        room_id=None,
        body_name=body_name,
        body_names=[body_name],
        site_names=[],
        asset_id=None,
        object_id=f"{task_name}_{label}_support",
        center=center,
        half_size=half_size,
        yaw=yaw,
        front=np.array([np.cos(yaw), np.sin(yaw)]),
        capabilities={"support", "task_calibrated"},
        regions={region.stable_id: region},
    )
    env.semantic_scene.add_fixture(fixture)
    return fixture, region


def set_free_joint_bottom_at(env, obj, pos, yaw=None):
    """Place a free-joint object with its bounding-box bottom at YAML XYZ."""
    pos = np.asarray(pos, dtype=float)
    if pos.shape != (3,):
        raise ValueError(f"{env.scene_name}: object position needs XYZ")
    joint_name = obj.joints[0]
    qpos = env.sim.data.get_joint_qpos(joint_name).copy()
    qpos[:2] = pos[:2]
    qpos[2] = float(pos[2] - obj.bottom_offset[2])
    if yaw is not None:
        half_yaw = 0.5 * float(yaw)
        qpos[3:7] = [np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]
    env.sim.data.set_joint_qpos(joint_name, qpos)
    joint_id = env.sim.model.joint_name2id(joint_name)
    qvel_addr = int(env.sim.model.jnt_dofadr[joint_id])
    env.sim.data.qvel[qvel_addr : qvel_addr + 6] = 0.0


def set_free_joint_oriented_bottom_at(env, obj, pos, yaw, pitch, contact_sink=0.001):
    """Place an oriented free object using its collidable-geometry bottom."""
    pos = np.asarray(pos, dtype=float)
    if pos.shape != (3,):
        raise ValueError(f"{env.scene_name}: object position needs XYZ")
    half_yaw = 0.5 * float(yaw)
    half_pitch = 0.5 * float(pitch)
    cy, sy = np.cos(half_yaw), np.sin(half_yaw)
    cp, sp = np.cos(half_pitch), np.sin(half_pitch)
    joint = obj.joints[0]
    qpos = env.sim.data.get_joint_qpos(joint).copy()
    qpos[:3] = pos
    qpos[3:7] = [cy * cp, cy * sp, sy * sp, sy * cp]
    env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.forward()

    root_id = env.sim.model.body_name2id(obj.root_body)
    body_ids = env._body_tree_ids([root_id])
    signs = np.array(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=float,
    )
    bottom = np.inf
    for geom_id in env._collidable_geoms(body_ids):
        aabb = np.asarray(env.sim.model.geom_aabb[geom_id], dtype=float)
        rotation = np.asarray(env.sim.data.geom_xmat[geom_id], dtype=float).reshape(
            3, 3
        )
        center = np.asarray(env.sim.data.geom_xpos[geom_id], dtype=float)
        corners = center + np.asarray(
            [rotation @ (aabb[:3] + aabb[3:] * sign) for sign in signs]
        )
        bottom = min(bottom, float(np.min(corners[:, 2])))
    if not np.isfinite(bottom):
        raise ValueError(f"{env.scene_name}: object has no collision geometry")

    contact_sink = float(contact_sink)
    if contact_sink < 0.0:
        raise ValueError(f"{env.scene_name}: contact sink must be non-negative")
    qpos[2] += float(pos[2] - contact_sink - bottom)
    env.sim.data.set_joint_qpos(joint, qpos)
    joint_id = env.sim.model.joint_name2id(joint)
    qvel_addr = int(env.sim.model.jnt_dofadr[joint_id])
    env.sim.data.qvel[qvel_addr : qvel_addr + 6] = 0.0


__all__ = [
    "make_task_support_region",
    "set_free_joint_bottom_at",
    "set_free_joint_oriented_bottom_at",
]
