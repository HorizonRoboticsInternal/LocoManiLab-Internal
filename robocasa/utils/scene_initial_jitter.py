"""Protocol-based initial-pose randomization for scene environments."""

import numpy as np


JITTER_RADIUS = 0.10
MAX_JITTER_ATTEMPTS = 32


def _sample_disk_offset(rng, radius=JITTER_RADIUS):
    angle = rng.uniform(0.0, 2.0 * np.pi)
    distance = float(radius) * np.sqrt(rng.uniform())
    return distance * np.array([np.cos(angle), np.sin(angle)])


def _object_has_illegal_penetration(env, name, tolerance=0.002):
    """Reject object overlaps, while permitting contact with its support."""
    object_bodies = env._body_tree_ids([env.obj_body_id[name]])
    allowed_bodies = set()
    cfg = next((cfg for cfg in env.object_cfgs if cfg["name"] == name), {})
    region = cfg.get("placement", {}).get("region")
    if getattr(region, "kind", None) is not None:
        fixture = env.get_fixture(id=region.fixture_id)
        if fixture is not None:
            roots = []
            for body_name in fixture.body_names:
                try:
                    roots.append(env.sim.model.body_name2id(body_name))
                except Exception:
                    continue
            allowed_bodies = env._body_tree_ids(roots)
    allowed_bodies.update(env._initial_jitter_allowed_contact_bodies(name))

    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        if float(contact.dist) >= -float(tolerance):
            continue
        body1 = int(env.sim.model.geom_bodyid[contact.geom1])
        body2 = int(env.sim.model.geom_bodyid[contact.geom2])
        if body1 in object_bodies:
            other = body2
        elif body2 in object_bodies:
            other = body1
        else:
            continue
        if other not in object_bodies and other not in allowed_bodies:
            return True
    return False


def _default_candidate_is_valid(env):
    if env.count_body_collisions() != 0:
        return False
    if not env._semantic_placements_are_valid():
        return False
    return not any(_object_has_illegal_penetration(env, name) for name in env.objects)


def apply_initial_pose_jitter(
    env,
    robot_radius=JITTER_RADIUS,
    object_radius=JITTER_RADIUS,
    rng=None,
    include_robot=True,
    include_objects=True,
):
    """Randomize G1 and task objects within configurable XY radii.

    The environment supplies a small placement protocol through the
    ``_initial_jitter_*`` hooks. This deliberately has no dependency on an
    atomic-task package, so other scene environment bases can reuse it.
    """
    rng = env.rng if rng is None else rng
    robot_joint = env._get_robot_free_joint()
    if include_robot and robot_joint is None:
        return
    movable_joints = []
    if include_objects:
        movable_joints.extend(model.joints[0] for model in env.objects.values())
    if include_objects:
        movable_joints.extend(env._initial_jitter_extra_joints())
    movable_joints = list(dict.fromkeys(movable_joints))
    joints = []
    if include_robot and robot_joint is not None:
        joints.append(robot_joint)
    joints.extend(movable_joints)
    if not joints:
        return False
    baseline = {joint: env.sim.data.get_joint_qpos(joint).copy() for joint in joints}

    for _ in range(MAX_JITTER_ATTEMPTS):
        for joint, qpos in baseline.items():
            candidate = qpos.copy()
            radius = robot_radius if joint == robot_joint else object_radius
            candidate[:2] += _sample_disk_offset(rng, radius)
            env.sim.data.set_joint_qpos(joint, candidate)
        env.sim.forward()
        env._constrain_initial_jitter_candidate()
        if (
            _default_candidate_is_valid(env)
            and env._initial_jitter_candidate_is_valid()
        ):
            if include_robot and robot_joint is not None:
                env.robot_base_pos[:2] = env._robot_base_xy()
                env.robot_base_ori[2] = env._robot_base_yaw()
            return True

    for joint, qpos in baseline.items():
        env.sim.data.set_joint_qpos(joint, qpos)
    env.sim.forward()
    # The baseline is the task's normal automatic / calibrated placement.  A
    # failed local perturbation must not discard an otherwise valid episode.
    return False
