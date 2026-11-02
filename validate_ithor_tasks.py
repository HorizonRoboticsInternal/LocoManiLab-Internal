"""Catalog-driven static and reset validation for iTHOR room tasks."""

import argparse
import os
import pathlib
import traceback

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

from robocasa.environments.ithor_atomic.pick_place import (
    SurfaceObjectInReceptacle,
)
from robocasa.environments.ithor_task_catalog import (
    BATHROOM_PACKAGE,
    BEDROOM_PACKAGE,
    discover_task_catalog,
    task_counts,
    validate_task_catalog,
)
from robocasa.models.scenes.thor_semantics import SemanticRegion


DEFAULT_PACKAGES = (
    BEDROOM_PACKAGE,
    BATHROOM_PACKAGE,
)
SUPPORT_SETTLE_STEPS = 30
MAX_OBJECT_SETTLE_XY = 0.03
MAX_OBJECT_SETTLE_Z = 0.03
SUPPORT_PENETRATION_TOLERANCE = 0.003
MAX_EDGE_ROW_DEPTH_DELTA = 0.02
MAX_CROSSWISE_VIEW_DOT = 0.30


def _load_reset_controller():
    """Load the G1 controller in policy-free pose-hold mode for reset audits."""
    config_path = (
        pathlib.Path(robosuite.__file__).parent
        / "controllers/config/robots/default_g1_teleop.json"
    )
    controller = load_composite_controller_config(controller=str(config_path))
    controller["composite_controller_specific_configs"][
        "wbt_controller_type"
    ] = "default_pos"
    return controller


def _csv(value):
    return {token.strip() for token in value.split(",") if token.strip()}


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package",
        action="append",
        dest="packages",
        help="room package to discover; repeat for multiple rooms",
    )
    parser.add_argument(
        "--task-filter", default=os.environ.get("TASK_FILTER", "")
    )
    parser.add_argument(
        "--scene-filter", default=os.environ.get("SCENE_FILTER", "")
    )
    parser.add_argument(
        "--resets",
        type=int,
        default=int(os.environ.get("RESET_COUNT", "0")),
        help=(
            "resets per selected task-scene pair; task-design validation should "
            "normally set --task-filter to the new or modified tasks"
        ),
    )
    parser.add_argument("--atomic-only", action="store_true")
    return parser.parse_args()


def _selected(entries, task_filter, scene_filter, atomic_only):
    task_tokens = _csv(task_filter)
    scene_tokens = _csv(scene_filter)
    for entry in entries:
        if atomic_only and entry.family == "composite":
            continue
        if task_tokens and not any(token in entry.name for token in task_tokens):
            continue
        for scene_id in entry.scene_ids:
            if scene_tokens and str(scene_id) not in scene_tokens:
                continue
            yield entry, scene_id


def _semantic_support_placements(env):
    seen = set()
    for cfg in getattr(env, "object_cfgs", ()):
        name = cfg.get("name")
        region = cfg.get("placement", {}).get("region")
        if (
            name in getattr(env, "objects", {})
            and isinstance(region, SemanticRegion)
            and region.kind == "support"
        ):
            seen.add((name, region.stable_id))
            yield name, region
    for name, region in getattr(env, "_task_support_contracts", {}).items():
        if (
            name in getattr(env, "objects", {})
            and isinstance(region, SemanticRegion)
            and region.kind == "support"
            and (name, region.stable_id) not in seen
        ):
            seen.add((name, region.stable_id))
            yield name, region
    support_regions = {
        region.stable_id: region
        for region in getattr(env, "region_refs", {}).values()
        if isinstance(region, SemanticRegion) and region.kind == "support"
    }
    for region in support_regions.values():
        for name in getattr(env, "objects", {}):
            if (name, region.stable_id) in seen:
                continue
            if env.object_on_region(
                name, region, z_tol=0.08, require_contact=True
            ):
                seen.add((name, region.stable_id))
                yield name, region


def _assert_raw_predicates_false(env, phase):
    for predicate_name in ("_raw_success", "_raw_place_success"):
        predicate = getattr(env, predicate_name, None)
        if callable(predicate) and predicate():
            raise AssertionError(
                f"{phase}: {predicate_name} is already satisfied"
            )


def _assert_surface_pnp_layout(env, phase):
    if not isinstance(env, SurfaceObjectInReceptacle):
        return
    region = env.work_surface_region
    source_name = env.SOURCE_NAME
    receptacle_name = env.RECEPTACLE_NAME
    operator_xy = (
        env._operator_reference_xy()
        if callable(getattr(env, "_operator_reference_xy", None))
        else env._robot_base_xy()
    )
    center, _ = region.world_pose(env.sim)
    direction = np.asarray(
        env.support_region_long_edge_approach_directions(
            region, toward_xy=operator_xy, sim=env.sim
        )[0],
        dtype=float,
    )
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    positions = {
        name: np.asarray(
            env.sim.data.get_body_xpos(env.objects[name].root_body), dtype=float
        )
        for name in (source_name, receptacle_name)
    }
    depths = {
        name: float(np.dot(position[:2] - center[:2], direction))
        for name, position in positions.items()
    }
    depth_delta = abs(depths[source_name] - depths[receptacle_name])
    if depth_delta > MAX_EDGE_ROW_DEPTH_DELTA:
        raise AssertionError(
            f"{phase}: source/receptacle edge-row depth delta "
            f"{depth_delta:.4f} m"
        )
    if getattr(env, "SOURCE_ON_OPERATOR_RIGHT", False):
        forward = np.asarray(center[:2], dtype=float) - np.asarray(
            operator_xy, dtype=float
        )
        forward /= max(float(np.linalg.norm(forward)), 1e-8)
        operator_right = np.array([forward[1], -forward[0]], dtype=float)
        right_separation = float(
            np.dot(
                positions[source_name][:2] - positions[receptacle_name][:2],
                operator_right,
            )
        )
        minimum = float(env.MIN_SOURCE_RIGHT_SEPARATION)
        if right_separation < minimum:
            raise AssertionError(
                f"{phase}: source is only {right_separation:.4f} m to the "
                f"operator's right (minimum {minimum:.4f} m)"
            )
    for name in (source_name, receptacle_name):
        points = env.object_bbox_points(name)
        if not all(region.contains_xy(point, env.sim) for point in points):
            raise AssertionError(
                f"{phase}: {name} bounding box leaves named work surface"
            )

    if getattr(env, "ENFORCE_EDGE_PARALLEL_LAYOUT", False):
        for name in (source_name, receptacle_name):
            half_xy = np.asarray(env._object_half_xy(name), dtype=float)
            if not (
                half_xy.shape == (2,)
                and float(np.min(half_xy)) > 1e-8
                and float(np.max(half_xy) / np.min(half_xy)) >= 1.15
            ):
                continue
            matrix = np.asarray(
                env.sim.data.get_body_xmat(env.objects[name].root_body),
                dtype=float,
            ).reshape(3, 3)
            long_axis = matrix[:2, int(np.argmax(half_xy))]
            long_axis /= max(float(np.linalg.norm(long_axis)), 1e-8)
            alignment = abs(float(np.dot(direction, long_axis)))
            if alignment > MAX_CROSSWISE_VIEW_DOT:
                raise AssertionError(
                    f"{phase}: elongated {name} is not parallel to the table "
                    f"edge (normal dot={alignment:.3f})"
                )


def _support_contact_distances(env, name, region):
    fixture = env.get_fixture(id=region.fixture_id)
    object_bodies = env._body_tree_ids([env.obj_body_id[name]])
    fixture_bodies = env._fixture_body_ids(fixture, include_siblings=True)
    distances = []
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        body1 = int(env.sim.model.geom_bodyid[contact.geom1])
        body2 = int(env.sim.model.geom_bodyid[contact.geom2])
        if (body1 in object_bodies and body2 in fixture_bodies) or (
            body2 in object_bodies and body1 in fixture_bodies
        ):
            distances.append(float(contact.dist))
    return distances


def _resolve_body_roots(env, roots, contract_name, role):
    model = env.sim.model
    root_ids = []
    for root in roots:
        try:
            body_id = int(root) if isinstance(root, (int, np.integer)) else int(
                model.body_name2id(root)
            )
        except Exception as exc:
            raise AssertionError(
                f"body contract {contract_name}: unknown {role} root {root!r}"
            ) from exc
        if body_id < 0 or body_id >= int(model.nbody):
            raise AssertionError(
                f"body contract {contract_name}: invalid {role} body id {body_id}"
            )
        root_ids.append(body_id)
    body_ids = env._body_tree_ids([body_id for body_id in root_ids if body_id])
    if 0 in root_ids:
        body_ids.add(0)
    return tuple(root_ids), body_ids


def _task_body_contact_contracts(env):
    contracts = getattr(env, "_task_body_contact_contracts", {})
    if not isinstance(contracts, dict):
        raise AssertionError("_task_body_contact_contracts must be a dict")
    for name, contract in contracts.items():
        if not isinstance(name, str) or not isinstance(contract, dict):
            raise AssertionError("invalid task body contact contract")
        subject_roots = contract.get("subject_body_roots")
        support_roots = contract.get("support_body_roots")
        if not subject_roots or not support_roots:
            raise AssertionError(
                f"body contract {name}: subject/support roots are required"
            )
        yield name, tuple(subject_roots), tuple(support_roots)


def _assert_body_contact_state(
    env, contract_name, subject_roots, support_roots, phase
):
    subject_root_ids, subject_bodies = _resolve_body_roots(
        env, subject_roots, contract_name, "subject"
    )
    support_root_ids, support_bodies = _resolve_body_roots(
        env, support_roots, contract_name, "support"
    )
    if subject_bodies & support_bodies:
        raise AssertionError(
            f"body contract {contract_name}: subject and support overlap"
        )
    distances = []
    for index in range(env.sim.data.ncon):
        contact = env.sim.data.contact[index]
        body1 = int(env.sim.model.geom_bodyid[contact.geom1])
        body2 = int(env.sim.model.geom_bodyid[contact.geom2])
        if (body1 in subject_bodies and body2 in support_bodies) or (
            body2 in subject_bodies and body1 in support_bodies
        ):
            distances.append(float(contact.dist))
    if not distances:
        raise AssertionError(
            f"{phase}: body contract {contract_name} lacks support contact"
        )
    if min(distances) < -SUPPORT_PENETRATION_TOLERANCE:
        raise AssertionError(
            f"{phase}: body contract {contract_name} penetrates support by "
            f"{-min(distances):.4f} m"
        )
    illegal = env.illegal_contact_pairs_for_bodies(
        subject_root_ids,
        support_root_ids,
        tolerance=SUPPORT_PENETRATION_TOLERANCE,
    )
    if illegal:
        raise AssertionError(
            f"{phase}: body contract {contract_name} penetrates unrelated geometry"
        )
    return subject_bodies


def _assert_support_state(env, name, region, phase):
    if not env.object_on_region(name, region, z_tol=0.08, require_contact=True):
        raise AssertionError(
            f"{phase}: {name} lacks contact with named support {region.stable_id}"
        )
    distances = _support_contact_distances(env, name, region)
    if not distances:
        raise AssertionError(f"{phase}: {name} has no physical support contact")
    if min(distances) < -SUPPORT_PENETRATION_TOLERANCE:
        raise AssertionError(
            f"{phase}: {name} penetrates support by {-min(distances):.4f} m"
        )
    if env._object_has_illegal_penetration(
        name, region, tolerance=SUPPORT_PENETRATION_TOLERANCE
    ):
        raise AssertionError(f"{phase}: {name} penetrates unrelated geometry")


def _capture_robot_joint_state(env):
    """Capture qpos/qvel slices for every joint in the robot body tree."""

    model = env.sim.model
    root_id = model.body_name2id(env.robots[0].robot_model.root_body)
    robot_bodies = env._body_tree_ids([root_id])
    joint_ids = []
    for body_id in sorted(robot_bodies):
        start = int(model.body_jntadr[body_id])
        joint_ids.extend(
            range(start, start + int(model.body_jntnum[body_id]))
        )
    state = []
    for joint_id in joint_ids:
        qpos_start = int(model.jnt_qposadr[joint_id])
        qpos_end = (
            int(model.jnt_qposadr[joint_id + 1])
            if joint_id + 1 < int(model.njnt)
            else int(model.nq)
        )
        qvel_start = int(model.jnt_dofadr[joint_id])
        qvel_end = (
            int(model.jnt_dofadr[joint_id + 1])
            if joint_id + 1 < int(model.njnt)
            else int(model.nv)
        )
        state.append(
            (
                slice(qpos_start, qpos_end),
                env.sim.data.qpos[qpos_start:qpos_end].copy(),
                slice(qvel_start, qvel_end),
                env.sim.data.qvel[qvel_start:qvel_end].copy(),
            )
        )
    return state


def _settle_scene_with_frozen_robot(env, steps):
    """Advance scene physics while pinning the uncontrolled robot state."""

    robot_state = _capture_robot_joint_state(env)
    for _ in range(steps):
        env.sim.step()
        for qpos_slice, qpos, qvel_slice, qvel in robot_state:
            env.sim.data.qpos[qpos_slice] = qpos
            env.sim.data.qvel[qvel_slice] = qvel
        env.sim.forward()


def _validate_support_stability(env, reset_index):
    placements = list(_semantic_support_placements(env))
    body_contracts = list(_task_body_contact_contracts(env))
    for name, region in placements:
        _assert_support_state(env, name, region, f"reset={reset_index}")
    initial_body_positions = {}
    for contract_name, subject_roots, support_roots in body_contracts:
        subject_bodies = _assert_body_contact_state(
            env,
            contract_name,
            subject_roots,
            support_roots,
            f"reset={reset_index}",
        )
        for body_id in subject_bodies:
            initial_body_positions[(contract_name, body_id)] = np.asarray(
                env.sim.data.body_xpos[body_id], dtype=float
            ).copy()
    initial = {
        name: np.asarray(
            env.sim.data.get_body_xpos(env.objects[name].root_body), dtype=float
        ).copy()
        for name in env.objects
    }
    _assert_surface_pnp_layout(env, f"reset={reset_index}")
    _settle_scene_with_frozen_robot(env, SUPPORT_SETTLE_STEPS)
    for name, region in placements:
        _assert_support_state(
            env, name, region, f"reset={reset_index} after settle"
        )
    for contract_name, subject_roots, support_roots in body_contracts:
        _assert_body_contact_state(
            env,
            contract_name,
            subject_roots,
            support_roots,
            f"reset={reset_index} after settle",
        )
    for (contract_name, body_id), initial_position in initial_body_positions.items():
        current = np.asarray(env.sim.data.body_xpos[body_id], dtype=float)
        xy_drift = float(np.linalg.norm(current[:2] - initial_position[:2]))
        z_drift = abs(float(current[2] - initial_position[2]))
        if xy_drift > MAX_OBJECT_SETTLE_XY or z_drift > MAX_OBJECT_SETTLE_Z:
            raise AssertionError(
                f"reset={reset_index}: body contract {contract_name} body "
                f"{body_id} drift xy={xy_drift:.4f} z={z_drift:.4f}"
            )
    for name, initial_position in initial.items():
        current = np.asarray(
            env.sim.data.get_body_xpos(env.objects[name].root_body), dtype=float
        )
        xy_drift = float(np.linalg.norm(current[:2] - initial_position[:2]))
        z_drift = abs(float(current[2] - initial_position[2]))
        if xy_drift > MAX_OBJECT_SETTLE_XY or z_drift > MAX_OBJECT_SETTLE_Z:
            raise AssertionError(
                f"reset={reset_index}: {name} object drift "
                f"xy={xy_drift:.4f} z={z_drift:.4f}"
            )
    collisions = env.count_body_collisions()
    if collisions:
        raise AssertionError(
            f"reset={reset_index} after settle: {collisions} robot collisions"
        )
    if not env.robot_footprint_is_on_floor(
        env._robot_base_xy(), radius=0.25
    ):
        raise AssertionError(
            f"reset={reset_index} after settle: robot footprint leaves floor"
        )
    if env._check_success():
        raise AssertionError(
            f"reset={reset_index} after settle: task became successful"
        )
    _assert_raw_predicates_false(env, f"reset={reset_index} after settle")
    _assert_surface_pnp_layout(env, f"reset={reset_index} after settle")


def _validate_resets(entries, resets, task_filter, scene_filter, atomic_only):
    if resets < 0:
        raise ValueError("--resets must be nonnegative")
    if resets == 0:
        return []

    controller = _load_reset_controller()
    failures = []
    for entry, scene_id in _selected(
        entries, task_filter, scene_filter, atomic_only
    ):
        label = f"{entry.name}:{scene_id}"
        env = None
        try:
            env = robosuite.make(
                env_name=entry.name,
                robots="G1",
                controller_configs=controller,
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                ignore_done=True,
                hard_reset=False,
                control_freq=20,
                renderer="mujoco",
                scene_ids=[scene_id],
            )
            for reset_index in range(resets):
                env.reset()
                collisions = env.count_body_collisions()
                if collisions:
                    raise AssertionError(
                        f"reset={reset_index}: {collisions} robot collisions"
                    )
                if env._check_success():
                    raise AssertionError(
                        f"reset={reset_index}: task starts successful"
                    )
                _assert_raw_predicates_false(env, f"reset={reset_index}")
                if not env.robot_footprint_is_on_floor(
                    env._robot_base_xy(), radius=0.25
                ):
                    raise AssertionError(
                        f"reset={reset_index}: robot footprint leaves floor"
                    )
                _validate_support_stability(env, reset_index)
            print("OK", label, flush=True)
        except Exception as exc:
            failures.append(f"{label}: {exc!r}")
            print("FAIL", label, repr(exc), flush=True)
            traceback.print_exc()
        finally:
            if env is not None:
                env.close()
    return failures


def main():
    args = _parse_args()
    entries = discover_task_catalog(args.packages or DEFAULT_PACKAGES)
    print("TASK_COUNTS", task_counts(entries), flush=True)
    failures = validate_task_catalog(entries)
    failures.extend(
        _validate_resets(
            entries,
            args.resets,
            args.task_filter,
            args.scene_filter,
            args.atomic_only,
        )
    )
    for failure in failures:
        print("FAILURE", failure, flush=True)
    print(
        "SUMMARY",
        f"tasks={len(entries)}",
        f"failures={len(failures)}",
        flush=True,
    )
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
