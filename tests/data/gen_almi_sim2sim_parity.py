#!/usr/bin/env python3
"""Regenerate ``almi_sim2sim_parity.npz`` -- the ALMI cross-implementation fixture.

The ALMI backend reimplements an observation layout that already has a working
reference implementation: mjlab-loco's own plain-MuJoCo deployment
(``deploy_mujoco/sim2sim.py:ObservationBuilder`` + ``policies.py``). The two
cannot be compared at runtime because they need different environments -- mjlab
pulls in warp/mujoco-warp and pins its own torch -- so instead this records what
the reference produces for a fixed pseudo-random closed-loop rollout, and
``tests/test_almi_sim2sim_parity.py`` replays the same rollout through this
repo's implementation.

Closed loop, not a set of independent frames: ``prev_action`` is the previous
tick's raw network output on both sides, so an off-by-one there (feeding the
current action, or resetting it per frame) has to show up.

The action fed back is clipped to +/-``FEEDBACK_CLIP`` first, and that clip is
load-bearing. The rollout drives a body that never responds -- joint states are
resampled at random every tick, so nothing the policy does moves anything --
and a branch that leans on ``prev_action`` will happily run away inside that
loop: the half_body checkpoint diverged at tick 4 and reached 1e30 by the end,
which makes every downstream tick a comparison of two numerical garbages
agreeing to a few ULPs. The clip breaks that feedback path without touching the
off-by-one property. It is set above any action a healthy rollout produces (the
whole_body reference peaks at 7.5 and this repo's ALMI notes put branch
saturation at ~10), so on a well-behaved checkpoint it never binds and the
fixture is bit-identical with or without it.

This script does NOT run in this repo's environment. Run it in mjlab-loco's::

    cd /path/to/mjlab-loco
    .venv/bin/python \\
        /path/to/g1_backend/tests/data/gen_almi_sim2sim_parity.py \\
        --checkpoint logs/rsl_rl/g1_velocity_dual/H100/flat/model_10000.pt \\
        --out /path/to/g1_backend/tests/data/almi_sim2sim_parity.npz

One fixture per observation layout, named for it -- the widths are read off the
checkpoint, so the only thing that has to be right by hand is the --out name.
The `locomotion_sim2real` checkout carries the same ObservationBuilder under
`locomotion_sim2real.sim2sim`; either works, they are the same code.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TICKS = 64
SEED = 20260810
#: Bound on the action fed back as the next tick's ``prev_action``. See the
#: module docstring -- this is a property of the synthetic rollout, not of the
#: policy (mjlab's RslRlBaseRunnerCfg.clip_actions is None for this task).
FEEDBACK_CLIP = 10.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    from deploy_mujoco.policies import CheckpointDualPolicy
    from deploy_mujoco.reference_sources import ReferenceFrame
    from deploy_mujoco.sim2sim import ALL_JOINT_NAMES, ObservationBuilder, RuntimeState
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.entity import Entity

    import mujoco

    model = Entity(get_g1_robot_cfg()).spec.compile()
    qpos_adrs = np.array(
        [model.joint(name).qposadr[0] for name in ALL_JOINT_NAMES], dtype=np.int32
    )
    default_joint_pos = model.key_qpos[0, qpos_adrs].astype(np.float32)

    rng = np.random.default_rng(args.seed)
    n = len(ALL_JOINT_NAMES)
    ticks = args.ticks

    # Plausible magnitudes, not standard normal: the point is to exercise the
    # layout, and a rollout the policy would never see makes a real disagreement
    # harder to read next to float32 noise.
    joint_pos = default_joint_pos[None, :] + rng.normal(0, 0.15, (ticks, n)).astype(np.float32)
    joint_vel = rng.normal(0, 0.8, (ticks, n)).astype(np.float32)
    base_ang_vel = rng.normal(0, 0.3, (ticks, 3)).astype(np.float32)
    root_quat = rng.normal(0, 1.0, (ticks, 4)).astype(np.float32)
    root_quat /= np.linalg.norm(root_quat, axis=1, keepdims=True)

    twist = np.stack(
        [
            rng.uniform(-0.8, 0.8, ticks),
            rng.uniform(-0.5, 0.5, ticks),
            rng.uniform(-1.8, 1.8, ticks),
            rng.uniform(0.35, 0.79, ticks),
        ],
        axis=1,
    ).astype(np.float32)
    ee_pos = rng.uniform(-0.3, 0.45, (ticks, 6)).astype(np.float32)
    ee_quat = rng.normal(0, 1.0, (ticks, 8)).astype(np.float32).reshape(ticks, 2, 4)
    ee_quat /= np.linalg.norm(ee_quat, axis=2, keepdims=True)
    ee_quat = ee_quat.reshape(ticks, 8)
    q_ref = rng.uniform(-0.6, 0.6, (ticks, 17)).astype(np.float32)

    policy = CheckpointDualPolicy(args.checkpoint, device="cpu")
    builder = ObservationBuilder(default_joint_pos)

    # Read the widths off the checkpoint rather than hardcoding them: ALMI has
    # shipped more than one observation layout, and the reference implementation
    # dispatches on exactly these numbers. Hardcoding them here would silently
    # record one layout under the name of another.
    lower_obs_dim = int(policy.lower_obs_dim)
    upper_obs_dim = int(policy.upper_obs_dim)
    history_steps, history_width = policy.history_shape
    print(
        f"[gen] lower_obs={lower_obs_dim} upper_obs={upper_obs_dim} "
        f"history={history_steps}x{history_width}"
    )

    lower_obs_out = np.zeros((ticks, lower_obs_dim), dtype=np.float32)
    upper_obs_out = np.zeros((ticks, upper_obs_dim), dtype=np.float32)
    history_out = np.zeros((ticks, history_steps, history_width), dtype=np.float32)
    action_out = np.zeros((ticks, n), dtype=np.float32)
    gravity_out = np.zeros((ticks, 3), dtype=np.float32)
    waist_rpy_out = np.zeros((ticks, 3), dtype=np.float32)

    for t in range(ticks):
        # Same gravity convention the runtime uses (R^T @ [0,0,-1] for the
        # pelvis rotation), derived here from the quaternion the fixture stores
        # so the other side can reach it by its own route.
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, root_quat[t].astype(np.float64))
        projected_gravity = (
            rotation.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        ).astype(np.float32)
        gravity_out[t] = projected_gravity

        state = RuntimeState(
            joint_pos=joint_pos[t],
            joint_vel=joint_vel[t],
            base_ang_vel=base_ang_vel[t],
            projected_gravity=projected_gravity,
        )
        reference = ReferenceFrame(
            twist=twist[t],
            waist_rpy=np.zeros(3, dtype=np.float32),
            ee_pos=ee_pos[t],
            ee_quat=ee_quat[t],
            upper_joint_pos=q_ref[t],
            upper_joint_ref_valid=True,
        )
        lower_obs, upper_obs, history = builder.build(
            state, reference,
            lower_obs_dim=lower_obs_dim, upper_obs_dim=upper_obs_dim,
        )
        lower_action, upper_action = policy.act(
            lower_obs, upper_obs, lower_history=history
        )
        builder.update_action(
            np.clip(lower_action, -FEEDBACK_CLIP, FEEDBACK_CLIP),
            np.clip(upper_action, -FEEDBACK_CLIP, FEEDBACK_CLIP),
        )

        lower_obs_out[t] = lower_obs
        upper_obs_out[t] = upper_obs
        history_out[t] = history
        action_out[t] = np.concatenate([lower_action, upper_action])
        waist_rpy_out[t] = reference.validated().waist_rpy

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        joint_names=np.array(ALL_JOINT_NAMES),
        default_joint_pos=default_joint_pos,
        # Recorded, not agreed by convention: the replay has to feed back the
        # same bounded action this rollout did, and a constant duplicated in
        # two repos-worth of files is a constant that eventually differs.
        feedback_clip=np.float32(FEEDBACK_CLIP),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        base_ang_vel=base_ang_vel,
        root_quat=root_quat,
        projected_gravity=gravity_out,
        twist=twist,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        q_ref=q_ref,
        waist_rpy=waist_rpy_out,
        lower_obs=lower_obs_out,
        upper_obs=upper_obs_out,
        lower_history=history_out,
        action=action_out,
    )
    print(f"wrote {args.out} ({ticks} ticks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
