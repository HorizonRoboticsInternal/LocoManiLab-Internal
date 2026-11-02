"""The standing reference every backend holds while disengaged.

Needs a real env (the joint ordering only exists once a policy is loaded), but
no PICO and no display: MUJOCO_GL=egl, --pico-source mock.

Why this exists. The reference is FK of the running policy's own
``default_dof_angles``, and *which list indexes that vector* is a trap:
``BasePolicy.__init__`` sizes it from ``joint_names_simulation`` but fills it
via ``action_manager.joint_names``. Those differ for sonic and amo. Permuting
from the wrong one produced a vector that was still 29 plausible-looking joint
angles -- knee values on hip-roll joints, elbow values on ankle-roll joints --
so the robot settled into a wide splayed crouch that reads as a policy or
tuning problem, not as a reordering bug. Nothing downstream can catch that;
only an assertion here can.

The oracle is ``SONIC_DEFAULT_ANGLES_MUJOCO``: the vendored WBT constant, known
good, in MuJoCo order. sonic's checkpoint declares the same pose, so the two
must agree bit-for-bit.

Run: MUJOCO_GL=egl python tests/test_standing_reference.py
"""

import sys

import numpy as np

from g1_bridge.g1_runtime import G1_JOINT_NAMES
from g1_bridge.vendored.wbt.controllers.wbt_controller_sonic import (
    SONIC_DEFAULT_ANGLES_MUJOCO,
)

from robocasa.utils.g1_data_collection import build_teleop_env, forward_scene_kwargs
from robocasa.g1_host.bridge import load_sim2real_controller_config

ENV = "IThorKitchenOpenFridge"
FAILURES = []


def check(cond, msg):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        FAILURES.append(msg)


def idx(name):
    return G1_JOINT_NAMES.index(name)


def run(backend):
    print(f"{backend}:")
    cfg = load_sim2real_controller_config(backend, "mock")
    env = build_teleop_env(
        ENV,
        "G1FloatingBody",
        cfg,
        forward_scene_kwargs(ENV, fixed_scene=True, scene_seed=0),
        headless=True,
        control_freq=50,
    )
    try:
        env.reset()
        # .session, not the composite: the standing reference belongs to the
        # host-agnostic TeleopSession (g1_bridge/session.py), which is
        # also what the backends see as their Sim2RealContext.
        comp = env.robots[0].composite_controller.session
        sj = comp.standing_joints()
        check(sj is not None, "standing_joints() resolves")
        if sj is None:
            return
        sj = np.asarray(sj, dtype=float)
        check(sj.shape == (len(G1_JOINT_NAMES),), f"29 entries, got {sj.shape}")

        # A standing pose is left/right symmetric except for the roll joints,
        # which mirror. Scrambling breaks this immediately.
        for a, b, sign in [
            ("left_hip_pitch_joint", "right_hip_pitch_joint", +1),
            ("left_knee_joint", "right_knee_joint", +1),
            ("left_ankle_pitch_joint", "right_ankle_pitch_joint", +1),
            ("left_shoulder_pitch_joint", "right_shoulder_pitch_joint", +1),
            ("left_shoulder_roll_joint", "right_shoulder_roll_joint", -1),
            ("left_elbow_joint", "right_elbow_joint", +1),
        ]:
            check(
                abs(sj[idx(a)] - sign * sj[idx(b)]) < 1e-6,
                f"{a} mirrors {b} ({sj[idx(a)]:+.3f} vs {sj[idx(b)]:+.3f})",
            )

        # Standing means bent knees and legs under the body, never splayed.
        check(sj[idx("left_knee_joint")] > 0.1, "knees are bent (a real stance)")
        for n in (
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
        ):
            check(abs(sj[idx(n)]) < 0.05, f"{n} ~0 (legs under the body)")

        if backend == "sonic":
            d = float(np.abs(sj - SONIC_DEFAULT_ANGLES_MUJOCO).max())
            check(
                d < 1e-6,
                f"matches SONIC_DEFAULT_ANGLES_MUJOCO bit-for-bit (max diff {d:.2e})",
            )
    finally:
        env.close()


if __name__ == "__main__":
    for b in ("sonic", "heft", "scalebfm", "amo"):
        run(b)
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print(
        "PASS: every backend's standing reference is a sane, correctly-ordered stance"
    )
