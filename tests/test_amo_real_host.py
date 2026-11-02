"""AMO on a host that has no simulator, and no operator consent yet.

Both cases here are hardware-only code paths -- they are unreachable in sim,
which is exactly why they shipped broken.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from g1_bridge.backends.amo import AmoBackend, _runtime_owns_the_robot
from g1_bridge.g1_runtime import G1_JOINT_NAMES

_ARM_AND_WRIST = [
    i
    for i, name in enumerate(G1_JOINT_NAMES)
    if "shoulder" in name or "elbow" in name or "wrist" in name
]


def _ctx(control_mode: str, target_q, *, sim=None):
    """The slice of Sim2RealContext ``_step_amo`` actually touches."""
    tracking = SimpleNamespace(state_dict={"control_mode": control_mode})
    return SimpleNamespace(
        sim=sim,
        tracking=tracking,
        step_policy=lambda qpos: (
            np.asarray(target_q, dtype=np.float32),
            np.full(29, 50.0, dtype=np.float32),
            np.full(29, 1.0, dtype=np.float32),
        ),
    )


class RuntimeOwnsTheRobotTest(unittest.TestCase):
    def test_only_policy_mode_hands_the_joints_over(self) -> None:
        self.assertTrue(_runtime_owns_the_robot(_ctx("policy", np.zeros(29))))
        self.assertFalse(_runtime_owns_the_robot(_ctx("zero", np.zeros(29))))
        self.assertFalse(_runtime_owns_the_robot(_ctx("init", np.zeros(29))))

    def test_a_host_without_control_modes_is_unchanged(self) -> None:
        """The simulator never leaves policy mode and never had a state_dict
        contract with this backend; it must keep the old behaviour."""
        self.assertTrue(_runtime_owns_the_robot(SimpleNamespace()))
        self.assertTrue(_runtime_owns_the_robot(SimpleNamespace(tracking=None)))
        self.assertTrue(
            _runtime_owns_the_robot(SimpleNamespace(tracking=SimpleNamespace()))
        )


class AmoArmOverrideGatingTest(unittest.TestCase):
    """The arms must not move while the operator has not armed the robot.

    Measured on hardware before this gate existed: the legs held to 0.0000 rad
    of error in zero mode while the arms were commanded up to 1.97 rad away
    from their measured pose in a single 20 ms tick.
    """

    def setUp(self) -> None:
        self.backend = AmoBackend()
        self.backend._build_amo_joint_mapping()
        self.backend._amo_arm_action = np.full(8, 0.5, dtype=np.float32)
        self.backend._amo_wrist_targets = np.full(6, -0.5, dtype=np.float32)
        # What zero mode produces: the measured pose, held.
        self.measured = np.linspace(-1.5, 1.5, 29).astype(np.float32)

    def test_zero_mode_passes_the_runtime_target_through_untouched(self) -> None:
        for mode in ("zero", "init"):
            with self.subTest(mode=mode):
                out, _kp, _kd = self.backend._step_amo(
                    _ctx(mode, self.measured), self.measured
                )
                np.testing.assert_array_equal(out, self.measured)

    def test_policy_mode_still_overwrites_the_fourteen_arm_joints(self) -> None:
        out, _kp, _kd = self.backend._step_amo(
            _ctx("policy", self.measured), self.measured
        )
        np.testing.assert_array_equal(
            out[self.backend._amo_arm_sim_indices], self.backend._amo_arm_action
        )
        np.testing.assert_array_equal(
            out[self.backend._amo_wrist_sim_indices], self.backend._amo_wrist_targets
        )
        untouched = [i for i in range(29) if i not in _ARM_AND_WRIST]
        np.testing.assert_array_equal(out[untouched], self.measured[untouched])


class AmoResetWithoutSimTest(unittest.TestCase):
    def test_reset_on_hardware_does_not_raise(self) -> None:
        """``_reset_amo``'s sim-less branch is what ``RealRobotRunner.arm()``
        runs first. It used to raise NameError on an unimported symbol, which
        made amo impossible to arm on a real robot at all -- and could not
        show up in sim, where the branch is never taken."""
        backend = AmoBackend()
        backend._build_amo_joint_mapping()
        backend.reset(SimpleNamespace(sim=None))
        np.testing.assert_array_equal(backend._amo_commands, np.zeros(8))
        self.assertFalse(backend._amo_arm_engaged)
        self.assertIsNone(backend._amo_yaw_anchor)


if __name__ == "__main__":
    unittest.main()
