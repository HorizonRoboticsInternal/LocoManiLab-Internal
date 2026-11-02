"""heading_yaw_offset must be bit-identical to the inline formula it replaced.

The three call sites in observations/sonic.py (and scalebfm's alignment reset)
used to spell the formula out; a refactor that changed the result by even one
ULP would shift every yaw-aligned reference the policy sees. This pins the
extraction to the original expression on randomized inputs.
"""

from __future__ import annotations

import unittest

import numpy as np

from sim2real.utils.math import (
    heading_yaw_offset,
    projected_yaw_quat,
    quat_conjugate,
    quat_mul,
)


def _random_unit_quats(rng, n):
    q = rng.standard_normal((n, 4)).astype(np.float32)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


class HeadingYawOffsetTest(unittest.TestCase):
    def test_matches_inline_formula_bitwise(self):
        rng = np.random.default_rng(0)
        robot = _random_unit_quats(rng, 256)
        ref = _random_unit_quats(rng, 256)
        inline = quat_mul(
            projected_yaw_quat(robot), quat_conjugate(projected_yaw_quat(ref))
        )
        np.testing.assert_array_equal(heading_yaw_offset(robot, ref), inline)

    def test_single_quat_shapes(self):
        rng = np.random.default_rng(1)
        robot = _random_unit_quats(rng, 1)
        ref = _random_unit_quats(rng, 1)
        out = heading_yaw_offset(robot, ref)
        self.assertEqual(out.shape, (1, 4))
        inline = quat_mul(
            projected_yaw_quat(robot), quat_conjugate(projected_yaw_quat(ref))
        )
        np.testing.assert_array_equal(out, inline)


if __name__ == "__main__":
    unittest.main()
