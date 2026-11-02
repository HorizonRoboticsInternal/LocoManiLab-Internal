from __future__ import annotations

import numpy as np
import pytest

from sim2real.utils.dex3 import DEX3_NUM_MOTORS, compute_pd_torques


def test_pd_torques_zero_when_at_target_and_still() -> None:
    q = np.zeros(DEX3_NUM_MOTORS, dtype=np.float32)
    torques = compute_pd_torques(q, q, np.zeros(DEX3_NUM_MOTORS, dtype=np.float32))
    np.testing.assert_allclose(torques, np.zeros(DEX3_NUM_MOTORS))


def test_pd_torques_uses_thumb_gains_only_for_first_motor() -> None:
    q_target = np.full(DEX3_NUM_MOTORS, 0.1, dtype=np.float32)
    q_measured = np.zeros(DEX3_NUM_MOTORS, dtype=np.float32)
    qvel_measured = np.zeros(DEX3_NUM_MOTORS, dtype=np.float32)

    torques = compute_pd_torques(
        q_target, q_measured, qvel_measured, kp=1.0, kd=0.2, thumb_kp=2.0, thumb_kd=0.5
    )

    assert torques[0] == pytest.approx(2.0 * 0.1)
    for i in range(1, DEX3_NUM_MOTORS):
        assert torques[i] == pytest.approx(1.0 * 0.1)


def test_pd_torques_damping_term_opposes_velocity() -> None:
    q = np.zeros(DEX3_NUM_MOTORS, dtype=np.float32)
    qvel = np.full(DEX3_NUM_MOTORS, 1.0, dtype=np.float32)

    torques = compute_pd_torques(q, q, qvel, kp=1.0, kd=0.3, thumb_kp=1.0, thumb_kd=0.3)

    np.testing.assert_allclose(torques, np.full(DEX3_NUM_MOTORS, -0.3, dtype=np.float32))


def test_pd_torques_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="expected shape"):
        compute_pd_torques(np.zeros(3), np.zeros(DEX3_NUM_MOTORS), np.zeros(DEX3_NUM_MOTORS))
