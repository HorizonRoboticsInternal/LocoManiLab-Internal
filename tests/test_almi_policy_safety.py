"""Safety behavior specific to ALMI's dual-branch policy adapter."""

from types import SimpleNamespace

import numpy as np
import pytest

from sim2real.rl_policy import tracking_factory
from sim2real.rl_policy.tracking_factory import DualBranchPolicyMixin


class _FakeModule:
    def __init__(self, action):
        self._action = np.asarray(action, dtype=np.float32)

    def __call__(self, _feed):
        return {"action": self._action.copy()}


class _Harness(DualBranchPolicyMixin):
    pass


def _make_policy(monkeypatch, *, safety=True, config_overrides=None):
    lower = np.array([10.0, -10.0] + [1.0] * 10, dtype=np.float32)
    upper = np.full(17, 4.0, dtype=np.float32)
    modules = iter((_FakeModule(lower), _FakeModule(upper)))
    monkeypatch.setattr(
        tracking_factory,
        "build_inference_module",
        lambda *_args, **_kwargs: next(modules),
    )

    harness = _Harness()
    harness.policy_config = {"upper_model_path": "upper.onnx"}
    if safety:
        harness.policy_config.update(
            action_clip=5.0,
            soft_joint_pos_limit_factor=0.9,
        )
    if config_overrides:
        harness.policy_config.update(config_overrides)
    harness.args = SimpleNamespace(policy_config="/tmp/almi/policy.yaml")
    harness.inference_backend = "onnx-cpu"
    harness.num_actions = 29
    harness.controlled_joint_indices = list(range(29))
    harness.default_dof_angles = np.zeros(29, dtype=np.float64)
    harness.action_scale = np.ones(29, dtype=np.float64)
    harness.joint_pos_lower_limit = np.full(29, -2.0, dtype=np.float64)
    harness.joint_pos_upper_limit = np.full(29, 2.0, dtype=np.float64)
    harness.setup_policy("/tmp/almi/lower.onnx")
    return harness


def _inputs():
    return {
        "lower_obs": np.zeros(46, dtype=np.float32),
        "lower_history": np.zeros(230, dtype=np.float32),
        "upper_obs": np.zeros(105, dtype=np.float32),
    }


def test_clips_action_feedback_and_soft_limits_q_target(monkeypatch):
    harness = _make_policy(monkeypatch)

    action, q_target, _ = harness.policy(_inputs())

    # The returned action becomes state_dict["action"] (the next prev_action).
    np.testing.assert_array_equal(action[:2], np.array([5.0, -5.0]))
    assert np.max(np.abs(action)) <= 5.0
    # Hardware range [-2, 2] contracted to 90% is [-1.8, 1.8].
    assert np.max(q_target) == pytest.approx(1.8)
    assert np.min(q_target) == pytest.approx(-1.8)


def test_safety_is_opt_in_for_older_almi_manifests(monkeypatch):
    harness = _make_policy(monkeypatch, safety=False)

    action, q_target, _ = harness.policy(_inputs())

    assert action[0] == pytest.approx(10.0)
    assert q_target[0] == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("key", "value"),
    (("action_clip", 0.0), ("soft_joint_pos_limit_factor", 1.1)),
)
def test_rejects_invalid_safety_config(monkeypatch, key, value):
    with pytest.raises(ValueError):
        _make_policy(
            monkeypatch,
            safety=False,
            config_overrides={key: value},
        )
