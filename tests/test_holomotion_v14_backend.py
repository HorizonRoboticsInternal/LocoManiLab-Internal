"""Check the real driver/reference queue boundary without ONNX or hardware."""
from types import SimpleNamespace
import numpy as np
import pytest

from sim2real.vla.driver import VlaDriver, VlaDriverConfig
from g1_bridge.backends import get_backend_class
from g1_bridge.holomotion_v14 import ensure_vendor_path
from g1_bridge.holomotion_v14.runtime import HolomotionV14Runtime


def driver(client):
    return VlaDriver(client, command_space='holomotion_reference_v1',
                     config=VlaDriverConfig(inference_rate=3.125))


def test_driver_runtime_queue_replan_and_episode_reset():
    ensure_vendor_path()
    from humanoid_policy.vr_reference import VrLatestObsReference
    class Client:
        calls = 0
        def get_action(self, observation):
            rows = np.zeros((50, 87), np.float32)
            rows[:, 0] = self.calls * 100 + np.arange(50)
            rows[:, 61] = 1
            rows[:, 65:] = self.calls + .25
            self.calls += 1
            return {'reference': rows}
    class Node:
        n_fut_frames, reference_delay_frames = 10, 0
        policy_enabled, current_policy_mode = True, 'velocity'
        _vr_reference = VrLatestObsReference(n_fut_frames=10, num_actions=29, expected_dim=65)
        runtime = SimpleNamespace(state=SimpleNamespace(motion_reference_frozen=False))
        action_target, action_kps, action_kds = np.zeros(29), np.ones(29), np.ones(29)
        def feed_lowstate(self, *args):
            pass
        def feed_reference(self, row, index, stamp):
            self._vr_reference.store(row, current_time=stamp / 1e9, frame_index=index)
        def feed_pico(self, sample):
            if sample.get('B'):
                self.current_policy_mode = 'motion'
        def run(self):
            pass
    client = Client()
    loop = driver(client)
    runtime = HolomotionV14Runtime.__new__(HolomotionV14Runtime)
    runtime.node, runtime.started, runtime.frame_index, runtime.steps = Node(), True, 0, 0
    cls = get_backend_class('holomotion')
    assert cls.__name__ == 'HolomotionV14Backend'
    backend = cls()
    for step in range(32):
        a = loop.step({})
        runtime.step(np.zeros(29), np.zeros(29), [1,0,0,0], [0,0,0], step / 50,
                     a['reference_window'], a['reference_new_chunk'])
        expected = (step // 16) * 100 + step % 16
        assert runtime.node._vr_reference.current_dof_pos()[0] == expected
        left, right = backend.vla_hand_targets(a)
        np.testing.assert_array_equal(left, np.full(11, step // 16 + .25))
        np.testing.assert_array_equal(left, right)
    assert client.calls == 2
    loop.reset()
    a = loop.step({})
    assert a['reference_step'] == 0 and a['reference_new_chunk']


def test_invalid_reference_and_execution_contract_fail_before_control():
    with pytest.raises(ValueError, match='synchronous'):
        VlaDriver(None, command_space='holomotion_reference_v1')
    invalid = np.zeros((50,87))
    with pytest.raises(ValueError, match='quaternion'):
        driver(SimpleNamespace(get_action=lambda _: {'reference': invalid})).step({})
    with pytest.raises(ValueError, match='finite'):
        driver(SimpleNamespace(get_action=lambda _: {'reference': invalid[:16]})).step({})
