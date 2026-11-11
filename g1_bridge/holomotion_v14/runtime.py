"""Host-clock adapter for the frozen HoloMotion v1.4 controller.

The controller's graphs and observation implementation remain unchanged. This
adapter replaces ROS transport, supplies named low-state inputs and consumes
explicit 11-frame reference windows. It does not import SIMPLE.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import hashlib
import os
import sys
import time

import numpy as np
import yaml

from . import ROBOT_CONFIG, VENDOR_DIR, ensure_vendor_path


@contextmanager
def controller_clock(now):
    """Limit the injected clock to this controller call, restoring all modules."""
    class Clock:
        def time(self):
            return float(now)
        monotonic = time
        def time_ns(self):
            return int(now * 1e9)
        monotonic_ns = time_ns
        def __getattr__(self, name):
            return getattr(time, name)
    changed = []
    for name, module in list(sys.modules.items()):
        if name.startswith(('humanoid_policy.', 'holomotion_policy_core.')) and hasattr(module, 'time'):
            changed.append((module, module.time))
            module.time = Clock()
    try:
        yield
    finally:
        for module, old in changed:
            module.time = old


class HolomotionV14Runtime:
    def __init__(self, manifest, inference_backend='onnx-cpu'):
        path = Path(manifest).resolve()
        cfg = yaml.safe_load(path.read_text())
        if cfg.get('controller') != 'holomotion_v14':
            raise ValueError('Expected controller: holomotion_v14')
        root = Path(cfg['models_dir']).expanduser()
        root = root if root.is_absolute() else path.parent / root
        if not root.is_dir():
            raise FileNotFoundError(f'HoloMotion models_dir does not exist: {root}')
        if inference_backend not in ('onnx-cpu', 'onnx-gpu'):
            raise ValueError('HoloMotion v14 currently supports onnx-cpu or onnx-gpu')
        expected = cfg.get('sha256') or {}
        for folder in ('velocity_tracking_model', 'motion_tracking_model'):
            config = root / folder / 'config.yaml'
            models = list((root / folder / 'exported').glob('*.onnx'))
            if not config.is_file() or len(models) != 1:
                raise ValueError(f'{folder} requires config.yaml and exactly one exported ONNX')
            for label, resource in ((folder, models[0]), (folder + '_config', config)):
                if label in expected:
                    digest = hashlib.sha256()
                    with resource.open('rb') as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b''):
                            digest.update(block)
                    if digest.hexdigest() != expected[label]:
                        raise ValueError(f'HoloMotion bundle hash mismatch: {label}')
        ensure_vendor_path()
        from .sim_node import build_sim_policy_node
        from . import ros_stubs
        ros_stubs.install(share_dir=str(VENDOR_DIR))
        from humanoid_policy import policy_node_29dof as policy
        original_loader = policy.load_dual_policy_bundle
        providers = ['CPUExecutionProvider'] if inference_backend == 'onnx-cpu' else ['CUDAExecutionProvider']
        def load_bundle(**kwargs):
            import onnxruntime
            if providers[0] not in onnxruntime.get_available_providers():
                raise RuntimeError(f'Requested ONNX provider is unavailable: {providers[0]}')
            return original_loader(**kwargs, providers=providers)
        old_root = os.environ.get('HOLOMOTION_UNITREE_POLICY_MODELS_DIR')
        os.environ['HOLOMOTION_UNITREE_POLICY_MODELS_DIR'] = str(root)
        policy.load_dual_policy_bundle = load_bundle
        try:
            self.node = build_sim_policy_node(quiet=True)
        finally:
            policy.load_dual_policy_bundle = original_loader
            if old_root is None:
                os.environ.pop('HOLOMOTION_UNITREE_POLICY_MODELS_DIR', None)
            else:
                os.environ['HOLOMOTION_UNITREE_POLICY_MODELS_DIR'] = old_root
        robot = yaml.safe_load(ROBOT_CONFIG.read_text())
        self.joint_names = tuple(robot['complete_dof_order'])
        self.default_dof_angles = np.array([robot['default_joint_angles'][n] for n in self.joint_names])
        self.reset()

    def reset(self):
        # Same fresh-start semantics as SIMPLE's node_state.fresh_start.
        from humanoid_policy.policy_runtime import PolicyRuntimeState
        from humanoid_policy.utils.remote_controller_filter import RemoteController
        n = self.node
        n.runtime.state = PolicyRuntimeState(num_actions=n.num_actions)
        n._init_obs_buffers()
        n.observation_evaluator.initialize_observation_state()
        n.observation_evaluator.clear_motion_yaw_alignment()
        n.observation_evaluator.clear_vr_fk_cache()
        if n.motion_kv_cache is not None:
            n.motion_kv_cache.fill(0)
        n.pico_controller = RemoteController()
        n._pico_enable_velocity = n._pico_switch_to_motion = False
        n._pico_last_receive_time = None
        self.started = False
        self.frame_index = 0
        self.steps = 0

    @staticmethod
    def _sample(**buttons):
        return dict(l_active=1, r_active=1, left_axis=[0., 0.], right_axis=[0., 0.], **buttons)

    def step(self, q, dq, quat, gyro, now, window=None, new_chunk=False):
        n = self.node
        with controller_clock(now):
            n.feed_lowstate(q, dq, quat, gyro)
            if not self.started:
                n.set_robot_state('MOVE_TO_DEFAULT')
                for pressed in (True, False):
                    n.feed_pico(self._sample(A=pressed))
                    n.run()
                if not n.policy_enabled or n.current_policy_mode != 'velocity':
                    raise RuntimeError('HoloMotion walking initialization failed')
                self.started = True
            if window is not None:
                if n.n_fut_frames != 10 or n.reference_delay_frames != 0:
                    raise ValueError('HoloMotion requires 10 future frames and zero delay')
                rows = window if new_chunk else window[-1:]
                first = self.frame_index if new_chunk else self.frame_index - 10
                for row in rows:
                    n.feed_reference(row[:65], self.frame_index, int(now * 1e9))
                    self.frame_index += 1
                n.feed_pico(self._sample(B=n.current_policy_mode != 'motion'))
            else:
                first = None
                n.feed_pico(self._sample())
            n.run()
            if window is not None:
                if not n.policy_enabled or n.current_policy_mode != 'motion':
                    raise RuntimeError('HoloMotion failed to enter motion mode')
                if n._vr_reference.current_frame_index() != first:
                    raise RuntimeError('HoloMotion consumed an unexpected reference frame')
                if n.runtime.state.motion_reference_frozen:
                    raise RuntimeError('HoloMotion reference unexpectedly froze')
            target = n.action_target
            if target is None or not np.isfinite(target).all():
                raise RuntimeError('HoloMotion returned no finite joint target')
            self.steps += 1
            return target, n.action_kps, n.action_kds
