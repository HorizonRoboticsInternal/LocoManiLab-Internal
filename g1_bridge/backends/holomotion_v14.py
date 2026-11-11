"""HoloMotion v1.4 as a native G1_BRIDGE backend, with Revo2 references."""
from __future__ import annotations

import numpy as np

from g1_bridge.backends.base import BackendResources, Sim2RealBackend


class HolomotionV14Backend(Sim2RealBackend):
    name = 'holomotion'
    controller_json = 'g1_holomotion_v14.json'
    uses_gmr = False
    mode_profile = 'tracking'
    owns_policy_runtime = True
    vla_action_space = 'holomotion_reference_v1'
    POLICY_CONFIGS = {'holomotion': 'checkpoints/holomotion/v14/policy.yaml'}

    def build_tracking(self, cfg, policy_config_path, inference_backend):
        from g1_bridge.holomotion_v14.runtime import HolomotionV14Runtime
        runtime = HolomotionV14Runtime(policy_config_path, inference_backend)
        return BackendResources(tracking=runtime, io=None)

    def on_load(self, ctx, cfg, res):
        from g1_bridge.g1_runtime import G1_JOINT_NAMES
        names = list(res.tracking.joint_names)
        if len(names) != 29 or set(names) != set(G1_JOINT_NAMES):
            raise ValueError('HoloMotion body joints do not match G1 host')
        self._to_policy = np.array([G1_JOINT_NAMES.index(n) for n in names])
        self._to_host = np.array([names.index(n) for n in G1_JOINT_NAMES])
        self.reset(ctx)

    def reset(self, ctx):
        ctx.tracking.reset()
        self._reference = None
        self._last_step = -1

    def vla_reset(self):
        self._reference = None
        self._last_step = -1

    def _run(self, ctx, tick, window=None, new_chunk=False):
        quat, gyro = ctx.read_pelvis_state()
        target, kp, kd = ctx.tracking.step(
            tick.qpos[self._to_policy], tick.qvel[self._to_policy],
            quat, gyro, ctx.clock_s(), window, new_chunk,
        )
        return tuple(np.asarray(v)[self._to_host] for v in (target, kp, kd))

    def step(self, ctx, tick):
        return self._run(ctx, tick)

    def vla_step(self, ctx, tick, action):
        window = np.asarray(action['reference_window'], dtype=np.float32)
        if window.shape != (11, 87) or not np.isfinite(window).all():
            raise ValueError('HoloMotion requires a finite (11,87) reference window')
        if np.any(np.abs(np.linalg.norm(window[:, 61:65], axis=1) - 1) > .1):
            raise ValueError('HoloMotion root quaternion is not normalized')
        step = int(action['reference_step'])
        new = bool(action['reference_new_chunk'])
        if step != self._last_step + 1 or new != (step % 16 == 0):
            raise ValueError('HoloMotion references require sequential steps and replanning every 16 steps')
        result = self._run(ctx, tick, window, new)
        self._last_step = step
        self._reference = window[0].copy()
        return result

    def vla_hand_targets(self, action):
        row = np.asarray(action['reference_window'], dtype=float)[0]
        return row[65:76].copy(), row[76:87].copy()

    def teleop_sample(self, ctx):
        raise NotImplementedError('HoloMotion v14 collection is not implemented; use the VLA evaluation path')
