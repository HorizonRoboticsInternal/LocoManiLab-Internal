"""AMO (Li/Cheng/Huang/Wang, RSS 2025) observations.

Verbatim port of ``AMOG1Composite._get_observation`` / ``_advance_gait``
(``robosuite_src/robosuite/controllers/composite/amo_g1_composite.py`` in
tele_humanoid_benchmark -- the pre-existing, standalone ``AMO_G1`` composite
this checkpoint family was already integrated under) into sim2real's
Observation framework, following the same "shared stateful core, one
Observation subclass per named output" pattern ``scalebfm.py`` already uses
for its own multi-tensor / cross-tick-state observations
(``_ScaleBFMCore`` / ``_get_core`` / ``output(name)``).

AMO's main policy takes two separate tensors -- ``obs`` (current proprio +
demo + priv + 10-step history) and ``extra`` (a 25-step raw-proprio history)
-- built here as ``amo_obs`` / ``amo_extra``. The adapter sub-model
(``adapter_jit.pt`` + its normalization stats) that produces part of ``obs``
runs *inside* this module (loaded via the ``torch-jit`` inference backend),
not as a separate top-level policy stage -- AMO's own deployment code
(``play_amo.py`` / Psi0 ``master_whole_body.py``) treats it the same way.

Two things this module cannot get from sim2real's own state (no GMR/motion
reference at all is used by AMO -- locomotion is command-velocity-driven,
not motion-tracking-driven):

* ``env.amo_commands`` -- an (8,) array ``[vx, target_yaw, vy, height_delta,
  torso_yaw, torso_pitch, torso_roll, arm_toggle]``, exactly
  ``AMOG1Composite._commands``. Set by the bridge (PICO joystick -> velocity
  command, ported verbatim as well) before each ``Tracking.step()``.
* ``env.amo_vyaw`` -- the current commanded yaw *rate* (separate from the
  integrated target heading in ``amo_commands[1]``), used only for the
  in-place-stand gate.

Both default to zero (stand still, no yaw-only turn) if the bridge hasn't set
them -- e.g. during a self-check that never touches the composite layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from sim2real.rl_policy.observations.base import Observation

# The canonical device-constant patcher lives with the TorchScript loader;
# the alias keeps the adapter on the same code the main policy already uses
# (this file used to carry a byte-identical private copy).
from sim2real.rl_policy.inference.torch_module import patch_jit_to_cpu as _patch_jit_to_cpu


def _resolve_path(env, path: str) -> str:
    """Relative ``adapter_path``/``norm_stats_path`` in the yaml resolve
    against the policy_config's own directory (same convention
    ``encoder_model_path``/``model_path`` use elsewhere)."""
    p = Path(path)
    if p.is_absolute():
        return str(p)
    policy_config = getattr(getattr(env, "args", None), "policy_config", None)
    if not policy_config:
        return str(p)
    return str(Path(policy_config).parent / p)


def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
    """[w, x, y, z] -> [roll, pitch, yaw]. Verbatim from play_amo.quatToEuler
    (via AMOG1Composite._quat_to_euler)."""
    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    sinr_cosp = 2 * (qw * qx + qy * qz)
    cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (qw * qy - qz * qx)
    if np.abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    siny_cosp = 2 * (qw * qz + qx * qy)
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


# ---- play_amo HumanoidEnv constants (== AMOG1Composite's own copies) ----
_CONTROL_DT = 0.02
_ACTION_SCALE = 0.25
_GAIT_FREQ = 1.3
_SCALE_ANG_VEL = 0.25
_SCALE_DOF_VEL = 0.05
_N_PRIV = 3
_N_PROPRIO = 3 + 2 + 2 + 23 * 3 + 2 + 15  # 93
_HISTORY_LEN = 10
_EXTRA_HISTORY_LEN = 25
_N_DEMO_DOF = 8

_DEFAULT_DOF_POS = np.array(
    [
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.5, 0.0, 0.2, 0.3,
        0.5, 0.0, -0.2, 0.3,
    ],
    dtype=np.float32,
)


class _AMOCore:
    """Shared per-``env`` state: proprio/extra history, gait phase, last
    action, and the adapter sub-model -- everything ``_get_observation``
    needs, cached exactly once regardless of how many named Observation
    instances (``amo_obs``, ``amo_extra``) reference it."""

    def __init__(
        self,
        env,
        *,
        joint_names: Sequence[str],
        adapter_path: str,
        norm_stats_path: str,
        device: str = "cpu",
    ):
        self.env = env
        self.joint_names = tuple(str(name) for name in joint_names)
        if len(self.joint_names) != 23:
            raise ValueError(
                f"AMO expects exactly 23 joint_names (AMO's own DOF order), got "
                f"{len(self.joint_names)}"
            )
        self.device = device
        self._joint_indices: list[int] | None = None

        self._adapter = _patch_jit_to_cpu(
            torch.jit.load(adapter_path, map_location=device)
        ).eval()
        for p in self._adapter.parameters():
            p.requires_grad = False
        norm_stats = torch.load(norm_stats_path, weights_only=False)
        self._in_mean = torch.tensor(norm_stats["input_mean"], dtype=torch.float32, device=device)
        self._in_std = torch.tensor(norm_stats["input_std"], dtype=torch.float32, device=device)
        self._out_mean = torch.tensor(norm_stats["output_mean"], dtype=torch.float32, device=device)
        self._out_std = torch.tensor(norm_stats["output_std"], dtype=torch.float32, device=device)

        self._gait_cycle = np.array([0.25, 0.25], dtype=np.float32)
        self._in_place_stand = True
        self._proprio_hist: list[np.ndarray] = []
        self._extra_hist: list[np.ndarray] = []
        self._demo_template = np.zeros(_N_DEMO_DOF + 9, dtype=np.float32)
        self._demo_template[:_N_DEMO_DOF] = _DEFAULT_DOF_POS[15:]
        self._demo_template[_N_DEMO_DOF + 6 : _N_DEMO_DOF + 9] = 0.75

        self._obs = np.zeros(_N_PROPRIO + (_N_DEMO_DOF + 9) + _N_PRIV + _HISTORY_LEN * _N_PROPRIO, dtype=np.float32)
        self._extra = np.zeros(_EXTRA_HISTORY_LEN * _N_PROPRIO, dtype=np.float32)
        self._last_update_token: tuple[int, int] | None = None
        self.reset()

    def reset(self) -> None:
        self._gait_cycle = np.array([0.25, 0.25], dtype=np.float32)
        self._in_place_stand = True
        self._proprio_hist = [np.zeros(_N_PROPRIO, dtype=np.float32) for _ in range(_HISTORY_LEN)]
        self._extra_hist = [np.zeros(_N_PROPRIO, dtype=np.float32) for _ in range(_EXTRA_HISTORY_LEN)]
        self._last_update_token = None

    def _refresh_joint_indices(self) -> None:
        if self._joint_indices is not None:
            return
        sim_joint_names = tuple(self.env.state_processor.joint_names)
        missing = [name for name in self.joint_names if name not in sim_joint_names]
        if missing:
            raise ValueError(f"AMO robot state missing joints: {missing}")
        self._joint_indices = [sim_joint_names.index(name) for name in self.joint_names]

    def _commands(self) -> np.ndarray:
        return np.asarray(
            getattr(self.env, "amo_commands", np.zeros(8, dtype=np.float32)),
            dtype=np.float32,
        )

    def _vyaw(self) -> float:
        return float(getattr(self.env, "amo_vyaw", 0.0))

    def _adapter_out(self, dof_pos: np.ndarray, commands: np.ndarray) -> np.ndarray:
        adapter_in = np.concatenate([np.zeros(4, dtype=np.float32), dof_pos[15:]]).astype(np.float32)
        adapter_in[0] = 0.75 + commands[3]
        adapter_in[1] = commands[4]
        adapter_in[2] = commands[5]
        adapter_in[3] = commands[6]
        ai = torch.tensor(adapter_in, device=self.device, dtype=torch.float32).unsqueeze(0)
        ai = (ai - self._in_mean) / (self._in_std + 1e-8)
        with torch.no_grad():
            ao = self._adapter(ai.view(1, -1)) * self._out_std + self._out_mean
        return ao.cpu().numpy().squeeze()

    def update_once(self, data: dict[str, Any]) -> None:
        token = (id(data), int(getattr(self.env, "total_inference_cnt", -1)))
        if self._last_update_token == token:
            return
        self._last_update_token = token

        self._refresh_joint_indices()
        state = self.env.state_processor
        joint_pos = np.asarray(state.joint_pos, dtype=np.float32).reshape(-1)
        joint_vel = np.asarray(state.joint_vel, dtype=np.float32).reshape(-1)
        dof_pos = joint_pos[self._joint_indices]
        dof_vel = joint_vel[self._joint_indices]
        quat = np.asarray(state.root_quat_w, dtype=np.float32).reshape(4)
        norm = float(np.linalg.norm(quat))
        quat = quat / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ang_vel = np.asarray(state.root_ang_vel_b, dtype=np.float32).reshape(3)

        last_action = np.asarray(
            data.get("action", np.zeros(len(self.joint_names), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1)

        c = self._commands()
        rpy = _quat_to_euler(quat)
        target_yaw = c[1]
        dyaw = np.remainder(rpy[2] - target_yaw + np.pi, 2 * np.pi) - np.pi
        if self._in_place_stand:
            dyaw = 0.0

        gait_obs = np.sin(self._gait_cycle * 2 * np.pi)
        adapter_out = self._adapter_out(dof_pos, c)

        obs_prop = np.concatenate(
            [
                ang_vel * _SCALE_ANG_VEL,
                rpy[:2],
                np.array([np.sin(dyaw), np.cos(dyaw)], dtype=np.float32),
                dof_pos - _DEFAULT_DOF_POS,
                dof_vel * _SCALE_DOF_VEL,
                last_action,
                gait_obs,
                adapter_out,
            ]
        ).astype(np.float32)

        obs_priv = np.zeros(_N_PRIV, dtype=np.float32)
        obs_hist = np.array(self._proprio_hist, dtype=np.float32).flatten()

        obs_demo = self._demo_template.copy()
        obs_demo[:_N_DEMO_DOF] = dof_pos[15:]
        obs_demo[_N_DEMO_DOF] = c[0]
        obs_demo[_N_DEMO_DOF + 1] = c[2]
        vyaw = self._vyaw()
        self._in_place_stand = bool(np.abs(c[0]) < 0.1 and np.abs(c[2]) < 0.1 and np.abs(vyaw) < 0.1)
        obs_demo[_N_DEMO_DOF + 3] = c[4]
        obs_demo[_N_DEMO_DOF + 4] = c[5]
        obs_demo[_N_DEMO_DOF + 5] = c[6]
        obs_demo[_N_DEMO_DOF + 6 : _N_DEMO_DOF + 9] = 0.75 + c[3]

        self._proprio_hist.append(obs_prop)
        self._proprio_hist.pop(0)
        self._extra_hist.append(obs_prop)
        self._extra_hist.pop(0)

        self._obs = np.concatenate((obs_prop, obs_demo, obs_priv, obs_hist)).astype(np.float32)
        self._extra = np.array(self._extra_hist, dtype=np.float32).flatten()

        # advance gait phase for the *next* tick's gait_obs -- matches
        # AMOG1Composite._advance_gait, called after this tick's obs+action.
        self._gait_cycle = np.remainder(self._gait_cycle + _CONTROL_DT * _GAIT_FREQ, 1.0)
        g = self._gait_cycle
        if self._in_place_stand and (np.abs(g[0] - 0.25) < 0.05 or np.abs(g[1] - 0.25) < 0.05):
            self._gait_cycle = np.array([0.25, 0.25], dtype=np.float32)
        if (not self._in_place_stand) and (np.abs(g[0] - 0.25) < 0.05 and np.abs(g[1] - 0.25) < 0.05):
            self._gait_cycle = np.array([0.25, 0.75], dtype=np.float32)

    def output(self, name: str) -> np.ndarray:
        return self._obs if name == "obs" else self._extra


def _core_key(*, joint_names, adapter_path, norm_stats_path, device):
    return ("amo", tuple(joint_names), str(adapter_path), str(norm_stats_path), str(device))


def _get_core(env, **kwargs) -> _AMOCore:
    key = _core_key(**kwargs)
    cache = getattr(env, "_amo_core_cache", None)
    if cache is None:
        cache = {}
        setattr(env, "_amo_core_cache", cache)
    if key not in cache:
        cache[key] = _AMOCore(env, **kwargs)
    return cache[key]


class _AMOObservation(Observation, namespace="amo"):
    output_name: str

    def __init__(
        self,
        joint_names: Sequence[str],
        adapter_path: str,
        norm_stats_path: str,
        device: str = "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.core = _get_core(
            self.env,
            joint_names=joint_names,
            adapter_path=_resolve_path(self.env, adapter_path),
            norm_stats_path=_resolve_path(self.env, norm_stats_path),
            device=device,
        )

    def reset(self) -> None:
        self.core.reset()

    def update(self, data: dict[str, Any]) -> None:
        self.core.update_once(data)

    def compute(self) -> np.ndarray:
        return self.core.output(self.output_name)


class amo_obs(_AMOObservation, namespace="amo"):
    output_name = "obs"


class amo_extra(_AMOObservation, namespace="amo"):
    output_name = "extra"
