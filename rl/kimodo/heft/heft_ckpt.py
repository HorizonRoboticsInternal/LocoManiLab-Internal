"""Load the frozen HEFT tracker on the robosuite side, without torch.

Peer of ``g1_bridge/sonic_utils/ckpt.py``: it
resolves which artifact to use and hands back something callable. The difference
is that this loader requires an existing, compatible HEFT tracker ONNX selected
with ``HEFT_POLICY_ONNX`` or an explicit path. It does not export or download a
policy.

**The no-torch rule is not stylistic.** Rollout workers create MuJoCo offscreen
contexts; a process holding both an EGL and a CUDA context deadlocks on shared
multi-GPU nodes (DEBUG_LOG #11), which is why ``vec_task_env`` runs its workers
with rendering off and no CUDA. Importing torch in a worker re-opens that door
even if no tensor is ever moved to a GPU. onnxruntime's CPU provider has no such
behaviour, and the robosuite env already carries it for SONIC.

``load_session`` therefore asserts the CPU provider explicitly rather than
trusting the default: onnxruntime silently prefers CUDA when onnxruntime-gpu is
installed, and that would put a CUDA context in every worker without any error.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from rl.kimodo.heft.tracker_const import ACTION_CLIP, ACTION_SCALE, N_JOINTS, OBS_DIM

_REPO = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_DIR = _REPO / "rl" / "kimodo" / "_out" / "heft"
ONNX_NAME = "heft_policy_loc.onnx"
REF_NAME = "heft_policy_ref.npz"

ENV_VAR = "HEFT_POLICY_ONNX"


def resolve_onnx(path: str | Path | None = None) -> Path:
    """Explicit path -> $HEFT_POLICY_ONNX -> the default artifact dir."""
    for cand in (path, os.environ.get(ENV_VAR), DEFAULT_ARTIFACT_DIR / ONNX_NAME):
        if cand:
            p = Path(cand)
            if p.exists():
                return p
    raise SystemExit(
        f"[heft_ckpt] no policy ONNX found. Set {ENV_VAR} to an existing compatible\n"
        "HEFT tracker ONNX file, or pass its path explicitly. This loader does\n"
        "not export or download policy artifacts.")


def resolve_reference(path: str | Path | None = None) -> Path | None:
    p = Path(path) if path else DEFAULT_ARTIFACT_DIR / REF_NAME
    return p if p.exists() else None


def load_session(path: str | Path | None = None, *, threads: int = 1):
    """An onnxruntime CPU session for the tracker.

    ``threads=1`` by default: these run one-per-rollout-worker and oversubscribing
    BLAS across workers costs more than it gains (the same reasoning
    ``vec_task_env._worker`` applies to OMP/MKL).
    """
    import onnxruntime as ort

    onnx_path = resolve_onnx(path)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx_path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    got = sess.get_providers()
    if got != ["CPUExecutionProvider"]:
        raise SystemExit(
            f"[heft_ckpt] session came up on {got}; a CUDA context in a rollout "
            f"worker deadlocks against EGL (DEBUG_LOG #11)")
    return sess


class HeftPolicy:
    """[N, 1729] -> (delta, raw), both in CANONICAL joint order.

    Both are returned because they are not interchangeable, and confusing them
    is one of the five silent bugs heft_work documented: the robot is driven by
    ``delta = raw * ACTION_SCALE``, while the ``PrevActions`` observation block
    is filled with ``raw`` -- clipped but NOT scaled. Storing the scaled value
    corrupts 232 of the 1729 input dims with nothing raised.

    Neither output is permuted here. The caller owns the mapping into whatever
    order its simulator exposes, and must resolve it by joint name.
    """

    def __init__(self, path: str | Path | None = None, *, threads: int = 1):
        self.session = load_session(path, threads=threads)
        self.path = resolve_onnx(path)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        # Static width check: a wrong-width ONNX would otherwise fail deep in the
        # first rollout step, long after the cause.
        width = inp.shape[-1]
        if isinstance(width, int) and width != OBS_DIM:
            raise SystemExit(f"[heft_ckpt] {self.path.name} takes {width} inputs, "
                             f"expected {OBS_DIM}")
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        if obs.shape[-1] != OBS_DIM:
            raise ValueError(f"expected [..., {OBS_DIM}], got {obs.shape}")
        raw = self.session.run([self.output_name], {self.input_name: obs})[0]
        raw = np.clip(raw, -ACTION_CLIP, ACTION_CLIP)
        return raw * ACTION_SCALE, raw

    def __repr__(self) -> str:
        return f"HeftPolicy({self.path.name}, {OBS_DIM}->{N_JOINTS}, CPU)"
