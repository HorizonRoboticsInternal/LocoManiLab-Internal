"""Kimodo motion -> HEFT motion-tracking reference.

The tracker (HEFT ``G1_PMG``) consumes a reference in the GMR-style layout its
datasets use: ``root_pos``, ``root_rot``, ``dof_pos``, ``local_body_pos``,
``local_body_rot`` at 50 Hz. Kimodo emits MuJoCo ``qpos[T, 36]`` at 30 Hz for the
same 29-DoF Unitree G1, so this is a re-slice plus one FK, not a retarget.

Every element of the contract below was read off a released HEFT reference file
(``sim2real/config/g1/motions/omni_extreme/omni_extreme_1.npz``), which ships both
the inputs (``debug_qpos``, [T, 36]) and the expected outputs, so the conversion
is checked against ground truth rather than against a reading of the docs:

  * joint order        identical to Kimodo/SONIC's 29-DoF order, element for
                       element -- no permutation
  * ``root_pos``       ``qpos[:, 0:3]``            exact (0.0)
  * ``root_rot``       ``qpos[:, 3:7]`` wxyz->xyzw exact (0.0)
  * ``dof_pos``        ``qpos[:, 7:36]``           exact (0.0)
  * ``local_body_pos`` R_root^T (x_world - root), i.e. the FULL root frame, not
                       a yaw-only heading frame (yaw-only is off by 0.63 m)
  * model              ``g1_5.xml``, NOT ``g1.xml``. Both load and both give
                       plausible numbers; g1.xml is off by 4.9 mm on every body
                       while g1_5.xml matches to 1.4e-06. Picking the wrong one
                       injects a silent systematic offset into every reference.
  * body set           the 30 names in the npz, a subset of the model's 35 in a
                       different order -- resolved by name, never by index.

What the tracker actually needs is a strict subset of that. Reading the deployed
runtime's observation assembly (``sim2real/src/runtime/policy.py`` and
``observation.py``), the only reference-derived terms are:

  ``ref_root_pos``   inter-step displacement in the reference root frame, and
                     root height z
  ``ref_root_quat``  relative rotation vs the robot's current quaternion (as
                     rot6d), and the projected gravity direction
  ``ref_joint_pos``  target joint angles, and target-minus-current

Everything else in the 1729-dim observation is the robot's own proprioception
history. ``local_body_pos`` / ``local_body_rot`` are **not consumed by the
policy at all** -- so driving the tracker needs no FK, just resample and slice.
The FK path below exists for writing files in their dataset format, which the
training-side command uses for keypoint rewards; those rewards are being
replaced anyway.

One convention trap: the npz stores ``root_rot`` as xyzw, but the runtime feeds
the quaternion to ``scipy`` with ``scalar_first=True``, i.e. wxyz. On-disk and
in-memory orders differ; ``build_policy_reference`` returns the in-memory (wxyz)
form the policy path wants, ``from_qpos36`` returns the on-disk (xyzw) form.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

KIMODO_FPS = 30.0
HEFT_FPS = 50.0


def resolve_reference_xml(xml_path: str | Path | None = None) -> Path:
    """Select the matching g1_5.xml explicitly; never substitute another G1 model."""
    selected = xml_path or os.environ.get("HEFT_G1_XML")
    if not selected:
        raise ValueError("Set HEFT_G1_XML or pass xml_path to the matching HEFT g1_5.xml model.")
    resolved = Path(selected).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            f"HEFT reference model not found: {resolved}. Select the matching g1_5.xml "
            "with HEFT_G1_XML or xml_path; g1.xml is not interchangeable."
        )
    return resolved


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """[..., 4] wxyz -> [..., 3, 3]."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(*q.shape[:-1], 3, 3)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, wxyz."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def resample(qpos: np.ndarray, src_fps: float = KIMODO_FPS,
             dst_fps: float = HEFT_FPS) -> np.ndarray:
    """Linear resample of ``qpos[T, 36]``, with the quaternion slerped-by-lerp.

    Root position and joints interpolate linearly; the quaternion is lerped and
    renormalised after hemisphere alignment, which is what a 30->50 Hz step over
    a smooth motion needs (the per-step angle is small enough that lerp and slerp
    agree to well under the tracker's sensitivity).
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    T = qpos.shape[0]
    if T < 2:
        return qpos.copy()
    n_out = max(2, int(round((T - 1) * dst_fps / src_fps)) + 1)
    t = np.linspace(0.0, T - 1, n_out)
    i0 = np.floor(t).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    w = (t - i0).astype(np.float32)[:, None]

    out = np.empty((n_out, qpos.shape[1]), dtype=np.float32)
    out[:, 0:3] = (1 - w) * qpos[i0, 0:3] + w * qpos[i1, 0:3]
    out[:, 7:] = (1 - w) * qpos[i0, 7:] + w * qpos[i1, 7:]

    q0, q1 = qpos[i0, 3:7], qpos[i1, 3:7].copy()
    flip = (q0 * q1).sum(-1) < 0            # take the short way round
    q1[flip] *= -1
    q = (1 - w) * q0 + w * q1
    out[:, 3:7] = q / np.linalg.norm(q, axis=-1, keepdims=True).clip(1e-8)
    return out


class HeftReference:
    """Builds HEFT reference dicts from Kimodo qpos, via MuJoCo FK."""

    def __init__(self, body_names, xml_path: str | Path | None = None):
        xml_path = str(resolve_reference_xml(xml_path))
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.body_names = [str(b) for b in body_names]
        self.body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
                         for n in self.body_names]
        missing = [n for n, i in zip(self.body_names, self.body_ids) if i < 0]
        if missing:
            raise ValueError(f"{xml_path} has no bodies named {missing}")
        if self.model.nq != 36:
            raise ValueError(f"expected nq=36 (3+4+29), got {self.model.nq} from {xml_path}")

    def from_qpos36(self, qpos36: np.ndarray, *, src_fps: float = KIMODO_FPS,
                    resample_to: float | None = HEFT_FPS) -> dict:
        """``qpos[T, 36]`` -> the field set a HEFT reference needs."""
        q = np.asarray(qpos36, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != 36:
            raise ValueError(f"expected [T, 36], got {q.shape}")
        if resample_to is not None and abs(src_fps - resample_to) > 1e-6:
            q = resample(q, src_fps, resample_to)
            fps = resample_to
        else:
            fps = src_fps

        root_pos = q[:, 0:3].copy()
        root_quat_wxyz = q[:, 3:7]
        dof_pos = q[:, 7:36].copy()

        n = q.shape[0]
        nb = len(self.body_ids)
        xw = np.empty((n, nb, 3), dtype=np.float32)
        qw = np.empty((n, nb, 4), dtype=np.float32)
        for f in range(n):
            self.data.qpos[:] = q[f]
            self._mj.mj_forward(self.model, self.data)
            xw[f] = self.data.xpos[self.body_ids]
            qw[f] = self.data.xquat[self.body_ids]

        # Full root frame, not a heading frame -- verified against a released
        # reference file; yaw-only is off by 0.63 m.
        R = _quat_wxyz_to_rotmat(root_quat_wxyz)                    # [T,3,3]
        local_body_pos = np.einsum("tij,tbi->tbj", R, xw - root_pos[:, None, :])
        root_conj = root_quat_wxyz * np.array([1, -1, -1, -1], dtype=np.float32)
        local_body_rot_wxyz = _quat_mul(root_conj[:, None, :], qw)

        return {
            "fps": np.float32(fps),
            "root_pos": root_pos,
            "root_rot": np.concatenate([root_quat_wxyz[:, 1:4],
                                        root_quat_wxyz[:, 0:1]], axis=1),  # xyzw
            "dof_pos": dof_pos,
            "local_body_pos": local_body_pos.astype(np.float32),
            "local_body_rot": np.concatenate([local_body_rot_wxyz[..., 1:4],
                                              local_body_rot_wxyz[..., 0:1]], axis=-1),
            "body_names": np.array(self.body_names),
        }
