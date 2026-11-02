"""Build the observation dict a GR00T-compatible policy server expects.

The schema is the same whether the robot is real or simulated -- which is the
point of putting it here. Both callers previously carried their own copy of
the joint-group table below, and a copy that drifts produces a policy that
runs, returns plausible numbers, and drives the robot wrong: the arm slice
lands in the leg slot and nothing raises.

Every tensor carries leading ``[batch=1, time=1]`` dims, as the servers expect.
"""

from __future__ import annotations

import numpy as np

#: Slices of the 43-DoF whole-body joint vector, matching the
#: ``unitree_g1_sonic`` embodiment's ``meta/modality.json`` state groups.
#: Order is load-bearing: this IS the layout the models were trained on.
G1_43DOF_STATE_SLICES: dict[str, tuple[int, int]] = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "left_hand": (22, 29),
    "right_arm": (29, 36),
    "right_hand": (36, 43),
}

DEFAULT_PROMPT_KEY = "annotation.human.task_description"


def projected_gravity(base_quat_wxyz) -> np.ndarray:
    """Gravity expressed in the robot base frame.

    Must match how the training data computed it; this is the same formula the
    collection scripts use.
    """
    from scipy.spatial.transform import Rotation

    quat = np.asarray(base_quat_wxyz, dtype=np.float64)
    r = Rotation.from_quat(quat[[1, 2, 3, 0]])  # wxyz -> xyzw
    return r.inv().apply([0.0, 0.0, -1.0]).astype(np.float32)


def split_state(
    joint_pos, slices: dict[str, tuple[int, int]] | None = None
) -> dict[str, np.ndarray]:
    """Split a flat joint vector into the policy's named state groups."""
    slices = slices or G1_43DOF_STATE_SLICES
    q = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
    expected = max(hi for _, hi in slices.values())
    if q.shape[0] != expected:
        raise ValueError(
            f"joint vector has {q.shape[0]} entries, but the state layout "
            f"covers {expected}. Wrong embodiment, or the wrong joint order."
        )
    return {name: q[lo:hi][None, None] for name, (lo, hi) in slices.items()}


def build_observation(
    *,
    joint_pos,
    images: dict[str, np.ndarray],
    prompt: str,
    base_quat_wxyz=None,
    state_slices: dict[str, tuple[int, int]] | None = None,
    prompt_key: str = DEFAULT_PROMPT_KEY,
    extra_state: dict[str, np.ndarray] | None = None,
) -> dict:
    """Assemble one observation.

    Args:
        joint_pos: flat whole-body joint vector (43-DoF for unitree_g1_sonic).
        images: camera name -> ``(H, W, 3)`` uint8 frame. ``ego_view`` is the
            head camera every current checkpoint reads; wrist views are
            optional and simply passed through under their own names.
        prompt: the language instruction. Must match what the checkpoint was
            trained with -- an out-of-distribution prompt is a silent failure,
            not an error.
        base_quat_wxyz: pelvis orientation, for ``state.projected_gravity``.
            Omit only for checkpoints that do not consume it.
        extra_state: additional ``state.*`` entries, already batched.
    """
    state = split_state(joint_pos, state_slices)
    if base_quat_wxyz is not None:
        state["projected_gravity"] = projected_gravity(base_quat_wxyz)[None, None]
    if extra_state:
        state.update(extra_state)

    video = {
        name: np.ascontiguousarray(frame, dtype=np.uint8)[None, None]
        for name, frame in images.items()
    }
    return {
        "video": video,
        "state": state,
        "language": {prompt_key: [[prompt]]},
    }
