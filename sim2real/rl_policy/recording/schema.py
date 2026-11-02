"""Derive LeRobot features + a frame-flattening function from a running BasePolicy.

The point: a policy family gets LeRobot-episode recording just by declaring
`--record --record-format lerobot` -- no new per-family Python code. Every
`observation.<group>` feature comes straight from the `observation` groups the
family's own `policy.yaml` already declares (resolved into `policy.observations`/
per-tick `obs_components` by `BasePolicy.setup_observations()`/`prepare_obs_for_rl()`);
baseline robot-state and action fields come from what `step()` already computes every
tick. Only truly out-of-band signals (camera images, hand-grip) need an explicit CLI
opt-in (`--camera-connect`, `--hand-grip-connect`), since they aren't part of the
`Observation` registry at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def derive_features(
    policy: Any,
    sample_frame: dict[str, Any],
    image_shapes: dict[str, tuple[int, int, int]],
    include_hand_grip: bool,
) -> dict[str, dict]:
    num_dofs = policy.num_dofs
    joint_names = list(policy.joint_names_simulation)

    features: dict[str, dict] = {
        "observation.joint_pos": {
            "dtype": "float32", "shape": (num_dofs,), "names": joint_names,
        },
        "observation.joint_vel": {
            "dtype": "float32", "shape": (num_dofs,), "names": joint_names,
        },
        "observation.root_quat_w": {
            "dtype": "float32", "shape": (4,), "names": ["w", "x", "y", "z"],
        },
        "observation.root_ang_vel_b": {
            "dtype": "float32", "shape": (3,), "names": ["x", "y", "z"],
        },
    }

    for group_name, group_obs in sample_frame["policy"]["obs"].items():
        dim = sum(int(np.asarray(v).reshape(-1).shape[0]) for v in group_obs.values())
        features[f"observation.{group_name}"] = {
            "dtype": "float32", "shape": (dim,), "names": None,
        }

    features["action.q_target"] = {
        "dtype": "float32", "shape": (num_dofs,), "names": joint_names,
    }
    features["action.raw"] = {
        "dtype": "float32",
        "shape": (len(policy.policy_joint_names),),
        "names": list(policy.policy_joint_names),
    }

    for name, shape in image_shapes.items():
        features[f"observation.images.{name}"] = {
            "dtype": "video", "shape": shape, "names": ["height", "width", "channel"],
        }

    if include_hand_grip:
        features["action.hand_grip"] = {
            "dtype": "float32", "shape": (2,), "names": ["left", "right"],
        }

    return features


def derive_modality_config(features: dict[str, dict]) -> dict:
    """meta/modality.json group layout: one group per top-level feature key."""
    modality_config: dict[str, dict] = {"state": {}, "action": {}}
    offsets: dict[str, int] = {"state": 0, "action": 0}
    for key, feature in features.items():
        if feature["dtype"] == "video":
            continue
        top, _, sub = key.partition(".")
        section = "state" if top == "observation" else "action"
        dim = int(feature["shape"][0])
        modality_config[section][sub or top] = {
            "start": offsets[section], "end": offsets[section] + dim,
        }
        offsets[section] += dim
    return modality_config


def flatten_frame(
    frame: dict[str, Any],
    task: str,
    image: dict[str, np.ndarray] | None,
    hand_grip: tuple[float, float] | None,
) -> dict[str, Any]:
    """Convert one BasePolicy per-tick frame dict into a flat LeRobot frame."""
    flat: dict[str, Any] = {
        "task": task,
        "observation.joint_pos": frame["robot"]["joint_pos"],
        "observation.joint_vel": frame["robot"]["joint_vel"],
        "observation.root_quat_w": frame["robot"]["root_quat_w"],
        "observation.root_ang_vel_b": frame["robot"]["root_ang_vel_b"],
        "action.q_target": frame["policy"]["q_target"],
        "action.raw": frame["policy"]["action"],
    }
    for group_name, group_obs in frame["policy"]["obs"].items():
        flat[f"observation.{group_name}"] = np.concatenate(
            [np.asarray(v).reshape(-1) for v in group_obs.values()]
        ).astype(np.float32)

    if image is not None:
        for name, array in image.items():
            flat[f"observation.images.{name}"] = array

    if hand_grip is not None:
        flat["action.hand_grip"] = np.asarray(hand_grip, dtype=np.float32)

    return flat
