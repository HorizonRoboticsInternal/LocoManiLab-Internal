"""Backward-compatible flat-npz episode recorder.

With ``episodes_enabled=False`` (today's default: ``--record`` alone, no
``--record-episodes``), this reproduces BasePolicy's original recording behavior
byte-for-byte: frames accumulate from the first step, and one ``.npz`` is written on
process exit (``policy_tracking_record_v1`` schema, same ``metadata``/``robot``/
``policy``/``runtime`` layout the old inline implementation produced).

``episodes_enabled=True`` is new, additive behavior: recording starts idle and only
``toggle()`` starts/stops it, writing one timestamped ``.npz`` per saved episode --
``discard()`` drops the in-progress buffer instead of saving it.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from sim2real.rl_policy.recording.base import EpisodeRecorderBase, record_array


class NpzEpisodeRecorder(EpisodeRecorderBase):
    def __init__(self, policy: Any, output_path: Path, episodes_enabled: bool = False) -> None:
        self._policy = policy
        self._output_path = Path(output_path)
        self._episodes_enabled = episodes_enabled
        self._frames: list[dict[str, Any]] = []
        self._start_time_ns: int | None = None
        # Legacy (episodes disabled) behavior: implicitly "recording" for the whole
        # process, matching what `--record` alone always did before episode support.
        self._recording_active = not episodes_enabled
        self._episode_index = 0

    def append_frame(self, frame: dict[str, Any]) -> None:
        if not self._recording_active:
            return
        if self._start_time_ns is None:
            self._start_time_ns = frame["time_ns"]
        self._frames.append(frame)

    def toggle(self, *, source: str) -> None:
        if not self._episodes_enabled:
            return
        if self._recording_active:
            self._save_current(source=source)
            self._recording_active = False
        else:
            self._frames = []
            self._start_time_ns = None
            self._recording_active = True
            logger.info("Recording started via {}", source)

    def discard(self, *, source: str) -> None:
        if not self._episodes_enabled:
            return
        if self._frames:
            logger.info("Discarding {} buffered frame(s) via {}", len(self._frames), source)
        self._frames = []
        self._start_time_ns = None
        self._recording_active = False

    def close(self) -> None:
        if self._episodes_enabled:
            if self._recording_active and self._frames:
                self._save_current(source="close")
            return
        # Legacy path: always save once, exactly like the original _save_recording().
        self._save_current(source="exit")

    def _resolve_output_path(self) -> Path:
        if not self._episodes_enabled:
            return self._output_path
        self._episode_index += 1
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        stem = self._output_path.stem
        suffix = self._output_path.suffix or ".npz"
        return self._output_path.parent / f"{stem}_ep{self._episode_index:04d}_{timestamp}{suffix}"

    def _save_current(self, *, source: str) -> None:
        if not self._frames:
            return
        data = self._build_record_data()
        path = self._resolve_output_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, data=np.asarray(data, dtype=object))
        logger.info(
            "Saved {} policy tracking frames to {} (via {})", len(self._frames), path, source
        )

    def _build_record_data(self) -> dict[str, Any]:
        frames = self._frames
        policy = self._policy

        def collect(group: str, key: str) -> Any:
            return record_array([frame[group].get(key) for frame in frames])

        obs_groups = sorted(
            {
                group_name
                for frame in frames
                for group_name in frame["policy"].get("obs", {}).keys()
            }
        )
        obs = {}
        for group_name in obs_groups:
            obs_names = sorted(
                {
                    obs_name
                    for frame in frames
                    for obs_name in frame["policy"].get("obs", {}).get(group_name, {}).keys()
                }
            )
            obs[group_name] = {
                obs_name: record_array(
                    [
                        frame["policy"].get("obs", {}).get(group_name, {}).get(obs_name)
                        for frame in frames
                    ]
                )
                for obs_name in obs_names
            }

        runtime_keys = sorted(
            {key for frame in frames for key in frame.get("runtime", {})}
        )

        return {
            "metadata": {
                "schema": "policy_tracking_record_v1",
                "robot": str(policy.args.robot),
                "policy_config": str(policy.args.policy_config),
                "model_path": str(policy.model_path),
                "rl_rate": float(policy.args.rl_rate),
                "joint_names_simulation": list(policy.joint_names_simulation),
                "body_names_simulation": list(policy.body_names_simulation),
                "frame_count": int(len(frames)),
                "recorded_at_unix_ns": int(time.time_ns()),
                "record_start_unix_ns": int(self._start_time_ns or 0),
                **policy._record_metadata_fields(),
            },
            "robot": {
                "joint_names": list(policy.state_processor.joint_names),
                "joint_pos": collect("robot", "joint_pos"),
                "joint_vel": collect("robot", "joint_vel"),
                "joint_torque": collect("robot", "joint_torque"),
                "root_quat_w": collect("robot", "root_quat_w"),
                "root_ang_vel_b": collect("robot", "root_ang_vel_b"),
                "low_state_tick": collect("robot", "low_state_tick"),
            },
            "policy": {
                "step_index": record_array([frame["step_index"] for frame in frames]),
                "time_ns": record_array([frame["time_ns"] for frame in frames]),
                "action": collect("policy", "action"),
                "q_target": collect("policy", "q_target"),
                "cmd_q": collect("policy", "cmd_q"),
                "cmd_dq": collect("policy", "cmd_dq"),
                "cmd_tau": collect("policy", "cmd_tau"),
                "obs": obs,
            },
            "runtime": {
                key: record_array([frame.get("runtime", {}).get(key) for frame in frames])
                for key in runtime_keys
            },
        }
