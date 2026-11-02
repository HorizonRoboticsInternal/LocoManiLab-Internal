from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim2real.rl_policy.recording.npz_writer import NpzEpisodeRecorder


class _FakePolicy:
    """Just enough of BasePolicy's surface for NpzEpisodeRecorder's metadata build."""

    def __init__(self) -> None:
        self.args = SimpleNamespace(robot="g1", policy_config="checkpoints/demo/policy.yaml", rl_rate=50.0)
        self.model_path = "checkpoints/demo/policy.onnx"
        self.joint_names_simulation = ["joint_a", "joint_b"]
        self.body_names_simulation = ["body_a"]
        self.state_processor = SimpleNamespace(joint_names=["joint_a", "joint_b"])

    def _record_metadata_fields(self) -> dict:
        return {}


def _make_frame(step_index: int, time_ns: int) -> dict:
    return {
        "step_index": step_index,
        "time_ns": time_ns,
        "robot": {
            "joint_pos": np.array([0.1, 0.2], dtype=np.float32),
            "joint_vel": np.array([0.0, 0.0], dtype=np.float32),
            "joint_torque": np.array([0.0, 0.0], dtype=np.float32),
            "root_quat_w": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "root_ang_vel_b": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "low_state_tick": step_index,
        },
        "policy": {
            "action": np.array([0.0], dtype=np.float32),
            "q_target": np.array([0.1, 0.2], dtype=np.float32),
            "cmd_q": np.array([0.1, 0.2], dtype=np.float32),
            "cmd_dq": np.array([0.0, 0.0], dtype=np.float32),
            "cmd_tau": np.array([0.0, 0.0], dtype=np.float32),
            "obs": {"policy": {"obs_a": np.array([1.0, 2.0], dtype=np.float32)}},
        },
        "runtime": {"control_mode": "policy", "paused": False},
    }


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as npz:
        return npz["data"].item()


def test_default_legacy_mode_records_from_the_start_with_no_toggle(tmp_path: Path) -> None:
    output = tmp_path / "record.npz"
    recorder = NpzEpisodeRecorder(policy=_FakePolicy(), output_path=output, episodes_enabled=False)

    for i in range(3):
        recorder.append_frame(_make_frame(i, time_ns=1_000 + i))
    recorder.close()

    assert output.exists()
    data = _load(output)
    assert data["metadata"]["schema"] == "policy_tracking_record_v1"
    assert data["metadata"]["frame_count"] == 3
    assert data["policy"]["step_index"].tolist() == [0, 1, 2]
    assert data["robot"]["joint_pos"].shape == (3, 2)


def test_legacy_mode_ignores_toggle_and_discard(tmp_path: Path) -> None:
    output = tmp_path / "record.npz"
    recorder = NpzEpisodeRecorder(policy=_FakePolicy(), output_path=output, episodes_enabled=False)

    recorder.append_frame(_make_frame(0, time_ns=1_000))
    recorder.toggle(source="test")  # no-op: legacy mode has no episode segmentation
    recorder.append_frame(_make_frame(1, time_ns=1_001))
    recorder.discard(source="test")  # no-op: legacy mode never discards mid-run
    recorder.append_frame(_make_frame(2, time_ns=1_002))
    recorder.close()

    data = _load(output)
    assert data["metadata"]["frame_count"] == 3
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_legacy_mode_saves_nothing_when_no_frames_were_recorded(tmp_path: Path) -> None:
    output = tmp_path / "record.npz"
    recorder = NpzEpisodeRecorder(policy=_FakePolicy(), output_path=output, episodes_enabled=False)

    recorder.close()

    assert not output.exists()


def test_episodes_enabled_buffers_nothing_until_first_toggle(tmp_path: Path) -> None:
    output = tmp_path / "record.npz"
    recorder = NpzEpisodeRecorder(policy=_FakePolicy(), output_path=output, episodes_enabled=True)

    recorder.append_frame(_make_frame(0, time_ns=1_000))
    recorder.close()

    assert list(tmp_path.glob("*.npz")) == []


def test_episodes_enabled_toggle_saves_one_file_per_episode(tmp_path: Path) -> None:
    output = tmp_path / "record.npz"
    recorder = NpzEpisodeRecorder(policy=_FakePolicy(), output_path=output, episodes_enabled=True)

    recorder.toggle(source="keyboard:c")
    recorder.append_frame(_make_frame(0, time_ns=1_000))
    recorder.append_frame(_make_frame(1, time_ns=1_001))
    recorder.toggle(source="keyboard:c")

    saved = sorted(tmp_path.glob("*.npz"))
    assert len(saved) == 1
    data = _load(saved[0])
    assert data["metadata"]["frame_count"] == 2


def test_episodes_enabled_discard_drops_buffer_without_saving(tmp_path: Path) -> None:
    output = tmp_path / "record.npz"
    recorder = NpzEpisodeRecorder(policy=_FakePolicy(), output_path=output, episodes_enabled=True)

    recorder.toggle(source="keyboard:c")
    recorder.append_frame(_make_frame(0, time_ns=1_000))
    recorder.discard(source="keyboard:x")
    recorder.close()

    assert list(tmp_path.glob("*.npz")) == []


def test_episodes_enabled_writes_distinct_files_across_two_episodes(tmp_path: Path) -> None:
    output = tmp_path / "record.npz"
    recorder = NpzEpisodeRecorder(policy=_FakePolicy(), output_path=output, episodes_enabled=True)

    recorder.toggle(source="c")
    recorder.append_frame(_make_frame(0, time_ns=1_000))
    recorder.toggle(source="c")  # save episode 1

    recorder.toggle(source="c")
    recorder.append_frame(_make_frame(0, time_ns=2_000))
    recorder.append_frame(_make_frame(1, time_ns=2_001))
    recorder.toggle(source="c")  # save episode 2

    saved = sorted(tmp_path.glob("*.npz"))
    assert len(saved) == 2
    assert _load(saved[0])["metadata"]["frame_count"] == 1
    assert _load(saved[1])["metadata"]["frame_count"] == 2
