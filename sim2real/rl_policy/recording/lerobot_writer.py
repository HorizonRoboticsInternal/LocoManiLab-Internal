"""LeRobot v2.1 episode writer + the EpisodeRecorderBase adapter that wires it into
BasePolicy's control loop.

``LeRobotEpisodeExporter``/``LeRobotEpisodeMetadata`` are a thin, generic wrapper
around the public ``lerobot`` package -- structurally the same as the already-proven
``HeftDataExporter``/``HeftDatasetMetadata`` in LocoManiLab's original HeFT integration
(historical ``sim2real/vla/dataset_writer.py``, also reused by the AMO integration), reimplemented here with no family-specific assumptions: ``features``/
``modality_config`` are always supplied by the caller (``schema.py``).
"""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub.errors import RepositoryNotFoundError
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    compute_episode_stats,
)
from lerobot.common.datasets.utils import (
    check_timestamps_sync,
    get_episode_data_index,
    validate_episode_buffer,
    validate_frame,
)
from loguru import logger

from sim2real.rl_policy.recording.base import EpisodeRecorderBase
from sim2real.rl_policy.recording.extras.base import RecordingExtra
from sim2real.rl_policy.recording.schema import derive_features, derive_modality_config, flatten_frame
from sim2real.rl_policy.recording.video_writer import VideoWriter


@dataclass
class ScriptConfig:
    """Minimal config dataclass for script_config serialization."""

    def to_dict(self):
        return asdict(self)


class LeRobotEpisodeMetadata(LeRobotDatasetMetadata):
    """Adds ``modality_config`` (meta/modality.json) + discarded-episode tracking."""

    MODALITY_CONFIG_REL_PATH = Path("meta/modality.json")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(self.root / self.MODALITY_CONFIG_REL_PATH, "rb") as f:
            self.modality_config = json.load(f)

    @classmethod
    def create(cls, modality_config: dict, script_config: dict, *args, **kwargs):
        obj = super().create(*args, **kwargs)
        obj.info["script_config"] = script_config
        obj.info["discarded_episode_indices"] = []
        with open(obj.root / "meta" / "info.json", "w") as f:
            json.dump(obj.info, f, indent=4)

        obj.__class__ = cls
        with open(obj.root / cls.MODALITY_CONFIG_REL_PATH, "w") as f:
            json.dump(modality_config, f, indent=4)
        obj.modality_config = modality_config
        return obj


class LeRobotEpisodeExporter(LeRobotDataset):
    """Records one collection session to a LeRobot v2.1 dataset.

    Lifecycle: create() -> add_frame() * N -> save_episode() (repeat per episode).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_writers = self.create_video_writer()

    @property
    def repo_id(self):
        return self.meta.repo_id

    @property
    def root(self):
        return self.meta.root

    @property
    def video_keys(self):
        return self.meta.video_keys

    @classmethod
    def create(
        cls,
        save_root: str | Path,
        fps: int,
        features: dict,
        modality_config: dict,
        task: str,
        script_config: ScriptConfig | dict | None = None,
        robot_type: str | None = None,
        tolerance_s: float = 1e-4,
        vcodec: str = "h264",
        overwrite_existing: bool = False,
    ) -> "LeRobotEpisodeExporter":
        if script_config is None:
            script_config = {}

        obj = cls.__new__(cls)
        repo_id = "tmp/tmp_dataset"

        if overwrite_existing and Path(save_root).exists():
            logger.info(f"Found existing dataset at {save_root}; overwrite_existing=True, removing.")
            shutil.rmtree(save_root)

        if Path(save_root).exists():
            try:
                obj.meta = LeRobotEpisodeMetadata(repo_id=repo_id, root=save_root)
            except RepositoryNotFoundError as e:
                raise ValueError(
                    f"Failed to resume from corrupted dataset at {save_root}") from e
        else:
            if not isinstance(script_config, dict):
                script_config = script_config.to_dict()
            obj.meta = LeRobotEpisodeMetadata.create(
                repo_id=repo_id, fps=fps, root=save_root, robot=None, robot_type=robot_type,
                features=features, modality_config=modality_config,
                script_config=script_config, use_videos=True,
            )

        obj.tolerance_s = tolerance_s
        obj.video_backend = "pyav"
        obj.vcodec = vcodec
        obj.task = task
        obj.image_writer = None
        obj.episode_buffer = obj.create_episode_buffer()
        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_writers = obj.create_video_writer()
        return obj

    def create_video_writer(self) -> dict[str, VideoWriter]:
        return {
            key: VideoWriter(
                self.root / self.meta.get_video_file_path(self.episode_buffer["episode_index"], key),
                self.meta.shapes[key][1], self.meta.shapes[key][0], self.fps, self.vcodec,
            )
            for key in self.meta.video_keys
        }

    def add_frame(self, frame: dict) -> None:
        frame = copy.deepcopy(frame)
        frame["task"] = frame.get("task", self.task)
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)
        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key == "task":
                self.episode_buffer["task"].append(frame["task"])
                continue
            if key not in self.features:
                raise ValueError(f"'{key}' not in features: {self.features.keys()}")
            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"],
                    image_key=key, frame_index=frame_index)
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self.video_writers[key].add_frame(frame[key])
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def stop_video_writers(self):
        for key in self.video_writers:
            self.video_writers[key].stop()

    def skip_and_start_new_episode(self) -> None:
        self.stop_video_writers()
        self.episode_buffer = self.create_episode_buffer()
        self.video_writers = self.create_video_writer()

    def save_episode(self, episode_data: dict | None = None) -> None:
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(
            self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        for task in episode_tasks:
            if self.meta.get_task_index(task) is None:
                self.meta.add_task(task)
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(t) for t in tasks])

        for key, ft in self.features.items():
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)

        non_video_features = {k: v for k, v in self.features.items() if v["dtype"] != "video"}
        non_vid_ep_buffer = {k: v for k, v in episode_buffer.items() if k in non_video_features}
        ep_stats = compute_episode_stats(non_vid_ep_buffer, non_video_features)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"], episode_buffer["episode_index"],
            ep_data_index_np, self.fps, self.tolerance_s)

        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(img_dir)

        if not episode_data:
            self.episode_buffer = self.create_episode_buffer()
            self.video_writers = self.create_video_writer()

    def encode_episode_videos(self, episode_index: int) -> dict:
        return {key: self.video_writers[key].stop() for key in self.meta.video_keys}

    def save_episode_as_discarded(self) -> None:
        self.meta.info["discarded_episode_indices"] = self.meta.info.get(
            "discarded_episode_indices", []) + [self.episode_buffer["episode_index"]]
        self.save_episode()


class LeRobotRecorder(EpisodeRecorderBase):
    """EpisodeRecorderBase adapter: BasePolicy frame dicts -> LeRobotEpisodeExporter.

    Starts idle (unlike the npz recorder's legacy always-on mode) -- recording is
    inherently episodic for LeRobot, so the first toggle() opens episode 1. The
    underlying dataset (and its `features`/`modality_config`) is created lazily, on the
    first frame for which all opt-in signals (camera, hand-grip) are actually present,
    so image shape/keys can be auto-detected from the live stream instead of requiring
    a separate `--camera-shape`-style flag.
    """

    def __init__(
        self,
        policy: Any,
        save_root: Path,
        fps: float,
        task: str,
        camera_client: Any | None,
        hand_grip_client: Any | None,
        extras: list[RecordingExtra] | None = None,
    ) -> None:
        self._policy = policy
        self._save_root = Path(save_root)
        self._fps = fps
        self._task = task
        self._camera_client = camera_client
        self._hand_grip_client = hand_grip_client
        self._extras = extras or []

        self._exporter: LeRobotEpisodeExporter | None = None
        self._recording_active = False
        self._has_episode_open = False

    def _sample_camera(self) -> dict[str, np.ndarray] | None:
        if self._camera_client is None:
            return None
        frame = self._camera_client.read()
        if frame is None:
            return None
        return frame["images"]

    def _sample_hand_grip(self) -> tuple[float, float] | None:
        if self._hand_grip_client is None:
            return None
        return self._hand_grip_client.read()

    @staticmethod
    def _merge_no_collision(base: dict, extra: dict, *, extra_source: str) -> dict:
        collisions = set(base) & set(extra)
        if collisions:
            raise ValueError(
                f"RecordingExtra {extra_source!r} redeclared existing field(s) "
                f"{sorted(collisions)} -- rename the field in that extra."
            )
        return {**base, **extra}

    def append_frame(self, frame: dict[str, Any]) -> None:
        if not self._recording_active:
            return

        image = self._sample_camera()
        if self._camera_client is not None and image is None:
            return  # waiting for the first/next camera frame

        hand_grip = self._sample_hand_grip()
        if self._hand_grip_client is not None and hand_grip is None:
            return  # waiting for the first hand-grip message

        if self._exporter is None:
            image_shapes = (
                {name: arr.shape for name, arr in image.items()} if image is not None else {}
            )
            features = derive_features(
                self._policy,
                sample_frame=frame,
                image_shapes=image_shapes,
                include_hand_grip=hand_grip is not None,
            )
            for extra in self._extras:
                features = self._merge_no_collision(
                    features, extra.features(), extra_source=type(extra).__name__
                )
            self._exporter = LeRobotEpisodeExporter.create(
                save_root=self._save_root,
                fps=self._fps,
                features=features,
                modality_config=derive_modality_config(features),
                task=self._task,
                script_config={
                    "robot": self._policy.args.robot,
                    "controller": self._policy.controller_type,
                    "policy_config": str(self._policy.args.policy_config),
                },
                robot_type=f"sim2real_{self._policy.args.robot}",
            )
            logger.info("Created LeRobot dataset at {}", self._save_root)

        flat_frame = flatten_frame(frame, task=self._task, image=image, hand_grip=hand_grip)
        for extra in self._extras:
            flat_frame = self._merge_no_collision(
                flat_frame, extra.compute(frame), extra_source=type(extra).__name__
            )
        self._exporter.add_frame(flat_frame)
        self._has_episode_open = True

    def toggle(self, *, source: str) -> None:
        if self._recording_active:
            self._recording_active = False
            if self._has_episode_open:
                self._exporter.save_episode()
                self._has_episode_open = False
                logger.info("Episode saved via {}", source)
        else:
            self._recording_active = True
            logger.info("Recording started via {}", source)

    def discard(self, *, source: str) -> None:
        self._recording_active = False
        if self._has_episode_open:
            self._exporter.save_episode_as_discarded()
            self._has_episode_open = False
            logger.info("Episode discarded via {}", source)

    def close(self) -> None:
        if self._has_episode_open:
            self._exporter.save_episode()
            self._has_episode_open = False
        if self._camera_client is not None:
            self._camera_client.close()
        if self._hand_grip_client is not None:
            self._hand_grip_client.close()
        for extra in self._extras:
            extra.close()


def build_lerobot_recorder(policy: Any) -> LeRobotRecorder:
    """Factory used by BasePolicy._build_recorder() for --record-format lerobot."""
    camera_client = None
    if policy.args.camera_connect:
        from sim2real.camera.client import CameraClient

        camera_client = CameraClient(connect=policy.args.camera_connect)

    hand_grip_client = None
    if policy.args.hand_grip_connect:
        from sim2real.rl_policy.recording.hand_grip_client import HandGripClient

        hand_grip_client = HandGripClient(connect=policy.args.hand_grip_connect)

    task = policy.args.record_task or Path(policy.args.policy_config).stem

    # policy.yaml's own recording_extras declares this family's default extras;
    # --record-extra adds ad-hoc ones on top (e.g. for a quick experiment) without
    # having to edit the checkpoint config.
    extra_names = set(policy.policy_config.get("recording_extras", [])) | set(
        policy.args.record_extra
    )
    if getattr(policy, "dex3_driver", None) is not None:
        # In-process Dex3 hand control (--dex3-hand-dir) always records its own
        # joint targets/measurements, no separate --record-extra dex3 needed.
        extra_names.add("dex3")
    extras = [RecordingExtra.resolve(name)(policy) for name in sorted(extra_names)]

    return LeRobotRecorder(
        policy=policy,
        save_root=policy._record_output,
        fps=policy.args.rl_rate,
        task=task,
        camera_client=camera_client,
        hand_grip_client=hand_grip_client,
        extras=extras,
    )
