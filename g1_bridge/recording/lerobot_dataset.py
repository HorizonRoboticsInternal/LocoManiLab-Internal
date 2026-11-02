"""``LerobotDatasetWrapper`` -- LeRobotDataset with the video-encoding controls
this project needs, and with the image-writer races closed.

Moved here from ``robocasa.utils.lerobot_utils`` so a real-robot collector can
write the same datasets as the sim one without importing the benchmark (that
module also carries robomimic/HDF5 conversion helpers, which pull in a much
heavier dependency set). ``robocasa.utils.lerobot_utils`` re-exports this class.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import write_info
from lerobot.datasets.video_utils import encode_video_frames


class LerobotDatasetWrapper(LeRobotDataset):
    """
    Wrapper class for creating LeRobotDataset. Class is needed so that we can override
    methods to get more control over video encoding parameters.
    """

    def _episode_image_paths(self, episode_buffer):
        paths = []
        for key, ft in self.features.items():
            if ft["dtype"] in ["image", "video"] and key in episode_buffer:
                paths.extend(Path(p) for p in episode_buffer[key])
        return paths

    def _wait_for_episode_images(self, episode_buffer, timeout_s=5.0):
        """Wait until every image path referenced by the buffer exists on disk."""
        self._wait_image_writer()
        paths = self._episode_image_paths(episode_buffer)
        deadline = time.monotonic() + timeout_s
        missing = [p for p in paths if not p.is_file()]
        while missing and time.monotonic() < deadline:
            time.sleep(0.05)
            missing = [p for p in paths if not p.is_file()]
        if missing:
            preview = ", ".join(str(p) for p in missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} image file(s) referenced by the episode buffer "
                f"were not written before save_episode; first missing: {preview}"
            )

    def save_episode(self, episode_data: dict | None = None) -> None:
        episode_buffer = self.episode_buffer if episode_data is None else episode_data
        self._wait_for_episode_images(episode_buffer)
        return super().save_episode(episode_data)

    def clear_episode_buffer(self) -> None:
        # Do not remove an episode image directory while async writer jobs for it
        # are still running.
        self._wait_image_writer()
        return super().clear_episode_buffer()

    def encode_episode_videos(self, episode_index: int) -> None:
        """
        Encode videos for a given episode index. Code is mostly copied from parent class
        but with modified video new encoding parameters.
        """
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            if video_path.is_file():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(
                img_dir,
                video_path,
                self.fps,
                overwrite=True,
                vcodec="h264",
                pix_fmt="yuv420p",
                crf=23,
                g=None,
                fast_decode=0,
            )
            shutil.rmtree(img_dir)

        # Update video info (only needed when first episode is encoded since it reads from episode 0)
        if len(self.meta.video_keys) > 0 and episode_index == 0:
            self.meta.update_video_info()
            write_info(
                self.meta.info, self.meta.root
            )  # ensure video info always written properly
