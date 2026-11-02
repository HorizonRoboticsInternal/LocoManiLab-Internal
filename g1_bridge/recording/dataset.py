"""Creating the LeRobot dataset a G1 teleop run records into -- host-agnostic.

Split out of ``robocasa.utils.g1_data_collection`` so the real-robot collector
creates its datasets through exactly the same code path as the sim one. What the
host still supplies: which extra features it can fill (``extra_features``), the
ego frame size, and -- in legacy/sonic schema mode only -- the static
``modality.json`` asset to copy.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import shutil

from termcolor import colored

from g1_bridge.recording.features import EGO_H, EGO_W, FPS, build_features, schema_version_for
from g1_bridge.recording.schema import modality_json_dict


def timestamped_out_dir(out_base, name: str, backend: str | None = None, suffix: str = ""):
    """``<out_base>[/<backend>]/<name><suffix>_<timestamp>``.

    The controller backend gets its own level above the task so runs of the same
    task under different controllers (sonic/amo/heft/scalebfm) stay separated.
    The dataset itself lives directly at this path -- there is no extra
    ``lerobot/`` level (see :func:`make_dataset`).
    """
    stamp = f"{datetime.datetime.now():%Y-%m-%d-%H-%M-%S}"
    base = pathlib.Path(out_base)
    if backend:
        base = base / backend
    return base / f"{name}{suffix}_{stamp}"


def make_dataset(
    out_root,
    teleop_keys,
    fps: int = FPS,
    collection_info: dict | None = None,
    *,
    extra_features: dict | None = None,
    ego_hw: tuple[int, int] = (EGO_H, EGO_W),
    robot_type: str = "G1FloatingBody",
    modality_asset=None,
):
    """Create a fresh LeRobot dataset AT ``out_root`` + drop in modality.json.

    ``collection_info`` is recorded verbatim under ``meta/info.json`` as
    ``collection_info``. It exists for the settings that change what the *values*
    in an episode mean while leaving the schema untouched -- the arm source is the
    motivating one: ``amo_arm_source="gmr"`` records the same
    ``left_arm_action``/``right_arm_action`` columns as ``"ik"`` does, but the two
    produce systematically different arm behaviour for the same task (measured:
    shoulder_yaw travels 0.31 rad under the IK and 0.97 under GMR, because that
    joint is exactly the DoF 3-point input cannot determine). Training on a mix of
    both is training on two different experts for one observation, and without
    this there is nothing in a finished dataset that says which one you have. The
    real-robot collector adds its own keys here (robot_io, camera, ...) for the
    same reason.

    ``out_root`` is the LeRobot dataset root itself (it must not exist yet --
    LeRobot's own ``root.mkdir(exist_ok=False)`` enforces that too); sidecar files
    written by callers land next to ``meta/`` and ``data/``.
    """
    from g1_bridge.recording.lerobot_dataset import LerobotDatasetWrapper

    lerobot_root = pathlib.Path(out_root)
    if lerobot_root.exists():
        raise FileExistsError(f"{lerobot_root} already exists — pick a new --out")

    features, blobs = build_features(
        teleop_keys, ego_hw=ego_hw, fps=fps, extra_features=extra_features
    )
    dataset = LerobotDatasetWrapper.create(
        repo_id=lerobot_root,
        robot_type=robot_type,
        fps=fps,
        features=features,
        image_writer_threads=8,
        image_writer_processes=0,
    )
    if blobs is not None:
        # No sort_keys: modality_json_dict() already inserts each blob's groups
        # in ascending start-offset order (see ModalityBlob.groups) -- matches the
        # reference sonic modality.json's own convention
        # (robocasa/models/assets/groot_dataset_assets/G1_modality.json).
        # sort_keys=True would re-sort alphabetically and destroy that.
        (dataset.root / "meta" / "modality.json").write_text(
            json.dumps(modality_json_dict(blobs), indent=2) + "\n"
        )
    elif modality_asset is not None:
        shutil.copy(modality_asset, dataset.root / "meta" / "modality.json")
    else:
        raise ValueError(
            "legacy (plain-tuple) schema mode needs modality_asset: the static "
            "G1_modality.json that describes those columns"
        )
    # Stamped here rather than by each collector: this is the one function both
    # hosts create a dataset through, so neither can forget it, and a dataset
    # with no `schema_version` is unambiguously one collected before the field
    # existed (see features.SCHEMA_VERSION for what each value means).
    # Per backend (see features.BACKEND_SCHEMA_VERSION): the last version whose
    # change touched this backend's columns, not the global head.
    backend = (collection_info or {}).get("backend")
    collection_info = {**(collection_info or {}), "schema_version": schema_version_for(backend)}
    if collection_info:
        # Into the live meta.info dict, NOT straight into info.json. LeRobot
        # rewrites that whole file from meta.info later (lerobot_utils'
        # save_episode -> write_info, when the first episode's video is encoded),
        # so a direct file write is silently clobbered -- measured. Setting it
        # here means every later rewrite carries it along. Passing it to create()
        # is not an option either: LerobotDatasetWrapper has no script_config
        # parameter, unlike sim2real's own LeRobotEpisodeExporter.
        from lerobot.datasets.utils import write_info

        dataset.meta.info["collection_info"] = collection_info
        write_info(dataset.meta.info, dataset.meta.root)
        print(
            colored(f"[collect] collection_info: {collection_info}", "green"),
            flush=True,
        )
    print(colored(f"[collect] LeRobot dataset at {dataset.root}", "green"), flush=True)
    return dataset


def purge_image_cache(dataset):
    """Delete LeRobot's temporary PNG staging tree (``<root>/images/``).

    Only the encoded mp4s are wanted on disk. LeRobot writes every recorded frame
    as a PNG under ``<root>/images/<key>/episode_<i>/`` and removes that episode's
    dir when it encodes the video (save) or drops the buffer (discard) -- but an
    episode that is neither saved nor discarded, i.e. one interrupted by Ctrl-C
    mid-recording, leaves its frames behind for good. That is how a finished
    dataset ends up with a stray ``images/`` tree next to its ``videos/``.
    Sweeping the whole directory is safe: by the time anything calls this, every
    kept episode is already encoded.

    Call ``dataset.stop_image_writer()`` first, or in-flight writer jobs can
    recreate files right after the sweep.
    """
    img_root = pathlib.Path(dataset.root) / "images"
    if not img_root.exists():
        return
    leftover = sum(1 for _ in img_root.rglob("*.png"))
    shutil.rmtree(img_root, ignore_errors=True)
    if leftover:
        print(
            colored(
                f"[collect] cleaned up {leftover} unencoded frame(s) from an "
                "interrupted episode (only mp4 video is kept).",
                "yellow",
            )
        )
