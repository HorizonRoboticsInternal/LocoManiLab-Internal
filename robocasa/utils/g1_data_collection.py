"""Scene-type-agnostic G1 teleop data-collection core.

This holds the parts of demo collection that do NOT depend on a particular scene
family (kitchen / living-room / MolmoSpaces iTHOR rooms / future types): the
LeRobot dataset schema + frame builder, the ``EpisodeRecorder`` hook, the env
builder, and the ``collect_demos_main`` entry that wires reset -> teleop loop ->
recording. ``collect_demos_g1.py`` provides the generic CLI; fixed-task collection
reuses these helpers with its own reset and recording policy.

Scene selection is generic: ``forward_scene_kwargs`` inspects the target env's
``__init__`` signature and forwards only the scene kwargs it actually accepts
(``--scene-id`` -> ``scene_ids`` for MolmoSpaces envs, ``--layout/--style`` ->
``layout_ids/style_ids`` for kitchen/living-room, ``--fixed-scene`` ->
``fixed_scene_seed`` universally). Adding a new scene family needs no change here.
"""

import datetime
import inspect
import json
import pathlib
import time
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple, Optional

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as sRot
from termcolor import colored

import robosuite
from robosuite.environments.base import REGISTERED_ENVS
from robosuite.controllers import load_composite_controller_config
from g1_bridge.pico.mode_manager import ManagerMode
from g1_bridge.sonic_utils.ckpt import (
    DEFAULT_SONIC_CKPT,
    SONIC_CKPT_CHOICES,
    apply_sonic_ckpt,
)

import robocasa  # noqa: F401  -- registers all envs

# Re-exported, never redefined: build_frame's isinstance(ModalityLayout) check
# only works if there is exactly one such class object in the process.
from g1_bridge.recording.schema import (  # noqa: F401
    ModalityBlob,
    ModalityGroup,
    ModalityLayout,
    blob_features,
    build_blob_frame_fields,
    layout_sources,
    modality_json_dict,
)

# The recording layer itself is host-agnostic and lives in g1_bridge, so the
# real-robot collector writes byte-compatible datasets through the same code.
# Only the sim-specific wrappers (obs -> frame, success -> episode) are here.
from g1_bridge.recording.dataset import (  # noqa: F401
    make_dataset as _make_dataset,
    purge_image_cache,
    timestamped_out_dir,
)
from g1_bridge.recording.episode import EpisodeRecorder as _EpisodeRecorder
from g1_bridge.recording.features import (  # noqa: F401
    LEGACY_MODALITY_ASSET,
    EGO_H,
    EGO_W,
    FPS,
    SIM_ONLY_FEATURES,
    TELEOP_FEATURE_SPECS,
    VIDEO_INFO,
)
from g1_bridge.recording.cli import add_shared_collection_args
from g1_bridge.recording.frames import (
    build_blob_frame,
    build_legacy_frame,
    sonic_state_43,
)

# Default output base: <repo root>/outputs_sim (gitignored). robocasa.__file__ is
# <repo>/robocasa/__init__.py, so parents[1] is the repo root. Note an ABSOLUTE
# --out wins LeRobot's HF_LEROBOT_HOME join, a relative one silently lands in
# ~/.cache/huggingface/lerobot -- so this default is resolved to absolute here.
REPO_ROOT = pathlib.Path(robocasa.__file__).resolve().parents[1]
#: Simulated demos only -- the real-robot collector (g1_bridge/collect_real.py)
#: files its own under ``outputs_real/``. Two trees rather than one because
#: nothing downstream re-derives which one a dataset came from: the schema, the
#: backend and the env id are all identical between a sim demo and a hardware
#: demo of the same task, so the path is the only place that distinction survives
#: an rsync or a glob into ``lab prepare``.
DEFAULT_OUT_BASE = REPO_ROOT / "outputs_sim"

EGO_CAMERA = "robot0_head_camera"
DEPTH_FEATURE = "observation.images.ego_depth_view"
DEPTH_PNG_PATH_PATTERN = (
    "depth/observation.images.ego_depth_view/"
    "episode_{episode_index:06d}/frame_{frame_index:06d}.png"
)
DEPTH_NPY_PATH_PATTERN = (
    "depth/observation.images.ego_depth_view/"
    "episode_{episode_index:06d}/frame_{frame_index:06d}.npy"
)
DEPTH_VIS_NEAR_MM = 250
DEPTH_VIS_FAR_MM = 5000
DEPTH_INVALID_MM = 0
DEPTH_MAX_MM = np.iinfo(np.uint16).max

# A compact spectral palette: far surfaces are blue, near surfaces are red.
# Keeping the palette local avoids adding matplotlib / OpenCV to collection.
_DEPTH_VIS_STOPS = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float32)
_DEPTH_VIS_COLORS = np.array(
    [
        [49, 54, 149],
        [69, 117, 180],
        [116, 173, 209],
        [255, 255, 191],
        [253, 174, 97],
        [165, 0, 38],
    ],
    dtype=np.float32,
)
_DEPTH_VIS_LEVELS = np.linspace(0.0, 1.0, 256, dtype=np.float32)
_DEPTH_VIS_LUT = (
    np.stack(
        [
            np.interp(
                _DEPTH_VIS_LEVELS, _DEPTH_VIS_STOPS, _DEPTH_VIS_COLORS[:, channel]
            )
            for channel in range(3)
        ],
        axis=-1,
    )
    .round()
    .astype(np.uint8)
)
_KEY_RESET = ord("Q")

# The LeRobot schema itself (FPS, VIDEO_INFO, the ego feature, the legacy
# TELEOP_FEATURE_SPECS table and the sim-only columns) is imported above from
# g1_bridge.recording.features -- one copy, shared with the real-robot collector.
# EGO_H/EGO_W are the robosuite render size, which is where the sim's ego frames
# come from; the real collector downscales its camera to the same shape.

# The per---g1-control teleop-key table and the WHOLE_BODY_G1 config loader
# that used to live here are gone: every collection entrypoint now goes through
# G1BridgeComposite, whose per-backend schemas live in
# g1_bridge/modality.py (G1_BACKEND_TELEOP_KEYS). The demo-only
# copies remain in robocasa/demos/demo_empty_teleop.py, which is the sole
# consumer of the WHOLE_BODY_G1 configs.


def _depth_millimeters(obs, sim):
    """Return vertically-correct ego depth as a uint16 millimetre array.

    Zero is reserved for invalid pixels. Finite positive depths are rounded to
    the nearest millimetre and saturated at the uint16 limit (65.535 m). The
    depth image must have the same height and width as its RGB frame.
    """
    from robosuite.utils.camera_utils import get_real_depth_map

    depth = np.asarray(obs[f"{EGO_CAMERA}_depth"])[::-1]
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = np.squeeze(depth, axis=-1)
    if depth.ndim != 2:
        raise ValueError(f"expected 2-D ego depth, got shape {depth.shape}")

    rgb_shape = np.asarray(obs[f"{EGO_CAMERA}_image"]).shape[:2]
    if depth.shape != rgb_shape:
        raise ValueError(
            f"ego depth resolution {depth.shape} does not match RGB {rgb_shape}"
        )

    depth_m = np.asarray(get_real_depth_map(sim, depth), dtype=np.float64)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    depth_mm = np.full(depth_m.shape, DEPTH_INVALID_MM, dtype=np.uint16)
    if np.any(valid):
        millimeters = np.rint(depth_m[valid] * 1000.0)
        depth_mm[valid] = np.clip(millimeters, 1, DEPTH_MAX_MM).astype(np.uint16)
    return np.ascontiguousarray(depth_mm)


def _depth_rgb_image(obs, sim):
    """Return metric ego depth as a human-readable pseudocolor RGB image."""
    return _depth_rgb_image_from_millimeters(_depth_millimeters(obs, sim))


def _depth_rgb_image_from_millimeters(depth_mm):
    """Map 250--5000 mm to near-red / far-blue RGB; zero stays black."""
    depth_mm = np.asarray(depth_mm, dtype=np.uint16)
    valid = depth_mm != DEPTH_INVALID_MM
    safe_depth = np.where(valid, depth_mm, DEPTH_VIS_FAR_MM).astype(np.float32)
    proximity = (DEPTH_VIS_FAR_MM - safe_depth) / (DEPTH_VIS_FAR_MM - DEPTH_VIS_NEAR_MM)
    indices = np.rint(np.clip(proximity, 0.0, 1.0) * 255.0).astype(np.uint8)
    rgb = _DEPTH_VIS_LUT[indices].copy()
    rgb[~valid] = 0
    return Image.fromarray(rgb, mode="RGB")


class DepthSequenceWriter:
    """Write pseudocolor RGB PNG and raw uint16-mm NPY for accepted depth frames.

    Frames first land in a private staging directory. Only a successfully saved
    LeRobot episode moves them into ``depth/.../episode_NNNNNN``; discarded or
    interrupted episodes remove their staging directory instead. This keeps the
    sidecar transactional with the RGB video and parquet episode.
    """

    def __init__(self, dataset_root, workers: int = 4):
        self.root = pathlib.Path(dataset_root)
        self.staging_root = self.root / ".depth_staging"
        self.pool = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="depth-png"
        )
        self.futures = {}

    def _staging_path(
        self, episode_index: int, frame_index: int, suffix: str
    ) -> pathlib.Path:
        return (
            self.staging_root
            / f"episode_{episode_index:06d}"
            / f"frame_{frame_index:06d}{suffix}"
        )

    def add(self, depth_mm, episode_index: int, frame_index: int) -> None:
        depth_mm = np.asarray(depth_mm)
        if depth_mm.dtype != np.uint16:
            raise TypeError(f"raw ego depth must be uint16, got {depth_mm.dtype}")
        if depth_mm.shape != (EGO_H, EGO_W):
            raise ValueError(
                "ego depth resolution must match RGB video "
                f"{(EGO_H, EGO_W)}, got {depth_mm.shape}"
            )
        depth_mm = np.ascontiguousarray(depth_mm)
        png_path = self._staging_path(episode_index, frame_index, ".png")
        npy_path = self._staging_path(episode_index, frame_index, ".npy")
        png_path.parent.mkdir(parents=True, exist_ok=True)
        image = _depth_rgb_image_from_millimeters(depth_mm)
        futures = self.futures.setdefault(episode_index, [])
        futures.append(self.pool.submit(image.save, png_path, format="PNG"))
        futures.append(
            self.pool.submit(np.save, npy_path, depth_mm, allow_pickle=False)
        )

    def flush(self, episode_index: int) -> None:
        for future in self.futures.get(episode_index, ()):
            future.result()

    def commit(self, episode_index: int) -> None:
        self.flush(episode_index)
        staging = self.staging_root / f"episode_{episode_index:06d}"
        final = self.root / "depth" / DEPTH_FEATURE / f"episode_{episode_index:06d}"
        if final.exists():
            raise FileExistsError(f"depth episode already exists: {final}")
        final.parent.mkdir(parents=True, exist_ok=True)
        staging.rename(final)
        self.futures.pop(episode_index, None)
        try:
            self.staging_root.rmdir()
        except OSError:
            pass

    def discard(self, episode_index: int) -> None:
        self.flush(episode_index)
        shutil.rmtree(
            self.staging_root / f"episode_{episode_index:06d}", ignore_errors=True
        )
        self.futures.pop(episode_index, None)
        try:
            self.staging_root.rmdir()
        except OSError:
            pass

    def close(self) -> None:
        for episode_index in list(self.futures):
            self.discard(episode_index)
        self.pool.shutdown(wait=True)


def _sim_only_fields(obs, reward, done):
    """The columns only a simulator can fill (see recording.features.SIM_ONLY_FEATURES).

    ``observation.eef_state`` here is robosuite's value: WORLD frame, quaternion
    xyzw, sampled by a robosuite observable (0-1 control tick before the state
    the recorder reads straight from ``sim.data``). The blob backends record it
    as-is. The SONIC legacy schema (schema_version >= 6) OVERRIDES it inside
    ``g1_bridge.recording.frames.build_legacy_frame`` with FK of the state --
    pelvis frame, wxyz, same tick -- so a sonic sim dataset and a sonic hardware
    dataset carry one definition (NVlabs' upstream one).
    """
    eef = np.concatenate(
        [
            obs["robot0_left_eef_pos"],
            obs["robot0_left_eef_quat"],
            obs["robot0_right_eef_pos"],
            obs["robot0_right_eef_quat"],
        ]
    ).astype(np.float64)
    return {
        "observation.eef_state": eef,
        "next.reward": np.array([reward], dtype=np.float32),
        "next.done": np.array([done], dtype=bool),
    }


def _base_quat_wxyz(obs):
    """``robot0_base_quat`` in the wxyz convention the shared builder expects.

    robosuite's sensor is ``T.mat2quat(...)``, which returns **xyzw**; the real
    host feeds ``read_pelvis_state()[0]``, which is **wxyz**, and that is what
    ``g1_bridge.recording.frames.projected_gravity`` documents and assumes. The
    sim used to hand its xyzw straight over, so the recorded
    ``observation.root_orientation``/``observation.projected_gravity`` columns
    were a reordered quaternion -- a plausible-looking unit vector that is not
    the robot's attitude. Measured on a standing G1: the column read 30.8 deg of
    tilt where the pelvis was at 0.0.

    Sim-only, and every backend was affected, so datasets collected in sim
    before this fix have both columns wrong (the two ``observation.state`` and
    every action column are unaffected -- they never went through here).
    """
    from robosuite.utils.transform_utils import convert_quat

    return convert_quat(
        np.asarray(obs["robot0_base_quat"], dtype=np.float64), to="wxyz"
    )


def build_frame(
    obs, teleop, reward, done, task_idx, teleop_keys, composite, require_ready=True
):
    """Assemble one LeRobot frame dict from an env obs + composite teleop sample.

    The sim adapter over ``g1_bridge.recording.frames``: it pulls the ego image,
    the base quaternion and the sim-only columns out of a robosuite ``obs``, and
    the shared builder does the rest. A real-robot host calls the same builder
    with a camera frame and an IMU quaternion instead -- which is what makes the
    two datasets the same schema.

    ``composite`` is the G1BridgeComposite, i.e. this host's implementation of
    the ``Sim2RealContext`` read surface. The legacy/sonic schema's 43-D
    ``observation.state`` is built from it (``sonic_state_43``) rather than from
    ``obs["robot0_joint_pos"]``: that observable is in MJCF body-tree order,
    whose per-hand joint order (thumb, **middle**, index) is not the Dex3
    actuator order (thumb, **index**, middle) that ``read_hand_state()`` and
    every other hand column in the dataset use. See ``sonic_state_43``.

    ``teleop_keys`` (see ``controller_teleop_keys``) is either a plain tuple
    (legacy mode, TELEOP_FEATURE_SPECS) or a ``ModalityLayout`` (blob mode) --
    fixes which teleop/action columns this run's controller populates, must
    match what ``teleop`` (a ``composite.get_teleop_data_sample()`` snapshot)
    can actually contain. Returns None while the controller's readiness
    sentinel is still None (before its first real command), so the caller can
    skip the frame -- unless ``require_ready=False`` (plumbing-only checks,
    see collect_demos_main's ``--debug-skip-readiness``), in which case a
    still-missing readiness signal is just zero-filled like any other absent
    key instead of skipping the frame: this proves the controller constructs
    and steps without crashing even when no real reference (GMR payload,
    live PICO data, ...) has arrived, e.g. because the GMR subprocess can't
    reach a real PICO service in this environment -- a separate concern from
    whether the recording plumbing itself works.
    """
    img = obs[f"{EGO_CAMERA}_image"][::-1]  # robosuite frames are vertically flipped
    extra = _sim_only_fields(obs, reward, done)
    root_quat = _base_quat_wxyz(obs)
    if isinstance(teleop_keys, ModalityLayout):
        return build_blob_frame(
            teleop_keys,
            teleop,
            image=img,
            root_quat=root_quat,
            extra_fields=extra,
            require_ready=require_ready,
        )
    return build_legacy_frame(
        teleop_keys,
        teleop,
        image=img,
        state=sonic_state_43(composite),
        root_quat=root_quat,
        task_idx=task_idx,
        extra_fields=extra,
        require_ready=require_ready,
    )


def default_out_dir(env_name, backend, suffix=""):
    """``<repo>/outputs_sim/<backend>/<env><suffix>_<timestamp>``."""
    return timestamped_out_dir(DEFAULT_OUT_BASE, env_name, backend, suffix)


def build_collection_info(args, controller_config):
    """What produced this dataset, for ``meta/info.json``.

    Only settings a reader cannot recover from the recorded columns. Read off
    the assembled controller config rather than the module defaults, so an
    override in the controller JSON is reported as what actually ran.
    """
    from g1_bridge.backends.amo import _AMO_VR_DEFAULTS

    spec = controller_config.get("composite_controller_specific_configs", {})
    info = {
        "backend": args.backend,
        "pico_source": args.pico_source,
        "policy_config": spec.get("sim2real_policy_config"),
        "teleop_duration": {
            "files": ["meta/episodes.jsonl", "meta/episodes_stats.jsonl"],
            "key": "teleop_duration_s",
            "unit": "seconds",
            "definition": "wall-clock time from the first engaged/recorded frame to task success, one value per episode",
            "dtype": "float64",
            "shape": [1],
        },
        "depth": {
            "key": DEPTH_FEATURE,
            "png_encoding": "rgb8_pseudocolor",
            "png_colormap": "spectral_far_blue_near_red",
            "png_near_mm": DEPTH_VIS_NEAR_MM,
            "png_far_mm": DEPTH_VIS_FAR_MM,
            "npy_dtype": "uint16",
            "npy_unit": "millimeter",
            "npy_invalid_value": DEPTH_INVALID_MM,
            "npy_saturation_value": int(DEPTH_MAX_MM),
            "shape": [EGO_H, EGO_W],
            "matches_rgb_feature": "observation.images.ego_view",
            "png_path": DEPTH_PNG_PATH_PATTERN,
            "npy_path": DEPTH_NPY_PATH_PATTERN,
        },
    }
    if args.backend == "amo":
        source = spec.get("amo_arm_source", _AMO_VR_DEFAULTS["amo_arm_source"])
        info["amo_arm_source"] = source
        if source == "gmr":
            info["amo_gmr_human_height"] = spec.get(
                "amo_gmr_human_height", _AMO_VR_DEFAULTS["amo_gmr_human_height"]
            )
    return info


#: modality.json for the legacy plain-tuple (SONIC) schema mode -- a static asset
#: rather than something generated from blob offsets. Imported from the recording
#: layer above, so the sim and the real collector copy the SAME file: two copies
#: of a schema is two schemas, eventually.


def make_dataset(out_root, teleop_keys, fps=FPS, collection_info=None):
    """Create the run's LeRobot dataset -- the sim adapter over
    ``g1_bridge.recording.dataset.make_dataset``.

    What this host adds on top of the shared schema: independent pseudocolor PNG
    previews plus lossless uint16 millimetre NPY depth sidecars under
    ``observation.images.ego_depth_view``, the other sim-only columns
    (``observation.eef_state``/``next.reward``/``next.done``), the robosuite
    render size, and the legacy modality.json asset. See that function for the
    ``collection_info`` rationale and the "out_root must not exist" rule.
    """
    return _make_dataset(
        out_root,
        teleop_keys,
        fps=fps,
        collection_info=collection_info,
        extra_features=SIM_ONLY_FEATURES,
        ego_hw=(EGO_H, EGO_W),
        robot_type="G1FloatingBody",
        modality_asset=LEGACY_MODALITY_ASSET,
    )


class EpisodeRecorder(_EpisodeRecorder):
    """Data-collection hook for ``run_sonic_viewer_loop``: records every engaged
    step into a LeRobot episode, saves on task success or S, and discards on Q.

    The sim's episode policy. Hardware has no success signal and uses
    ``g1_bridge.recording.episode.OperatorEpisodeRecorder`` instead; both share
    the frame/dataset bookkeeping in the base class.
    """

    def __init__(self, dataset, *args, depth_writer=None, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self._teleop_started_at = None
        self.depth_writer = depth_writer or DepthSequenceWriter(dataset.root)

    def reset_episode(self):
        self._teleop_started_at = None
        return super().reset_episode()

    def add_frame(self, frame, task=None, depth_mm=None):
        if frame is not None and depth_mm is None:
            raise ValueError("accepted simulator frame has no matching metric depth")
        added = super().add_frame(frame, task=task)
        if added:
            self.depth_writer.add(
                depth_mm,
                episode_index=int(self.dataset.meta.total_episodes),
                frame_index=self.ep_frames - 1,
            )
        return added

    def save(self, reason="SUCCESS"):
        if self.episode_saved or self.ep_frames == 0:
            return False
        episode_index = int(self.dataset.meta.total_episodes)
        self.depth_writer.flush(episode_index)
        try:
            saved = super().save(reason=reason)
        except KeyboardInterrupt:
            if self.episode_saved:
                self.depth_writer.commit(episode_index)
            raise
        if saved:
            elapsed = 0.0 if self._teleop_started_at is None else max(0.0, time.monotonic() - self._teleop_started_at)
            meta = pathlib.Path(self.dataset.root) / "meta"
            episodes_path = meta / "episodes.jsonl"
            lines = []
            if episodes_path.exists():
                for line in episodes_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        if int(row.get("episode_index", -1)) == episode_index:
                            row["teleop_duration_s"] = elapsed
                        lines.append(row)
            if not any(int(r.get("episode_index", -1)) == episode_index for r in lines):
                lines.append({"episode_index": episode_index, "teleop_duration_s": elapsed})
            episodes_path.write_text("".join(json.dumps(r) + "\n" for r in lines), encoding="utf-8")
            stats_path = meta / "episodes_stats.jsonl"
            stat_lines = []
            if stats_path.exists():
                for line in stats_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        row = json.loads(line)
                        if int(row.get("episode_index", -1)) == episode_index:
                            row.setdefault("stats", {})["teleop_duration_s"] = elapsed
                        stat_lines.append(row)
            if not any(int(r.get("episode_index", -1)) == episode_index for r in stat_lines):
                stat_lines.append({"episode_index": episode_index, "stats": {"teleop_duration_s": elapsed}})
            stats_path.write_text("".join(json.dumps(r) + "\n" for r in stat_lines), encoding="utf-8")
            self.depth_writer.commit(episode_index)
        return saved

    def discard(self, reason="Q"):
        self.depth_writer.discard(int(self.dataset.meta.total_episodes))
        return super().discard(reason=reason)

    def close(self):
        try:
            return super().close()
        finally:
            self.depth_writer.close()

    def record_step(self, env, composite, obs):
        base = getattr(env, "unwrapped", env)
        if self.lang is None:
            self.lang = base.get_ep_meta().get("lang", "")
        success = bool(base._check_success())
        mm = getattr(composite, "_mode_manager", None)
        engaged = True if mm is None else mm.current != ManagerMode.OFF
        if engaged:
            if self._teleop_started_at is None:
                self._teleop_started_at = time.monotonic()
            teleop = composite.get_teleop_data_sample()
            frame = build_frame(
                obs,
                teleop,
                reward=float(success),
                done=success,
                task_idx=0,
                teleop_keys=self.teleop_keys,
                composite=composite,
            )
            if frame is not None:
                self.add_frame(frame, depth_mm=_depth_millimeters(obs, base.sim))
        if success and not self.episode_saved and self.ep_frames > 0:
            return self.save(reason="SUCCESS")
        return False


# --------------------------------------------------------------------------- #
# Generic scene selection + env building
# --------------------------------------------------------------------------- #


def env_accepts(env_name):
    """Return the constructor keyword names a registered env effectively accepts.

    Walks the MRO and unions every ``__init__``'s explicit parameters: a task
    subclass often defines ``__init__(self, *args, **kwargs)`` and forwards
    scene kwargs (e.g. ``fixed_scene_seed``) up to its base env, so inspecting
    only the leaf class would miss them.
    """
    cls = REGISTERED_ENVS.get(env_name)
    if cls is None:
        # Holodeck keeps a family registry in addition to robosuite's global
        # registry. Use it as a fallback so --scene-id is not silently dropped
        # when this utility is imported before all environment side effects.
        try:
            from robocasa.environments.holodeck.holodeck import (
                REGISTERED_HOLODECK_ENVS,
            )

            cls = REGISTERED_HOLODECK_ENVS.get(env_name)
        except ImportError:
            cls = None
    if cls is None:
        return set()
    names = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for pname, p in inspect.signature(init).parameters.items():
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY):
                names.add(pname)
    return names


def forward_scene_kwargs(
    env_name,
    *,
    scene_id=None,
    layout=None,
    style=None,
    robot_scale=None,
    fixed_scene=False,
    fixed_objects=False,
    scene_seed=0,
    warn=True,
):
    """Build scene kwargs by forwarding only what the target env accepts.

    Maps CLI scene options to whichever constructor kwarg the env exposes:
      --scene-id      -> scene_ids        (MolmoSpaces iTHOR room envs)
      --layout/-style -> layout_ids/style_ids  (kitchen / living-room)
      --fixed-scene   -> fixed_scene_seed  (universal)
      --fixed-objects -> fixed_objects_seed
    Unsupported options for the chosen env are skipped (with a warning).
    """
    params = env_accepts(env_name)
    sk = {}

    def put(value, param, label):
        if value is None:
            return
        if param in params:
            sk[param] = value
        elif warn:
            print(
                colored(
                    f"[collect] note: {env_name} does not take '{param}'; "
                    f"ignoring --{label}.",
                    "yellow",
                )
            )

    # Holodeck tasks maintain a dedicated family registry. Forward this
    # explicitly as well as through signature discovery so --scene-id cannot
    # silently fall back to the task's default scene set.
    if scene_id is not None and env_name.startswith("Holodeck"):
        sk["scene_ids"] = scene_id
    else:
        put(scene_id, "scene_ids", "scene-id")
    put(layout, "layout_ids", "layout")
    put(style, "style_ids", "style")
    put(robot_scale, "robot_scale", "robot-scale")

    if fixed_scene:
        if "fixed_scene_seed" in params:
            sk["fixed_scene_seed"] = scene_seed
            print(
                colored(
                    f"[collect] FIXED scene (seed {scene_seed}) — identical "
                    "scene every episode.",
                    "green",
                )
            )
        elif warn:
            print(
                colored(
                    f"[collect] note: {env_name} has no fixed_scene_seed.", "yellow"
                )
            )
    elif fixed_objects:
        if "fixed_objects_seed" in params:
            sk["fixed_objects_seed"] = scene_seed
            print(
                colored(f"[collect] FIXED objects (seed {scene_seed}).", "green"),
                flush=True,
            )
        elif warn:
            print(
                colored(
                    f"[collect] note: {env_name} has no fixed_objects_seed.", "yellow"
                )
            )
    return sk


def build_teleop_env(
    env_name,
    robot,
    controller_config,
    scene_kwargs,
    headless=False,
    control_freq=FPS,
):
    """robosuite.make a G1 teleop/collection env (ego cam + the given
    controller config). Callers build ``controller_config`` themselves --
    e.g. ``robocasa.g1_host.bridge.load_sim2real_controller_config()`` for
    the unified collect_demos_g1.py/collect_fixed_task.py -- so this stays
    agnostic to which composite controller family is in use."""
    return robosuite.make(
        env_name=env_name,
        robots=robot,
        controller_configs=controller_config,
        has_renderer=not headless,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[EGO_CAMERA],
        camera_heights=EGO_H,
        camera_widths=EGO_W,
        camera_depths=True,
        render_camera=None,
        ignore_done=True,
        control_freq=control_freq,
        renderer="mjviewer" if not headless else "mujoco",
        **scene_kwargs,
    )


def collect_demos_main(args):
    """Run a full collection session from a parsed argparse Namespace.

    Required attrs: env, robot, scene_id, layout, style, fixed_scene,
    fixed_objects, scene_seed, backend, pico_source, policy_config,
    gmr_viewer, task_prompt, out, max_episodes, control_freq, no_viewer,
    no_show_camera, debug_record_steps, force_engage. Optional:
    amo_arm_source, inference_backend (both read with getattr).

    Every backend (sonic/amo/heft/scalebfm/almi) drives
    ``G1BridgeComposite`` ("G1_BRIDGE", robocasa/g1_host/bridge.py).

    ``g1_bridge`` is imported here rather than at module level. That
    started as a circular-import workaround (this module used to define the
    Modality* types that g1_bridge.modality needs; they now live in
    robocasa.utils.modality_schema, so the cycle is gone). It stays function-
    local for a second, independent reason: importing g1_bridge pulls in
    torch, onnxruntime and the whole external sim2real package, and everything
    else in this module works without them.
    """
    from robocasa.g1_host.bridge import load_sim2real_controller_config
    from g1_bridge.modality import G1_BACKEND_TELEOP_KEYS

    scene_kwargs = forward_scene_kwargs(
        args.env,
        scene_id=getattr(args, "scene_id", None),
        layout=getattr(args, "layout", None),
        style=getattr(args, "style", None),
        robot_scale=getattr(args, "robot_scale", None),
        fixed_scene=args.fixed_scene,
        fixed_objects=getattr(args, "fixed_objects", False),
        scene_seed=args.scene_seed,
    )

    teleop_keys = G1_BACKEND_TELEOP_KEYS[args.backend]

    out_root = args.out or default_out_dir(args.env, args.backend)
    # Built before the dataset so its resolved settings can be recorded as
    # collection_info. Only assembles dicts -- it does not touch the PICO SDK,
    # so this stays well clear of the ordering constraint below.
    controller_config = load_sim2real_controller_config(
        args.backend,
        args.pico_source,
        policy_config=getattr(args, "policy_config", None),
        gmr_viewer=getattr(args, "gmr_viewer", False),
        inference_backend=getattr(args, "inference_backend", None),
        amo_arm_source=getattr(args, "amo_arm_source", None),
    )
    # Create/import the LeRobot dataset before robosuite.make starts the real
    # PICO SDK.  The XRoboToolkit SDK leaves C++ streaming threads alive after
    # xrt.init(); importing pyarrow/video/dataset native modules while those
    # threads are running has been observed to segfault on Ubuntu.
    dataset = make_dataset(
        out_root,
        teleop_keys,
        fps=args.control_freq,
        collection_info=build_collection_info(args, controller_config),
    )

    headless = args.no_viewer or args.debug_record_steps > 0
    run_sonic_viewer_loop = None
    if not headless:
        from robocasa.demos.demo_empty_teleop import run_sonic_viewer_loop

    env = build_teleop_env(
        args.env,
        args.robot,
        controller_config,
        scene_kwargs,
        headless=headless,
        control_freq=args.control_freq,
    )

    # ---- debug plumbing path: reset + record N steps + save + exit ----
    if args.debug_record_steps > 0:
        base = getattr(env, "unwrapped", env)
        lang = args.task_prompt or base.get_ep_meta().get("lang", args.env)
        composite = env.robots[0].composite_controller
        if getattr(args, "force_engage", False) and composite._mode_manager is not None:
            # POSE, not PLANNER: PLANNER is the idle/hold state for the GMR
            # tracking backends (heft/scalebfm) and the pose-tracking state for
            # sonic-smpl, so forcing PLANNER would engage without ever
            # producing a tracked reference -- no frames, nothing recorded.
            composite._mode_manager.force_mode(ManagerMode.POSE)
            print(
                colored(
                    "[collect] debug: forced ModeManager to POSE (tracking).",
                    "yellow",
                )
            )
        zero = np.zeros(env.action_dim)
        env.step(zero)
        print(
            colored(
                f"[collect] debug: recording {args.debug_record_steps} steps", "yellow"
            )
        )
        require_ready = not getattr(args, "debug_skip_readiness", False)
        depth_writer = DepthSequenceWriter(dataset.root)
        n = 0
        for _ in range(args.debug_record_steps):
            obs, _, _, _ = env.step(zero)
            teleop = composite.get_teleop_data_sample()
            frame = build_frame(
                obs,
                teleop,
                reward=0.0,
                done=False,
                task_idx=0,
                teleop_keys=teleop_keys,
                composite=composite,
                require_ready=require_ready,
            )
            if frame is not None:
                dataset.add_frame(frame, task=lang)
                depth_writer.add(
                    _depth_millimeters(obs, env.sim), episode_index=0, frame_index=n
                )
                n += 1
        if n > 0:
            depth_writer.flush(0)
            dataset.save_episode()
            depth_writer.commit(0)
            if require_ready:
                print(colored(f"[collect] debug episode saved ({n} frames).", "green"))
            else:
                print(
                    colored(
                        f"[collect] debug: plumbing OK, env/controller construct "
                        f"and step cleanly ({n} frames, --debug-skip-readiness "
                        "so this does NOT confirm a real GMR/PICO reference was "
                        "received).",
                        "green",
                    )
                )
        elif require_ready:
            print(
                colored(
                    "[collect] debug: no frames (readiness signal never ready).", "red"
                )
            )
        else:
            print(
                colored(
                    "[collect] debug: no frames (env.step or build_frame failed).",
                    "red",
                )
            )
            depth_writer.discard(0)
        depth_writer.close()
        # no EpisodeRecorder on this path, so sweep the PNG staging tree here
        dataset.stop_image_writer()
        purge_image_cache(dataset)
        if composite._pico_source is not None:
            composite._pico_source.stop()
        return

    # ---- interactive teleop + record (shared G1 control loop) ----
    if args.pico_source == "mock":
        print(
            colored(
                "[collect] mock PICO — buttons never fire; use " "--pico-source real.",
                "yellow",
            )
        )
    recorder = EpisodeRecorder(
        dataset,
        args.task_prompt,
        teleop_keys,
        max_episodes=args.max_episodes,
        # Lets a backend take its per-episode origin (heft's episode-relative
        # reference root). Late-bound through the env: robosuite rebuilds the
        # composite on every env.reset(), so a reference captured now goes stale.
        on_episode_start=lambda: env.robots[
            0
        ].composite_controller.session.on_episode_start(),
    )

    run_sonic_viewer_loop(
        env,
        recorder=recorder,
        show_ego=not args.no_show_camera,
        initial_reset=False,
    )


def add_collection_args(parser):
    """Register the collection CLI args on an argparse parser.

    The host-agnostic half comes from ``g1_bridge.recording.cli`` and is shared
    verbatim with the real-robot collector -- anything defined twice drifts, and
    a drifted flag means two datasets that look alike and are not. What is added
    here is only what a simulator has: scene selection and the viewer.
    """
    add_shared_collection_args(parser)
    parser.add_argument("--robot", type=str, default="G1FloatingBody")
    parser.add_argument(
        "--scene-id",
        type=int,
        default=None,
        help="FloorPlan id for MolmoSpaces room envs (ignored by "
        "envs that don't take scene_ids).",
    )
    parser.add_argument(
        "--layout",
        type=int,
        default=None,
        help="layout id for kitchen/living-room envs.",
    )
    parser.add_argument(
        "--style", type=int, default=None, help="style id for kitchen/living-room envs."
    )
    parser.add_argument(
        "--robot-scale",
        type=float,
        default=None,
        help="override iTHOR G1 model scale for envs that support robot_scale.",
    )
    parser.add_argument(
        "--fixed-scene",
        action="store_true",
        help="identical, deterministic scene every episode " "(seed = --scene-seed).",
    )
    parser.add_argument(
        "--fixed-objects",
        action="store_true",
        help="same object instances every episode, randomized "
        "positions (mutually exclusive with --fixed-scene).",
    )
    parser.add_argument("--scene-seed", type=int, default=0)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="headless (no operator window); for plumbing tests.",
    )
    parser.add_argument(
        "--no-show-camera",
        action="store_true",
        help="don't open the live ego-camera preview window.",
    )
    return parser
