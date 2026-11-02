"""Export usable h7 rollouts as a **heft-format** LeRobot v2.1 dataset.

The format is not defined here. It is defined by ``g1_bridge`` in
``LocoManiLab/g1_backend``, which is the same module LocoManiLab's own collectors
use, and this file IMPORTS it rather than restating it:

    observation.state             29   left_leg6 right_leg6 waist3 left_arm7 right_arm7
    observation.root_orientation   4   wxyz
    observation.projected_gravity  3
    action.ref_qpos               36   ref_root_pos3 + ref_root_quat4 + ref_joint_pos29
    action.hands                  14   7 target finger angles per side
    observation.images.ego_view  240x320 video @ 50 fps
    observation.eef_state         14 |  next.reward 1 | next.done 1   (SIM_ONLY)

⚠️ **Restating that table in our own code is the failure this module exists to
avoid.** g1_bridge's own header says a heft dataset recorded on hardware must
carry the same columns as one recorded in sim "or nothing downstream (LocoManiLab's
variant detection, every model's data config) can treat the two as the same
thing" -- and a copy drifts silently, because a dataset with the wrong column
widths still writes, still validates (``lab validate`` only hard-fails on a
missing ``meta/``/``data/``), and only fails at training time as a shape error
in someone else's repo.

⚠️ NOT ``action.motion_token``. That is the SONIC family's 64-D learned latent
and this pipeline has no SONIC in it: ``robocasa.utils.g1_data_collection``
zero-fills that column when the controller cannot produce one, which would hand
gr00t/fastwam/lingbot-va a dataset whose PRIMARY ACTION IS ALL ZEROS. The heft
layout records the tracking policy's reference frame instead, which is exactly
what Kimodo generates.

TWO PHASES, AND THE SPLIT IS FORCED
-----------------------------------
``HeftFrameRecorder`` runs inside the rollout, where Kimodo holds a CUDA
context, so it renders NOTHING -- it only collects vectors. The ego camera is
rendered afterwards by ``main()`` in a separate process: an EGL offscreen
context in a process that also holds CUDA deadlocks on shared multi-GPU nodes
(DEBUG_LOG #11), which is the same reason ``--video`` already shells out.

    python -m rl.kimodo.heft.export --bundle <ep.heft.npz> --out <dataset dir>
"""

from __future__ import annotations

import os
import sys

import numpy as np

#: Where the schema lives. A sibling checkout, the same way ``h7`` reaches
#: ``heft_work`` -- see the module docstring for why it is imported and not
#: copied. Pinned by LocoManiLab's submodule pointer; ``schema_provenance()``
#: reports the commit so a dataset can be traced to the layout that made it.
G1_BACKEND = os.environ.get("G1_BACKEND", "")

#: The 29 body joints, grouped the way ``observation.state`` is sliced.
#:
#: ⚠️ THIS IS THE **DATASET** ORDER, NOT THE CANONICAL ONE. The two are
#: permutations of each other and both are 29 long, so picking the wrong one
#: produces a full dataset with every joint in the wrong column and nothing that
#: raises. g1_bridge calls its order "the canonical 29-order qpos", but that word
#: means the opposite here (``tracker_const.CANONICAL_JOINT_NAMES`` is the
#: policy's interleaved action order). The grouping settles it: only the dataset
#: order is contiguous as leg/leg/waist/arm/arm, which ``_check_joint_groups``
#: asserts by NAME at construction rather than trusting this comment.
JOINT_GROUPS = (
    ("left_leg", 6), ("right_leg", 6), ("waist", 3),
    ("left_arm", 7), ("right_arm", 7),
)


def _ensure_g1_bridge():
    """Put ``g1_bridge`` on the path, with an actionable error if it is absent."""
    global G1_BACKEND
    selected = os.path.abspath(os.path.expanduser(G1_BACKEND)) if G1_BACKEND else ""
    if not selected or not os.path.isdir(os.path.join(selected, "g1_bridge")):
        raise SystemExit(
            f"[export] no g1_bridge under {selected or '(G1_BACKEND is unset)'}. It defines the dataset\n"
            "         format and is deliberately not copied into this repo:\n"
            "           https://github.com/LocoManiLab/g1_backend\n"
            "         Set G1_BACKEND to the local checkout containing g1_bridge/.")
    G1_BACKEND = selected
    if G1_BACKEND not in sys.path:
        sys.path.insert(0, G1_BACKEND)


def heft_layout():
    """The ``ModalityLayout`` LocoManiLab's heft backend records, straight from source."""
    _ensure_g1_bridge()
    from g1_bridge.modality import G1_BACKEND_TELEOP_KEYS

    return G1_BACKEND_TELEOP_KEYS["heft"]


def schema_provenance() -> dict:
    """Which g1_backend commit defined the layout. Stored in the dataset."""
    import subprocess

    try:
        sha = subprocess.run(["git", "-C", G1_BACKEND, "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        sha = ""
    return {"g1_backend_path": G1_BACKEND, "g1_backend_commit": sha,
            "layout": "heft (GENERIC_TRACKING_MODALITY_LAYOUT)"}


def _check_joint_groups(joint_names) -> None:
    """Assert the 29 names really are contiguous leg/leg/waist/arm/arm.

    ⚠️ The check is on the NAMES, not on the count. Both joint orders in this
    repo are 29 long, both slice without error, and the canonical one puts a
    right knee where ``left_leg`` should be -- an error that survives all the way
    to a trained policy. This is the assertion that the comment on
    ``JOINT_GROUPS`` cannot be.
    """
    want = {"left_leg": ("left_hip", "left_knee", "left_ankle"),
            "right_leg": ("right_hip", "right_knee", "right_ankle"),
            "waist": ("waist",),
            "left_arm": ("left_shoulder", "left_elbow", "left_wrist"),
            "right_arm": ("right_shoulder", "right_elbow", "right_wrist")}
    if len(joint_names) != sum(n for _, n in JOINT_GROUPS):
        raise ValueError(f"expected 29 joints, got {len(joint_names)}")
    i = 0
    for group, n in JOINT_GROUPS:
        for name in joint_names[i:i + n]:
            if not name.startswith(want[group]):
                raise ValueError(
                    f"[export] joint order mismatch: '{name}' landed in group "
                    f"'{group}'. observation.state would be written with the "
                    f"joints permuted and nothing downstream would notice. This "
                    f"is the DATASET order, not tracker_const's CANONICAL one.")
        i += n


def split_joint_groups(q29) -> dict:
    """29 joint values -> the five named groups ``observation.state`` wants."""
    q = np.asarray(q29, dtype=np.float32)
    out, i = {}, 0
    for group, n in JOINT_GROUPS:
        out[group] = q[i:i + n]
        i += n
    return out


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def episode_usable(row: dict, *, require_natural: bool = True) -> bool:
    """Is this episode DATA, as opposed to merely a success.

    ⚠️ Success alone is not the gate, and the gap is not academic:
    ``IThorBathroomApproachSink`` succeeds 100% of the time and is usable 20% of
    the time -- the criterion (root within 1.0 m of the sink) is satisfied with
    the robot's shoulder inside the sink. Exporting on success would put those
    in the dataset with a success label attached.

    ``physics_ok`` is None when the physics screen did not run; that is treated
    as NOT usable rather than as a pass, because "we did not check" and "we
    checked and it was fine" must not export the same episode.

    ``require_natural=False`` drops the smoothness screen from the gate -- what
    ``--no-natural`` passes. ⚠️ It is an explicit ARGUMENT and not "treat None
    as a pass" for the reason above: the caller has to say it out loud, the
    numbers are still measured and still on the row, and the summary records
    which gate produced its ``usable_rate``. Two runs whose gates differ are not
    comparable, and a rate that quietly means something else is the mistake this
    line of work has already made twice.
    """
    return bool(row.get("success")
                and not row.get("fell")
                and row.get("physics_ok") is True
                and (row.get("natural_ok") is True or not require_natural))


def usable_rate(rows, *, require_natural: bool = True) -> float | None:
    """Fraction of episodes that pass the same gate the exporter uses.

    Deliberately computed from ``episode_usable`` rather than from the three
    component rates in the summary: a reported rate whose population differs
    from the filtered population is the mistake this line of work has already
    made twice (P3-2's contact term, P3-3's DLJ), and both times the threshold
    was fine and the POPULATION was wrong.
    """
    rows = [r for r in rows if r is not None]
    return (round(float(np.mean([episode_usable(r, require_natural=require_natural)
                                 for r in rows])), 4)
            if rows else None)


# --------------------------------------------------------------------------- #
# Phase 1: collect, inside the rollout process. Renders nothing.
# --------------------------------------------------------------------------- #
class HeftFrameRecorder:
    """Per-tick heft columns, gathered as a ``rollout(on_tick=...)`` watcher.

    Everything here is a read of state that already exists; no FK, no rendering,
    no allocation beyond one append per tick, because this runs inside the 50 Hz
    loop that every timing number in this line of work was measured on.
    """

    def __init__(self, tr):
        from rl.kimodo.heft.tracker_const import DATASET_JOINT_NAMES

        _check_joint_groups(DATASET_JOINT_NAMES)
        self.tr = tr
        self.joint_names = list(DATASET_JOINT_NAMES)
        comp = tr.composite
        # The commanded finger angles, not the trigger. `_write_gripper_torques`
        # drives a PD loop toward `trigger * CLOSE_TARGETS`, so THAT product is
        # the 7-per-side target the heft contract asks for -- a raw 0/1 scalar
        # broadcast to 7 columns would be a different quantity wearing the right
        # shape.
        self._close_l = np.asarray(comp._HAND_LEFT_CLOSE_TARGETS, dtype=np.float32)
        self._close_r = np.asarray(comp._HAND_RIGHT_CLOSE_TARGETS, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        self.rows: list[dict] = []
        self.success_tick = None

    # -- per tick ----------------------------------------------------------- #
    def on_tick(self, i=None, _tr=None) -> None:
        tr = self.tr
        q29 = np.asarray(tr.joint_pos_ds(), dtype=np.float32)
        root_quat = np.asarray(tr.root_quat(), dtype=np.float32)
        if root_quat.shape != (4,):
            raise ValueError(f"root_quat must be wxyz [4], got {root_quat.shape}")
        ref = self._ref_now()
        comp = tr.composite
        lt = float(np.clip(getattr(comp, "_left_trigger", 0.0), 0.0, 1.0))
        rt = float(np.clip(getattr(comp, "_right_trigger", 0.0), 0.0, 1.0))
        row = {
            **split_joint_groups(q29),
            "root_quat": root_quat,
            "ref_root_pos": ref[0:3].astype(np.float32),
            "ref_root_quat": ref[3:7].astype(np.float32),
            "ref_joint_pos": ref[7:36].astype(np.float32),
            "left_hand_action": (lt * self._close_l).astype(np.float32),
            "right_hand_action": (rt * self._close_r).astype(np.float32),
            "eef_state": self._eef_state(),
        }
        if self.success_tick is None and tr._check_success():
            self.success_tick = len(self.rows)
        self.rows.append(row)

    def _ref_now(self) -> np.ndarray:
        """The reference frame being tracked THIS tick, as a 36-vector.

        ⚠️ Reads the tracker's own cursor (``_ref_tick``) rather than counting
        ticks here. A separately-counted index drifts the moment a chunk is
        reloaded, and the resulting dataset would pair each observation with a
        reference from a slightly different time -- which is indistinguishable
        from a tracking-error dataset until a policy trained on it fails.
        """
        ref = getattr(self.tr, "_last_ref", None)
        if ref is None:
            return np.zeros(36, dtype=np.float32)
        k = int(np.clip(getattr(self.tr, "_ref_tick", 0), 0, len(ref) - 1))
        return np.asarray(ref[k], dtype=np.float32)

    def _eef_state(self) -> np.ndarray:
        """[14] left pos+quat then right pos+quat, in the episode's own frame."""
        d, m = self.tr.data, self.tr.model
        out = []
        for side in ("left", "right"):
            bid = self.tr._bid_wrist[side]
            out.append(self.tr.to_local(d.xpos[bid]))
            out.append(d.xquat[bid].copy())
        return np.concatenate(out).astype(np.float64)

    # -- per episode -------------------------------------------------------- #
    def save(self, path: str, *, lang: str, env_name: str, seed: int,
             usable: bool, extra: dict | None = None) -> str:
        """Write the per-tick columns beside the states npz.

        ``usable`` is stored rather than recomputed later: the screens that
        decide it live in the rollout process, and an exporter that re-derived
        it from whatever it could see would be a second, quieter definition.
        """
        if not self.rows:
            raise ValueError("[export] nothing recorded -- no ticks were run")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        cols = {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        n = len(self.rows)
        reward = np.zeros((n, 1), dtype=np.float32)
        if self.success_tick is not None:
            # The RISING EDGE only, matching EpisodeRecorder: an episode that
            # succeeds and then drops the object still succeeded, and marking
            # every later tick would make the reward a "still holding" signal.
            reward[self.success_tick] = 1.0
        done = np.zeros((n, 1), dtype=bool)
        done[-1] = True
        np.savez_compressed(
            path, lang=str(lang), env_name=str(env_name), seed=int(seed),
            usable=bool(usable), joint_names=np.asarray(self.joint_names, dtype=object),
            success_tick=(-1 if self.success_tick is None else int(self.success_tick)),
            reward=reward, done=done, **cols)
        return path


# --------------------------------------------------------------------------- #
# Phase 2: render + write, in a process with no CUDA.
# --------------------------------------------------------------------------- #
#: The robot's own head camera, the same one ``g1_data_collection`` records from,
#: so a Kimodo episode and a teleop episode show the same view of the same robot.
EGO_CAMERA = "robot0_head_camera"


def render_ego(model_path: str, qpos, size) -> list:
    """Ego frames for a recorded episode, from the COMPILED scene. No torch.

    The compiled ``.mjb`` is used rather than a rebuilt scene for the reason
    ``render_saved`` already documents: a scene that resamples its objects on
    reset cannot be reproduced from env+seed, and a rebuilt room would render a
    sharp, plausible video of an episode that never happened.
    """
    import mujoco

    m = mujoco.MjModel.from_binary_path(model_path)
    d = mujoco.MjData(m)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, EGO_CAMERA)
    if cid < 0:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i)
                 for i in range(m.ncam)]
        raise SystemExit(f"[export] no camera {EGO_CAMERA!r} in {model_path}; "
                         f"the model has {names}")
    from rl.kimodo.heft.render import visual_only_option

    h, w = size
    frames = []
    with mujoco.Renderer(m, height=h, width=w) as r:
        # ⚠️ Without this the dataset's images are the COLLISION geometry --
        # olive slabs for walls, green blobs for the plant -- rather than the
        # scene. robosuite's camera path sets the same thing; see
        # `visual_only_option`. A VLA trained on the wrong one would be trained
        # on a different world than the one it is deployed into.
        r._scene_option = visual_only_option()
        for q in np.asarray(qpos):
            d.qpos[:] = q
            mujoco.mj_forward(m, d)
            r.update_scene(d, camera=cid)
            # ⚠️ NOT flipped. `g1_data_collection` does `[::-1]` because
            # robosuite hands back an OpenGL buffer bottom-up; mujoco.Renderer
            # already returns it top-down, so flipping here would store this
            # pipeline's episodes upside down relative to every teleop one.
            frames.append(r.render().copy())
    return frames


def export_bundle(bundle_path: str, model_path: str, states_path: str,
                  out_root: str, *, size=None, fps=None, dataset=None,
                  robot_type: str = "G1FloatingBody") -> int:
    """Turn one recorded episode into LeRobot frames. Returns the frame count."""
    _ensure_g1_bridge()
    from g1_bridge.recording.dataset import make_dataset
    from g1_bridge.recording.episode import EpisodeRecorder
    from g1_bridge.recording.features import EGO_H, EGO_W, FPS, SIM_ONLY_FEATURES
    from g1_bridge.recording.frames import build_blob_frame

    size = (EGO_H, EGO_W) if size is None else size
    fps = FPS if fps is None else fps
    layout = heft_layout()

    z = np.load(bundle_path, allow_pickle=True)
    states = np.load(states_path, allow_pickle=True)
    qpos = states["qpos"]
    n = len(z["ref_joint_pos"])
    if len(qpos) != n:
        raise SystemExit(f"[export] {len(qpos)} states against {n} recorded ticks "
                         "-- the two recorders disagree about the episode length, "
                         "so a frame would pair one tick's image with another "
                         "tick's action")

    frames_img = render_ego(model_path, qpos, size)
    own = dataset is None
    if own:
        dataset = make_dataset(out_root, layout, fps=fps,
                               collection_info=schema_provenance(),
                               extra_features=SIM_ONLY_FEATURES,
                               ego_hw=size, robot_type=robot_type)
    lang = str(z["lang"])
    rec = EpisodeRecorder(dataset, lang, layout)
    rec.reset_episode()
    written = 0
    for t in range(n):
        teleop = {k: z[k][t] for k in
                  ("left_leg", "right_leg", "waist", "left_arm", "right_arm",
                   "ref_root_pos", "ref_root_quat", "ref_joint_pos",
                   "left_hand_action", "right_hand_action")}
        frame = build_blob_frame(
            layout, teleop, image=frames_img[t], root_quat=z["root_quat"][t],
            extra_fields={"observation.eef_state": z["eef_state"][t],
                          "next.reward": z["reward"][t],
                          "next.done": z["done"][t]})
        if frame is None:      # readiness sentinel absent: before the first chunk
            continue
        written += rec.append_frame(frame)
    rec.save(reason="USABLE")
    return written


def main() -> int:
    import argparse
    import glob

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", nargs="+", required=True,
                    help="one or more *.heft.npz written by HeftFrameRecorder.save")
    ap.add_argument("--out", required=True, help="dataset root; must not exist")
    ap.add_argument("--size", type=int, nargs=2, default=None,
                    help="ego frame HxW; defaults to g1_bridge's sim size")
    ap.add_argument("--robot-type", default="G1FloatingBody")
    args = ap.parse_args()

    _ensure_g1_bridge()
    from g1_bridge.recording.dataset import make_dataset
    from g1_bridge.recording.features import EGO_H, EGO_W, FPS, SIM_ONLY_FEATURES

    bundles = sorted(b for pat in args.bundle for b in glob.glob(pat))
    if not bundles:
        raise SystemExit(f"[export] no bundles matched {args.bundle}")
    size = tuple(args.size) if args.size else (EGO_H, EGO_W)
    dataset = make_dataset(args.out, heft_layout(), fps=FPS,
                           collection_info=schema_provenance(),
                           extra_features=SIM_ONLY_FEATURES, ego_hw=size,
                           robot_type=args.robot_type)
    total = 0
    for b in bundles:
        stem = b[:-len(".heft.npz")] if b.endswith(".heft.npz") else os.path.splitext(b)[0]
        model, states = stem + ".mjb", stem + ".npz"
        for p in (model, states):
            if not os.path.exists(p):
                raise SystemExit(f"[export] {b} has no {p} beside it")
        k = export_bundle(b, model, states, args.out, size=size, dataset=dataset)
        print(f"[export] {os.path.basename(b)} -> {k} frames")
        total += k
    print(f"[export] {len(bundles)} episodes, {total} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
