"""Watch a P2 scene rollout: live in a viewer window, or as an mp4 afterwards.

Two outputs, and they deliberately take DIFFERENT paths through MuJoCo, because
the constraint that shaped P4-2 has not gone away:

    live      GLFW onscreen, in the rollout process, drawn as it happens
    video     states recorded during the rollout, rendered LATER in a process
              that holds no CUDA context

An EGL offscreen context in a process that also holds a CUDA context deadlocks on
shared multi-GPU nodes (DEBUG_LOG #11), and Kimodo lives on the GPU. So the
rollout process must never render offscreen -- ``--video`` therefore saves states
and hands them to ``view_rollout.py`` in a subprocess. The live window is a
different backend (GLFW, onscreen) and is not known to hit that deadlock, but it
is still opt-in for the same reason: if it ever hangs, ``--video`` is unaffected
because it does its rendering somewhere else entirely.

⚠️ Watching costs correctness nothing, but ``--view`` DOES change timing: the
viewer syncs on the sim clock so the episode runs in real time instead of as fast
as the machine can go. Never quote wall-clock numbers from a viewed run.

⚠️ ``mj_forward`` before every draw and every record. ``mj_step`` runs the
forward dynamics and THEN integrates, so when ``env.step()`` returns, ``d.qpos``
is the new state while ``d.xpos`` still describes the pose one integration
earlier. Drawing straight after a step shows a stale body. Measured cost of
getting this wrong: 86 grey levels (h12).
"""

from __future__ import annotations

import os
import time

import numpy as np


#: Marker colours. FOUR THINGS, all of them ASKS, plus the ghost -- what Kimodo
#: was told to do and what it generated. Where the robot actually is, is the
#: robot: it is already on screen and a sphere on top of it adds nothing.
#:
#: Distinct hues rather than shades, because the thing being judged is often a
#: 10 cm gap between two of them.
MARK = {"root_goal": (0.9, 0.1, 0.1, 0.9),    # target root position
        "goal_left": (0.1, 0.9, 0.2, 0.9),    # target LEFT end-effector
        "goal_right": (1.0, 1.0, 1.0, 0.95)}  # target RIGHT end-effector


def add_marker(scene, pos, rgba=(0.1, 0.9, 0.2, 0.9), size: float = 0.03) -> bool:
    """Append a sphere to an MjvScene. False if the scene is full.

    Markers are why watching beats reading the log. "palm 0.136" says the hand
    stopped 13.6 cm from the object; two spheres say whether it stopped SHORT of
    it, went PAST it, or is beside it with the fingers pointing the wrong way --
    which are three different bugs with the same number.
    """
    import mujoco

    if scene.ngeom >= scene.maxgeom:
        return False
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        g, type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([size, size, size], dtype=np.float64),
        pos=np.asarray(pos, dtype=np.float64).reshape(3),
        mat=np.eye(3, dtype=np.float64).reshape(9),
        rgba=np.asarray(rgba, dtype=np.float32).reshape(4))
    scene.ngeom += 1
    return True


def _draw_markers(scene, markers) -> None:
    """``markers`` is a list of (pos, rgba) or (pos, rgba, size), or a callable
    returning one -- callable so a marker can follow a moving object."""
    if markers is None:
        return
    items = markers() if callable(markers) else markers
    for m in (items or []):
        pos, rgba = m[0], m[1]
        size = m[2] if len(m) > 2 else 0.03
        if pos is not None:
            add_marker(scene, pos, rgba, size)


#: The reference's own body positions, drawn translucent. Not a second robot:
#: MjvScene can only take primitives, so the ghost is one sphere per body. That
#: is enough to answer the question a ghost is for -- is the robot BEHIND the
#: reference, or is the reference itself in the wrong place.
GHOST_RGBA = (0.6, 0.6, 0.95, 0.35)


def _draw_ghost(scene, tr, ghost, rgba=GHOST_RGBA, size: float = 0.035) -> None:
    """``ghost`` is a callable returning [B, 3] world positions, or None."""
    pts = ghost() if callable(ghost) else ghost
    if pts is None:
        return
    for p in np.asarray(pts, dtype=np.float64).reshape(-1, 3):
        if np.isfinite(p).all():
            add_marker(scene, p, rgba, size)


class ViewerClosed(RuntimeError):
    """The window was closed. Raised from the tick hook so the run can stop."""


class ResetRequested(RuntimeError):
    """R was pressed in the window: abandon this episode and start another.

    Raised rather than flagged, for the same reason `ViewerClosed` is: the
    rollout loop is several frames deep inside `tr.rollout`, and a flag would
    only be noticed after the current 3 s chunk had finished playing.
    """


class LiveViewer:
    """A passive GLFW window that follows a running rollout.

    Used as a ``rollout(on_tick=...)`` observer::

        with LiveViewer(tr) as v:
            tr.rollout(300, on_tick=v.on_tick)

    Passive rather than managed: the simulation is stepped by robosuite, not by
    the viewer, so the viewer must not own the loop. ``launch_passive`` is the
    only mode that leaves stepping to the caller.
    """

    def __init__(self, tr, *, realtime: bool = True, every: int = 1,
                 track_root: bool = True, distance: float = 3.0,
                 azimuth: float = 90.0, elevation: float = -12.0,
                 markers=None, ghost=None, success=None):
        self.tr = tr
        #: A ``success_view.success_probe`` callable, or None. Drawn as the
        #: closest robot/fixture geom pair and the segment between them, because
        #: the criterion IS that distance -- there is no radius that stands in
        #: for it (see success_view's header, and I-14 for what happened the last
        #: time a radius was treated as the criterion).
        self.success = success
        self._success_state = None
        # Either a list, or a callable returning one. h7 uses set_marks below,
        # which is the same mechanism the recorder uses -- the live and offline
        # paths must show the SAME markers or watching one teaches you something
        # untrue about the other.
        self.markers = markers
        self.ghost = ghost
        self.realtime = realtime
        self.every = max(1, int(every))
        self.track_root = track_root
        self._cam = (distance, azimuth, elevation)
        self.viewer = None
        self._t0 = None
        self._i = 0
        self._goal = None
        self._root_goal = None
        self._side = "right"

    def set_marks(self, goal=None, obj=None, root_goal=None,
                  side: str = "right") -> None:
        """Aim the ask markers, in WORLD. Same signature as the recorder.

        ``obj`` is accepted and ignored: the object is visible in the scene, and
        a sphere on it competes with the goal marker it sits 0.13 m behind.
        """
        self._goal = None if goal is None else np.asarray(goal, dtype=np.float64)
        self._side = "left" if str(side).lower().startswith("l") else "right"
        self._root_goal = (None if root_goal is None
                           else np.asarray(root_goal, dtype=np.float64))

    def _auto_marks(self):
        out = []
        if self._root_goal is not None:
            out.append((self._root_goal, MARK["root_goal"], 0.06))
        if self._goal is not None:
            out.append((self._goal, MARK[f"goal_{self._side}"], 0.045))
        return out

    def __enter__(self):
        import mujoco.viewer

        # No MUJOCO_GL override: onscreen wants the platform GL, and forcing
        # "egl" here (as the offline renderer does) produces a context that
        # cannot present to a window.
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            raise SystemExit(
                "[view] no DISPLAY -- a live viewer needs one. Use --video "
                "instead, which renders offscreen in a separate process and "
                "works headless.")
        self._reset_asked = False
        self.viewer = mujoco.viewer.launch_passive(
            self.tr.model, self.tr.data, show_left_ui=False, show_right_ui=False,
            key_callback=self._on_key)
        d, az, el = self._cam
        self.viewer.cam.distance = d
        self.viewer.cam.azimuth = az
        self.viewer.cam.elevation = el
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        return False

    def _on_key(self, keycode) -> None:
        """R restarts the episode. The viewer keeps SPACE (pause) and the rest."""
        if int(keycode) in (82, 114):          # 'R' / 'r'
            self._reset_asked = True

    @property
    def alive(self) -> bool:
        return self.viewer is not None and self.viewer.is_running()

    def on_tick(self, i=None, _t=None) -> None:
        """Draw the CURRENT state. Safe to pass straight to ``rollout``.

        ⚠️ Raises `ViewerClosed` once the window is gone. Returning quietly --
        which is what this did -- leaves the rollout running to completion with
        nothing to look at, and on a multi-chunk episode that is minutes of GPU
        after the user has already said they are done.
        """
        if self.viewer is None:
            return
        if not self.viewer.is_running():
            raise ViewerClosed()
        if self._reset_asked:
            self._reset_asked = False
            raise ResetRequested()
        self._i += 1
        if self._i % self.every:
            return
        import mujoco

        # See the module docstring: the body poses are one integration stale
        # until this runs.
        mujoco.mj_forward(self.tr.model, self.tr.data)
        if self.track_root:
            self.viewer.cam.lookat[:] = self.tr.data.xpos[self.tr.bid_root]
        self.viewer.user_scn.ngeom = 0
        _draw_markers(self.viewer.user_scn,
                      self.markers if self.markers is not None else self._auto_marks())
        if self.ghost is not None:
            _draw_ghost(self.viewer.user_scn, self.tr, self.ghost)
        if self.success is not None:
            from rl.kimodo.heft import success_view

            # Evaluated here rather than cached from the rollout: the criterion
            # is read off the CURRENT pose, and this hook already had to run
            # `mj_forward` for the same reason the markers do.
            self._success_state = self.success()
            success_view.draw(self.viewer.user_scn, self._success_state,
                              self.success)
        self.viewer.sync()
        if self.realtime:
            # Pace against the SIM clock, so a slow machine falls behind rather
            # than fast-forwarding, and a fast one does not blur past the motion.
            target = self._t0 + self.tr.data.time
            dt = target - time.time()
            if dt > 0:
                time.sleep(min(dt, 0.05))


def _stack_ragged(frames: list) -> np.ndarray:
    """[T, B, 3] from per-tick point sets of possibly differing B, NaN-padded."""
    if not frames:
        return np.zeros((0, 1, 3), dtype=np.float32)
    b = max(int(np.asarray(f).reshape(-1, 3).shape[0]) for f in frames)
    out = np.full((len(frames), b, 3), np.nan, dtype=np.float32)
    for i, f in enumerate(frames):
        pts = np.asarray(f, dtype=np.float32).reshape(-1, 3)
        out[i, :pts.shape[0]] = pts
    return out


class RolloutRecorder:
    """Records states for later rendering, and remembers WHICH episode they are.

    Thin wrapper over ``render.StateRecorder`` that also keeps the per-tick
    annotations worth drawing later (waypoint index, grasp state), and knows how
    to write itself to an npz the offline viewer can read without importing
    torch, robosuite, or anything else from the rollout process.
    """

    def __init__(self, tr):
        from rl.kimodo.heft.render import StateRecorder

        self.tr = tr
        self.rec = StateRecorder(tr)
        self.labels: list[str] = []
        self._label = ""
        # Where the hand WOULD close, per tick. Recorded rather than derived
        # later because it depends on the wrist's orientation, which the saved
        # qpos has but which nothing downstream should have to re-derive.
        self.palm: list = []
        # Static per chunk: where the wrist was asked to go, and where the object
        # is. Recorded PER TICK so the video shows them changing between
        # waypoints -- which is the difference between "it reached the wrong
        # place" and "it reached the right place for the previous waypoint".
        self.goal: list = []
        self.obj: list = []
        # The robot's own ROOT, and where the root was asked to go. The palm/goal
        # pair answers "did the hand arrive"; without these there is no way to see
        # whether the BODY arrived, which is the question every navigation task and
        # most of the failed reaches actually turn on.
        self.root: list = []
        self.root_goal: list = []
        #: Which hand each tick's goal belongs to -- the two are drawn in
        #: different colours and a bimanual scene would otherwise be unreadable.
        self.side: list = []
        # Kimodo's own output, per tick, so the video can show the reference
        # BESIDE the robot. Previously view-only (known_issues I-1), which meant
        # the recorded episodes could not answer "is the robot behind its
        # reference, or is the reference in the wrong place" -- two different bugs
        # that produce the same distance in the log.
        self.ref: list = []
        self._goal = None
        self._obj = None
        self._root_goal = None
        self._side = "right"

    def set_marks(self, goal=None, obj=None, root_goal=None,
                  side: str = "right") -> None:
        """Aim this chunk's ask markers, in WORLD.

        ``obj`` is still RECORDED -- the offline analysis reads it -- but it is
        no longer drawn: the object is in the scene already, and a sphere on it
        competes with the wrist goal sitting 0.13 m in front of it.
        """
        self._goal = None if goal is None else np.asarray(goal, dtype=np.float32)
        self._obj = None if obj is None else np.asarray(obj, dtype=np.float32)
        self._side = "left" if str(side).lower().startswith("l") else "right"
        self._root_goal = (None if root_goal is None
                           else np.asarray(root_goal, dtype=np.float32))

    def label(self, text: str) -> None:
        """Set the annotation applied to subsequent ticks (e.g. 'wp0 pick mug')."""
        self._label = str(text)

    def on_tick(self, i=None, _t=None) -> None:
        import mujoco

        mujoco.mj_forward(self.tr.model, self.tr.data)
        self.rec.step()
        self.labels.append(self._label)
        try:
            self.palm.append(np.asarray(self.tr.wrist_pos("right", frame="world"))
                             + self._palm_axis() * 0.13)
        except Exception:
            self.palm.append(np.full(3, np.nan))
        self.goal.append(self._goal if self._goal is not None else np.full(3, np.nan))
        self.obj.append(self._obj if self._obj is not None else np.full(3, np.nan))
        self.root.append(np.asarray(self.tr.data.xpos[self.tr.bid_root],
                                    dtype=np.float32).copy())
        self.root_goal.append(self._root_goal if self._root_goal is not None
                              else np.full(3, np.nan))
        self.side.append(self._side)
        try:
            pts = self.tr.reference_points()
        except Exception:  # noqa: BLE001 -- a marker must never kill a rollout
            pts = None
        self.ref.append(np.full((1, 3), np.nan, dtype=np.float32) if pts is None
                        else np.asarray(pts, dtype=np.float32))

    def _palm_axis(self) -> np.ndarray:
        from rl.kimodo.heft.scene_frame import palm_axis

        return np.asarray(palm_axis(self.tr.model, self.tr.data,
                                    self.tr.prefix, "right"), dtype=np.float64)

    def save(self, path: str) -> str:
        d = self.rec.to_dict()
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        np.savez_compressed(
            path, qpos=d["qpos"], qvel=d["qvel"], fingerprint=d["fingerprint"],
            labels=np.asarray(self.labels, dtype=object),
            env_name=str(getattr(self.tr, "env_name", "")),
            seed=int(getattr(self.tr, "seed", 0)),
            # How far into the reset sequence this episode is. Under hard_reset
            # the objects are resampled every reset, so this is part of the
            # scene's identity, not bookkeeping.
            palm=np.asarray(self.palm, dtype=np.float32),
            goal=np.asarray(self.goal, dtype=np.float32),
            obj=np.asarray(self.obj, dtype=np.float32),
            root=np.asarray(self.root, dtype=np.float32),
            root_goal=np.asarray(self.root_goal, dtype=np.float32),
            side=np.asarray(self.side, dtype=object),
            # ⚠️ NOT uniformly shaped as recorded: `reference_points` returns
            # None until the first chunk has been generated, and those ticks are
            # placeholders of a different width. Padding here rather than at
            # record time keeps a missing ghost from being a missing episode.
            ref=_stack_ragged(self.ref),
            n_resets=int(getattr(self.tr, "n_resets", 1)),
            # Named rather than indexed: a model loaded from `save_model` has no
            # tracker to ask, and the camera has to know what to follow.
            root_body=str(self.tr.mujoco.mj_id2name(
                self.tr.model, self.tr.mujoco.mjtObj.mjOBJ_BODY,
                int(self.tr.bid_root)) or ""),
            hard_reset=bool(getattr(self.tr, "hard_reset", True)))
        return path

    def save_model(self, path: str) -> str:
        """Write the COMPILED scene beside the states, meshes and all.

        The alternative -- rebuilding from ``env_name`` + ``seed`` + reset index
        -- does not work for a scene whose objects are resampled per reset, and
        the failure is not subtle: on ``IThorLivingRoomLiftStatue`` the rebuild came
        back with ngeom 653 against the recorded 672, and the fingerprint guard
        (correctly) refused to draw. Reproducing a random draw from its inputs
        means reproducing the whole RNG history of the recording process, which
        includes whatever the generator did; that is not a contract worth having.

        A binary model is self-contained, so nothing has to be resolved, sampled
        or looked up at render time. It costs ~170 MB for a living room, which is
        why the caller writes it for the render and then deletes it rather than
        keeping one per episode.
        """
        self.tr.mujoco.mj_saveModel(self.tr.model, path, None)
        return path

    def __len__(self) -> int:
        return len(self.rec.qpos)


class _SavedScene:
    """Just enough of a tracker for ``render_states``: a model, its data and the
    body the camera follows. Built from ``RolloutRecorder.save_model``, so there
    is no scene to rebuild and no resampling to reproduce."""

    def __init__(self, model_path: str, root_body: str = ""):
        import mujoco

        self.model = mujoco.MjModel.from_binary_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        bid = (mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, root_body)
               if root_body else -1)
        # Body 0 is the world, and following it would put the camera at the
        # origin looking at nothing -- better to fail loudly than to render a
        # correct episode nobody can see.
        if bid <= 0:
            raise SystemExit(
                f"[render] the recording names its root body {root_body!r}, "
                "which this model does not have. Re-record with a matching "
                "model rather than guessing a body for the camera to follow.")
        self.bid_root = int(bid)

    def close(self) -> None:
        pass


def _rebuild(env_name: str, seed: int, z):
    """Rebuild the scene the recording came from, at the right point in its life.

    ⚠️ Under ``hard_reset`` the objects are resampled on EVERY reset, so it is
    not enough to build the env with the same seed -- the renderer has to arrive
    at the same index in the reset sequence. The recording stores how many resets
    had happened; this replays exactly that many. Getting it wrong by one was
    caught by the fingerprint guard (ngeom 178 vs 179), which is the whole reason
    that guard is a hard failure rather than a warning.
    """
    from rl.kimodo.heft.scene_track import KimodoHeftTracker

    hard = bool(z["hard_reset"]) if "hard_reset" in getattr(z, "files", []) else False
    n = int(z["n_resets"]) if "n_resets" in getattr(z, "files", []) else 1
    tr = KimodoHeftTracker(env_name=env_name, seed=seed, verbose=False,
                           hard_reset=hard)
    for _ in range(max(1, n) - tr.n_resets):
        tr.reset()
    if tr.n_resets == 0:
        tr.reset()
    return tr


def render_saved(states_path: str, video_path: str, *, env_name: str | None = None,
                 seed: int | None = None, model_path: str | None = None,
                 size=(480, 640), fps: int = 50, stride: int = 1, **cam) -> str:
    """Replay saved states into the scene they came from, write an mp4. NO torch.

    ``model_path`` is the reliable route: a compiled model written by
    ``RolloutRecorder.save_model`` is the scene, so nothing is rebuilt and there
    is nothing to get wrong.

    Without it the scene is rebuilt from ``env_name`` + ``seed`` + reset index,
    and ⚠️ **that only works for scenes whose reset is deterministic.** Under a
    hard reset robocasa resamples the objects, and reproducing the draw means
    reproducing the recording process's entire RNG history -- measured on
    ``IThorLivingRoomLiftStatue``: rebuilt ngeom 653 against the recorded 672. The
    fingerprint check turns that into a refusal rather than a perfectly sharp
    video of an episode that never happened, which is the only reason this is
    safe to run detached from the rollout.
    """
    from rl.kimodo.heft.render import render_states, write_mp4

    z = np.load(states_path, allow_pickle=True)
    env_name = env_name or str(z["env_name"])
    seed = int(z["seed"]) if seed is None else seed
    if model_path:
        root = str(z["root_body"]) if "root_body" in getattr(z, "files", []) else ""
        tr = _SavedScene(model_path, root)
    else:
        if not env_name:
            raise SystemExit(f"[view] {states_path} has no env_name; pass --env")
        tr = _rebuild(env_name, seed, z)
    try:
        def track(name):
            return z[name] if name in getattr(z, "files", []) else None

        palm, goal, obj = track("palm"), track("goal"), track("obj")
        root, root_goal, ref = track("root"), track("root_goal"), track("ref")
        side = z["side"] if "side" in getattr(z, "files", []) else None

        def marks(k):
            # Palm at 0.05, not the 0.025 that "a marker" would suggest: the palm
            # point sits INSIDE the hand's own geometry, so a small sphere is
            # occluded by the very thing it marks -- measured 2 of 9 frames
            # visible at 0.025, 14 of 14 at 0.05. The goal and object markers sit
            # in free space and can be smaller.
            out = []
            if root_goal is not None and k < len(root_goal) \
                    and np.isfinite(root_goal[k]).all():
                out.append((root_goal[k], MARK["root_goal"], 0.06))
            if goal is not None and k < len(goal) and np.isfinite(goal[k]).all():
                sd = "right"
                if side is not None and k < len(side):
                    sd = "left" if str(side[k]).lower().startswith("l") else "right"
                out.append((goal[k], MARK[f"goal_{sd}"], 0.045))
            # Kimodo's own output for this tick, translucent, one sphere per body.
            # Recorded episodes could not show this before (known_issues I-1), so a
            # video could not tell "the robot is behind its reference" from "the
            # reference is in the wrong place".
            if ref is not None and k < len(ref):
                for pt in np.asarray(ref[k]).reshape(-1, 3):
                    if np.isfinite(pt).all():
                        out.append((pt, GHOST_RGBA, 0.035))
            return out

        frames = render_states(tr, {"qpos": z["qpos"], "qvel": z["qvel"],
                                    "fingerprint": z["fingerprint"]},
                               size=size, stride=stride, marks=marks, **cam)
        return write_mp4(frames, video_path, fps=max(1, fps // max(1, stride)))
    finally:
        tr.close()


def replay_saved_live(states_path: str, *, env_name: str | None = None,
                      seed: int | None = None, fps: int = 50,
                      loop: bool = False) -> None:
    """Play saved states back into a live window. Also torch-free.

    This is the path to use when the rollout itself ran headless (on a server, or
    under ``--video``): the episode is over, but it can still be inspected
    interactively -- paused, orbited, stepped -- which a video cannot be.
    """
    import mujoco
    import mujoco.viewer

    from rl.kimodo.heft.scene_track import KimodoHeftTracker

    z = np.load(states_path, allow_pickle=True)
    env_name = env_name or str(z["env_name"])
    seed = int(z["seed"]) if seed is None else seed
    qpos, qvel = z["qpos"], z["qvel"]

    tr = _rebuild(env_name, seed, z)
    try:
        fp = (int(tr.model.nq), int(tr.model.nv), int(tr.model.nbody),
              int(tr.model.ngeom))
        got = tuple(int(x) for x in z["fingerprint"])
        if fp != got:
            raise SystemExit(
                f"[view] model fingerprint {fp} != recording's {got}. These "
                "states describe a different scene; replaying them would show "
                "something that never happened.")
        dt = 1.0 / max(1, fps)
        with mujoco.viewer.launch_passive(tr.model, tr.data, show_left_ui=False,
                                          show_right_ui=False) as v:
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 3.0, 90.0, -12.0
            while v.is_running():
                for k in range(len(qpos)):
                    if not v.is_running():
                        break
                    t0 = time.time()
                    tr.data.qpos[:] = qpos[k]
                    tr.data.qvel[:] = qvel[k]
                    mujoco.mj_forward(tr.model, tr.data)
                    v.cam.lookat[:] = tr.data.xpos[tr.bid_root]
                    v.sync()
                    rest = dt - (time.time() - t0)
                    if rest > 0:
                        time.sleep(rest)
                if not loop:
                    break
    finally:
        tr.close()
