"""Kimodo chunks -> HEFT tracker -> a robosuite task scene, through ``env.step()``.

Structural peer of ``rl/kimodo/sonic_track.py::KimodoSonicTracker``, with SONIC
swapped for the ported HEFT tracker. The differences from that class are not
cosmetic, and each one is a bug that does not raise if you carry the SONIC habit
across:

  * **Never step the sim before a reference is loaded.** ``KimodoSonicTracker``
    warms up with one ``env.step`` to lazy-init the kPlanner. Doing that here
    knocks the robot over first: HEFT's gains are soft on purpose and the gravity
    compensation lives in the policy's delta, so commanding the default pose with
    no reference drops the root 0.72 m over 200 ticks (measured). ``reset()``
    therefore does no warm-up, and ``rollout`` asserts
    ``heft_ticks_without_reference() == 0`` rather than trusting the ordering.

  * **The root body is ``mobilebase0_support``, not ``robot0_pelvis``.** The
    latter is the floating base at floor level. Reading it as the pelvis tells
    the generator the robot is crouching 0.79 m, every frame, silently.

  * **The floor is not z = 0**, and the reference's root height is measured from
    it. Shared with the driver via ``scene_frame.resolve_floor_z`` so the two
    paths cannot drift onto floors a couple of centimetres apart.

  * **The reference is not re-anchored in XY, on purpose.** The tracker's
    observation uses inter-step displacement in the reference's own root frame,
    so it is translation-invariant; what is NOT invariant is height (measured
    from the floor) and heading (rot6d against the robot's current quaternion).
    ``gen.continue_chunk`` already matches the heading to the prefix. Shifting XY
    here would be a no-op on the policy and a lie in the recorded metrics.

The frame contract, in one place, because three of the four bugs above are frame
bugs:

    world        what MuJoCo reports; the scene's own coordinates
    local        world minus (spawn XY, floor z). Kimodo's convention: ground at
                 0, and the prefix near the origin. Everything handed to or read
                 back from the generator is in this frame -- prefixes, wrist
                 goals, root sub-goals.

``local`` is the single-env analogue of mjlab's ``scene.env_origins``.
"""

from __future__ import annotations

import numpy as np

from rl.kimodo.heft.scene_frame import resolve_floor_z

#: Kimodo emits 30 Hz; the tracker consumes 50 Hz.
KIMODO_FPS, HEFT_FPS = 30.0, 50.0


class KimodoHeftTracker:
    """A scene, a HEFT tracker in its composite controller, and a chunk cursor."""

    def __init__(
        self,
        env=None,
        *,
        env_name: str = "EmptyEnv",
        robot: str = "G1FloatingBody",
        control_freq: int = 50,
        seed: int = 0,
        offscreen: bool = False,
        hard_reset: bool = True,
        onnx_path: str | None = None,
        verbose: bool = True,
        **env_kwargs,
    ):
        """``offscreen=False`` by default, and that default is load-bearing.

        An EGL offscreen context in a process that also holds a CUDA context
        deadlocks on shared multi-GPU nodes (DEBUG_LOG #11), and this class is
        meant to be driven by a generator sitting on the GPU. Record states and
        render offline from them (``rl.kimodo.viz.render_qpos``) instead.
        """
        import mujoco

        import robocasa  # noqa: F401  -- registers the scenes and the G1 robot
        import robosuite
        from robosuite.controllers import load_composite_controller_config

        self.mujoco = mujoco
        self.prefix = "robot0_"
        self.verbose = bool(verbose)
        # Kept so a recording can name the scene it came from. Replaying states
        # into the wrong scene draws a convincing video of a different episode,
        # so the identity has to travel WITH the states -- see view.render_saved.
        self.n_resets = 0
        self.env_name = env_name if env is None else type(env).__name__
        self.seed = int(seed)
        self.hard_reset = bool(hard_reset)

        if env is None:
            cfg = self._controller_config_path()
            ctrl = load_composite_controller_config(controller=cfg)
            if onnx_path:
                ctrl["composite_controller_specific_configs"]["heft_policy_onnx"] = str(onnx_path)
            self.env = robosuite.make(
                env_name=env_name, robots=robot, controller_configs=ctrl,
                has_renderer=False, has_offscreen_renderer=bool(offscreen),
                use_camera_obs=False, ignore_done=True,
                control_freq=control_freq, seed=seed, **env_kwargs,
            )
            self.env.hard_reset = bool(hard_reset)
        else:
            self.env = env

        self._zero = np.zeros(self.env.action_dim, dtype=np.float32)
        self.substeps = int(round(1.0 / (control_freq * self.env.sim.model.opt.timestep)))
        self.reset()

    @staticmethod
    def _controller_config_path() -> str:
        import pathlib

        import robosuite

        return str(pathlib.Path(robosuite.__file__).parent
                   / "controllers/config/robots/default_g1_heft.json")

    # ------------------------------------------------------------------ frame
    def _bind(self) -> None:
        """Re-resolve everything that ``env.reset()`` invalidates.

        ``reset()`` rebuilds the composite controller (via ``_load_controller``)
        and, under ``hard_reset``, the MjModel itself. A cached handle to either
        silently drives nothing / reads a dead array.
        """
        from rl.kimodo.heft import model_patch as mp

        self.composite = self.env.robots[0].composite_controller
        self.model = self.env.sim.model._model
        self.data = self.env.sim.data._data

        if getattr(self.composite, "_ctrl_type", None) != "heft_tracker":
            raise SystemExit(
                f"[scene_track] composite is {getattr(self.composite, '_ctrl_type', None)!r}, "
                "not 'heft_tracker' -- load default_g1_heft.json")

        self.root_body, self.root_offset = mp.resolve_root_body(self.model, self.prefix)
        self.bid_root = self.mujoco.mj_name2id(
            self.model, self.mujoco.mjtObj.mjOBJ_BODY, self.root_body)
        self.floor_z = resolve_floor_z(self.model, self.data)
        self.origin_xy = self.data.xpos[self.bid_root][:2].copy()

        self._bid_wrist = {
            side: self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY,
                                         f"{self.prefix}{side}_wrist_yaw_link")
            for side in ("left", "right")
        }
        self._q_ds, self._v_ds = self._joint_addrs()

    def _joint_addrs(self):
        from rl.kimodo.heft.tracker_const import DATASET_JOINT_NAMES

        q, v = [], []
        for n in DATASET_JOINT_NAMES:
            jid = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT,
                                         f"{self.prefix}{n}")
            if jid < 0:
                raise SystemExit(f"[scene_track] no joint {self.prefix}{n}")
            q.append(int(self.model.jnt_qposadr[jid]))
            v.append(int(self.model.jnt_dofadr[jid]))
        return np.asarray(q), np.asarray(v)

    # ------------------------------------------------------------------ state
    def root_pos_local(self) -> np.ndarray:
        p = self.data.xpos[self.bid_root].copy()
        p[0] -= self.origin_xy[0]
        p[1] -= self.origin_xy[1]
        p[2] -= self.floor_z
        return p

    def root_quat(self) -> np.ndarray:
        return self.data.xquat[self.bid_root].copy()          # wxyz

    def joint_pos_ds(self) -> np.ndarray:
        return self.data.qpos[self._q_ds].copy()

    def realized_qpos36(self) -> np.ndarray:
        """[36] the ROBOT's pose in the local frame, dataset joint order.

        This seeds the next chunk, and it must be the robot's state rather than
        the reference's: the accumulated tracking error is the thing the whole
        pipeline exists to measure, and feeding the reference back would erase it
        from the generator's input while every number still looked reasonable.
        """
        return np.concatenate([self.root_pos_local(), self.root_quat(),
                               self.joint_pos_ds()]).astype(np.float32)

    def wrist_pos(self, side: str = "right", frame: str = "local") -> np.ndarray:
        """Wrist position. ``frame='world'`` to compare against scene objects."""
        p = self.data.xpos[self._bid_wrist[side]].copy()
        if frame == "world":
            return p
        p[0] -= self.origin_xy[0]
        p[1] -= self.origin_xy[1]
        p[2] -= self.floor_z
        return p

    def to_local(self, world_xyz) -> np.ndarray:
        """A scene point in the generator's frame. Goals go through here."""
        p = np.asarray(world_xyz, dtype=np.float64).copy()
        p[0] -= self.origin_xy[0]
        p[1] -= self.origin_xy[1]
        p[2] -= self.floor_z
        return p

    def to_world(self, local_xyz) -> np.ndarray:
        p = np.asarray(local_xyz, dtype=np.float64).copy()
        p[0] += self.origin_xy[0]
        p[1] += self.origin_xy[1]
        p[2] += self.floor_z
        return p

    def penetrations(self, tol: float = 1e-3) -> list[tuple[str, str, float]]:
        """Robot bodies currently interpenetrating scene geometry, deepest first.

        Ground and walls are excluded by GEOM name, not body name: robocasa hangs
        the floor and wall geoms off the worldbody, so every one of them reports
        a body called ``world`` and a body-name filter passes them all through --
        which reports a robot standing quietly on the floor as eight collisions
        with the furniture.

        Checked at reset because a spawn that overlaps the furniture is the most
        expensive failure in this pipeline: MuJoCo resolves the overlap by
        shoving the robot, so the very first ticks of the episode contain a push
        that has nothing to do with the reference. Measured on
        ``SquatPickPlaceTVStand``: 50 mm of knee-in-coffee-table at tick 0 turned
        a 0.24 m reach error into 0.57 m and 2-4x the commanded root travel, and
        nothing in the rollout reported anything wrong.
        """
        out = []
        for c in range(self.data.ncon):
            con = self.data.contact[c]
            if con.dist > -abs(tol):
                continue
            g1, g2 = con.geom1, con.geom2
            b1 = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY,
                                        self.model.geom_bodyid[g1]) or ""
            b2 = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY,
                                        self.model.geom_bodyid[g2]) or ""
            r1, r2 = b1.startswith(self.prefix), b2.startswith(self.prefix)
            if r1 == r2:
                continue
            gn = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM,
                                        g2 if r1 else g1) or ""
            if gn.startswith("floor") or gn.startswith("wall"):
                continue
            out.append(((b1 if r1 else b2).replace(self.prefix, ""), gn, float(con.dist)))
        return sorted(out, key=lambda t: t[2])

    def sole_clearance(self) -> float:
        """Lowest foot geom above the floor datum. Confirms the datum is right."""
        zs = []
        for g in range(self.model.ngeom):
            bn = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY,
                                        self.model.geom_bodyid[g]) or ""
            if "ankle_roll" in bn and (self.model.geom_contype[g]
                                       | self.model.geom_conaffinity[g]):
                zs.append(float(self.data.geom_xpos[g][2] - self.model.geom_size[g][0]))
        return min(zs) - self.floor_z if zs else float("nan")

    # ------------------------------------------------------------------ cycle
    def reset(self) -> np.ndarray:
        """New episode. Returns the realized qpos36 -- no sim steps are taken.

        Deliberately does NOT warm up. See the class docstring: a step here is a
        step of the robot collapsing.
        """
        self.env.reset()
        # Counted because under hard_reset the object set is RESAMPLED on every
        # reset, so "which episode this is" is part of the scene's identity.
        # Offline rendering has to arrive at the same point in that sequence or
        # it draws a different scene -- caught in practice as a fingerprint
        # mismatch of exactly one geom (ngeom 178 vs 179).
        self.n_resets = getattr(self, "n_resets", 0) + 1
        self._bind()
        self.n_chunks = 0
        # The scratch MjData the ghost draws into belongs to the OLD model. Two
        # scenes can share an nq and still differ, so the shape check in
        # `reference_points` is the backstop and this is the actual fix: the
        # model is rebuilt here, so the cache dies here.
        self._ghost_data = None
        self._success_seen = False
        self._start_root_z = float(self.root_pos_local()[2])
        self._min_root_z = self._start_root_z
        self.spawn_penetrations = self.penetrations()
        if self.verbose:
            print(f"[scene_track] root={self.root_body} floor_z={self.floor_z:.4f} "
                  f"origin_xy={np.round(self.origin_xy, 3)} "
                  f"sole_clearance={self.sole_clearance():+.4f} m")
        if self.spawn_penetrations:
            worst = self.spawn_penetrations[0]
            print(f"[scene_track] WARNING spawn overlaps the scene: "
                  f"{len(self.spawn_penetrations)} contacts, deepest "
                  f"{worst[0]} in {worst[1]} by {worst[2]*1000:.1f} mm. "
                  "The first ticks contain a shove that is not in the reference.")
        return self.realized_qpos36()

    def prefix_frames(self, k: int = 5, true_prefix: bool = False,
                      history: np.ndarray | None = None) -> np.ndarray:
        """The K-frame conditioning prefix for the next chunk.

        ``true_prefix=False`` tiles ONE realized frame K times, which makes the
        prefix's velocity identically zero -- so the generator is told "the robot
        is standing still here" at every chunk boundary. That is a defect, and it
        is the default anyway, because the fine-tuned weights were trained under
        it: switching it on for an evaluation of those weights introduces a
        distribution shift into the very thing being measured. Turn it on only
        with a matching training run, and report both.
        """
        if true_prefix:
            if history is None or len(history) < k:
                raise ValueError("true_prefix needs >= k realized frames of history")
            return np.asarray(history[-k:], dtype=np.float32)
        return np.tile(self.realized_qpos36(), (int(k), 1)).astype(np.float32)

    def load_chunk(self, qpos_30hz: np.ndarray, reboot: bool | None = None) -> int:
        """Resample a Kimodo chunk to 50 Hz and hand it to the tracker.

        ``reboot`` defaults to True for the first chunk of an episode and False
        afterwards: the boot indicator says "a new command started", and
        re-arming it at every boundary is one of the three stacked causes of the
        documented chunk-boundary stall. Pass it explicitly to override.
        """
        from rl.kimodo.heft_bridge import resample

        q = np.asarray(qpos_30hz, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != 36:
            raise ValueError(f"expected [T, 36] Kimodo qpos, got {q.shape}")
        if not np.isfinite(q).all():
            # MuJoCo does not raise on a non-finite reference; it poisons the
            # state, every later number is NaN and the step rate collapses ~10x.
            raise ValueError("[scene_track] non-finite chunk from the generator")

        ref = resample(q, KIMODO_FPS, HEFT_FPS).astype(np.float64)
        self.composite.set_heft_reference(
            ref, reboot=(self.n_chunks == 0) if reboot is None else bool(reboot))
        self.n_chunks += 1
        self._last_ref = ref
        self._ref_tick = 0
        self._ref_t0 = None
        return int(ref.shape[0])

    def reference_points(self, side_bodies=True) -> np.ndarray | None:
        """Where the REFERENCE says the body is right now, in world. [B, 3] or None.

        This is what a ghost draws. It is worth having next to the robot because
        the two failures it separates look identical from the log: a robot BEHIND
        its reference is an execution problem, a reference in the wrong place is a
        generation problem, and ``palm 0.136`` says neither.

        Costs one FK into a scratch MjData per call, so it is a viewer-only
        facility -- never put it in a rollout loop that is being timed.
        """
        ref = getattr(self, "_last_ref", None)
        if ref is None:
            return None
        k = int(np.clip(getattr(self, "_ref_tick", 0), 0, len(ref) - 1))
        # ⚠️ Rebuilt whenever the MODEL changes, not just once. Under hard_reset
        # every episode recompiles the scene, and these rooms resample the floor
        # plan too -- `IThorBathroomApproachToilet` went FloorPlan426 -> 420 ->
        # 405 -> 416 across four resets, and nq with it. A scratch MjData cached
        # from the first scene then raises on the second:
        #     ValueError: could not broadcast input array from shape (117,)
        #     into shape (126,)
        # which killed `--view` on episode 1 while episode 0 looked fine.
        if (getattr(self, "_ghost_data", None) is None
                or self._ghost_data.qpos.shape[0] != self.model.nq):
            self._ghost_data = self.mujoco.MjData(self.model)
            names = [self.mujoco.mj_id2name(
                self.model, self.mujoco.mjtObj.mjOBJ_BODY, b) or ""
                for b in range(self.model.nbody)]
            bids = [b for b, n in enumerate(names) if n.startswith(self.prefix)]
            self._ghost_bids = np.asarray(
                sorted(set(bids) | {int(self.bid_root)}) if bids
                else range(self.model.nbody), dtype=np.int32)
        d = self._ghost_data
        d.qpos[:] = self.data.qpos
        # The reference is [root_pos(3), root_quat wxyz(4), 29 joints] in dataset
        # order; the scene's qpos is in SIM order and may hold objects too, so
        # only the robot's own addresses are overwritten and everything else
        # keeps the live scene's value.
        q = ref[k]
        rid = self.model.body_jntadr[self.bid_root]
        if rid >= 0:
            adr = self.model.jnt_qposadr[rid]
            d.qpos[adr:adr + 3] = q[0:3] + np.array(
                [self.origin_xy[0], self.origin_xy[1], self.floor_z])
            d.qpos[adr + 3:adr + 7] = q[3:7]
        d.qpos[self._q_ds] = q[7:36]
        self.mujoco.mj_forward(self.model, d)
        # ⚠️ ROBOT BODIES ONLY. `d.xpos` is every body in the model, and in a
        # furnished room that is 168 of them -- the ghost then decorates the
        # toilet and the walls with spheres that never move, which reads as "the
        # reference includes the furniture". Only the robot's bodies were driven
        # by the reference above; the rest still hold the live scene's value.
        return d.xpos[self._ghost_bids].copy()

    # ------------------------------------------------------------------ run
    def rollout(self, n_ticks: int, *, track: str | None = "right",
                grip: tuple[float, float] | None = None, on_tick=None,
                record_qpos: bool = True, stop_on_success: bool = True,
                success_hold: int = 0, stop_when=None) -> dict:
        """Step the scene ``n_ticks`` control steps and record what happened.

        One ``env.step`` is one 50 Hz policy tick and ten physics substeps; the
        ratio is asserted rather than assumed, because the whole observation
        history is garbage at 10x and nothing else would notice.

        ``on_tick(i, tr)`` runs after every control step. It is the seam a
        scripted grasp needs: closing on ARRIVAL rather than on schedule means
        the decision is taken from state that only exists between steps, so it
        cannot be hoisted out to chunk granularity.
        """
        if (self.composite.heft_ticks_without_reference() != 0
                and getattr(self.composite._wbt_ctrl, "_replay", None) is None):
            raise RuntimeError(
                f"[scene_track] {self.composite.heft_ticks_without_reference()} ticks ran "
                "before a reference was loaded -- the robot was falling for all of them")

        t0 = self.composite.heft_tick_count()
        if getattr(self, "_ref_t0", None) is None or self._ref_tick == 0:
            self._ref_t0 = t0
        if grip is not None:
            self.set_grip(*grip)

        qpos36, wrist_w, root_w, success_at = [], [], [], None
        success_run, success_run_from = 0, 0
        targets, grips = [], []
        i = -1
        for i in range(int(n_ticks)):
            # Where the reference is, for anything that wants to draw it. Kept in
            # step with the tracker's own cursor rather than counted separately,
            # so a ghost cannot silently drift out of sync with the body.
            self._ref_tick = self.composite.heft_tick_count() - self._ref_t0 \
                if getattr(self, "_ref_t0", None) is not None else i
            self.env.step(self._zero)
            if on_tick is not None:
                on_tick(i, self)
            if record_qpos:
                qpos36.append(self.realized_qpos36())
            if track is not None:
                wrist_w.append(self.wrist_pos(track, frame="world"))
            root_w.append(self.data.xpos[self.bid_root].copy())
            self._min_root_z = min(self._min_root_z, float(self.root_pos_local()[2]))
            # The exported action column, recorded as it is commanded. P3-4 asks
            # whether these alone reproduce the episode, which needs them stored
            # from the same place the recorder reads them.
            tq = getattr(self.composite, "_cached_target_q", None)
            if tq is not None:
                targets.append(np.asarray(tq, dtype=np.float32).copy())
            # The grip triggers are a SECOND action channel. action.wbc is 29
            # body joints; the 14 finger DoF are driven from these two scalars and
            # appear nowhere in it. An episode replayed from action.wbc alone has
            # an open hand where the original had a closed one -- invisible in an
            # empty arena, and the reason h10 failed the moment there was an
            # object to hold.
            grips.append((float(getattr(self.composite, "_left_trigger", 0.0)),
                          float(getattr(self.composite, "_right_trigger", 0.0))))
            # ⚠️ SUCCESS HAS TO HOLD, not just happen. `_check_success` is
            # evaluated every tick, and crediting the first tick it fires counts
            # a hand that sweeps through the target, a lid that bounces past its
            # threshold, and an object that is momentarily in a region on its way
            # out of it. `success_hold` is the number of CONSECUTIVE ticks the
            # predicate must stay true; `success_at` is then stamped at the tick
            # the run STARTED, so the reported time still means "when it got
            # there" rather than "when we finally believed it".
            #
            # 0 reproduces the old behaviour exactly, because every earlier
            # number in this project was measured that way.
            if self._check_success():
                if success_run == 0:
                    success_run_from = i
                success_run += 1
                if success_at is None and success_run > int(success_hold):
                    success_at = success_run_from
                    self._success_seen = True
                    if stop_on_success:
                        break
            else:
                success_run = 0
            # ⚠️ AN ARRIVED ROBOT MUST STOP WALKING AT THE THING IT ARRIVED AT.
            # A chunk is a 3 s reference and it used to be executed in full, so a
            # robot that entered the target's radius at tick 60 spent the next 90
            # leaning into the furniture -- the ask is aimed at (or clamped to)
            # the target itself, and nothing in the tracker knows the job is
            # done. `stop_on_success` could not serve here: it is tied to the
            # task's own `_check_success`, while what has to stop the walk is
            # whatever the WAYPOINT counts as arrival, which is a different
            # predicate as soon as `--success-radius` is given.
            if stop_when is not None and stop_when():
                break

        n = i + 1 if n_ticks else 0
        ticks = self.composite.heft_tick_count() - t0
        if ticks != n:
            raise RuntimeError(
                f"[scene_track] {n} env.step() produced {ticks} policy ticks; the "
                "50 Hz gate is not 1:1 with the control step")

        root_w = np.asarray(root_w)
        return {
            "target_q_29": np.asarray(targets) if targets else None,      # action.wbc
            "grip": np.asarray(grips) if grips else None,                 # the other channel
            "realized_qpos36": np.asarray(qpos36) if qpos36 else None,   # local frame
            "wrist_world": np.asarray(wrist_w) if wrist_w else None,
            "root_world": root_w,
            "root_z_local": root_w[:, 2] - self.floor_z if len(root_w) else root_w,
            "ticks": ticks,
            "success_at": success_at,
            "fell": self.fell(),
            "ref_done": bool(self.composite.heft_reference_done),
        }

    # ------------------------------------------------------------------ replay
    def snapshot(self):
        """The full sim state, for restoring an episode's exact initial value.

        ``sim.get_state()`` is what the batch path records instead of frames
        (P4-2), so a reproducibility check that starts from anything else would
        be testing a different episode than the one that will be exported.

        The solver's warm-start (``qacc_warmstart``) is captured alongside it,
        because the wrapper's state does not carry it and two runs from the
        "same" state would then seed the contact solve differently. It was not
        what broke h10 -- that was ``hard_reset`` resampling the scene between
        record and replay -- but it costs one array and removes a whole class of
        "nearly reproducible" from the answer.

        ⚠️ **Restoring is only meaningful into the SAME model.** With
        ``hard_reset=True`` an ``env.reset()`` rebuilds the MjModel and resamples
        the objects, so writing an old qpos into it replays a different episode.
        Measured: joint drift 3.1e-2 with a hard reset, exactly 0.0 with a soft
        one. The batch and re-render paths (P4-2, P4-3) inherit this.
        """
        return (self.env.sim.get_state(), self.data.qacc_warmstart.copy())

    def restore(self, snap) -> None:
        state, warm = snap if isinstance(snap, tuple) else (snap, None)
        self.env.sim.set_state(state)
        if warm is not None:
            self.data.qacc_warmstart[:] = warm
        self.env.sim.forward()
        self._start_root_z = float(self.root_pos_local()[2])
        self._min_root_z = self._start_root_z
        self._success_seen = False

    def replay(self, targets_q29, n_ticks: int | None = None, grip=None,
               on_tick=None, **kw) -> dict:
        """Drive from a recorded action stream. Peer of ``rollout``.

        The tracker is bypassed entirely: no reference is needed and none is
        used, which is the point -- if the actions reproduce the episode then the
        actions are a sufficient label, and if they do not then the dataset is
        recording something the actions do not determine.

        ``grip`` is the second half of the action and is not optional when
        anything is being held. Omitting it replays a body with an open hand.
        """
        t = np.asarray(targets_q29, dtype=np.float64)
        g = None if grip is None else np.asarray(grip, dtype=np.float64).reshape(-1, 2)

        def hook(i, tr):
            if g is not None:
                l, r = g[min(i, len(g) - 1)]
                tr.set_grip(l, r)
            if on_tick is not None:
                on_tick(i, tr)

        # Set the first grip BEFORE the first step: the hook runs after env.step,
        # so without this the opening tick is always played with an open hand.
        if g is not None:
            self.set_grip(*g[0])
        self.composite.set_heft_replay(t)
        try:
            return self.rollout(n_ticks if n_ticks is not None else len(t),
                                on_tick=hook, **kw)
        finally:
            self.composite.set_heft_replay(None)

    def set_grip(self, left: float, right: float) -> None:
        """Scripted grasp channel. 0 = open, 1 = closed.

        The tracker controls 29 body DoF; the 14 finger DoF are an independent
        channel the composite drives from these two triggers, so a scripted grasp
        needs no new plumbing and cannot perturb what the tracker is doing.
        """
        self.composite._left_trigger = float(np.clip(left, 0.0, 1.0))
        self.composite._right_trigger = float(np.clip(right, 0.0, 1.0))

    def _check_success(self) -> bool:
        fn = getattr(self.env, "_check_success", None)
        if fn is None:
            return False
        try:
            return bool(fn())
        except Exception:  # noqa: BLE001 -- a scene without placed objects
            return False

    def fell(self, drop: float = 0.30) -> bool:
        """Did the root ever drop ``drop`` below where it started.

        Against the MINIMUM over the episode, not the current height. A stumble
        that recovers is still a stumble, and judging on the last frame alone
        passed an episode whose root went 0.798 -> 0.341 -> back up as "did not
        fall" -- which is exactly the kind of episode a data pipeline must not
        keep.
        """
        return (self._start_root_z - self._min_root_z) > drop

    @property
    def min_root_z(self) -> float:
        return self._min_root_z

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:  # noqa: BLE001
            pass
