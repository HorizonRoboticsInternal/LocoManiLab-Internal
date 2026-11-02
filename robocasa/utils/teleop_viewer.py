"""Operator ergonomics for the MuJoCo window during G1 teleop / data collection.

Two things only bite when a human is driving, and robosuite handles neither:

* **The camera loses the robot on every reset.** ``env.reset()`` unconditionally
  tears the mjviewer down (robosuite ``environments/base.py``: "always terminate
  mjviewer"), and the replacement window comes up on
  ``MjviewerRenderer.DEFAULT_FREE_CAM`` -- lookat = the *world origin*, 2 m out.
  In a room-scale scene (iTHOR FloorPlans, kitchen layouts) the origin is
  nowhere near the robot and is regularly on the far side of a wall, so every
  reset drops the operator somewhere they have to re-orbit from.
  ``TeleopViewer`` re-frames the free camera on the robot after each reset and
  carries the operator's own orbit across it.

  It deliberately does **not** use MuJoCo's tracking camera (it did, briefly).
  ``mjCAMERA_TRACKING`` drives ``cam.lookat`` toward the tracked body's *subtree
  centre of mass*, and a walking humanoid's torso COM bobs and sways with every
  footfall -- so the entire world visibly shakes, which is worse to operate than
  a camera that occasionally needs re-centring. The camera stays a plain free
  camera owned by the operator's mouse; ``C`` re-frames it on demand.

* **The task instruction, the active controller and the control mode scroll
  away.** All three are printed when they change and are immediately buried by
  the per-step FPS/status lines, so mid-episode there is nothing on screen
  telling the operator what the task is, which controller is driving the robot,
  or whether their last chord actually engaged. ``TeleopViewer`` pins all three
  -- plus live success/distance -- to the top-left of the MuJoCo window for the
  whole episode.

* **The controls themselves are unmemorable and differ per controller.** The
  key/chord list is printed once at startup and is gone by the time it is
  needed; the operator is usually in a headset, cannot see the terminal at all,
  and each controller means something different by the same chord (A+X is
  PLANNER<->POSE on SONIC, IDLE<->tracking on heft/scalebfm, and nothing at all
  on AMO, whose whole mapping is joystick-and-button-based). So the controller
  supplies its own ``hud_cheatsheet`` -- the state-machine transitions plus its
  movement bindings, in two aligned columns -- and ``TeleopViewer`` pins it to
  the top-right of the window (``H`` toggles it).

* **MuJoCo's own keyboard shortcuts fire on those same keys**, and they change
  what is rendered: Q (the discard/reset key) toggles simulate's *camera*
  visualization, which draws a cylinder per camera, and nearly every other
  teleop key is bound to some visualization flag or geom group too. So
  ``TeleopViewer`` pins the viewer's ``MjvOption`` every frame and undoes those
  toggles (see ``_PINNED_OPT_FIELDS``).

The HUD uses the passive viewer's **overlay text** (``handle.set_texts``, a
list of ``(fontscale, gridpos, left_column, right_column)``). That is MuJoCo's
own on-screen text layer: it renders at a real selectable font size, aligns two
columns for free, and is drawn over the scene, so nothing can occlude it.

It replaces an earlier hand-rolled version that appended ``mjGEOM_LABEL`` geoms
to ``user_scn`` positioned in camera space, with a black box geom behind them
for contrast. That worked, but label geoms are drawn at one fixed pixel size no
matter how the geom is scaled -- there is no way to make that text bigger --
and the whole layout had to be recomputed from fovy/aspect/eye every frame.
"""

import math

import mujoco
import numpy as np
from termcolor import colored

# Camera framing applied after every reset. The operator's own orbit overrides
# these (see TeleopViewer.remember_orbit); 'C' restores them.
DEFAULT_DISTANCE = 3.0  # m from the robot
DEFAULT_ELEVATION = -15.0  # deg, looking slightly down
DEFAULT_AZIMUTH_OFFSET = 0.0  # deg relative to the robot's facing (0 = behind it)

# HUD font. ``set_texts``'s first field is an **mjtFont** (NORMAL / SHADOW /
# BIG), not the mjtFontScale its docstring names -- the pixel size itself lives
# in the render context's fontScale, which the passive viewer owns and does not
# expose. So BIG (measured ~2x NORMAL) is the largest size reachable from here,
# and it is the default: this HUD exists to be read from across the room while
# the operator is in a headset, not to be squinted at.
_HUD_FONTS = (
    ("big", mujoco.mjtFont.mjFONT_BIG),
    ("shadow", mujoco.mjtFont.mjFONT_SHADOW),
    ("normal", mujoco.mjtFont.mjFONT_NORMAL),
)
_HUD_FONT_DEFAULT = 0  # index into _HUD_FONTS -> big
# The left column is a fixed-width label, so the right column lines up.
_HUD_WRAP = 44  # wrap the instruction; the big font fits fewer characters

# The cheat sheet is a dozen-odd rows against the status HUD's four, so it is
# drawn one font size down from whatever the status block is using -- at BIG it
# would take most of the window height.
_CHEATSHEET_FONT_OFFSET = 1

# Body names to frame the camera on, best first. torso_link sits at chest height
# (~0.81 m) and is the natural thing to center on; the G1's pelvis/root body
# origin is low and off-center, so it is only a fallback.
_TRACK_BODY_SUFFIXES = ("torso_link", "pelvis", "base")

# --- MuJoCo's built-in shortcuts vs. ours ---------------------------------- #
# The passive viewer feeds every keypress to simulate's own UI (the Rendering
# and "Group enable" sections) as well as to our ``key_callback``, and it does
# so whether or not that UI is on screen -- ``show_right_ui=False`` hides the
# panel, not its shortcuts. The letters simulate binds there are letters teleop
# already owns:
#
#   Q      mjVIS_CAMERA <- our discard/reset key. Camera visualization draws a
#                          box + a *cylinder* per camera, so every Q sprinkled
#                          cylinders through the scene and left them there (the
#                          flag is a toggle, so the next Q cleared them again).
#   C T F H             contact point / transparent / contact force / convex
#                          hull -- i.e. all four TeleopViewer keys below.
#   Z X A D Y I J       light, texture, auto-connect, mjVIS_STATIC (the whole
#                          static kitchen), rangefinder, inertia, joint --
#                          the keyboard-PICO gripper and arm-jog keys.
#   B , ; ' E P         perturb force, activation, skin, scale inertia,
#                          equality, contact split -- backend mode chords.
#   0-5                 geom groups: "1" hides every visual geom in the scene,
#                          and 0-9 is the locomotion-mode selector.
#
# The flags live in the ``MjvOption`` that ``launch_passive`` shares with
# simulate (``Handle.opt`` *is* that struct), so we snapshot it when we adopt a
# window and rewrite it every frame: simulate's toggle lands, we undo it before
# the next sync. The window redraws faster than we sync, so a stray toggle can
# still flash for a frame -- it does not stick, which is the point.
_PINNED_OPT_FIELDS = (
    "flags",  # mjVIS_* visualization flags
    "geomgroup",
    "sitegroup",
    "jointgroup",
    "tendongroup",
    "actuatorgroup",
    "label",
    "frame",
)

KEY_RECENTER = ord("C")
KEY_TOGGLE_HUD = ord("T")
KEY_FONT = ord("F")
KEY_TOGGLE_CHEATSHEET = ord("H")
KEY_HELP = (
    "  C          - re-center the camera on the robot (after Esc / getting lost)\n"
    "  T          - toggle the on-screen task/mode text\n"
    "  H          - toggle the on-screen controls cheat sheet (top right)\n"
    "  F          - cycle the on-screen text font (big / shadow / normal)\n"
)


def _wrap(text, width=_HUD_WRAP):
    """Greedy word wrap; never emits a line longer than ``width`` unless a
    single word is."""
    lines, cur = [], ""
    for word in str(text).split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        lines.append(cur)
    return lines


class TeleopViewer:
    """Keeps the mjviewer's free camera on the robot and draws the task HUD.

    Lifecycle, mirroring the teleop loop:

        view = TeleopViewer(env)
        view.frame_on_robot()          # before the first env.render()
        while True:
            env.render()
            view.sync(status="...")    # every frame, after env.render()
            ...
            view.remember_orbit()      # before env.reset()
            env.reset()
            view.frame_on_robot()      # after env.reset()

    Every method is defensive: a missing viewer, an unexpected mujoco build or a
    robot without the expected bodies disables the feature with one printed line
    rather than taking the collection session down.
    """

    def __init__(
        self,
        env,
        distance=DEFAULT_DISTANCE,
        elevation=DEFAULT_ELEVATION,
        show_hud=True,
        show_cheatsheet=True,
    ):
        self.env = env
        self.distance = distance
        self.elevation = elevation
        self.azimuth_offset = DEFAULT_AZIMUTH_OFFSET
        self.show_hud = show_hud
        self.show_cheatsheet = show_cheatsheet
        self.instruction = ""
        self.controller = ""
        self.mode = ""
        self.cheatsheet = ()
        self._font_idx = _HUD_FONT_DEFAULT
        self._configured_handle = None  # the passive handle we last set up
        self._track_bid = None
        self._disabled = False
        self._opt_baseline = None  # rendering options as the window opened them
        self._opt_baseline_handle = None
        self._opt_pin_disabled = False

    # ------------------------------------------------------------------ #
    # plumbing
    # ------------------------------------------------------------------ #
    @property
    def _renderer(self):
        """The robosuite MjviewerRenderer (recreated by every env.reset())."""
        return getattr(self.env, "viewer", None)

    @property
    def _handle(self):
        """The live ``mujoco.viewer`` passive handle, or None before the window
        is launched (MjviewerRenderer launches it lazily on the first render)."""
        return getattr(self._renderer, "viewer", None)

    def _disable(self, exc, what):
        self._disabled = True
        print(
            colored(f"[viewer] {what} disabled ({type(exc).__name__}: {exc})", "yellow")
        )

    def _body_id(self):
        """Body to center on, resolved fresh each reset (ids move when the
        model is rebuilt)."""
        model = self.env.sim.model
        prefix = self.env.robots[0].robot_model.naming_prefix
        names = [prefix + s for s in _TRACK_BODY_SUFFIXES]
        root = getattr(self.env.robots[0].robot_model, "root_body", None)
        if root:
            names.append(root)
        for name in names:
            try:
                return model.body_name2id(name)
            except Exception:
                continue
        return None

    def _robot_yaw_deg(self, bid):
        """Facing of the tracked body, in degrees, as a camera azimuth. The
        camera then sits *behind* the robot looking the way it walks."""
        xmat = np.array(self.env.sim.data.body_xmat[bid]).reshape(3, 3)
        forward = xmat[:, 0]  # body +x is the G1's forward
        return math.degrees(math.atan2(forward[1], forward[0]))

    def _tracked_point(self, bid):
        """Where MuJoCo's tracking camera actually centers: the *subtree* center
        of mass of the tracked body, not its frame origin. Framing the
        pre-launch camera on the same point keeps the window from jumping the
        moment tracking takes over (for a humanoid torso the two differ by most
        of a torso height)."""
        data = self.env.sim.data
        com = getattr(data, "subtree_com", None)
        return np.array(com[bid] if com is not None else data.body_xpos[bid])

    # ------------------------------------------------------------------ #
    # camera
    # ------------------------------------------------------------------ #
    def remember_orbit(self):
        """Capture the operator's current orbit so the next framing reuses it.
        Call *before* env.reset(), while the old viewer still exists. The
        azimuth is stored relative to the robot's facing, so "over the left
        shoulder" stays over the left shoulder even if the next episode spawns
        the robot pointing somewhere else."""
        if self._disabled:
            return
        handle = self._handle
        if handle is None or self._track_bid is None:
            return
        try:
            self.distance = float(handle.cam.distance)
            self.elevation = float(handle.cam.elevation)
            yaw = self._robot_yaw_deg(self._track_bid)
            self.azimuth_offset = (
                float(handle.cam.azimuth) - yaw + 180.0
            ) % 360.0 - 180.0
        except Exception as e:
            self._disable(e, "camera framing")

    def frame_on_robot(self, reset_orbit=False):
        """Point the free camera at the robot. Call right after env.reset() and
        before the next render: the renderer has not launched its window yet, so
        this writes ``camera_config`` and the window opens already framed (no
        one-frame jump from the world origin).

        One-shot, by design -- the camera does not follow afterwards (see the
        module docstring), so the operator's orbit is never fought over."""
        if self._disabled:
            return
        try:
            if reset_orbit:
                self.distance = DEFAULT_DISTANCE
                self.elevation = DEFAULT_ELEVATION
                self.azimuth_offset = DEFAULT_AZIMUTH_OFFSET
            self._track_bid = self._body_id()
            if self._track_bid is None:
                self._disable(
                    RuntimeError("no torso/pelvis/root body on robot0"),
                    "camera framing",
                )
                return
            lookat = self._tracked_point(self._track_bid)
            azimuth = self._robot_yaw_deg(self._track_bid) + self.azimuth_offset
            renderer = self._renderer
            if renderer is not None:
                renderer.camera_config = {
                    "lookat": lookat.tolist(),
                    "distance": self.distance,
                    "azimuth": azimuth,
                    "elevation": self.elevation,
                }
            handle = self._handle
            if handle is not None:
                # Window already open (the 'C' key path) -- apply immediately.
                with handle.lock():
                    handle.cam.lookat = lookat
                    handle.cam.distance = self.distance
                    handle.cam.azimuth = azimuth
                    handle.cam.elevation = self.elevation
                self._adopt(handle)
        except Exception as e:
            self._disable(e, "camera framing")

    def _adopt(self, handle):
        """Take ownership of a freshly launched window.

        Explicitly a free camera: MuJoCo's own ``Esc`` binding and any earlier
        session could have left ``cam.type`` elsewhere, and a stale
        ``trackbodyid`` points into a model that no longer exists after a reset.
        Setting it here (rather than leaving it to the default) is what
        guarantees the camera never follows the robot."""
        with handle.lock():
            handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            handle.cam.trackbodyid = -1
        self._configured_handle = handle
        self._snapshot_render_opts(handle)

    # ------------------------------------------------------------------ #
    # rendering options (see _PINNED_OPT_FIELDS)
    # ------------------------------------------------------------------ #
    def _snapshot_render_opts(self, handle):
        """Record the rendering options the fresh window came up with -- the
        MuJoCo defaults, minus whatever the renderer changed on launch (robosuite
        hides geom group 0, the collision geoms). This is the state teleop keys
        must not be able to change.

        Once per window, never on re-adoption: ``_adopt`` also runs on the 'C'
        re-center key, and 'C' is itself one of the shortcuts we are undoing --
        re-snapshotting there would bake the stray toggle in as the baseline."""
        if self._opt_pin_disabled or handle is self._opt_baseline_handle:
            return
        try:
            self._opt_baseline = {
                name: np.array(getattr(handle.opt, name), copy=True)
                for name in _PINNED_OPT_FIELDS
            }
            self._opt_baseline_handle = handle
        except Exception as e:
            self._opt_pin_disabled = True
            self._opt_baseline = None
            print(
                colored(
                    f"[viewer] rendering-option pin disabled "
                    f"({type(e).__name__}: {e}); MuJoCo's own C/T/F/H/Q/digit "
                    "shortcuts will toggle visualization flags",
                    "yellow",
                )
            )

    def _pin_render_opts(self, handle):
        """Undo any rendering-option change simulate's built-in shortcuts made
        since the last frame. Cheap: an array compare per field, and the write
        only happens on the frames where a key actually landed."""
        if self._opt_baseline is None:
            return
        stale = {
            name: want
            for name, want in self._opt_baseline.items()
            if not np.array_equal(np.asarray(getattr(handle.opt, name)), want)
        }
        if not stale:
            return
        with handle.lock():
            for name, want in stale.items():
                field = getattr(handle.opt, name)
                if np.ndim(field) == 0:
                    setattr(handle.opt, name, want.item())
                else:
                    field[:] = want

    # ------------------------------------------------------------------ #
    # HUD
    # ------------------------------------------------------------------ #
    def set_instruction(self, text):
        self.instruction = text or ""

    def set_controller(self, text):
        """The controller line: which whole-body controller is actually driving
        the robot (``sonic_manager``, ``sim2real_heft``, ...). It is fixed for a
        session, but the session is long and the startup banner that named it
        scrolled off minutes ago -- and every mode chord below means something
        different per controller, so it is the header the MODE line is read
        against."""
        self.controller = text or ""

    def set_mode(self, text):
        """The control-mode line: which state the operator's chords have the
        robot in. Pinned next to the task because the two questions mid-episode
        are always "what am I doing" and "is it listening"."""
        self.mode = text or ""

    def set_cheatsheet(self, rows):
        """The controls cheat sheet: an iterable of ``(key, meaning)`` pairs,
        supplied by whichever composite controller is driving the robot (its
        ``hud_cheatsheet``). A pair with an empty key is a section heading.

        Re-set every frame like the other HUD fields, because the answer can
        change under the operator: ``env.reset()`` rebuilds the composite, and
        AMO's sheet depends on a config flag read at load time."""
        self.cheatsheet = tuple(rows) if rows else ()

    def _hud_rows(self, status):
        """(left, right) column pairs for the overlay, top to bottom."""
        rows = []
        if self.controller:
            rows.append(("CTRL", self.controller))
        if self.mode:
            rows.append(("MODE", self.mode))
        for i, line in enumerate(_wrap(self.instruction)):
            rows.append(("TASK" if i == 0 else "", line))
        if status:
            for i, line in enumerate(_wrap(status)):
                rows.append(("" if i else "STATUS", line))
        return rows

    @staticmethod
    def _text_block(font, gridpos, rows):
        """One overlay entry from ``(left, right)`` rows. MuJoCo joins each
        column on newlines itself, so a single (font, gridpos, left, right)
        entry renders the whole table aligned. Separate entries at the *same*
        gridpos would stack on top of each other instead -- which is why the
        status block and the cheat sheet sit in different grid corners."""
        return (
            font,
            gridpos,
            "\n".join(left for left, _ in rows),
            "\n".join(right for _, right in rows),
        )

    def sync(self, status=None):
        """Per-frame upkeep, called right after ``env.render()``: adopt a
        freshly launched viewer and refresh the HUD text."""
        if self._disabled:
            return
        handle = self._handle
        if handle is None:
            return
        try:
            if handle is not self._configured_handle:
                self._adopt(handle)
            self._pin_render_opts(handle)
            blocks = []
            rows = self._hud_rows(status) if self.show_hud else []
            if rows:
                blocks.append(
                    self._text_block(
                        _HUD_FONTS[self._font_idx][1],
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        rows,
                    )
                )
            if self.show_cheatsheet and self.cheatsheet:
                font_idx = min(
                    self._font_idx + _CHEATSHEET_FONT_OFFSET, len(_HUD_FONTS) - 1
                )
                blocks.append(
                    self._text_block(
                        _HUD_FONTS[font_idx][1],
                        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                        self.cheatsheet,
                    )
                )
            if not blocks:
                handle.clear_texts()
                return
            handle.set_texts(blocks)
        except Exception as e:
            self._disable(e, "viewer HUD")

    # ------------------------------------------------------------------ #
    # keyboard
    # ------------------------------------------------------------------ #
    def handle_keypress(self, keycode):
        """Returns True if the key was consumed here."""
        if keycode == KEY_RECENTER:
            self.frame_on_robot(reset_orbit=True)
            print("[viewer] camera re-centered on the robot")
            return True
        if keycode == KEY_TOGGLE_HUD:
            self.show_hud = not self.show_hud
            print(f"[viewer] on-screen text {'on' if self.show_hud else 'off'}")
            return True
        if keycode == KEY_TOGGLE_CHEATSHEET:
            self.show_cheatsheet = not self.show_cheatsheet
            print(
                f"[viewer] controls cheat sheet "
                f"{'on' if self.show_cheatsheet else 'off'}"
            )
            return True
        if keycode == KEY_FONT:
            self._font_idx = (self._font_idx + 1) % len(_HUD_FONTS)
            print(f"[viewer] text font: {_HUD_FONTS[self._font_idx][0]}")
            return True
        return False
