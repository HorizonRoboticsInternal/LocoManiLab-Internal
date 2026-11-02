"""SONIC-style PICO button mode manager.

Full 5-mode state machine, mirroring
``gear_sonic/scripts/pico_manager_thread_server.py`` (lines 1881-2039).

Modes (per SONIC's state diagram):

  OFF                  -- policy disengaged
  POSE                 -- full-body SMPL teleop (encoder mode_id=2)
  PLANNER              -- kPlanner full body (encoder mode_id=0)
  PLANNER_FROZEN_UPPER -- kPlanner lower body, upper body frozen at last snapshot
  POSE_PAUSE           -- POSE held still while the left menu button is down
  PLANNER_VR_3PT       -- kPlanner lower body, VR 3-point upper body (mode_id=1)

Buttons:

  A+B+X+Y (start_combo): toggle OFF <-> PLANNER (engage / emergency stop).
                         First press from OFF triggers CALIB_FULL.
  A+X:    Chain 2 toggle    POSE <-> PLANNER  (also VR_3PT -> POSE)
  B+Y:    Chain 1 toggle    POSE <-> PLANNER_FROZEN_UPPER  (also VR_3PT -> POSE).
                            Entering PLANNER_FROZEN_UPPER triggers
                            FREEZE_SNAPSHOT so the composite can capture the
                            current upper-body pose.
  L-Stick Click:           PLANNER       <-> PLANNER_VR_3PT (triggers CALIB_WRISTS)
                           PLANNER_FROZEN_UPPER <-> PLANNER_VR_3PT (also CALIB)
                           Returning from VR_3PT goes back to the parent.
  L-Menu (held):           POSE -> POSE_PAUSE while down, POSE again on release.
                           A level, not an edge -- see ``tick``.

Calibration events emitted by transitions:

  CALIB_FULL    -- head + wrists re-zeroed vs all-zero reference pose.
                   First OFF -> PLANNER.
  CALIB_WRISTS  -- wrists re-zeroed vs robot's current pose.  Every entry
                   into VR_3PT.
  FREEZE_SNAPSHOT -- composite should capture upper-body 17 joints into
                     its snapshot buffer.  Every entry into
                     PLANNER_FROZEN_UPPER.

Profiles
--------

Which chords a controller *listens* to, selected by ``profile``.  The
transition table itself is shared and never varies -- a profile only decides
which rising edges are allowed to reach it, so ``"sonic"`` (the default) is
byte-identical to having no profile at all.

  "sonic"    -- everything above.  SONIC and WholeBodyG1Composite.
  "tracking" -- A+B+X+Y and A+X only (no L-menu, so no POSE_PAUSE), giving
                OFF <-> PLANNER <-> POSE and nothing else.  Used by the GMR
                motion-tracking backends (heft/scalebfm), which have neither a
                kPlanner nor a VR-3pt controller: for them PLANNER is simply
                "engaged, holding the current pose" (IDLE) and A+X is the
                tracking on/off switch.  B+Y and the L-stick are muted rather
                than remapped, because PLANNER_FROZEN_UPPER has no rising_ax
                case -- an operator who reached it with B+Y could not get back
                to POSE with the chord they had been using.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

#: chord name -> whether ``tick`` acts on its rising edge, per profile.
_PROFILE_CHORDS = {
    "sonic": frozenset({"start", "ax", "by", "click", "menu"}),
    "tracking": frozenset({"start", "ax"}),
}


class ManagerMode(IntEnum):
    """Values match StreamMode in SONIC's pico_manager_thread_server.py.

    They did NOT until 2026-08-26: VR_3PT used to sit at 4, which is upstream's
    POSE_PAUSE, and upstream's VR 3-point mode is 5. The two collectors wrote
    the same ``teleop.stream_mode`` column with different meanings for the same
    number -- see recording.features SCHEMA_VERSION bullet 9.
    """

    OFF = 0
    POSE = 1
    PLANNER = 2
    PLANNER_FROZEN_UPPER = 3
    POSE_PAUSE = 4
    PLANNER_VR_3PT = 5

    #: Pre-2026-08-26 spelling. An IntEnum alias, so `is` comparisons and
    #: `.name` all resolve to PLANNER_VR_3PT -- existing callers (notably
    #: tele_humanoid_benchmark's robocasa/g1_host/wbt.py) keep working, and
    #: they move with the value rather than being pinned to the old 4.
    VR_3PT = 5


class CalibrationEvent(IntEnum):
    """Side-effect events emitted on mode transitions."""

    NONE = 0
    CALIB_FULL = 1
    CALIB_WRISTS = 2
    FREEZE_SNAPSHOT = 3


class ModeManager:
    """Button-driven state machine for sonic_manager mode.

    Call :meth:`tick` once per control tick with the latest button snapshot
    from ``PicoVRSource.get_latest()['buttons']``.  Returns (new_mode, event).
    """

    def __init__(
        self,
        initial_mode: ManagerMode = ManagerMode.OFF,
        verbose: bool = True,
        profile: str = "sonic",
    ):
        self._mode = initial_mode
        self._verbose = verbose
        if profile not in _PROFILE_CHORDS:
            raise ValueError(
                f"unknown ModeManager profile {profile!r}; "
                f"expected one of {sorted(_PROFILE_CHORDS)}"
            )
        self._profile = profile
        self._chords = _PROFILE_CHORDS[profile]

        # Tracks which parent VR_3PT was entered from, so L-Stick Click goes
        # back to the right place (PLANNER vs PLANNER_FROZEN_UPPER).
        self._vr3pt_parent: ManagerMode = ManagerMode.PLANNER

        # Edge-detection state for chord buttons.
        self._prev_start_combo = False
        self._prev_ax = False
        self._prev_by = False
        self._prev_left_axis_click = False

    @property
    def current(self) -> ManagerMode:
        return self._mode

    @property
    def vr3pt_parent(self) -> ManagerMode:
        return self._vr3pt_parent

    @property
    def profile(self) -> str:
        return self._profile

    # ------------------------------------------------------------------ #
    def tick(self, buttons: Optional[dict]) -> tuple[ManagerMode, CalibrationEvent]:
        """Advance the state machine by one tick.

        Args:
            buttons: snapshot from ``PicoVRSource.get_latest()['buttons']``,
                or ``None`` if no PICO data this tick.

        Returns:
            (new_mode, event).  ``new_mode`` equals ``self.current`` after
            this call.  ``event`` describes any side-effect the composite
            controller should execute.
        """
        if buttons is None:
            return self._mode, CalibrationEvent.NONE

        a = bool(buttons.get("a", False))
        b = bool(buttons.get("b", False))
        x = bool(buttons.get("x", False))
        y = bool(buttons.get("y", False))
        left_click = bool(buttons.get("left_axis_click", False))
        # POSE_PAUSE is a LEVEL, not an edge: upstream enters while the left
        # menu button is held and leaves the moment it is released
        # (pico_manager_thread_server.py:2038 / :2053). Muting it for a profile
        # therefore means "never enter", which is why it is read through the
        # profile gate rather than latched like the chords below.
        left_menu = bool(buttons.get("left_menu", False)) and "menu" in self._chords

        start_combo = a and b and x and y
        # AX / BY are deliberately NOT guarded against the full chord — this
        # mirrors SONIC (pico_manager_thread_server.py:1917-1923).  The guard
        # ``not (b and y)`` looks safer but is actively harmful: during an
        # A+B+X+Y hold it forces ``ax`` False, so ``prev_ax`` stays False, and
        # when the operator releases the chord with B/Y lifting a tick before
        # A/X, ``ax`` rises (a&x & not b&y) and spuriously fires PLANNER->POSE
        # — exactly the "engage and it jumps to POSE" symptom.  Leaving ``ax``
        # unguarded means it is already True throughout the chord (so
        # ``prev_ax`` is True and no rising edge fires on release), while the
        # start_combo branch is still reached first because every state checks
        # ``rising_start`` before ``rising_ax`` / ``rising_by``.
        ax = a and x
        by = b and y

        rising_start = start_combo and not self._prev_start_combo
        rising_ax = ax and not self._prev_ax
        rising_by = by and not self._prev_by
        rising_click = left_click and not self._prev_left_axis_click

        # Mute the chords this profile does not use (see "Profiles" above).
        # Only the rising edges are suppressed, never the ``_prev_*`` latches
        # below -- those must keep tracking the real button state so that
        # switching profiles, or a chord that is muted here but meaningful to
        # another consumer of the same snapshot, cannot desync them.
        rising_start = rising_start and "start" in self._chords
        rising_ax = rising_ax and "ax" in self._chords
        rising_by = rising_by and "by" in self._chords
        rising_click = rising_click and "click" in self._chords

        new_mode = self._mode
        event = CalibrationEvent.NONE

        if self._mode == ManagerMode.OFF:
            if rising_start:
                new_mode = ManagerMode.PLANNER
                event = CalibrationEvent.CALIB_FULL

        elif self._mode == ManagerMode.PLANNER:
            if rising_start:
                new_mode = ManagerMode.OFF
            elif rising_ax:
                new_mode = ManagerMode.POSE
            elif rising_click:
                new_mode = ManagerMode.PLANNER_VR_3PT
                self._vr3pt_parent = ManagerMode.PLANNER
                event = CalibrationEvent.CALIB_WRISTS

        elif self._mode == ManagerMode.POSE:
            if rising_start:
                new_mode = ManagerMode.OFF
            elif rising_ax:
                new_mode = ManagerMode.PLANNER  # Chain 2 exit
            elif rising_by:
                new_mode = ManagerMode.PLANNER_FROZEN_UPPER
                event = CalibrationEvent.FREEZE_SNAPSHOT
            elif left_menu:
                new_mode = ManagerMode.POSE_PAUSE

        elif self._mode == ManagerMode.POSE_PAUSE:
            if rising_start:
                new_mode = ManagerMode.OFF
            elif not left_menu:
                new_mode = ManagerMode.POSE

        elif self._mode == ManagerMode.PLANNER_FROZEN_UPPER:
            if rising_start:
                new_mode = ManagerMode.OFF
            elif rising_by:
                new_mode = ManagerMode.POSE  # Chain 1 exit
            elif rising_click:
                new_mode = ManagerMode.PLANNER_VR_3PT
                self._vr3pt_parent = ManagerMode.PLANNER_FROZEN_UPPER
                event = CalibrationEvent.CALIB_WRISTS

        elif self._mode == ManagerMode.PLANNER_VR_3PT:
            if rising_start:
                new_mode = ManagerMode.OFF
            elif rising_click:
                # Return to whichever parent we came from.
                new_mode = self._vr3pt_parent
            elif rising_ax:
                new_mode = ManagerMode.POSE  # Chain 2 exit
            elif rising_by:
                new_mode = ManagerMode.POSE  # Chain 1 exit

        # ------ commit edges ------
        self._prev_start_combo = start_combo
        self._prev_ax = ax
        self._prev_by = by
        self._prev_left_axis_click = left_click

        if new_mode != self._mode:
            if self._verbose:
                evt_str = (
                    f" (event={event.name})" if event != CalibrationEvent.NONE else ""
                )
                print(f"[ModeManager] {self._mode.name} -> {new_mode.name}{evt_str}")
            self._mode = new_mode

        return self._mode, event

    # ------------------------------------------------------------------ #
    def force_mode(self, mode: ManagerMode) -> None:
        """Force a specific mode (e.g. from keyboard).  Emits no event."""
        if mode != self._mode:
            if self._verbose:
                print(f"[ModeManager] forced {self._mode.name} -> {mode.name}")
            self._mode = mode
