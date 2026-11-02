"""
Teleop a robot around a kitchen-free floor scene: the bare EmptyEnv (one native
MuJoCo ground plane, nothing else), the free-roam LivingRoomEnv (furnished room, no
task), or a living-room task like TidyLivingRoom. These are independent
ManipulationEnv subclasses. Pass ``--env`` to pick one directly, or omit it to
choose from an interactive G1 menu. LivingRoom envs
accept ``--layout`` / ``--style`` to select a specific scene.

G1 humanoids use the whole-body WHOLE_BODY_G1 composite controller (SONIC +
GR00T kPlanner) for balance and locomotion. The MuJoCo viewer keyboard feeds
locomotion commands directly into the controller.

    G1 has three selectable control modes via ``--g1-control``:
      - ``keyboard``: ``default_g1.json`` (``sonic_planner``) — lower-body
        locomotion only, driven by the viewer keyboard.
      - ``pico-teleop``: ``default_g1_teleop.json`` (``sonic_teleop``,
        encoder mode_id=1) — kPlanner legs + PICO VR 3-point arms.
      - ``pico-manager``: ``default_g1_manager.json`` (``sonic_manager``) —
        full 5-mode PICO state machine (OFF/PLANNER/VR_3PT/POSE/FROZEN).
    If ``--g1-control`` is omitted (and the robot is a G1), the demo prompts you
    to pick one interactively after startup, just like the scene picker. The
    PICO data source is chosen with ``--pico-source`` (``real`` needs
    xrobotoolkit_sdk + a connected headset on Linux); omit it and, when a
    ``pico-*`` mode is chosen, you'll be prompted for ``mock``/``real`` too.
    With ``mock`` the arms hold a fixed pose and buttons never fire — useful for
    plumbing tests without hardware; locomotion via keyboard still works.

Usage (Linux, needs a display for the viewer):

    conda activate tele_benchmark
    cd /path/to/humanoid_benchmark
    python -m robocasa.demos.demo_empty_teleop                       # pick a G1 scene from the menu
    python -m robocasa.demos.demo_empty_teleop --env TidyLivingRoom  # a specific living-room task
    python -m robocasa.demos.demo_empty_teleop --robot G1FloatingBody   # G1 + SONIC (keyboard locomotion)
    python -m robocasa.demos.demo_empty_teleop --robot G1FloatingBody --g1-control pico-manager  # G1 + PICO (mock)
    python -m robocasa.demos.demo_empty_teleop --robot G1FloatingBody --g1-control pico-teleop --pico-source real  # real PICO
    python -m robocasa.demos.demo_empty_teleop --env LivingRoomEnv --robot G1FloatingBody
    python -m robocasa.demos.demo_empty_teleop --env LivingRoomEnv --layout 2  # specific layout
    python -m robocasa.demos.demo_empty_teleop --env TidyLivingRoom --check  # headless smoke test

On macOS, run with ``mjpython`` instead of ``python``.
"""

import argparse
import pathlib
import time
from collections import OrderedDict

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
from g1_bridge.pico.mode_manager import ManagerMode
from g1_bridge.sonic_utils.ckpt import (
    DEFAULT_SONIC_CKPT,
    SONIC_CKPT_CHOICES,
    apply_sonic_ckpt,
)
from robosuite.environments.base import REGISTERED_ENVS
from termcolor import colored

import robocasa  # noqa: F401  -- triggers EmptyEnv / LivingRoom registration
from robocasa.environments.living_room.living_room import (
    LivingRoomEnv,
    REGISTERED_LIVING_ROOM_ENVS,
)
from robocasa.utils import teleop_viewer
from robocasa.utils.robot_utils import DEFAULT_ROBOT, G1_ROBOTS
from robocasa.utils.ego_stream import EgoFramePublisher
from robocasa.utils.teleop_viewer import TeleopViewer


# short descriptions for the scene picker; any registered env without an entry
# falls back to showing its class name.
_SCENE_DESCRIPTIONS = {
    "EmptyEnv": "asset-free native MuJoCo ground plane (no walls or furniture)",
    "TeleLabEnv": "empty TeleLab room with authored floor, walls, and lights",
    "TeleLabConveyor": "TeleLab room with a controllable conveyor fixture",
    "TeleLabMani": "TeleLab manipulation room with a fridge, shelf, and desk",
    "TeleLabApproachOpenFridge": "task: walk to the fridge, then open its door",
    "TeleLabCarryCupFromShelfToDesk": (
        "task: carry a shelf cup to the adjacent desk"
    ),
    "LivingRoomEnv": "furnished living room, random layout/style, no task "
    "(use --layout/--style to fix a scene)",
    "TidyLivingRoom": "task: tidy the apple on the coffee table (placeholder)",
    "ApproachSofa": "task: walk over to the sofa (navigation; G1-friendly)",
    "ApproachTable": "task: walk over to the side table (navigation; G1-friendly)",
    "CarryObjectToTable": "task: carry a mug to the table across the room "
    "(pick-and-carry; needs grasping)",
}


def is_whole_body_g1(controller_config):
    """True if the loaded controller is one of the G1 whole-body composites.

    Both drive all 29 joints themselves and expose the KEY_HELP/handle_keypress
    surface run_sonic_viewer_loop needs, so both take the whole-body branch.
    ``--g1-control amo`` used to fall through here (its config says AMO_G1,
    which this only checked for WHOLE_BODY_G1) and silently ended up in the
    manipulator EEF-teleop loop instead.
    """
    return controller_config.get("type") in ("WHOLE_BODY_G1", "G1_BRIDGE")


# G1 control-mode -> WHOLE_BODY_G1 controller config file (in robosuite's
# controllers/config/robots/). keyboard = locomotion only; the pico-* modes add
# PICO VR upper-body teleop on top of the kPlanner lower body. ``amo`` is not
# here: it is a G1_BRIDGE backend, built by load_sim2real_controller_config.
_G1_CONTROL_CONFIGS = {
    "keyboard": "default_g1.json",  # sonic_planner: kPlanner legs, keyboard-driven
    "pico-teleop": "default_g1_teleop.json",  # sonic_teleop (mode_id=1): + VR 3-point arms
    "pico-manager": "default_g1_manager.json",  # sonic_manager: full 5-mode PICO state machine
}

# Human-readable descriptions for the interactive G1 control-mode picker (shown
# when the robot is a G1 and --g1-control wasn't passed on the command line).
_G1_CONTROL_DESCRIPTIONS = OrderedDict(
    [
        ("keyboard", "keyboard only — lower-body locomotion (no PICO needed)"),
        (
            "pico-teleop",
            "PICO VR — kPlanner legs + 3-point VR arms (encoder mode_id=1)",
        ),
        (
            "pico-manager",
            "PICO VR — full 5-mode state machine (OFF/PLANNER/VR_3PT/POSE/FROZEN)",
        ),
        (
            "amo",
            "AMO whole-body policy — PICO joystick locomotion + 3-point IK arms",
        ),
    ]
)

# Descriptions for the PICO data-source picker (shown after a pico-* mode is
# chosen interactively).
_PICO_SOURCE_DESCRIPTIONS = OrderedDict(
    [
        (
            "mock",
            "mock — no hardware; arms hold a fixed pose, buttons never fire (plumbing test)",
        ),
        (
            "keyboard",
            "keyboard — starts in VR_3PT. WASD=left arm IJKL=right arm "
            "RF/YH=up/down Num8/2/4/6=Y/A/X/B Z/X=grip \\=reset",
        ),
        ("real", "real — needs xrobotoolkit_sdk + a connected PICO headset (Linux)"),
    ]
)


def is_g1_robot(robot):
    """True if ``robot`` is a G1 variant (uses the WHOLE_BODY_G1 controller)."""
    return robot in G1_ROBOTS


def load_g1_controller_config(g1_control, pico_source, sonic_ckpt=DEFAULT_SONIC_CKPT):
    """Load the WHOLE_BODY_G1 controller config for the chosen G1 control mode.

    For the ``pico-*`` modes the JSON ships with ``pico_source: "real"``; we
    override it with ``pico_source`` here so the demo defaults to the mock
    source (no xrobotoolkit_sdk / headset needed) unless the user opts into
    ``--pico-source real``.

    ``sonic_ckpt`` (``old``/``new``) selects which SONIC encoder/decoder ONNX
    checkpoint the controller loads — see ``sonic_ckpt.apply_sonic_ckpt``.

    ``amo`` is not a WHOLE_BODY_G1 mode at all — it routes to the G1_BRIDGE
    composite's amo backend, which is where AMO whole-body control now lives.
    """
    if g1_control == "amo":
        # Imported lazily: g1_bridge pulls in the sim2real package, and
        # the other (SONIC) modes must stay usable without it installed.
        from robocasa.g1_host.bridge import load_sim2real_controller_config

        return load_sim2real_controller_config("amo", pico_source)

    fname = _G1_CONTROL_CONFIGS[g1_control]
    cfg_path = (
        pathlib.Path(robosuite.__file__).parent / "controllers/config/robots" / fname
    )
    cfg = load_composite_controller_config(controller=str(cfg_path))
    apply_sonic_ckpt(cfg, sonic_ckpt)
    if g1_control != "keyboard":
        spec = cfg.setdefault("composite_controller_specific_configs", {})
        spec["pico_source"] = pico_source
        # The JSON ships pico_source_kwargs tuned for the real source (e.g.
        # ``verbose``); MockPicoVRSource and KeyboardPicoVRSource don't accept
        # those, so clear them to avoid a spurious construction error.
        if pico_source in ("mock", "keyboard"):
            spec["pico_source_kwargs"] = {}
    return cfg


def is_living_room_env(env_name):
    """True if ``env_name`` is a registered LivingRoom scene/task subclass."""
    cls = REGISTERED_ENVS.get(env_name)
    return isinstance(cls, type) and issubclass(cls, LivingRoomEnv)


def scene_options():
    """Selectable empty, TeleLab, and registered living-room environments.

    Returns an ordered
    ``{env_name: description}`` map for the picker."""
    options = OrderedDict()
    options["EmptyEnv"] = _SCENE_DESCRIPTIONS["EmptyEnv"]
    for name in (
        "TeleLabEnv",
        "TeleLabConveyor",
        "TeleLabMani",
        "TeleLabApproachOpenFridge",
        "TeleLabCarryCupFromShelfToDesk",
    ):
        if name in REGISTERED_ENVS:
            options[name] = _SCENE_DESCRIPTIONS[name]
    for name in sorted(REGISTERED_LIVING_ROOM_ENVS):
        options[name] = _SCENE_DESCRIPTIONS.get(name, name)
    return options


def choose_option(options, option_name, default=None):
    """Print numbered ``{key: description}`` options and return the chosen key.
    Any non-numeric / out-of-range input falls back to ``default``."""
    if default is None:
        default = list(options.keys())[0]

    print("Here is a list of {}s:\n".format(option_name))
    for i, (k, v) in enumerate(options.items()):
        print("[{}] {}: {}".format(i, k, v))
    print()
    try:
        s = input(
            "Choose an option 0 to {}, or any other key for default ({}): ".format(
                len(options) - 1, default
            )
        )
        k = min(max(int(s), 0), len(options) - 1)
        choice = list(options.keys())[k]
    except Exception:
        choice = default
        print("Use {} by default.\n".format(choice))
    return choice


# GLFW reports letter keys as their uppercase ASCII code, so 'q' arrives as
# ord("Q"). 'Q' is not bound to a destructive action by the MuJoCo passive
# viewer, but it is not free either: simulate also toggles its own "Camera"
# visualization flag on Q (one box + one cylinder per camera, so a G1 scene
# sprouts cylinders), and the same is true of most other teleop keys.
# TeleopViewer pins the viewer's MjvOption every frame to undo those -- see
# robocasa/utils/teleop_viewer.py (_PINNED_OPT_FIELDS).
_KEY_RESET = ord("Q")
_KEY_SAVE = ord("S")

# Warm-up steps run after every reset, before the pause gate hands control to the
# operator. More than one (collect_human_trajectory uses a single warm-up) so the
# disengaged arms settle from the kinematic init pose to their gravity-loaded OFF
# equilibrium (~15-20 steps) -- otherwise the paused, pre-first-keypress pose
# shows the arms mid-drop, inconsistent with the settled OFF pose once stepping.
_WARMUP_SETTLE_STEPS = 20

# The G1's first-person head camera. It is mounted on the robot (in the G1
# MJCF), so the ego view is a *robot-level* feature available in any env/script
# that loads G1 with this camera in its observations -- the SONIC control loop
# below shows it whenever it's present.
EGO_CAMERA = "robot0_head_camera"
# 320x240 (W x H): offscreen camera render is ~3x cheaper than 640x480, which
# keeps the per-step time under the realtime budget (the camera render was the
# main extra cost making the viewer choppy). Still plenty for an ego view / VLA.
EGO_VIEW_H, EGO_VIEW_W = 240, 320


class EgoCameraPreview:
    """Live first-person (head-camera) preview window, tied to the G1 robot.

    Uses Tkinter + PIL (the env's cv2 build is headless, so cv2.imshow is not
    available). Degrades gracefully: if there's no display / no Tk it disables
    itself and prints one line instead of crashing.
    """

    def __init__(
        self, title=f"G1 ego view ({EGO_CAMERA})", stride=2, display_scale=2.5
    ):
        # Refresh the window only every ``stride`` calls so the per-step Tkinter
        # cost (main-thread GUI update) is cut: at control_freq=20, stride=2 ->
        # ~10 FPS preview, which is smooth enough to teleop by and keeps the
        # control loop from falling behind realtime.
        # ``display_scale`` enlarges the shown window (the camera still renders at
        # EGO_VIEW_W x EGO_VIEW_H, so the sim/recording cost is unchanged — only
        # the on-screen image is upscaled, for the operator to see more clearly).
        self._stride = max(1, int(stride))
        self._display_scale = float(display_scale)
        self._i = 0
        self._base_title = title
        self._tk = None
        self._photo = None  # keep a ref so Tk doesn't GC the image
        try:
            import tkinter as tk
            from PIL import Image, ImageTk

            self._Image, self._ImageTk = Image, ImageTk
            root = tk.Tk()
            root.title(title)
            # Place the ego window on the right side of the screen so it
            # doesn't overlap the main MuJoCo viewer.
            display_w = int(root.winfo_screenwidth() * 0.65)
            display_h = 50
            root.geometry(f"+{display_w}+{display_h}")
            label = tk.Label(root)
            label.pack()
            self._tk = (root, label)
        except Exception as e:  # no display / no Tk -> disable, don't crash
            print(
                colored(f"[ego] preview disabled ({type(e).__name__}: {e})", "yellow")
            )
            self._tk = None

    def update(self, obs):
        """Refresh the window with the latest head-camera frame from ``obs``
        (throttled to every ``stride`` calls)."""
        if self._tk is None:
            return
        self._i += 1
        if self._i % self._stride != 0:
            return
        img = obs.get(f"{EGO_CAMERA}_image")
        if img is None:
            return
        try:
            root, label = self._tk
            # robosuite frames are vertically flipped; PIL takes RGB as-is.
            pil = self._Image.fromarray(img[::-1])
            if self._display_scale != 1.0:
                w, h = pil.size
                pil = pil.resize(
                    (int(w * self._display_scale), int(h * self._display_scale)),
                    self._Image.BILINEAR,
                )
            self._photo = self._ImageTk.PhotoImage(pil)
            label.configure(image=self._photo)
            root.update_idletasks()
            root.update()
        except Exception as e:
            self._tk = None
            print(
                colored(
                    f"[ego] preview error ({type(e).__name__}); disabling", "yellow"
                )
            )

    def set_fps(self, fps, work_ms=None, budget_ms=None):
        """Show the live sim FPS (and per-step work time) in the window title."""
        if self._tk is None:
            return
        try:
            t = f"{self._base_title} — {fps:.1f} FPS"
            if work_ms is not None and budget_ms:
                t += f"  ({work_ms:.0f}/{budget_ms:.0f} ms)"
            self._tk[0].title(t)
        except Exception:
            pass

    def close(self):
        if self._tk is not None:
            try:
                self._tk[0].destroy()
            except Exception:
                pass


def task_status_line(env):
    """``(success, one-line live task state)``. The line goes in the viewer HUD
    (so the operator can see how far off they are without reading the terminal)
    and, throttled, in the terminal readout below."""
    base = getattr(env, "unwrapped", env)
    success = base._check_success()
    parts = ["SUCCESS" if success else "in progress"]
    if hasattr(base, "success_distance"):
        parts.append(f"dist {base.success_distance():.2f} m")
        if hasattr(base, "HOLD_STEPS"):
            parts.append(
                f"held {getattr(base, '_in_range_steps', 0)}/{base.HOLD_STEPS}"
            )
    return success, "   ".join(parts)


def report_task_status(env, state):
    """Print task success on the rising edge, plus a periodic distance readout
    for tasks that expose ``success_distance``. ``state`` (a dict) persists the
    previous success flag, the latest status line (for the HUD) and a step
    counter across calls; clear it on reset."""
    state["i"] = state.get("i", 0) + 1
    success, line = task_status_line(env)
    state["line"] = line
    if success and not state.get("prev", False):
        print(colored("[task] SUCCESS — task condition met!", "green"))
    state["prev"] = success
    base = getattr(env, "unwrapped", env)
    if hasattr(base, "success_distance") and state["i"] % 20 == 0:
        print(f"[task] {line}")


#: ManagerMode -> (label, what it means) for the composite's chord profile.
#: The GMR tracking backends run the "tracking" profile, where PLANNER is not a
#: planner at all -- it is the engaged-but-holding idle state, and B+Y /
#: L-stick are muted so the other two modes are unreachable. Showing the raw
#: enum name there would tell the operator they are in "PLANNER" when nothing
#: is planning. See mode_manager._PROFILE_CHORDS.
_MODE_LABELS = {
    "tracking": {
        ManagerMode.OFF: ("OFF", "disengaged, holding standing pose"),
        ManagerMode.PLANNER: ("IDLE", "engaged, holding pose - A+X to track"),
        ManagerMode.POSE: ("POSE", "tracking your body"),
    },
    "sonic": {
        ManagerMode.OFF: ("OFF", "disengaged"),
        ManagerMode.PLANNER: ("PLANNER", "kPlanner locomotion"),
        ManagerMode.POSE: ("POSE", "full-body SMPL tracking"),
        ManagerMode.PLANNER_FROZEN_UPPER: ("FROZEN", "planner legs, frozen upper"),
        ManagerMode.POSE_PAUSE: ("PAUSED", "POSE frozen while L-menu is held"),
        ManagerMode.PLANNER_VR_3PT: ("VR_3PT", "planner legs, 3-point arms"),
    },
}


def mode_line(env, recorder=None):
    """One line describing the control state, for the viewer HUD.

    Returns "" for controllers with no ModeManager (the keyboard/device teleop
    paths), which simply drops the row.
    """
    composite = env.robots[0].composite_controller
    manager = getattr(composite, "_mode_manager", None)
    if manager is None:
        return ""
    profile = getattr(manager, "profile", "sonic")
    mode = manager.current
    label, meaning = _MODE_LABELS.get(profile, _MODE_LABELS["sonic"]).get(
        mode, (mode.name, "")
    )
    text = f"{label} - {meaning}" if meaning else label
    if recorder is not None:
        # Recording is gated on the mode (and on the controller having produced
        # a real reference), so say which it is rather than leaving the operator
        # to infer it from the mode.
        text += f" | REC {recorder.ep_frames}f" if recorder.ep_frames else " | not rec"
    return text


#: sonic encoder path fragment -> the ``--sonic-ckpt`` name that selected it.
_CKPT_LABELS = {"low_latency": "new", "release": "old"}


def controller_line(env):
    """One line naming the controller actually driving the robot, for the HUD.

    ``_ctrl_type`` is what both G1 composites already call themselves:
    ``sonic_planner`` / ``sonic_teleop`` / ``sonic_manager`` for WHOLE_BODY_G1,
    ``g1_<backend>`` for G1_BRIDGE. Manipulator composites carry no such
    field, so fall back to the class name -- still the answer to "what is
    driving this robot".

    The SONIC checkpoint flavor rides along when the config names one: old
    (release) vs new (low-latency) is a per-run choice that changes the encoder
    layout, and nothing else on screen says which one got loaded.
    """
    composite = env.robots[0].composite_controller
    name = getattr(composite, "_ctrl_type", None) or type(composite).__name__
    cfg = getattr(composite, "composite_controller_specific_config", None) or {}
    encoder = str(cfg.get("sonic_encoder_path", ""))
    for fragment, label in _CKPT_LABELS.items():
        if fragment in encoder:
            return f"{name} ({label} ckpt)"
    return name


def announce_task(env, view=None):
    """Print the language instruction and pin it in the viewer HUD, where it
    stays put instead of scrolling off under the per-step status spam."""
    lang = env.get_ep_meta().get("lang", "")
    if lang:
        print(colored(f"[task] {lang}", "magenta", attrs=["bold"]))
    if view is not None:
        view.set_instruction(lang)


def run_sonic_viewer_loop(
    env,
    recorder=None,
    show_ego=True,
    initial_reset=True,
    extra_keypress_callback=None,
    viewer_overlay_callback=None,
    reset_callback=None,
):
    """Whole-body G1 control loop, mirroring ``collect_human_trajectory``'s
    rigorous design: physics is held PAUSED until the user issues the first
    locomotion keypress, then every frame steps SONIC. The viewer keeps
    rendering while paused so the user can orbit / inspect the scene. During
    collection, S saves the current episode and Q discards it; both reset.

    ``recorder`` is an optional data-collection hook so a collection script can
    reuse this exact control loop (identical keyboard handling). When given it
    must implement:
      - ``record_step(env, composite, obs) -> bool`` — record the current step;
        return True if it just saved an episode (loop then resets for the next).
      - ``discard()`` — drop the in-progress episode (called on Q before reset).
      - ``reset_episode()`` — clear per-episode state (called after each reset).
      - ``close()`` — finalize (called once on exit).
    Environments can opt into S-to-save by setting ``manual_episode_save=True``;
    this is intended for free-teleoperation scenes without a success predicate.
    ``reset_callback(env, reason, recorder)`` is an optional hook called just
    before discard, manual-save, or success resets. ``reason`` is ``"discard"``
    for Q, ``"manual"`` for S, and ``"success"`` after the recorder saves an
    episode automatically.
    With ``recorder=None`` this is the plain teleop demo (Q simply resets)."""
    # The MuJoCo passive viewer's key_callback runs on the main thread during
    # sync(); calling env.reset() from inside it would tear down the very viewer
    # whose callback is executing. So the callback only raises a flag, and the
    # main loop performs the reset between steps.
    reset_requested = {"flag": False}
    manual_save_requested = {"flag": False}
    base_env = getattr(env, "unwrapped", env)
    manual_save_enabled = recorder is not None and bool(
        getattr(base_env, "manual_episode_save", False)
    )
    # Peer of ``collect_human_trajectory``'s ``nonzero_ac_seen`` gate: stays
    # False until the user actually presses a locomotion key. While False, we
    # skip env.step (physics paused) but still env.render the viewer.
    command_received = {"flag": True}  # physics starts immediately
    zero_action = np.zeros(env.action_dim)
    task_state = {}
    # Keeps the free camera on the robot across resets and draws the task text
    # in the MuJoCo window (robocasa/utils/teleop_viewer.py).
    view = TeleopViewer(env)

    def setup_after_reset():
        """(Re-)wire the controller's keyboard handler into the freshly created
        viewer. env.reset() builds a new MjviewerRenderer with a null callback,
        and launch_passive bakes the callback in at first render, so this must
        run after every reset and before the next step."""
        composite = env.robots[0].composite_controller
        if hasattr(env.viewer, "add_keypress_callback") and hasattr(
            composite, "handle_keypress"
        ):

            def handler(keycode, _composite=composite):
                if extra_keypress_callback is not None and extra_keypress_callback(
                    keycode
                ):
                    return
                if keycode == _KEY_RESET:
                    reset_requested["flag"] = True
                    return
                if manual_save_enabled and keycode == _KEY_SAVE:
                    manual_save_requested["flag"] = True
                    return
                if view.handle_keypress(keycode):  # camera / HUD keys
                    return
                # any non-reset keypress is a locomotion command -> start stepping
                command_received["flag"] = True
                _composite.handle_keypress(keycode)

            env.viewer.add_keypress_callback(handler)
            env.viewer.keypress_callback = handler
            print(composite.KEY_HELP)
            print(
                "  Q          - {}".format(
                    "discard current episode + reset"
                    if recorder is not None
                    else "reset the environment"
                )
            )
            if manual_save_enabled:
                print("  S          - save current episode + reset")
            print(teleop_viewer.KEY_HELP)
        else:
            print("[teleop] viewer/composite does not support a keyboard hook")

    def do_reset(reason):
        """Reset env and re-arm the pause gate after the given episode event."""
        nonlocal composite
        if reset_callback is not None:
            reset_callback(env, reason=reason, recorder=recorder)
        if recorder is not None and reason == "discard":
            recorder.discard()
        # Capture the operator's orbit before the reset destroys the viewer, so
        # the fresh window comes up on the same view of the robot rather than on
        # MjviewerRenderer's world-origin default.
        view.remember_orbit()
        env.reset()
        # env.reset() rebuilds the robot and its composite controller, so the
        # previously captured ``composite`` is now STALE -- re-fetch it. Without
        # this, the recorder reads token_state / finger qpos from the dead
        # episode-0 controller (frozen) while obs comes from the live env, so
        # every episode after the first records constant motion_token/hand_joints.
        composite = env.robots[0].composite_controller
        # The reset also rebuilt the model (body ids move) and swapped in a new,
        # unframed MjviewerRenderer -- re-frame it on the robot before the next
        # render launches its window.
        view.frame_on_robot()
        setup_after_reset()
        # warm-up: settle the disengaged arms to their OFF equilibrium so the
        # paused pose matches the engaged-OFF pose (mirrors collect_human_trajectory,
        # which warms up before handing over control).
        for _ in range(_WARMUP_SETTLE_STEPS):
            env.step(zero_action)
        command_received["flag"] = True  # physics stays active after reset
        task_state.clear()  # reset success/distance reporting
        if recorder is not None:
            recorder.reset_episode()
        announce_task(env, view)

    if initial_reset:
        env.reset()
    view.frame_on_robot()
    setup_after_reset()
    # warm-up: initialize the controller and settle the disengaged arms to their
    # OFF equilibrium (see _WARMUP_SETTLE_STEPS) before the pause gate.
    for _ in range(_WARMUP_SETTLE_STEPS):
        env.step(zero_action)
    if recorder is not None:
        recorder.reset_episode()

    composite = env.robots[0].composite_controller
    ctrl_type = getattr(composite, "_ctrl_type", "?")
    pico = getattr(composite, "_pico_source", None)
    pico_name = type(pico).__name__ if pico is not None else "none"
    print(
        colored(
            f"G1 + SONIC ready (ctrl_type={ctrl_type}, pico_source={pico_name}). "
            "Focus the MuJoCo window, then use the arrow keys to walk; Q resets.",
            "green",
        )
    )
    announce_task(env, view)
    print(
        colored(
            "Physics is ACTIVE — the robot will respond to keyboard input immediately.",
            "green",
        )
    )
    if pico_name == "MockPicoVRSource":
        print(
            colored(
                "[teleop] mock PICO source — arms hold a fixed pose and PICO "
                "buttons never fire (manager stays OFF). Keyboard locomotion still "
                "works. Use --pico-source real with a connected headset for VR arms.",
                "yellow",
            )
        )
    elif pico_name == "KeyboardPicoVRSource":
        print(
            colored(
                "[teleop] keyboard PICO — starts in VR_3PT. "
                "WASD=left_arm  IJKL=right_arm  RF/YH=up/down  "
                "Num 8/2/4/6=Y/A/X/B  Z/X=grippers  \\=reset",
                "yellow",
            )
        )
    print(
        f"=== sim {1.0 / env.sim.model.opt.timestep:.0f} Hz, "
        f"env.step {env.control_freq} Hz, realtime throttle ON. "
        f"Close the window or Ctrl-C to exit. ==="
    )

    target_dt = env.control_timestep
    next_tick = time.monotonic()
    # First-person preview window, created lazily on the first frame that
    # actually carries the head-camera image (so it only opens for a G1 env
    # whose obs include robot0_head_camera, and not as an empty window up front).
    ego_preview = None
    # ... and, when LOCOMANILAB_EGO_STREAM is set, the same frame also goes out on a
    # ZMQ PUB socket so locomanilab.runtime.headset_video can put it in the PICO.
    # None (the default) when the env var is unset -- costs nothing then.
    ego_stream = EgoFramePublisher.maybe()
    # Live FPS / per-step work tracking (work = step+render+camera, excludes the
    # realtime-pacing sleep). Prints once/sec and updates the ego-window title.
    budget_ms = env.control_timestep * 1000.0
    work_ms_ema = None
    fps_count = 0
    fps_since = time.monotonic()
    last_fps_print = fps_since

    def render_with_overlay():
        env.render()
        # after env.render(), so the passive-viewer handle exists on the first
        # frame after every reset (that is where the camera is re-framed and the
        # HUD text is refreshed)
        view.set_controller(controller_line(env))
        view.set_mode(mode_line(env, recorder))
        # Re-read every frame from the live composite: env.reset() rebuilds it,
        # and the sim2real one only learns its backend at load_controller_config
        # time, so a sheet captured at startup would be the wrong controller's.
        view.set_cheatsheet(
            getattr(env.robots[0].composite_controller, "hud_cheatsheet", ())
        )
        view.sync(status=task_state.get("line"))
        if viewer_overlay_callback is not None:
            viewer_overlay_callback(env)

    try:
        while True:
            # reset is allowed at any time, paused or stepping
            if reset_requested["flag"]:
                reset_requested["flag"] = False
                print(
                    "[teleop] Q pressed — {}".format(
                        "discarding episode + resetting"
                        if recorder is not None
                        else "resetting environment"
                    )
                )
                do_reset(reason="discard")
                next_tick = time.monotonic()  # don't count reset time against pacing
                fps_count = 0
                fps_since = time.monotonic()
                continue

            if manual_save_requested["flag"]:
                manual_save_requested["flag"] = False
                saved = manual_save_enabled and recorder.save(reason="MANUAL")
                if saved:
                    print("[teleop] S pressed — saved current episode + resetting")
                    do_reset(reason="manual")
                    next_tick = time.monotonic()
                    fps_count = 0
                    fps_since = time.monotonic()
                    continue
                print("[teleop] S pressed — no recorded frames to save")

            if command_received["flag"]:
                _work_t0 = time.perf_counter()
                obs = env.step(zero_action)[0]
                render_with_overlay()
                if f"{EGO_CAMERA}_image" in obs:
                    if show_ego:
                        if ego_preview is None:
                            ego_preview = EgoCameraPreview()
                        ego_preview.update(obs)
                    if ego_stream is not None:
                        # robosuite frames are vertically flipped, same as the
                        # preview and the recorder do (g1_data_collection.py).
                        ego_stream.publish(obs[f"{EGO_CAMERA}_image"][::-1])
                saved = (
                    recorder.record_step(env, composite, obs)
                    if recorder is not None
                    else False
                )
                report_task_status(env, task_state)

                # --- FPS / work-time readout (once per second) ---
                work_ms = (time.perf_counter() - _work_t0) * 1000.0
                work_ms_ema = (
                    work_ms
                    if work_ms_ema is None
                    else 0.9 * work_ms_ema + 0.1 * work_ms
                )
                fps_count += 1
                now = time.monotonic()
                if now - last_fps_print >= 1.0:
                    fps = fps_count / max(1e-6, now - fps_since)
                    print(
                        colored(
                            f"[fps] {fps:.1f} Hz (target {env.control_freq}) | "
                            f"step work {work_ms_ema:.1f}/{budget_ms:.0f} ms "
                            f"({100.0 * work_ms_ema / budget_ms:.0f}% of budget)",
                            "cyan",
                        )
                    )
                    if ego_preview is not None:
                        ego_preview.set_fps(fps, work_ms_ema, budget_ms)
                    fps_count = 0
                    fps_since = now
                    last_fps_print = now
                if saved:
                    # episode saved on task success -> reset for the next one
                    do_reset(reason="success")
                    next_tick = time.monotonic()
                    continue
            else:
                # paused: keep the viewer alive so the user can orbit / inspect,
                # but don't advance physics until they press a key
                render_with_overlay()

            next_tick += target_dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()  # fell behind realtime; drop the lag
    except KeyboardInterrupt:
        print("\n[teleop] Ctrl-C — exiting")
    finally:
        env.close()
        if ego_preview is not None:
            ego_preview.close()
        if ego_stream is not None:
            ego_stream.close()
        if recorder is not None:
            recorder.close()
        if pico is not None:
            pico.stop()


def run_headless_check(
    env_name, robot, controller_config, steps=300, scene_kwargs=None
):
    """No-display smoke test: build the env, step ``steps`` times, and report.
    For G1 it also checks that the torso stays upright (SONIC balancing)."""
    env = robosuite.make(
        env_name=env_name,
        robots=robot,
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        **(scene_kwargs or {}),
    )
    env.reset()
    whole_body = is_whole_body_g1(controller_config)
    prefix = env.robots[0].robot_model.naming_prefix
    tid = None
    if whole_body:
        # Measure torso_link (stands ~0.81 m); the pelvis body origin sits low
        # in the G1 MJCF and is a poor upright proxy.
        tid = env.sim.model.body_name2id(prefix + "torso_link")
    z0 = float(env.sim.data.body_xpos[tid][2]) if tid is not None else None
    zero_action = np.zeros(env.action_dim)
    for _ in range(steps):
        env.step(zero_action)
    print(
        f"[check] env={env_name} robot={robot} action_dim={env.action_dim}: "
        f"stepped {steps}x without error"
    )
    if tid is not None:
        z1 = float(env.sim.data.body_xpos[tid][2])
        upright = z1 > 0.6
        print(f"[check] torso_link z: reset={z0:.3f} -> after {steps} steps={z1:.3f}")
        print(
            f"[check] {'PASS' if upright else 'FAIL'}: robot "
            f"{'stays upright' if upright else 'collapsed'}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="scene/task name (e.g. EmptyEnv, LivingRoomEnv, TidyLivingRoom). "
        "If omitted, you'll be prompted to pick one.",
    )
    parser.add_argument(
        "--robot",
        type=str,
        default=DEFAULT_ROBOT,
        choices=G1_ROBOTS,
        help="G1 robot variant to load (default: G1FloatingBody).",
    )
    parser.add_argument(
        "--layout",
        type=int,
        default=None,
        help="living-room layout id (only for LivingRoom scenes/tasks).",
    )
    parser.add_argument(
        "--style",
        type=int,
        default=None,
        help="living-room style id (only for LivingRoom scenes/tasks).",
    )
    parser.add_argument(
        "--scene-id",
        type=int,
        default=None,
        help="TeleLab task scene id (currently 1 for formal TeleLab tasks).",
    )
    parser.add_argument(
        "--conveyor-speed",
        type=float,
        default=0.0,
        help="initial TeleLabConveyor speed in m/s (U/O adjust, P stops).",
    )
    parser.add_argument(
        "--conveyor-speed-step",
        type=float,
        default=0.1,
        help="TeleLabConveyor speed change per U/O keypress in m/s.",
    )
    parser.add_argument(
        "--conveyor-speed-range",
        type=float,
        nargs=2,
        default=(-1.5, 1.5),
        metavar=("MIN", "MAX"),
        help="TeleLabConveyor minimum and maximum speed in m/s.",
    )
    parser.add_argument(
        "--g1-control",
        type=str,
        default=None,
        # _G1_CONTROL_DESCRIPTIONS, not _G1_CONTROL_CONFIGS: 'amo' is a
        # G1_BRIDGE backend and so has no WHOLE_BODY_G1 config file.
        choices=list(_G1_CONTROL_DESCRIPTIONS.keys()),
        help="G1 control mode. 'keyboard' = "
        "lower-body locomotion only; 'pico-teleop'/'pico-manager' add PICO VR "
        "upper-body teleop; 'amo' runs the AMO whole-body policy through the "
        "G1_BRIDGE composite. If omitted, you'll be "
        "prompted to pick one interactively. See module docstring.",
    )
    parser.add_argument(
        "--pico-source",
        type=str,
        default=None,
        choices=["mock", "real", "keyboard"],
        help="PICO VR data source for the pico-* G1 control modes. 'mock' "
        "needs no hardware; 'keyboard' simulates PICO buttons via keyboard; "
        "'real' needs xrobotoolkit_sdk + a connected PICO headset on Linux. "
        "If omitted (and a pico-* mode is chosen), you'll be "
        "prompted to pick one interactively.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="headless smoke test (no viewer); use on machines without a display.",
    )
    parser.add_argument(
        "--ego-view",
        action="store_true",
        help="ALSO open the G1 first-person (head-camera) preview window. Off by "
        "default: it adds an offscreen camera render that contends with the "
        "on-screen MuJoCo viewer (GL context switch every step) and can make a "
        "heavy scene choppy. Turn it on only when you want the ego view.",
    )
    parser.add_argument(
        "--control-freq",
        type=int,
        default=20,
        help="env control frequency (Hz). Higher = smoother viewer; with SONIC "
        "on GPU there's headroom to try 30. The [fps] readout shows the budget %%.",
    )
    parser.add_argument(
        "--sonic-ckpt",
        type=str,
        default=DEFAULT_SONIC_CKPT,
        choices=list(SONIC_CKPT_CHOICES),
        help="which SONIC encoder/decoder ONNX checkpoint to load: 'old' "
        "(release) or 'new' (low-latency). Default 'new' -- only the "
        "low-latency export ships in the sim2real checkpoint tree.",
    )
    args = parser.parse_args()

    # pick a scene/task interactively when one isn't given (skip in headless
    # --check, which must not block on input -- default to EmptyEnv there).
    if args.env is None:
        if args.check:
            args.env = "EmptyEnv"
        else:
            args.env = choose_option(
                scene_options(), "scene/task", default="LivingRoomEnv"
            )

    # Resolve the G1 control mode + PICO source. When a flag wasn't passed
    # on the command line, prompt interactively (but never block in headless
    # --check, where we fall back to the keyboard/mock defaults).
    if args.g1_control is None:
        if args.check:
            args.g1_control = "keyboard"
        else:
            args.g1_control = choose_option(
                _G1_CONTROL_DESCRIPTIONS, "G1 control mode", default="keyboard"
            )
    # PICO source only matters for the pico-* modes; keyboard ignores it.
    if args.g1_control != "keyboard" and args.pico_source is None:
        if args.check:
            args.pico_source = "mock"
        else:
            args.pico_source = choose_option(
                _PICO_SOURCE_DESCRIPTIONS, "PICO data source", default="mock"
            )
    if args.pico_source is None:
        args.pico_source = "mock"
    controller_config = load_g1_controller_config(
        args.g1_control, args.pico_source, args.sonic_ckpt
    )

    # layout/style only apply to LivingRoom envs; pass them through when given.
    scene_kwargs = {}
    if is_living_room_env(args.env):
        if args.layout is not None:
            scene_kwargs["layout_ids"] = args.layout
        if args.style is not None:
            scene_kwargs["style_ids"] = args.style
    if args.env == "TeleLabConveyor":
        scene_kwargs["conveyor_speed"] = args.conveyor_speed
        scene_kwargs["conveyor_speed_step"] = args.conveyor_speed_step
        scene_kwargs["conveyor_speed_range"] = args.conveyor_speed_range
    if args.scene_id is not None and args.env in {
        "TeleLabApproachOpenFridge",
        "TeleLabCarryCupFromShelfToDesk",
    }:
        scene_kwargs["scene_ids"] = args.scene_id

    if args.check:
        run_headless_check(
            args.env, args.robot, controller_config, scene_kwargs=scene_kwargs
        )
        return

    whole_body = is_whole_body_g1(controller_config)

    # The G1's head camera can show its first-person view while teleoping, but
    # the offscreen render contends with the on-screen viewer (GL context switch
    # per step) and slows a heavy scene -> off by default; opt in with --ego-view.
    show_ego = whole_body and args.ego_view
    ego_cam_kwargs = (
        dict(
            use_camera_obs=True,
            has_offscreen_renderer=True,
            camera_names=[EGO_CAMERA],
            camera_heights=EGO_VIEW_H,
            camera_widths=EGO_VIEW_W,
        )
        if show_ego
        else dict(use_camera_obs=False, has_offscreen_renderer=False)
    )

    print(colored(f"Initializing {args.env} with {args.robot}...", "yellow"))
    env = robosuite.make(
        env_name=args.env,
        robots=args.robot,
        controller_configs=controller_config,
        has_renderer=True,
        # Free camera; Tab cycles the fixed arena cameras.
        render_camera=None,
        ignore_done=True,
        control_freq=args.control_freq,
        renderer="mjviewer",
        **ego_cam_kwargs,
        **scene_kwargs,
    )

    if whole_body:
        scene_keypress = getattr(env, "handle_scene_keypress", None)
        if hasattr(env, "scene_control_help"):
            print(env.scene_control_help())
        run_sonic_viewer_loop(
            env,
            show_ego=show_ego,
            extra_keypress_callback=scene_keypress,
        )
        return

    raise ValueError("G1 teleoperation requires WHOLE_BODY_G1 or G1_BRIDGE")


if __name__ == "__main__":
    main()
