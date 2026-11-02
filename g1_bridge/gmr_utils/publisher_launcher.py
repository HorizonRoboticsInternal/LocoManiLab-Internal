"""Thin wrapper around ``sim2real.teleop.pico_retarget_pub`` that starts the
GMR publisher already LIVE instead of paused.

``LiveRetargetPublisher`` hardcodes ``self.paused = True`` at construction
(no CLI flag to change it) and only flips to live tracking on a rising edge
of the left controller's plain X button, read by *this process's own*
independent PICO SDK connection. In practice this has proven confusing and
fragile to operate: this process's stdout (where the pause-toggle
confirmation prints) usually isn't the terminal the operator is watching,
so there's no feedback on whether the press registered, and debugging
across two independent SDK connections (this one vs. tele's own) for what
looks like one physical button press is error-prone.

Rather than editing sim2real's ``pico_retarget_pub.py`` (zero-invasion
constraint -- see the g1_bridge design notes), this is a small
subclass + monkeypatch, run instead of the module directly, that forces the
publisher to start live. The X button still works as a normal live/pause
toggle afterwards, for an operator who wants to step away.

Runs under sim2real's isolated ``venv/pico/.venv`` (Python 3.10)
interpreter, same as ``pico_retarget_pub.py`` itself -- see
``gmr_subprocess.py``, which launches this file instead of
``-m sim2real.teleop.pico_retarget_pub``.
"""

import tyro

from sim2real.teleop import pico_retarget_pub as _prp


class _AutoLiveRetargetPublisher(_prp.LiveRetargetPublisher):
    def __init__(self, args):
        super().__init__(args)
        self.paused = False
        self._needs_live_pelvis_init = True
        self._needs_smpl_heading_init = True
        print(
            "[g1_bridge] GMR publisher starting LIVE (auto-unpaused); "
            "its A+X pause toggle is disabled so A+X stays the ModeManager "
            "PLANNER<->POSE switch, as in the official gear_sonic deploy."
        )

    def _poll_pause_toggle(self) -> bool:
        """Never pause on A+X -- but still run the base implementation for
        its side effects.

        ``_poll_pause_toggle`` does double duty upstream: it reads the
        controller and *publishes* both the A/B/X/Y state and the hand-grip
        triggers over ZMQ (``_publish_controller_state`` /
        ``_publish_hand_grip``), which is the only channel
        ``gmr_button_relay.GMRButtonRelay`` has for buttons and gripper
        control. So it must keep being called; only its return value (the
        pause toggle) is suppressed.

        Why suppress it: A+X is the ModeManager's PLANNER<->POSE transition
        (mode_manager.py, matching gear_sonic's own
        ``pico_manager_thread_server.py`` state machine). Upstream also
        binds A+X to pause/resume retargeting, so on any GMR-backed backend
        one press would switch mode AND freeze the reference motion -- the
        robot silently stops following mid-session. The official deploy has
        no such binding; A+X is a mode switch there and nothing else.
        """
        super()._poll_pause_toggle()
        return False


_prp.LiveRetargetPublisher = _AutoLiveRetargetPublisher


if __name__ == "__main__":
    _prp.run_publish(tyro.cli(_prp.PublisherArgs))
