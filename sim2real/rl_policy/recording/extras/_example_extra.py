"""Template ``RecordingExtra`` -- demonstrates the extension point, not a real field.

This does NOT correspond to any real heft/sonic/etc. signal. It exists to prove the
``RecordingExtra`` registry/features()/compute()/close() lifecycle actually works
end-to-end (see ``tests/test_recording_extras.py``), and to give whoever writes the
first real one (e.g. ``extras/heft_extra.py`` for Dex3 hand state, ``extras/sonic_extra.py`` for a
forward-kinematics end-effector pose) a concrete skeleton to copy.

Enable it for a smoke test with ``--record-extra example`` -- it adds a single
monotonically increasing ``action.example_step_parity`` field, computed from the
frame's own ``step_index`` (no external IO), so it needs no additional hardware/ZMQ
setup to try out.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sim2real.rl_policy.recording.extras.base import RecordingExtra


class ExampleRecordingExtra(RecordingExtra, extra_name="example"):
    def features(self) -> dict[str, dict]:
        return {
            "action.example_step_parity": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["is_even_step"],
            },
        }

    def compute(self, frame: dict[str, Any]) -> dict[str, Any]:
        step_index = int(frame["step_index"])
        return {
            "action.example_step_parity": np.array(
                [1.0 if step_index % 2 == 0 else 0.0], dtype=np.float32
            ),
        }
