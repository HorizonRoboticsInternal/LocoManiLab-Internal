"""Controller-specific recording field sources, resolved by name.

Add a new one by dropping a file here (e.g. ``heft_extra.py``, ``sonic_extra.py`` --
named ``<family>_extra.py`` to match the corresponding
``rl_policy/observations/<family>.py``) that defines a ``RecordingExtra`` subclass
with an ``extra_name``, then importing it below so its ``__init_subclass__``
registration actually runs.
"""

from sim2real.rl_policy.recording.extras.base import RecordingExtra
from sim2real.rl_policy.recording.extras import _example_extra  # noqa: F401
from sim2real.rl_policy.recording.extras import dex3_extra  # noqa: F401
from sim2real.rl_policy.recording.extras import heft_extra  # noqa: F401
from sim2real.rl_policy.recording.extras import sonic_extra  # noqa: F401

__all__ = ["RecordingExtra"]
