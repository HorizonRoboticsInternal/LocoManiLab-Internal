"""``RecordingExtra``: a controller-specific recording field source.

Mirrors ``rl_policy.observations.Observation``'s name-registry pattern
(``__init_subclass__`` auto-registration, ``resolve(name)`` lookup) so this reuses a
convention every family already knows, instead of inventing a new one.

Where this differs from ``Observation``: an ``Observation`` feeds the RL policy's own
input vector and is declared under ``policy_config["observation"]``; a
``RecordingExtra`` only adds fields to the *recorded episode* and is declared under
``policy_config["recording_extras"]`` (plus optional ``--record-extra`` CLI overrides,
see ``base_policy.py``'s ``BasePolicyArgs.record_extra``). Generic, cross-family
concepts (camera images, hand-grip) stay on their own dedicated CLI flags
(``--camera-connect``/``--hand-grip-connect``) rather than becoming extras --
``RecordingExtra`` is reserved for fields that only make sense for one controller
(e.g. a forward-kinematics end-effector pose, or a policy-internal action token).
"""

from __future__ import annotations

from abc import ABC
from typing import Any


class RecordingExtra(ABC):
    registry: dict[str, type["RecordingExtra"]] = {}
    extra_name: str | None = None

    def __init_subclass__(cls, *, extra_name: str | None = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if extra_name is not None:
            cls.extra_name = extra_name
            if cls.extra_name in RecordingExtra.registry:
                conflicting = RecordingExtra.registry[cls.extra_name]
                raise ValueError(
                    f"RecordingExtra name {cls.extra_name!r} already registered by "
                    f"{conflicting.__module__}.{conflicting.__qualname__}"
                )
            RecordingExtra.registry[cls.extra_name] = cls

    @classmethod
    def resolve(cls, name: str) -> type["RecordingExtra"]:
        if name not in cls.registry:
            raise ValueError(
                f"RecordingExtra {name!r} not found. Available: {sorted(cls.registry)}"
            )
        return cls.registry[name]

    def __init__(self, policy: Any) -> None:
        self.policy = policy

    def features(self) -> dict[str, dict]:
        """Extra LeRobot feature declarations, merged into the generic baseline.

        Called once, right before dataset creation (see ``LeRobotRecorder``), after
        the generic schema has already been derived -- so this only needs to
        contribute what's specific to this controller.
        """
        return {}

    def compute(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Return extra flat LeRobot fields for one frame, merged before add_frame().

        ``frame`` is the same per-tick dict ``BasePolicy._append_record_frame``
        already built (``step_index``/``time_ns``/``robot``/``policy``/``runtime``) --
        pull whatever generic signal you need from there. Anything this controller
        alone knows about (its own ZMQ topics, a forward-kinematics model, ...) is
        this subclass's own responsibility to open in ``__init__`` and query here.
        """
        return {}

    def close(self) -> None:
        """Release any resources (sockets, models, ...) opened in __init__."""
        return None
