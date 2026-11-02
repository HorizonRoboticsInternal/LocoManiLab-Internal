"""Install the appearance randomizer on a live env, through robosuite's own
xml-processor seam.

``MujocoEnv._initialize_sim`` runs every registered processor over
``self.model.get_xml()`` immediately before ``MjSim.from_xml_string``
(robosuite ``environments/base.py``), and both scene families force
``hard_reset=True``, so the processor runs on EVERY reset -- the same cadence
the old per-episode variant xml had. RoboCasa's kitchens already ride this
seam for their built-in ``generative_textures`` flag.

Usage::

    handle = install_appearance(env, cfg)      # after the env exists
    handle.new_episode(seed)                   # before each env.reset()
    ...                                        # handle.report has the choices
    handle.restore()

The randomization is regenerated from ``seed`` on every compile rather than
cached, so a reset retried after a ``RandomizationError`` lands on the same
appearance (the candidate set does not depend on where the objects ended up).
"""

from __future__ import annotations

import numpy as np

from .config import AppearanceConfig
from .engine import randomize_xml
from .ownership import protected_prefixes
from .profiles import AppearanceProfile, profile_for


class AppearanceHandle:
    """A registered xml processor, plus the knobs the caller drives it with."""

    def __init__(self, env, cfg: AppearanceConfig,
                 profile: AppearanceProfile) -> None:
        self.env = env
        self.cfg = cfg
        self.profile = profile
        self.report: dict = {}
        self.seed: int | None = None
        self._protected = protected_prefixes(env)
        self._announced = False

    # -- the processor ------------------------------------------------------ #
    def __call__(self, xml: str) -> str:
        if self.seed is None:
            return xml
        # env.objects changes when distractors are wrapped in, so the
        # protected set is refreshed at compile time rather than pinned. It is
        # also EMPTY before the env's first _load_model, which is why this is
        # the first moment worth announcing it.
        self._protected = protected_prefixes(self.env)
        if not self._announced:
            self._announced = True
            print(f"[appearance] profile {self.profile.name!r}, protected "
                  f"prefixes {list(self._protected)}", flush=True)
        self.report.clear()
        return randomize_xml(xml, np.random.default_rng(self.seed), self.cfg,
                             self.profile, self._protected, self.report)

    # -- driving it --------------------------------------------------------- #
    def new_episode(self, seed: int) -> None:
        """Point the next compile(s) at a fresh appearance."""
        self.seed = int(seed)

    def disable(self) -> None:
        """Leave the next compile(s) un-randomized (the baseline)."""
        self.seed = None
        self.report.clear()

    @property
    def protected(self) -> tuple[str, ...]:
        """The live protected set. Robots and task objects only exist after
        ``_load_model``, so this is empty until the env's first reset."""
        self._protected = protected_prefixes(self.env)
        return self._protected

    def restore(self) -> None:
        processors = getattr(self.env, "_xml_processors", None)
        if processors is not None and self in processors:
            processors.remove(self)


def install_appearance(env, cfg: AppearanceConfig,
                       profile: AppearanceProfile | None = None
                       ) -> AppearanceHandle:
    """Register an appearance randomizer on ``env``; returns its handle.

    ``profile`` defaults to :func:`profiles.profile_for`, which raises
    ``UnsupportedScene`` for an env that is neither MolmoSpaces-style nor a
    robocasa kitchen.
    """
    base = getattr(env, "unwrapped", env)
    handle = AppearanceHandle(base, cfg, profile or profile_for(base))
    base.set_xml_processor(handle)
    return handle
