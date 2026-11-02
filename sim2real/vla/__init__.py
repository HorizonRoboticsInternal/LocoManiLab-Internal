"""Driving a whole-body policy from a VLA served over the GR00T contract.

Three pieces, deliberately separate:

* :mod:`~sim2real.vla.client` -- talk to the policy server.
* :mod:`~sim2real.vla.driver` -- reconcile a 2.5-10 Hz policy with a 50 Hz
  controller (chunking, latency, smoothing).
* :mod:`~sim2real.vla.adapters` -- deliver the action into whichever command
  space the controller consumes.

Only the last one knows anything about controllers, and it has one
implementation per *command space*, not per controller -- which is why adding
a policy family costs nothing here as long as it consumes a space that already
exists. See ``sim2real.rl_policy.families``.
"""

from sim2real.vla.adapters import (
    ActionAdapter,
    check_action_keys,
    command_spaces,
    get_adapter,
)
from sim2real.vla.client import PolicyClient, action_field, as_chunk
from sim2real.vla.driver import ReplayPolicy, VlaDriver, VlaDriverConfig, ZeroPolicy
from sim2real.vla.observation import build_observation, split_state

__all__ = [
    "ActionAdapter",
    "PolicyClient",
    "ReplayPolicy",
    "VlaDriver",
    "VlaDriverConfig",
    "ZeroPolicy",
    "action_field",
    "as_chunk",
    "build_observation",
    "check_action_keys",
    "command_spaces",
    "get_adapter",
    "split_state",
]
