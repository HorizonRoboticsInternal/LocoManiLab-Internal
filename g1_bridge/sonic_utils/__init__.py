"""SONIC-specific helpers for the g1_bridge runtime.

Importing this package registers ``sonic.sonic_g1_low_latency_encoder_input``
and ``sonic.sonic_g1_v1_1_encoder_input`` into sim2real's Observation registry
(an import side effect of
``encoder_obs``) -- ``bridge_runtime.py`` relies on exactly that when it does
``import g1_bridge.sonic_utils`` before constructing a tracking from a
policy.yaml that references the observation. Without this import the
registration only happened when ``backends/sonic.py`` was pulled in first,
so ``make_bridge_tracking()`` called on its own raised "Observation target
not found".

``kplanner``/``ckpt`` are NOT imported here: they pull in ONNX runtime and
checkpoint I/O that callers who only need the observation registration
should not pay for.
"""

from g1_bridge.sonic_utils import encoder_obs  # noqa: F401 — registration side effect
