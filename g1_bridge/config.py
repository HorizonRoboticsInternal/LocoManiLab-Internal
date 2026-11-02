"""Backend/policy/PICO resolution for the G1 teleop bridge -- host-agnostic.

``TeleopSession.configure`` takes one dict: the ``composite_controller_specific_configs``
block. Building that block is *not* a robosuite concern -- it picks the backend, resolves
its checkpoint against the g1_backend repo root, merges the backend's own config overlay
and stamps the PICO source -- but it used to live inside ``load_sim2real_controller_config``
in the robosuite composite, which meant the only way to get one was to import robosuite.

The split:

* here: :func:`build_bridge_spec`, the spec dict itself, usable by *any* host
  (the robosuite composite, a real-robot host, a test).
* the robosuite host: wraps this with ``load_composite_controller_config`` so the
  returned dict also carries ``type``/``body_parts``, which only robosuite reads.

The per-backend JSONs under ``configs/`` stay the single source of the *defaults*
(blend seconds, inference backend, kPlanner settings, ...). Both hosts read the same
file, so a real-robot run and a sim run cannot drift apart on a default.
"""

from __future__ import annotations

import json
import pathlib

from g1_bridge.backends import get_backend_class

#: Controller JSONs. Package-level, not next to this module, because the
#: robosuite host loads the same files through its own loader.
CONFIG_DIR = pathlib.Path(__file__).resolve().parent / "configs"

#: Values ``sim2real.rl_policy.inference.build_inference_module`` accepts.
INFERENCE_BACKENDS = ("onnx-cpu", "onnx-gpu", "tensorrt", "torch-jit")

#: The key holding the bridge spec inside a controller JSON. Named by robosuite's
#: composite-controller schema; the real host only ever reads this one block.
SPEC_KEY = "composite_controller_specific_configs"


def sim2real_root() -> pathlib.Path:
    """Repo root of the installed ``sim2real`` package (the dir holding ``checkpoints/``).

    Imports sim2real, so callers that must stay import-light (checkpoint
    downloaders, schema-only consumers) should use
    ``g1_bridge.g1_runtime.sim2real_root`` instead, which uses ``find_spec``.
    """
    import sim2real

    return pathlib.Path(sim2real.__file__).resolve().parent.parent


def config_path(backend: str) -> pathlib.Path:
    """The controller JSON a backend wants (``g1_sonic.json`` / ``g1_generic.json``)."""
    return CONFIG_DIR / get_backend_class(backend).controller_json


def resolve_policy_config_path(backend: str, policy_config: str | None = None) -> str:
    if policy_config:
        return str(pathlib.Path(policy_config).expanduser())
    path = sim2real_root() / get_backend_class(backend).POLICY_CONFIGS[backend]
    if not path.exists():
        raise FileNotFoundError(f"Default policy_config not found at {path}.")
    return str(path)


def build_bridge_spec(
    backend: str,
    pico_source: str,
    policy_config: str | None = None,
    gmr_viewer: bool = False,
    inference_backend: str | None = None,
    amo_arm_source: str | None = None,
) -> dict:
    """The ``composite_controller_specific_configs`` block for one run.

    ``inference_backend`` overrides the controller JSON's
    ``sim2real_inference_backend`` (default ``onnx-cpu``). Note the policy is only
    one of several things competing for the tick budget in sim -- MuJoCo, rendering
    and the GMR subprocess share the machine -- so moving it to the GPU helps the
    policy's own latency, not necessarily the loop rate.
    """
    backend_cls = get_backend_class(backend)
    overlay = backend_cls.config_overlay(pico_source=pico_source, gmr_viewer=gmr_viewer)
    with open(config_path(backend)) as f:
        spec = dict(json.load(f).get(SPEC_KEY, {}))
    spec["sim2real_backend"] = backend
    spec["sim2real_policy_config"] = resolve_policy_config_path(backend, policy_config)
    spec["pico_source"] = pico_source
    if inference_backend is not None:
        if inference_backend not in INFERENCE_BACKENDS:
            raise ValueError(
                f"unknown inference backend {inference_backend!r}; "
                f"expected one of {list(INFERENCE_BACKENDS)}"
            )
        spec["sim2real_inference_backend"] = inference_backend
    if backend == "amo":
        # Resolved here, once, so meta/info.json and the backend cannot
        # disagree about what actually ran. See amo.resolve_arm_source for the
        # precedence and for why the default depends on pico_source.
        from g1_bridge.backends.amo import resolve_arm_source

        spec["amo_arm_source"] = resolve_arm_source(amo_arm_source, pico_source)
    if pico_source in ("mock", "keyboard"):
        spec["pico_source_kwargs"] = {}
    for key, value in overlay.items():
        spec[key] = (
            list(spec.get(key, [])) + list(value) if isinstance(value, list) else value
        )
    return spec
