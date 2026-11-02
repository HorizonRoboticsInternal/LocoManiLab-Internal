"""Import torch before ``sim2real``. Import this module first, for that effect.

``sim2real.rl_policy.inference.cuda_runtime.preload_cuda_runtime_libraries()``
runs at ``sim2real`` import time and ``ctypes.CDLL``s the CUDA runtime it finds
on the system. On a box whose ``/usr/local/cuda`` is older than the toolkit the
installed wheels were built against, that fails outright::

    OSError: /usr/local/cuda/lib64/libcusparse.so.12: undefined symbol:
             __nvJitLinkCreate_12_8, version libnvJitLink.so.12

Importing torch first loads the matched ``nvidia-*`` wheel libraries that ship
with it, so sim2real's preload finds a consistent set already resident and
succeeds.

This used to happen by accident: ``amo_g1_composite.py`` imported torch at
module scope and robosuite's ``controllers/composite/__init__.py`` imported that
module eagerly, so torch was always in memory before anything reached sim2real.
That module is gone (AMO now runs as a G1_BRIDGE backend), which turned a
working setup into an import-time crash.

Kept here rather than in ``g1_bridge/__init__.py`` so that
``g1_bridge.modality`` -- which promises to be importable without the
sim2real environment -- does not drag torch in as a side effect of the package
import. Only the modules that actually import ``sim2real`` need this.
"""

import torch as _torch  # noqa: F401  -- see module docstring
