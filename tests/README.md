# CPU checks

## Minimal installed-wheel resource check

The [CPU workflow](../.github/workflows/cpu.yml) builds a Python 3.10 wheel,
installs it with `--no-deps`, then runs `test_package_resources.py` from a
temporary directory outside the checkout. An isolated Python import also
asserts that the FK module comes from `site-packages`.

This catches resources that exist in an editable checkout but are missing from
the wheel. It reads the shipped G1 URDF, traverses both arm chains, and computes
a finite 14-value end-effector state with normalized quaternions. It needs only
NumPy and SciPy at runtime, plus pytest for the check and `build` for packaging.
It does not install Torch, robot/camera SDKs, model weights, or optional motion
assets, and does not open hardware or run a policy.

To reproduce from the repository root with a separate temporary environment:

```bash
qa_dir=$(mktemp -d)
python3.10 -m venv "$qa_dir/env"
qa_python="$qa_dir/env/bin/python"
"$qa_python" -m pip install build numpy scipy pytest
"$qa_python" -m build --wheel --outdir "$qa_dir/dist"
"$qa_python" -m pip install --no-deps "$qa_dir"/dist/*.whl
cp tests/test_package_resources.py "$qa_dir/"
cd "$qa_dir"
unset PYTHONPATH
"$qa_python" -I -c 'import g1_bridge.recording.g1_fk as fk; print(fk.__file__)'
"$qa_python" -I -m pytest -q test_package_resources.py
```

The printed module path should be under the temporary environment's
`site-packages`, outside your source checkout.

## Broader runtime unit tests

The rest of `tests/` covers controller adapters, observation construction,
message formats, recording, hand helpers, and operator flow. It expects an
already provisioned runtime environment; it is not supported by the minimal
NumPy/SciPy environment above. From the repository root, use:

```bash
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
  uv run --no-sync python -m pytest -q tests
```

Run this in the native Python 3.10 environment declared in `pyproject.toml`.
LocoManiLab's `humanoid_benchmark` integration uses its separately provisioned
Python 3.11 environment; that is a separate compatibility check, not the
backend's declared installation environment.

Optional checkpoint parity cases skip when their local model files are absent.
Missing Python dependencies can fail test collection instead: provision only
the environment needed for the tests you intend to run. Do not download every
controller's weights or SDKs just to remove skips. Hardware and model timing
utilities under `scripts/` are explicit operator commands and are excluded
from automatic pytest discovery by `pytest.ini`.
