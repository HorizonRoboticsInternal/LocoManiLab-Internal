---
title: Root Project
sidebar_position: 2
---

Use the root project for inference, tracking policy, MuJoCo simulation, and robot I/O.

## Setup

```bash
uv sync --extra inference-cpu
```

If this machine will run `--robot-io inline` or `scripts/g1/real_bridge_cpp.py`,
install the G1 robot extra as well:

```bash
uv sync --extra inference-cpu --extra robot-g1
```

If `unitree_sdk2py` setup cannot locate `cyclonedds`, refer to the upstream
FAQ:

[https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq](https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq)

Typical error:

```text
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

Compile and install CycloneDDS first:

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
```

Then rerun your environment setup.

## Verify Installation

Run a CPU unit check without weights, robot SDKs or hardware:

```bash
uv run --no-sync python -m unittest discover -s tests -p test_math_helpers.py
```

An optional inference check requires the HEFT weight from
[Download Artifacts](/reference/artifacts):

```bash
uv run --no-sync python scripts/test_policy_inference.py \
  --policy_config checkpoints/heft/pmg/policy.yaml \
  --inference_backend onnx-cpu --single
```

Motor/calibration scripts are separate hardware operations, not installation
checks. Run them only for an explicitly requested hardware procedure.

## Next Steps

- [Offline Motion Tracking](/tutorials/offline-motion-tracking)
- [Robot I/O](/reference/robot-io)
