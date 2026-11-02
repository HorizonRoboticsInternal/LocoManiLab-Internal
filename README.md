# g1_backend

Shared G1 controller and teleoperation runtime for [LocoManiLab](https://github.com/LocoManiLab/LocoManiLab) and [humanoid_benchmark](https://github.com/LocoManiLab/humanoid_benchmark).

[中文说明](README_zh.md) · [Documentation](docs/README.md) · [Teleoperation and recording](g1_bridge/README.md)

- `g1_bridge/` provides the controller session, PICO input, host adapters and training-data recording shared by simulation and real-robot collection.
- `sim2real/` provides inference, reference-motion handling, MuJoCo execution and robot I/O. The existing Python namespace is retained for compatibility.
- `checkpoints/` contains versioned manifests; model weights are optional downloads. `assets/g1_unified/` contains the shared robot description.
- `venv/pico/` isolates optional XRoboToolkit/GMR dependencies. Robot SDKs and camera packages are also optional.

This repository builds on the upstream [sim2real deployment stack](https://github.com/EGalahad/sim2real). Its original citation appears below. Fork-specific installation and asset instructions are maintained here.

## Install from a checkout

The standalone environment requires Python 3.10 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/LocoManiLab/g1_backend.git
cd g1_backend
uv sync --extra inference-cpu
```

Use this editable checkout for runtime execution: manifests, robot descriptions and launch scripts live beside the Python packages. This is not a self-contained wheel installation. When using LocoManiLab, follow its `lab` setup and environment instructions instead of installing another copy.

The generic CPU environment does not require robot SDKs, PICO, camera packages or weights. After installation, run a local check:

```bash
uv run --no-sync python -m unittest discover -s tests -p test_math_helpers.py
```

For hardware, install only the extras needed by the selected [Robot I/O mode](docs/robot_io.md). PICO setup is documented separately for [PC](docs/getting-started/teleop-x86-64.md) and [onboard Orin](docs/getting-started/teleop-onboard-orin.md).

## Select a controller and its assets

The five shared controller families are SONIC, HEFT, ScaleBFM, AMO and ALMI:

| Family / variant | Tracked manifest | Input contract |
| --- | --- | --- |
| SONIC | `checkpoints/sonic/low_latency/g1/policy.yaml` | Motion reference / motion token; default SONIC variant |
| SONIC v1.1 | `checkpoints/sonic/v1_1/g1/policy.yaml` | Separate encoder/decoder variant; tokens are not interchangeable with low-latency |
| HEFT | `checkpoints/heft/pmg/policy.yaml` | Motion reference |
| ScaleBFM | `checkpoints/scalebfm/humanoid_transformer_m/policy.yaml` | Motion reference |
| AMO | `checkpoints/amo/policy.yaml` | Controller-specific commands |
| ALMI | `checkpoints/almi/flat_dual/policy.yaml` | Dual-branch controller-specific commands |

Keep the family, manifest and data/action schema together. Do not substitute one controller's weights or recorded actions for another's. No weights are needed for source review or the check above. Download only the chosen family's weights; see [Download Artifacts](docs/artifacts.md). The standalone HEFT repository is not required; the HEFT controller here remains supported.

For simulation collection, follow [humanoid_benchmark](https://github.com/LocoManiLab/humanoid_benchmark) or LocoManiLab's collection guide. For standalone collection, see [g1_bridge](g1_bridge/README.md). For offline motion playback, see the [tracking tutorial](docs/tutorials/offline-motion-tracking.md).

## License

g1_backend is released under the [MIT License](LICENSE), Copyright (c) 2026 Gamma Robotics. Vendored code, robot assets, and controller configurations keep their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the per-directory license files. The citation below is attribution, not a license grant.

## Upstream citation

Retained from the original sim2real project:

```bibtex
@misc{sim2real2026,
  author       = {{RoboParty Lab Team}},
  title        = {sim2real: A Lightweight and Modular Sim2sim and Sim2real Deployment Stack},
  year         = {2026},
  howpublished = {\url{https://github.com/EGalahad/sim2real}},
  note         = {Documentation: \url{https://egalahad.github.io/sim2real/}}
}
```
