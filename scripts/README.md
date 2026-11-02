# Maintenance and operator tools

Run these tools from the repository root, after installing the environment for
the workflow you need. `scripts/` is source-checkout tooling; the shared runtime
and collection APIs live in `sim2real/` and `g1_bridge/`.

For LocoManiLab data collection, start with its `lab collect` guide. For direct
use of this backend, see [the bridge guide](../g1_bridge/README.md) and
[robot I/O modes](../docs/robot_io.md).

The examples below use `--no-sync` to reuse an installed environment. They do
not install dependencies or download model weights. Runtime commands still need
the specific policy manifests, weights, motion files, or devices they consume;
see [artifacts](../docs/artifacts.md) before preparing them.

## Motion and tracking

| Tool | Purpose | Requirements |
| --- | --- | --- |
| `record_motion.py` | Record a live retargeted motion stream as an any4hdmi qpos clip. | A running motion publisher. |
| `view_motion.py` | Open the any4hdmi motion viewer. | A local motion file and the viewer dependencies. |
| `process_qpos_data.py` | Smooth older qpos/timestamp NPZ recordings and compute qvel. | NumPy, SciPy, and tyro; this legacy converter drops the first 10 samples. |
| `tracking_experiment/run_tracking_metrics_eval.py` | Run policy/motion/seed comparisons through the integrated MuJoCo evaluator. | Selected local policies, weights, and motions. |
| `tracking_experiment/compute_tracking_metrics.py` | Recompute metrics from existing full trajectory NPZ files. | NumPy; no policy execution. |
| `tracking_experiment/visualize_root_trajectory.py` | Plot saved root trajectories. | Saved trajectories and plotting dependencies. |
| `tracking_experiment/convert_xrobot_raw_zip_to_any4hdmi.py` | Convert local XRobot raw recordings. | A local raw archive and reference manifest. |

Inspect the available arguments without starting collection:

```bash
uv run --no-sync python -m scripts.record_motion --help
uv run --no-sync python -m scripts.tracking_experiment.compute_tracking_metrics --help
```

The [tracking evaluation guide](tracking_experiment/README.md) documents inputs,
metrics, and examples. Its `run_root_final_error_eval.py`,
`compute_root_final_error.py`, and `plot_root_final_error_bars.py` remain for
reproducing older root-displacement results; use the full tracking metrics for
new comparisons.

## Model preparation

| Tool | Purpose |
| --- | --- |
| `export_almi_onnx.py` | Export a supplied ALMI checkpoint to lower/upper ONNX branches and check parity. |
| `export_scalebfm_onnx.py` | Export a supplied ScaleBFM checkpoint using a local ScaleBFM source checkout. |
| `test_policy_inference.py` | Measure an existing model with an explicitly selected inference backend. |

These tools require the relevant model dependencies and your own local input
files. They are optional preparation or benchmarking tools, not unit tests.
Use the CPU inference backend for CPU-only measurements.

## Robot and camera operation

| Tool | Purpose |
| --- | --- |
| `g1/real_bridge.py` | Bridge Unitree SDK2 Python DDS to the ZMQ policy bus. |
| `g1/real_bridge_cpp.py` | Provide the same ZMQ bridge through `unitree_interface`. |
| `dex3_control.py` | Optional separate Dex3 hand process; normal tracking can control Dex3 in process. |
| `g1/start_orin_camera.sh` | Select/start a camera publisher on a configured Orin over SSH. |
| `ankle_swing.py` | Manual Unitree motor diagnostic that commands physical ankle and wrist motion. |

Use `python -m sim2real.camera.server` for the generic camera publisher. Its
`dummy` source supports camera transport checks without a camera SDK.
The motor diagnostic is a manual hardware tool and is not part of CPU tests.

The Orin camera helper requires `--host user@robot-host` (or `G1_HOST`).
Its `listen` mode additionally needs `--listen address:port` (or `ORIN_LISTEN`)
on the headset network. It checks these settings before connecting over SSH.
Use `--mode headless` when there is no headset; `--status` and `--stop` need
only the host.

## Optional machine setup

| Tool | Purpose |
| --- | --- |
| `setup/setup_xrobot_pybind.sh` | Build the PICO binding from separately prepared local XRoboToolkit sources. |
| `setup/setup_g1_hotspot.sh` | Configure a G1 NetworkManager hotspot with its own upstream interface. |
| `setup/setup_g1_hotspot_via_laptop.sh` | Configure a G1 hotspot whose upstream gateway is a laptop. |
| `setup/setup_laptop_g1_gateway.sh` | Configure the matching laptop Wi-Fi connection and NAT gateway. |

The network tools change the selected machine's network configuration. Read
their `--help` and the [network guide](../docs/getting-started/network-configuration.md)
before running them. None is required for a CPU unit-test run.
