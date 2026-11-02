---
title: Download Artifacts
slug: /reference/artifacts
---

# Download Artifacts

A fresh clone already contains controller manifests and the shared robot description. Source checks, configuration review and pure CPU unit tests need no model download. Install native dependencies and weights only for the workflow you will run.

## Checkpoint manifests and weights

`checkpoints/**/policy.yaml` and associated JSON/YAML metadata are versioned with the runtime. They define joint order, scales and observation layout. Weights are stored separately in the `checkpoint/controller/` folder of the LocoManiLab release repository [GammaRobotics/LocoManiLab](https://huggingface.co/datasets/GammaRobotics/LocoManiLab), which mirrors `checkpoints/`. Download **one selected variant** from the repository root into a staging directory and copy it into `checkpoints/`, for example HEFT:

```bash
hf download GammaRobotics/LocoManiLab --repo-type dataset --local-dir /tmp/g1-weights \
  --include 'checkpoint/controller/heft/pmg/policy.onnx'
cp -r /tmp/g1-weights/checkpoint/controller/. checkpoints/
```

Use these filters for other controllers:

| Controller / variant | Weight paths relative to `checkpoints/` (prefix each with `checkpoint/controller/` in `--include`) |
| --- | --- |
| SONIC low-latency | `'sonic/low_latency/g1/model_encoder.onnx' 'sonic/low_latency/g1/model_decoder.onnx'` |
| SONIC v1.1 | `'sonic/v1_1/g1/model_encoder.onnx' 'sonic/v1_1/g1/model_decoder.onnx'` |
| HEFT PMG | `'heft/pmg/policy.onnx'` |
| ScaleBFM M | `'scalebfm/humanoid_transformer_m/policy.onnx'` |
| AMO | `'amo/amo_jit.pt' 'amo/adapter_jit.pt' 'amo/adapter_norm_stats.pt'` |
| ALMI | `'almi/flat_dual/lower.onnx' 'almi/flat_dual/upper.onnx'` |

Only filter weight files. Older artifact revisions also contain manifests; downloading those over the checkout can silently replace the layout expected by its code. Keep custom exports under separate paths and retain their matching manifests.

From a `humanoid_benchmark` checkout, its downloader finds the backend root, filters weight files and skips existing downloads:

```bash
python -m robocasa.scripts.download_g1_checkpoints --family heft
python -m robocasa.scripts.download_g1_checkpoints --family sonic
```

Run `--help` there for current family options. Do not use an all-family download for ordinary setup.

## SONIC variants

| Variant | Manifest directory | Encoder contract |
| --- | --- | --- |
| Default low-latency | `checkpoints/sonic/low_latency/g1` | 1247D, body-frame anchors, step1 reference |
| v1.1 | `checkpoints/sonic/v1_1/g1` | 1751D, heading-normalized anchors, step5 reference |

Both use a 64D token and a 994→29 decoder shape, but the token meanings and decoder weights differ. **Collect and deploy with the same variant.** The default stays low-latency for existing datasets.

PLANNER mode additionally needs the shared `sonic/low_latency/g1/planner_sonic.onnx`, including when using v1.1. Add this exact filter only if that mode is needed:

```bash
hf download GammaRobotics/LocoManiLab --repo-type dataset --local-dir /tmp/g1-weights \
  --include 'checkpoint/controller/sonic/low_latency/g1/planner_sonic.onnx'
cp -r /tmp/g1-weights/checkpoint/controller/. checkpoints/
```

LocoManiLab's `lab eval --check` checks the selected runtime and assets before evaluation. It does not make tokens from different SONIC variants compatible.

## Optional native dependencies

`third_party/wheels/` and `third_party/prebuilt/` are optional local artifact directories, not a bundle every user must download. Generic CPU inference needs neither. G1-specific packages are installed only with the relevant extra:

```bash
uv sync --extra inference-cpu --extra robot-g1
```

Choose `inference-gpu` instead of `inference-cpu` only for a supported GPU deployment. Do not install both ONNX Runtime extras together. PICO on JetPack 5 may need platform-specific prebuilt libraries under `third_party/prebuilt/jetpack5-aarch64/`; see [onboard PICO setup](/getting-started/teleop-onboard-orin) and the linked source-build guides. This is separate from simulation data collection.

Do not commit downloaded weights, native binaries, credentials or private training outputs. Real-robot deployments use `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1`; prepare the selected files beforehand and transfer them to an explicitly chosen host/path. Machine-specific synchronization and upload workflows are kept outside this repository.
