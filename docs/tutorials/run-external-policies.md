---
title: Run External Policies
slug: /tutorials/run-external-policies
---

# Run External Policies

The tracked manifests support five shared controller families. Download only the chosen variant's weights using [Download Artifacts](/reference/artifacts).

| Family | Default manifest | External input |
| --- | --- | --- |
| SONIC | `checkpoints/sonic/low_latency/g1/policy.yaml` | SONIC motion token; the bridge builds matching encoder input |
| HEFT | `checkpoints/heft/pmg/policy.yaml` | `ref_qpos` reference motion |
| ScaleBFM | `checkpoints/scalebfm/humanoid_transformer_m/policy.yaml` | `ref_qpos` reference motion |
| AMO | `checkpoints/amo/policy.yaml` | `velocity_cmd` commands |
| ALMI | `checkpoints/almi/flat_dual/policy.yaml` | `twist_ee` dual-branch commands |

HEFT and ScaleBFM share a reference-motion contract. SONIC tokens, AMO commands and ALMI commands need their matching backend and dataset schema; changing only `--policy-config` cannot translate between them. SONIC v1.1 is an opt-in variant at `checkpoints/sonic/v1_1/g1/policy.yaml`, with tokens that differ from low-latency.

For standalone HEFT motion playback in simulation, follow [Offline Motion Tracking](/tutorials/offline-motion-tracking). For simulator-driven collection, use humanoid_benchmark/LocoManiLab; for the shared runtime API and real collection, see `g1_bridge/README.md` in this checkout.

## Adapting another controller

The optional `.agents/skills/adapt-policy-to-sim2real/` helper documents how to trace the source observations, joint order, action scales and export pipeline. Its conversion scripts are maintained in that directory. Supply your own matching manifest and exported weights; historical examples do not imply the corresponding checkpoint is distributed by this fork.

Preserve controller-specific input names, normalization, history order and joint layout. Verify an export numerically against its source implementation before deployment. Do not select a checkpoint on shape alone.

For SONIC SMPL-specific observations and streams, see [SONIC SMPL Input](/reference/sonic-smpl-input). Such specialized exports require a matching manifest and weights; this checkout distributes G1 manifests only.
