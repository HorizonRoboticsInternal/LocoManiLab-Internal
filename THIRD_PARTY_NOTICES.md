# Third-Party Notices

g1_backend is licensed under the MIT License (see [LICENSE](LICENSE)), except
for the third-party components listed below, which remain under their own
terms. Directory-level license files take precedence over the root license for
the files they cover.

## Upstream project

| Component | Location | Origin | License / status |
|---|---|---|---|
| sim2real deployment stack | `sim2real/`, `docs/`, `scripts/` (original layout) | [EGalahad/sim2real](https://github.com/EGalahad/sim2real), RoboParty Lab Team / sim2real Contributors | The upstream repository does not publish a license. Attribution is retained in [README.md](README.md) ("Upstream citation"). |

## Vendored code and assets

| Component | Location | Origin | License / status |
|---|---|---|---|
| G1 arm / whole-body IK and URDFs | [`g1_bridge/vendored/amo/`](g1_bridge/vendored/amo/) | [Psi0](https://github.com/physical-superintelligence-lab/Psi0) (PSI-lab, USC) and [AMO](https://github.com/OpenTeleVision/AMO) | Apache-2.0, modified. See [LICENSE](g1_bridge/vendored/amo/LICENSE) and [NOTICE](g1_bridge/vendored/amo/NOTICE). |
| WBT / SONIC controllers | [`g1_bridge/vendored/wbt/`](g1_bridge/vendored/wbt/) | [TeleHuman/whole_body_teleop](https://github.com/TeleHuman/whole_body_teleop); SONIC logic derived from [NVIDIA GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl) (Apache-2.0) | Redistribution permission pending confirmation. See [LICENSE](g1_bridge/vendored/wbt/LICENSE). |
| Unitree G1 meshes | [`assets/g1_unified/meshes/`](assets/g1_unified/) | Unitree Robotics G1 description | BSD-3-Clause, Copyright (c) 2016-2022 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"). See [LICENSE](assets/g1_unified/LICENSE). |
| Unified G1 MJCF (`robot.xml`) | [`assets/g1_unified/`](assets/g1_unified/) | Derived from [zengweishuai/ScaleBFM](https://github.com/zengweishuai/ScaleBFM) | License pending confirmation from the author. See [LICENSE](assets/g1_unified/LICENSE). |
| FK MJCF (`robot_fk.xml`) | [`assets/g1_unified/`](assets/g1_unified/) | Derived from [Axellwppr/motion_tracking](https://github.com/Axellwppr/motion_tracking) | MIT, Copyright (c) 2026 Axell. See [LICENSE](assets/g1_unified/LICENSE). |
| HEFT observation / controller support | `sim2real/rl_policy/observations/heft.py`, `checkpoints/heft/` | Derived from [Axellwppr/motion_tracking](https://github.com/Axellwppr/motion_tracking) (HEFT) | MIT, Copyright (c) 2026 Axell. |

## Controller configurations and weights

`checkpoints/` contains only configuration and metadata files for third-party
controllers (SONIC, HEFT, ScaleBFM, AMO, ALMI). Model weights are not included;
they are downloaded separately and remain subject to the terms of their
publishers (for example, NVIDIA's SONIC checkpoints on Hugging Face). The SONIC
configuration files under `checkpoints/sonic/` are reproduced from NVIDIA's
GR00T-WholeBodyControl / GEAR-SONIC release (Apache-2.0).

## Dependencies that are not redistributed

Python dependencies (including git dependencies such as `mjhub` and `any4hdmi`)
and the deployment wheels described in
[third_party/wheels/README.md](third_party/wheels/README.md) are installed from
their own sources and are not part of this repository. They are governed by
their own licenses.
