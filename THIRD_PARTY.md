# Third-party source notices

The main LocoManiLab package is MIT-licensed. Dependencies, robot assets, weights,
and datasets retain their own terms; the root license does not relicense them.

`scripts/aidi/submit_locomanilab_eval.py` was developed as a RoboOrchard submit
entry point and is retained here for reproducible internal delivery. Its Apache
2.0 license and copyright notice are preserved in the file.

The HoloMotion runtime and SIMPLE transport glue are documented in
`external/g1_backend/THIRD_PARTY_NOTICES.md` and
`external/g1_backend/g1_bridge/holomotion_v14/SOURCE.md`.
The Revo2 robot description, meshes and calibration are documented in
`external/humanoid_benchmark/robosuite_src/robosuite/models/assets/robots/g1_revo2/SOURCE.md`.
Both dependencies preserve source revisions and hashes. Their inclusion in an
internal dependency branch does not grant additional redistribution rights.
No model weights or downloaded task asset archives are included in this change.
