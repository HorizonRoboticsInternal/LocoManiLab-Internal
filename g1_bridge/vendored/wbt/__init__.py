"""Vendored copy of selected wbt (whole_body_teleop) controllers.

Files vendored (7 modules + 1 data file):
    controllers/base_controller.py
    controllers/default_pos_controller.py
    controllers/wbt_controller_sonic.py
    controllers/wbt_controller_sonic_smpl.py
    controllers/wbt_controller_sonic_teleop.py
    utils/bydmimic_utils.py
    utils/trt_utils.py
    data/smpl_human_joints_info.npz

This listing is the vendoring boundary -- keep it exact: it is what a cleanup
pass checks before touching anything here. Consumers span two repos (this one
and tele_humanoid_benchmark's robocasa/g1_host/wbt.py), so an in-repo grep
alone does NOT prove a file dead.

These are copied from the wbt project rather than imported via PYTHONPATH so
that this repo is self-contained and does not depend on an external wbt
checkout. Modify with care — upstream changes should be re-synced manually
rather than auto-merged (do not refactor vendored code; whole-file re-sync
only, so diffs against upstream stay meaningful).
"""
