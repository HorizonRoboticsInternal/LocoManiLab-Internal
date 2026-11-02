"""Phase 0 plumbing for closed-loop Kimodo -> SONIC in the tele G1 sim.

Sub-packages / modules (built incrementally, each independently testable):
- gen.py        : 0a  Kimodo-G1 chunk generation -> MuJoCo qpos (30 Hz)
- viz.py        : offscreen MuJoCo rendering of a qpos sequence to mp4/gif
- (later) ref_convert / track / readback / closed_loop

See rl/kimodo/README.md for how to run each functional test + visualization.
"""
