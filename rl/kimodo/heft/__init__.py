"""Kimodo x HEFT inside robocasa.

The RL fine-tuning of Kimodo happened in mjlab against HEFT's frozen G1 tracking
policy (see ../../../heft_work). Every task scene, object asset, success
predicate and dataset exporter lives here instead, on robosuite/MuJoCo-CPU. This
package is the port of the tracker side into that simulator, so the fine-tuned
weights stay valid while gaining access to actual tasks.
"""
