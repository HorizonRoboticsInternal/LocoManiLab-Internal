"""Bridge between robocasa/robosuite and the external ``sim2real`` package.

Deliberately empty of imports: ``g1_bridge.modality`` is meant to be
importable without the sim2real environment (torch, onnxruntime, pinocchio),
and anything imported here would be pulled in by every submodule import.
Modules that do import ``sim2real`` must import ``g1_bridge._cuda_order``
first -- see that module for why.
"""
