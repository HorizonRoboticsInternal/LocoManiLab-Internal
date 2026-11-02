"""Compatibility re-export of the modality-blob schema.

The types moved to ``g1_bridge.recording.schema`` when the recording layer became
host-agnostic (a real-robot collector writes the same datasets as the sim one, and
neither should have to import robocasa to know what a ``ModalityBlob`` is).

This module must stay a *re-export* and never a redefinition: ``build_frame`` does
an ``isinstance(..., ModalityLayout)`` check, which only works while there is
exactly one such class object in the process.
"""

from g1_bridge.recording.schema import (  # noqa: F401
    ModalityBlob,
    ModalityGroup,
    ModalityLayout,
    blob_features,
    build_blob_frame_fields,
    layout_sources,
    modality_json_dict,
)
