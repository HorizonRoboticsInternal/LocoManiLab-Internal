"""GR00T-format modality-blob schema: the types and the frame/JSON builders.

Deliberately dependency-free (numpy only). It imports nothing from robocasa or
sim2real, which is what lets ``g1_bridge.modality`` declare each backend's
schema without importing ``robocasa.utils.g1_data_collection`` -- the two used
to form an import cycle that both sides broke with function-local imports.

Schema mode: matches GR00T-format reference datasets exactly (e.g.
``G1WholebodyBendPickMP-v0/meta/modality.json``) -- several named groups packed
into ONE concatenated LeRobot feature ("blob"), with a real ``meta/modality.json``
written declaring each group's byte offset, as opposed to the legacy plain-tuple
mode's one-LeRobot-column-per-signal layout (still used by the SONIC family, see
``TELEOP_FEATURE_SPECS`` in g1_data_collection).

``g1_data_collection`` re-exports every name here; import from either. It must
re-export rather than redefine -- ``build_frame`` does an ``isinstance`` check
against ``ModalityLayout``, which only works if there is exactly one class
object in the process.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np


class ModalityGroup(NamedTuple):
    """One named slice within a ModalityBlob.

    ``name`` is the label written into modality.json (state/action groups in
    different blobs may reuse the same ``name``, e.g. "left_arm" for both the
    current and commanded arm pose -- modality.json keeps state/action in
    separate namespaces). ``source`` is the unique key this group's actual
    per-frame value is looked up under in the ``values`` dict passed to
    ``build_blob_frame_fields`` -- always distinct, even when ``name`` repeats.
    """

    name: str
    dim: int
    source: str
    dtype: str = "float32"
    rotation_type: Optional[str] = None


class ModalityBlob(NamedTuple):
    """Several ModalityGroups concatenated, in order, into one LeRobot column."""

    feature_name: str
    section: str  # "state" or "action"
    groups: tuple

    @property
    def dim(self):
        return sum(g.dim for g in self.groups)


class ModalityLayout(NamedTuple):
    """A controller's full modality-blob schema.

    ``readiness_source`` is the ``values`` key (see ModalityGroup.source)
    whose absence (None, or missing from the dict) means the controller
    hasn't produced its first real command yet -- build_frame() gates
    frame-emission on it, mirroring the plain-tuple mode's
    ``teleop_keys[0]`` convention.
    """

    readiness_source: str
    blobs: tuple


_MODALITY_DEFAULT_FEATURE = {"state": "observation.state", "action": "action"}


def layout_sources(teleop_keys):
    """Every ``values`` key a schema refers to, for either schema mode.

    ``teleop_keys`` is what the per-backend registry stores: a ``ModalityLayout``
    (blob mode) or a plain tuple of signal names (legacy mode). Centralising the
    type test here keeps it out of the call sites, which otherwise each have to
    know that the registry's values are heterogeneous.
    """
    if isinstance(teleop_keys, ModalityLayout):
        return tuple(g.source for blob in teleop_keys.blobs for g in blob.groups)
    return tuple(teleop_keys)


def modality_json_dict(
    blobs, video_feature="observation.images.ego_view", video_key="ego_view"
):
    """Build a GR00T-format modality.json dict from a controller's blobs."""
    modality = {
        "state": {},
        "action": {},
        "video": {video_key: {"original_key": video_feature}},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    for blob in blobs:
        offset = 0
        for g in blob.groups:
            entry = {"start": offset, "end": offset + g.dim}
            if blob.feature_name != _MODALITY_DEFAULT_FEATURE[blob.section]:
                entry["original_key"] = blob.feature_name
            if g.rotation_type:
                entry["rotation_type"] = g.rotation_type
            modality[blob.section][g.name] = entry
            offset += g.dim
    return modality


def blob_features(blobs):
    """LeRobot feature specs (dtype/shape) for each blob's concatenated column."""
    return {
        blob.feature_name: {"dtype": "float32", "shape": (blob.dim,)} for blob in blobs
    }


def build_blob_frame_fields(values, blobs):
    """Concatenate ``values`` (keyed by ModalityGroup.source) into each blob's
    LeRobot column. A missing/None group value zero-fills (mirrors the
    plain-tuple mode's "controller hasn't produced this signal yet" handling)."""
    out = {}
    for blob in blobs:
        parts = []
        for g in blob.groups:
            v = values.get(g.source)
            parts.append(
                np.zeros(g.dim, dtype=g.dtype)
                if v is None
                else np.asarray(v, dtype=g.dtype).reshape(-1)
            )
        out[blob.feature_name] = np.concatenate(parts).astype(np.float32)
    return out
