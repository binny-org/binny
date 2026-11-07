"""Compatibility exports for legacy tomography binning utilities."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from binny.binning.core import build_bins_on_edges
from binny.binning.edges import resolve_bin_edges, validate_bin_edges
from binny.utils.metadata import build_tomo_bins_metadata, save_metadata_txt
from binny.utils.types import FloatArray

__all__ = [
    "resolve_bin_edges",
    "validate_bin_edges",
    "build_bins_on_edges",
    "finalize_tomo_metadata",
]


def finalize_tomo_metadata(
    *,
    kind: str,
    z: FloatArray,
    parent_nz: FloatArray,
    bin_edges: FloatArray,
    bins: Mapping[int, FloatArray],
    inputs: Mapping[str, Any],
    parent_norm: float | None,
    bins_norms: Mapping[int, float] | None,
    include_metadata: bool,
    save_metadata_path: str | None,
) -> dict[str, Any] | None:
    """Builds tomography metadata and optionally writes it to disk.

    Args:
        kind: Tomography kind label (e.g. `"photoz"`, `"specz"`).
        z: Redshift grid.
        parent_nz: Parent redshift distribution.
        bin_edges: Bin-edge array used to construct the bins.
        bins: Mapping of bin index to bin curve.
        inputs: Input settings to record in the metadata.
        parent_norm: Integral of the parent distribution, if available.
        bins_norms: Per-bin integrals, if available.
        include_metadata: Whether to return the metadata dictionary.
        save_metadata_path: Optional path for writing a text metadata file.

    Returns:
        Metadata dictionary if requested, otherwise `None`.
    """
    need_meta = include_metadata or (save_metadata_path is not None)
    if not need_meta:
        return None

    frac_per_bin: dict[int, float] | None
    if parent_norm is None or parent_norm == 0.0 or bins_norms is None:
        frac_per_bin = None
    else:
        frac_per_bin = {i: float(bins_norms[i] / parent_norm) for i in bins_norms}

    meta = build_tomo_bins_metadata(
        kind=kind,
        z=np.asarray(z, dtype=float),
        parent_nz=np.asarray(parent_nz, dtype=float),
        bin_edges=np.asarray(bin_edges, dtype=float),
        bins_returned=dict(bins),
        inputs=dict(inputs),
        parent_norm=parent_norm,
        bins_norms=dict(bins_norms) if bins_norms is not None else None,
        frac_per_bin=frac_per_bin,
    )

    if save_metadata_path is not None:
        save_metadata_txt(meta, save_metadata_path)

    return meta if include_metadata else None
