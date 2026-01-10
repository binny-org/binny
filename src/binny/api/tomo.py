"""API for tomographic binning functions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from binny.ztomo.photoz import build_photoz_bins
from binny.ztomo.specz import build_specz_bins

__all__ = [
    "photoz_bins",
    "specz_bins",
    "tomo_bins",
]


def photoz_bins(z: Any, nz: Any, bin_edges: Any, **params: Any):
    return build_photoz_bins(z=z, nz=nz, bin_edges=bin_edges, **params)


def specz_bins(z: Any, nz: Any, bin_edges: Any, **params: Any):
    return build_specz_bins(z=z, nz=nz, bin_edges=bin_edges, **params)


def tomo_bins(
    kind: str,
    z: Any,
    nz: Any,
    bin_edges: Any,
    *,
    params: Mapping[str, Any] | None = None,
):
    p = dict(params or {})
    k = kind.lower()
    if k in {"photoz", "photo"}:
        return photoz_bins(z, nz, bin_edges, **p)
    if k in {"specz", "spec"}:
        return specz_bins(z, nz, bin_edges, **p)
    raise ValueError("kind must be 'photoz' or 'specz'.")
