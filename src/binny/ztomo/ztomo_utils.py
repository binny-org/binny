"""Uitility functions for z tomography binning schemes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from binny.axes.bin_edges import equal_number_edges, equidistant_edges
from binny.utils.normalization import normalize_1d
from binny.utils.validators import (
    validate_axis_and_weights,
    validate_n_bins,
)

FloatArray: TypeAlias = NDArray[np.float64]
BinningScheme: TypeAlias = str | Sequence[Mapping[str, Any]] | Mapping[str, Any]

__all__ = [
    "mixed_edges",
]


def mixed_edges(
    segments: Sequence[Mapping[str, Any]],
    *,
    z_axis: FloatArray,
    nz_axis: FloatArray,
    z_ph: FloatArray | None,
    nz_ph: FloatArray | None,
    normalize_input: bool,
    norm_method: Literal["trapezoid", "simpson"],
) -> np.ndarray:
    """Builds observed-z bin edges from mixed per-segment binning schemes.

    Each segment is a mapping with keys:
      - "z_min": float
      - "z_max": float
      - "scheme": str  ("equidistant"/"equal_number" and aliases)
      - "n_bins": int  (number of bins in this segment)

    Segments must be contiguous and increasing: segment[j].z_min == segment[j-1].z_max.
    """
    if len(segments) == 0:
        raise ValueError("binning_scheme segments must be a non-empty sequence.")

    # choose axis/dist used for equal-number edges
    if (z_ph is None) ^ (nz_ph is None):
        raise ValueError(
            "Provide both z_ph and nz_ph, or neither. "
            "If neither is provided, (z, nz) is used as a proxy for (z_ph, nz_ph)."
        )

    if z_ph is not None:
        x_full, w_full = validate_axis_and_weights(z_ph, nz_ph)
    else:
        x_full, w_full = z_axis, nz_axis

    edges_all: list[np.ndarray] = []
    prev_zmax: float | None = None

    for j, seg in enumerate(segments):
        try:
            z_min = float(seg["z_min"])
            z_max = float(seg["z_max"])
            scheme = str(seg["scheme"]).lower()
            nseg = int(seg["n_bins"])
        except KeyError as e:
            raise ValueError(f"Missing key in mixed bin spec: {e.args[0]!r}") from e

        validate_n_bins(nseg)

        if not (np.isfinite(z_min) and np.isfinite(z_max)):
            raise ValueError("Segment z_min/z_max must be finite.")
        if z_max <= z_min:
            raise ValueError("Each segment must satisfy z_max > z_min.")

        if prev_zmax is not None:
            if z_min < prev_zmax:
                raise ValueError(
                    "Segments must be non-overlapping and in increasing order."
                )
            if not np.isclose(z_min, prev_zmax, rtol=0.0, atol=1e-10):
                raise ValueError(
                    "Segments must be contiguous: each z_min must equal previous z_max."
                )

        if scheme in {"equidistant", "eq", "linear"}:
            edges = equidistant_edges(z_min, z_max, nseg)

        elif scheme in {"equal_number", "equipopulated", "en"}:
            m = (x_full >= z_min) & (x_full <= z_max)
            if np.count_nonzero(m) < 2:
                raise ValueError(
                    "Segment has too few points to compute equal-number edges."
                )

            x_seg = x_full[m]
            w_seg = w_full[m]

            if normalize_input:
                total = np.trapezoid(w_seg, x=x_seg)
                if not np.isclose(total, 1.0, rtol=1e-3, atol=1e-4):
                    w_seg = normalize_1d(x_seg, w_seg, method=norm_method)

            edges = equal_number_edges(x_seg, w_seg, nseg)

            # numeric safety: force exact segment endpoints
            edges[0] = z_min
            edges[-1] = z_max

        else:
            raise ValueError(
                "Unsupported segment scheme. Supported: "
                "'equidistant' (eq/linear) and 'equal_number' (equipopulated/en)."
            )

        if j > 0:
            edges = edges[1:]  # drop duplicate boundary
        edges_all.append(edges)
        prev_zmax = z_max

    out = np.concatenate(edges_all)

    if not np.all(np.isfinite(out)):
        raise ValueError("Mixed bin edges contain non-finite values.")
    if not np.all(np.diff(out) > 0):
        raise ValueError(
            "Mixed bin edges are not strictly increasing. Check segment specs."
        )

    return out
