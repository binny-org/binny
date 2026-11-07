"""Resolve and validate tomographic bin edges."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from binny.axes.bin_edges import equal_number_edges, equidistant_edges
from binny.nz_tomo.nz_tomo_utils import mixed_edges_from_segments
from binny.utils.normalization import normalize_1d
from binny.utils.types import BinningScheme, FloatArray
from binny.utils.validators import validate_axis_and_weights, validate_n_bins

__all__ = ["resolve_bin_edges", "validate_bin_edges"]


def resolve_bin_edges(
    *,
    z_axis: FloatArray,
    nz_axis: FloatArray,
    bin_edges: Any | None,
    binning_scheme: BinningScheme | None,
    n_bins: int | None,
    bin_range: tuple[float, float] | None = None,
    equal_number_axis: Any | None = None,
    equal_number_weights: Any | None = None,
    z_ph: Any | None = None,
    nz_ph: Any | None = None,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    normalize_equal_number_weights: bool = True,
) -> FloatArray:
    """Returns tomographic bin edges.

    Args:
        z_axis: Default axis used when edges are derived from the input range.
        nz_axis: Default weights used when edges are derived from weights.
        bin_edges: Explicit bin edges. Mutually exclusive with `binning_scheme`
            and `n_bins`.
        binning_scheme: Binning description. May be a scheme name (e.g.
            `"equidistant"`, `"equal_number"`) or a mixed-segment specification.
        n_bins: Number of bins for simple scheme names.
        bin_range: Optional `(min, max)` range for equidistant binning.
        equal_number_axis: Optional axis used for equal-number binning.
        equal_number_weights: Optional weights used for equal-number binning.
        z_ph: Optional axis used by mixed-segment specifications.
        nz_ph: Optional weights used by mixed-segment specifications.
        norm_method: Normalization method used when normalizing weights.
        normalize_equal_number_weights: Whether to normalize the weights used
            for equal-number binning.

    Returns:
        A 1D array of bin edges.

    Raises:
        ValueError: If inputs are inconsistent or the binning specification is
            not supported.
    """
    z_arr, nz_arr = validate_axis_and_weights(z_axis, nz_axis)

    if bin_edges is not None:
        if (binning_scheme is not None) or (n_bins is not None):
            raise ValueError("Provide either bin_edges or (binning_scheme, n_bins), not both.")
        return np.asarray(bin_edges, dtype=float)

    if binning_scheme is None:
        raise ValueError("bin_edges is None. You must provide binning_scheme.")

    if isinstance(binning_scheme, str):
        if n_bins is None:
            raise ValueError(
                "bin_edges is None and binning_scheme is a string. You must provide n_bins."
            )
        validate_n_bins(n_bins)

        scheme = binning_scheme.lower()
        if scheme in {"equidistant", "eq", "linear"}:
            if bin_range is not None:
                lo, hi = float(bin_range[0]), float(bin_range[1])
                if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                    raise ValueError("bin_range must be finite with bin_range[0] < bin_range[1].")
                return equidistant_edges(lo, hi, n_bins)
            return equidistant_edges(float(z_arr[0]), float(z_arr[-1]), n_bins)

        if scheme in {"equal_number", "equipopulated", "en"}:
            if (equal_number_axis is None) ^ (equal_number_weights is None):
                raise ValueError(
                    "Provide both equal_number_axis and equal_number_weights, or neither."
                )

            if equal_number_axis is None:
                x, w = z_arr, nz_arr
            else:
                x, w = validate_axis_and_weights(equal_number_axis, equal_number_weights)

            if normalize_equal_number_weights:
                total = float(np.trapezoid(w, x=x))
                if total > 0.0 and not np.isclose(total, 1.0, rtol=1e-3, atol=1e-4):
                    w = normalize_1d(x, w, method=norm_method)

            return equal_number_edges(x, w, n_bins)

        raise ValueError(
            "Unsupported binning_scheme. Supported: "
            "'equidistant' (eq/linear) and 'equal_number' (equipopulated/en)."
        )

    if n_bins is not None:
        raise ValueError("In mixed binning mode, set n_bins per segment and leave n_bins=None.")

    if isinstance(binning_scheme, Mapping):
        if "segments" not in binning_scheme:
            raise ValueError(
                "Mixed binning dict must contain key 'segments' (e.g. {'segments': [...] })."
            )
        segments = binning_scheme["segments"]
    else:
        segments = binning_scheme

    if not isinstance(segments, Sequence) or isinstance(segments, str | bytes):
        raise ValueError("Mixed binning requires a sequence of segment dicts.")

    return mixed_edges_from_segments(
        segments,
        z_axis=z_arr,
        nz_axis=nz_arr,
        z_ph=z_ph,
        nz_ph=nz_ph,
    )


def validate_bin_edges(
    bin_edges: Any,
    *,
    require_within: tuple[float, float] | None = None,
) -> FloatArray:
    """Validates a bin-edge array.

    Args:
        bin_edges: Candidate bin edges.
        require_within: Optional `(min, max)` interval that the edges must lie
            within.

    Returns:
        A validated 1D array of bin edges as ``float64``.

    Raises:
        ValueError: If the edges are not 1D, not finite, not strictly
            increasing, or fall outside `require_within`.
    """
    be = np.asarray(bin_edges, dtype=float)

    if be.ndim != 1:
        raise ValueError("bin_edges must be 1D.")
    if be.size < 2:
        raise ValueError("bin_edges must have at least two entries.")
    if not np.all(np.isfinite(be)):
        raise ValueError("bin_edges must contain only finite values.")
    if not np.all(np.diff(be) > 0):
        raise ValueError("bin_edges must be strictly increasing.")

    if require_within is not None:
        lo, hi = float(require_within[0]), float(require_within[1])
        if be[0] < lo or be[-1] > hi:
            raise ValueError(f"bin_edges must lie within [{lo}, {hi}], got [{be[0]}, {be[-1]}].")

    n_bins_eff = int(be.size - 1)
    validate_n_bins(n_bins_eff)
    return be.astype(np.float64, copy=False)
