"""Module for computing bin edges based on different strategies."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Sequence, Mapping
from typing import Any

from src.binny.utils.validation import (
    validate_n_bins,
    validate_interval,
    validate_axis_and_weights,
    validate_mixed_segments,
    resolve_binning_method,
)

__all__ = [
    "equidistant_edges",
    "equal_number_edges",
    "log_edges",
    "equidistant_chi_edges",
    "equal_information_edges",
    "geometric_edges_n",
    "mixed_edges",
]


def equidistant_edges(x_min: float, x_max: float, n_bins: int) -> np.ndarray:
    """Returns uniformly spaced bin edges between x_min and x_max (inclusive).

    Works for any 1D axis: redshift, ell, k, etc. Useful for linear scales.

    Args:
        x_min: Minimum value of the axis.
        x_max: Maximum value of the axis.
        n_bins: Number of bins.

    Returns:
        Array of bin edges of shape (n_bins + 1,).
    """
    validate_interval(x_min, x_max, n_bins, log=False)

    if n_bins == 1:
        return np.array([x_min, x_max], dtype=float)

    return np.linspace(x_min, x_max, n_bins + 1)


def log_edges(x_min: float, x_max: float, n_bins: int) -> np.ndarray:
    """Returns log-spaced bin edges between x_min and x_max (inclusive).

    Works for any 1D axis: redshift, ell, k, etc. Useful for logarithmic scales.

    Args:
        x_min: Minimum value of the axis (must be > 0).
        x_max: Maximum value of the axis.
        n_bins: Number of bins.

    Returns:
        Array of bin edges of shape (n_bins + 1,).
    """
    validate_interval(x_min, x_max, n_bins, log=True)

    if n_bins == 1:
        return np.array([x_min, x_max], dtype=float)

    return np.logspace(np.log10(x_min), np.log10(x_max), n_bins + 1)


def equal_number_edges(x: ArrayLike, weights: ArrayLike, n_bins: int) -> np.ndarray:
    """Returns bin edges such that each bin contains the same integrated *number* (or weight).

    Works for any 1D axis: redshift, ell, k, etc. Useful for binning data with varying density.

    Args:
        x: 1D array of axis values.
        weights: 1D array of weights (e.g., counts) corresponding to `x
        n_bins: Number of bins.

    Returns:
        Array of bin edges of shape (n_bins + 1,).
    """
    validate_n_bins(n_bins)
    return _equal_weight_edges(x, weights, n_bins)


def equal_information_edges(
    x: ArrayLike,
    info_density: ArrayLike,
    n_bins: int,
) -> np.ndarray:
    """Returns bin edges such that each bin contains the same integrated *information*.

    Works for any 1D axis: redshift, ell, k, etc. Useful for binning data with
    varying information content. Conceptually identical to `equal_number_edges`,
    but the weights represent an "information density" instead of counts.

    Args:
        x: 1D array of axis values.
        info_density: 1D array of information density corresponding to `x`.
        n_bins: Number of bins.

    Returns:
        Array of bin edges of shape (n_bins + 1,).
    """
    validate_n_bins(n_bins)
    return _equal_weight_edges(x, info_density, n_bins)


def equidistant_chi_edges(
    z: ArrayLike,
    chi: ArrayLike,
    n_bins: int,
) -> np.ndarray:
    """Returns bin edges that are uniformly spaced in comoving distance chi, but returned in z.

    Assumes chi(z) is monotonic and `z` and `chi` are matched 1D arrays.
    Useful for cosmological applications where uniform spacing in comoving distance is desired.

    Args:
        z: 1D array of redshift values.
        chi: 1D array of comoving distance values corresponding to `z`.
        n_bins: Number of bins.

    Returns:
        Array of bin edges in redshift of shape (n_bins + 1,).
    """
    validate_n_bins(n_bins)

    z_arr, chi_arr = validate_axis_and_weights(z, chi)

    chi_edges = np.linspace(chi_arr[0], chi_arr[-1], n_bins + 1)
    z_edges = np.interp(chi_edges, chi_arr, z_arr)
    return z_edges


def geometric_edges_n(x_min: float, x_max: float, n_bins: int) -> np.ndarray:
    """Retruns geometrically (log-spaced) bin edges between x_min and x_max with n_bins bins.

    Works for any 1D axis: redshift, ell, k, etc. Useful for logarithmic scales.

    Args:
        x_min: Minimum value of the axis (must be > 0).
        x_max: Maximum value of the axis (must be > 0 and > x_min).
        n_bins: Number of bins.

    Returns:
        Array of length n_bins + 1 with geometrically spaced bin edges.
    """
    validate_interval(x_min, x_max, n_bins, log=True)
    return np.geomspace(x_min, x_max, n_bins + 1, dtype=float)


def _cumulative_trapz(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Computes a cumulative integral of `weights` over `x` using the trapezoidal rule.

    Args:
        x: 1D array of axis values.
        weights: 1D array of weights corresponding to `x`.

    Returns:
        1D array of cumulative integral values at each point in `x`.
    """
    # x, weights already checked by validate_axis_and_weights
    cumul = np.empty_like(weights, dtype=float)
    cumul[0] = 0.0

    # mid-bin values times bin widths
    mid = 0.5 * (weights[1:] + weights[:-1]) * np.diff(x)
    cumul[1:] = np.cumsum(mid)
    return cumul


def _equal_weight_edges(x: ArrayLike, weights: ArrayLike, n_bins: int) -> np.ndarray:
    """Returns bin edges such that each bin contains the same integrated weight.

    Args:
        x: 1D array of axis values.
        weights: 1D array of weights corresponding to `x`.
        n_bins: Number of bins.

    Returns:
        Array of bin edges of shape (n_bins + 1).
    """
    x_arr, w_arr = validate_axis_and_weights(x, weights)
    cumul_trapz = _cumulative_trapz(x_arr, w_arr)
    total = cumul_trapz[-1]

    if total <= 0:
        raise ValueError("Total weight must be positive for equal-weight binning.")

    edges = [x_arr[0]]
    for i in range(1, n_bins):
        target = (i / n_bins) * total
        edge = np.interp(target, cumul_trapz, x_arr)
        edges.append(edge)
    edges.append(x_arr[-1])
    return np.array(edges, dtype=float)





def mixed_edges(
    segments: Sequence[Mapping[str, Any]],
    *,
    # global arrays that segments can reference
    x: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    info_density: ArrayLike | None = None,
    z: ArrayLike | None = None,
    chi: ArrayLike | None = None,
    total_n_bins: int | None = None,
) -> np.ndarray:
    """Build bin edges by stitching together multiple binning strategies.

    Parameters
    ----------
    segments
        Sequence of segment dicts. Each segment must have:
          - 'method': str  (with aliases; e.g. 'eq', 'equal_number', 'chi', ...)
          - 'n_bins': int
          - optional 'params': mapping of extra kwargs for that method.

        Example:
            segments = [
                {"method": "eq", "n_bins": 3,
                 "params": {"x_min": 0.0, "x_max": 1.0}},
                {"method": "equal_number", "n_bins": 2},
            ]

    x, weights, info_density, z, chi
        Optional global arrays passed to methods that need them, unless overridden
        by per-segment parameters.

        For example, a segment with method 'equal_number' will use:
            x_seg = seg_params.get("x", x)
            w_seg = seg_params.get("weights", weights)

    total_n_bins
        Optional check that the sum of all segment['n_bins'] equals total_n_bins.

    Returns
    -------
    edges : np.ndarray
        Global bin edges array of length sum_i n_bins_i + 1.
    """
    validate_mixed_segments(segments, total_n_bins=total_n_bins)

    all_edges: list[np.ndarray] = []

    for i, seg in enumerate(segments):
        method = resolve_binning_method(seg["method"])
        n_bins = int(seg["n_bins"])
        params: Mapping[str, Any] = seg.get("params", {}) or {}

        # Decide which low-level function to call and with which arguments
        if method == "equidistant":
            x_min = params.get("x_min")
            x_max = params.get("x_max")
            if x_min is None or x_max is None:
                raise ValueError(
                    f"Segment {i} (equidistant) requires 'x_min' and 'x_max' in params."
                )
            edges = equidistant_edges(float(x_min), float(x_max), n_bins)

        elif method == "log":
            x_min = params.get("x_min")
            x_max = params.get("x_max")
            if x_min is None or x_max is None:
                raise ValueError(
                    f"Segment {i} (log) requires 'x_min' and 'x_max' in params."
                )
            edges = log_edges(float(x_min), float(x_max), n_bins)

        elif method == "equal_number":
            x_seg = params.get("x", x)
            w_seg = params.get("weights", weights)
            if x_seg is None or w_seg is None:
                raise ValueError(
                    f"Segment {i} (equal_number) requires 'x' and 'weights' either "
                    f"in params or as global x/weights."
                )
            edges = equal_number_edges(x_seg, w_seg, n_bins)

        elif method == "equal_information":
            x_seg = params.get("x", x)
            info_seg = params.get("info_density", info_density)
            if x_seg is None or info_seg is None:
                raise ValueError(
                    f"Segment {i} (equal_information) requires 'x' and 'info_density' "
                    f"either in params or as global x/info_density."
                )
            edges = equal_information_edges(x_seg, info_seg, n_bins)

        elif method == "equidistant_chi":
            z_seg = params.get("z", z)
            chi_seg = params.get("chi", chi)
            if z_seg is None or chi_seg is None:
                raise ValueError(
                    f"Segment {i} (equidistant_chi) requires 'z' and 'chi' either "
                    f"in params or as global z/chi."
                )
            edges = equidistant_chi_edges(z_seg, chi_seg, n_bins)

        elif method == "geometric":
            x_min = params.get("x_min")
            x_max = params.get("x_max")
            if x_min is None or x_max is None:
                raise ValueError(
                    f"Segment {i} (geometric) requires 'x_min' and 'x_max' in params."
                )
            edges = geometric_edges_n(float(x_min), float(x_max), n_bins)

        else:
            # defensive, should never happen because _resolve_binning_method validates
            raise RuntimeError(f"Unhandled binning method {method!r} in mixed_edges.")

        # Stitch segments: keep left edge of first, skip left edge of others
        if i == 0:
            all_edges.append(edges)
        else:
            all_edges.append(edges[1:])

    return np.concatenate(all_edges, axis=0)
