"""Module for computing bin edges based on different strategies."""

from __future__ import annotations

from typing import Any

import numpy as np

from binny.utils.validators import (
    validate_axis_and_weights,
    validate_interval,
    validate_n_bins,
)

__all__ = [
    "equidistant_edges",
    "equal_number_edges",
    "log_edges",
    "equidistant_chi_edges",
    "equal_information_edges",
    "geometric_edges",
    "_equal_weight_edges",
    "_cumulative_trapz",
]


def equidistant_edges(x_min: float, x_max: float, n_bins: int) -> np.ndarray:
    """Returns uniformly spaced bin edges between ``x_min`` and ``x_max``.

    This method includes both endpoints.
    Works for any 1D axis: redshift, ell, k, etc. Useful for linear scales.

    Args:
        x_min: Minimum value of the axis.
        x_max: Maximum value of the axis.
        n_bins: Number of bins.

    Returns:
        Array of bin edges of shape (n_bins + 1,), uniformly spaced.
    """
    validate_interval(x_min, x_max, n_bins, log=False)

    if n_bins == 1:
        return np.array([x_min, x_max], dtype=float)

    return np.linspace(x_min, x_max, n_bins + 1)


def log_edges(x_min: float, x_max: float, n_bins: int) -> np.ndarray:
    """Returns logarithmically spaced bin edges between ``x_min`` and
    ``x_max``.

    The returned edges are equally spaced in ``log10(x)`` and include both
    endpoints.

    Notes:
        The edges are equally spaced in ``log10(x)`` (via ``np.logspace``).
        For positive endpoints, this produces the same spacing as geometric
        spacing (constant ratio between successive edges), up to floating-point
        rounding.

    Args:
        x_min: Lower endpoint of the axis (must be > 0).
        x_max: Upper endpoint of the axis (must be > 0 and > x_min).
        n_bins: Number of bins.

    Returns:
        Array of bin edges with shape ``(n_bins + 1,)``.
    """
    validate_interval(x_min, x_max, n_bins, log=True)

    if n_bins == 1:
        return np.array([x_min, x_max], dtype=float)

    return np.logspace(np.log10(x_min), np.log10(x_max), n_bins + 1)


def equal_number_edges(x: Any, weights: Any, n_bins: int) -> np.ndarray:
    """Returns bin edges with equal integrated weight per bin.

    This constructs edges along ``x`` such that each bin encloses the same
    integrated weight over ``x``. When ``weights`` represent galaxy counts or
    number density, this yields equal-number (equipopulated) bins.

    Args:
        x: 1D axis values (must be strictly increasing).
        weights: 1D weights defined on ``x`` (e.g. counts, number density).
        n_bins: Number of bins.

    Returns:
        1D array of bin edges with shape ``(n_bins + 1,)``.

    Raises:
        ValueError: If inputs are not 1D, have mismatched shapes, contain
            non-finite values, are not strictly increasing in ``x``, or if the
            total integrated weight is non-positive.
    """
    validate_n_bins(n_bins)
    return _equal_weight_edges(x, weights, n_bins)


def equal_information_edges(x: Any, info_density: Any, n_bins: int) -> np.ndarray:
    """Return bin edges with equal integrated information per bin.

    This is the same construction as :func:`equal_number_edges`, but the
    supplied weights represent an information density rather than counts.
    Each bin contains the same integral of ``info_density`` over ``x``.

    Args:
        x: 1D axis values (must be strictly increasing).
        info_density: 1D information density defined on ``x``.
        n_bins: Number of bins.

    Returns:
        1D array of bin edges with shape ``(n_bins + 1,)``.

    Raises:
        ValueError: If inputs are invalid or the total integrated information
            is non-positive.
    """
    validate_n_bins(n_bins)
    return _equal_weight_edges(x, info_density, n_bins)


def equidistant_chi_edges(
    z: Any,
    chi: Any,
    n_bins: int,
) -> np.ndarray:
    """Returns bin edges uniform in comoving distance, expressed in ``z``.

    Requires ``chi(z)`` is monotonic and ``z`` and ``chi`` are matched 1D
    arrays. Useful for cosmological applications where uniform spacing in
    comoving distance is desired.

    Args:
        z: 1D array of redshift values.
        chi: 1D array of comoving distance values corresponding to ``z``.
        n_bins: Number of bins.

    Returns:
        Array of bin edges in redshift of shape (n_bins + 1,), uniform in
        comoving distance.

    Raises:
        ValueError: If ``chi`` is not strictly increasing.
    """
    validate_n_bins(n_bins)

    z_arr, chi_arr = validate_axis_and_weights(z, chi)

    if not np.all(np.diff(chi_arr) > 0):
        raise ValueError("chi must be strictly increasing for interpolation.")

    chi_edges = np.linspace(chi_arr[0], chi_arr[-1], n_bins + 1)
    z_edges = np.interp(chi_edges, chi_arr, z_arr)
    return z_edges


def geometric_edges(x_min: float, x_max: float, n_bins: int) -> np.ndarray:
    """Returns geometrically spaced bin edges between ``x_min`` and ``x_max``.

    Geometric spacing means the ratio between successive edges is constant,
    i.e. ``edges[k+1] / edges[k]`` is the same for all ``k``.
    The returned edges include both endpoints.

    Notes:
        Geometric spacing is the multiplicative analogue of linear spacing.
        It is effectively the same as “log-spaced edges” (equal spacing in
        ``log(x)``), just expressed in ratio form (via ``np.geomspace``) rather
        than via a chosen log base (via ``np.logspace``).

    Args:
        x_min: Lower endpoint of the axis (must be > 0).
        x_max: Upper endpoint of the axis (must be > 0 and > x_min).
        n_bins: Number of bins.

    Returns:
        Array of bin edges with shape ``(n_bins + 1,)``.
    """
    validate_interval(x_min, x_max, n_bins, log=True)
    return np.geomspace(x_min, x_max, n_bins + 1, dtype=float)


def _cumulative_trapz(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Computes a cumulative trapezoidal integral of ``weights`` over ``x``.

    Args:
        x: 1D array of axis values.
        weights: 1D array of weights corresponding to ``x``.

    Returns:
        1D array of cumulative integral values at each point in ``x``.
    """
    cumul = np.empty_like(weights, dtype=float)
    cumul[0] = 0.0

    mid = 0.5 * (weights[1:] + weights[:-1]) * np.diff(x)
    cumul[1:] = np.cumsum(mid)
    return cumul


def _equal_weight_edges(x: Any, weights: Any, n_bins: int) -> np.ndarray:
    """Return bin edges with equal integrated weight per bin.

    This constructs edges along ``x`` so that each bin contains the same integral
    of ``weights`` over ``x`` (using a cumulative trapezoidal integral). This is
    the core helper behind “equal-number” binning when ``weights`` represent
    counts or a number density, and behind “equal-information” binning when
    ``weights`` represent an information density.

    Args:
        x: 1D axis values (must be strictly increasing).
        weights: 1D non-negative weights defined on ``x``.
        n_bins: Number of bins.

    Returns:
        1D array of bin edges with shape ``(n_bins + 1,)``.

    Raises:
        ValueError: If ``weights`` contains negative values, if the total
            integrated weight is not positive or not finite, or if the
            computed edges are not strictly increasing (e.g. if weights are
            too concentrated relative to the resolution of ``x`)
    """
    x_arr, w_arr = validate_axis_and_weights(x, weights)

    # 1) Equal-weight binning assumes non-negative weights
    # (counts/info density)
    if np.any(w_arr < 0):
        raise ValueError("weights must be non-negative for equal-weight binning.")

    # 2) Must contain *some* positive mass.
    if not np.any(w_arr > 0):
        raise ValueError("weights must contain at least one positive value.")

    cumul_trapz = _cumulative_trapz(x_arr, w_arr)
    total = cumul_trapz[-1]

    # 3) Guard against overflow / nonsense totals.
    if not np.isfinite(total):
        raise ValueError("Total integrated weight is not finite.")

    if total <= 0:
        raise ValueError("Total weight must be positive for equal-weight binning.")

    edges = [x_arr[0]]
    for i in range(1, n_bins):
        target = (i / n_bins) * total
        edge = np.interp(target, cumul_trapz, x_arr)
        edges.append(edge)
    edges.append(x_arr[-1])

    edges_arr = np.array(edges, dtype=float)

    # 4) Interp can return repeated edges
    # if weights are too spiky / grid too coarse.
    if np.any(np.diff(edges_arr) <= 0):
        raise ValueError(
            "Cannot construct strictly increasing bin edges "
            "for equal-weight binning. Insufficient x resolution or "
            "overly concentrated weights can cause repeated edges."
        )

    return edges_arr
