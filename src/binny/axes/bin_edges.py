"""Module for computing bin edges based on different strategies."""

from __future__ import annotations

from typing import Any

import numpy as np

from binny.core.validators import (
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
    "geometric_edges_n",
]


def equidistant_edges(x_min: float, x_max: float, n_bins: int) -> np.ndarray:
    """Returns uniformly spaced bin edges between ``x_min`` and ``x_max`` (inclusive).

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
    """Returns log-spaced bin edges between ``x_min`` and ``x_max`` (inclusive).

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


def equal_number_edges(x: Any, weights: Any, n_bins: int) -> np.ndarray:
    """Returns bin edges such that each bin contains the same integrated number.

    Works for any 1D axis: redshift, ell, k, etc.
    This is useful for binning data with varying density.

    Args:
        x: 1D array of axis values.
        weights: 1D array of weights (e.g., counts) corresponding to ``x``.
        n_bins: Number of bins.

    Returns:
        Array of bin edges of shape ``(n_bins + 1,)``.
    """
    validate_n_bins(n_bins)
    return _equal_weight_edges(x, weights, n_bins)


def equal_information_edges(
    x: Any,
    info_density: Any,
    n_bins: int,
) -> np.ndarray:
    """Returns bin edges such that each bin contains the same integrated information.

    Information is defined as the integral of the supplied information density
    along the axis. The method applies to any 1D axis (redshift, ell, k, etc.) and
    is useful for binning data with non-uniform information content. It is
    conceptually identical to ``equal_number_edges``, except the weights encode
    information density rather than object counts.

    Args:
        x: 1D array of axis values.
        info_density: 1D array of information density corresponding to ``x``.
        n_bins: Number of bins.

    Returns:
        Array of bin edges of shape (n_bins + 1,).
    """
    validate_n_bins(n_bins)
    return _equal_weight_edges(x, info_density, n_bins)


def equidistant_chi_edges(
    z: Any,
    chi: Any,
    n_bins: int,
) -> np.ndarray:
    """Returns bin edges uniform in comoving distance, expressed in ``z``.

    Assumes ``chi(z)`` is monotonic and ``z`` and ``chi`` are matched 1D arrays.
    Useful for cosmological applications where uniform spacing in comoving distance
    is desired.

    Args:
        z: 1D array of redshift values.
        chi: 1D array of comoving distance values corresponding to ``z``.
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
    """Returns geometrically (log-spaced) bin edges between ``x_min`` and ``x_max``.

    Works for any 1D axis: redshift, ell, k, etc. Useful for logarithmic scales.

    Args:
        x_min: Minimum value of the axis (must be > 0).
        x_max: Maximum value of the axis (must be > 0 and > x_min).
        n_bins: Number of bins.

    Returns:
        Array of length ``n_bins + 1`` with geometrically spaced bin edges.
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
    """Returns bin edges such that each bin contains the same integrated weight.

    Args:
        x: 1D array of axis values.
        weights: 1D array of weights corresponding to ``x``.
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
