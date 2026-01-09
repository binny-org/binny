"""Normalization utilities for 1D data arrays."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import simpson

from binny.utils.validators import validate_axis_and_weights

FloatArray = np.ndarray
PrepMode: TypeAlias = Literal["curves", "segments_prob"]
NormMode: TypeAlias = Literal["none", "normalize", "check"]

__all__ = [
    "normalize_1d",
    "integrate_bins",
    "cdf_from_curve",
    "weighted_quantile_from_cdf",
    "mass_per_segment",
    "normalize_or_check_curves",
    "trapz_weights",
    "normalize_edges",
    "prepare_metric_inputs",
    "curve_norm_mode",
]


FloatArray = NDArray[np.float64]


def normalize_1d(
    x: ArrayLike,
    y: ArrayLike,
    *,
    method: Literal["trapezoid", "simpson"] = "trapezoid",
) -> FloatArray:
    """Returns ``y`` scaled so that its integral over ``x`` is 1.

    This is commonly used to normalize sampled 1D curves (e.g., probability
    densities or redshift distributions) defined on a strictly increasing grid,
    so they can be compared consistently or interpreted as unit-mass functions.

    Args:
        x: One-dimensional grid of sample locations.
        y: Values evaluated on ``x``.
        method: Numerical integration rule used to compute the normalization.

    Returns:
        The normalized values as a ``float64`` NumPy array.

    Raises:
        ValueError: If ``x`` or ``y`` are not 1D, have mismatched shapes, contain
            non-finite values, have fewer than two points, or if ``x`` is not
            strictly increasing.
        ValueError: If ``method`` is not one of ``"trapezoid"`` or ``"simpson"``.
        ValueError: If the computed normalization factor is non-positive.
    """
    x_arr, y_arr = validate_axis_and_weights(x, y)

    if method == "simpson":
        norm = float(simpson(y_arr, x=x_arr))
    elif method == "trapezoid":
        norm = float(np.trapezoid(y_arr, x=x_arr))
    else:
        raise ValueError("method must be 'trapezoid' or 'simpson'.")

    if norm <= 0.0:
        raise ValueError("Normalization factor must be positive.")

    return (y_arr / norm).astype(np.float64, copy=False)


def integrate_bins(
    z: ArrayLike,
    bins: Mapping[int, ArrayLike],
) -> dict[int, float]:
    """Computes trapezoid integrals for multiple curves evaluated on a shared grid.

    This is useful for quickly checking per-bin masses of a collection of sampled
    distributions (e.g., tomographic ``n_i(z)`` curves) defined on the same
    strictly increasing axis.

    Args:
        z: One-dimensional grid shared by all curves.
        bins: Mapping from bin index to curve values evaluated on ``z``.

    Returns:
        A mapping ``{bin_idx: integral}`` of trapezoid areas.

    Raises:
        ValueError: If ``bins`` is empty.
        ValueError: If ``z`` or any curve is not 1D, has mismatched length with ``z``,
            contains non-finite values, has fewer than two points, or if ``z`` is not
            strictly increasing. The error message is annotated with the offending
            bin index.
    """
    if len(bins) == 0:
        raise ValueError("bins must not be empty.")

    z_arr = np.asarray(z, dtype=float)

    integrals: dict[int, float] = {}
    for idx, nz_bin in bins.items():
        try:
            _, nz_arr = validate_axis_and_weights(z_arr, nz_bin)
        except ValueError as e:
            raise ValueError(f"Invalid bin {idx}: {e}") from e

        integrals[int(idx)] = float(np.trapezoid(nz_arr, x=z_arr))

    return integrals


def cdf_from_curve(
    z: ArrayLike,
    nz: ArrayLike,
) -> tuple[FloatArray, float]:
    """Builds a trapezoid cumulative mass function from a nonnegative curve.

    The result is returned at the grid nodes, starting at zero, and accumulating
    trapezoid segment masses. This representation is convenient for computing
    weighted quantiles on a discrete grid while keeping the total mass explicit.

    Args:
        z: One-dimensional grid of nodes.
        nz: Nonnegative curve values evaluated on ``z``.

    Returns:
        A tuple ``(cdf, norm)`` where ``cdf`` is the cumulative trapezoid mass at
        each node (dtype ``float64``) and ``norm`` is the total mass.

    Raises:
        ValueError: If ``z`` or ``nz`` are not 1D, have mismatched shapes, contain
            non-finite values, have fewer than two points, or if ``z`` is not strictly
            increasing.
        ValueError: If any values of ``nz`` are negative.
        ValueError: If the total mass (trapezoid integral) is non-positive.
    """
    z_arr, nz_arr = validate_axis_and_weights(z, nz)

    if np.any(nz_arr < 0):
        raise ValueError("nz must be nonnegative to build a CDF.")

    norm = float(np.trapezoid(nz_arr, x=z_arr))
    if norm <= 0.0:
        raise ValueError("Total weight must be positive.")

    dz = np.diff(z_arr)
    seg = 0.5 * (nz_arr[:-1] + nz_arr[1:]) * dz
    cdf = np.concatenate(([0.0], np.cumsum(seg))).astype(np.float64, copy=False)
    return cdf, norm


def weighted_quantile_from_cdf(
    z_arr: np.ndarray,
    cdf: np.ndarray,
    norm: float,
    q: float,
    *,
    side: Literal["left", "right"] = "left",
) -> float:
    """Returns a weighted quantile from a precomputed cumulative mass array.

    This finds the location where the cumulative mass reaches ``q * norm`` and
    linearly interpolates between adjacent grid nodes. It is intended to be used
    with cumulative masses produced by trapezoid integration on the same node grid.

    Args:
        z_arr: 1D array of strictly increasing grid nodes.
        cdf: 1D array of cumulative masses at the nodes (nondecreasing).
        norm: Total mass associated with the CDF.
        q: Quantile in the interval ``[0, 1]``.
        side: Side argument forwarded to ``np.searchsorted`` for locating the target.

    Returns:
        The weighted quantile value on the ``z_arr`` grid.

    Raises:
        ValueError: If ``q`` is outside ``[0, 1]``.
        ValueError: If ``norm`` is not positive.
        ValueError: If ``z_arr`` and ``cdf`` are not 1D arrays of the same nonzero
            length.
        ValueError: If ``z_arr`` is not strictly increasing.
        ValueError: If ``cdf`` is not nondecreasing.
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be between 0 and 1.")
    if norm <= 0.0:
        raise ValueError("norm must be positive.")

    z_arr = np.asarray(z_arr, dtype=float)
    cdf = np.asarray(cdf, dtype=float)

    if z_arr.ndim != 1 or cdf.ndim != 1 or z_arr.size != cdf.size or z_arr.size == 0:
        raise ValueError("z_arr and cdf must be 1D arrays of the same nonzero length.")
    if not np.all(np.diff(z_arr) > 0):
        raise ValueError("z_arr must be strictly increasing.")
    if np.any(np.diff(cdf) < 0):
        raise ValueError("cdf must be nondecreasing.")

    target = q * float(norm)
    j = int(np.searchsorted(cdf, target, side=side))

    if j <= 0:
        return float(z_arr[0])
    if j >= cdf.size:
        return float(z_arr[-1])

    c0 = float(cdf[j - 1])
    c1 = float(cdf[j])
    if np.isclose(c1, c0):
        return float(z_arr[j])

    t = (target - c0) / (c1 - c0)
    return float(z_arr[j - 1] + t * (z_arr[j] - z_arr[j - 1]))


def trapz_weights(z_arr: np.ndarray) -> FloatArray:
    """Returns trapezoid-rule integration weights for a strictly increasing 1D grid.

    The returned weights satisfy ``np.trapezoid(f, x=z_arr) == np.sum(w * f)`` for
    arrays ``f`` evaluated at the grid nodes, which is useful for vectorized
    integrations and repeated inner products on a fixed axis.

    Args:
        z_arr: 1D grid of nodes.

    Returns:
        A ``float64`` array of node weights with the same shape as ``z_arr``. For
        grids with fewer than two points, the weights are all zeros.

    Raises:
        ValueError: If ``z_arr`` is not 1D.
        ValueError: If ``z_arr`` is not strictly increasing.
    """
    z_arr = np.asarray(z_arr, dtype=float)
    if z_arr.ndim != 1:
        raise ValueError("z_arr must be a 1D array.")

    if z_arr.size < 2:
        return np.zeros_like(z_arr, dtype=np.float64)

    if not np.all(np.diff(z_arr) > 0):
        raise ValueError("z_arr must be strictly increasing.")

    dz = np.diff(z_arr)

    w = np.empty_like(z_arr, dtype=np.float64)
    w[0] = 0.5 * dz[0]
    w[-1] = 0.5 * dz[-1]
    w[1:-1] = 0.5 * (dz[:-1] + dz[1:])
    return w


def mass_per_segment(z_arr: np.ndarray, p_arr: np.ndarray) -> FloatArray:
    """Returns trapezoid masses per grid segment for a curve sampled at nodes.

    This converts node values into per-interval masses using the trapezoid rule,
    which is useful for building cumulative masses, rebinning, or diagnostics that
    operate on segment contributions rather than node values.

    Args:
        z_arr: 1D array of grid nodes.
        p_arr: 1D array of curve values at the nodes.

    Returns:
        A ``float64`` array of length ``len(z_arr) - 1`` containing trapezoid masses
        for each adjacent node interval.

    Raises:
        ValueError: If inputs are not 1D arrays of the same length.
        ValueError: If ``z_arr`` is not strictly increasing.
    """
    z_arr = np.asarray(z_arr, dtype=float)
    p_arr = np.asarray(p_arr, dtype=float)

    if z_arr.ndim != 1 or p_arr.ndim != 1 or z_arr.size != p_arr.size:
        raise ValueError("z_arr and p_arr must be 1D arrays of the same length.")

    if not np.all(np.diff(z_arr) > 0):
        raise ValueError("z_arr must be strictly increasing.")

    dz = np.diff(z_arr)
    mass = 0.5 * (p_arr[:-1] + p_arr[1:]) * dz
    return mass.astype(np.float64, copy=False)


def normalize_or_check_curves(
    z_arr: np.ndarray,
    p: Mapping[int, np.ndarray],
    *,
    normalize: bool,
    check_normalized: bool,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    warn_if_already_normalized: bool = False,
) -> dict[int, np.ndarray]:
    """Returns curves that are normalized and/or validated for unit integral.

    This is a convenience helper for collections of sampled curves on a shared,
    strictly increasing grid. It can enforce that each curve integrates to one
    (within tolerance), and it can also normalize curves by dividing by their
    trapezoid integral.

    Args:
        z_arr: Shared 1D grid of nodes.
        p: Mapping from bin id to curve values evaluated on ``z_arr``.
        normalize: Whether to divide each curve by its trapezoid integral.
        check_normalized: Whether to require each curve to have unit integral within
            ``rtol``/``atol``.
        rtol: Relative tolerance used for the unit-integral check.
        atol: Absolute tolerance used for the unit-integral check.
        warn_if_already_normalized: Whether to warn (when normalizing) if a curve
            already appears normalized within tolerance.

    Returns:
        A new mapping from bin id to curve arrays (normalized if requested).

    Raises:
        ValueError: If ``z_arr`` or any curve is not 1D, has mismatched length with
            ``z_arr``, contains non-finite values, has fewer than two points, or if
            ``z_arr`` is not strictly increasing.
        ValueError: If any curve has a non-positive trapezoid integral.
        ValueError: If ``check_normalized`` is True and any curve is not within
            tolerance of unit integral.
    """
    z_arr = np.asarray(z_arr, dtype=float)

    out: dict[int, np.ndarray] = {}
    for idx, curve in p.items():
        _, curve_arr = validate_axis_and_weights(z_arr, curve)

        area = float(np.trapezoid(curve_arr, x=z_arr))
        if area <= 0.0:
            raise ValueError(f"bin {idx} has non-positive integral: {area}.")

        is_norm = bool(np.isclose(area, 1.0, rtol=rtol, atol=atol))

        if check_normalized and not is_norm:
            raise ValueError(
                f"bin {idx} does not appear normalized (integral={area}). "
                "Set check_normalized=False or normalize=True."
            )

        if normalize:
            if warn_if_already_normalized and is_norm:
                warnings.warn(
                    f"bin {idx} appears already normalized (integral={area}).",
                    stacklevel=2,
                )
            out[int(idx)] = (curve_arr / area).astype(np.float64, copy=False)
        else:
            out[int(idx)] = curve_arr.astype(np.float64, copy=False)

    return out


def normalize_edges(
    bin_indices: Sequence[int],
    bin_edges: Mapping[int, tuple[float, float]] | Sequence[float] | np.ndarray,
) -> dict[int, tuple[float, float]]:
    """Returns a mapping from bin index to ``(lo, hi)`` edge pairs.

    This normalizes bin-edge inputs into a consistent dictionary form. It supports
    either an explicit per-bin edge mapping or a single strictly increasing edge
    array interpreted in the standard way, where bin ``j`` corresponds to
    ``(edges[j], edges[j+1])``.

    Args:
        bin_indices: Bin indices that must be present in the returned mapping.
        bin_edges: Either a mapping ``{idx: (lo, hi)}`` or a 1D strictly increasing
            edge array ``[e0, e1, ..., eN]``.

    Returns:
        A mapping ``{idx: (lo, hi)}`` with float-valued edge pairs.

    Raises:
        ValueError: If a required bin index is missing from a mapping input.
        ValueError: If an edge array is not 1D, has fewer than two entries, contains
            non-finite values, or is not strictly increasing.
        ValueError: If any requested bin index is out of range for an edge array.
    """
    bin_indices = [int(i) for i in bin_indices]
    edges_map: dict[int, tuple[float, float]] = {}

    if isinstance(bin_edges, Mapping):
        for j in bin_indices:
            try:
                lo, hi = bin_edges[j]
            except KeyError as e:
                raise ValueError(f"bin_edges is missing bin index {e.args[0]}.") from e
            edges_map[j] = (float(lo), float(hi))
        return edges_map

    edges_arr = np.asarray(bin_edges, dtype=float)
    if edges_arr.ndim != 1 or edges_arr.size < 2:
        raise ValueError("bin_edges must be a 1D sequence with length at least 2.")
    if not np.all(np.isfinite(edges_arr)):
        raise ValueError("bin_edges must be finite.")
    if not np.all(np.diff(edges_arr) > 0):
        raise ValueError("bin_edges must be strictly increasing.")

    max_bin = edges_arr.size - 2
    for j in bin_indices:
        if j < 0 or j > max_bin:
            raise ValueError(
                f"bin index {j} is out of range for edges of length {edges_arr.size}."
            )
        edges_map[j] = (float(edges_arr[j]), float(edges_arr[j + 1]))

    return edges_map


def prepare_metric_inputs(
    z_arr: np.ndarray,
    p: Mapping[int, np.ndarray],
    *,
    mode: PrepMode,
    curve_norm: NormMode = "none",
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Prepares inputs for pairwise metrics (validate once; optionally normalize).

    This is a convenience wrapper that standardizes the common boilerplate for
    pairwise curve metrics:

    - Validates ``z_arr`` and each curve in ``p`` using
        :func:`validate_axis_and_weights`.
    - Optionally normalizes curves to unit trapezoid integral or
        checks they already are.
    - Optionally converts curves to per-segment probability vectors (segment masses
      normalized to sum to 1), suitable for discrete probability metrics.

    Args:
        z_arr: 1D strictly increasing grid of nodes.
        p: Mapping from id to curve values evaluated on ``z_arr``.
        mode: Output mode:
            - ``"curves"``: return validated (and possibly normalized) node curves.
            - ``"segments_prob"``: return per-segment mass *probability* vectors.
        curve_norm: How to treat curve normalization before any conversion:
            - ``"none"``: no normalization checks beyond basic validation.
            - ``"normalize"``: divide each curve by its trapezoid integral.
            - ``"check"``: require each curve integrates to 1 within tolerance.
        rtol: Relative tolerance for the unit-integral check when
            ``curve_norm="check"``.
        atol: Absolute tolerance for the unit-integral check when
            ``curve_norm="check"``.

    Returns:
        ``(z_arr, out)`` where ``z_arr`` is float64 and ``out`` maps ids to arrays:
        - For ``mode="curves"``: arrays have length ``len(z_arr)``.
        - For ``mode="segments_prob"``: arrays have length ``len(z_arr) - 1``
            and sum to 1.

    Raises:
        ValueError: If ``z_arr`` or any curve fails validation, if a curve has
            non-positive trapezoid integral (needed for normalize/check),
            if a check fails, or if a curve yields non-positive total
            segment mass in ``"segments_prob"`` mode.
    """
    z_arr = np.asarray(z_arr, dtype=float)

    curves: dict[int, np.ndarray] = {}
    for idx, curve in p.items():
        _, c = validate_axis_and_weights(z_arr, curve)
        area = float(np.trapezoid(c, x=z_arr))

        if curve_norm == "normalize":
            if area <= 0.0:
                raise ValueError(f"bin {idx} has non-positive integral: {area}.")
            c = (c / area).astype(np.float64, copy=False)

        elif curve_norm == "check":
            if area <= 0.0:
                raise ValueError(f"bin {idx} has non-positive integral: {area}.")
            if not np.isclose(area, 1.0, rtol=rtol, atol=atol):
                raise ValueError(
                    f"bin {idx} does not appear normalized (integral={area}). "
                    "Set curve_norm='normalize' or curve_norm='none'."
                )
            c = c.astype(np.float64, copy=False)

        else:  # "none"
            c = c.astype(np.float64, copy=False)

        curves[int(idx)] = c

    if mode == "curves":
        return z_arr.astype(np.float64, copy=False), curves

    if mode == "segments_prob":
        probs: dict[int, np.ndarray] = {}
        for idx, c in curves.items():
            m = mass_per_segment(z_arr, c)
            s = float(np.sum(m))
            if s <= 0.0:
                raise ValueError(f"bin {idx} has non-positive mass on segments.")
            probs[idx] = (m / s).astype(np.float64, copy=False)
        return z_arr.astype(np.float64, copy=False), probs

    raise ValueError('mode must be "curves" or "segments_prob".')


def curve_norm_mode(
    *,
    required: bool,
    assume_normalized: bool,
    normalize_if_needed: bool,
) -> Literal["none", "normalize", "check"]:
    """Resolves how to treat curve normalization for a given metric call.

    Args:
        required: Whether the chosen metric expects normalized curves.
        assume_normalized: User intent: treat curves as normalized.
        normalize_if_needed: If True, renormalize curves when they do not appear
            normalized and normalization is required.

    Returns:
        One of ``"none"``, ``"normalize"``, or ``"check"`` to pass as
        ``curve_norm`` into :func:`binny.utils.normalization.prepare_metric_inputs`.
    """
    if not required:
        return "none"
    if not assume_normalized:
        return "none"
    return "normalize" if normalize_if_needed else "check"
