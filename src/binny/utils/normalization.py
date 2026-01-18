"""Normalization utilities for 1D data arrays."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson

from binny.utils.validators import validate_axis_and_weights

FloatArray = NDArray[np.float64]

__all__ = [
    "normalize_1d",
    "integrate_bins",
    "cdf_from_curve",
    "weighted_quantile_from_cdf",
    "normalize_or_check_curves",
    "as_float_array",
    "as_bins_dict",
    "require_bins",
    "curve_norm_mode",
    "trapz_weights",
    "normalize_over_z",
    "normalize_edges",
    "prepare_metric_inputs",
]


def normalize_1d(
    x: FloatArray,
    y: FloatArray,
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
    z: FloatArray,
    bins: Mapping[int, FloatArray],
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
    z: FloatArray,
    nz: FloatArray,
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


def as_float_array(x: Any, *, name: str) -> NDArray[np.float64]:
    """Coerce an array-like input to a float64 NumPy array.

    This helper standardizes user inputs to a consistent dtype for numerical
    routines. It is used to keep user-facing APIs forgiving while ensuring that
    downstream computations receive a predictable array type.

    Args:
        x: Array-like input.
        name: Name of the input (used in error messages).

    Returns:
        A 1D or nD NumPy array with dtype float64.

    Raises:
        ValueError: If the input cannot be converted to a float array.
    """
    try:
        return np.asarray(x, dtype=np.float64)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Could not convert {name} to a float array.") from e


def as_bins_dict(bins: Mapping[int, Any]) -> dict[int, NDArray[np.float64]]:
    """Coerce a bins mapping to ``dict[int, float64 array]``.

    This helper normalizes bin curve mappings provided by users (or returned by
    builders) into a consistent representation used across diagnostics. It keeps
    the public API flexible while ensuring diagnostics can assume a stable type.

    Args:
        bins: Mapping of bin identifiers to bin curves.

    Returns:
        A dictionary mapping integer bin indices to float64 arrays.
    """
    return {int(k): as_float_array(v, name=f"bins[{k!r}]") for k, v in bins.items()}


def require_bins(
    bins: Mapping[int, Any] | None,
    *,
    cached: Mapping[int, Any] | None = None,
    name: str = "bins",
) -> dict[int, NDArray[np.float64]]:
    """Resolves bins from an explicit argument or cached bins.

    This helper supports wrapper-style APIs where diagnostics accept an optional
    ``bins`` argument but may also use bins cached on an instance.

    Args:
        bins: Optional bins mapping provided by the caller.
        cached: Optional cached bins mapping (for wrapper classes).
        name: Name used in error messages.

    Returns:
        A bins dictionary with integer keys and float64 arrays.

    Raises:
        ValueError: If neither ``bins`` nor ``cached`` are provided.
    """
    b = cached if bins is None else bins
    if b is None:
        raise ValueError(f"{name} is not set. Build bins first or pass {name}=...")
    return as_bins_dict(b)


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


def normalize_over_z(z: FloatArray, nz: FloatArray) -> FloatArray:
    """Normalizes ``nz`` so that it integrates to 1 over ``z``."""
    z_arr = np.asarray(z, dtype=np.float64)
    nz_arr = np.asarray(nz, dtype=np.float64)

    if z_arr.ndim != 1 or nz_arr.ndim != 1:
        raise ValueError("normalize_over_z requires 1D arrays for z and nz.")
    if z_arr.size < 2:
        raise ValueError("normalize_over_z requires at least two z points.")
    if z_arr.shape != nz_arr.shape:
        raise ValueError("normalize_over_z requires z and nz to have the same shape.")
    if not np.all(np.isfinite(z_arr)) or not np.all(np.isfinite(nz_arr)):
        raise ValueError("normalize_over_z requires finite z and nz.")
    if not np.all(np.diff(z_arr) > 0):
        raise ValueError("normalize_over_z requires z to be strictly increasing.")

    area = float(np.trapezoid(nz_arr, x=z_arr))
    if not np.isfinite(area) or area <= 0.0:
        raise ValueError("Cannot normalize: non-positive or non-finite integral.")

    return nz_arr / area


def normalize_edges(
    bin_indices: Sequence[int],
    bin_edges: Mapping[int, tuple[float, float]] | Sequence[float] | np.ndarray,
) -> dict[int, tuple[float, float]]:
    """Normalizes bin-edge specifications to a mapping of (lo, hi) per bin index.

    Args:
        bin_indices: Sorted bin indices present in the bins mapping.
        bin_edges: Either a mapping {i: (lo, hi)} or an edge array where bin i uses
            (edges[i], edges[i+1]).

    Returns:
        Mapping {i: (lo, hi)} for all i in bin_indices.

    Raises:
        ValueError: If required edges are missing or invalid.
    """
    idx = [int(i) for i in bin_indices]

    # Mapping case: {i: (lo, hi)}
    if isinstance(bin_edges, Mapping):
        out: dict[int, tuple[float, float]] = {}
        for i in idx:
            if i not in bin_edges:
                raise ValueError(f"missing bin index {i} in bin_edges.")
            lo, hi = bin_edges[i]
            lo_f = float(lo)
            hi_f = float(hi)
            out[i] = (lo_f, hi_f)
        return out

    # Sequence case: edges array-like
    edges = np.asarray(bin_edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bin_edges must be a 1D sequence with at least two edges.")
    if not np.all(np.isfinite(edges)):
        raise ValueError("bin_edges must contain only finite values.")

    max_i = max(idx) if idx else -1
    if edges.size < (max_i + 2):
        raise ValueError(
            f"bin_edges must have at least {max_i + 2} entries for bins up to {max_i}."
        )

    out = {}
    for i in idx:
        out[i] = (float(edges[i]), float(edges[i + 1]))
    return out


def prepare_metric_inputs(
    z: Any,
    bins: Mapping[int, Any],
    *,
    mode: Literal["curves", "segments_prob"],
    curve_norm: Literal["none", "normalize", "check"] = "none",
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Validates bin curves and prepares curve- or segment-mass inputs for metrics.

    Args:
        z: Shared 1D grid of nodes.
        bins: Mapping from bin index to curve values evaluated on ``z``.
        mode: ``"curves"`` to return node values; ``"segments_prob"`` to return
            per-segment probability masses (length ``len(z)-1``).
        curve_norm: Normalization handling:
            - ``"none"``: no normalization/checking beyond basic validation.
            - ``"normalize"``: divide each curve by its trapezoid integral.
            - ``"check"``: require each curve to have unit integral (within tol).
        rtol: Relative tolerance for unit-integral checks.
        atol: Absolute tolerance for unit-integral checks.

    Returns:
        (z_arr, out) where out is:
            - curves dict {i: y(z)} if mode="curves"
            - segment probs dict {i: p_k} if mode="segments_prob"

    Raises:
        ValueError: If inputs are invalid, or normalization checks fail.
    """
    z_arr = np.asarray(z, dtype=float)
    if z_arr.ndim != 1 or z_arr.size < 2:
        raise ValueError("z must be a 1D array with at least two points.")
    if not np.all(np.isfinite(z_arr)):
        raise ValueError("z must contain only finite values.")
    if not np.all(np.diff(z_arr) > 0):
        raise ValueError("z must be strictly increasing.")

    if len(bins) == 0:
        return z_arr, {}

    if mode not in {"curves", "segments_prob"}:
        raise ValueError("mode must be 'curves' or 'segments_prob'.")

    curves: dict[int, np.ndarray] = {}
    for k, v in bins.items():
        i = int(k)
        try:
            _, y = validate_axis_and_weights(z_arr, v)
        except ValueError as e:
            raise ValueError(f"Invalid bin {i}: {e}") from e

        area = float(np.trapezoid(y, x=z_arr))

        needs_positive_mass = False
        if curve_norm != "none":
            needs_positive_mass = True
        elif mode == "segments_prob":
            needs_positive_mass = True

        if needs_positive_mass and (area <= 0.0 or not np.isfinite(area)):
            raise ValueError(f"bin {i} has non-positive or non-finite integral: {area}.")

        if curve_norm == "check":
            is_unit = bool(np.isclose(area, 1.0, rtol=rtol, atol=atol))
            if not is_unit:
                raise ValueError(
                    f"bin {i} does not appear normalized (integral={area}). "
                    "Set curve_norm='normalize' or pass already-normalized curves."
                )

        if curve_norm == "normalize":
            y = (y / area).astype(np.float64, copy=False)
        else:
            y = y.astype(np.float64, copy=False)

        curves[i] = y

    if mode == "curves":
        return z_arr, curves

    dz = np.diff(z_arr)
    masses: dict[int, np.ndarray] = {}
    for i, y in curves.items():
        seg = 0.5 * (y[:-1] + y[1:]) * dz
        total = float(np.sum(seg))
        if total <= 0.0 or not np.isfinite(total):
            raise ValueError(f"bin {i} has non-positive segment mass total: {total}.")
        masses[i] = (seg / total).astype(np.float64, copy=False)

    return z_arr, masses
