"""Combine multiple binning strategies into a single set of bin edges."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from binny.axes.bin_edges import (
    equal_information_edges,
    equal_number_edges,
    equidistant_chi_edges,
    equidistant_edges,
    geometric_edges,
    log_edges,
)
from binny.utils.validators import (
    resolve_binning_method,
    validate_mixed_segments,
)

__all__ = ["mixed_edges"]

EdgeFunc = Callable[..., np.ndarray]

_METHODS: dict[str, tuple[EdgeFunc, tuple[str, ...]]] = {
    "equidistant": (equidistant_edges, ("x_min", "x_max")),
    "log": (log_edges, ("x_min", "x_max")),
    "geometric": (geometric_edges, ("x_min", "x_max")),
    "equal_number": (equal_number_edges, ("x", "weights")),
    "equal_information": (equal_information_edges, ("x", "info_density")),
    "equidistant_chi": (equidistant_chi_edges, ("z", "chi")),
}


def _get(
    seg_i: int,
    params: Mapping[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    """Resolve a segment parameter, falling back to a global value."""
    value = params.get(key, fallback)

    if value is None:
        raise ValueError(f"Segment {seg_i} requires {key!r} in params or as a global argument.")

    return value


def _call_with(
    seg_i: int,
    params: Mapping[str, Any],
    n_bins: int,
    g: Mapping[str, Any],
    *,
    func: EdgeFunc,
    required: tuple[str, ...],
) -> np.ndarray:
    """Call an existing edge builder with resolved segment parameters."""
    kwargs = {key: _get(seg_i, params, key, g.get(key)) for key in required}

    for key in ("x_min", "x_max"):
        if key in kwargs:
            kwargs[key] = float(kwargs[key])

    return func(**kwargs, n_bins=n_bins)


def _maybe_get_range(
    params: Mapping[str, Any],
) -> tuple[float, float] | None:
    """Return an optional segment range, accepting legacy redshift keys."""
    lo = params.get("x_min", params.get("z_min"))
    hi = params.get("x_max", params.get("z_max"))

    if lo is None or hi is None:
        return None

    lo, hi = float(lo), float(hi)

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid segment range x_min/x_max: ({lo}, {hi}).")

    return lo, hi


def _slice_axis_weights(
    seg_i: int,
    x: np.ndarray,
    w: np.ndarray,
    lo: float,
    hi: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Restrict an axis and weights to a segment, interpolating its bounds.

    Args:
        seg_i: Segment index for error messages.
        x: Strictly increasing sampling axis.
        w: Weights evaluated on the axis.
        lo: Lower segment boundary.
        hi: Upper segment boundary.

    Returns:
        Restricted axis and weights, including both exact boundaries.

    Raises:
        ValueError: If the arrays or requested range are invalid.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    if x.ndim != 1 or w.ndim != 1 or x.size != w.size:
        raise ValueError(f"Segment {seg_i}: x/weights must be 1D and same length.")

    if x.size < 2 or not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0):
        raise ValueError(f"Segment {seg_i}: x must be finite and strictly increasing.")

    if not np.all(np.isfinite(w)):
        raise ValueError(f"Segment {seg_i}: weights must be finite.")

    if not (np.isfinite(lo) and np.isfinite(hi) and x[0] <= lo < hi <= x[-1]):
        raise ValueError(f"Segment {seg_i}: range [{lo}, {hi}] lies outside the axis.")

    interior = (x > lo) & (x < hi)

    xs = np.concatenate(([lo], x[interior], [hi]))
    ws = np.concatenate(
        (
            [np.interp(lo, x, w)],
            w[interior],
            [np.interp(hi, x, w)],
        )
    )

    return xs, ws


def _validate_segment_edges(
    seg_i: int,
    edges: np.ndarray,
    *,
    n_bins: int,
    prev_right: float | None,
    atol: float = 1e-12,
) -> float:
    """Validate a segment's edges and its connection to the previous one."""
    edges = np.asarray(edges, dtype=float)

    if edges.ndim != 1:
        raise ValueError(f"Segment {seg_i}: edges must be 1D, got shape {edges.shape}.")

    if edges.size != n_bins + 1:
        raise ValueError(f"Segment {seg_i}: expected {n_bins + 1} edges, got {edges.size}.")

    if not np.all(np.isfinite(edges)):
        raise ValueError(f"Segment {seg_i}: edges must be finite.")

    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"Segment {seg_i}: edges must be strictly increasing.")

    if prev_right is not None and not np.isclose(edges[0], prev_right, rtol=0, atol=atol):
        raise ValueError(
            f"Segment {seg_i}: left edge {edges[0]} does not match "
            f"previous right edge {prev_right}."
        )

    return float(edges[-1])


def mixed_edges(
    segments: Sequence[Mapping[str, Any]],
    *,
    x: Any | None = None,
    weights: Any | None = None,
    info_density: Any | None = None,
    z: Any | None = None,
    chi: Any | None = None,
    total_n_bins: int | None = None,
) -> np.ndarray:
    """Construct bin edges by combining multiple binning strategies.

    Each segment specifies a ``method``, ``n_bins``, and optional
    ``params`` mapping. Parameters can be supplied per segment or
    through the global keyword arguments.

    Supported methods are ``equidistant``, ``log``, ``geometric``,
    ``equal_number``, ``equal_information``, and ``equidistant_chi``.

    Weighted methods can be restricted to a segment using
    ``x_min``/``x_max`` (or legacy ``z_min``/``z_max``).
    The weights are interpolated at the exact segment boundaries.

    Adjacent segments must share a boundary. Duplicate boundaries
    are removed when combining their edges.

    Args:
        segments: Ordered segment specifications.
        x: Sampling axis for weighted binning methods.
        weights: Weights for equal-number binning.
        info_density: Weights for equal-information binning.
        z: Redshift axis for equal-distance binning.
        chi: Comoving distances corresponding to ``z``.
        total_n_bins: Optional expected total number of bins.

    Returns:
        Strictly increasing array of combined bin edges.

    Raises:
        ValueError: If segment specifications, inputs, or resulting
            edges are invalid.
    """
    validate_mixed_segments(segments, total_n_bins=total_n_bins)

    global_params = {
        "x": x,
        "weights": weights,
        "info_density": info_density,
        "z": z,
        "chi": chi,
    }

    all_edges: list[np.ndarray] = []
    prev_right: float | None = None

    for i, segment in enumerate(segments):
        method = resolve_binning_method(segment["method"])
        n_bins = int(segment["n_bins"])
        params = dict(segment.get("params", {}) or {})

        # Accept ranges specified directly on the segment.
        for key in ("x_min", "x_max", "z_min", "z_max"):
            if key in segment and key not in params:
                params[key] = segment[key]

        if method not in _METHODS:
            raise ValueError(f"Unknown binning method {method!r} in mixed_edges.")

        func, required = _METHODS[method]

        # Restrict weighted methods to their requested segment range.
        if method in {"equal_number", "equal_information"}:
            bounds = _maybe_get_range(params)

            if bounds is not None:
                weight_key = "weights" if method == "equal_number" else "info_density"

                x_in = _get(i, params, "x", global_params["x"])
                w_in = _get(i, params, weight_key, global_params[weight_key])

                params["x"], params[weight_key] = _slice_axis_weights(i, x_in, w_in, *bounds)

        edges = _call_with(
            i,
            params,
            n_bins,
            global_params,
            func=func,
            required=required,
        )

        prev_right = _validate_segment_edges(
            i,
            edges,
            n_bins=n_bins,
            prev_right=prev_right,
        )

        edges = np.asarray(edges, dtype=float)
        all_edges.append(edges if i == 0 else edges[1:])

    out = np.concatenate(all_edges)

    if out.ndim != 1 or not np.all(np.diff(out) > 0):
        raise ValueError("Combined mixed edges are not strictly increasing.")

    return out
