"""Computes bin edges using different binning strategies per segment."""

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

CastFunc = Callable[[Any], Any]
EdgeFunc = Callable[..., np.ndarray]


def _get(seg_i: int, params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Resolve a parameter value from segment params or global fallback.

    Args:
        seg_i: Segment index (used for error messages).
        params: Segment-level parameter mapping.
        key: Parameter name to resolve.
        fallback: Value from global arguments.

    Returns:
        The resolved value.

    Raises:
        ValueError: If neither the segment params nor the global fallback
            provides a value for ``key``.
    """
    val = params.get(key, fallback)
    if val is None:
        raise ValueError(f"Segment {seg_i} requires {key!r} in params or as a global argument.")
    return val


_MIXED_SPEC: dict[str, dict[str, Any]] = {
    "equidistant": {
        "required": ("x_min", "x_max"),
        "casts": {"x_min": float, "x_max": float},
    },
    "log": {
        "required": ("x_min", "x_max"),
        "casts": {"x_min": float, "x_max": float},
    },
    "geometric": {
        "required": ("x_min", "x_max"),
        "casts": {"x_min": float, "x_max": float},
    },
    "equal_number": {"required": ("x", "weights")},
    "equal_information": {"required": ("x", "info_density")},
    "equidistant_chi": {"required": ("z", "chi")},
}

_FUNCS: dict[str, EdgeFunc] = {
    "equidistant": equidistant_edges,
    "log": log_edges,
    "geometric": geometric_edges,
    "equal_number": equal_number_edges,
    "equal_information": equal_information_edges,
    "equidistant_chi": equidistant_chi_edges,
}


def _call_with(
    seg_i: int,
    params: Mapping[str, Any],
    n_bins: int,
    g: Mapping[str, Any],
    *,
    func: EdgeFunc,
    required: tuple[str, ...],
    casts: Mapping[str, CastFunc] | None = None,
) -> np.ndarray:
    """Call a bin-edge function using segment params with global fallbacks.

    Args:
        seg_i: Segment index (used for error messages).
        params: Segment-level parameter mapping.
        n_bins: Number of bins for the segment.
        g: Global arguments mapping.
        func: Bin-edge function to call.
        required: Required parameter names to pass to ``func``.
        casts: Optional mapping from parameter name to a cast/convert function.

    Returns:
        1D array of bin edges produced by ``func``.
    """
    casts = casts or {}
    kwargs: dict[str, Any] = {}
    for k in required:
        v = _get(seg_i, params, k, g.get(k))
        if k in casts:
            v = casts[k](v)
        kwargs[k] = v
    return func(**kwargs, n_bins=n_bins)


def _validate_segment_edges(
    seg_i: int,
    edges: np.ndarray,
    *,
    n_bins: int,
    prev_right: float | None,
    atol: float = 1e-12,
) -> float:
    """Validate segment edges and return the segment's right endpoint."""
    edges = np.asarray(edges, dtype=float)

    if edges.ndim != 1:
        raise ValueError(f"Segment {seg_i}: edges must be 1D, got shape {edges.shape}.")

    if edges.size != n_bins + 1:
        raise ValueError(f"Segment {seg_i}: expected {n_bins + 1} edges, got {edges.size}.")

    if not np.all(np.isfinite(edges)):
        raise ValueError(f"Segment {seg_i}: edges must be finite.")

    if not np.all(np.diff(edges) > 0):
        raise ValueError(f"Segment {seg_i}: edges must be strictly increasing.")

    mismatch = prev_right is not None and not np.isclose(edges[0], prev_right, rtol=0, atol=atol)
    if mismatch:
        raise ValueError(
            f"Segment {seg_i}: left edge {edges[0]} does not match previous "
            f"right edge {prev_right}."
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
    """Compute bin edges for a mixed binning strategy across multiple segments.

    Each segment specifies a binning method and a number of bins. Segment edge
    arrays are concatenated in order; shared boundaries are de-duplicated (the
    first edge of each subsequent segment is dropped) so the output is a single
    increasing edge array.

    Segment specification:

        Each element of ``segments`` is a mapping with keys:

        - ``"method"``: Binning method name or alias (resolved via
          :func:`binny.utils.validators.resolve_binning_method`).
        - ``"n_bins"``: Number of bins in this segment (integer).
        - ``"params"``: Optional mapping of method-specific parameters. Any missing
          required parameter may be provided as a global keyword argument to
          :func:`mixed_edges`.

    Global inputs:

        Some methods require arrays (e.g. ``x``/``weights``). You can provide them
        globally, or per-segment via ``params``. If a required input is missing,
        a ``ValueError`` is raised.

    Args:
        segments: Sequence of segment specifications.
        x: 1D axis values (used by equal-number / equal-information methods).
        weights: 1D weights on ``x`` (used by ``"equal_number"``).
        info_density: 1D information density on ``x``
            (used by ``"equal_information"``).
        z: 1D redshift grid (used by ``"equidistant_chi"``).
        chi: 1D comoving distance grid corresponding to ``z``
            (used by ``"equidistant_chi"``).
        total_n_bins: Optional total number of bins for validation
            (sum of segment ``n_bins``).

    Returns:
        1D array of combined bin edges with shape ``(sum(n_bins) + 1,)``.

    Raises:
        ValueError: If segment specs are invalid, a required input is missing,
            a method is unknown, or segment edges are invalid/incompatible.
    """
    validate_mixed_segments(segments, total_n_bins=total_n_bins)

    g = {
        "x": x,
        "weights": weights,
        "info_density": info_density,
        "z": z,
        "chi": chi,
    }

    all_edges: list[np.ndarray] = []
    prev_right: float | None = None

    for i, seg in enumerate(segments):
        method = resolve_binning_method(seg["method"])
        n_bins = int(seg["n_bins"])
        params: Mapping[str, Any] = seg.get("params", {}) or {}

        spec = _MIXED_SPEC.get(method)
        func = _FUNCS.get(method)
        if spec is None or func is None:
            raise ValueError(f"Unknown binning method {method!r} in mixed_edges.")

        edges = _call_with(
            i,
            params,
            n_bins,
            g,
            func=func,
            required=spec["required"],
            casts=spec.get("casts"),
        )

        prev_right = _validate_segment_edges(i, edges, n_bins=n_bins, prev_right=prev_right)

        edges = np.asarray(edges, dtype=float)
        all_edges.append(edges if i == 0 else edges[1:])

    out = np.concatenate(all_edges, axis=0)
    # Final sanity check: strictly increasing combined edges
    if out.ndim != 1 or not np.all(np.diff(out) > 0):
        raise ValueError("Combined mixed edges are not strictly increasing.")
    return out
