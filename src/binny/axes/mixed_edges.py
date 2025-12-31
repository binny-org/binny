"""Module to compute mixed bin edges based on different strategies for each segment."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from binny.axes.bin_edges import (
    equal_information_edges,
    equal_number_edges,
    equidistant_chi_edges,
    equidistant_edges,
    geometric_edges_n,
    log_edges,
)
from binny.core.validators import resolve_binning_method, validate_mixed_segments


def _get(seg_i: int, params: Mapping[str, Any], key: str, fallback: Any) -> Any:
    """Gets a parameter value from params or fallback, raising an error if not found.

    Args:
        seg_i: Index of the current segment.
        params: Parameters dictionary for the segment.
        key: Key to look for in params.
        fallback: Fallback value if key is not in params.

    Returns:
        The value associated with key in params or the fallback value.
    """
    val = params.get(key, fallback)

    if val is None:
        raise ValueError(
            f"Segment {seg_i} requires {key!r} in params or as a global argument."
        )

    return val

_MIXED_SPEC: dict[str, dict[str, Any]] = {
    "equidistant": {"required": ("x_min", "x_max"),
                    "casts": {"x_min": float, "x_max": float}},
    "log": {"required": ("x_min", "x_max"),
            "casts": {"x_min": float, "x_max": float}},
    "geometric": {"required": ("x_min", "x_max"),
                  "casts": {"x_min": float, "x_max": float}},
    "equal_number": {"required": ("x", "weights")},
    "equal_information": {"required": ("x", "info_density")},
    "equidistant_chi": {"required": ("z", "chi")},
}

_FUNCS: dict[str, Callable[..., np.ndarray]] = {
    "equidistant": equidistant_edges,
    "log": log_edges,
    "geometric": geometric_edges_n,
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
    func: Callable[..., np.ndarray],
    required: tuple[str, ...],
    casts: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Calls a bin edge function with parameters from params or global arguments.

    Args:
        seg_i: Index of the current segment.
        params: Parameters dictionary for the segment.
        n_bins: Number of bins for the segment.
        g: Global arguments mapping.
        func: Bin edge function to call.
        required: Tuple of required parameter names.
        casts: Optional mapping of parameter names to casting functions.

    Returns:
        Array of bin edges.
    """
    casts = casts or {}
    kwargs: dict[str, Any] = {}
    for k in required:
        v = _get(seg_i, params, k, g.get(k))
        if k in casts:
            v = casts[k](v)
        kwargs[k] = v
    return func(**kwargs, n_bins=n_bins)


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
    """Computes bin edges for a mixed binning strategy across multiple segments.

    Each segment can use a different binning method, specified in
    the ``segments` argument.

    Args:
        segments: Sequence of segment specifications. Each segment is
            a mapping with keys:
            - "method": Binning method name
                (e.g., ``"equidistant"``, ``"log"``, ``"equal_number"``, etc.).
            - "n_bins": Number of bins for the segment.
            - "params": Optional mapping of parameters specific to the segment.
        x: 1D array of axis values (required for some methods).
        weights: 1D array of weights corresponding to ``x``
            (required for "equal_number").
        info_density: 1D array of information density corresponding to ``x``
            (required for ``"equal_information"``).
        z: 1D array of redshift values (required for ``"equidistant_chi"``).
        chi: 1D array of comoving distances corresponding to ``z``
            (required for "equidistant_chi").
        total_n_bins: Optional total number of bins across all segments for validation.

    Returns:
        Array of bin edges combining all segments.

    Raises:
        RuntimeError: If an unhandled binning method is specified.
    """
    validate_mixed_segments(segments, total_n_bins=total_n_bins)

    g = {"x": x, "weights": weights, "info_density": info_density, "z": z, "chi": chi}

    all_edges: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        method = resolve_binning_method(seg["method"])
        n_bins = int(seg["n_bins"])
        params: Mapping[str, Any] = seg.get("params", {}) or {}

        spec = _MIXED_SPEC.get(method)
        func = _FUNCS.get(method)
        if spec is None or func is None:
            raise RuntimeError(f"Unhandled binning method {method!r} in mixed_edges.")

        edges = _call_with(
            i,
            params,
            n_bins,
            g,
            func=func,
            required=spec["required"],
            casts=spec.get("casts"),
        )

        all_edges.append(edges if i == 0 else edges[1:])

    return np.concatenate(all_edges, axis=0)
