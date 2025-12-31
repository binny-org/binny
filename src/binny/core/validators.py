"""Validation utilities for binning and axis-related functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from collections.abc import Mapping, Sequence
from typing import Any


__all__ = [
    "validate_interval",
    "validate_axis_and_weights",
    "validate_n_bins",
    "validate_mixed_segments",
    "resolve_binning_method",
]

# Normalised names for binning methods and short aliases
_BIN_METHOD_ALIASES: dict[str, str] = {
    # equidistant (linear in x)
    "equidistant": "equidistant",
    "eq": "equidistant",
    "linear": "equidistant",

    # log-spaced
    "log": "log",
    "log_edges": "log",

    # equal-number / equipopulated
    "equal_number": "equal_number",
    "equipop": "equal_number",
    "en": "equal_number",

    # equal-information
    "equal_information": "equal_information",
    "info": "equal_information",

    # chi-spaced in comoving distance
    "equidistant_chi": "equidistant_chi",
    "chi": "equidistant_chi",

    # geometric in x
    "geometric": "geometric",
    "geom": "geometric",
    "geometric_edges_n": "geometric",
}


def resolve_binning_method(name: str) -> str:
    """Resolve a user-facing binning method name (with aliases) to a canonical key."""
    key = str(name).lower()
    if key not in _BIN_METHOD_ALIASES:
        raise ValueError(
            f"Unknown binning method {name!r}. "
            f"Supported methods: {sorted(set(_BIN_METHOD_ALIASES.values()))}"
        )
    return _BIN_METHOD_ALIASES[key]


def validate_n_bins(
    n_bins: int,
    *,
    allow_one: bool = True,
    max_bins: int = 1_000_000,
) -> None:
    """Validate the number of bins."""
    if not isinstance(n_bins, int):
        raise TypeError("n_bins must be an integer.")

    if n_bins < 0:
        raise ValueError("n_bins must be non-negative.")

    if not allow_one and n_bins == 1:
        raise ValueError("n_bins must be greater than 1.")

    if n_bins == 0:
        raise ValueError("n_bins must be positive.")

    if n_bins > max_bins:
        raise ValueError(
            f"n_bins={n_bins} is too large; may cause memory issues "
            f"(max allowed={max_bins})."
        )


def validate_interval(
    x_min: float,
    x_max: float,
    n_bins: int,
    *,
    log: bool = False,
) -> None:
    """Validate scalar interval [x_min, x_max] + n_bins."""
    validate_n_bins(n_bins)

    if np.isnan(x_min) or np.isnan(x_max):
        raise ValueError("x_min and x_max must be valid finite numbers.")

    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("x_min and x_max must be finite numbers.")

    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min.")

    if log:
        if x_min <= 0 or x_max <= 0:
            raise ValueError("log-/geometric-spaced bins require x_min > 0 and x_max > 0.")


def validate_axis_and_weights(
    x: ArrayLike,
    weights: ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate 1D axis and weights arrays and return them as float ndarrays."""
    x_arr = np.asarray(x, dtype=float)
    w_arr = np.asarray(weights, dtype=float)

    if x_arr.shape != w_arr.shape:
        raise ValueError("x and weights must have the same shape.")

    if x_arr.ndim != 1:
        raise ValueError("x must be 1D.")

    if w_arr.ndim != 1:
        raise ValueError("weights must be 1D.")

    if not np.all(np.isfinite(x_arr)):
        raise ValueError("x must contain only finite values.")

    if not np.all(np.isfinite(w_arr)):
        raise ValueError("weights must contain only finite values.")

    if x_arr.size < 2:
        raise ValueError("x must contain at least two points.")

    if not np.all(np.diff(x_arr) > 0):
        raise ValueError("x must be strictly increasing for binning.")

    return x_arr, w_arr

def validate_mixed_segments(
    segments: Sequence[Mapping[str, Any]],
    *,
    total_n_bins: int | None = None,
) -> None:
    """Validate a list of mixed-binning segments.

    Each segment must have at least:
      - 'method': str   (e.g. 'equidistant', 'eq', 'equal_number', 'chi', ...)
      - 'n_bins': int > 0

    If total_n_bins is given, the sum over all segments must match it.
    """
    if not segments:
        raise ValueError("segments must be a non-empty sequence.")

    n_sum = 0
    for i, seg in enumerate(segments):
        if not isinstance(seg, Mapping):
            raise TypeError(f"Segment {i} must be a mapping, got {type(seg).__name__}.")

        if "method" not in seg or "n_bins" not in seg:
            raise ValueError(
                f"Segment {i} must contain at least 'method' and 'n_bins' keys."
            )

        method = resolve_binning_method(seg["method"])
        n_bins = int(seg["n_bins"])

        if n_bins <= 0:
            raise ValueError(f"Segment {i}: n_bins must be positive, got {n_bins}.")

        # This will raise if the method name is not known
        _ = method
        n_sum += n_bins

    if total_n_bins is not None and n_sum != total_n_bins:
        raise ValueError(
            f"Sum of segment n_bins = {n_sum}, but total_n_bins={total_n_bins}."
        )
