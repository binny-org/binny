"""Validation utilities for binning and axis-related functions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

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
    """Resolves a binning method name or alias to its canonical name.

    Args:
        name: Binning method name or alias.

    Returns:
        Canonical binning method name.

    Raises:
        ValueError: If the method name is unknown.
    """
    key = str(name).strip().lower()

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
    """Validates the number of bins.

    Args:
        n_bins: Number of bins to validate.
        allow_one: Whether to allow n_bins == 1.
        max_bins: Maximum allowed number of bins to prevent memory issues.

    Raises:
        TypeError: If n_bins is not an integer.
        ValueError: If n_bins is not positive, or if n_bins == 1 when
            allow_one is False, or if n_bins exceeds max_bins.
    """
    if not isinstance(n_bins, int):
        raise TypeError("n_bins must be an integer.")

    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")

    if not allow_one and n_bins == 1:
        raise ValueError("n_bins must be greater than 1.")

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
    """Validates the interval ``[x_min, x_max]`` and number of bins.

    Args:
        x_min: Minimum value of the axis.
        x_max: Maximum value of the axis.
        n_bins: Number of bins.
        log: Whether the bins are logarithmically spaced.

    Raises:
        ValueError: If ``x_min`` or ``x_max`` are not finite, if ``x_max <= x_min``,
                    or if ``log`` is ``True`` and ``x_min <= 0`` or ``x_max <= 0``.
    """
    validate_n_bins(n_bins)

    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("x_min and x_max must be finite numbers.")

    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min.")

    if log and (x_min <= 0 or x_max <= 0):
        raise ValueError("log-/geometric-spaced bins require x_min > 0 and x_max > 0.")


def validate_axis_and_weights(
    x: Any,
    weights: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Validates axis values and weights for binning.

    Args:
        x: 1D array-like of axis values.
        weights: 1D array-like of weights corresponding to ``x``.

    Returns:
        Tuple of validated numpy arrays: ``(x, weights)``.

    Raises:
        ValueError: If ``x`` and ``weights`` do not have the same shape, are not 1D,
                    contain non-finite values, have less than two points,
                    or if ``x`` is not strictly increasing.
    """
    x_arr = np.asarray(x, dtype=float)
    w_arr = np.asarray(weights, dtype=float)

    if x_arr.ndim != 1:
        raise ValueError("x must be 1D.")

    if w_arr.ndim != 1:
        raise ValueError("weights must be 1D.")

    if x_arr.shape != w_arr.shape:
        raise ValueError("x and weights must have the same shape.")

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
    """Validates mixed binning segments.

    Args:
        segments: Sequence of segment specifications. Each segment is a mapping
        with keys:
            - ``"method"``: Binning method name (e.g., ``"equidistant"``, ``"log"``,
              ``"equal_number"``, ``"equidistant_chi"``, etc.).
            - ``"n_bins"``: Number of bins for the segment.
            - ``"params"``: Optional mapping of parameters specific to the segment.
        total_n_bins: Optional total number of bins across all segments. If provided,
            the sum of segment ``"n_bins"`` values must match this number.

    Raises:
        ValueError: If ``segments`` is empty, required keys are missing,
            ``"n_bins"`` values are invalid, or the sum of ``"n_bins"`` does not
            match ``total_n_bins``.
        TypeError: If segment specifications or their fields are of incorrect types.
    """
    if not segments:
        raise ValueError("segments must be a non-empty sequence.")

    n_sum = 0
    for i, seg in enumerate(segments):
        if not isinstance(seg, Mapping):
            raise TypeError(
                f"Segment {i} must be a mapping, got {type(seg).__name__}."
            )

        if "method" not in seg or "n_bins" not in seg:
            raise ValueError(
                f"Segment {i} must contain at least 'method' and 'n_bins' keys."
            )

        if not isinstance(seg["method"], str):
            raise TypeError(f"Segment {i}: 'method' must be a string.")

        if not isinstance(seg["n_bins"], int):
            raise TypeError(f"Segment {i}: 'n_bins' must be an int.")

        resolve_binning_method(seg["method"])

        n_bins = seg["n_bins"]
        validate_n_bins(n_bins)

        params = seg.get("params", None)
        if params is not None and not isinstance(params, Mapping):
            raise TypeError(
                f"Segment {i}: 'params' must be a mapping when provided."
            )

        n_sum += n_bins

    if total_n_bins is not None and n_sum != total_n_bins:
        raise ValueError(
            f"Sum of segment n_bins is {n_sum}, but total_n_bins is {total_n_bins}."
        )
