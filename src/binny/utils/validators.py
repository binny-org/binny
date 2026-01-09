"""Validation utilities for binning and axis-related functions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "validate_interval",
    "validate_axis_and_weights",
    "validate_n_bins",
    "validate_mixed_segments",
    "resolve_binning_method",
    "validate_response_matrix",
    "validated_float_arrays",
    "validate_probability_vector",
    "validate_same_shape",
]

FloatArray2D: TypeAlias = NDArray[np.float64]

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
    """Returns the canonical binning method identifier for a user-supplied name.

    This provides a small normalization layer for user input by accepting common
    aliases (case-insensitive) and mapping them to the internal method names used
    throughout the package. Normalizing method names early makes downstream binning
    code simpler and ensures consistent behavior across APIs.

    Args:
        name: Binning method name or alias (case-insensitive).

    Returns:
        Canonical method name: one of ``"equidistant"``, ``"log"``, ``"equal_number"``,
        ``"equal_information"``, ``"equidistant_chi"``, or ``"geometric"``.

    Raises:
        ValueError: If ``name`` is not a recognized method name or alias.
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
    """Validates a requested number of bins.

    This guards against invalid bin counts and accidental huge allocations by
    enforcing positivity, optional constraints on allowing a single bin, and an
    upper bound. It is typically used at API boundaries before constructing bin
    edges or allocating arrays that scale with ``n_bins``.

    Args:
        n_bins: Number of bins.
        allow_one: If False, requires ``n_bins > 1``.
        max_bins: Upper bound to guard against accidental huge allocations.

    Raises:
        TypeError: If ``n_bins`` is not an integer.
        ValueError: If ``n_bins <= 0``, if ``allow_one`` is False and ``n_bins == 1``,
            or if ``n_bins > max_bins``.
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
    """Validates an axis interval and binning mode for edge construction.

    This checks that the interval endpoints are finite and ordered, and that the
    interval is compatible with the requested spacing mode. It is useful for
    bin-edge builders that assume a well-defined interval (and for log/geometric
    spacing, strictly positive bounds).

    Args:
        x_min: Minimum value of the axis.
        x_max: Maximum value of the axis.
        n_bins: Number of bins.
        log: Whether the bins are logarithmically (or geometric) spaced.

    Raises:
        TypeError: If ``n_bins`` is not an integer (via ``validate_n_bins``).
        ValueError: If ``n_bins`` is not positive (via ``validate_n_bins``), if
            ``x_min`` or ``x_max`` are not finite, if ``x_max <= x_min``, or if
            ``log`` is True and either bound is non-positive.
    """
    validate_n_bins(n_bins)

    if not np.isfinite(x_min) or not np.isfinite(x_max):
        raise ValueError("x_min and x_max must be finite numbers.")

    if x_max <= x_min:
        raise ValueError("x_max must be greater than x_min.")

    if log and (x_min <= 0 or x_max <= 0):
        raise ValueError("log-/geometric-spaced bins require x_min > 0 and x_max > 0.")


def validate_axis_and_weights(
    x: ArrayLike,
    weights: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Returns validated 1D axis values and weights as float64 arrays.

    This validates a sampling axis and a corresponding weight array for use in
    binning routines (e.g., equal-number or equal-information edges). It ensures
    both inputs are 1D, aligned in length, finite, and suitable for algorithms
    that assume a strictly increasing axis.

    Args:
        x: 1D array-like of axis values.
        weights: 1D array-like of weights corresponding to ``x``.

    Returns:
        Tuple ``(x_arr, w_arr)`` as 1D ``float64`` NumPy arrays.

    Raises:
        ValueError: If ``x`` is not 1D, if ``weights`` is not 1D, if they have
            different shapes, if either contains non-finite values, if ``x`` has
            fewer than two points, or if ``x`` is not strictly increasing.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)

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
    """Validates a mixed-binning segment specification.

    This checks that a sequence of segment dictionaries is well-formed for mixed
    binning workflows, where different binning methods are applied over different
    regions. It verifies required fields, validates each segment bin count, ensures
    each method name is recognized, and optionally enforces that segment bin counts
    sum to an expected total.

    Args:
        segments: Sequence of segment specifications. Each segment must be a mapping
            containing:
            - ``"method"``: Binning method name or alias.
            - ``"n_bins"``: Number of bins in the segment.
            Optionally, a segment may include:
            - ``"params"``: Mapping of method-specific parameters.
        total_n_bins: Optional expected total number of bins across all segments.

    Raises:
        ValueError: If ``segments`` is empty, if a segment is missing required keys,
            or if ``total_n_bins`` is provided and the sum of segment ``"n_bins"``
            does not match it.
        TypeError: If ``segments`` is not a sequence of mappings, if a segment
            ``"method"`` is not a string, if a segment ``"n_bins"`` is not an int,
            or if a provided ``"params"`` is not a mapping.
        ValueError: If any segment ``"method"`` is not recognized (via
            ``resolve_binning_method``), or if any segment ``"n_bins"`` is invalid
            (via ``validate_n_bins``).
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

        if not isinstance(seg["method"], str):
            raise TypeError(f"Segment {i}: 'method' must be a string.")

        if not isinstance(seg["n_bins"], int):
            raise TypeError(f"Segment {i}: 'n_bins' must be an int.")

        resolve_binning_method(seg["method"])

        n_bins = seg["n_bins"]
        validate_n_bins(n_bins)

        params = seg.get("params", None)
        if params is not None and not isinstance(params, Mapping):
            raise TypeError(f"Segment {i}: 'params' must be a mapping when provided.")

        n_sum += n_bins

    if total_n_bins is not None and n_sum != total_n_bins:
        raise ValueError(
            f"Sum of segment n_bins is {n_sum}, but total_n_bins is {total_n_bins}."
        )


def validate_response_matrix(matrix: FloatArray2D, n_bins: int) -> None:
    """Validates a bin-to-bin misassignment (response) matrix.

    This checks that a response matrix used to model bin misassignment is compatible
    with a given number of bins and behaves like a column-stochastic mapping. It is
    commonly used for photo-z or classification confusion matrices where each column
    represents the distribution of assigned bins for a true bin.

    Args:
        matrix: 2D NumPy array representing the response/misassignment matrix.
        n_bins: Expected number of bins; ``matrix`` must have shape
            ``(n_bins, n_bins)``.

    Raises:
        TypeError: If ``n_bins`` is not an integer (via ``validate_n_bins``).
        ValueError: If ``n_bins`` is not positive (via ``validate_n_bins``), if
            ``matrix`` does not have shape ``(n_bins, n_bins)``, if ``matrix`` contains
            non-finite values, if it contains entries less than ``-1e-15``, or if the
            (clipped) column sums are not close to 1 within tolerance.
    """
    validate_n_bins(n_bins)
    if matrix.shape != (n_bins, n_bins):
        raise ValueError(f"misassignment_matrix must have shape ({n_bins}, {n_bins}).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("misassignment_matrix must be finite.")
    if np.any(matrix < -1e-15):
        raise ValueError("misassignment_matrix must be non-negative.")
    matrix_clip = np.maximum(matrix, 0.0)
    col_sums = matrix_clip.sum(axis=0)
    if not np.allclose(col_sums, 1.0, rtol=1e-6, atol=1e-10):
        raise ValueError(
            "Each column of misassignment_matrix must sum to 1 (column-stochastic)."
        )


def validated_float_arrays(x: ArrayLike, y: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Returns two validated 1D float64 arrays with matched shape and finite values.

    This is a convenience wrapper for workflows that take paired 1D arrays and need
    them validated and converted to ``float64`` consistently. It is commonly used
    before numerical operations that assume aligned samples (e.g., an axis and an
    associated function evaluated on that axis).

    Args:
        x: First array-like input.
        y: Second array-like input.

    Returns:
        Tuple ``(x_arr, y_arr)`` as 1D ``float64`` NumPy arrays.

    Raises:
        ValueError: If either input is not 1D, if the shapes differ, if either contains
            non-finite values, if the first input has fewer than two points, or if the
            first input is not strictly increasing.
    """
    x_arr, y_arr = validate_axis_and_weights(x, y)
    return x_arr, y_arr


def validate_probability_vector(
    p: ArrayLike,
    *,
    name: str = "p",
    rtol: float = 1e-6,
    atol: float = 1e-12,
    allow_empty: bool = False,
) -> FloatArray:
    """Returns a validated 1D probability vector as float64.

    Checks:
    - 1D (and non-empty unless allow_empty=True)
    - finite
    - nonnegative
    - sums to 1 within tolerance

    Args:
        p: Array-like probability vector.
        name: Name used in error messages.
        rtol: Relative tolerance for the sum-to-one check.
        atol: Absolute tolerance for the sum-to-one check.
        allow_empty: If True, allows empty vectors (returns empty float64 array).

    Returns:
        1D float64 NumPy array.

    Raises:
        ValueError: If the input is not a valid probability vector.
    """
    arr = np.asarray(p, dtype=np.float64)

    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    if arr.size == 0:
        if allow_empty:
            return arr
        raise ValueError(f"{name} must be non-empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite.")
    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be nonnegative.")

    s = float(np.sum(arr))
    if not np.isclose(s, 1.0, rtol=rtol, atol=atol):
        raise ValueError(f"{name} must sum to 1 within tolerance (got {s}).")

    return arr


def validate_same_shape(
    a: ArrayLike,
    b: ArrayLike,
    *,
    name_a: str = "a",
    name_b: str = "b",
) -> None:
    """Validates that two array-likes have the same shape."""
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"{name_a} and {name_b} must have the same shape.")
