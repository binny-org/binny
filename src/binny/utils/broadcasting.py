"""Utility to broadcast parameters to per-bin arrays."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["as_per_bin"]


def as_per_bin(x: float | int | Sequence | None, n_bins: int, name: str) -> np.ndarray:
    """Broadcasts a scalar/None/sequence to a per-bin array of length
    ``n_bins``.

    Rules:
    - If ``x`` is a float or int -> repeat for all bins.
    - If ``x`` is None -> return an object array of ``None`` with length ``n_bins``.
    - If ``x`` is a sequence -> validate length ``n_bins`` and broadcast elementwise.
    - Raise ValueError if lengths mismatch.

    Args:
        x: A scalar, None, or sequence.
        n_bins: Number of bins.
        name: Name of the parameter (for error messages).

    Returns:
        A NumPy array of length ``n_bins``. Uses ``dtype=float`` when possible,
        otherwise ``dtype=object`` (e.g. when values include ``None``).

    Raises:
        ValueError: If ``n_bins`` < 1 or if ``x`` is a sequence of incorrect length.
        TypeError: If ``x`` is not a scalar, None, or sequence.
    """
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")

    # Case 1: None -> propagate as None for all bins (object array)
    if x is None:
        return np.array([None] * n_bins, dtype=object)

    # Case 2: scalar -> broadcast (float array)
    if isinstance(x, float | int):
        return np.full(n_bins, float(x), dtype=float)

    # Case 3: sequence -> check length
    try:
        seq = list(x)  # type: ignore[arg-type]
    except TypeError as err:
        raise TypeError(f"{name} must be scalar, None, or a sequence.") from err

    if len(seq) != n_bins:
        raise ValueError(f"{name} must have length {n_bins}, got {len(seq)}.")

    # If any entry is None, we must preserve None -> object array
    if any(v is None for v in seq):
        return np.array(
            [float(v) if v is not None else None for v in seq], dtype=object
        )

    # Otherwise, safe to return float array
    return np.asarray([float(v) for v in seq], dtype=float)
