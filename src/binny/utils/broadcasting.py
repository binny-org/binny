# src/binny/utils/broadcasting.py

from __future__ import annotations
from collections.abc import Sequence

__all__ = ["as_per_bin"]


def as_per_bin(x: float | int | Sequence | None, n_bins: int, name: str):
    """
    Broadcast a scalar/None/sequence to a per-bin list of length n_bins.

    Rules:
    - If x is a float or int → repeat for all bins.
    - If x is None → return [None] * n_bins.
    - If x is a sequence → validate length n_bins.
    - Raise ValueError if lengths mismatch.

    Args:
        x: A scalar, None, or sequence.
        n_bins: Number of bins.
        name: Name of the parameter (for error messages).

    Returns:
        List of length n_bins.
    """
    # Case 1: None → propagate as None for all bins
    if x is None:
        return [None] * n_bins

    # Case 2: scalar → broadcast
    if isinstance(x, (float, int)):
        return [float(x)] * n_bins

    # Case 3: sequence → check length
    try:
        seq = list(x)
    except TypeError:
        raise TypeError(f"{name} must be scalar, None, or a sequence.")

    if len(seq) != n_bins:
        raise ValueError(f"{name} must have length {n_bins}, got {len(seq)}.")

    return [float(v) if v is not None else None for v in seq]
