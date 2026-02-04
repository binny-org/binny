"""Axis grid builders.

Small helpers for building 1D grids commonly used in numerical work
(e.g. k, ell, z).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from binny.axes.bin_edges import log_edges
from binny.utils.validators import validate_grid_spec

__all__ = [
    "linear_grid",
    "log_grid",
    "grid",
]


def linear_grid(start: float, stop: float, n: int) -> np.ndarray:
    """Builds a linearly spaced 1D grid including endpoints.

    Args:
        start: Lower endpoint of the grid.
        stop: Upper endpoint of the grid.
        n: Number of grid points (>= 2).

    Returns:
        A 1D NumPy array of shape ``(n,)`` with dtype ``float64``.

    Raises:
        TypeError: If ``start``/``stop`` are not real numbers or ``n`` is not
            integer-like.
        ValueError: If the grid specification is invalid (e.g., non-finite or
            non-increasing endpoints, or ``n < 2``).
    """
    x0, x1, n_int = validate_grid_spec(start, stop, n, log=False)
    return np.linspace(x0, x1, n_int, dtype=np.float64)


def log_grid(start: float, stop: float, n: int) -> np.ndarray:
    """Builds a log-spaced 1D grid including endpoints.

    Values are evenly spaced in ``log(x)`` (geometric progression).

    Args:
        start: Lower endpoint of the grid (must be > 0).
        stop: Upper endpoint of the grid (must be > ``start``).
        n: Number of grid points (>= 2).

    Returns:
        A 1D NumPy array of shape ``(n,)`` with dtype ``float64``.

    Raises:
        TypeError: If ``start``/``stop`` are not real numbers or ``n`` is not
            integer-like.
        ValueError: If the grid specification is invalid (e.g., non-finite,
            non-increasing, or non-positive endpoints, or ``n < 2``).
    """
    x0, x1, n_int = validate_grid_spec(start, stop, n, log=True)
    return log_edges(x0, x1, n_int - 1).astype(np.float64, copy=False)


def grid(kind: str, /, **kwargs: Any):
    """Constructs a 1D sampling grid using a named strategy.

    Supported kinds (case-insensitive) and common aliases:
        - ``"linear"``: ``"lin"``, ``"uniform"``
        - ``"log"``: ``"log_grid"``, ``"logarithmic"``, ``"geom"``

    Args:
        kind: Grid strategy selector (case-insensitive).
        **kwargs: Parameters forwarded to the underlying grid builder.

    Returns:
        1D NumPy array of grid points (dtype ``float64``).

    Raises:
        ValueError: If ``kind`` is not recognized.
    """
    k = kind.lower()
    if k in {"linear", "lin", "uniform"}:
        return linear_grid(**kwargs)
    if k in {"log", "log_grid", "logarithmic", "geom", "geometric"}:
        return log_grid(**kwargs)
    raise ValueError(f"Unknown grid kind {kind!r}.")
