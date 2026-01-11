"""Axis grid builders.

This module provides small helpers to construct 1D sampling grids for
numerical work (e.g., k, ell, z). The grids include both endpoints and are
returned as ``float64`` NumPy arrays.

Functions
---------
- ``linear_grid``: Uniform spacing in x.
- ``log_grid``: Uniform spacing in log(x) (geometric progression).
"""

from __future__ import annotations

import numpy as np

from binny.axes.bin_edges import log_edges
from binny.utils.validators import validate_grid_spec

__all__ = [
    "linear_grid",
    "log_grid",
]


def linear_grid(x_min: float, x_max: float, n: int) -> np.ndarray:
    """Builds a linearly spaced 1D grid including endpoints.

    Args:
        x_min: Lower endpoint of the grid.
        x_max: Upper endpoint of the grid.
        n: Number of grid points (>= 2).

    Returns:
        A 1D NumPy array of shape ``(n,)`` with dtype ``float64``.

    Raises:
        TypeError: If ``x_min``/``x_max`` are not real numbers or ``n`` is not
            integer-like.
        ValueError: If the grid specification is invalid (e.g., non-finite or
            non-increasing endpoints, or ``n < 2``).
    """
    x0, x1, n_int = validate_grid_spec(x_min, x_max, n, log=False)
    return np.linspace(x0, x1, n_int, dtype=np.float64)


def log_grid(x_min: float, x_max: float, n: int) -> np.ndarray:
    """Builds a log-spaced 1D grid including endpoints.

    Values are evenly spaced in ``log(x)`` (geometric progression).

    Args:
        x_min: Lower endpoint of the grid (must be > 0).
        x_max: Upper endpoint of the grid (must be > ``x_min``).
        n: Number of grid points (>= 2).

    Returns:
        A 1D NumPy array of shape ``(n,)`` with dtype ``float64``.

    Raises:
        TypeError: If ``x_min``/``x_max`` are not real numbers or ``n`` is not
            integer-like.
        ValueError: If the grid specification is invalid (e.g., non-finite,
            non-increasing, or non-positive endpoints, or ``n < 2``).
    """
    x0, x1, n_int = validate_grid_spec(x_min, x_max, n, log=True)
    return log_edges(x0, x1, n_int - 1).astype(np.float64, copy=False)
