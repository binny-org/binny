"""Grid construction API.

This module provides a single public entry point, :func:`grid`, that dispatches
to 1D sampling-grid builders in :mod:`binny.axes.grids`.
"""

from __future__ import annotations

from typing import Any

from binny.axes.grids import linear_grid, log_grid

__all__ = ["grid"]


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
