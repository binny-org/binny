"""Bin-edge construction API.

This module provides a single public entry point, :func:`bin_edges`, that
dispatches to the available bin-edge construction routines in
:mod:`binny.axes.bin_edges`.
"""

from __future__ import annotations

from typing import Any

from binny.axes.bin_edges import (
    equal_information_edges,
    equal_number_edges,
    equidistant_chi_edges,
    equidistant_edges,
    geometric_edges,
    log_edges,
)

__all__ = ["bin_edges"]


def bin_edges(method: str, /, **kwargs: Any):
    """Constructs bin edges using a named binning strategy.

    This function is a convenience dispatcher over bin-edge constructors such as
    equidistant, logarithmic, geometric, equal-number, equal-information, and
    equidistant-in-comoving-distance (chi) edges.

    The accepted keyword arguments depend on the selected method and are passed
    through to the corresponding implementation. Use the individual functions in
    :mod:`binny.axes.bin_edges` for full parameter documentation.

    Supported methods (case-insensitive) and common aliases:
        - ``"equidistant"``: ``"linear"``, ``"eq"``
        - ``"log"``: ``"log_edges"``, ``"logarithmic"``
        - ``"geometric"``: ``"geom"``, ``"geomspace"``
        - ``"equal_number"``: ``"equipopulated"``, ``"en"``
        - ``"equal_information"``: ``"ei"``
        - ``"equidistant_chi"``: ``"chi"``

    Args:
        method: Name of the bin-edge construction strategy (case-insensitive).
        **kwargs: Parameters forwarded to the underlying bin-edge constructor.

    Returns:
        A 1D NumPy array of bin edges with length ``n_bins + 1``.

    Raises:
        ValueError: If ``method`` is not recognized.

    Examples:
        >>> import numpy as np
        >>> from binny.api.edges import bin_edges
        >>> edges = bin_edges("equidistant", x_min=0.0, x_max=3.0, n_bins=5)
        >>> edges.shape
        (6,)

        >>> z = np.linspace(0.0, 2.0, 101)
        >>> nz = np.exp(-z)
        >>> edges = bin_edges("equal_number", x=z, weights=nz, n_bins=4)
        >>> edges.shape
        (5,)
    """
    m = method.lower()

    if m in {"equidistant", "linear", "eq"}:
        return equidistant_edges(**kwargs)
    if m in {"log", "log_edges", "logarithmic"}:
        return log_edges(**kwargs)
    if m in {"geometric", "geom", "geomspace"}:
        return geometric_edges(**kwargs)
    if m in {"equal_number", "equipopulated", "en"}:
        return equal_number_edges(**kwargs)
    if m in {"equal_information", "ei"}:
        return equal_information_edges(**kwargs)
    if m in {"equidistant_chi", "chi"}:
        return equidistant_chi_edges(**kwargs)

    raise ValueError(f"Unknown bin edge method '{method}'.")
