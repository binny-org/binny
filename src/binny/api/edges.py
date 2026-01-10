"""Public API for bin-edge construction methods."""

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
    """Unified public interface for bin-edge construction.

    Examples:
        edges = bin_edges("equidistant", x_min=0.0, x_max=3.0, n_bins=5)
        edges = bin_edges("log", x_min=10.0, x_max=2000.0, n_bins=20)
        edges = bin_edges("equal_number", x=z, weights=nz, n_bins=5)
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
