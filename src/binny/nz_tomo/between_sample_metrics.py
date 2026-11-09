"""Backward-compatible wrappers for legacy tomography comparison metrics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from binny.tomography.comparison import (
    between_bin_overlap as _between_bin_overlap,
)
from binny.tomography.comparison import (
    between_interval_mass_matrix,
    bin_overlap,
    leakage_matrix,
    overlap_pairs,
    pearson_matrix,
)
from binny.tomography.comparison import (
    between_overlap_pairs as _between_overlap_pairs,
)
from binny.tomography.comparison import (
    between_pearson_matrix as _between_pearson_matrix,
)

__all__ = [
    "bin_overlap",
    "between_bin_overlap",
    "overlap_pairs",
    "between_overlap_pairs",
    "leakage_matrix",
    "between_interval_mass_matrix",
    "pearson_matrix",
    "between_pearson_matrix",
]


def between_bin_overlap(
    z: Any,
    bins_a: Mapping[int, Any],
    bins_b: Mapping[int, Any],
    **kwargs: Any,
) -> dict[int, dict[int, float]]:
    """Compute between-sample overlap using a shared redshift grid."""
    return _between_bin_overlap(z, bins_a, z, bins_b, **kwargs)


def between_overlap_pairs(
    z: Any,
    bins_a: Mapping[int, Any],
    bins_b: Mapping[int, Any],
    **kwargs: Any,
) -> list[tuple[int, int, float]]:
    """Select between-sample pairs using a shared redshift grid."""
    return _between_overlap_pairs(z, bins_a, z, bins_b, **kwargs)


def between_pearson_matrix(
    z: Any,
    bins_a: Mapping[int, Any],
    bins_b: Mapping[int, Any],
    **kwargs: Any,
) -> dict[int, dict[int, float]]:
    """Compute between-sample Pearson correlations using a shared grid."""
    return _between_pearson_matrix(z, bins_a, z, bins_b, **kwargs)
