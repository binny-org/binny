"""Backward-compatible imports for tomographic bin comparisons."""

from binny.tomography.comparison import (
    bin_overlap,
    leakage_matrix,
    overlap_pairs,
    pearson_matrix,
)

__all__ = [
    "bin_overlap",
    "leakage_matrix",
    "overlap_pairs",
    "pearson_matrix",
]
