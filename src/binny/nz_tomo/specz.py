"""Backward-compatible spectroscopic-redshift binning utilities."""

from binny.binning.assignments import specz_selection_in_bin
from binny.binning.specz import (
    apply_response_matrix,
    build_specz_bins,
    build_specz_response_matrix,
    specz_gaussian_response_matrix,
)

__all__ = [
    "build_specz_bins",
    "specz_selection_in_bin",
    "build_specz_response_matrix",
    "apply_response_matrix",
    "specz_gaussian_response_matrix",
]
