"""Binning, assignment models, and redshift-error calibration."""

from binny.binning.assignments import (
    deterministic_assignment,
    specz_selection_in_bin,
)
from binny.binning.core import build_bins
from binny.binning.edges import (
    mixed_edges_from_segments,
    resolve_bin_edges,
    segments_to_axes,
    validate_bin_edges,
)
from binny.binning.photoz import (
    build_photoz_bins,
    photoz_assignment,
    photoz_assignment_probability,
    true_redshift_distribution,
)
from binny.binning.photoz_fitting import (
    fit_photoz_errors,
    to_binny_photoz_params,
)
from binny.binning.scheme_registry import (
    available_binning_schemes,
    get_binning_scheme,
    register_binning_scheme,
)
from binny.binning.specz import (
    apply_response_matrix,
    build_specz_bins,
    build_specz_response_matrix,
    specz_assignment,
    specz_gaussian_response_matrix,
)

__all__ = [
    "apply_response_matrix",
    "available_binning_schemes",
    "build_bins",
    "build_photoz_bins",
    "build_specz_bins",
    "build_specz_response_matrix",
    "deterministic_assignment",
    "fit_photoz_errors",
    "get_binning_scheme",
    "mixed_edges_from_segments",
    "photoz_assignment",
    "photoz_assignment_probability",
    "register_binning_scheme",
    "resolve_bin_edges",
    "segments_to_axes",
    "specz_assignment",
    "specz_gaussian_response_matrix",
    "specz_selection_in_bin",
    "to_binny_photoz_params",
    "true_redshift_distribution",
    "validate_bin_edges",
]
