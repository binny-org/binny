"""Binning, assignment models, and redshift-error calibration."""

from binny.binning.core import build_bins
from binny.binning.edges import resolve_bin_edges, validate_bin_edges
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
from binny.binning.specz import (
    apply_response_matrix,
    build_specz_bins,
    build_specz_response_matrix,
    specz_assignment,
    specz_gaussian_response_matrix,
)

__all__ = [
    "build_bins",
    "resolve_bin_edges",
    "validate_bin_edges",
    "build_photoz_bins",
    "photoz_assignment",
    "photoz_assignment_probability",
    "true_redshift_distribution",
    "fit_photoz_errors",
    "to_binny_photoz_params",
    "apply_response_matrix",
    "build_specz_bins",
    "build_specz_response_matrix",
    "specz_assignment",
    "specz_gaussian_response_matrix",
]
