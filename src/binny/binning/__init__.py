"""Bin-edge construction, binning schemes, and assignment models."""

from binny.axes import (
    chebyshev_edges,
    equal_information_edges,
    equal_number_edges,
    equidistant_chi_edges,
    equidistant_edges,
    generalized_tomographic_edges,
    geometric_edges,
    log_edges,
    mixed_edges,
    power_edges,
    sinh_edges,
)
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
    photoz_assignment,
    photoz_assignment_probability,
    true_redshift_distribution,
)
from binny.binning.scheme_registry import (
    available_binning_schemes,
    get_binning_scheme,
    register_binning_scheme,
)
from binny.binning.specz import (
    apply_response_matrix,
    build_specz_response_matrix,
    specz_assignment,
    specz_gaussian_response_matrix,
)
from binny.utils.io import load_binning_recipe

__all__ = [
    "apply_response_matrix",
    "available_binning_schemes",
    "build_bins",
    "build_specz_response_matrix",
    "chebyshev_edges",
    "deterministic_assignment",
    "equal_information_edges",
    "equal_number_edges",
    "equidistant_chi_edges",
    "equidistant_edges",
    "generalized_tomographic_edges",
    "geometric_edges",
    "get_binning_scheme",
    "load_binning_recipe",
    "log_edges",
    "mixed_edges",
    "mixed_edges_from_segments",
    "photoz_assignment",
    "photoz_assignment_probability",
    "power_edges",
    "register_binning_scheme",
    "resolve_bin_edges",
    "segments_to_axes",
    "sinh_edges",
    "specz_assignment",
    "specz_gaussian_response_matrix",
    "specz_selection_in_bin",
    "true_redshift_distribution",
    "validate_bin_edges",
]
