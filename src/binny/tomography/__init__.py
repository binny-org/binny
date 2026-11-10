"""Tomographic redshift distributions and scientific operations."""

from binny.tomography.builders import build_tomography_from_spec
from binny.tomography.combinations import make_bin_combo_filter
from binny.tomography.comparison import (
    between_bin_overlap,
    between_interval_mass_matrix,
    between_overlap_pairs,
    between_pearson_matrix,
    bin_overlap,
    leakage_matrix,
    overlap_pairs,
    pearson_matrix,
)
from binny.tomography.composition import (
    align_bin_grids,
    combine_parent_nz,
    combine_tomographies,
    interpolate_tomography,
    sample_bin_labels,
    sample_bins,
    sample_combinations,
)
from binny.tomography.core import Tomography, build_tomography
from binny.tomography.predefined import (
    build_predefined_tomography,
    load_predefined_bins,
)
from binny.tomography.stats import (
    bin_centers,
    bin_moments,
    bin_quantiles,
    comoving_density_per_bin,
    galaxy_count_per_bin,
    galaxy_density_per_bin,
    galaxy_fraction_per_bin,
    in_range_fraction,
    in_range_fraction_per_bin,
    parent_stats,
    peak_flags,
    peak_flags_per_bin,
    population_stats,
    shape_stats,
)

__all__ = [
    "Tomography",
    "build_tomography",
    "build_tomography_from_spec",
    "build_predefined_tomography",
    "load_predefined_bins",
    "make_bin_combo_filter",
    "bin_centers",
    "bin_moments",
    "bin_quantiles",
    "comoving_density_per_bin",
    "galaxy_count_per_bin",
    "galaxy_density_per_bin",
    "galaxy_fraction_per_bin",
    "in_range_fraction",
    "in_range_fraction_per_bin",
    "parent_stats",
    "peak_flags",
    "peak_flags_per_bin",
    "population_stats",
    "shape_stats",
    "bin_overlap",
    "overlap_pairs",
    "leakage_matrix",
    "pearson_matrix",
    "between_bin_overlap",
    "between_overlap_pairs",
    "between_interval_mass_matrix",
    "between_pearson_matrix",
    "combine_parent_nz",
    "interpolate_tomography",
    "align_bin_grids",
    "combine_tomographies",
    "sample_bin_labels",
    "sample_bins",
    "sample_combinations",
]
