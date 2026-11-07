"""Backward-compatible imports for tomographic statistics."""

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
]
