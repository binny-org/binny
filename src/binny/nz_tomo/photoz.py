"""Backward-compatible photometric-redshift binning utilities."""

from binny.binning.photoz import (
    build_photoz_bins,
    photoz_assignment_probability,
    true_redshift_distribution,
)

__all__ = [
    "build_photoz_bins",
    "true_redshift_distribution",
    "photoz_assignment_probability",
]
