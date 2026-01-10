"""Public API for tomographic bin metrics (statistics + cross-bin similarity)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from binny.ztomo.bin_similarity import (
    bin_overlap,
    leakage_matrix,
    overlap_pairs,
    pearson_matrix,
)
from binny.ztomo.bin_stats import (
    bin_centers,
    bin_moments,
    bin_quantiles,
    galaxy_count_per_bin,
    galaxy_density_per_bin,
    galaxy_fraction_per_bin,
    in_range_fraction,
    in_range_fraction_per_bin,
    peak_flags,
    peak_flags_per_bin,
    summarize_bins,
)

__all__ = [
    # bin stats primitives
    "bin_moments",
    "bin_centers",
    "summarize_bins",
    "galaxy_fraction_per_bin",
    "galaxy_density_per_bin",
    "galaxy_count_per_bin",
    "bin_quantiles",
    "in_range_fraction",
    "in_range_fraction_per_bin",
    "peak_flags",
    "peak_flags_per_bin",
    # similarity metrics
    "bin_overlap",
    "overlap_pairs",
    "leakage_matrix",
    "pearson_matrix",
    # convenience
    "bin_summary",
]


def bin_summary(
    z: Any,
    bins: Mapping[int, Any],
    *,
    center_method: str = "mean",
    decimal_places: int | None = 3,
    density_total: float | None = None,
    survey_area: float | None = None,
) -> dict[str, Any]:
    """Convenience wrapper: centers + optional density/counts + summarize_bins.

    Returns a dict with:
      - centers
      - density_per_bin (or None)
      - count_per_bin (or None)
      - summary (output of summarize_bins)
    """
    centers = bin_centers(z, bins, method=center_method, decimal_places=decimal_places)

    density_per_bin = None
    count_per_bin = None

    if density_total is not None:
        density_per_bin, _ = galaxy_density_per_bin(
            z,
            bins,
            density_total=density_total,
        )

        if survey_area is not None:
            count_per_bin = galaxy_count_per_bin(density_per_bin, survey_area)

    summary = summarize_bins(
        z,
        bins,
        count_per_bin=count_per_bin,
        density_per_bin=density_per_bin,
        survey_area=survey_area,
    )

    return {
        "centers": centers,
        "density_per_bin": density_per_bin,
        "count_per_bin": count_per_bin,
        "summary": summary,
    }
