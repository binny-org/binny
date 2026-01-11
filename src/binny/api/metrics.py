"""Tomographic bin metrics API.

This module exposes a public, stable interface for two families of tomographic
bin diagnostics:

- **Bin statistics**: centers, moments, quantiles, in-range fractions, per-bin
  number density / counts, and compact summaries.
- **Cross-bin similarity**: overlap-based measures, leakage/overlap matrices,
  and correlation-style similarity (e.g., Pearson matrices).

Most functions are re-exported from :mod:`binny.ztomo.bin_stats` and
:mod:`binny.ztomo.bin_similarity` for convenience.
"""

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
    """Summarizes tomographic bins with optional densities, counts, and fractions.

    This is a convenience wrapper that computes:

    - Bin centers via :func:`binny.ztomo.bin_stats.bin_centers`.
    - Per-bin galaxy fraction via :func:`binny.ztomo.bin_stats.galaxy_fraction_per_bin`.
    - Per-bin number density via :func:`binny.ztomo.bin_stats.galaxy_density_per_bin`
      if ``density_total`` is provided.
    - Per-bin galaxy counts via :func:`binny.ztomo.bin_stats.galaxy_count_per_bin`
      if both ``density_total`` and ``survey_area`` are provided.
    - A compact bin report via :func:`binny.ztomo.bin_stats.summarize_bins`.

    The returned dictionary is designed for quick inspection, logging, or lightweight
    reporting. For full control, call the underlying functions directly (they are
    also re-exported from this module).

    Args:
        z: Redshift grid associated with the binned distributions.
        bins: Mapping from bin index to a per-bin distribution/weights array sampled
            on ``z`` (e.g., ``{0: nz0, 1: nz1, ...}``).
        center_method: Method used to compute bin centers (forwarded to
            :func:`binny.ztomo.bin_stats.bin_centers`).
        decimal_places: Optional rounding applied to the reported centers. Use
            ``None`` to disable rounding.
        density_total: Total number density (e.g., galaxies per arcmin^2) to apportion
            into per-bin densities. If ``None``, densities and counts are not computed.
        survey_area: Survey area used to convert per-bin densities to counts.
            Interpretation follows :func:`binny.ztomo.bin_stats.galaxy_count_per_bin`.
            Only used if ``density_total`` is provided.

    Returns:
        Dictionary with keys:

        - ``"centers"``: Bin centers (type depends on ``bin_centers`` output).
        - ``"fraction_per_bin"``: Fraction of the total distribution assigned to each
          bin (output of :func:`binny.ztomo.bin_stats.galaxy_fraction_per_bin`).
        - ``"density_per_bin"``: Per-bin densities, or ``None`` if not requested.
        - ``"count_per_bin"``: Per-bin counts, or ``None`` if not requested.
        - ``"summary"``: Output of :func:`binny.ztomo.bin_stats.summarize_bins`.

    Notes:
        This function does not enforce a particular bin representation beyond
        requiring that each bin can be interpreted consistently with the downstream
        helpers (centers, fractions, densities, summaries).

    Examples:
        >>> import numpy as np
        >>> from binny.api.metrics import bin_summary
        >>> z = np.linspace(0.0, 2.0, 5)
        >>> bins = {0: np.array([0.0, 1.0, 2.0, 1.0, 0.0]),
        ...         1: np.array([0.0, 0.5, 1.0, 0.5, 0.0])}
        >>> out = bin_summary(z, bins, center_method="mean", decimal_places=None)
        >>> sorted(out.keys())
        ['centers', 'count_per_bin', 'density_per_bin', 'fraction_per_bin', 'summary']
    """
    centers = bin_centers(z, bins, method=center_method, decimal_places=decimal_places)
    fraction_per_bin, _ = galaxy_fraction_per_bin(z, bins)

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
        "fraction_per_bin": fraction_per_bin,
        "density_per_bin": density_per_bin,
        "count_per_bin": count_per_bin,
        "summary": summary,
    }
