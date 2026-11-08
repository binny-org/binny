"""Photometric-redshift tomographic assignment probabilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import erf

from binny.utils.broadcasting import as_per_bin
from binny.utils.types import FloatArray

__all__ = [
    "photoz_assignment",
    "photoz_assignment_probability",
]


def photoz_assignment_probability(
    z: FloatArray,
    bin_min: float,
    bin_max: float,
    scatter_scale: float,
    mean_offset: float,
    *,
    mean_scale: float = 1.0,
    outlier_frac: float = 0.0,
    outlier_scatter_scale: float | None = None,
    outlier_mean_offset: float = 0.0,
    outlier_mean_scale: float = 1.0,
) -> FloatArray:
    """Compute the probability of assignment to one photo-z bin.

    Args:
        z: True-redshift grid.
        bin_min: Lower observed photo-z edge.
        bin_max: Upper observed photo-z edge.
        scatter_scale: Core Gaussian scatter scale.
        mean_offset: Core Gaussian mean offset.
        mean_scale: Core Gaussian mean scale.
        outlier_frac: Outlier mixture fraction.
        outlier_scatter_scale: Outlier Gaussian scatter scale.
        outlier_mean_offset: Outlier Gaussian mean offset.
        outlier_mean_scale: Outlier Gaussian mean scale.

    Returns:
        Bin-assignment probabilities evaluated on the true-z grid.

    Raises:
        ValueError: If the photo-z parameters are invalid.
    """
    z_arr = np.asarray(z, dtype=float)

    if not (0.0 <= outlier_frac <= 1.0):
        raise ValueError("outlier_frac must lie in [0, 1].")

    if outlier_frac > 0.0 and outlier_scatter_scale is None:
        raise ValueError(
            "outlier_scatter_scale must be set when outlier_frac > 0. "
            "Use 0.0 for a deterministic (no-uncertainty) outlier component."
        )

    if mean_scale <= 0.0:
        raise ValueError("mean_scale must be > 0.")
    if scatter_scale < 0.0:
        raise ValueError("scatter_scale must be >= 0.")

    outliers_enabled = outlier_frac > 0.0
    if outliers_enabled:
        if outlier_mean_scale <= 0.0:
            raise ValueError("outlier_mean_scale must be > 0.")
        if outlier_scatter_scale < 0.0:
            raise ValueError("outlier_scatter_scale must be >= 0.")

    p_core = _bin_prob_gaussian_photoz(
        z_arr,
        bin_min=bin_min,
        bin_max=bin_max,
        scatter_scale=scatter_scale,
        mean_offset=mean_offset,
        mean_scale=mean_scale,
    )

    if not outliers_enabled:
        return p_core

    p_out = _bin_prob_gaussian_photoz(
        z_arr,
        bin_min=bin_min,
        bin_max=bin_max,
        scatter_scale=outlier_scatter_scale,
        mean_offset=outlier_mean_offset,
        mean_scale=outlier_mean_scale,
    )

    return (1.0 - outlier_frac) * p_core + outlier_frac * p_out


def _bin_prob_gaussian_photoz(
    z: FloatArray,
    *,
    bin_min: float,
    bin_max: float,
    scatter_scale: float,
    mean_offset: float,
    mean_scale: float = 1.0,
) -> FloatArray:
    """Evaluate a Gaussian photo-z selection probability.

    Args:
        z: True-redshift grid.
        bin_min: Lower observed photo-z edge.
        bin_max: Upper observed photo-z edge.
        scatter_scale: Scatter scale multiplying ``1 + z``.
        mean_offset: Offset subtracted from the Gaussian mean.
        mean_scale: Multiplicative factor in the Gaussian mean.

    Returns:
        Probability of assignment to the observed bin.

    Raises:
        ValueError: If scatter_scale is negative.
    """
    z_arr = np.asarray(z, dtype=float)

    if scatter_scale < 0.0:
        raise ValueError("scatter_scale must be >= 0.")

    mu = mean_scale * z_arr - mean_offset

    if scatter_scale == 0.0:
        return ((mu >= bin_min) & (mu < bin_max)).astype(np.float64)

    sigma = np.maximum(scatter_scale * (1.0 + z_arr), 1e-10)

    sqrt2 = np.sqrt(2.0)
    t_max = (bin_max - mu) / (sqrt2 * sigma)
    t_min = (bin_min - mu) / (sqrt2 * sigma)

    return 0.5 * (erf(t_max) - erf(t_min))


def photoz_assignment(
    z: FloatArray,
    edges: FloatArray,
    *,
    scatter_scale: Sequence[float] | float = 0.0,
    mean_offset: Sequence[float] | float = 0.0,
    mean_scale: Sequence[float] | float = 1.0,
    outlier_frac: Sequence[float] | float = 0.0,
    outlier_scatter_scale: Sequence[float] | float | None = None,
    outlier_mean_offset: Sequence[float] | float = 0.0,
    outlier_mean_scale: Sequence[float] | float = 1.0,
) -> FloatArray:
    """Return photo-z bin-assignment probabilities.

    Model parameters may be specified globally or independently
    for each tomographic bin.

    Args:
        z: True-redshift grid.
        edges: Bin edges in observed photo-z space.
        scatter_scale: Core Gaussian scatter scale.
        mean_offset: Core Gaussian mean offset.
        mean_scale: Core Gaussian mean scale.
        outlier_frac: Outlier mixture fraction.
        outlier_scatter_scale: Outlier Gaussian scatter scale.
        outlier_mean_offset: Outlier Gaussian mean offset.
        outlier_mean_scale: Outlier Gaussian mean scale.

    Returns:
        Selection probabilities with shape ``(n_bins, n_z)``.
    """
    z_arr = np.asarray(z, dtype=float)
    edges_arr = np.asarray(edges, dtype=float)
    n_bins = edges_arr.size - 1

    scatter = as_per_bin(scatter_scale, n_bins, "scatter_scale")
    offset = as_per_bin(mean_offset, n_bins, "mean_offset")
    scale = as_per_bin(mean_scale, n_bins, "mean_scale")
    frac = as_per_bin(outlier_frac, n_bins, "outlier_frac")

    if outlier_scatter_scale is None:
        out_scatter = [None] * n_bins
    else:
        out_scatter = as_per_bin(
            outlier_scatter_scale,
            n_bins,
            "outlier_scatter_scale",
        )

    out_offset = as_per_bin(
        outlier_mean_offset,
        n_bins,
        "outlier_mean_offset",
    )
    out_scale = as_per_bin(
        outlier_mean_scale,
        n_bins,
        "outlier_mean_scale",
    )

    return np.stack(
        [
            photoz_assignment_probability(
                z_arr,
                bin_min=float(edges_arr[i]),
                bin_max=float(edges_arr[i + 1]),
                scatter_scale=float(scatter[i]),
                mean_offset=float(offset[i]),
                mean_scale=float(scale[i]),
                outlier_frac=float(frac[i]),
                outlier_scatter_scale=(None if out_scatter[i] is None else float(out_scatter[i])),
                outlier_mean_offset=float(out_offset[i]),
                outlier_mean_scale=float(out_scale[i]),
            )
            for i in range(n_bins)
        ],
        axis=0,
    )
