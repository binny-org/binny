"""Selection functions for generic binning."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from binny.nz_tomo.photoz import photoz_assignment_probability
from binny.nz_tomo.specz import specz_selection_in_bin
from binny.utils.broadcasting import as_per_bin
from binny.utils.validators import validate_response_matrix

__all__ = ["deterministic_assignment", "photoz_assignment"]


def deterministic_assignment(
    z: np.ndarray,
    edges: np.ndarray,
    *,
    completeness: Sequence[float] | float = 1.0,
    response_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Return deterministic bin selections with optional bin mixing.

    Args:
        z: Grid on which the selections are evaluated.
        edges: Bin edges in the same coordinate as ``z``.
        completeness: Scalar or per-bin completeness in [0, 1].
        response_matrix: Optional column-stochastic matrix mapping
            true bins to observed bins.

    Returns:
        Selection array with shape ``(n_bins, n_z)``.
    """
    z_arr = np.asarray(z, dtype=float)
    edges_arr = np.asarray(edges, dtype=float)
    n_bins = edges_arr.size - 1

    completeness_arr = as_per_bin(
        completeness,
        n_bins,
        "completeness",
    )

    selection = np.stack(
        [
            specz_selection_in_bin(
                z_arr,
                float(edges_arr[i]),
                float(edges_arr[i + 1]),
                completeness=float(completeness_arr[i]),
            )
            for i in range(n_bins)
        ],
        axis=0,
    )

    if response_matrix is not None:
        matrix = np.asarray(response_matrix, dtype=float)
        validate_response_matrix(matrix, n_bins)
        selection = matrix @ selection

    return selection


def photoz_assignment(
    z: np.ndarray,
    edges: np.ndarray,
    *,
    scatter_scale: Sequence[float] | float = 0.0,
    mean_offset: Sequence[float] | float = 0.0,
    mean_scale: Sequence[float] | float = 1.0,
    outlier_frac: Sequence[float] | float = 0.0,
    outlier_scatter_scale: Sequence[float] | float | None = None,
    outlier_mean_offset: Sequence[float] | float = 0.0,
    outlier_mean_scale: Sequence[float] | float = 1.0,
) -> np.ndarray:
    """Return photo-z bin-assignment probabilities on the true-z grid.

    The assignment is based on a Gaussian photo-z model with an optional
    Gaussian outlier component. All model parameters may be specified
    globally or independently for each tomographic bin.

    Args:
        z: True-redshift grid.
        edges: Bin edges in observed photo-z space.
        scatter_scale: Core Gaussian scatter scale, scalar or per bin.
        mean_offset: Core Gaussian mean offset, scalar or per bin.
        mean_scale: Core Gaussian mean scale, scalar or per bin.
        outlier_frac: Outlier mixture fraction, scalar or per bin.
        outlier_scatter_scale: Outlier scatter scale, scalar or per bin.
        outlier_mean_offset: Outlier Gaussian mean offset, scalar or per bin.
        outlier_mean_scale: Outlier Gaussian mean scale, scalar or per bin.

    Returns:
        Selection probabilities with shape ``(n_bins, n_z)``.
    """
    z_arr = np.asarray(z, dtype=float)
    edges_arr = np.asarray(edges, dtype=float)
    n_bins = edges_arr.size - 1

    scatter = as_per_bin(
        scatter_scale,
        n_bins,
        "scatter_scale",
    )
    offset = as_per_bin(
        mean_offset,
        n_bins,
        "mean_offset",
    )
    scale = as_per_bin(
        mean_scale,
        n_bins,
        "mean_scale",
    )
    frac = as_per_bin(
        outlier_frac,
        n_bins,
        "outlier_frac",
    )

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
