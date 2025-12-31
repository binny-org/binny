"""Functions to build photometric-redshift–smeared n(z) distributions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.special import erf

from binny.core.validators import validate_axis_and_weights, validate_n_bins
from binny.utils.broadcasting import as_per_bin
from binny.utils.normalization import normalize_1d

__all__ = [
    "build_photoz_bins",
    "true_redshift_distribution",
]


def build_photoz_bins(
    z: Any,
    nz: Any,
    bin_edges: Any,
    scatter_scale_per_bin: Sequence[float],
    mean_offset_per_bin: Sequence[float],
    *,
    mean_scale_per_bin: Sequence[float] | float = 1.0,
    outlier_frac_per_bin: Sequence[float] | float = 0.0,
    outlier_scatter_scale_per_bin: Sequence[float] | float | None = None,
    outlier_mean_offset_per_bin: Sequence[float] | float = 0.0,
    outlier_mean_scale_per_bin: Sequence[float] | float = 1.0,
    normalize_input: bool = True,
    normalize_bins: bool = True,
    norm_method: str = "trapezoid",
) -> dict[int, np.ndarray]:
    """Builds photo-z-smeared redshift distributions per tomographic bin.

    This function constructs per-bin true-redshift distributions by combining an
    intrinsic parent distribution ``n(z)`` with a photo-z selection in observed
    redshift, parameterized by per-bin scatter and mean-mapping parameters. For
    each bin defined by ``bin_edges``, the corresponding distribution is computed
    with :func:`true_redshift_distribution`.

    The core photo-z model uses:
    - Mean mapping: ``mu(z) = mean_scale * z - mean_offset``
    - Scatter: ``sigma(z) = scatter_scale * (1 + z)``

    An optional outlier mixture component can be included via the ``outlier_*``
    parameters.

    Args:
        z: One-dimensional redshift grid.
        nz: Parent redshift distribution evaluated on ``z``.
        bin_edges: One-dimensional array of tomographic bin edges in observed
            (photo-z) space. Must have length ``n_bins + 1``.
        scatter_scale_per_bin: Per-bin core scatter amplitudes. Must have length
            ``n_bins`` and is interpreted as ``scatter_scale * (1 + z)``.
        mean_offset_per_bin: Per-bin additive mean offsets in the mapping from
            true to observed redshift. Must have length ``n_bins``.
        mean_scale_per_bin: Optional per-bin multiplicative mean scales. May be a
            scalar (applied to all bins) or a sequence of length ``n_bins``.
        outlier_frac_per_bin: Optional per-bin outlier fractions in ``[0, 1]``.
            May be a scalar or a sequence of length ``n_bins``.
        outlier_scatter_scale_per_bin: Optional per-bin outlier scatter
            amplitudes. May be a scalar, a sequence of length ``n_bins``, or
            None to disable the outlier component.
        outlier_mean_offset_per_bin: Optional per-bin additive mean offsets for
            the outlier component. May be a scalar or a sequence of length
            ``n_bins``.
        outlier_mean_scale_per_bin: Optional per-bin multiplicative mean scales
            for the outlier component. May be a scalar or a sequence of length
            ``n_bins``.
        normalize_input: Whether to normalize the input ``nz`` before binning.
        normalize_bins: Whether to normalize each output bin distribution.
        norm_method: Normalization method passed to :func:`normalize_1d`.

    Returns:
        A mapping from bin index to the corresponding photo-z-smeared true
        redshift distribution evaluated on ``z``.

    Raises:
        ValueError: If ``bin_edges`` does not define a valid number of bins.
        ValueError: If ``scatter_scale_per_bin`` or ``mean_offset_per_bin`` does
            not have length ``n_bins``.
        ValueError: If ``normalize_input`` is True and the input ``nz`` already
            appears normalized.
    """
    z_arr, n_arr = validate_axis_and_weights(z, nz)

    bin_edges_arr = np.asarray(bin_edges, dtype=float)
    n_bins = bin_edges_arr.size - 1
    validate_n_bins(n_bins)

    if len(scatter_scale_per_bin) != n_bins or len(mean_offset_per_bin) != n_bins:
        raise ValueError(
            "scatter_scale_per_bin and mean_offset_per_bin must have length n_bins."
        )

    if normalize_input:
        total = np.trapezoid(n_arr, z_arr)
        if np.isclose(total, 1.0, rtol=1e-3, atol=1e-4):
            raise ValueError(
                "build_photoz_bins: normalize_input=True but intrinsic nz already "
                f"looks normalised (integral n(z) dz approx {total:.4f}). "
                "Set normalize_input=False if nz is already normalised."
            )
        n_arr = normalize_1d(z_arr, n_arr, method=norm_method)

    mean_scale = as_per_bin(mean_scale_per_bin, n_bins, "mean_scale_per_bin")
    outlier_frac = as_per_bin(outlier_frac_per_bin, n_bins, "outlier_frac_per_bin")
    outlier_scatter = as_per_bin(
        outlier_scatter_scale_per_bin,
        n_bins,
        "outlier_scatter_scale_per_bin",
    )
    outlier_mean_offset = as_per_bin(
        outlier_mean_offset_per_bin,
        n_bins,
        "outlier_mean_offset_per_bin",
    )
    outlier_mean_scale = as_per_bin(
        outlier_mean_scale_per_bin,
        n_bins,
        "outlier_mean_scale_per_bin",
    )

    bins: dict[int, np.ndarray] = {}

    for i, (z_min, z_max) in enumerate(
        zip(bin_edges_arr[:-1], bin_edges_arr[1:], strict=False)
    ):
        nz_bin = true_redshift_distribution(
            z_arr,
            n_arr,
            bin_min=float(z_min),
            bin_max=float(z_max),
            scatter_scale=float(scatter_scale_per_bin[i]),
            mean_offset=float(mean_offset_per_bin[i]),
            mean_scale=float(mean_scale[i]),
            outlier_frac=float(outlier_frac[i]),
            outlier_scatter_scale=(
                None if outlier_scatter[i] is None else float(outlier_scatter[i])
            ),
            outlier_mean_offset=float(outlier_mean_offset[i]),
            outlier_mean_scale=float(outlier_mean_scale[i]),
        )
        if normalize_bins:
            nz_bin = normalize_1d(z_arr, nz_bin, method=norm_method)

        bins[i] = nz_bin

    return bins


def true_redshift_distribution(
    z: Any,
    nz: Any,
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
) -> np.ndarray:
    """Computes the true-redshift distribution for a single photo-z bin.

    The output distribution is computed as ``n(z) * P(bin | z)``, where
    ``P(bin | z)`` is the probability that an object at true redshift ``z`` is
    assigned to the observed photo-z bin ``[bin_min, bin_max]`` under a Gaussian
    error model with scatter scaling as ``scatter_scale * (1 + z)``.

    The core photo-z model uses:
    - Mean mapping: ``mu(z) = mean_scale * z - mean_offset``
    - Scatter: ``sigma(z) = scatter_scale * (1 + z)``

    An optional outlier mixture component is supported. When enabled, the
    mixture probability is::

        P(bin | z) =
            (1 - outlier_frac) * P_core(bin | z)
            + outlier_frac * P_out(bin | z)

    Args:
        z: One-dimensional redshift grid.
        nz: Parent redshift distribution evaluated on ``z``.
        bin_min: Lower edge of the photo-z bin.
        bin_max: Upper edge of the photo-z bin.
        scatter_scale: Core scatter amplitude. The per-object scatter is
            ``scatter_scale * (1 + z)``.
        mean_offset: Additive mean offset in the mapping from true to observed
            redshift.
        mean_scale: Multiplicative mean scale in the core component.
        outlier_frac: Outlier mixture fraction in ``[0, 1]``.
        outlier_scatter_scale: Outlier scatter amplitude. The per-object scatter
            is ``outlier_scatter_scale * (1 + z)``. The outlier component is used
            only when ``outlier_frac > 0`` and ``outlier_scatter_scale`` is not
            None.
        outlier_mean_offset: Additive mean offset in the outlier component.
        outlier_mean_scale: Multiplicative mean scale in the outlier component.

    Returns:
        The photo-z-selected true-redshift distribution evaluated on ``z``.

    Raises:
        ValueError: If ``outlier_frac`` is not in the interval ``[0, 1]``.
    """
    z_arr, n_arr = validate_axis_and_weights(z, nz)

    if not (0.0 <= outlier_frac <= 1.0):
        raise ValueError("outlier_frac must lie in [0, 1].")

    scatter_core = np.maximum(scatter_scale * (1.0 + z_arr), 1e-10)
    mu_core = mean_scale * z_arr - mean_offset

    sqrt2 = np.sqrt(2.0)
    upper_core = (bin_max - mu_core) / (sqrt2 * scatter_core)
    lower_core = (bin_min - mu_core) / (sqrt2 * scatter_core)
    p_core = 0.5 * (erf(upper_core) - erf(lower_core))

    if outlier_frac > 0.0 and outlier_scatter_scale is not None:
        scatter_out = np.maximum(outlier_scatter_scale * (1.0 + z_arr), 1e-10)
        mu_out = outlier_mean_scale * z_arr + outlier_mean_offset

        upper_out = (bin_max - mu_out) / (sqrt2 * scatter_out)
        lower_out = (bin_min - mu_out) / (sqrt2 * scatter_out)
        p_out = 0.5 * (erf(upper_out) - erf(lower_out))

        p_bin_given_z = (1.0 - outlier_frac) * p_core + outlier_frac * p_out
    else:
        p_bin_given_z = p_core

    return n_arr * p_bin_given_z
