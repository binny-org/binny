"""Functions to build photometric-redshift–smeared tomographic bins.

This module provides helpers to construct true-redshift distributions selected
by observed-redshift (photo-z) tomographic bins. The main entry point is
:func:`build_photoz_bins`, which returns a dictionary mapping bin index to the
corresponding photo-z-selected true-redshift distribution evaluated on a common
true-z grid.

Examples
--------
>>> import numpy as np
>>> from binny.ztomo.photoz import build_photoz_bins
>>> z = np.linspace(0.0, 3.0, 501)
>>> nz = z**2 * np.exp(-z)
>>> bin_edges = [0.0, 0.5, 1.0, 1.5]
>>> bins = build_photoz_bins(
...     z, nz, bin_edges,
...     scatter_scale=0.05,
...     mean_offset=0.01,
... )
>>> sorted(bins)
[0, 1, 2]
>>> bins[0].shape
(501,)

Notes
-----
- All outputs are evaluated on the same true-z grid ``z``.
- If ``normalize_bins=True`` (default), each returned bin distribution integrates
  to 1 on ``z``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import erf

from binny.utils.broadcasting import as_per_bin
from binny.utils.normalization import normalize_1d
from binny.utils.validators import validate_axis_and_weights, validate_n_bins

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "build_photoz_bins",
    "true_redshift_distribution",
]


def build_photoz_bins(
    z: ArrayLike,
    nz: ArrayLike,
    bin_edges: ArrayLike,
    scatter_scale: Sequence[float] | float,
    mean_offset: Sequence[float] | float,
    *,
    mean_scale: Sequence[float] | float = 1.0,
    outlier_frac: Sequence[float] | float = 0.0,
    outlier_scatter_scale: Sequence[float] | float | None = None,
    outlier_mean_offset: Sequence[float] | float = 0.0,
    outlier_mean_scale: Sequence[float] | float = 1.0,
    normalize_input: bool = True,
    normalize_bins: bool = True,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
) -> dict[int, FloatArray]:
    """Builds photo-z-smeared redshift distributions per tomographic bin.

    This function constructs one true-redshift distribution per observed-redshift
    (photo-z) bin defined by ``bin_edges``. It combines a parent true-redshift
    distribution ``n(z)`` with a Gaussian photo-z selection model in observed
    redshift.

    Parameters that can vary by photo-z bin (e.g. scatter, mean mapping, outlier
    properties) may be provided either as a scalar (applied to all bins) or as a
    sequence of length ``n_bins`` (one value per bin). In all cases, ``n_bins`` is
    inferred from ``bin_edges`` as ``len(bin_edges) - 1``.

    For a galaxy at true redshift ``z``, the probability of being assigned to a
    photo-z bin ``[z_ph_min, z_ph_max]`` is::

        P(bin | z) = (1 - outlier_frac) * P_ph(bin | z)
            + outlier_frac * P_out(bin | z)

    where the core and outlier terms are Gaussian integrals over the photo-z bin
    edges::

        P_core(bin | z) =
            integral_{z_ph_min}^{z_ph_max}
            N(z_ph | mu_ph(z), sigma_ph(z)) dz_ph,
        <BLANKLINE>
        P_out(bin | z) =
            integral_{z_ph_min}^{z_ph_max}
            N(z_ph | mu_out(z), sigma_out(z)) dz_ph.

    The core photo-z model is a Gaussian conditional distribution for the observed
    redshift ``z_ph`` at fixed true redshift ``z``::

        z_ph approx N(mu_ph(z), sigma_ph(z))

    with::

        mu_ph(z) = mean_scale * z - mean_offset

    and::

        sigma_ph(z) = scatter_scale * (1 + z) * mean_scale

    (Equivalently: write the Gaussian in terms of ``(z - c z_ph - z0)`` with
    ``c = 1 / mean_scale`` and ``z0 = mean_offset / mean_scale``. Integrating over
    ``z_ph`` introduces a Jacobian factor ``1 / c`` in the bin probability.)

    The outlier component is an optional second Gaussian with its own mean and
    scatter::

        z_ph approx N(mu_out(z), sigma_out(z))

    where::

        mu_out(z) = outlier_mean_scale * z - outlier_mean_offset

    and::

        sigma_out(z) = outlier_scatter_scale * (1 + z) * outlier_mean_scale

    when enabled (and it contributes with mixture weight ``outlier_frac``).

    The final true-redshift distribution for a given photo-z bin is::

        n_bin(z) = n(z) * P(bin | z)

    Args:
        z: One-dimensional grid of true-redshift values where inputs are defined and
            outputs are evaluated.
        nz: Parent (intrinsic) true-redshift distribution evaluated on ``z``.
            If ``normalize_input=True``, this input is renormalized to integrate to 1.
        bin_edges: One-dimensional array of tomographic bin edges in observed
            redshift (photo-z) space. Adjacent entries define a bin
            ``[bin_edges[i], bin_edges[i+1]]``. Must have length ``n_bins + 1``.
        scatter_scale: Random photo-z scatter (the "noise" or uncertainty). This
            controls the standard deviation of the core photo-z error distribution
            via ``sigma_ph(z) = scatter_scale * (1 + z) * mean_scale``.
            Larger values correspond to broader photo-z errors and increased leakage
            between tomographic bins. May be a scalar (applied to all bins) or
            a sequence of length ``n_bins``.
        mean_offset: Systematic photo-z bias in the mean relation (an additive
            shift). This shifts the center of the core photo-z Gaussian according to::

                mu(z) = mean_scale * z - mean_offset

            with ``mean_scale = 1``.
            A positive ``mean_offset`` shifts the mean observed redshift downward by
            that amount. May be a scalar (applied to all bins) or a sequence of length
            ``n_bins``.
        mean_scale: Systematic calibration error in the mean relation (a
            multiplicative slope change). This stretches or compresses the mapping
            between true redshift and the mean observed redshift in
            ``mu(z) = mean_scale * z - mean_offset``. Values greater than 1 stretch
            the mapping, while values less than 1 compress it. May be a scalar
            (applied to all bins) or a sequence of length ``n_bins``.
        outlier_frac: Fraction of objects treated as "outliers" in the photo-z error
            model. Must lie in the interval ``[0, 1]`` and represents the mixture
            weight in a two-component model::

                P(bin | z) = (1 - outlier_frac) * P_core(bin | z)
                    + outlier_frac * P_out(bin | z)

            It controls how many galaxies have catastrophic photo-z errors. May be
            a scalar (applied to all bins) or a sequence of length ``n_bins``.
        outlier_scatter_scale: Random scatter (standard deviation) of the outlier
            component. This is the "error on the observed photo-z" for the outlier
            population, i.e. the width of the outlier Gaussian via::

                sigma_out(z) = outlier_scatter_scale * (1 + z) * outlier_mean_scale

            Set to ``None`` to disable the outlier component entirely
            (even if ``outlier_frac > 0``).
            May be a scalar or a sequence of length ``n_bins``.
        outlier_mean_offset: Systematic bias in the mean relation for the outlier
            component (an additive shift). This shifts the center of the outlier
            Gaussian mean mapping. Conceptually, this is an error on the mean for
            catastrophic outliers. May be a scalar (applied to all bins) or a
            sequence of length ``n_bins``.
        outlier_mean_scale: Systematic calibration (slope) error in the mean
            relation for the outlier component (a multiplicative scale). This stretches
            or compresses the mapping from true redshift to observed redshift, i.e.
            another form of error on the mean. May be a scalar (applied to all
            bins) or a sequence of length ``n_bins``.
        normalize_input: Whether to normalize the input ``nz`` before binning.
        normalize_bins: Whether to normalize each output bin distribution to
            integrate to 1 on ``z``.
        norm_method: Normalization method passed to :func:`normalize_1d`
            (e.g. ``"trapezoid"``).

    Returns:
        A mapping from photo-z bin index to the corresponding photo-z-smeared true
        redshift distribution evaluated on ``z``.

    Raises:
        ValueError: If ``bin_edges`` does not define a valid number of bins.
        ValueError: If any scalar-or-sequence parameter cannot be broadcast to length
            ``n_bins`` (e.g. wrong-length sequence).
    """
    z_arr, n_arr = validate_axis_and_weights(z, nz)

    bin_edges_arr = np.asarray(bin_edges, dtype=float)

    if bin_edges_arr.ndim != 1:
        raise ValueError("bin_edges must be 1D.")
    if bin_edges_arr.size < 2:
        raise ValueError("bin_edges must have at least two entries.")
    if not np.all(np.isfinite(bin_edges_arr)):
        raise ValueError("bin_edges must contain only finite values.")
    if not np.all(np.diff(bin_edges_arr) > 0):
        raise ValueError("bin_edges must be strictly increasing.")

    n_bins = bin_edges_arr.size - 1
    validate_n_bins(n_bins)

    if normalize_input:
        total = np.trapezoid(n_arr, x=z_arr)
        if not np.isclose(total, 1.0, rtol=1e-3, atol=1e-4):
            n_arr = normalize_1d(z_arr, n_arr, method=norm_method)

    scatter_scale_arr = as_per_bin(scatter_scale, n_bins, "scatter_scale")
    mean_offset_arr = as_per_bin(mean_offset, n_bins, "mean_offset")
    mean_scale_arr = as_per_bin(mean_scale, n_bins, "mean_scale")
    outlier_frac_arr = as_per_bin(outlier_frac, n_bins, "outlier_frac")

    if outlier_scatter_scale is None:
        outlier_scatter_arr = np.array([None] * n_bins, dtype=object)
    else:
        outlier_scatter_arr = as_per_bin(
            outlier_scatter_scale, n_bins, "outlier_scatter_scale"
        )

    outlier_mean_offset_arr = as_per_bin(
        outlier_mean_offset, n_bins, "outlier_mean_offset"
    )
    outlier_mean_scale_arr = as_per_bin(
        outlier_mean_scale, n_bins, "outlier_mean_scale"
    )

    bins: dict[int, FloatArray] = {}

    for i, (z_min, z_max) in enumerate(
        zip(bin_edges_arr[:-1], bin_edges_arr[1:], strict=False)
    ):
        out_scatter = (
            None if outlier_scatter_arr[i] is None else float(outlier_scatter_arr[i])
        )

        nz_bin = true_redshift_distribution(
            z_arr,
            n_arr,
            bin_min=float(z_min),
            bin_max=float(z_max),
            scatter_scale=float(scatter_scale_arr[i]),
            mean_offset=float(mean_offset_arr[i]),
            mean_scale=float(mean_scale_arr[i]),
            outlier_frac=float(outlier_frac_arr[i]),
            outlier_scatter_scale=out_scatter,
            outlier_mean_offset=float(outlier_mean_offset_arr[i]),
            outlier_mean_scale=float(outlier_mean_scale_arr[i]),
        )

        if normalize_bins:
            nz_bin = normalize_1d(z_arr, nz_bin, method=norm_method)

        bins[i] = nz_bin

    return bins


def true_redshift_distribution(
    z: FloatArray,
    nz: FloatArray,
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
    """Computes the true-redshift distribution for a single photo-z bin.

    The output distribution is computed as ``n(z) * P(bin | z)``, where
    ``P(bin | z)`` is the probability that an object at true redshift ``z`` is
    assigned to the observed photo-z bin ``[bin_min, bin_max]``.

    Photo-z model
    -------------
    This function implements a two-component (core + outlier) Gaussian mixture
    model for the conditional observed redshift ``z_ph`` at fixed true redshift
    ``z``.

    Core component (always present)::

        z_ph ~ Normal(mu_core(z), sigma_core(z))

        mu_core(z)    = mean_scale * z - mean_offset
        sigma_core(z) = scatter_scale * (1 + z)

    Outlier component (optional; enabled only if ``outlier_frac > 0`` and
    ``outlier_scatter_scale is not None``)::

        z_ph ~ Normal(mu_out(z), sigma_out(z))

        mu_out(z)    = outlier_mean_scale * z - outlier_mean_offset
        sigma_out(z) = outlier_scatter_scale * (1 + z)

    For a photo-z bin ``[bin_min, bin_max]``, the bin-assignment probability is::

        P(bin | z) = (1 - outlier_frac) * P_core(bin | z)
                     + outlier_frac * P_out(bin | z),

    where ``P_core`` and ``P_out`` are the analytic Gaussian integrals over
    ``z_ph`` in ``[bin_min, bin_max]`` computed using the error function.

    Args:
        z: One-dimensional grid of true-redshift values.
        nz: Parent (intrinsic) true-redshift distribution evaluated on ``z``.
            This function does not normalize ``nz``.
        bin_min: Lower edge of the photo-z bin in observed-redshift space.
        bin_max: Upper edge of the photo-z bin in observed-redshift space.
        scatter_scale: Core photo-z scatter amplitude. The observed-redshift scatter
            is ``sigma_core(z) = scatter_scale * (1 + z)``.
        mean_offset: Additive mean offset in the core mean mapping
            ``mu_core(z) = mean_scale * z - mean_offset``.
        mean_scale: Multiplicative slope in the core mean mapping. Defaults to 1.
        outlier_frac: Outlier mixture fraction in ``[0, 1]``. If zero (default),
            the outlier component is ignored.
        outlier_scatter_scale: Outlier scatter amplitude. If ``None`` (default),
            the outlier component is disabled even if ``outlier_frac > 0``.
            When enabled, ``sigma_out(z) = outlier_scatter_scale * (1 + z)``.
        outlier_mean_offset: Additive mean offset for the outlier mean mapping
            ``mu_out(z) = outlier_mean_scale * z - outlier_mean_offset``.
        outlier_mean_scale: Multiplicative slope for the outlier mean mapping
            ``mu_out(z) = outlier_mean_scale * z - outlier_mean_offset``.

    Returns:
        The photo-z-selected true-redshift distribution
        ``n_bin(z) = n(z) * P(bin | z)`` evaluated on the input grid ``z``.

    Raises:
        ValueError: If ``outlier_frac`` is not in the interval ``[0, 1]``.
        ValueError: If any scale or scatter parameter is non-positive.

    Notes:
        The returned array represents a *shape* in true redshift. If absolute
        normalizations are required (e.g. number densities), they should be
        applied at a higher level.
    """
    z_arr = np.asarray(z, dtype=float)
    n_arr = np.asarray(nz, dtype=float)

    # --- validate mixture weight
    if not (0.0 <= outlier_frac <= 1.0):
        raise ValueError("outlier_frac must lie in [0, 1].")

    # --- validate core params (always)
    if mean_scale <= 0.0:
        raise ValueError("mean_scale must be > 0.")
    if scatter_scale <= 0.0:
        raise ValueError("scatter_scale must be > 0.")

    # --- validate outlier params ONLY when the outlier component is active
    outliers_enabled = (outlier_frac > 0.0) and (outlier_scatter_scale is not None)
    if outliers_enabled:
        if outlier_mean_scale <= 0.0:
            raise ValueError("outlier_mean_scale must be > 0.")
        if outlier_scatter_scale <= 0.0:
            raise ValueError("outlier_scatter_scale must be > 0.")

    # --- core probability
    p_core = _bin_prob_gaussian_photoz(
        z_arr,
        bin_min=bin_min,
        bin_max=bin_max,
        scatter_scale=scatter_scale,
        mean_offset=mean_offset,
        mean_scale=mean_scale,
    )

    # --- optional outliers
    if outliers_enabled:
        p_out = _bin_prob_gaussian_photoz(
            z_arr,
            bin_min=bin_min,
            bin_max=bin_max,
            scatter_scale=outlier_scatter_scale,  # not None here
            mean_offset=outlier_mean_offset,
            mean_scale=outlier_mean_scale,
        )
        p_bin = (1.0 - outlier_frac) * p_core + outlier_frac * p_out
    else:
        p_bin = p_core

    return n_arr * p_bin


def _bin_prob_gaussian_photoz(
    z: FloatArray,
    *,
    bin_min: float,
    bin_max: float,
    scatter_scale: float,
    mean_offset: float,
    mean_scale: float = 1.0,
) -> FloatArray:
    """P(bin | z) for z_ph ~ N(mean_scale*z - mean_offset, scatter_scale*(1+z))."""
    z_arr = np.asarray(z, dtype=float)

    if mean_scale <= 0.0:
        raise ValueError("mean_scale must be > 0.")
    if scatter_scale <= 0.0:
        raise ValueError("scatter_scale must be > 0.")

    mu = mean_scale * z_arr - mean_offset
    sigma = np.maximum(scatter_scale * (1.0 + z_arr) * mean_scale, 1e-10)

    sqrt2 = np.sqrt(2.0)
    t_max = (bin_max - mu) / (sqrt2 * sigma)
    t_min = (bin_min - mu) / (sqrt2 * sigma)

    return 0.5 * (erf(t_max) - erf(t_min))
