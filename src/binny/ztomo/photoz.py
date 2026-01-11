"""Functions to build photometric-redshift tomographic bins.

This module builds true-redshift distributions selected by observed-redshift
(photo-z) tomographic bins. The primary entry point is :func:`build_photoz_bins`,
which returns a dict mapping bin index -> photo-z-selected ``n_bin(z)`` evaluated
on a common true-z grid.

Model
-----
For each observed-redshift bin ``[z_ph_min, z_ph_max]``, we compute

    n_bin(z) = n(z) * P(bin | z),

where ``P(bin | z)`` is the probability that an object at true redshift ``z``
is assigned to that photo-z bin. The core photo-z model is Gaussian:

    z_ph ~ Normal(mu(z), sigma(z)),
    mu(z)    = mean_scale * z - mean_offset,
    sigma(z) = scatter_scale * (1 + z) * mean_scale.

Optionally, a second Gaussian outlier component may be included with mixture
weight ``outlier_frac`` (enabled only when ``outlier_scatter_scale`` is not None).

All bin-assignment probabilities are computed analytically by integrating the
Gaussian(s) between the photo-z bin edges using the error function.

Examples
--------
>>> import numpy as np
>>> from binny.ztomo.photoz import build_photoz_bins
>>> z = np.linspace(0.0, 3.0, 501)
>>> nz = z**2 * np.exp(-z)
>>> bin_edges = [0.0, 0.5, 1.0, 1.5]
>>> bins = build_photoz_bins(z, nz, bin_edges, scatter_scale=0.05, mean_offset=0.01)
>>> sorted(bins)
[0, 1, 2]
>>> bins[0].shape
(501,)

Notes
-----
- All outputs are evaluated on the input true-z grid ``z``.
- If ``normalize_bins=True`` (default), each returned bin distribution is
  normalized to integrate to 1 on ``z``.
- ``bin_edges`` live in observed-redshift (photo-z) space.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from binny.axes.bin_edges import equal_number_edges, equidistant_edges
from binny.utils.broadcasting import as_per_bin
from binny.utils.normalization import normalize_1d
from binny.utils.validators import (
    validate_axis_and_weights,
    validate_n_bins,
)
from binny.ztomo.ztomo_utils import mixed_edges

FloatArray: TypeAlias = NDArray[np.float64]
BinningScheme: TypeAlias = str | Sequence[Mapping[str, Any]] | Mapping[str, Any]

__all__ = [
    "build_photoz_bins",
    "true_redshift_distribution",
]


def build_photoz_bins(
    z: FloatArray,
    nz: FloatArray,
    bin_edges: FloatArray | None = None,
    *,
    scatter_scale: Sequence[float] | float,
    mean_offset: Sequence[float] | float,
    binning_scheme: BinningScheme | None = None,
    n_bins: int | None = None,
    z_ph: FloatArray | None = None,
    nz_ph: FloatArray | None = None,
    mean_scale: Sequence[float] | float = 1.0,
    outlier_frac: Sequence[float] | float = 0.0,
    outlier_scatter_scale: Sequence[float] | float | None = None,
    outlier_mean_offset: Sequence[float] | float = 0.0,
    outlier_mean_scale: Sequence[float] | float = 1.0,
    normalize_input: bool = True,
    normalize_bins: bool = True,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
) -> dict[int, FloatArray]:
    """Builds photo-z-selected true-redshift distributions per tomographic bin.

    Bins are defined either by explicit ``bin_edges`` (in observed redshift) or by
    ``(binning_scheme, n_bins)`` to construct edges. Per-bin photo-z parameters
    (scatter/mean/outlier terms) may be scalars or sequences of length equal to
    ``n_bins`` (single-scheme mode) or the total number of bins implied by the
    resolved bin edges (mixed mode).

    Args:
        z: True-redshift grid where outputs are evaluated.
        nz: Parent true-redshift distribution evaluated on ``z``.
        bin_edges: Observed-redshift bin edges (length ``n_bins + 1``). Mutually
            exclusive with ``binning_scheme`` / ``n_bins``.
        scatter_scale: Core photo-z scatter amplitude (scalar or per-bin sequence).
        mean_offset: Core mean offset (scalar or per-bin sequence).
        binning_scheme: If ``bin_edges`` is None, scheme to build edges
            (e.g. ``"equidistant"``, ``"equal_number"``), or a mixed binning
            specification (sequence of segment dicts, or a dict with key
            ``"segments"`` containing that sequence).
        n_bins: Number of bins when using a string ``binning_scheme``.
        z_ph: Observed redshift grid for equal-number edges.
        nz_ph: Observed distribution for equal-number edges.
        mean_scale: Core mean scale (scalar or per-bin sequence).
        outlier_frac: Outlier mixture fraction in [0, 1] (scalar or per-bin sequence).
        outlier_scatter_scale: Outlier scatter amplitude; set None to disable outliers.
        outlier_mean_offset: Outlier mean offset.
        outlier_mean_scale: Outlier mean scale.
        normalize_input: If True, normalize ``nz`` on ``z`` before binning.
        normalize_bins: If True, normalize each output bin on ``z``.
        norm_method: Normalization method passed to :func:`normalize_1d`.

    Returns:
        Dict mapping bin index -> ``n_bin(z)`` array evaluated on ``z``.

    Raises:
        ValueError: If bin specification is inconsistent, or if ``z_ph``/``nz_ph``
            are provided inconsistently.
    """
    z_arr, n_arr = validate_axis_and_weights(z, nz)

    # resolve bin edges (explicit or scheme-based)
    if bin_edges is not None:
        if (binning_scheme is not None) or (n_bins is not None):
            raise ValueError(
                "Provide either bin_edges or (binning_scheme, n_bins), not both."
            )
        bin_edges_arr = np.asarray(bin_edges, dtype=float)

    else:
        if binning_scheme is None:
            raise ValueError("bin_edges is None. You must provide binning_scheme.")

        # Case 1: single scheme (string)
        if isinstance(binning_scheme, str):
            if n_bins is None:
                raise ValueError(
                    "bin_edges is None and binning_scheme is a string. "
                    "You must provide n_bins."
                )

            validate_n_bins(n_bins)

            scheme = binning_scheme.lower()
            if scheme in {"equidistant", "eq", "linear"}:
                x_min = float(z_arr[0])
                x_max = float(z_arr[-1])
                bin_edges_arr = equidistant_edges(x_min, x_max, n_bins)

            elif scheme in {"equal_number", "equipopulated", "en"}:
                # Prefer explicit observed-z distribution if provided;
                # otherwise fall back to using (z, nz) as a proxy for (z_ph,
                # nz_ph).
                if (z_ph is None) ^ (nz_ph is None):
                    raise ValueError(
                        "Provide both z_ph and nz_ph, or neither. "
                        "If neither is provided, (z, nz) is used as a proxy for"
                        " (z_ph, nz_ph)."
                    )

                if z_ph is None and nz_ph is None:
                    bin_edges_arr = equal_number_edges_proxy(
                        z_arr,
                        n_arr,
                        n_bins,
                        normalize_input=normalize_input,
                        norm_method=norm_method,
                    )
                else:
                    zph_arr, nzph_arr = validate_axis_and_weights(z_ph, nz_ph)
                    bin_edges_arr = equal_number_edges(zph_arr, nzph_arr, n_bins)

            else:
                raise ValueError(
                    "Unsupported binning_scheme. Supported: "
                    "'equidistant' (eq/linear) and 'equal_number'"
                    " (equipopulated/en)."
                )

        # Case 2: mixed segments (sequence/dict)
        else:
            if n_bins is not None:
                raise ValueError(
                    "In mixed binning mode, set n_bins per segment and leave"
                    " n_bins=None."
                )

            if isinstance(binning_scheme, Mapping):
                if "segments" in binning_scheme:
                    segments = binning_scheme["segments"]
                else:
                    raise ValueError(
                        "Mixed binning dict must contain key 'segments' "
                        "(e.g. {'segments': [ ... ]})."
                    )
            else:
                segments = binning_scheme

            if not isinstance(segments, Sequence) or isinstance(segments, str | bytes):
                raise ValueError("Mixed binning requires a sequence of segment dicts.")

            bin_edges_arr = mixed_edges(
                segments,
                z_axis=z_arr,
                nz_axis=n_arr,
                z_ph=z_ph,
                nz_ph=nz_ph,
                normalize_input=normalize_input,
                norm_method=norm_method,
            )

    # validate edges
    if bin_edges_arr.ndim != 1:
        raise ValueError("bin_edges must be 1D.")
    if bin_edges_arr.size < 2:
        raise ValueError("bin_edges must have at least two entries.")
    if not np.all(np.isfinite(bin_edges_arr)):
        raise ValueError("bin_edges must contain only finite values.")
    if not np.all(np.diff(bin_edges_arr) > 0):
        raise ValueError("bin_edges must be strictly increasing.")

    n_bins_eff = bin_edges_arr.size - 1
    validate_n_bins(n_bins_eff)

    # normalize parent nz if requested
    if normalize_input:
        total = np.trapezoid(n_arr, x=z_arr)
        if not np.isclose(total, 1.0, rtol=1e-3, atol=1e-4):
            n_arr = normalize_1d(z_arr, n_arr, method=norm_method)

    # broadcast per-bin params
    scatter_scale_arr = as_per_bin(scatter_scale, n_bins_eff, "scatter_scale")
    mean_offset_arr = as_per_bin(mean_offset, n_bins_eff, "mean_offset")
    mean_scale_arr = as_per_bin(mean_scale, n_bins_eff, "mean_scale")
    outlier_frac_arr = as_per_bin(outlier_frac, n_bins_eff, "outlier_frac")

    if outlier_scatter_scale is None:
        outlier_scatter_arr = np.array([None] * n_bins_eff, dtype=object)
    else:
        outlier_scatter_arr = as_per_bin(
            outlier_scatter_scale, n_bins_eff, "outlier_scatter_scale"
        )

    outlier_mean_offset_arr = as_per_bin(
        outlier_mean_offset, n_bins_eff, "outlier_mean_offset"
    )
    outlier_mean_scale_arr = as_per_bin(
        outlier_mean_scale, n_bins_eff, "outlier_mean_scale"
    )

    # build bins
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
    """Computes ``n_bin(z) = n(z) * P(bin | z)`` for one photo-z bin.

    ``P(bin | z)`` is computed from a Gaussian core photo-z model, optionally
    including a second Gaussian outlier component with mixture fraction
    ``outlier_frac`` when ``outlier_scatter_scale`` is not None.

    Returns:
        Photo-z-selected true-redshift distribution evaluated on ``z``.

    Raises:
        ValueError: If mixture fraction is outside [0, 1] or any active scale
            parameter is non-positive.
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


def equal_number_edges_proxy(
    z: FloatArray,
    nz: FloatArray,
    n_bins: int,
    *,
    normalize_input: bool = True,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
) -> np.ndarray:
    """Computes equal-number bin edges using (z, nz) as a fallback proxy for
     (z_ph, nz_ph).

    This is a convenience helper. If you do not have an observed-redshift
    distribution n(z_ph), it approximates it with the provided true-redshift
    distribution n(z).

    Args:
        z: Grid values (used as the proxy observed-redshift axis).
        nz: Weights on ``z`` (used as the proxy observed-redshift distribution).
        n_bins: Number of equal-number bins.
        normalize_input: If True, normalize ``nz`` before computing quantiles.
        norm_method: Normalization method for :func:`normalize_1d`.

    Returns:
        A strictly increasing array of length ``n_bins + 1``.
    """
    validate_n_bins(n_bins)
    z_arr, n_arr = validate_axis_and_weights(z, nz)

    if normalize_input:
        total = np.trapezoid(n_arr, x=z_arr)
        if not np.isclose(total, 1.0, rtol=1e-3, atol=1e-4):
            n_arr = normalize_1d(z_arr, n_arr, method=norm_method)

    return equal_number_edges(z_arr, n_arr, n_bins)


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
