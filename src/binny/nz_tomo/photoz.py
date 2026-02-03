r"""Functions to build photometric-redshift tomographic bins.

This module builds true-redshift distributions selected by observed-redshift
(photo-z) tomographic bins. The primary entry point is ``build_photoz_bins``,
which returns a dict mapping bin index -> photo-z-selected ``n_bin(z)`` evaluated
on a common true-z grid.

Model
-----
For each observed-redshift bin ``[z_ph_min, z_ph_max]``, we compute::

    n_bin(z) = n(z) * P(bin | z),

where ``P(bin | z)`` is the probability that an object at true redshift ``z`` is
assigned to that photo-z bin. The core photo-z model is Gaussian::

    z_ph ~ Normal(mu(z), sigma(z)),
    mu(z)    = mean_scale * z - mean_offset,
    sigma(z) = scatter_scale * (1 + z).

Optionally, a second Gaussian outlier component may be included with mixture
weight ``outlier_frac`` (enabled only when ``outlier_scatter_scale`` is not None).

All bin-assignment probabilities are computed analytically by integrating the
Gaussian(s) between the photo-z bin edges using the error function.

Notes
-----
- All outputs are evaluated on the input true-z grid ``z``.
- If ``normalize_bins=True`` (default), each returned bin distribution is
  normalized to integrate to 1 on ``z``.
- ``bin_edges`` live in observed-redshift (photo-z) space.

Metadata / population fractions
-------------------------------
If metadata is requested (``include_metadata=True`` or ``save_metadata_path`` is set),
the builder records how much of the *parent* distribution falls into each observed
(photo-z) bin *before* any per-bin normalization is applied.

Specifically, it stores:

- ``parent_norm``: the total area under the parent curve ``nz`` on the provided grid
- ``bins_norms[i]``: the area under the raw (pre-normalization) bin curve for bin ``i``
- ``frac_per_bin[i]``: ``bins_norms[i] / parent_norm`` when ``parent_norm > 0``

If ``normalize_bins=True``, the returned bin curves are each normalized to integrate
to 1 on ``z`` and should be treated as *shape-only* PDFs. Use ``frac_per_bin`` (or
survey-level inputs) for population statistics such as relative bin weights, number
densities, or counts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf

from binny.nz_tomo.binning_core import (
    build_bins_on_edges,
    finalize_tomo_metadata,
    resolve_bin_edges,
    validate_bin_edges,
)
from binny.utils.broadcasting import as_per_bin
from binny.utils.normalization import normalize_1d
from binny.utils.validators import validate_axis_and_weights

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
    scatter_scale: Sequence[float] | float = 0.0,
    mean_offset: Sequence[float] | float = 0.0,
    binning_scheme: BinningScheme | None = None,
    n_bins: int | None = None,
    bin_range: tuple[float, float] | None = None,
    mean_scale: Sequence[float] | float = 1.0,
    outlier_frac: Sequence[float] | float = 0.0,
    outlier_scatter_scale: Sequence[float] | float | None = None,
    outlier_mean_offset: Sequence[float] | float = 0.0,
    outlier_mean_scale: Sequence[float] | float = 1.0,
    z_ph: FloatArray | None = None,
    nz_ph: FloatArray | None = None,
    normalize_input: bool = True,
    normalize_bins: bool = True,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    include_metadata: bool = False,
    save_metadata_path: str | None = None,
) -> dict[int, FloatArray] | tuple[dict[int, FloatArray], dict[str, Any]]:
    """Builds photo-z-selected true-redshift distributions per tomographic bin."""
    z_arr, parent_arr0 = validate_axis_and_weights(z, nz)
    need_meta = include_metadata or (save_metadata_path is not None)

    parent_arr = parent_arr0
    if normalize_input:
        parent_arr = normalize_1d(z_arr, parent_arr, method=norm_method)

    # --- optional restriction: define bin edges only within a sub-range
    # Important: do NOT snap to the grid; pass the numeric range through so
    # edges match survey specs exactly (e.g. 0.2..1.2).
    edge_axis = z_ph if z_ph is not None else z_arr
    edge_weights = nz_ph if (z_ph is not None and nz_ph is not None) else parent_arr

    edge_axis = np.asarray(edge_axis, dtype=float)
    edge_weights = np.asarray(edge_weights, dtype=float)

    # 1) Resolve + validate photo-z bin edges (edges live in *photo-z* space)
    bin_edges_arr = resolve_bin_edges(
        z_axis=edge_axis,
        nz_axis=edge_weights,
        bin_edges=bin_edges,
        binning_scheme=binning_scheme,
        n_bins=n_bins,
        bin_range=bin_range,
        # equal-number in photo-z space if (z_ph, nz_ph) provided, else fallback to (z, nz)
        equal_number_axis=z_ph,
        equal_number_weights=nz_ph,
        # mixed segments equal-number should also use (z_ph, nz_ph) if provided
        z_ph=z_ph,
        nz_ph=nz_ph,
        norm_method=norm_method,
        normalize_equal_number_weights=True,
    )

    bin_edges_arr = validate_bin_edges(bin_edges_arr, require_within=None)

    n_bins_eff = int(bin_edges_arr.size - 1)

    # 2) Broadcast per-bin model params (photoz-specific)
    scatter_scale_arr = as_per_bin(scatter_scale, n_bins_eff, "scatter_scale")
    mean_offset_arr = as_per_bin(mean_offset, n_bins_eff, "mean_offset")
    mean_scale_arr = as_per_bin(mean_scale, n_bins_eff, "mean_scale")
    outlier_frac_arr = as_per_bin(outlier_frac, n_bins_eff, "outlier_frac")

    if outlier_scatter_scale is None:
        outlier_scatter_arr = np.array([None] * n_bins_eff, dtype=object)
    else:
        outlier_scatter_arr = as_per_bin(outlier_scatter_scale, n_bins_eff, "outlier_scatter_scale")

    outlier_mean_offset_arr = as_per_bin(outlier_mean_offset, n_bins_eff, "outlier_mean_offset")
    outlier_mean_scale_arr = as_per_bin(outlier_mean_scale, n_bins_eff, "outlier_mean_scale")

    # 3) Raw-bin callback (edge in photo-z space, but output is always on true-z grid z_arr)
    def raw_bin_for_edge(i: int, zmin: float, zmax: float) -> FloatArray:
        out_scatter = None if outlier_scatter_arr[i] is None else float(outlier_scatter_arr[i])

        return true_redshift_distribution(
            z_arr,
            parent_arr,
            bin_min=float(zmin),
            bin_max=float(zmax),
            scatter_scale=float(scatter_scale_arr[i]),
            mean_offset=float(mean_offset_arr[i]),
            mean_scale=float(mean_scale_arr[i]),
            outlier_frac=float(outlier_frac_arr[i]),
            outlier_scatter_scale=out_scatter,
            outlier_mean_offset=float(outlier_mean_offset_arr[i]),
            outlier_mean_scale=float(outlier_mean_scale_arr[i]),
        ).astype(np.float64, copy=False)

    # 4) Build bins + norms (agnostic)
    bins, bins_norms, parent_norm = build_bins_on_edges(
        z=z_arr,
        nz_parent_for_meta=parent_arr0,
        bin_edges=bin_edges_arr,
        raw_bin_for_edge=raw_bin_for_edge,
        normalize_bins=normalize_bins,
        norm_method=norm_method,
        mixer=None,
        need_meta=need_meta,
    )

    # 5) Metadata (agnostic)
    meta = finalize_tomo_metadata(
        kind="photoz",
        z=z_arr,
        parent_nz=parent_arr0,
        bin_edges=bin_edges_arr,
        bins=bins,
        inputs={
            "bin_edges_provided": bin_edges is not None,
            "binning_scheme": binning_scheme,
            "n_bins": n_bins,
            "bin_range": bin_range,
            "normalize_bins": normalize_bins,
            "norm_method": norm_method,
            "scatter_scale": scatter_scale,
            "mean_offset": mean_offset,
            "mean_scale": mean_scale,
            "outlier_frac": outlier_frac,
            "outlier_scatter_scale": outlier_scatter_scale,
            "outlier_mean_offset": outlier_mean_offset,
            "outlier_mean_scale": outlier_mean_scale,
            "z_ph_provided": z_ph is not None,
            "nz_ph_provided": nz_ph is not None,
            "bin_edges_space": "photoz",
        },
        parent_norm=parent_norm,
        bins_norms=bins_norms,
        include_metadata=include_metadata,
        save_metadata_path=save_metadata_path,
    )

    return (bins, meta) if include_metadata else bins


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
    outlier_frac when outlier_frac > 0 (requires outlier_scatter_scale, which
    may be 0.0 for deterministic).

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

    if outlier_frac > 0.0 and outlier_scatter_scale is None:
        raise ValueError(
            "outlier_scatter_scale must be set when outlier_frac > 0. "
            "Use 0.0 for a deterministic (no-uncertainty) outlier component."
        )

    # --- validate core params (always)
    if mean_scale <= 0.0:
        raise ValueError("mean_scale must be > 0.")
    if scatter_scale < 0.0:
        raise ValueError("scatter_scale must be >= 0.")

    # --- validate outlier params ONLY when the outlier component is active
    outliers_enabled = outlier_frac > 0.0
    if outliers_enabled:
        if outlier_mean_scale <= 0.0:
            raise ValueError("outlier_mean_scale must be > 0.")
        if outlier_scatter_scale < 0.0:
            raise ValueError("outlier_scatter_scale must be >= 0.")

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

    if scatter_scale < 0.0:
        raise ValueError("scatter_scale must be >= 0.")

    # Reference convention:
    mu = mean_scale * z_arr - mean_offset

    # no-uncertainty limit: deterministic assignment in photo-z space
    if scatter_scale == 0.0:
        return ((mu >= bin_min) & (mu < bin_max)).astype(np.float64)

    sigma = np.maximum(scatter_scale * (1.0 + z_arr), 1e-10)

    sqrt2 = np.sqrt(2.0)
    t_max = (bin_max - mu) / (sqrt2 * sigma)
    t_min = (bin_min - mu) / (sqrt2 * sigma)
    return 0.5 * (erf(t_max) - erf(t_min))
