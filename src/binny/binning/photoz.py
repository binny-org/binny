"""Photometric-redshift tomographic assignment probabilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from scipy.special import erf

from binny.binning.core import build_bins_on_edges
from binny.binning.edges import resolve_bin_edges, validate_bin_edges
from binny.tomography.metadata import finalize_tomo_metadata
from binny.utils.broadcasting import as_per_bin
from binny.utils.normalization import normalize_1d
from binny.utils.types import FloatArray
from binny.utils.validators import validate_axis_and_weights

__all__ = [
    "build_photoz_bins",
    "true_redshift_distribution",
    "photoz_assignment",
    "photoz_assignment_probability",
]


def _broadcast_photoz_params(
    n_bins: int,
    *,
    scatter_scale: Sequence[float] | float,
    mean_offset: Sequence[float] | float,
    mean_scale: Sequence[float] | float,
    outlier_frac: Sequence[float] | float,
    outlier_scatter_scale: Sequence[float] | float | None,
    outlier_mean_offset: Sequence[float] | float,
    outlier_mean_scale: Sequence[float] | float,
) -> list[dict[str, float | None]]:
    """Broadcast photo-z parameters to individual tomographic bins.

    Args:
        n_bins: Number of tomographic bins.
        scatter_scale: Core Gaussian scatter scale.
        mean_offset: Core Gaussian mean offset.
        mean_scale: Core Gaussian mean scale.
        outlier_frac: Outlier mixture fraction.
        outlier_scatter_scale: Optional outlier Gaussian scatter scale.
        outlier_mean_offset: Outlier Gaussian mean offset.
        outlier_mean_scale: Outlier Gaussian mean scale.

    Returns:
        List of parameter dictionaries, one per tomographic bin.
    """
    values = {
        "scatter_scale": scatter_scale,
        "mean_offset": mean_offset,
        "mean_scale": mean_scale,
        "outlier_frac": outlier_frac,
        "outlier_mean_offset": outlier_mean_offset,
        "outlier_mean_scale": outlier_mean_scale,
    }

    arrays = {name: as_per_bin(value, n_bins, name) for name, value in values.items()}

    if outlier_scatter_scale is None:
        outlier_scatter: list[float | None] = [None] * n_bins
    else:
        outlier_scatter = [
            float(value)
            for value in as_per_bin(
                outlier_scatter_scale,
                n_bins,
                "outlier_scatter_scale",
            )
        ]

    return [
        {
            **{name: float(array[i]) for name, array in arrays.items()},
            "outlier_scatter_scale": outlier_scatter[i],
        }
        for i in range(n_bins)
    ]


def _resolve_photoz_edges(
    z: FloatArray,
    nz: FloatArray,
    bins: Mapping[str, Any],
    z_ph: FloatArray | None,
    nz_ph: FloatArray | None,
    norm_method: Literal["trapezoid", "simpson"],
) -> FloatArray:
    """Resolve bin edges in observed photo-z space."""
    edge_axis = z_ph if z_ph is not None else z
    edge_weights = nz_ph if (z_ph is not None and nz_ph is not None) else nz

    edges = resolve_bin_edges(
        z_axis=np.asarray(edge_axis, dtype=float),
        nz_axis=np.asarray(edge_weights, dtype=float),
        bins=bins,
        equal_number_axis=z_ph,
        equal_number_weights=nz_ph,
        z_ph=z_ph,
        nz_ph=nz_ph,
        norm_method=norm_method,
        normalize_equal_number_weights=True,
    )

    return validate_bin_edges(edges, require_within=None)


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

    if mean_scale <= 0.0:
        raise ValueError("mean_scale must be > 0.")
    if scatter_scale < 0.0:
        raise ValueError("scatter_scale must be >= 0.")

    outliers_enabled = outlier_frac > 0.0
    if outliers_enabled:
        if outlier_mean_scale <= 0.0:
            raise ValueError("outlier_mean_scale must be > 0.")
        if outlier_scatter_scale is None:
            raise ValueError(
                "outlier_scatter_scale must be set when outlier_frac > 0. "
                "Use 0.0 for a deterministic (no-uncertainty) outlier component."
            )
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

    params = _broadcast_photoz_params(
        n_bins,
        scatter_scale=scatter_scale,
        mean_offset=mean_offset,
        mean_scale=mean_scale,
        outlier_frac=outlier_frac,
        outlier_scatter_scale=outlier_scatter_scale,
        outlier_mean_offset=outlier_mean_offset,
        outlier_mean_scale=outlier_mean_scale,
    )

    return np.stack(
        [
            photoz_assignment_probability(
                z_arr,
                bin_min=float(edges_arr[i]),
                bin_max=float(edges_arr[i + 1]),
                **params[i],
            )
            for i in range(n_bins)
        ],
        axis=0,
    )


def build_photoz_bins(
    z: FloatArray,
    nz: FloatArray,
    *,
    bins: Mapping[str, Any],
    scatter_scale: Sequence[float] | float = 0.0,
    mean_offset: Sequence[float] | float = 0.0,
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
    include_photoz_diagnostics: bool = False,
    save_metadata_path: str | None = None,
) -> dict[int, FloatArray] | tuple[dict[int, FloatArray], dict[str, Any]]:
    """Builds photo-z-selected true-redshift distributions per tomographic bin.

    Args:
        z: True-redshift grid.
        nz: Parent distribution evaluated on ``z``.
        bins: Dictionary specifying the observed photo-z bin edges or their
            construction. Supports ``{"edges": [...]}``,
            ``{"scheme": name, "n_bins": n, "range": (lo, hi),
            "params": {...}}``, or ``{"segments": [...]}``.
        scatter_scale: Per-bin (or constant scalar) photo-z scatter scale.
        mean_offset: Per-bin (or constant scalar) photo-z mean offset.
        mean_scale: Per-bin (or constant scalar) photo-z mean scale.
        outlier_frac: Per-bin (or constant scalar) outlier mixture fraction.
        outlier_scatter_scale: Per-bin (or scalar) outlier scatter scale. When
            provided, enables an outlier component.
        outlier_mean_offset: Per-bin (or constant scalar) outlier mean offset.
        outlier_mean_scale: Per-bin (or constant scalar) outlier mean scale.
        z_ph: Optional photo-z axis used for edge derivation in population-based
            schemes.
        nz_ph: Optional photo-z weights evaluated on ``z_ph``.
        normalize_input: Whether to normalize ``nz`` before computing bins.
        normalize_bins: Whether to normalize each returned bin curve.
        norm_method: Integration method used for normalization.
        include_metadata: Whether to return metadata alongside the bins.
        include_photoz_diagnostics: Whether to include photo-z bin-assignment
            diagnostics in the returned metadata.
        save_metadata_path: Optional path for writing metadata.

    Returns:
        Mapping from bin index to ``n_bin(z)``. If ``include_metadata=True`` or
        ``include_photoz_diagnostics=True``, returns ``(bins, metadata)``.

    Raises:
        ValueError: If edge inputs are inconsistent, the binning specification
            is invalid, or photo-z model parameters are invalid.
    """

    z_arr, parent_arr = validate_axis_and_weights(z, nz)

    return_metadata = include_metadata or include_photoz_diagnostics
    need_meta = return_metadata or save_metadata_path is not None

    if normalize_input:
        parent_arr = normalize_1d(z_arr, parent_arr, method=norm_method)

    # Resolve edges in observed photo-z space.
    edges = _resolve_photoz_edges(z_arr, parent_arr, bins, z_ph, nz_ph, norm_method)

    # Prepare one set of model parameters per tomographic bin.
    params = _broadcast_photoz_params(
        len(edges) - 1,
        scatter_scale=scatter_scale,
        mean_offset=mean_offset,
        mean_scale=mean_scale,
        outlier_frac=outlier_frac,
        outlier_scatter_scale=outlier_scatter_scale,
        outlier_mean_offset=outlier_mean_offset,
        outlier_mean_scale=outlier_mean_scale,
    )

    assignment_probabilities: dict[int, FloatArray] = {}

    def raw_bin_for_edge(i: int, zmin: float, zmax: float) -> FloatArray:
        p_bin = photoz_assignment_probability(
            z_arr,
            bin_min=float(zmin),
            bin_max=float(zmax),
            **params[i],
        )

        if include_photoz_diagnostics:
            assignment_probabilities[i] = p_bin.astype(np.float64, copy=True)

        return (parent_arr * p_bin).astype(np.float64, copy=False)

    # Construct and optionally normalize the true-redshift distributions.
    tomo_bins, bins_norms, parent_norm = build_bins_on_edges(
        z=z_arr,
        nz_parent_for_meta=parent_arr,
        bin_edges=edges,
        raw_bin_for_edge=raw_bin_for_edge,
        normalize_bins=normalize_bins,
        norm_method=norm_method,
        mixer=None,
        need_meta=need_meta,
    )

    # Preserve the existing metadata schema and saving behavior.
    meta = finalize_tomo_metadata(
        kind="photoz",
        z=z_arr,
        parent_nz=parent_arr,
        bin_edges=edges,
        bins=tomo_bins,
        inputs={
            "bins": dict(bins),
            "normalize_input": normalize_input,
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
        include_metadata=return_metadata,
        save_metadata_path=save_metadata_path,
    )

    if include_photoz_diagnostics and meta is not None:
        meta["photoz_diagnostics"] = {
            "observed_bin_edges": edges.copy(),
            "true_z": z_arr.copy(),
            "assignment_probabilities": assignment_probabilities,
            "description": (
                "P(i | z) for each observed photo-z bin, evaluated on the true-z grid."
            ),
        }

    if return_metadata:
        if meta is None:
            raise RuntimeError("Metadata was requested but not generated.")
        return tomo_bins, meta

    return tomo_bins


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

    The photo-z-selected true-redshift distribution is
    ``n_bin(z) = n(z) * P(bin | z)``,
    where ``n(z)`` is the parent distribution and ``P(bin | z)` is the
    probability that an object at true redshift ``z`` falls into the photo-z bin.
    The probability is modeled with a Gaussian core photo-z relation, with an
    optional Gaussian outlier component controlled by outlier_frac and the
    corresponding outlier parameters.

    Args:
        z: True-redshift grid.
        nz: Parent distribution evaluated on ``z``.
        bin_min: Lower photo-z edge for the bin.
        bin_max: Upper photo-z edge for the bin.
        scatter_scale: Core photo-z scatter scale.
        mean_offset: Core photo-z mean offset.
        mean_scale: Core photo-z mean scale.
        outlier_frac: Outlier mixture fraction.
        outlier_scatter_scale: Outlier scatter scale.
        outlier_mean_offset: Outlier mean offset.
        outlier_mean_scale: Outlier mean scale.

    Returns:
        Photo-z-selected true-redshift distribution evaluated on ``z``.

    Raises:
        ValueError: If ``outlier_frac`` is outside [0, 1], if required outlier
            parameters are missing, or if any active scale parameter is invalid.
    """
    z_arr = np.asarray(z, dtype=float)
    n_arr = np.asarray(nz, dtype=float)

    p_bin = photoz_assignment_probability(
        z_arr,
        bin_min=bin_min,
        bin_max=bin_max,
        scatter_scale=scatter_scale,
        mean_offset=mean_offset,
        mean_scale=mean_scale,
        outlier_frac=outlier_frac,
        outlier_scatter_scale=outlier_scatter_scale,
        outlier_mean_offset=outlier_mean_offset,
        outlier_mean_scale=outlier_mean_scale,
    )

    return n_arr * p_bin
