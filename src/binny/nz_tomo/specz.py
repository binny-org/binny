r"""Build spectroscopic-redshift tomographic bins on a common true-z grid.

This module constructs tomographic-bin redshift distributions for **true-z
(spectroscopic) binning**. The main entry point is :func:`build_specz_bins`,
which returns a mapping from observed-bin index to an observed-bin distribution
``n_obs_i(z)``, all evaluated on the same true-redshift grid ``z``.

Selection model
---------------
True bins are defined by edges ``[z_j, z_{j+1}]`` and a per-bin completeness
factor ``c_j``. The true-bin selection window is

    ``S_j(z) = c_j * 1_{[z_j, z_{j+1})}(z)``,

and the corresponding true-bin distribution is
    ``n_true_j(z) = n(z) * S_j(z)``.

Bin-to-bin response (optional)
------------------------------
Observed bins may differ from true bins due to survey-style response effects.
These are represented by a column-stochastic response matrix ``M`` with

    ``M[i, j] = P(i_obs | j_true)``,
    ``sum_i M[i, j] = 1``  for every true bin ``j``,

which mixes true bins into observed bins via

    ``n_obs_i(z) = sum_j M[i, j] * n_true_j(z)``.

Two response components are supported:

1) Catastrophic reassignment (bin-level)
   A per-bin fraction ``f_j`` is redistributed away from the diagonal according
   to a leakage prescription (uniform, neighbor, or Gaussian in bin-index
   space). If an explicit ``response_matrix`` is provided, it is used directly.

2) Measurement scatter (optional)
   A Gaussian measurement model for ``z_hat | z_true`` is integrated over
   observed bin edges to form a response, then averaged within each true bin to
   obtain a column-stochastic matrix ``M_scatter``.

If both are enabled, the total response is applied as

    ``M_total = M_scatter @ M_cat``.

Normalization
-------------
- If ``normalize_input=True`` (default), the parent distribution ``n(z)`` is
  normalized to integrate to 1 over the grid ``z`` before binning.
- If ``normalize_bins=True`` (default), each returned ``n_obs_i(z)`` is
  normalized to integrate to 1 over ``z`` when the bin has nonzero support.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from binny.binning.assignments import specz_selection_in_bin
from binny.binning.specz import (
    apply_response_matrix,
    build_specz_response_matrix,
    specz_gaussian_response_matrix,
)
from binny.nz_tomo.binning_core import (
    build_bins_on_edges,
    finalize_tomo_metadata,
    resolve_bin_edges,
    validate_bin_edges,
)
from binny.utils.broadcasting import as_per_bin
from binny.utils.normalization import normalize_1d
from binny.utils.types import BinningScheme, FloatArray1D, FloatArray2D
from binny.utils.validators import (
    validate_axis_and_weights,
)

__all__ = [
    "build_specz_bins",
    "specz_selection_in_bin",
    "build_specz_response_matrix",
    "apply_response_matrix",
    "specz_gaussian_response_matrix",
]


def build_specz_bins(
    z: FloatArray1D,
    nz: FloatArray1D,
    bin_edges: FloatArray1D | None = None,
    *,
    binning_scheme: BinningScheme | None = None,
    n_bins: int | None = None,
    bin_range: tuple[float, float] | None = None,
    completeness: Sequence[float] | float = 1.0,
    catastrophic_frac: Sequence[float] | float = 0.0,
    leakage_model: Literal["uniform", "neighbor", "gaussian"] = "neighbor",
    leakage_sigma: Sequence[float] | float = 1.0,
    response_matrix: FloatArray2D | None = None,
    specz_scatter: Sequence[float] | float | None = None,
    specz_scatter_model: Literal["sigma0_plus_sigma1_1pz"] = "sigma0_plus_sigma1_1pz",
    sigma0: float = 0.0,
    sigma1: float = 0.0,
    normalize_input: bool = True,
    normalize_bins: bool = True,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    include_metadata: bool = False,
    save_metadata_path: str | None = None,
) -> dict[int, FloatArray1D] | tuple[dict[int, FloatArray1D], dict[str, Any]]:
    """Build spectroscopic-redshift tomographic bins on a common true-z grid.

    Constructs per-bin true-redshift distributions from a parent ``n(z)`` using
    true-redshift bin edges. Optionally applies survey-style bin-response effects
    that mix true bins into observed bins via a column-stochastic response matrix.

    If response effects are enabled, the returned bins correspond to *observed*
    bins, each expressed as a distribution on the *true-z* grid.

    Args:
        z: True-redshift grid where all outputs are evaluated.
        nz: Parent true-redshift distribution evaluated on ``z``.
        bin_edges: Optional true-redshift bin edges. If provided, ``binning_scheme``
            and ``n_bins`` are ignored.
        binning_scheme: Scheme used to derive edges when ``bin_edges`` is not
            provided. May be a simple scheme name or a mixed-segment specification.
        n_bins: Number of bins for simple scheme names.
        bin_range: Optional interval used when deriving equidistant edges.
        completeness: Per-bin completeness factor(s) applied to the true-bin
            selection (scalar or per-bin sequence).
        catastrophic_frac: Per-bin fraction(s) reassigned to other observed bins
            through the catastrophic response model (scalar or per-bin sequence).
        leakage_model: Redistribution rule for catastrophic reassignment.
        leakage_sigma: Width parameter for Gaussian leakage in bin-index space.
        response_matrix: Optional explicit bin-response matrix overriding the
            catastrophic leakage model.
        specz_scatter: Optional measurement-scatter scale(s) used to build a
            scatter response matrix (scalar or per-bin sequence).
        specz_scatter_model: Scatter parameterization used when ``specz_scatter`` is
            not provided.
        sigma0: Additive scatter component for ``specz_scatter_model``.
        sigma1: Redshift-dependent scatter component for ``specz_scatter_model``.
        normalize_input: Whether to normalize the parent ``nz`` before binning.
        normalize_bins: Whether to normalize each returned bin to integrate to 1
            on ``z`` when the bin has nonzero support.
        norm_method: Integration method used for normalization.
        include_metadata: Whether to return a metadata mapping alongside the bins.
        save_metadata_path: Optional path where metadata is written in text form.

    Returns:
        A dict mapping observed bin index to a true-z distribution ``n_obs_i(z)``.
        If ``include_metadata=True``, returns ``(bins, metadata)``.

    Raises:
        ValueError: If edge inputs are inconsistent, response settings are invalid,
            or any parameter constraints are violated.
    """
    z_arr, parent_arr0 = validate_axis_and_weights(z, nz)
    need_meta = include_metadata or (save_metadata_path is not None)

    # Resolve and validate true-z bin edges
    parent_for_edges = parent_arr0
    if normalize_input:
        parent_for_edges = normalize_1d(z_arr, parent_for_edges, method=norm_method)

    bin_edges_arr = resolve_bin_edges(
        z_axis=z_arr,
        nz_axis=parent_for_edges,
        bin_edges=bin_edges,
        binning_scheme=binning_scheme,
        n_bins=n_bins,
        bin_range=bin_range,  # NEW
        equal_number_axis=None,
        equal_number_weights=None,
        z_ph=None,
        nz_ph=None,
        norm_method=norm_method,
        normalize_equal_number_weights=False,
    )

    bin_edges_arr = validate_bin_edges(
        bin_edges_arr,
        require_within=(float(z_arr[0]), float(z_arr[-1])),
    )
    n_bins_eff = int(bin_edges_arr.size - 1)

    # Scatter knobs: either explicit specz_scatter OR sigma0/sigma1 model,
    # not both
    if specz_scatter is not None and (sigma0 != 0.0 or sigma1 != 0.0):
        raise ValueError("Provide either specz_scatter OR ``sigma0/sigma1``, not both.")

    # Normalize parent for true-bin construction if requested
    parent_arr = parent_arr0
    if normalize_input:
        parent_arr = normalize_1d(z_arr, parent_arr, method=norm_method)

    # Broadcast per-bin params
    completeness_arr = as_per_bin(completeness, n_bins_eff, "completeness")

    # Build response matrices (catastrophic + optional scatter)
    matrix_cat = build_specz_response_matrix(
        n_bins_eff,
        catastrophic_frac=catastrophic_frac,
        leakage_model=leakage_model,
        leakage_sigma=leakage_sigma,
        response_matrix=response_matrix,
    )
    matrix_total = matrix_cat

    scatter_enabled = (specz_scatter is not None) or (
        specz_scatter is None
        and specz_scatter_model == "sigma0_plus_sigma1_1pz"
        and (sigma0 > 0.0 or sigma1 > 0.0)
    )

    if scatter_enabled:
        # Validate explicit per-bin scatter early
        if specz_scatter is not None:
            sig = np.asarray(as_per_bin(specz_scatter, n_bins_eff, "specz_scatter"), dtype=float)
            if np.any(sig < 0.0):
                raise ValueError("specz_scatter must be >= 0.")
            if np.allclose(sig, 0.0):
                scatter_enabled = False

    if scatter_enabled:
        matrix_scatter = specz_gaussian_response_matrix(
            z_arr=z_arr,
            bin_edges=bin_edges_arr,
            specz_scatter=specz_scatter,
            model=specz_scatter_model,
            sigma0=sigma0,
            sigma1=sigma1,
        )
        matrix_total = matrix_scatter @ matrix_cat
    else:
        matrix_total = matrix_cat

    # Raw true-bin callback (top-hat * completeness)
    def raw_true_bin_for_edge(j: int, zmin: float, zmax: float) -> FloatArray1D:
        """Returns the unnormalized true-bin distribution for bin ``j``.

        The true-bin distribution is constructed by applying the bin selection window
        to the parent distribution,

            ``n_true_j(z) = n(z) * S_j(z)``,

        where

            ``S_j(z) = c_j * 1_{[zmin, zmax)}(z)``

        with completeness ``c_j``.

        Args:
            j: True-bin index.
            zmin: Lower true-z edge of the bin.
            zmax: Upper true-z edge of the bin.

        Returns:
            Unnormalized true-bin distribution evaluated on the common grid ``z_arr``.
        """
        sel = specz_selection_in_bin(
            z_arr,
            float(zmin),
            float(zmax),
            completeness=float(completeness_arr[j]),
        )
        return (parent_arr * sel).astype(np.float64, copy=False)

    # Mixer: apply response matrix to raw true bins
    def mixer(raw_bins: dict[int, FloatArray1D]) -> dict[int, FloatArray1D]:
        """Applies the total bin-to-bin response to the set of true-bin distributions.

        The input is a mapping of true-bin distributions ``n_true_j(z)``. The output is
        a mapping of observed-bin distributions ``n_obs_i(z)`` obtained by mixing bins
        with the total response matrix,

            ``n_obs_i(z) = sum_j M_total[i, j] * n_true_j(z)``.

        Args:
            raw_bins: Mapping from true-bin index ``j`` to unnormalized true-bin
                distribution ``n_true_j(z)``.

        Returns:
            Mapping from observed-bin index ``i`` to unnormalized observed-bin
            distribution ``n_obs_i(z)``.
        """
        return apply_response_matrix(raw_bins, matrix_total)

    # Build observed bins + norms + optional per-bin normalization
    bins, bins_norms, parent_norm = build_bins_on_edges(
        z=z_arr,
        nz_parent_for_meta=parent_arr,  # Same normalization as raw bins
        bin_edges=bin_edges_arr,
        raw_bin_for_edge=raw_true_bin_for_edge,
        normalize_bins=normalize_bins,
        norm_method=norm_method,
        mixer=mixer,
        need_meta=need_meta,
    )

    # Optional: avoid normalizing empty bins
    if normalize_bins:
        for i in range(n_bins_eff):
            area = float(np.trapezoid(bins[i], x=z_arr))
            if np.isclose(area, 0.0, atol=1e-12):
                continue

    # Optionally report metadata
    meta = finalize_tomo_metadata(
        kind="specz",
        z=z_arr,
        parent_nz=parent_arr,
        bin_edges=bin_edges_arr,
        bins=bins,
        inputs={
            "bin_edges_provided": bin_edges is not None,
            "binning_scheme": binning_scheme,
            "n_bins": n_bins,
            "normalize_input": normalize_input,
            "normalize_bins": normalize_bins,
            "norm_method": norm_method,
            "completeness": completeness,
            "catastrophic_frac": catastrophic_frac,
            "leakage_model": leakage_model,
            "leakage_sigma": leakage_sigma,
            "response_matrix_provided": response_matrix is not None,
            "specz_scatter": specz_scatter,
            "specz_scatter_model": specz_scatter_model,
            "sigma0": sigma0,
            "sigma1": sigma1,
            "bin_edges_space": "truez",
            "response": "M_total = M_scatter @ M_cat" if scatter_enabled else "M_total = M_cat",
        },
        parent_norm=parent_norm,
        bins_norms=bins_norms,
        include_metadata=include_metadata,
        save_metadata_path=save_metadata_path,
    )

    return (bins, meta) if include_metadata else bins
