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

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np
from scipy.special import erf

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
    validate_n_bins,
    validate_response_matrix,
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
        nz_parent_for_meta=parent_arr0,  # always record fractions relative to original parent
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
        parent_nz=parent_arr0,
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


def specz_selection_in_bin(
    z: FloatArray1D,
    bin_min: float,
    bin_max: float,
    completeness: float = 1.0,
    *,
    inclusive_right: bool = False,
) -> FloatArray1D:
    """Evaluates a true-z bin selection window on a redshift grid.

    Returns a top-hat selection function over ``[bin_min, bin_max)`` (or
    ``[bin_min, bin_max]`` when ``inclusive_right=True``), scaled by a completeness
    factor. This selection is multiplied by the parent ``n(z)`` to form a
    true-bin distribution.

    Args:
        z: True-redshift grid.
        bin_min: Lower edge of the true-z bin.
        bin_max: Upper edge of the true-z bin.
        completeness: Multiplicative completeness factor in [0, 1].
        inclusive_right: Whether to include ``bin_max`` in the selection.

    Returns:
        Array of selection values on ``z``.

    Raises:
        ValueError: If ``completeness`` is outside [0, 1].
    """
    z_arr = np.asarray(z, dtype=float)

    if not (0.0 <= completeness <= 1.0):
        raise ValueError("completeness must be in [0, 1].")

    if inclusive_right:
        mask = (z_arr >= bin_min) & (z_arr <= bin_max)
    else:
        mask = (z_arr >= bin_min) & (z_arr < bin_max)

    return completeness * mask.astype(np.float64)


def build_specz_response_matrix(
    n_bins: int,
    *,
    catastrophic_frac: Sequence[float] | float = 0.0,
    leakage_model: Literal["uniform", "neighbor", "gaussian"] = "neighbor",
    leakage_sigma: Sequence[float] | float = 1.0,
    response_matrix: FloatArray2D | None = None,
) -> FloatArray2D:
    """Constructs a bin-to-bin response matrix for catastrophic reassignment.

    Builds a column-stochastic matrix ``M`` with entries ``M[i, j] = P(i_obs|j_true)``
    that describes reassignment of a fraction of objects in each true bin to other
    observed bins. If an explicit ``response_matrix`` is provided, it is validated
    and returned directly.

    Args:
        n_bins: Number of tomographic bins.
        catastrophic_frac: Per-bin catastrophic fraction(s) in [0, 1].
        leakage_model: Redistribution rule for the catastrophic fraction.
        leakage_sigma: Width parameter for Gaussian leakage in bin-index space.
        response_matrix: Optional explicit response matrix overriding the model.

    Returns:
        A float array of shape ``(n_bins, n_bins)`` that is column-stochastic.

    Raises:
        ValueError: If inputs are invalid or the resulting matrix fails validation.
    """
    validate_n_bins(n_bins)

    if response_matrix is not None:
        matrix = np.asarray(response_matrix, dtype=float)
        validate_response_matrix(matrix, n_bins)
        return matrix.astype(np.float64, copy=False)

    f = as_per_bin(catastrophic_frac, n_bins, "catastrophic_frac").astype(float)
    if np.any((f < 0.0) | (f > 1.0)):
        raise ValueError("catastrophic_frac must be in [0, 1].")

    matrix = np.eye(n_bins, dtype=float)

    if np.allclose(f, 0.0):
        return matrix.astype(np.float64, copy=False)

    q = np.zeros((n_bins, n_bins), dtype=float)

    leakage_model = leakage_model.lower()
    match leakage_model:
        case "uniform":
            for j in range(n_bins):
                if n_bins == 1:
                    q[0, 0] = 1.0
                else:
                    q[:, j] = 1.0 / (n_bins - 1)
                    q[j, j] = 0.0

        case "neighbor":
            for j in range(n_bins):
                if n_bins == 1:
                    q[0, 0] = 1.0
                elif j == 0:
                    q[1, 0] = 1.0
                elif j == n_bins - 1:
                    q[n_bins - 2, j] = 1.0
                else:
                    q[j - 1, j] = 0.5
                    q[j + 1, j] = 0.5

        case "gaussian":
            sig = as_per_bin(leakage_sigma, n_bins, "leakage_sigma").astype(float)
            if np.any(sig <= 0.0):
                raise ValueError("leakage_sigma must be > 0 for leakage_model='gaussian'.")

            idx = np.arange(n_bins, dtype=float)
            for j in range(n_bins):
                t = (idx - float(j)) / float(sig[j])
                t = np.clip(t, -50.0, 50.0)  # prevent overflow in t*t;
                w = np.exp(-0.5 * t * t)

                w[j] = 0.0
                s = w.sum()
                if s <= 0.0:
                    if n_bins == 1:
                        q[0, 0] = 1.0
                    elif j == 0:
                        q[1, 0] = 1.0
                    elif j == n_bins - 1:
                        q[n_bins - 2, j] = 1.0
                    else:
                        q[j - 1, j] = 0.5
                        q[j + 1, j] = 0.5
                else:
                    q[:, j] = w / s

        case _:
            raise ValueError("leakage_model must be 'uniform', 'neighbor', or 'gaussian'.")

    for j in range(n_bins):
        matrix[:, j] = (1.0 - f[j]) * matrix[:, j] + f[j] * q[:, j]

    validate_response_matrix(matrix, n_bins)
    return matrix.astype(np.float64, copy=False)


def apply_response_matrix(
    bins: Mapping[int, FloatArray1D],
    matrix: FloatArray2D,
) -> dict[int, FloatArray1D]:
    """Applies a bin-response matrix to a set of tomographic bin distributions.

    Treats the input bins as a stacked array with shape ``(n_bins, ...)`` and
    returns the mixed bins ``obs = M @ true``. The response matrix is expected to be
    column-stochastic and compatible with the number of bins and array shapes.

    Args:
        bins: Mapping from bin index to bin distribution arrays. Must contain
            exactly keys ``0..n_bins-1`` and all arrays must share the same shape.
        matrix: Response matrix of shape ``(n_bins, n_bins)``.

    Returns:
        Dict mapping observed bin index to the mixed bin distribution.

    Raises:
        ValueError: If keys/shapes are inconsistent or the matrix is invalid.
    """
    n_bins = int(len(bins))
    validate_response_matrix(matrix, n_bins)

    expected = set(range(n_bins))
    keys = set(bins.keys())
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        raise ValueError(
            f"bins must contain exactly keys 0..{n_bins - 1}. missing={missing}, extra={extra}"
        )

    shapes = {bins[j].shape for j in range(n_bins)}
    if len(shapes) != 1:
        raise ValueError(f"All bin arrays must have the same shape; got {sorted(shapes)}.")

    arr = np.stack([bins[j] for j in range(n_bins)], axis=0)
    obs = matrix @ arr

    out: dict[int, FloatArray1D] = {}
    for i in range(n_bins):
        out[i] = obs[i].astype(np.float64, copy=False)
    return out


def specz_gaussian_response_matrix(
    *,
    z_arr: FloatArray1D,
    bin_edges: FloatArray1D,
    specz_scatter: Sequence[float] | float | None,
    model: Literal["sigma0_plus_sigma1_1pz"] = "sigma0_plus_sigma1_1pz",
    sigma0: float = 0.0,
    sigma1: float = 0.0,
) -> FloatArray2D:
    """Builds a Gaussian measurement-scatter response matrix for spec-z
    binning.

    Constructs a column-stochastic matrix that maps true bins to observed bins
    under a Gaussian measurement model for ``z_hat`` at fixed true redshift.
    Probabilities are computed by integrating the Gaussian over observed-bin edges
    and then averaging within each true bin.

    The scatter scale may be provided explicitly per bin, or derived from a simple
    parameterization controlled by ``sigma0`` and ``sigma1``.

    Args:
        z_arr: True-redshift grid used to evaluate within-bin averaging.
        bin_edges: True-redshift bin edges defining both true and observed bins.
        specz_scatter: Optional scatter scale(s) (scalar or per-bin sequence).
        model: Scatter parameterization used when ``specz_scatter`` is None.
        sigma0: Additive scatter component for ``model``.
        sigma1: Redshift-dependent scatter component for ``model``.

    Returns:
        A float array of shape ``(n_bins, n_bins)`` that is column-stochastic.

    Raises:
        ValueError: If scatter settings are inconsistent or invalid.
    """
    z_arr = np.asarray(z_arr, dtype=float)
    bin_edges_arr = np.asarray(bin_edges, dtype=float)
    n_bins = int(bin_edges_arr.size - 1)

    if specz_scatter is not None and (sigma0 != 0.0 or sigma1 != 0.0):
        raise ValueError("Provide either specz_scatter OR ``sigma0/sigma1``, not both.")

    if specz_scatter is not None:
        sig_bin = as_per_bin(specz_scatter, n_bins, "specz_scatter").astype(float)
        if np.any(sig_bin < 0.0):
            raise ValueError("specz_scatter must be >= 0 when provided.")
        # no-uncertainty limit: identity
        if np.allclose(sig_bin, 0.0):
            return np.eye(n_bins, dtype=np.float64)
    else:
        # specz_scatter not provided -> use sigma0/sigma1 model knobs
        if sigma0 < 0.0 or sigma1 < 0.0:
            raise ValueError("sigma0 and sigma1 must be >= 0.")
        if sigma0 == 0.0 and sigma1 == 0.0:
            return np.eye(n_bins, dtype=np.float64)

    sqrt2 = np.sqrt(2.0)
    matrix = np.zeros((n_bins, n_bins), dtype=float)

    masks: list[np.ndarray] = []
    for j in range(n_bins):
        zmin, zmax = bin_edges_arr[j], bin_edges_arr[j + 1]
        masks.append((z_arr >= zmin) & (z_arr < zmax))

    for j in range(n_bins):
        mask_j = masks[j]
        if not np.any(mask_j):
            matrix[j, j] = 1.0
            continue

        z_in = z_arr[mask_j]

        if specz_scatter is not None:
            sigma = float(sig_bin[j]) * np.ones_like(z_in)
            # if this bin’s scatter is exactly 0, it contributes no mixing
            if np.allclose(sigma, 0.0):
                matrix[j, j] = 1.0
                continue
        else:
            sigma = sigma0 + sigma1 * (1.0 + z_in)

        sigma = np.maximum(sigma, 1e-12)

        for i in range(n_bins):
            a = bin_edges_arr[i]
            b = bin_edges_arr[i + 1]
            t1 = (b - z_in) / (sqrt2 * sigma)
            t0 = (a - z_in) / (sqrt2 * sigma)
            p = 0.5 * (erf(t1) - erf(t0))
            matrix[i, j] = float(np.mean(p))

        colsum = matrix[:, j].sum()
        if colsum <= 0.0:
            matrix[j, j] = 1.0
        else:
            matrix[:, j] /= colsum

    validate_response_matrix(matrix, n_bins)
    return matrix.astype(np.float64, copy=False)
