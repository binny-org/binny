r"""Functions to build spectroscopic-redshift tomographic bins.

This module constructs true-redshift tomographic-bin distributions for
spectroscopic (true-z) binning, with optional survey-style bin-response effects
that map true bins to observed bins.

The main entry point is :func:`build_specz_bins`, which returns a dictionary
mapping observed bin index -> observed-bin distribution ``n_obs_i(z)``, all
evaluated on a common true-z grid ``z``.

Model overview
--------------
1) True-bin selection (top-hat in true z).

For true bin ``j`` with edges ``[z_j, z_{j+1}]``, define a selection window
``S_j(z)`` (left-closed, right-open by default) scaled by per-bin completeness
``c_j``::

    S_j(z) = c_j * 1_{[z_j, z_{j+1})}(z)

and build the true-bin distribution from the parent ``n(z)``::

    n_true_j(z) = n(z) * S_j(z)

2) Bin-to-bin response (optional).

Observed bins are constructed by mixing the true bins with a column-stochastic
response matrix ``M``::

    M[i, j] = P(i_obs | j_true)
    sum_i M[i, j] = 1  for every true bin j

The observed-bin distributions are::

    n_obs_i(z) = sum_j M[i, j] * n_true_j(z)

Supported response effects
--------------------------
- Catastrophic response (bin-level).
  A per-bin catastrophic fraction ``f_j`` is redistributed from the diagonal to
  other observed bins using a leakage prescription (uniform, neighbor, or
  Gaussian in bin-index space). If an explicit ``response_matrix`` is provided,
  it is used directly and overrides the leakage model.

- Finite spec-z measurement scatter (optional).
  Builds a second response matrix by integrating a Gaussian measurement model
  for ``z_hat | z_true`` over observed bin edges to get::

      P(z_hat in bin_i | z_true)

  then averages within each true bin to obtain a column-stochastic matrix
  ``M_scatter``.

  If both effects are enabled, the total response is applied as::

      M_total = M_scatter @ M_cat

  meaning catastrophic reassignment is applied first (at the bin level), followed
  by additional mixing from measurement scatter.

Normalization
-------------
- If ``normalize_input=True`` (default), the parent ``n(z)`` is normalized to
  integrate to 1 over ``z`` before binning.
- If ``normalize_bins=True`` (default), each output ``n_obs_i(z)`` is normalized
  to integrate to 1 over ``z`` (for non-empty bins).
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

FloatArray1D: TypeAlias = NDArray[np.float64]
FloatArray2D: TypeAlias = NDArray[np.float64]
BinningScheme: TypeAlias = str | Sequence[Mapping[str, Any]] | Mapping[str, Any]


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
    """Builds spectroscopic-redshift tomographic bins.

    Args:
        z: True-redshift grid where outputs are evaluated.
        nz: Parent true-redshift distribution evaluated on ``z`` (raw or normalized).
        bin_edges: True-redshift bin edges (length ``n_bins + 1``).
            Mutually exclusive with ``binning_scheme`` / ``n_bins``.
        binning_scheme: If ``bin_edges`` is None, scheme to build edges (e.g.
            ``"equidistant"``, ``"equal_number"``), or a mixed binning specification
            (sequence of segment dicts, or a dict with key ``"segments"``).
        n_bins: Number of bins when using a string ``binning_scheme``.
        bin_range: If provided, overrides ``bin_edges`` to build a range of bins.
        completeness: Per-bin completeness factors (scalar or per-bin sequence).
        catastrophic_frac: Per-bin catastrophic fractions (scalar or per-bin sequence).
        leakage_model: Leakage prescription for catastrophes ("uniform",
            "neighbor", "gaussian").
        leakage_sigma: Width for Gaussian leakage in bin-index space.
        response_matrix: Optional explicit catastrophic response matrix.
            If provided, it overrides ``catastrophic_frac`` and leakage settings.
        specz_scatter: Optional per-bin (or scalar) measurement scatter. If None,
            scatter can be enabled via ``specz_scatter_model`` with
            ``sigma0``/``sigma1``.
        specz_scatter_model: Scatter parameterization used when ``specz_scatter``
            is None.
        sigma0: Additive scatter component for ``specz_scatter_model``.
        sigma1: Redshift-dependent scatter component for ``specz_scatter_model``.
        normalize_input: If True, normalize the parent ``nz`` before binning.
        normalize_bins: If True, normalize each returned bin to integrate to 1 on ``z``.
        norm_method: Normalization method passed to :func:`normalize_1d`.
        include_metadata: If True, return metadata dict along with bins.
        save_metadata_path: If provided, save metadata to this path as a text file.

    Returns:
        If ``include_metadata`` is False, returns a dict mapping observed bin index to
        ``n_obs_i(z)`` evaluated on ``z``.

        If ``include_metadata`` is True, returns ``(bins, metadata)``.
    """
    z_arr, parent_arr0 = validate_axis_and_weights(z, nz)
    need_meta = include_metadata or (save_metadata_path is not None)

    # --- resolve + validate true-z bin edges
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

    # --- scatter knobs: either explicit specz_scatter OR sigma0/sigma1 model, not both
    if specz_scatter is not None and (sigma0 != 0.0 or sigma1 != 0.0):
        raise ValueError("Provide either specz_scatter OR ``sigma0/sigma1``, not both.")

    # --- normalize parent for true-bin construction if requested
    parent_arr = parent_arr0
    if normalize_input:
        parent_arr = normalize_1d(z_arr, parent_arr, method=norm_method)

    # --- broadcast per-bin params
    completeness_arr = as_per_bin(completeness, n_bins_eff, "completeness")

    # --- build response matrices (catastrophic + optional scatter)
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
        # Validate explicit per-bin scatter early (nice error message here).
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

    # --- raw true-bin callback (top-hat * completeness)
    def raw_true_bin_for_edge(j: int, zmin: float, zmax: float) -> FloatArray1D:
        sel = specz_selection_in_bin(
            z_arr,
            float(zmin),
            float(zmax),
            completeness=float(completeness_arr[j]),
        )
        return (parent_arr * sel).astype(np.float64, copy=False)

    # --- mixer: apply response matrix to raw true bins
    def mixer(raw_bins: dict[int, FloatArray1D]) -> dict[int, FloatArray1D]:
        return apply_response_matrix(raw_bins, matrix_total)

    # --- build observed bins + norms + optional per-bin normalization
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

    # Optional: avoid normalizing empty bins (keep consistent with old behavior).
    # If you want this globally, move the guard into build_bins_on_edges instead.
    if normalize_bins:
        for i in range(n_bins_eff):
            area = float(np.trapezoid(bins[i], x=z_arr))
            if np.isclose(area, 0.0, atol=1e-12):
                # keep as-is (all zeros)
                continue
            # build_bins_on_edges already normalized; nothing to do here.

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
    """Builds the spectroscopic (true-z) bin selection function on a redshift grid."""
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
    """Builds a catastrophic-response response matrix between tomographic bins."""
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
    """Applies a bin-to-bin response matrix to tomographic bin distributions."""
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
    """Builds a Gaussian measurement-scatter response matrix for spec-z binning."""
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
