"""Spectroscopic-redshift bin-response models."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from scipy.special import erf

from binny.binning.assignments import specz_selection_in_bin
from binny.utils.broadcasting import as_per_bin
from binny.utils.types import FloatArray1D, FloatArray2D
from binny.utils.validators import (
    validate_n_bins,
    validate_response_matrix,
)

__all__ = [
    "build_specz_response_matrix",
    "apply_response_matrix",
    "specz_assignment",
    "specz_gaussian_response_matrix",
]


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

    if np.allclose(f, 0.0):
        sigma_arr = np.asarray(as_per_bin(leakage_sigma, n_bins, "leakage_sigma"), dtype=float)
        if leakage_model != "neighbor" or not np.allclose(sigma_arr, 1.0):
            warnings.warn(
                "catastrophic_frac is zero for all bins, so catastrophic leakage "
                "parameters (leakage_model, leakage_sigma) are ignored.",
                RuntimeWarning,
                stacklevel=2,
            )
        return np.eye(n_bins, dtype=np.float64)

    matrix = np.eye(n_bins, dtype=float)
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
                t = np.clip(t, -50.0, 50.0)
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


def specz_assignment(
    z: FloatArray1D,
    edges: FloatArray1D,
    *,
    completeness: Sequence[float] | float = 1.0,
    catastrophic_frac: Sequence[float] | float = 0.0,
    leakage_model: Literal["uniform", "neighbor", "gaussian"] = "neighbor",
    leakage_sigma: Sequence[float] | float = 1.0,
    response_matrix: FloatArray2D | None = None,
    specz_scatter: Sequence[float] | float | None = None,
    specz_scatter_model: Literal["sigma0_plus_sigma1_1pz"] = "sigma0_plus_sigma1_1pz",
    sigma0: float = 0.0,
    sigma1: float = 0.0,
) -> FloatArray2D:
    """Return spectroscopic-redshift selection probabilities for each observed bin.

    Args:
        z: True-redshift grid.
        edges: True-redshift bin edges.
        completeness: Per-bin selection completeness.
        catastrophic_frac: Per-bin catastrophic reassignment fraction.
        leakage_model: Catastrophic reassignment prescription.
        leakage_sigma: Gaussian leakage width in bin-index space.
        response_matrix: Optional explicit catastrophic response matrix.
        specz_scatter: Optional spectroscopic measurement scatter.
        specz_scatter_model: Scatter parameterization.
        sigma0: Constant scatter component.
        sigma1: Redshift-dependent scatter component.

    Returns:
        Selection probabilities with shape ``(n_bins, n_z)``.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    edges_arr = np.asarray(edges, dtype=np.float64)
    n_bins = edges_arr.size - 1

    completeness_arr = as_per_bin(completeness, n_bins, "completeness")

    true_selection = np.stack(
        [
            specz_selection_in_bin(
                z_arr,
                float(edges_arr[i]),
                float(edges_arr[i + 1]),
                completeness=float(completeness_arr[i]),
            )
            for i in range(n_bins)
        ]
    )

    matrix_cat = build_specz_response_matrix(
        n_bins,
        catastrophic_frac=catastrophic_frac,
        leakage_model=leakage_model,
        leakage_sigma=leakage_sigma,
        response_matrix=response_matrix,
    )

    matrix_total = matrix_cat

    if specz_scatter is not None or sigma0 > 0.0 or sigma1 > 0.0:
        matrix_scatter = specz_gaussian_response_matrix(
            z_arr=z_arr,
            bin_edges=edges_arr,
            specz_scatter=specz_scatter,
            model=specz_scatter_model,
            sigma0=sigma0,
            sigma1=sigma1,
        )
        matrix_total = matrix_scatter @ matrix_cat

    return matrix_total @ true_selection
