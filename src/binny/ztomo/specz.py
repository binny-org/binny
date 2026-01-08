"""Helpers for spectroscopic redshift samples and binning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
from scipy.special import erf

from binny.core.validators import validate_axis_and_weights, validate_n_bins
from binny.utils.broadcasting import as_per_bin
from binny.utils.normalization import normalize_1d

__all__ = [
    "build_specz_bins",
    "specz_selection_in_bin",
    # error-model utilities
    "build_specz_misassignment_matrix",
    "apply_misassignment_matrix",
    "specz_gaussian_misassignment_matrix",
]


def build_specz_bins(
    z: Any,
    nz: Any,
    bin_edges: Any,
    *,
    completeness: Sequence[float] | float = 1.0,
    catastrophic_frac: Sequence[float] | float = 0.0,
    leakage_model: Literal["uniform", "neighbor", "gaussian"] = "neighbor",
    leakage_sigma: Sequence[float] | float = 1.0,
    misassignment_matrix: Any | None = None,
    specz_scatter: Sequence[float] | float | None = None,
    specz_scatter_model: Literal["const", "sigma0_plus_sigma1_1pz"] = "const",
    sigma0: float = 0.0,
    sigma1: float = 0.0,
    normalize_input: bool = True,
    normalize_bins: bool = True,
    norm_method: str = "trapezoid",
) -> dict[int, np.ndarray]:
    """Constructs spectroscopic tomographic redshift bins from a parent
     true-redshift distribution.

    This function constructs one redshift distribution per spectroscopic (true-z)
    tomographic bin defined by ``bin_edges``. The baseline model is a hard top-hat
    selection in true redshift, multiplied by an optional per-bin completeness.

    In addition, the function supports a compact bin-response error model that
    captures the two dominant spectroscopic systematics in a survey-ready way:

    1) **Catastrophic misassignment between bins.**
       A fraction of objects in a true bin can be reassigned to a different observed
       bin. This is modeled by a response matrix

       ``M[i, j] = P(i_obs | j_true)``,

       where each column sums to 1 (column-stochastic). If ``misassignment_matrix``
       is not provided, ``M`` is built from ``catastrophic_frac`` together with a
       simple leakage prescription (uniform, neighbor, or Gaussian in bin index).

    2) **Finite spec-z measurement scatter (optional).**
       If enabled, an additional response matrix is built by integrating a Gaussian
       measurement model for ``z_hat | z`` over the bin edges, yielding
       ``P(z_hat in bin_i | z_true)``. This produces small additional mixing between
       bins and can be combined with the catastrophic response.

    With these definitions, the returned observed-bin distributions are

    ``n_obs_i(z) = sum_j M_total[i, j] * n_true_j(z)``,

    where ``n_true_j(z)`` is the top-hat-selected distribution in true bin ``j``,
    and ``M_total`` is the (optional) combined response.

    Args:
        z: One-dimensional redshift grid where inputs are defined and outputs are
            evaluated.
        nz: Parent (intrinsic) redshift distribution evaluated on ``z``.
            If ``normalize_input=True``, this is renormalized to integrate to 1.
        bin_edges: One-dimensional array of tomographic bin edges in true redshift.
            Adjacent entries define a bin ``[bin_edges[i], bin_edges[i+1]]``.
            Must have length ``n_bins + 1`` and lie within the range spanned by ``z``.
        completeness: Per-true-bin completeness factors in ``[0, 1]``. May be a scalar
            (applied to all bins) or a sequence of length ``n_bins``.

        catastrophic_frac: Per-true-bin fraction in ``[0, 1]`` that is catastrophically
            assigned to the wrong observed bin. May be a scalar or a sequence of
            length ``n_bins``. The default (0) disables catastrophic leakage.
        leakage_model: Leakage prescription used to distribute catastrophically
            misassigned objects across observed bins when ``misassignment_matrix`` is
            not provided.
            Supported values are:
            - ``"uniform"``: distribute to all other bins equally
            - ``"neighbor"``: distribute to adjacent bins (edge bins renormalized)
            - ``"gaussian"``: distribute in bin-index space with width ``leakage_sigma``
        leakage_sigma: Width parameter for ``leakage_model="gaussian"``. May be a scalar
            or a sequence of length ``n_bins``.
        misassignment_matrix: Optional explicit response matrix with shape
            ``(n_bins, n_bins)`` where ``M[i, j] = P(i_obs | j_true)`` and each column
            sums to 1. If provided, this overrides ``catastrophic_frac`` and
            ``leakage_model``.

        specz_scatter: Optional Gaussian measurement scatter for ``z_hat | z``.
            When provided, a measurement-scatter response matrix is built by integrating
            the Gaussian model over the bin edges. May be a scalar or a sequence of
            length ``n_bins``. The default (``None``) disables this contribution.
        specz_scatter_model: Parameterization used when building the measurement-scatter
            response. If ``specz_scatter`` is provided, this argument is ignored.
        sigma0: Additive component of the measurement scatter used by
            ``specz_scatter_model="sigma0_plus_sigma1_1pz"``.
        sigma1: Redshift-dependent component of the measurement scatter used by
            ``specz_scatter_model="sigma0_plus_sigma1_1pz"``.

        normalize_input: Whether to normalize the input ``nz`` before binning.
        normalize_bins: Whether to normalize each returned bin distribution so it
            integrates to 1 on ``z`` (for non-empty bins).
        norm_method: Normalization method passed to :func:`normalize_1d`
            (e.g. ``"trapezoid"`` or ``"simpson"``).

    Returns:
        A mapping from observed bin index to the corresponding binned distribution
        evaluated on ``z``.

    Raises:
        ValueError: If ``bin_edges`` does not define a valid number of bins.
        ValueError: If ``bin_edges`` is not strictly increasing or lies outside the
            range of ``z``.
        ValueError: If any scalar-or-sequence parameter cannot be broadcast to length
            ``n_bins``.
        ValueError: If any probability parameter is outside its valid range, or if an
            explicit ``misassignment_matrix`` has invalid shape or normalization.

    Notes:
        If both a catastrophic response and a measurement-scatter response are enabled,
        the combined response is applied as

        ``M_total = M_scatter @ M_cat``,

        meaning the catastrophic reassignment is applied at the bin level and then
        additional mixing from finite measurement scatter is applied. Both matrices are
        validated to be column-stochastic.
    """
    z_arr, n_arr = validate_axis_and_weights(z, nz)

    bin_edges_arr = np.asarray(bin_edges, dtype=float)
    n_bins = bin_edges_arr.size - 1
    validate_n_bins(n_bins)

    if np.any(np.diff(bin_edges_arr) <= 0):
        raise ValueError("bin_edges must be strictly increasing.")

    if bin_edges_arr[0] < z_arr[0] or bin_edges_arr[-1] > z_arr[-1]:
        raise ValueError(
            f"bin_edges must lie within z-range [{z_arr[0]}, {z_arr[-1]}], "
            f"got [{bin_edges_arr[0]}, {bin_edges_arr[-1]}]."
        )

    if normalize_input:
        n_arr = normalize_1d(z_arr, n_arr, method=norm_method)

    completeness_arr = as_per_bin(completeness, n_bins, "completeness")
    true_bins: dict[int, np.ndarray] = {}

    for j, (z_min, z_max) in enumerate(
        zip(bin_edges_arr[:-1], bin_edges_arr[1:], strict=False)
    ):
        sel = specz_selection_in_bin(
            z_arr,
            float(z_min),
            float(z_max),
            completeness=float(completeness_arr[j]),
        )
        true_bins[j] = n_arr * sel

    M_cat = build_specz_misassignment_matrix(
        n_bins,
        catastrophic_frac=catastrophic_frac,
        leakage_model=leakage_model,
        leakage_sigma=leakage_sigma,
        misassignment_matrix=misassignment_matrix,
    )

    M_total = M_cat

    # Optional Gaussian measurement scatter -> response matrix (often tiny)
    if specz_scatter is not None or (
        specz_scatter is None
        and specz_scatter_model == "sigma0_plus_sigma1_1pz"
        and (sigma0 > 0.0 or sigma1 > 0.0)
    ):
        M_scatter = specz_gaussian_misassignment_matrix(
            z_arr=z_arr,
            bin_edges=bin_edges_arr,
            specz_scatter=specz_scatter,
            model=specz_scatter_model,
            sigma0=sigma0,
            sigma1=sigma1,
        )
        M_total = M_scatter @ M_total

    # --- apply response to get observed bins ---
    bins = apply_misassignment_matrix(true_bins, M_total)

    # --- normalize outputs if requested ---
    if normalize_bins:
        for i in range(n_bins):
            integral = np.trapezoid(bins[i], z_arr)
            if not np.isclose(integral, 0.0, atol=1e-12):
                bins[i] = normalize_1d(z_arr, bins[i], method=norm_method)

    return bins


def specz_selection_in_bin(
    z: Any,
    bin_min: float,
    bin_max: float,
    completeness: float = 1.0,
    *,
    inclusive_right: bool = False,
) -> np.ndarray:
    """Computes a top-hat selection function for a spectroscopic bin."""
    z_arr = np.asarray(z, dtype=float)

    if not (0.0 <= completeness <= 1.0):
        raise ValueError("completeness must be in [0, 1].")

    if inclusive_right:
        mask = (z_arr >= bin_min) & (z_arr <= bin_max)
    else:
        mask = (z_arr >= bin_min) & (z_arr < bin_max)

    return completeness * mask.astype(float)


def build_specz_misassignment_matrix(
    n_bins: int,
    *,
    catastrophic_frac: Sequence[float] | float = 0.0,
    leakage_model: Literal["uniform", "neighbor", "gaussian"] = "neighbor",
    leakage_sigma: Sequence[float] | float = 1.0,
    misassignment_matrix: Any | None = None,
) -> np.ndarray:
    """Builds a bin-to-bin response matrix M[i, j] = P(i_obs | j_true).

    Defaults to identity (no misassignment).
    """
    validate_n_bins(n_bins)

    if misassignment_matrix is not None:
        M = np.asarray(misassignment_matrix, dtype=float)
        _validate_response_matrix(M, n_bins)
        return M

    f = as_per_bin(catastrophic_frac, n_bins, "catastrophic_frac").astype(float)
    if np.any((f < 0.0) | (f > 1.0)):
        raise ValueError("catastrophic_frac must be in [0, 1].")

    # Start with identity (stay in same bin)
    M = np.eye(n_bins, dtype=float)

    if np.allclose(f, 0.0):
        return M

    # Build leakage distributions q_{i|j} for catastrophes, then:
    # M[:, j] = (1 - f_j) * e_j + f_j * q[:, j]
    q = np.zeros((n_bins, n_bins), dtype=float)

    if leakage_model == "uniform":
        for j in range(n_bins):
            if n_bins == 1:
                q[0, 0] = 1.0
            else:
                q[:, j] = 1.0 / (n_bins - 1)
                q[j, j] = 0.0

    elif leakage_model == "neighbor":
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

    elif leakage_model == "gaussian":
        sig = as_per_bin(leakage_sigma, n_bins, "leakage_sigma").astype(float)
        if np.any(sig <= 0.0):
            raise ValueError("leakage_sigma must be > 0 for leakage_model='gaussian'.")

        idx = np.arange(n_bins, dtype=float)
        for j in range(n_bins):
            w = np.exp(-0.5 * ((idx - float(j)) / float(sig[j])) ** 2)
            w[j] = 0.0  # catastrophes go elsewhere
            s = w.sum()
            if s <= 0.0:
                # fallback: if too narrow, push to neighbors
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
    else:
        raise ValueError("leakage_model must be 'uniform', 'neighbor', or 'gaussian'.")

    for j in range(n_bins):
        M[:, j] = (1.0 - f[j]) * M[:, j] + f[j] * q[:, j]

    _validate_response_matrix(M, n_bins)
    return M


def apply_misassignment_matrix(
    bins: dict[int, np.ndarray],
    M: np.ndarray,
) -> dict[int, np.ndarray]:
    """Apply response matrix to a set of true-bin distributions."""
    n_bins = len(bins)
    _validate_response_matrix(M, n_bins)

    # Stack in true-bin order
    z_shape = next(iter(bins.values())).shape
    arr = np.stack([bins[j] for j in range(n_bins)], axis=0)  # (n_bins, n_z)

    if arr.shape[0] != n_bins:
        raise ValueError("bins dict must contain keys 0..n_bins-1.")

    obs = M @ arr  # (n_bins, n_z)

    out: dict[int, np.ndarray] = {}
    for i in range(n_bins):
        out[i] = obs[i].reshape(z_shape)
    return out


def specz_gaussian_misassignment_matrix(
    *,
    z_arr: np.ndarray,
    bin_edges: np.ndarray,
    specz_scatter: Sequence[float] | float | None,
    model: Literal["const", "sigma0_plus_sigma1_1pz"] = "const",
    sigma0: float = 0.0,
    sigma1: float = 0.0,
) -> np.ndarray:
    """Build a response matrix from a Gaussian measurement model z_hat | z.

    This computes M[i, j] = P(z_hat in bin_i | z in bin_j) by averaging
    P(bin_i | z) over z in bin_j with weights proportional to the top-hat
    indicator on the provided grid.

    This is an optional “theoretical completeness” piece; in practice spec-z
    scatter is often negligible compared to catastrophes.
    """
    bin_edges_arr = np.asarray(bin_edges, dtype=float)
    n_bins = bin_edges_arr.size - 1

    # Build per-z sigma(z)
    if specz_scatter is not None:
        # treat as constant per true bin: scalar or length n_bins
        sig_bin = as_per_bin(specz_scatter, n_bins, "specz_scatter").astype(float)
        if np.any(sig_bin <= 0.0):
            raise ValueError("specz_scatter must be > 0 when provided.")
    else:
        if model != "sigma0_plus_sigma1_1pz":
            raise ValueError(
                "If specz_scatter is None, model must be 'sigma0_plus_sigma1_1pz'."
            )
        if sigma0 < 0.0 or sigma1 < 0.0:
            raise ValueError("sigma0 and sigma1 must be >= 0.")
        if sigma0 == 0.0 and sigma1 == 0.0:
            # no scatter -> identity
            return np.eye(n_bins, dtype=float)

    sqrt2 = np.sqrt(2.0)
    M = np.zeros((n_bins, n_bins), dtype=float)

    # Precompute bin membership masks for “true bin j”
    masks = []
    for j in range(n_bins):
        zmin, zmax = bin_edges_arr[j], bin_edges_arr[j + 1]
        masks.append((z_arr >= zmin) & (z_arr < zmax))

    for j in range(n_bins):
        mask_j = masks[j]
        if not np.any(mask_j):
            # no grid support -> fall back to identity for this column
            M[j, j] = 1.0
            continue

        z_in = z_arr[mask_j]

        if specz_scatter is not None:
            sigma = float(sig_bin[j]) * np.ones_like(z_in)
        else:
            sigma = sigma0 + sigma1 * (1.0 + z_in)

        sigma = np.maximum(sigma, 1e-12)

        # For each observed bin i, compute average P(i | z)
        # over z in true bin j.
        for i in range(n_bins):
            a = bin_edges_arr[i]
            b = bin_edges_arr[i + 1]
            # P(a <= z_hat < b | z) for Gaussian N(z, sigma)
            t1 = (b - z_in) / (sqrt2 * sigma)
            t0 = (a - z_in) / (sqrt2 * sigma)
            p = 0.5 * (erf(t1) - erf(t0))
            M[i, j] = float(np.mean(p))

        # normalize column (numerical safety)
        colsum = M[:, j].sum()
        if colsum <= 0.0:
            M[j, j] = 1.0
        else:
            M[:, j] /= colsum

    _validate_response_matrix(M, n_bins)
    return M


def _validate_response_matrix(M: np.ndarray, n_bins: int) -> None:
    if M.shape != (n_bins, n_bins):
        raise ValueError(f"misassignment_matrix must have shape ({n_bins}, {n_bins}).")
    if not np.all(np.isfinite(M)):
        raise ValueError("misassignment_matrix must be finite.")
    if np.any(M < -1e-15):
        raise ValueError("misassignment_matrix must be non-negative.")
    M = np.maximum(M, 0.0)
    col_sums = M.sum(axis=0)
    if not np.allclose(col_sums, 1.0, rtol=1e-6, atol=1e-10):
        raise ValueError(
            "Each column of misassignment_matrix must sum to 1 (column-stochastic)."
        )
