"""Spectroscopic-redshift bin-response models."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr

from binny.binning.assignments import (
    deterministic_assignment,
)
from binny.binning.edges import validate_bin_edges
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


def _gaussian_bin_probability(
    t: float,
    scatter_lo: float,
    scatter_hi: float,
    lower_offset: float,
    upper_offset: float,
    width: float,
    same_bin: bool,
) -> float:
    """Returns the Gaussian assignment probability within a true-bin interval.

    Evaluates the probability that an object at a given true redshift is
    assigned to an observed redshift bin under a Gaussian measurement model.

    The true-redshift coordinate is expressed as a dimensionless position
    within the source bin. The scatter is interpolated linearly between its
    values at the lower and upper source-bin boundaries.

    For positive standardized lower bounds, the reflected Gaussian CDF is
    used to reduce numerical cancellation in the upper tail.

    Args:
        t: Dimensionless position within the true bin, between 0 and 1.
        scatter_lo: Measurement scatter at the lower true-bin boundary.
        scatter_hi: Measurement scatter at the upper true-bin boundary.
        lower_offset: Lower observed-bin edge relative to the lower
            true-bin edge.
        upper_offset: Upper observed-bin edge relative to the lower
            true-bin edge.
        width: Width of the true-redshift bin.
        same_bin: Whether the observed and true bin indices coincide.

    Returns:
        Probability of assignment to the observed bin.
    """
    sigma = (1.0 - t) * scatter_lo + t * scatter_hi

    if sigma == 0.0:
        return float(same_bin)

    lower = (lower_offset - t * width) / sigma
    upper = (upper_offset - t * width) / sigma

    if lower >= 0.0:
        return float(ndtr(-lower) - ndtr(-upper))

    return float(ndtr(upper) - ndtr(lower))


def specz_gaussian_response_matrix(
    *,
    z_arr: FloatArray1D,
    bin_edges: FloatArray1D,
    specz_scatter: Sequence[float] | float | None,
    model: Literal["sigma0_plus_sigma1_1pz"] = "sigma0_plus_sigma1_1pz",
    sigma0: float = 0.0,
    sigma1: float = 0.0,
) -> FloatArray2D:
    """Builds a Gaussian measurement-scatter response matrix for spec-z binning.

    Constructs a column-stochastic matrix that maps true bins to observed
    bins under a Gaussian measurement model.

    Response probabilities are averaged continuously over each true-bin
    interval, assuming uniform weighting in true redshift. The resulting
    matrix is independent of the sampling resolution and spacing of the
    redshift grid.

    The scatter scale may be provided explicitly as a scalar or per-bin
    sequence, or derived from the redshift-dependent parameterization
    ``sigma(z) = sigma0 + sigma1 * (1 + z)``.

    Numerical integration is performed over a dimensionless coordinate
    within each true-bin interval. Additional integration points near
    the bin boundaries improve resolution of narrow Gaussian scatter.

    Columns are normalized to conserve the population within the modeled
    tomographic bins. Probability outside the modeled redshift range is
    therefore excluded through column renormalization rather than treated
    as a loss of objects.

    Args:
        z_arr: True-redshift grid. Retained as an input for consistency
            with the spectroscopic binning interface. The response calculation
            does not depend on the grid sampling.
        bin_edges: True-redshift bin edges defining true and observed bins.
        specz_scatter: Optional scalar or per-bin constant scatter scale.
        model: Scatter parameterization used when specz_scatter is None.
            The supported model is ``"sigma0_plus_sigma1_1pz"``.
        sigma0: Additive scatter component.
        sigma1: Redshift-dependent scatter component.

    Returns:
        A column-stochastic float array of shape ``(n_bins, n_bins)``.

    Raises:
        ValueError: If bin edges or scatter settings are invalid, or if the
            resulting response matrix cannot be normalized or validated.
    """
    bin_edges_arr = validate_bin_edges(bin_edges)
    n_bins = bin_edges_arr.size - 1

    if specz_scatter is not None and (sigma0 != 0.0 or sigma1 != 0.0):
        raise ValueError("Provide either specz_scatter OR ``sigma0/sigma1``, not both.")

    if not np.all(np.isfinite([sigma0, sigma1])):
        raise ValueError("sigma0 and sigma1 must be finite.")

    if sigma0 < 0.0 or sigma1 < 0.0:
        raise ValueError("sigma0 and sigma1 must be >= 0.")

    if specz_scatter is not None:
        sig_bin = as_per_bin(specz_scatter, n_bins, "specz_scatter").astype(float)

        if not np.all(np.isfinite(sig_bin)):
            raise ValueError("specz_scatter must be finite.")

        if np.any(sig_bin < 0.0):
            raise ValueError("specz_scatter must be >= 0 when provided.")

        if np.all(sig_bin == 0.0):
            return np.eye(n_bins, dtype=np.float64)
    else:
        if model != "sigma0_plus_sigma1_1pz":
            raise ValueError("model must be 'sigma0_plus_sigma1_1pz'.")

        if sigma0 == 0.0 and sigma1 == 0.0:
            return np.eye(n_bins, dtype=np.float64)

        sig_bin = None

    matrix = np.zeros((n_bins, n_bins), dtype=np.float64)

    for j in range(n_bins):
        lo = float(bin_edges_arr[j])
        hi = float(bin_edges_arr[j + 1])
        width = hi - lo

        scatter_lo = float(sig_bin[j]) if sig_bin is not None else sigma0 + sigma1 * (1.0 + lo)
        scatter_hi = float(sig_bin[j]) if sig_bin is not None else sigma0 + sigma1 * (1.0 + hi)

        if scatter_lo < 0.0 or scatter_hi < 0.0:
            raise ValueError("Measurement scatter must be non-negative.")

        if scatter_lo == 0.0 and scatter_hi == 0.0:
            matrix[j, j] = 1.0
            continue

        points = sorted(
            {
                point
                for point in (
                    12.0 * scatter_lo / width,
                    1.0 - 12.0 * scatter_hi / width,
                )
                if 0.0 < point < 1.0
            }
        )

        for i in range(n_bins):
            a = float(bin_edges_arr[i])
            b = float(bin_edges_arr[i + 1])

            matrix[i, j], _ = quad(
                _gaussian_bin_probability,
                0.0,
                1.0,
                args=(
                    scatter_lo,
                    scatter_hi,
                    a - lo,
                    b - lo,
                    width,
                    i == j,
                ),
                points=points,
                epsabs=1e-14,
                epsrel=1e-9,
            )

        colsum = matrix[:, j].sum()

        if not np.isfinite(colsum) or colsum <= 0.0:
            raise ValueError("Invalid Gaussian scatter response column.")

        matrix[:, j] /= colsum

    validate_response_matrix(matrix, n_bins)
    return matrix


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
    """Returns spectroscopic-redshift selection probabilities for each observed bin.

    Constructs true-redshift top-hat selection functions with optional
    per-bin completeness factors and applies the combined catastrophic
    reassignment and Gaussian measurement-scatter response matrix.

    The resulting selection probabilities are evaluated on the supplied
    true-redshift grid and are independent of the parent redshift
    distribution.

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

    Raises:
        ValueError: If completeness, response matrix, or scatter settings
            are invalid.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    edges_arr = np.asarray(edges, dtype=np.float64)

    matrix_total, _ = _build_total_response_matrix(
        z_arr,
        edges_arr,
        catastrophic_frac=catastrophic_frac,
        leakage_model=leakage_model,
        leakage_sigma=leakage_sigma,
        response_matrix=response_matrix,
        specz_scatter=specz_scatter,
        specz_scatter_model=specz_scatter_model,
        sigma0=sigma0,
        sigma1=sigma1,
    )

    return deterministic_assignment(
        z_arr,
        edges_arr,
        completeness=completeness,
        response_matrix=matrix_total,
    )


def _specz_scatter_enabled(
    n_bins: int,
    *,
    specz_scatter: Sequence[float] | float | None = None,
    sigma0: float = 0.0,
    sigma1: float = 0.0,
) -> bool:
    """Return whether validated spectroscopic scatter settings enable scatter."""
    if specz_scatter is not None:
        return bool(np.any(as_per_bin(specz_scatter, n_bins, "specz_scatter") > 0.0))

    return sigma0 > 0.0 or sigma1 > 0.0


def _build_total_response_matrix(
    z: FloatArray1D,
    edges: FloatArray1D,
    *,
    catastrophic_frac: Sequence[float] | float = 0.0,
    leakage_model: Literal["uniform", "neighbor", "gaussian"] = "neighbor",
    leakage_sigma: Sequence[float] | float = 1.0,
    response_matrix: FloatArray2D | None = None,
    specz_scatter: Sequence[float] | float | None = None,
    specz_scatter_model: Literal["sigma0_plus_sigma1_1pz"] = "sigma0_plus_sigma1_1pz",
    sigma0: float = 0.0,
    sigma1: float = 0.0,
) -> tuple[FloatArray2D, bool]:
    """Builds the combined catastrophic and measurement-scatter response.

    Constructs the catastrophic reassignment matrix and the Gaussian
    measurement-scatter response matrix using their respective builders.

    When measurement scatter is enabled, the total response is given by
    ``M_total = M_scatter @ M_cat``, corresponding to catastrophic
    reassignment followed by measurement scatter.

    When measurement scatter is exactly zero, the catastrophic response
    matrix is returned directly. Scatter parameters are validated in both
    cases, including when measurement scatter is disabled.

    Args:
        z: True-redshift grid.
        edges: True-redshift bin edges.
        catastrophic_frac: Scalar or per-bin catastrophic reassignment fraction.
        leakage_model: Redistribution rule for catastrophic reassignment.
        leakage_sigma: Gaussian leakage width in bin-index space.
        response_matrix: Optional explicit catastrophic response matrix.
        specz_scatter: Optional scalar or per-bin measurement-scatter scale.
        specz_scatter_model: Scatter parameterization used when
            ``specz_scatter`` is None.
        sigma0: Additive measurement-scatter component.
        sigma1: Redshift-dependent measurement-scatter component.

    Returns:
        A tuple containing the combined column-stochastic response matrix
        and a boolean indicating whether measurement scatter is enabled.

    Raises:
        ValueError: If catastrophic reassignment or measurement-scatter
            settings are invalid.
    """
    n_bins = len(edges) - 1

    matrix_cat = build_specz_response_matrix(
        n_bins,
        catastrophic_frac=catastrophic_frac,
        leakage_model=leakage_model,
        leakage_sigma=leakage_sigma,
        response_matrix=response_matrix,
    )

    matrix_scatter = specz_gaussian_response_matrix(
        z_arr=z,
        bin_edges=edges,
        specz_scatter=specz_scatter,
        model=specz_scatter_model,
        sigma0=sigma0,
        sigma1=sigma1,
    )

    scatter_enabled = _specz_scatter_enabled(
        n_bins,
        specz_scatter=specz_scatter,
        sigma0=sigma0,
        sigma1=sigma1,
    )

    if not scatter_enabled:
        return matrix_cat, False

    return matrix_scatter @ matrix_cat, True
