"""Module defining various redshift distributions for astronomical sources.

All functions here return *shapes* by default (unnormalized). If ``normalize=True``,
the returned curve is normalized to integrate to 1 over the provided redshift
grid ``z`` using :func:`binny.utils.normalization.normalize_over_z`.

Notes:
    - Normalization requires a 1D, strictly increasing ``z`` grid with at least
      two points.
    - Passing a scalar ``z`` with ``normalize=True`` will raise.
"""

from __future__ import annotations

import numpy as np
from numpy import exp
from scipy.special import erf

from binny.utils.normalization import normalize_over_z
from binny.utils.types import FloatArray

__all__ = [
    "smail_like_distribution",
    "gaussian_distribution",
    "gaussian_mixture_distribution",
    "gamma_distribution",
    "schechter_like_distribution",
    "lognormal_distribution",
    "tophat_distribution",
    "shifted_smail_distribution",
    "skew_normal_distribution",
    "student_t_distribution",
    "tabulated_distribution",
]


def _maybe_normalize(z: FloatArray, nz: FloatArray, normalize: bool) -> FloatArray:
    """Optionally normalizes ``nz`` to integrate to 1 over ``z``."""
    if not normalize:
        return nz
    return normalize_over_z(z, nz)


def smail_like_distribution(
    z: float | FloatArray,
    z0: float,
    alpha: float,
    beta: float,
    *,
    normalize: bool = False,
) -> FloatArray:
    """Returns a Smail-type redshift distribution.

    This function defines an unnormalized Smail-like shape often used to model
    galaxy redshift distributions in weak lensing studies::

        n(z) = (z / z0)^alpha * exp[-(z / z0)^beta]

    Args:
        z: Redshift or array of redshifts.
        z0: Characteristic redshift scale (must be positive).
        alpha: Power-law index for low redshifts.
        beta: Exponential cutoff index for high redshifts.
        normalize: If ``True``, normalizes the curve to have
            integral 1 over ``z``.

    Returns:
        Smail-like distribution evaluated at ``z`` (normalized if requested).

    Raises:
        ValueError: If ``z0`` is not positive.
    """
    if z0 <= 0:
        raise ValueError("z0 must be positive.")

    z_arr = np.asarray(z, dtype=np.float64)
    ratio = z_arr / z0
    nz = (ratio**alpha) * exp(-(ratio**beta))
    return _maybe_normalize(z_arr, nz, normalize)


def gaussian_distribution(
    z: float | FloatArray,
    mu: float,
    sigma: float,
    *,
    normalize: bool = False,
) -> FloatArray:
    """Returns a Gaussian redshift distribution.

    This function defines an unnormalized Gaussian shape::

        n(z) = exp[-0.5 * ((z - mu) / sigma)^2]

    Args:
        z: Redshift or array of redshifts.
        mu: Mean redshift.
        sigma: Standard deviation (must be positive).
        normalize: If ``True``, normalizes the curve to have
            integral 1 over ``z``.

    Returns:
        Gaussian evaluated at ``z`` (normalized if requested).

    Raises:
        ValueError: If ``sigma`` is not positive.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    z_arr = np.asarray(z, dtype=np.float64)
    t = (z_arr - mu) / sigma
    nz = exp(-0.5 * t**2)
    return _maybe_normalize(z_arr, nz, normalize)


def gaussian_mixture_distribution(
    z: float | FloatArray,
    mus: FloatArray,
    sigmas: FloatArray,
    weights: FloatArray | None = None,
    *,
    normalize: bool = False,
) -> FloatArray:
    """Returns a Gaussian mixture redshift distribution.

    This function defines an unnormalized Gaussian mixture shape::

        n(z) = sum_{i=1}^K w_i * exp[-0.5 * ((z - mu_i) / sigma_i)^2]

    Args:
        z: Redshift or array of redshifts.
        mus: Array of component means, shape ``(K,)``.
        sigmas: Array of component std devs, shape ``(K,)``. Must be positive.
        weights: Optional nonnegative weights, shape ``(K,)``. If ``None``,
            equal weights are used. Weights do not need to sum to ``1``.
        normalize: If ``True``, normalizes the curve to have
            integral 1 over ``z``.

    Returns:
        Gaussian mixture evaluated at ``z`` (normalized if requested).

    Raises:
        ValueError: If shapes mismatch, any sigma is non-positive,
            or any weight is negative.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    mus_arr = np.asarray(mus, dtype=np.float64)
    sigmas_arr = np.asarray(sigmas, dtype=np.float64)

    if mus_arr.ndim != 1 or sigmas_arr.ndim != 1:
        raise ValueError("mus and sigmas must be 1D arrays.")
    if mus_arr.shape[0] != sigmas_arr.shape[0]:
        raise ValueError("mus and sigmas must have the same length.")
    if np.any(sigmas_arr <= 0):
        raise ValueError("All sigmas must be positive.")

    if weights is None:
        w = np.ones_like(mus_arr, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.shape[0] != mus_arr.shape[0]:
            raise ValueError("weights must be a 1D array with the same length as mus.")
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative.")

    z2 = np.atleast_1d(z_arr)
    t = (z2[None, :] - mus_arr[:, None]) / sigmas_arr[:, None]
    comps = exp(-0.5 * t**2)
    nz = (w[:, None] * comps).sum(axis=0)

    if z_arr.ndim == 0:
        nz = nz.reshape(())

    return _maybe_normalize(np.asarray(z_arr), np.asarray(nz), normalize)


def gamma_distribution(
    z: float | FloatArray,
    k: float,
    theta: float,
    *,
    normalize: bool = False,
) -> FloatArray:
    """Returns a gamma-shaped redshift distribution.

    This function defines an unnormalized gamma-shaped distribution::

        n(z) = z^(k-1) * exp(-z/theta)    for z >= 0
        n(z) = 0                         for z < 0

    Args:
        z: Redshift or array of redshifts.
        k: Shape parameter (must be positive).
        theta: Scale parameter (must be positive).
        normalize: If ``True``, normalizes the curve to have
            integral 1 over ``z``.

    Returns:
        Gamma-shaped distribution evaluated at ``z`` (normalized if requested).

    Raises:
        ValueError: If ``k`` or ``theta`` is not positive.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    if theta <= 0:
        raise ValueError("theta must be positive.")

    z_arr = np.asarray(z, dtype=np.float64)
    out = np.zeros_like(z_arr, dtype=np.float64)
    m = z_arr >= 0
    out[m] = (z_arr[m] ** (k - 1.0)) * exp(-(z_arr[m] / theta))
    return _maybe_normalize(z_arr, out, normalize)


def schechter_like_distribution(
    z: float | FloatArray,
    z0: float,
    alpha: float,
    *,
    normalize: bool = False,
) -> FloatArray:
    """Returns a Schechter-like redshift distribution.

    This function defines an unnormalized Schechter-like shape::

        n(z) = (z / z0)^alpha * exp(-(z / z0))

    This is equivalent to a Smail-like form with ``beta = 1``.

    Args:
        z: Redshift or array of redshifts.
        z0: Characteristic redshift scale (must be positive).
        alpha: Power-law index.
        normalize: If ``True``, normalizes the curve to
            have integral 1 over ``z``.

    Returns:
        Schechter-like distribution evaluated at ``z``.

    Raises:
        ValueError: If ``z0`` is not positive.
    """
    if z0 <= 0:
        raise ValueError("z0 must be positive.")

    z_arr = np.asarray(z, dtype=np.float64)
    ratio = z_arr / z0
    nz = (ratio**alpha) * exp(-ratio)
    return _maybe_normalize(z_arr, nz, normalize)


def lognormal_distribution(
    z: float | FloatArray,
    mu_ln: float,
    sigma_ln: float,
    *,
    normalize: bool = False,
) -> FloatArray:
    """Returns a lognormal redshift distribution.

    This function defines an unnormalized lognormal shape::

        n(z) approx (1/z) * exp[- (ln z - mu_ln)^2 / (2 sigma_ln^2)]  for z > 0
        n(z) = 0  for z <= 0

    Args:
        z: Redshift or array of redshifts.
        mu_ln: Mean of ``ln(z)``.
        sigma_ln: Std dev of ``ln(z)`` (must be positive).
        normalize: If ``True``, normalizes the curve to
            have integral 1 over ``z``.

    Returns:
        Lognormal shape evaluated at ``z`` (normalized if requested).

    Raises:
        ValueError: If ``sigma_ln`` is not positive.
    """
    if sigma_ln <= 0:
        raise ValueError("sigma_ln must be positive.")

    z_arr = np.asarray(z, dtype=np.float64)
    out = np.zeros_like(z_arr, dtype=np.float64)
    m = z_arr > 0
    lnz = np.log(z_arr[m])
    t = (lnz - mu_ln) / sigma_ln
    out[m] = (1.0 / z_arr[m]) * exp(-0.5 * t**2)
    return _maybe_normalize(z_arr, out, normalize)


def tophat_distribution(
    z: float | FloatArray,
    zmin: float,
    zmax: float,
    *,
    normalize: bool = False,
) -> FloatArray:
    """Returns a tophat (uniform) redshift distribution.

    This function defines a top-hat (uniform) distribution::

        n(z) = 1    for zmin <= z <= zmax
        n(z) = 0    otherwise

    Args:
        z: Redshift or array of redshifts.
        zmin: Lower edge.
        zmax: Upper edge (must be greater than ``zmin``).
        normalize: If ``True``, normalizes the curve to
            have integral 1 over ``z``.

    Returns:
        Tophat evaluated at ``z`` (normalized if requested).

    Raises:
        ValueError: If ``zmax <= zmin``.
    """
    if zmax <= zmin:
        raise ValueError("``zmax`` must be greater than ``zmin``.")

    z_arr = np.asarray(z, dtype=np.float64)
    nz = ((z_arr >= zmin) & (z_arr <= zmax)).astype(np.float64)
    return _maybe_normalize(z_arr, nz, normalize)


def shifted_smail_distribution(
    z: float | FloatArray,
    z0: float,
    alpha: float,
    beta: float,
    z_shift: float = 0.0,
    *,
    normalize: bool = False,
) -> FloatArray:
    """Returns a shifted Smail-type redshift distribution.

    This function defines a shifted Smail-like shape::

        n(z) = ((z - z_shift)/z0)^alpha * exp(-((z - z_shift)/z0)^beta)  for z >= z_shift
        n(z) = 0                                                        for z < z_shift

    Args:
        z: Redshift or array of redshifts.
        z0: Characteristic redshift scale (must be positive).
        alpha: Power-law index for low redshifts.
        beta: Exponential cutoff index for high redshifts.
        z_shift: Shift applied to redshift (domain cutoff at ``z_shift``).
        normalize: If ``True``, normalizes the curve to
            have integral 1 over ``z``.

    Returns:
        Shifted Smail-like distribution evaluated at ``z``.

    Raises:
        ValueError: If ``z0`` is not positive.
    """
    if z0 <= 0:
        raise ValueError("z0 must be positive.")

    z_arr = np.asarray(z, dtype=np.float64)
    out = np.zeros_like(z_arr, dtype=np.float64)
    m = z_arr >= z_shift
    zz = (z_arr[m] - z_shift) / z0
    out[m] = (zz**alpha) * exp(-(zz**beta))
    return _maybe_normalize(z_arr, out, normalize)


def skew_normal_distribution(
    z: float | FloatArray,
    xi: float,
    omega: float,
    alpha: float,
    *,
    normalize: bool = False,
) -> FloatArray:
    """Returns a skew-normal-like redshift distribution.

    This function defines an unnormalized skew-normal-like shape::

        n(z) = exp[-0.5 * ((z - xi) / omega)^2] * Phi(alpha * (z - xi) / omega)

    where ``Phi`` is the standard normal CDF approximated via ``tanh``.
    Parameters are ``xi`` (location), ``omega > 0`` (scale), and ``alpha``
    (shape/skewness).

    Args:
        z: Redshift or array of redshifts.
        xi: Location parameter.
        omega: Scale parameter (must be positive).
        alpha: Shape (skewness) parameter.
        normalize: If ``True``, normalizes the curve to have
            integral 1 over ``z``.

    Returns:
        Skew-normal-like shape evaluated at ``z`` (normalized if requested).

    Raises:
        ValueError: If ``omega`` is not positive.
    """
    if omega <= 0:
        raise ValueError("omega must be positive.")

    z_arr = np.asarray(z, dtype=np.float64)
    x = (z_arr - xi) / omega
    u = alpha * x

    # Phi(u) = 0.5 * (1 + erf(u / sqrt(2)))
    phi = 0.5 * (1.0 + np.vectorize(erf)(u / np.sqrt(2.0)))

    nz = np.exp(-0.5 * x**2) * phi
    return _maybe_normalize(z_arr, nz, normalize)


def student_t_distribution(
    z: float | FloatArray,
    mu: float,
    sigma: float,
    nu: float,
    *,
    normalize: bool = False,
) -> FloatArray:
    """Returns a Student-t redshift distribution.

    This function defines an unnormalized Student-t shape::

        n(z) = [1 + ((z - mu) / sigma)^2 / nu]^(-(nu + 1) / 2)

    with parameters ``mu`` (location), ``sigma > 0`` (scale), and ``nu > 0``
    (degrees of freedom).

    Args:
        z: Redshift or array of redshifts.
        mu: Location parameter.
        sigma: Scale parameter (must be positive).
        nu: Degrees of freedom (must be positive).
        normalize: If ``True``, normalizes the curve to have integral 1 over ``z``.

    Returns:
        Student-t shape evaluated at ``z`` (normalized if requested).

    Raises:
        ValueError: If ``sigma`` or ``nu`` is not positive.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if nu <= 0:
        raise ValueError("nu must be positive.")

    z_arr = np.asarray(z, dtype=np.float64)
    t = (z_arr - mu) / sigma
    nz = (1.0 + (t**2) / nu) ** (-(nu + 1.0) / 2.0)
    return _maybe_normalize(z_arr, nz, normalize)


def tabulated_distribution(
    z: float | FloatArray,
    z_input: FloatArray,
    nz_input: FloatArray,
    *,
    normalize: bool = False,
) -> FloatArray:
    """Returns a tabulated redshift distribution interpolated onto ``z``.

    This function linearly interpolates a tabulated redshift distribution
    onto the requested redshift grid. Values outside the tabulated redshift
    range are set to zero.

    Args:
        z: Redshift or array of redshifts where the distribution is evaluated.
        z_input: Tabulated redshift values. Must be 1D and strictly increasing.
        nz_input: Tabulated distribution values. Must have the same shape as
            ``z_input``.
        normalize: If ``True``, normalizes the interpolated curve to have
            integral 1 over ``z``.

    Returns:
        Tabulated distribution interpolated onto ``z``.

    Raises:
        ValueError: If ``z_table`` and ``nz_table`` are not valid 1D arrays,
            have different shapes, contain fewer than two points, or if
            ``z_table`` is not strictly increasing.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    z_tab = np.asarray(z_input, dtype=np.float64)
    nz_tab = np.asarray(nz_input, dtype=np.float64)

    if z_tab.ndim != 1 or nz_tab.ndim != 1:
        raise ValueError("z_table and nz_table must be 1D arrays.")
    if z_tab.shape != nz_tab.shape:
        raise ValueError("z_table and nz_table must have the same shape.")
    if z_tab.size < 2:
        raise ValueError("z_table and nz_table must contain at least two points.")
    if np.any(np.diff(z_tab) <= 0.0):
        raise ValueError("z_table must be strictly increasing.")

    nz = np.interp(z_arr, z_tab, nz_tab, left=0.0, right=0.0)
    return _maybe_normalize(z_arr, np.asarray(nz, dtype=np.float64), normalize)
