"""Module defining various redshift distributions for astronomical sources."""

from __future__ import annotations

import numpy as np
from numpy import exp
from numpy.typing import NDArray

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
]


def smail_like_distribution(
    z: float | NDArray[np.floating],
    z0: float,
    alpha: float,
    beta: float,
) -> NDArray[np.floating]:
    """Returns a Smail-type redshift distribution.

    This function defines an unnormalized Smail distribution for redshift
    distributions, often used to model galaxy redshift distributions in weak
    lensing studies. The form is given by::

        n(z) = (z / z0)^alpha * exp[-(z / z0)^beta]

    Args:
        z: Redshift or array of redshifts.
        z0: Characteristic redshift scale.
        alpha: Power-law index for low redshifts.
        beta: Exponential cutoff index for high redshifts.

    Returns:
        The value of the unnormalized Smail distribution at the given redshift(s).
    """
    z_arr = np.asarray(z, dtype=float)
    smail_distr = (z_arr / z0) ** alpha * exp(-((z_arr / z0) ** beta))
    return smail_distr


def gaussian_distribution(
    z: float | NDArray[np.floating],
    mu: float,
    sigma: float,
) -> NDArray[np.floating]:
    """Returns a Gaussian redshift distribution.

    This function defines an unnormalized Gaussian distribution for redshift
    distributions, given by the form::

        n(z) = exp[-0.5 * ((z - mu) / sigma)^2]

    with ``mu`` as the mean redshift and ``sigma`` as the standard deviation.

    Args:
        z: Redshift or array of redshifts.
        mu: Mean redshift.
        sigma: Standard deviation (must be positive).

    Returns:
        Unnormalized Gaussian evaluated at ``z``.

    Raises:
        ValueError: If ``sigma`` is not positive.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    z_arr = np.asarray(z, dtype=float)
    return exp(-0.5 * ((z_arr - mu) / sigma) ** 2)


def gaussian_mixture_distribution(
    z: float | NDArray[np.floating],
    mus: NDArray[np.floating],
    sigmas: NDArray[np.floating],
    weights: NDArray[np.floating] | None = None,
) -> NDArray[np.floating]:
    """Returns a Gaussian mixture redshift distribution.

    This function defines an unnormalized Gaussian mixture distribution for redshift
    distributions, given by the form::

        n(z) = sum_{i=1}^K w_i * exp[-0.5 * ((z - mu_i) / sigma_i)^2]

    where ``K`` is the number of components, ``mu_i`` and ``sigma_i`` are the
    mean and standard deviation of the i-th Gaussian component, and ``w_i``
    are the nonnegative weights for each component. If no weights are provided,
    equal weights are assumed.

    Args:
        z: Redshift or array of redshifts.
        mus: Array of component means, shape ``(K,)``.
        sigmas: Array of component std devs, shape ``(K,)``. Must be positive.
        weights: Optional nonnegative weights, shape ``(K,)``. If ``None``,
            equal weights are used. Weights do not need to sum to ``1``.

    Returns:
        Unnormalized Gaussian mixture evaluated at ``z``.

    Raises:
        ValueError: If shapes mismatch, any sigma is non-positive,
        or any weight is negative.
    """
    z_arr = np.asarray(z, dtype=float)
    mus_arr = np.asarray(mus, dtype=float)
    sigmas_arr = np.asarray(sigmas, dtype=float)

    if mus_arr.ndim != 1 or sigmas_arr.ndim != 1:
        raise ValueError("mus and sigmas must be 1D arrays.")
    if mus_arr.shape[0] != sigmas_arr.shape[0]:
        raise ValueError("mus and sigmas must have the same length.")
    if np.any(sigmas_arr <= 0):
        raise ValueError("All sigmas must be positive.")

    if weights is None:
        w = np.ones_like(mus_arr, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or w.shape[0] != mus_arr.shape[0]:
            raise ValueError("weights must be a 1D array with the same length as mus.")
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative.")

    zz = z_arr[None, :]
    comps = exp(-0.5 * ((zz - mus_arr[:, None]) / sigmas_arr[:, None]) ** 2)
    return (w[:, None] * comps).sum(axis=0)


def gamma_distribution(
    z: float | NDArray[np.floating],
    k: float,
    theta: float,
) -> NDArray[np.floating]:
    """Returns a gamma-shaped redshift distribution.

    This function defines an unnormalized gamma-shaped distribution for redshift
    distributions, given by the form::

        n(z) = z^(k-1) * exp(-z/theta)

    for ``z >= 0``, with ``n(z) = 0`` for ``z < 0`` and parameters ``k > 0``,
    ``theta > 0``.

    Args:
        z: Redshift or array of redshifts.
        k: Shape parameter (must be positive).
        theta: Scale parameter (must be positive).

    Returns:
        Unnormalized gamma-shaped distribution evaluated at ``z`` (0 for z < 0).

    Raises:
        ValueError: If ``k`` or ``theta`` is not positive.
    """
    if k <= 0:
        raise ValueError("k must be positive.")
    if theta <= 0:
        raise ValueError("theta must be positive.")
    z_arr = np.asarray(z, dtype=float)
    out = np.zeros_like(z_arr, dtype=float)
    m = z_arr >= 0
    out[m] = (z_arr[m] ** (k - 1.0)) * exp(-(z_arr[m] / theta))
    return out


def schechter_like_distribution(
    z: float | NDArray[np.floating],
    z0: float,
    alpha: float,
) -> NDArray[np.floating]:
    """Returns a Schechter-like redshift distribution.

    This function defines an unnormalized Schechter-like distribution for
    redshift distributions, given by the form::

        n(z) = (z / z0)^alpha * exp(-(z / z0))

    It is similar to the Schechter function commonly used in luminosity functions.
    It also reducses to Smail form with ``beta=1``.

    Args:
        z: Redshift or array of redshifts.
        z0: Characteristic redshift scale (must be positive).
        alpha: Power-law index.

    Returns:
        Unnormalized Schechter-like distribution evaluated at ``z``.

    Raises:
        ValueError: If ``z0`` is not positive.
    """
    if z0 <= 0:
        raise ValueError("z0 must be positive.")
    z_arr = np.asarray(z, dtype=float)
    return (z_arr / z0) ** alpha * exp(-(z_arr / z0))


def lognormal_distribution(
    z: float | NDArray[np.floating],
    mu_ln: float,
    sigma_ln: float,
) -> NDArray[np.floating]:
    """Returns a lognormal redshift distribution.

    This function defines an unnormalized lognormal distribution for redshift
    distributions, given by the form::

        n(z) approx (1/z) * exp(- (ln z - mu_ln)^2 / (2 sigma_ln^2))

    with ``n(z) = 0`` for ``z <= 0``. Parameters ``mu_ln`` and ``sigma_ln`` are the
    mean and standard deviation of ``ln(z)``.

    Args:
        z: Redshift or array of redshifts.
        mu_ln: Mean of ``ln(z)``.
        sigma_ln: Std dev of ``ln(z)`` (must be positive).

    Returns:
        Unnormalized lognormal shape evaluated at ``z`` (``0`` for ``z <= 0``).

    Raises:
        ValueError: If ``sigma_ln`` is not positive.
    """
    if sigma_ln <= 0:
        raise ValueError("sigma_ln must be positive.")
    z_arr = np.asarray(z, dtype=float)
    out = np.zeros_like(z_arr, dtype=float)
    m = z_arr > 0
    lnz = np.log(z_arr[m])
    out[m] = (1.0 / z_arr[m]) * exp(-0.5 * ((lnz - mu_ln) / sigma_ln) ** 2)
    return out


def tophat_distribution(
    z: float | NDArray[np.floating],
    zmin: float,
    zmax: float,
) -> NDArray[np.floating]:
    """Returns a tophat (uniform) redshift distribution.

    This function defines a top-hat (uniform) distribution for redshift
    distributions, given by::

        n(z) = 1`` for ``zmin <= z <= zmax

    and ``n(z) = 0`` elsewhere.

    Args:
        z: Redshift or array of redshifts.
        zmin: Lower edge.
        zmax: Upper edge (must be greater than ``zmin``).

    Returns:
        Array equal to 1 where ``zmin <= z <= zmax`` and 0 elsewhere.

    Raises:
        ValueError: If ``zmax <= zmin``.
    """
    if zmax <= zmin:
        raise ValueError("zmax must be greater than zmin.")
    z_arr = np.asarray(z, dtype=float)
    return ((z_arr >= zmin) & (z_arr <= zmax)).astype(float)


def shifted_smail_distribution(
    z: float | NDArray[np.floating],
    z0: float,
    alpha: float,
    beta: float,
    z_shift: float = 0.0,
) -> NDArray[np.floating]:
    """Returns a shifted Smail-type redshift distribution.

    This function defines a shifted version of the Smail distribution,
    given by::

        n(z) = ((z - z_shift)/z0)^alpha * exp(-((z - z_shift)/z0)^beta)

    with ``n(z)=0`` for ``z < z_shift``.

    Args:
        z: Redshift or array of redshifts.
        z0: Characteristic redshift scale (must be positive).
        alpha: Power-law index for low redshifts.
        beta: Exponential cutoff index for high redshifts.
        z_shift: Shift applied to redshift (domain cutoff at ``z_shift``).

    Returns:
        Unnormalized shifted Smail distribution evaluated at ``z``.

    Raises:
        ValueError: If ``z0`` is not positive.
    """
    if z0 <= 0:
        raise ValueError("z0 must be positive.")
    z_arr = np.asarray(z, dtype=float)
    out = np.zeros_like(z_arr, dtype=float)
    m = z_arr >= z_shift
    zz = (z_arr[m] - z_shift) / z0
    out[m] = (zz**alpha) * exp(-(zz**beta))
    return out


def skew_normal_distribution(
    z: float | NDArray[np.floating],
    xi: float,
    omega: float,
    alpha: float,
) -> NDArray[np.floating]:
    """Returns a skew-normal-like redshift distribution.

    This function defines an unnormalized skew-normal-like distribution for redshift
    distributions, given by the form::

        n(z) = exp[-0.5 * ((z - xi) / omega)^2] * Phi(alpha * (z - xi) / omega)

    where ``Phi`` is the standard normal CDF approximated via tanh. Parameters
    are ``xi`` (location), ``omega > 0`` (scale), and ``alpha`` (shape/skewness).

    Args:
        z: Redshift or array of redshifts.
        xi: Location parameter.
        omega: Scale parameter (must be positive).
        alpha: Shape (skewness) parameter.

    Returns:
        Unnormalized skew-normal-like shape evaluated at ``z``.

    Raises:
        ValueError: If ``omega`` is not positive.
    """
    if omega <= 0:
        raise ValueError("omega must be positive.")
    z_arr = np.asarray(z, dtype=float)
    t = (z_arr - xi) / omega
    u = alpha * t
    approx_phi = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (u + 0.044715 * u**3)))
    return exp(-0.5 * t**2) * approx_phi


def student_t_distribution(
    z: float | NDArray[np.floating],
    mu: float,
    sigma: float,
    nu: float,
) -> NDArray[np.floating]:
    """Returns a Student-t redshift distribution.

    This function defines an unnormalized Student-t distribution for redshift
    distributions, given by the form::

        n(z) = [1 + ((z - mu) / sigma)^2 / nu]^(-(nu + 1) / 2)

    with parameters ``mu`` (location), ``sigma > 0`` (scale), and ``nu > 0``
    (degrees of freedom).

    Args:
        z: Redshift or array of redshifts.
        mu: Location parameter.
        sigma: Scale parameter (must be positive).
        nu: Degrees of freedom (must be positive).

    Returns:
        Unnormalized Student-t shape evaluated at ``z``.

    Raises:
        ValueError: If ``sigma`` or ``nu`` is not positive.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    if nu <= 0:
        raise ValueError("nu must be positive.")
    z_arr = np.asarray(z, dtype=float)
    t = (z_arr - mu) / sigma
    return (1.0 + (t**2) / nu) ** (-(nu + 1.0) / 2.0)
