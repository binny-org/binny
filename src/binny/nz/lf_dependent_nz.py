"""Luminosity-function-dependent redshift distribution model.

This module defines a redshift distribution built from a luminosity function
by internally converting an apparent-magnitude grid into absolute magnitude,
integrating the LF over magnitude, and weighting by the comoving volume
element.

The intended public interface is a plain function, consistent with Binny's
other nz models.
"""

from __future__ import annotations

import numpy as np

from binny.utils.normalization import normalize_over_z
from binny.utils.types import FloatArray

__all__ = ["lf_nz_model"]


def _distance_modulus_from_luminosity_distance_mpc(d_l_mpc: FloatArray) -> FloatArray:
    """Return distance modulus from luminosity distance in Mpc."""
    d_l_pc = np.asarray(d_l_mpc, dtype=np.float64) * 1.0e6
    return 5.0 * np.log10(d_l_pc / 10.0)


def _absolute_magnitude_grid(
    z: FloatArray,
    m_grid: FloatArray,
    luminosity_distance_mpc_fn,
    k_correction_fn=None,
) -> FloatArray:
    """Return absolute-magnitude grid with shape (Nz, Nm)."""
    z_arr = np.asarray(z, dtype=np.float64)
    m_arr = np.asarray(m_grid, dtype=np.float64)

    d_l = np.asarray(luminosity_distance_mpc_fn(z_arr), dtype=np.float64)
    if d_l.shape != z_arr.shape:
        raise ValueError("luminosity_distance_mpc_fn(z) must return shape (len(z),).")

    mu = _distance_modulus_from_luminosity_distance_mpc(d_l)

    if k_correction_fn is None:
        kcorr = np.zeros_like(z_arr)
    else:
        kcorr = np.asarray(k_correction_fn(z_arr), dtype=np.float64)
        if kcorr.shape != z_arr.shape:
            raise ValueError("k_correction_fn(z) must return shape (len(z),).")

    # M = m - DM - K(z)
    return m_arr[None, :] - mu[:, None] - kcorr[:, None]


def lf_nz_model(
    z: float | FloatArray,
    lf,
    *,
    m_lim: float = 22.0,
    m_min: float = 14.0,
    n_m: int = 512,
    luminosity_distance_mpc_fn,
    comoving_volume_integrand_fn,
    k_correction_fn=None,
    normalize: bool = False,
    **lf_kwargs,
) -> FloatArray:
    """Return an LF-dependent redshift distribution.

    This constructs

        n(z) ∝ [∫ dM Phi(M, z)] * dV/dz

    but uses an internally built apparent-magnitude grid up to ``m_lim``,
    converts it to absolute magnitude, evaluates the luminosity function,
    integrates over magnitude, and weights by the comoving volume integrand.

    Args:
        z:
            Redshift or 1D redshift grid.
        lf:
            Luminosity-function callable with signature
            ``lf(M, z, **lf_kwargs)`` returning shape ``(len(z), len(M_grid))``
            when passed a 2D absolute-magnitude grid ``M`` and 1D ``z``.
        m_lim:
            Apparent magnitude limit. Default is ``22.0``.
        m_min:
            Bright-end lower bound of the internal apparent-magnitude grid.
        n_m:
            Number of apparent-magnitude samples in the internal grid.
        luminosity_distance_mpc_fn:
            Callable returning luminosity distance in Mpc as a function of ``z``.
        comoving_volume_integrand_fn:
            Callable returning the comoving-volume weight as a function of ``z``.
        k_correction_fn:
            Optional callable returning K-correction as a function of ``z``.
            If omitted, zero K-correction is assumed.
        normalize:
            If ``True``, normalize the output over ``z``.
        **lf_kwargs:
            Extra keyword arguments passed to ``lf``.

    Returns:
        Array ``n(z)`` evaluated on the input redshift grid.

    Raises:
        ValueError:
            If the input grid or returned shapes are invalid.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    if z_arr.ndim != 1:
        raise ValueError("z must be a 1D array.")
    if z_arr.size < 2:
        raise ValueError("z must contain at least two points.")
    if n_m < 2:
        raise ValueError("n_m must be at least 2.")
    if m_lim <= m_min:
        raise ValueError("m_lim must be greater than m_min.")

    m_grid = np.linspace(m_min, m_lim, n_m, dtype=np.float64)

    absolute_magnitude = _absolute_magnitude_grid(
        z_arr,
        m_grid,
        luminosity_distance_mpc_fn=luminosity_distance_mpc_fn,
        k_correction_fn=k_correction_fn,
    )

    phi = np.asarray(lf(absolute_magnitude, z_arr, **lf_kwargs), dtype=np.float64)
    if phi.shape != absolute_magnitude.shape:
        raise ValueError("lf(M, z, ...) must return an array of shape (len(z), n_m).")

    dV_dz = np.asarray(comoving_volume_integrand_fn(z_arr), dtype=np.float64)
    if dV_dz.shape != z_arr.shape:
        raise ValueError("comoving_volume_integrand_fn(z) must return shape (len(z),).")

    nz = np.trapezoid(phi, x=absolute_magnitude, axis=1) * dV_dz

    if normalize:
        nz = normalize_over_z(z_arr, nz)

    return nz
