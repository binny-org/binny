"""Luminosity function-dependent redshift distribution model.

This module defines a parent redshift distribution from a luminosity function.
It converts an apparent magnitude grid into absolute magnitude, evaluates the
luminosity function on that grid, integrates over apparent magnitude, and
weights the result by a redshift-dependent volume factor.

The model is intentionally backend-agnostic: distances, volume weights,
K-corrections, and luminosity functions are supplied as callables. This keeps
the interface compatible with different cosmology backends and LF
implementations, including LFKit ``LuminosityFunction`` objects.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from binny.cosmology.ccl_wrappers import (
    comoving_volume_weight,
    luminosity_distance_mpc,
)
from binny.utils.normalization import normalize_over_z
from binny.utils.types import FloatArray

__all__ = ["luminosity_function_distribution"]


def _distance_modulus_from_luminosity_distance_mpc(
    luminosity_distance_mpc: FloatArray,
) -> FloatArray:
    """Return distance modulus from luminosity distance in Mpc.

    Args:
        luminosity_distance_mpc: Luminosity distance in Mpc.

    Returns:
        Distance modulus evaluated from the supplied luminosity distance.

    Raises:
        ValueError: If luminosity distance is non-finite or not positive.
    """
    d_l = np.asarray(luminosity_distance_mpc, dtype=np.float64)

    if not np.all(np.isfinite(d_l)):
        raise ValueError("Luminosity distance must contain only finite values.")
    if np.any(d_l <= 0.0):
        raise ValueError("Luminosity distance must be positive.")

    d_l_pc = d_l * 1.0e6
    return 5.0 * np.log10(d_l_pc / 10.0)


def _absolute_magnitude_grid(
    z: FloatArray,
    m_grid: FloatArray,
    luminosity_distance_mpc_fn: Callable[[FloatArray], FloatArray],
    k_correction_fn: Callable[[FloatArray], FloatArray] | None = None,
) -> FloatArray:
    """Return absolute magnitudes on a redshift-apparent-magnitude grid.

    Args:
        z: Redshift grid.
        m_grid: Apparent-magnitude grid.
        luminosity_distance_mpc_fn: Callable returning luminosity distance in
            Mpc as a function of redshift.
        k_correction_fn: Optional callable returning K-correction as a function
            of redshift.

    Returns:
        Two-dimensional absolute-magnitude grid with shape ``(len(z), len(m_grid))``.

    Raises:
        ValueError: If distance or K-correction callables return incompatible
            shapes or invalid values.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    m_arr = np.asarray(m_grid, dtype=np.float64)

    d_l = np.asarray(luminosity_distance_mpc_fn(z_arr), dtype=np.float64)
    if d_l.shape != z_arr.shape:
        raise ValueError("luminosity_distance_mpc_fn(z) must return shape (len(z),).")

    distance_modulus = _distance_modulus_from_luminosity_distance_mpc(d_l)

    if k_correction_fn is None:
        k_correction = np.zeros_like(z_arr)
    else:
        k_correction = np.asarray(k_correction_fn(z_arr), dtype=np.float64)
        if k_correction.shape != z_arr.shape:
            raise ValueError("k_correction_fn(z) must return shape (len(z),).")
        if not np.all(np.isfinite(k_correction)):
            raise ValueError("k_correction_fn(z) must return only finite values.")

    return m_arr[None, :] - distance_modulus[:, None] - k_correction[:, None]


def _as_lf_callable(
    lf: Any,
) -> tuple[Callable[..., FloatArray], bool]:
    """Return an LF callable and whether it expects two-dimensional redshift input.

    LFKit ``LuminosityFunction`` objects expose ``_as_callable`` and/or
    ``phi``. Those interfaces broadcast naturally when redshift is supplied as
    ``z[:, None]`` against the two-dimensional absolute-magnitude grid.

    Plain callables are left unchanged so existing Binny callables continue to
    receive the original one-dimensional redshift grid.

    Args:
        lf: Luminosity-function callable or LFKit-style object.

    Returns:
        A callable luminosity function and a flag indicating whether redshift
        should be passed as ``z[:, None]``.
    """
    if hasattr(lf, "_as_callable"):
        return lf._as_callable(), True

    if hasattr(lf, "phi"):
        return lambda absolute_mag, z: lf.phi(absolute_mag, z), True

    return lf, False


def luminosity_function_distribution(
    z: FloatArray,
    lf: Callable[..., FloatArray],
    *,
    cosmo: Any | None = None,
    m_lim: float = 22.0,
    m_bright: float = 14.0,
    n_m: int = 512,
    luminosity_distance_mpc_fn: Callable[[FloatArray], FloatArray] | None = None,
    volume_weight_fn: Callable[[FloatArray], FloatArray] | None = None,
    k_correction_fn: Callable[[FloatArray], FloatArray] | None = None,
    normalize: bool = False,
    **lf_kwargs: Any,
) -> FloatArray:
    """Return a luminosity function-weighted parent redshift distribution.

    This constructs a parent redshift distribution proportional to

    .. math::

        n(z) \\propto W_V(z)
        \\int_{m_{\\rm bright}}^{m_{\\rm lim}} \\Phi(M(m, z), z)\\, dm,

    where ``W_V(z)`` is the redshift-dependent volume weight. Apparent
    magnitudes are converted to absolute magnitudes using the luminosity
    distance and optional K-correction.

    Args:
        z:
            One-dimensional redshift grid.
        lf:
            Luminosity-function callable or LFKit ``LuminosityFunction`` object.
            Plain callables must accept ``lf(M, z, **kwargs)``. LFKit objects
            are evaluated through their ``_as_callable`` or ``phi`` interface.
        cosmo:
            Optional PyCCL cosmology object. If supplied, Binny uses its
            CCL-backed luminosity-distance and comoving-volume helpers whenever
            explicit helper callables are not provided.
        m_lim:
            Faint-end apparent-magnitude limit.
        m_bright:
            Bright-end apparent-magnitude bound of the integration grid.
        n_m:
            Number of apparent-magnitude samples used for the magnitude
            integral.
        luminosity_distance_mpc_fn:
            Optional callable returning luminosity distance in Mpc as a
            function of redshift. If omitted, ``cosmo`` must be supplied.
        volume_weight_fn:
            Optional callable returning the redshift-dependent volume weight.
            If omitted, ``cosmo`` must be supplied.
        k_correction_fn:
            Optional callable returning K-correction as a function of redshift.
            If omitted, zero K-correction is assumed.
        normalize:
            If ``True``, normalize the output over the redshift grid.
        **lf_kwargs:
            Extra keyword arguments passed directly to plain LF callables.

    Returns:
        Parent redshift distribution evaluated on ``z``.

    Raises:
        ValueError:
            If inputs are invalid, neither ``cosmo`` nor the required helper
            callables are supplied, or supplied callables return arrays with
            incompatible shapes or invalid values.
    """
    z_arr = np.asarray(z, dtype=np.float64)

    if z_arr.ndim != 1:
        raise ValueError("z must be a 1D array.")
    if z_arr.size < 2:
        raise ValueError("z must contain at least two points.")
    if not np.all(np.isfinite(z_arr)):
        raise ValueError("z must contain only finite values.")
    if np.any(z_arr < 0.0):
        raise ValueError("z must be non-negative.")

    if n_m < 2:
        raise ValueError("n_m must be at least 2.")
    if m_lim <= m_bright:
        raise ValueError("m_lim must be greater than m_bright.")

    if cosmo is None and (luminosity_distance_mpc_fn is None or volume_weight_fn is None):
        raise ValueError(
            "Either cosmo or both luminosity_distance_mpc_fn and volume_weight_fn must be supplied."
        )

    if luminosity_distance_mpc_fn is None:

        def luminosity_distance_mpc_fn(z_eval: FloatArray) -> FloatArray:
            return luminosity_distance_mpc(cosmo, z_eval)

    if volume_weight_fn is None:

        def volume_weight_fn(z_eval: FloatArray) -> FloatArray:
            return comoving_volume_weight(cosmo, z_eval)

    m_grid = np.linspace(m_bright, m_lim, n_m, dtype=np.float64)

    positive = z_arr > 0.0
    nz = np.zeros_like(z_arr, dtype=np.float64)

    if not np.any(positive):
        return nz

    z_positive = z_arr[positive]

    absolute_magnitude = _absolute_magnitude_grid(
        z_positive,
        m_grid,
        luminosity_distance_mpc_fn=luminosity_distance_mpc_fn,
        k_correction_fn=k_correction_fn,
    )

    lf_callable, wants_2d_z = _as_lf_callable(lf)
    z_for_lf = z_positive[:, None] if wants_2d_z else z_positive

    phi = np.asarray(
        lf_callable(absolute_magnitude, z_for_lf, **lf_kwargs),
        dtype=np.float64,
    )

    if phi.shape != absolute_magnitude.shape:
        raise ValueError("lf(M, z, ...) must return an array of shape (len(z), n_m).")
    if not np.all(np.isfinite(phi)):
        raise ValueError("lf(M, z, ...) must return only finite values.")
    if np.any(phi < 0.0):
        raise ValueError("lf(M, z, ...) must return non-negative values.")

    volume_weight = np.asarray(volume_weight_fn(z_positive), dtype=np.float64)

    if volume_weight.shape != z_positive.shape:
        raise ValueError("volume_weight_fn(z) must return shape (len(z),).")
    if not np.all(np.isfinite(volume_weight)):
        raise ValueError("volume_weight_fn(z) must return only finite values.")
    if np.any(volume_weight < 0.0):
        raise ValueError("volume_weight_fn(z) must return non-negative values.")

    nz[positive] = np.trapezoid(phi, x=m_grid, axis=1) * volume_weight

    if normalize:
        nz = normalize_over_z(z_arr, nz)

    return nz
