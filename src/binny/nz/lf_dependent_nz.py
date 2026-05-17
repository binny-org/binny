"""Luminosity-function-dependent redshift distribution model.

This module defines a redshift distribution built from a luminosity function.
The model converts an apparent-magnitude grid into absolute magnitude,
evaluates the luminosity function, integrates over magnitude, and weights the
result by a redshift-dependent volume factor.

The model is intentionally backend-agnostic: distances, volume weights,
K-corrections, and luminosity functions are supplied as callables. This keeps
the redshift-distribution interface compatible with different cosmology and LF
implementations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from binny.utils.normalization import normalize_over_z
from binny.utils.types import FloatArray

__all__ = ["lf_nz_model"]


def _distance_modulus_from_luminosity_distance_mpc(
    luminosity_distance_mpc: FloatArray,
) -> FloatArray:
    """Return distance modulus from luminosity distance in Mpc.

    Args:
        luminosity_distance_mpc:
            Luminosity distance in Mpc.

    Returns:
        Distance modulus evaluated at each input distance.

    Raises:
        ValueError:
            If any luminosity distance is non-finite or non-positive.
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
    """Return absolute magnitudes for a redshift and apparent-magnitude grid.

    Args:
        z:
            One-dimensional redshift grid.
        m_grid:
            One-dimensional apparent-magnitude grid.
        luminosity_distance_mpc_fn:
            Callable returning luminosity distance in Mpc as a function of
            redshift.
        k_correction_fn:
            Optional callable returning the K-correction as a function of
            redshift. If omitted, zero K-correction is assumed.

    Returns:
        Absolute-magnitude grid with shape ``(len(z), len(m_grid))``.

    Raises:
        ValueError:
            If the supplied callables return arrays with invalid shapes or
            non-finite values.
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


def lf_nz_model(
    z: FloatArray,
    lf: Callable[..., FloatArray],
    *,
    m_lim: float = 22.0,
    m_min: float = 14.0,
    n_m: int = 512,
    luminosity_distance_mpc_fn: Callable[[FloatArray], FloatArray],
    volume_weight_fn: Callable[[FloatArray], FloatArray],
    k_correction_fn: Callable[[FloatArray], FloatArray] | None = None,
    normalize: bool = False,
    **lf_kwargs: Any,
) -> FloatArray:
    """Return an LF-dependent redshift distribution.

    This constructs a redshift distribution proportional to

    .. math::

        n(z) \\propto W_V(z) \\int \\Phi(M, z)\\, dM,

    where ``W_V(z)`` is the redshift-dependent volume weight. The absolute
    magnitude range is determined from an internal apparent-magnitude grid and
    the supplied luminosity-distance relation.

    Args:
        z:
            One-dimensional redshift grid.
        lf:
            Luminosity-function callable. It must accept an absolute-magnitude
            grid with shape ``(len(z), n_m)`` and the one-dimensional redshift
            grid, then return LF values with the same shape.
        m_lim:
            Faint-end apparent-magnitude limit.
        m_min:
            Bright-end lower bound of the internal apparent-magnitude grid.
        n_m:
            Number of apparent-magnitude samples used for the magnitude
            integral.
        luminosity_distance_mpc_fn:
            Callable returning luminosity distance in Mpc as a function of
            redshift.
        volume_weight_fn:
            Callable returning the redshift-dependent volume weight. This can
            be a comoving-volume element, a survey-area-weighted volume
            element, or any equivalent redshift-dependent weight. If
            ``normalize=True``, constant normalization factors cancel.
        k_correction_fn:
            Optional callable returning the K-correction as a function of
            redshift. If omitted, zero K-correction is assumed.
        normalize:
            If ``True``, normalize the output over the redshift grid.
        **lf_kwargs:
            Extra keyword arguments passed directly to ``lf``.

    Returns:
        Redshift distribution evaluated on ``z``.

    Raises:
        ValueError:
            If inputs are invalid or the supplied callables return arrays with
            incompatible shapes.
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
    if not np.all(np.isfinite(phi)):
        raise ValueError("lf(M, z, ...) must return only finite values.")
    if np.any(phi < 0.0):
        raise ValueError("lf(M, z, ...) must return non-negative values.")

    volume_weight = np.asarray(volume_weight_fn(z_arr), dtype=np.float64)
    if volume_weight.shape != z_arr.shape:
        raise ValueError("volume_weight_fn(z) must return shape (len(z),).")
    if not np.all(np.isfinite(volume_weight)):
        raise ValueError("volume_weight_fn(z) must return only finite values.")
    if np.any(volume_weight < 0.0):
        raise ValueError("volume_weight_fn(z) must return non-negative values.")

    nz = np.trapezoid(phi, x=m_grid, axis=1) * volume_weight

    if normalize:
        nz = normalize_over_z(z_arr, nz)

    return nz
