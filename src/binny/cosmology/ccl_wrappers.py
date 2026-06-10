"""PyCCL cosmology helpers for Binny.

This module provides small CCL-backed helpers for distances and volume weights.
They are useful for LF-dependent redshift distributions and other
cosmology-aware tomography extensions.
"""

from __future__ import annotations

import numpy as np
import pyccl as ccl

from binny.utils.types import FloatArray

__all__ = [
    "scale_factor_from_redshift",
    "comoving_radial_distance_mpc",
    "luminosity_distance_mpc",
    "comoving_volume_weight",
]


def scale_factor_from_redshift(z: FloatArray) -> FloatArray:
    """Return scale factor from redshift.

    Args:
        z: Redshift value or array.

    Returns:
        Scale factor values.

    Raises:
        ValueError: If any redshift is negative.
    """
    z_arr = np.asarray(z, dtype=np.float64)

    if np.any(z_arr < 0.0):
        raise ValueError("z must be non-negative.")

    return np.asarray(1.0 / (1.0 + z_arr), dtype=np.float64)


def comoving_radial_distance_mpc(
    cosmo: ccl.Cosmology,
    z: FloatArray,
) -> FloatArray:
    """Return comoving radial distance in Mpc.

    Args:
        cosmo: PyCCL cosmology object.
        z: Redshift value or array.

    Returns:
        Comoving radial distance in Mpc.
    """
    a = scale_factor_from_redshift(z)
    return np.asarray(ccl.comoving_radial_distance(cosmo, a), dtype=np.float64)


def luminosity_distance_mpc(
    cosmo: ccl.Cosmology,
    z: FloatArray,
) -> FloatArray:
    """Return luminosity distance in Mpc.

    Args:
        cosmo: PyCCL cosmology object.
        z: Redshift value or array.

    Returns:
        Luminosity distance in Mpc.
    """
    a = scale_factor_from_redshift(z)
    return np.asarray(ccl.luminosity_distance(cosmo, a), dtype=np.float64)


def comoving_volume_weight(
    cosmo: ccl.Cosmology,
    z: FloatArray,
) -> FloatArray:
    r"""Return comoving volume element per redshift and steradian.

    PyCCL's ``comoving_volume_element`` returns the comoving volume element
    per unit scale factor and steradian,

    .. math::

        \frac{dV}{da\,d\Omega}.

    This helper converts it to a redshift weight using

    .. math::

        \frac{dV}{dz\,d\Omega}
        =
        \left|\frac{dV}{da\,d\Omega}\right|
        \left|\frac{da}{dz}\right|
        =
        \left|\frac{dV}{da\,d\Omega}\right| a^2.

    Args:
        cosmo: PyCCL cosmology object.
        z: Redshift value or array.

    Returns:
        Comoving volume element per redshift and steradian in Mpc^3.
    """
    z_arr = np.asarray(z, dtype=np.float64)
    a = scale_factor_from_redshift(z_arr)

    volume_per_scale_factor = np.asarray(
        ccl.comoving_volume_element(cosmo, a),
        dtype=np.float64,
    )

    return np.asarray(np.abs(volume_per_scale_factor) * a**2, dtype=np.float64)
