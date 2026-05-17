"""Luminosity-function redshift-distribution models.

This module provides Binny redshift-distribution models that build ``n(z)``
from LFKit luminosity-function objects. The luminosity-function physics and
magnitude-limit integration are delegated to LFKit. Binny supplies the redshift
grid and CCL-backed cosmology helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyccl as ccl
from lfkit import LuminosityFunction

from binny.cosmology.ccl_wrappers import (
    comoving_volume_weight,
    luminosity_distance_mpc,
)
from binny.utils.types import FloatArray

if TYPE_CHECKING:
    from lfkit.api.corrections import Corrections


__all__ = [
    "lf_nz_model",
]


def lf_nz_model(
    z: FloatArray,
    lf: LuminosityFunction,
    *,
    cosmo: ccl.Cosmology,
    m_lim: float,
    m_bright: float,
    n_m: int = 512,
    corrections: Corrections | None = None,
    normalize: bool = True,
) -> FloatArray:
    r"""Return an LF-weighted redshift distribution.

    This computes an LF-weighted redshift density of the form

    .. math::

        n(z) \propto
        \frac{dV}{dz\,d\Omega}
        \int_{M_{\mathrm{bright}}}^{M_{\mathrm{lim}}(z)}
        \phi(M, z)\,dM,

    where ``M_lim(z)`` is set by the apparent-magnitude limit ``m_lim`` and
    the luminosity distance from the supplied PyCCL cosmology.

    Args:
        z: Redshift grid.
        lf: LFKit ``LuminosityFunction`` object.
        cosmo: PyCCL cosmology object.
        m_lim: Apparent-magnitude limit.
        m_bright: Bright absolute-magnitude integration bound.
        n_m: Number of absolute-magnitude grid points used by LFKit.
        corrections: Optional LFKit corrections object.
        normalize: If True, normalize the returned curve to integrate to one
            over ``z``.

    Returns:
        LF-weighted redshift distribution evaluated on ``z``.
    """
    return np.asarray(
        lf.lf_weighted_redshift_density(
            z,
            m_lim=m_lim,
            m_bright=m_bright,
            n_m=n_m,
            luminosity_distance_mpc_fn=lambda z_eval: luminosity_distance_mpc(
                cosmo,
                z_eval,
            ),
            volume_weight_fn=lambda z_eval: comoving_volume_weight(
                cosmo,
                z_eval,
            ),
            corrections=corrections,
            normalize=normalize,
        ),
        dtype=np.float64,
    )
