"""Module defining various redshift distributions for astronomical sources."""

from __future__ import annotations

import numpy as np
from numpy import exp
from numpy.typing import NDArray

__all__ = [
    "smail_distribution",
]


def smail_distribution(
    z: float | NDArray[np.floating],
    z0: float,
    alpha: float,
    beta: float,
) -> NDArray[np.floating]:
    """Smail-type distribution.

    This is a standard parametric form for redshift distributions of
    astronomical sources, often used in weak lensing and galaxy surveys.
    The form is given by:
        ``n(z) = (z / z0)^alpha * exp[-(z / z0)^beta]``

    Args:
        z: Redshift or array of redshifts.
        z0: Characteristic redshift scale.
        alpha: Power-law index for low redshifts.
        beta: Exponential cutoff index for high redshifts.

    Returns:
        The value of the Smail distribution at the given redshift(s).
    """
    z_arr = np.asarray(z, dtype=float)
    smail_distr = (z_arr / z0) ** alpha * exp(-((z_arr / z0) ** beta))
    return smail_distr
