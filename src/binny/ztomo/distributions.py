"""Module defining various redshift distributions for astronomical sources."""

import numpy as np
from numpy import exp

__all__ = [
    "smail_distribution",
]

def smail_distribution(z, z0, alpha, beta):
    """Smail-type distribution N(z) = (z/z0)^alpha * exp[-(z/z0)^beta]."""
    z = np.asarray(z)
    return (z / z0) ** alpha * exp(-(z / z0) ** beta)
