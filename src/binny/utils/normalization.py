"""Normalization utilities for 1D data arrays."""

import numpy as np
from scipy.integrate import simpson

__all__ = [
    "normalize_1d",
]

def normalize_1d(z, nz, method="trapz"):
    z = np.asarray(z)
    nz = np.asarray(nz)

    if method == "simpson":
        norm = simpson(nz, x=z)
    elif method == "trapz":
        norm = np.trapz(nz, x=z)
    else:
        raise ValueError("method must be 'trapz' or 'simpson'.")

    if np.isclose(norm, 0, atol=1e-10):
        raise ValueError("Normalization factor too small / zero.")
    return nz / norm
