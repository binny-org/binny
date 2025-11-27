"""Normalization utilities for 1D data arrays."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import simpson

__all__ = ["normalize_1d"]


def normalize_1d(
    x: ArrayLike,
    y: ArrayLike,
    method: str = "trapz",
) -> np.ndarray:
    """Return y normalized so that ∫ y(x) dx = 1."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if method == "simpson":
        norm = simpson(y_arr, x=x_arr)
    elif method == "trapz":
        norm = np.trapezoid(y_arr, x_arr)
    else:
        raise ValueError("method must be 'trapz' or 'simpson'.")

    if np.isclose(norm, 0.0, atol=1e-10):
        raise ValueError("Normalization factor too small / zero.")

    return y_arr / norm
