"""Normalization utilities for 1D data arrays."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import simpson

__all__ = ["normalize_1d"]


def normalize_1d(
    x: Any,
    y: Any,
    method: Literal["trapezoid", "simpson"] = "trapezoid",
) -> NDArray[np.float64]:
    """Returns y normalized so that the integral over x is 1."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if method == "simpson":
        norm = simpson(y_arr, x=x_arr)
    elif method == "trapezoid":
        norm = np.trapezoid(y_arr, x=x_arr)
    else:
        raise ValueError("method must be 'trapezoid' or 'simpson'.")

    if np.isclose(norm, 0.0, atol=1e-10):
        raise ValueError("Normalization factor too small / zero.")

    return y_arr / norm
