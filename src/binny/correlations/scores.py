"""Per-index score utilities for curves on a shared grid.

These helpers reduce each curve to a single scalar per index (e.g. peak, mean,
median, or a credible width). The returned dictionaries can be used as inputs
to tuple filters that compare indices via these scores.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from binny.utils.types import FloatArray1D

__all__ = [
    "score_peak_location",
    "score_mean_location",
    "score_median_location",
    "score_credible_width",
]


def score_peak_location(
    *,
    z: FloatArray1D,
    curves: Mapping[int, FloatArray1D],
) -> dict[int, float]:
    """Compute peak locations for a collection of curves on a shared grid.

    The peak location is defined as the grid coordinate corresponding to the
    maximum value of each curve.

    Args:
        z: One-dimensional coordinate grid.
        curves: Mapping from index to curve values evaluated on z.

    Returns:
        Mapping from index to peak location along the grid.

    Raises:
        ValueError: If a curve does not have the same shape as z.
    """
    zz = np.asarray(z, dtype=float)
    out: dict[int, float] = {}
    for k, c in curves.items():
        cc = np.asarray(c, dtype=float)
        if cc.shape != zz.shape:
            raise ValueError(f"curves[{k}] must have same shape as z.")
        out[int(k)] = float(zz[int(np.argmax(cc))])
    return out


def score_mean_location(
    *,
    z: FloatArray1D,
    curves: Mapping[int, FloatArray1D],
) -> dict[int, float]:
    """Compute mean locations for a collection of curves on a shared grid.

    The mean location is defined as the first moment of each curve with respect
    to the grid coordinate.

    Args:
        z: One-dimensional coordinate grid.
        curves: Mapping from index to curve values evaluated on z.

    Returns:
        Mapping from index to mean location values.

    Raises:
        ValueError: If a curve does not have the same shape as z.
    """
    zz = np.asarray(z, dtype=float)
    out: dict[int, float] = {}
    for k, c in curves.items():
        cc = np.asarray(c, dtype=float)
        if cc.shape != zz.shape:
            raise ValueError(f"curves[{k}] must have same shape as z.")
        norm = float(np.trapezoid(cc, zz))
        if norm == 0.0:
            out[int(k)] = float("nan")
            continue
        out[int(k)] = float(np.trapezoid(zz * cc, zz) / norm)
    return out


def score_median_location(
    *,
    z: FloatArray1D,
    curves: Mapping[int, FloatArray1D],
) -> dict[int, float]:
    """Compute median locations for a collection of curves on a shared grid.

    The median location is defined as the grid coordinate at which the
    cumulative integrated curve reaches one half of its total area.

    Args:
        z: One-dimensional coordinate grid.
        curves: Mapping from index to curve values evaluated on z.

    Returns:
        Mapping from index to median location values.

    Raises:
        ValueError: If z has fewer than two points or curve shapes are invalid.
    """
    zz = np.asarray(z, dtype=float)
    out: dict[int, float] = {}
    for k, c in curves.items():
        cc = np.asarray(c, dtype=float)
        if cc.shape != zz.shape:
            raise ValueError(f"curves[{k}] must have same shape as z.")
        dz = np.diff(zz)
        if dz.size == 0:
            raise ValueError("z must have at least 2 points.")
        area = 0.5 * (cc[:-1] + cc[1:]) * dz
        total = float(np.sum(area))
        if total == 0.0:
            out[int(k)] = float("nan")
            continue
        cdf = np.concatenate([[0.0], np.cumsum(area) / total])
        out[int(k)] = float(np.interp(0.5, cdf, zz))
    return out


def score_credible_width(
    *,
    z: FloatArray1D,
    curves: Mapping[int, FloatArray1D],
    mass: float = 0.68,
) -> dict[int, float]:
    """Compute central credible widths for a collection of curves.

    The credible width is defined as the width of the central interval
    containing the specified fraction of the total integrated curve area.

    Args:
        z: One-dimensional coordinate grid.
        curves: Mapping from index to curve values evaluated on z.
        mass: Fraction of total area contained within the interval.

    Returns:
        Mapping from index to credible interval widths.

    Raises:
        ValueError: If mass is not in the open interval (0, 1) or curve shapes
            are invalid.
    """
    if not (0.0 < mass < 1.0):
        raise ValueError("mass must be in (0, 1).")
    zz = np.asarray(z, dtype=float)
    out: dict[int, float] = {}
    lo_q = (1.0 - mass) / 2.0
    hi_q = 1.0 - lo_q

    for k, c in curves.items():
        cc = np.asarray(c, dtype=float)
        if cc.shape != zz.shape:
            raise ValueError(f"curves[{k}] must have same shape as z.")
        dz = np.diff(zz)
        area = 0.5 * (cc[:-1] + cc[1:]) * dz
        total = float(np.sum(area))
        if total == 0.0:
            out[int(k)] = float("nan")
            continue
        cdf = np.concatenate([[0.0], np.cumsum(area) / total])
        z_lo = float(np.interp(lo_q, cdf, zz))
        z_hi = float(np.interp(hi_q, cdf, zz))
        out[int(k)] = z_hi - z_lo
    return out
