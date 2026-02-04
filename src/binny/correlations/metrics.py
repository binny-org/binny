"""Pairwise metrics for curve comparisons.

These helpers build (i, j) -> scalar callables from collections of curves. Even
when used alongside tuple machinery, these metrics are pairwise: they compare
one curve selected by i to one curve selected by j.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from binny.utils.types import FloatArray1D

__all__ = [
    "metric_min_overlap_fraction",
    "metric_overlap_coefficient",
    "metric_from_curves",
]


def _prepare_indexed_curves_and_norms(
    *,
    z: FloatArray1D,
    curves: Sequence[Mapping[int, FloatArray1D]],
    label: str = "curves",
) -> tuple[FloatArray1D, list[dict[int, FloatArray1D]], list[dict[int, float]]]:
    """Prepares curve arrays and trapezoid integrals keyed by integer indices.

    Args:
        z: One-dimensional coordinate grid.
        curves: Sequence of curve mappings, one per index slot.
        label: Base label used in error messages.

    Returns:
        Tuple (zz, arrs, norms) where:
            - zz is the float coordinate grid,
            - arrs[k] maps indices -> float curve arrays for slot k,
            - norms[k] maps indices -> trapezoid integral of the curve for slot k.

    Raises:
        ValueError: If curves is empty or if any curve does not match the shape
            of z.
    """
    zz = np.asarray(z, dtype=float)
    if len(curves) == 0:
        raise ValueError("curves must contain at least one mapping.")

    arrs: list[dict[int, FloatArray1D]] = []
    norms: list[dict[int, float]] = []

    for slot, mapping in enumerate(curves):
        slot_arr: dict[int, FloatArray1D] = {}
        slot_norm: dict[int, float] = {}

        for idx, c in mapping.items():
            cc = np.asarray(c, dtype=float)
            if cc.shape != zz.shape:
                raise ValueError(f"{label}[{slot}][{idx}] must have same shape as z.")
            ii = int(idx)
            slot_arr[ii] = cc
            slot_norm[ii] = float(np.trapezoid(cc, zz))

        arrs.append(slot_arr)
        norms.append(slot_norm)

    return zz, arrs, norms


def _metric_from_curves_n(
    arrs: Sequence[Mapping[int, FloatArray1D]],
    kernel: Callable[..., float],
) -> Callable[..., float]:
    """Builds an index-metric by applying kernel to selected curve arrays.

    Args:
        arrs: Sequence of index -> curve mappings, one per index slot.
        kernel: Callable applied as kernel(curve0, curve1, ...).

    Returns:
        Callable metric function taking N indices and returning a scalar.

    Raises:
        KeyError: If any requested index is missing in its corresponding slot.
        TypeError: If called with the wrong number of indices.
    """

    def _metric(*idx: int) -> float:
        """Evaluate the index metric for a tuple of indices.

        Selects the curve for each slot by the corresponding index and applies
        the provided curve-level kernel. The number of indices must match the
        number of curve slots.

        Args:
            *idx: Indices selecting one curve from each slot.

        Returns:
            Scalar metric value from applying the kernel to the selected curves.

        Raises:
            TypeError: If the number of indices does not match the number of
                slots.
            KeyError: If any index is missing from its slot mapping.
        """
        if len(idx) != len(arrs):
            raise TypeError(f"Expected {len(arrs)} indices, got {len(idx)}.")
        curves: list[FloatArray1D] = []
        for slot, ii in enumerate(idx):
            mapping = arrs[slot]
            if ii not in mapping:
                raise KeyError(f"Missing curves for slot {slot} index {ii}.")
            curves.append(mapping[ii])
        return float(kernel(*curves))

    return _metric


def _min_overlap_integral_kernel(zz: FloatArray1D) -> Callable[..., float]:
    """Return a kernel that computes ∫ min(cs) dzz for curves on grid ``zz``."""

    def _kernel(*cs: FloatArray1D) -> float:
        """Compute the unnormalized overlap integral of the provided curves.

        Args:
            *cs: Curve arrays evaluated on the common grid ``zz``.

        Returns:
            The integral over ``zz`` of the pointwise minimum of the curves.
        """
        m = np.minimum.reduce(cs)
        return float(np.trapezoid(m, zz))

    return _kernel


def metric_min_overlap_fraction(
    *,
    z: FloatArray1D,
    curves: Sequence[Mapping[int, FloatArray1D]],
) -> Callable[..., float]:
    """Construct a normalized minimum-overlap metric for curve tuples.

    The returned metric evaluates the fractional overlap between N curves as the
    integral of their pointwise minimum divided by the product of their
    individual integrals.

    For N=2 this reduces to the usual pairwise overlap fraction. Supplying three
    curve mappings yields a triple-overlap fraction, and so on.

    Args:
        z: One-dimensional coordinate grid.
        curves: Sequence of curve mappings, one per index slot.

    Returns:
        Callable metric function taking N indices and returning a scalar overlap
        fraction.

    Raises:
        ValueError: If curves is empty or if any curve does not match the shape
            of z.
        KeyError: If a requested index is not present in the curve mappings.
        TypeError: If the returned callable is called with the wrong number of
            indices.
    """
    zz, arrs, norms = _prepare_indexed_curves_and_norms(z=z, curves=curves)
    base = _metric_from_curves_n(arrs, _min_overlap_integral_kernel(zz))

    def _metric(*idx: int) -> float:
        """Evaluate the minimum-overlap fraction for a tuple of indices.

        Computes the overlap integral (via the base metric) and normalizes it by
        the product of the individual curve integrals for the selected indices.

        Args:
            *idx: Indices selecting one curve from each slot.

        Returns:
            Overlap fraction in [0, 1] (0 if any selected curve has zero
            integral).

        Raises:
            TypeError: If the number of indices does not match the number of
                slots.
            KeyError: If any index is missing from its slot mapping.
        """
        denom = 1.0
        for slot, ii in enumerate(idx):
            di = norms[slot][ii]  # KeyError if missing.
            if di == 0.0:
                return 0.0
            denom *= di
        return float(base(*idx)) / denom  # base checks arity

    return _metric


def metric_overlap_coefficient(
    *,
    z: FloatArray1D,
    curves: Sequence[Mapping[int, FloatArray1D]],
) -> Callable[..., float]:
    """Construct an overlap coefficient metric for curve tuples.

    The overlap coefficient is defined as the integral of the pointwise minimum
    of N curves divided by the smallest of their individual integrals.

    For N=2 this reduces to the usual overlap coefficient. Supplying three curve
    mappings yields a triple-overlap coefficient, and so on.

    Args:
        z: One-dimensional coordinate grid.
        curves: Sequence of curve mappings, one per index slot.

    Returns:
        Callable metric function taking N indices and returning a scalar overlap
        coefficient.

    Raises:
        ValueError: If curves is empty or if any curve does not match the shape
            of z.
        KeyError: If a requested index is not present in the curve mappings.
        TypeError: If the returned callable is called with the wrong number of
            indices.
    """
    zz, arrs, norms = _prepare_indexed_curves_and_norms(z=z, curves=curves)
    base = _metric_from_curves_n(arrs, _min_overlap_integral_kernel(zz))

    def _metric(*idx: int) -> float:
        """Evaluate the overlap coefficient for a tuple of indices.

        Computes the overlap integral (via the base metric) and normalizes it by
        the smallest of the individual curve integrals for the selected indices.

        Args:
            *idx: Indices selecting one curve from each slot.

        Returns:
            Overlap coefficient in [0, 1] (0 if the minimum integral is zero or
            undefined).

        Raises:
            TypeError: If the number of indices does not match the number of
                slots.
            KeyError: If any index is missing from its slot mapping.
        """
        denom = np.inf
        for slot, ii in enumerate(idx):
            denom = min(denom, norms[slot][ii])  # KeyError if missing.
        if denom == 0.0 or not np.isfinite(denom):
            return 0.0
        return float(base(*idx)) / float(denom)  # base checks arity

    return _metric


def metric_from_curves(
    *,
    curves: Sequence[Mapping[int, FloatArray1D]],
    kernel: Callable[..., float],
) -> Callable[..., float]:
    """Construct an index metric from a curve-level kernel.

    The returned metric evaluates index tuples by applying the provided kernel
    to the corresponding curve arrays.

    With two curve mappings, the callable is pairwise (i, j). With three, it is
    (i, j, k), and so on.

    Args:
        curves: Sequence of curve mappings, one per index slot.
        kernel: Callable computing a scalar value from N curve arrays.

    Returns:
        Callable metric function taking N indices and returning a scalar value.

    Raises:
        KeyError: If a requested index is not present in the curve mappings.
        TypeError: If the returned callable is called with the wrong number of
            indices.
    """
    arrs = [{int(i): np.asarray(c, dtype=float) for i, c in m.items()} for m in curves]
    return _metric_from_curves_n(arrs, kernel)
