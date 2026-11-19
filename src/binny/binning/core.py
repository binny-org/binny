"""Generic selection-based tomographic binning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from binny.binning.edges import validate_bin_edges
from binny.utils.normalization import integrate_1d
from binny.utils.types import FloatArray
from binny.utils.validators import validate_axis_and_weights

__all__ = ["build_bins"]


def build_bins(
    z: FloatArray,
    nz: FloatArray,
    edges: FloatArray,
    assignment: Callable[[FloatArray, FloatArray], FloatArray],
    *,
    selection_kind: Literal["probability", "weight"] = "probability",
    normalize_bins: bool = False,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    return_integrals: bool = False,
) -> dict[int, FloatArray] | tuple[dict[int, FloatArray], dict[int, float], float]:
    """Build tomographic bins using a selection function.

    Args:
        z: Parent redshift grid.
        nz: Parent redshift distribution.
        edges: Bin edges in the selection coordinate.
        assignment: Function returning selections of shape
            ``(n_bins, n_z)``.
        selection_kind: Whether selections are probabilities in
            ``[0, 1]`` or non-negative weights.
        normalize_bins: Whether to normalize each bin.
        norm_method: Integration method for normalization and population integrals.
        return_integrals: Whether to return raw bin integrals
            and the parent integral.

    Returns:
        Dictionary of bin distributions, optionally accompanied
        by bin integrals and the parent integral.

    Raises:
        ValueError: If the selection type, shape, or values are invalid.
    """
    z_arr, nz_arr = validate_axis_and_weights(z, nz)
    edges_arr = validate_bin_edges(edges)

    if selection_kind not in {"probability", "weight"}:
        raise ValueError("selection_kind must be 'probability' or 'weight'.")

    selection = np.asarray(assignment(z_arr, edges_arr), dtype=np.float64)
    expected_shape = (edges_arr.size - 1, z_arr.size)

    if selection.shape != expected_shape:
        raise ValueError(f"assignment returned shape {selection.shape}, expected {expected_shape}.")

    if not np.all(np.isfinite(selection)):
        raise ValueError("Selection values must be finite.")

    if np.any(selection < 0.0):
        raise ValueError("Selection values must be non-negative.")

    if selection_kind == "probability" and np.any(selection > 1.0):
        raise ValueError("Selection probabilities must not exceed 1.")

    bins: dict[int, FloatArray] = {}
    bin_integrals: dict[int, float] = {}

    for i, probability in enumerate(selection):
        raw = nz_arr * probability
        area = integrate_1d(z_arr, raw, method=norm_method)

        if return_integrals:
            bin_integrals[i] = area

        if normalize_bins and area > 0.0:
            raw = raw / area

        bins[i] = raw

    if return_integrals:
        parent_integral = integrate_1d(z_arr, nz_arr, method=norm_method)
        return bins, bin_integrals, parent_integral

    return bins
