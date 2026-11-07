"""Generic selection-based binning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, overload

import numpy as np

from binny.nz_tomo.binning_core import (
    build_bins_on_edges,
    validate_bin_edges,
)
from binny.utils.types import FloatArray
from binny.utils.validators import validate_axis_and_weights

__all__ = ["build_bins"]


@overload
def build_bins(
    z: FloatArray,
    nz: FloatArray,
    edges: FloatArray,
    assignment: Callable[[FloatArray, FloatArray], FloatArray],
    *,
    selection_kind: Literal["probability", "weight"] = "probability",
    normalize: bool = False,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    return_integrals: Literal[False] = False,
) -> dict[int, FloatArray]: ...


@overload
def build_bins(
    z: FloatArray,
    nz: FloatArray,
    edges: FloatArray,
    assignment: Callable[[FloatArray, FloatArray], FloatArray],
    *,
    selection_kind: Literal["probability", "weight"] = "probability",
    normalize: bool = False,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    return_integrals: Literal[True],
) -> tuple[dict[int, FloatArray], dict[int, float], float]: ...


def build_bins(
    z: FloatArray,
    nz: FloatArray,
    edges: FloatArray,
    assignment: Callable[[FloatArray, FloatArray], FloatArray],
    *,
    selection_kind: Literal["probability", "weight"] = "probability",
    normalize: bool = False,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    return_integrals: bool = False,
) -> dict[int, FloatArray] | tuple[dict[int, FloatArray], dict[int, float], float]:
    """Build tomographic bins from selection probabilities or weights.

    Args:
        z: Parent redshift grid.
        nz: Parent redshift distribution.
        edges: Bin edges in the selected coordinate.
        assignment: Callable returning a selection array with shape
            ``(n_bins, n_z)``.
        selection_kind: Whether the selection represents probabilities
            in [0, 1] or arbitrary non-negative weights.
        normalize: Whether to normalize each returned bin.
        norm_method: Numerical integration method for normalization.
        return_integrals: Whether to also return pre-normalization
            bin integrals and the parent integral.

    Returns:
        Dictionary mapping bin indices to redshift distributions.
        If ``return_integrals=True``, also returns the raw bin
        integrals and the parent integral.
    """
    z_arr, nz_arr = validate_axis_and_weights(z, nz)
    edges_arr = validate_bin_edges(edges)

    if selection_kind not in {"probability", "weight"}:
        raise ValueError("selection_kind must be 'probability' or 'weight'.")

    selection = np.asarray(
        assignment(z_arr, edges_arr),
        dtype=np.float64,
    )

    expected_shape = (edges_arr.size - 1, z_arr.size)
    if selection.shape != expected_shape:
        raise ValueError(f"assignment returned shape {selection.shape}, expected {expected_shape}.")

    if not np.all(np.isfinite(selection)):
        raise ValueError("Selection values must be finite.")

    if np.any(selection < 0.0):
        raise ValueError("Selection values must be non-negative.")

    if selection_kind == "probability" and np.any(selection > 1.0):
        raise ValueError("Selection probabilities must not exceed 1.")

    bins, bin_integrals, parent_integral = build_bins_on_edges(
        z=z_arr,
        nz_parent_for_meta=nz_arr,
        bin_edges=edges_arr,
        raw_bin_for_edge=lambda i, a, b: nz_arr * selection[i],
        normalize_bins=normalize,
        norm_method=norm_method,
        need_meta=return_integrals,
    )

    if not return_integrals:
        return bins

    return bins, bin_integrals, parent_integral
