"""Generic selection-based binning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, overload

import numpy as np

from binny.binning.edges import validate_bin_edges
from binny.utils.normalization import normalize_1d
from binny.utils.types import FloatArray
from binny.utils.validators import validate_axis_and_weights

__all__ = ["build_bins", "build_bins_on_edges"]


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


def build_bins_on_edges(
    *,
    z: FloatArray,
    nz_parent_for_meta: FloatArray,
    bin_edges: FloatArray,
    raw_bin_for_edge: Callable[[int, float, float], FloatArray],
    normalize_bins: bool,
    norm_method: Literal["trapezoid", "simpson"],
    mixer: Callable[[dict[int, FloatArray]], dict[int, FloatArray]] | None = None,
    need_meta: bool,
) -> tuple[dict[int, FloatArray], dict[int, float] | None, float | None]:
    """Builds tomographic bins on fixed edges.

    Args:
        z: Redshift grid for all returned curves.
        nz_parent_for_meta: Parent distribution used for metadata.
        bin_edges: Validated 1D array of bin edges.
        raw_bin_for_edge: Callable that returns the raw bin curve for a bin
            index and edge pair.
        normalize_bins: Whether to normalize each bin curve.
        norm_method: Normalization method used when `normalize_bins=True`.
        mixer: Optional transformation applied to the set of raw bins.
        need_meta: Whether to compute per-bin norms and the parent norm.

    Returns:
        A tuple `(bins, bins_norms, parent_norm)`, where `bins` is a mapping
        of bin index to bin curve, `bins_norms` is the per-bin integral (or
        `None`), and `parent_norm` is the parent integral (or `None`).

    Raises:
        ValueError: If a raw bin curve does not match the shape of `z`.
    """
    z_arr = np.asarray(z, dtype=float)
    parent_arr = np.asarray(nz_parent_for_meta, dtype=float)

    n_bins = int(bin_edges.size - 1)

    # Build raw bins first
    raw_bins: dict[int, FloatArray] = {}
    for i, (a, b) in enumerate(zip(bin_edges[:-1], bin_edges[1:], strict=False)):
        raw = raw_bin_for_edge(i, float(a), float(b))
        arr = np.asarray(raw, dtype=np.float64)
        if arr.shape != z_arr.shape:
            raise ValueError(
                f"raw_bin_for_edge returned shape {arr.shape}, expected {z_arr.shape}."
            )
        raw_bins[i] = arr

    # Optional mixing (spec-z)
    if mixer is not None:
        raw_bins = mixer(raw_bins)

    bins_norms: dict[int, float] | None = {} if need_meta else None
    parent_norm: float | None = float(np.trapezoid(parent_arr, x=z_arr)) if need_meta else None

    # Track norms and optionally normalize
    out: dict[int, FloatArray] = {}
    for i in range(n_bins):
        arr = raw_bins[i]

        area = float(np.trapezoid(arr, x=z_arr))
        if bins_norms is not None:
            bins_norms[i] = area

        if normalize_bins:
            if np.isclose(area, 0.0, atol=1e-12):
                out[i] = arr
                continue
            arr = normalize_1d(z_arr, arr, method=norm_method)

        out[i] = arr

    return out, bins_norms, parent_norm
