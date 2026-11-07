"""Generic selection-based tomographic binning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np

from binny.binning.edges import validate_bin_edges
from binny.utils.normalization import normalize_1d
from binny.utils.types import FloatArray
from binny.utils.validators import validate_axis_and_weights

__all__ = ["build_bins", "build_bins_on_edges"]


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
    """Build tomographic bins using a selection function.

    Args:
        z: Parent redshift grid.
        nz: Parent redshift distribution.
        edges: Bin edges in the selection coordinate.
        assignment: Function returning selections of shape
            ``(n_bins, n_z)``.
        selection_kind: Whether selections are probabilities in
            ``[0, 1]`` or non-negative weights.
        normalize: Whether to normalize each bin.
        norm_method: Integration method for normalization.
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

    bins, bin_integrals, parent_integral = build_bins_on_edges(
        z=z_arr,
        nz_parent_for_meta=nz_arr,
        bin_edges=edges_arr,
        raw_bin_for_edge=lambda i, a, b: nz_arr * selection[i],
        normalize_bins=normalize,
        norm_method=norm_method,
        need_meta=return_integrals,
    )

    if return_integrals:
        assert bin_integrals is not None
        assert parent_integral is not None
        return bins, bin_integrals, parent_integral

    return bins


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
    """Construct bins from raw distributions on fixed edges.

    Args:
        z: Redshift grid for the returned distributions.
        nz_parent_for_meta: Parent distribution used for metadata integrals.
        bin_edges: Validated bin edges.
        raw_bin_for_edge: Function returning a raw bin distribution
            for a bin index and its lower and upper edges.
        normalize_bins: Whether to normalize each bin.
        norm_method: Integration method for normalization.
        mixer: Optional transformation of the raw bin distributions,
            applied before normalization.
        need_meta: Whether to calculate raw bin and parent integrals.

    Returns:
        Tuple containing the bin distributions, optional raw bin
        integrals, and optional parent integral.

    Raises:
        ValueError: If a raw bin has an incompatible shape.
    """
    z_arr = np.asarray(z, dtype=float)
    parent_arr = np.asarray(nz_parent_for_meta, dtype=float)

    raw_bins: dict[int, FloatArray] = {}

    for i, (lower, upper) in enumerate(zip(bin_edges[:-1], bin_edges[1:], strict=False)):
        raw = np.asarray(
            raw_bin_for_edge(i, float(lower), float(upper)),
            dtype=np.float64,
        )

        if raw.shape != z_arr.shape:
            raise ValueError(
                f"raw_bin_for_edge returned shape {raw.shape}, expected {z_arr.shape}."
            )

        raw_bins[i] = raw

    if mixer is not None:
        raw_bins = mixer(raw_bins)

    bin_integrals: dict[int, float] | None = {} if need_meta else None
    parent_integral = float(np.trapezoid(parent_arr, x=z_arr)) if need_meta else None

    bins: dict[int, FloatArray] = {}

    for i in range(len(bin_edges) - 1):
        raw = raw_bins[i]
        area = float(np.trapezoid(raw, x=z_arr))

        if bin_integrals is not None:
            bin_integrals[i] = area

        if normalize_bins and not np.isclose(area, 0.0, atol=1e-12):
            raw = normalize_1d(z_arr, raw, method=norm_method)

        bins[i] = raw

    return bins, bin_integrals, parent_integral
