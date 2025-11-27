"""Helpers for spectroscopic redshift samples and binning."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike

from src.binny.utils.normalization import normalize_1d
from src.binny.utils.broadcasting import as_per_bin
from src.binny.core.validation import (
    validate_axis_and_weights,
    validate_n_bins,
)

__all__ = [
    "build_specz_bins",
    "specz_selection_in_bin",
]


def build_specz_bins(
    z: ArrayLike,
    nz: ArrayLike,
    bin_edges: ArrayLike,
    *,
    completeness_per_bin: Sequence[float] | float = 1.0,
    normalize_input: bool = True,
    normalize_bins: bool = True,
    norm_method: str = "trapz",
) -> dict[int, np.ndarray]:
    ...
    z_arr, n_arr = validate_axis_and_weights(z, nz)

    bin_edges = np.asarray(bin_edges, dtype=float)
    n_bins = bin_edges.size - 1
    validate_n_bins(n_bins)

    if bin_edges[0] < z_arr[0] or bin_edges[-1] > z_arr[-1]:
        raise ValueError(
            f"bin_edges must lie within z-range [{z_arr[0]}, {z_arr[-1]}], "
            f"got [{bin_edges[0]}, {bin_edges[-1]}]."
        )

    # --- guard against double-normalisation of the parent nz ---
    if normalize_input:
        total = np.trapezoid(n_arr, z_arr)
        if np.isclose(total, 1.0, rtol=1e-3, atol=1e-4):
            raise ValueError(
                "build_specz_bins: normalize_input=True but intrinsic nz already looks "
                f"normalised (∫ n(z) dz ≈ {total:.4f}). "
                "Set normalize_input=False if nz is already normalised."
            )
        n_arr = normalize_1d(z_arr, n_arr, method=norm_method)

    completeness = as_per_bin(completeness_per_bin, n_bins, "completeness_per_bin")

    bins: dict[int, np.ndarray] = {}

    for i, (z_min, z_max) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        selection_i = specz_selection_in_bin(z_arr, z_min, z_max, completeness=completeness[i])
        nz_bin = n_arr * selection_i

        if normalize_bins and np.trapezoid(nz_bin, z_arr) > 0:
            nz_bin = normalize_1d(z_arr, nz_bin, method=norm_method)

        bins[i] = nz_bin

    return bins


def specz_selection_in_bin(
    z: ArrayLike,
    bin_min: float,
    bin_max: float,
    completeness: float = 1.0,
    *,
    inclusive_right: bool = False,
) -> np.ndarray:
    """Return a top-hat selection function S_i(z) for a spectroscopic bin."""
    z_arr = np.asarray(z, dtype=float)

    if not (0.0 <= completeness <= 1.0):
        raise ValueError("completeness must be in [0, 1].")

    if inclusive_right:
        mask = (z_arr >= bin_min) & (z_arr <= bin_max)
    else:
        mask = (z_arr >= bin_min) & (z_arr < bin_max)

    return completeness * mask.astype(float)
