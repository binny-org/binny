"""Helpers for spectroscopic redshift samples and binning."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from binny.core.validators import validate_axis_and_weights, validate_n_bins
from binny.utils.broadcasting import as_per_bin
from binny.utils.normalization import normalize_1d

__all__ = [
    "build_specz_bins",
    "specz_selection_in_bin",
]


def build_specz_bins(
    z: Any,
    nz: Any,
    bin_edges: Any,
    *,
    completeness_per_bin: Sequence[float] | float = 1.0,
    normalize_input: bool = True,
    normalize_bins: bool = True,
    norm_method: str = "trapz",
) -> dict[int, np.ndarray]:
    """Build spectroscopic redshift distributions per tomographic bin.

    This function constructs per-bin distributions by applying a top-hat
    selection in true redshift to an intrinsic parent distribution ``n(z)``. A
    per-bin completeness factor can be provided to downweight each bin by a
    constant fraction.

    Args:
        z: One-dimensional redshift grid.
        nz: Parent redshift distribution evaluated on ``z``.
        bin_edges: One-dimensional array of bin edges in true redshift. Must have
            length ``n_bins + 1`` and lie within the range spanned by ``z``.
        completeness_per_bin: Per-bin completeness factors in ``[0, 1]``. May be
            a scalar (applied to all bins) or a sequence of length ``n_bins``.
        normalize_input: Whether to normalize the input ``nz`` before binning.
        normalize_bins: Whether to normalize each output bin distribution.
        norm_method: Normalization method passed to :func:`normalize_1d`.

    Returns:
        A mapping from bin index to the corresponding binned distribution
        evaluated on ``z``.

    Raises:
        ValueError: If ``bin_edges`` does not define a valid number of bins.
        ValueError: If ``bin_edges`` extends outside the range of ``z``.
        ValueError: If ``normalize_input`` is True and the input ``nz`` already
            appears normalized.
    """
    z_arr, n_arr = validate_axis_and_weights(z, nz)

    bin_edges_arr = np.asarray(bin_edges, dtype=float)
    n_bins = bin_edges_arr.size - 1
    validate_n_bins(n_bins)

    if bin_edges_arr[0] < z_arr[0] or bin_edges_arr[-1] > z_arr[-1]:
        raise ValueError(
            f"bin_edges must lie within z-range [{z_arr[0]}, {z_arr[-1]}], "
            f"got [{bin_edges_arr[0]}, {bin_edges_arr[-1]}]."
        )

    if normalize_input:
        total = np.trapezoid(n_arr, z_arr)
        if np.isclose(total, 1.0, rtol=1e-3, atol=1e-4):
            raise ValueError(
                "build_specz_bins: normalize_input=True but intrinsic nz already "
                f"looks normalised (integral n(z) dz approx {total:.4f}). "
                "Set normalize_input=False if nz is already normalised."
            )
        n_arr = normalize_1d(z_arr, n_arr, method=norm_method)

    completeness = as_per_bin(completeness_per_bin, n_bins, "completeness_per_bin")

    bins: dict[int, np.ndarray] = {}

    for i, (z_min, z_max) in enumerate(
        zip(bin_edges_arr[:-1], bin_edges_arr[1:], strict=False)
    ):
        selection_i = specz_selection_in_bin(
            z_arr,
            float(z_min),
            float(z_max),
            completeness=float(completeness[i]),
        )
        nz_bin = n_arr * selection_i

        if normalize_bins and np.trapezoid(nz_bin, z_arr) > 0:
            nz_bin = normalize_1d(z_arr, nz_bin, method=norm_method)

        bins[i] = nz_bin

    return bins


def specz_selection_in_bin(
    z: Any,
    bin_min: float,
    bin_max: float,
    completeness: float = 1.0,
    *,
    inclusive_right: bool = False,
) -> np.ndarray:
    """Compute a top-hat selection function for a spectroscopic bin.

    The selection is defined on a redshift grid ``z`` as an indicator function
    for the interval ``[bin_min, bin_max)`` by default, optionally including the
    right edge. The output is scaled by ``completeness``.

    Args:
        z: One-dimensional redshift grid.
        bin_min: Lower edge of the redshift bin.
        bin_max: Upper edge of the redshift bin.
        completeness: Multiplicative completeness factor in ``[0, 1]`` applied to
            the selection.
        inclusive_right: Whether to include the right edge of the interval,
            selecting ``z == bin_max`` when True.

    Returns:
        A one-dimensional array ``S(z)`` evaluated on ``z``, with values in
        ``[0, completeness]``.

    Raises:
        ValueError: If ``completeness`` is not in the interval ``[0, 1]``.
    """
    z_arr = np.asarray(z, dtype=float)

    if not (0.0 <= completeness <= 1.0):
        raise ValueError("completeness must be in [0, 1].")

    if inclusive_right:
        mask = (z_arr >= bin_min) & (z_arr <= bin_max)
    else:
        mask = (z_arr >= bin_min) & (z_arr < bin_max)

    return completeness * mask.astype(float)
