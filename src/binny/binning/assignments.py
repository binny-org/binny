"""Selection functions for generic tomographic binning."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from binny.utils.broadcasting import as_per_bin
from binny.utils.types import FloatArray
from binny.utils.validators import validate_response_matrix

__all__ = [
    "deterministic_assignment",
    "specz_selection_in_bin",
]


def specz_selection_in_bin(
    z: FloatArray,
    bin_min: float,
    bin_max: float,
    completeness: float = 1.0,
    *,
    inclusive_right: bool = False,
) -> FloatArray:
    """Evaluate a deterministic selection window.

    Args:
        z: Redshift or other selection-coordinate grid.
        bin_min: Lower bin edge.
        bin_max: Upper bin edge.
        completeness: Selection completeness in [0, 1].
        inclusive_right: Whether to include the upper edge.

    Returns:
        Selection weights evaluated on the input grid.

    Raises:
        ValueError: If completeness is outside [0, 1].
    """
    z_arr = np.asarray(z, dtype=float)

    if not (0.0 <= completeness <= 1.0):
        raise ValueError("completeness must be in [0, 1].")

    if inclusive_right:
        mask = (z_arr >= bin_min) & (z_arr <= bin_max)
    else:
        mask = (z_arr >= bin_min) & (z_arr < bin_max)

    return completeness * mask.astype(np.float64)


def deterministic_assignment(
    z: FloatArray,
    edges: FloatArray,
    *,
    completeness: Sequence[float] | float = 1.0,
    response_matrix: FloatArray | None = None,
) -> FloatArray:
    """Return deterministic bin selections with optional mixing.

    Args:
        z: Grid of the binning coordinate.
        edges: Bin edges in the same coordinate.
        completeness: Scalar or per-bin completeness.
        response_matrix: Optional response matrix mapping true bins
            to observed bins.

    Returns:
        Selection array with shape ``(n_bins, n_z)``.
    """
    z_arr = np.asarray(z, dtype=float)
    edges_arr = np.asarray(edges, dtype=float)
    n_bins = edges_arr.size - 1

    completeness_arr = as_per_bin(completeness, n_bins, "completeness")

    selection = np.stack(
        [
            specz_selection_in_bin(
                z_arr,
                float(edges_arr[i]),
                float(edges_arr[i + 1]),
                completeness=float(completeness_arr[i]),
            )
            for i in range(n_bins)
        ],
        axis=0,
    )

    if response_matrix is not None:
        matrix = np.asarray(response_matrix, dtype=float)
        validate_response_matrix(matrix, n_bins)
        selection = matrix @ selection

    return selection
