"""Tomographic redshift distributions and construction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from binny.binning.core import build_bins
from binny.utils.types import FloatArray
from binny.utils.validators import validate_axis_and_weights

__all__ = ["Tomography", "build_tomography"]


@dataclass
class Tomography:
    """Tomographic distributions evaluated on a common redshift grid.

    Args:
        z: True-redshift grid.
        nz: Parent redshift distribution.
        bins: Mapping from bin index to redshift distribution.
        metadata: Additional information about the binning.
    """

    z: FloatArray
    nz: FloatArray
    bins: dict[int, FloatArray]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_bins(self) -> int:
        """Number of tomographic bins."""
        return len(self.bins)

    @property
    def bin_indices(self) -> list[int]:
        """Sorted tomographic bin indices."""
        return sorted(self.bins)

    def __getitem__(self, index: int) -> np.ndarray:
        """Return the distribution for a tomographic bin."""
        return self.bins[index]


def build_tomography(
    z: FloatArray,
    nz: FloatArray,
    edges: FloatArray,
    assignment: Callable[[FloatArray, FloatArray], FloatArray],
    *,
    selection_kind: Literal["probability", "weight"] = "probability",
    normalize: bool = True,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
) -> Tomography:
    """Build a tomography result from a selection function.

    Args:
        z: True-redshift grid.
        nz: Parent redshift distribution.
        edges: Bin edges in the selected coordinate.
        assignment: Function returning selections of shape
            ``(n_bins, n_z)``.
        selection_kind: Whether selections represent probabilities
            or arbitrary non-negative weights.
        normalize: Whether to normalize individual bin distributions.
        norm_method: Numerical integration method for normalization.

    Returns:
        Tomography result containing bin distributions and metadata.
    """
    z_arr, nz_arr = validate_axis_and_weights(z, nz)

    bins, integrals, parent_integral = build_bins(
        z_arr,
        nz_arr,
        edges,
        assignment,
        selection_kind=selection_kind,
        normalize=normalize,
        norm_method=norm_method,
        return_integrals=True,
    )

    fractions = {
        i: area / parent_integral if parent_integral > 0.0 else 0.0 for i, area in integrals.items()
    }

    metadata = {
        "bin_edges": np.asarray(edges, dtype=float).copy(),
        "selection_kind": selection_kind,
        "normalize_bins": normalize,
        "norm_method": norm_method,
        "bin_integrals": integrals,
        "parent_integral": parent_integral,
        "frac_per_bin": fractions,
    }

    return Tomography(
        z=z_arr,
        nz=nz_arr,
        bins=bins,
        metadata=metadata,
    )
