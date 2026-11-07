"""Container for tomographic redshift distributions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from binny.utils.types import FloatArray

__all__ = ["Tomography"]


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
