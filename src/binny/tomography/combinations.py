"""Bin-combination filtering for tomographic redshift distributions."""

from __future__ import annotations

import numpy as np

from binny.correlations.bin_combo_filter import BinComboFilter
from binny.tomography.core import Tomography


def make_bin_combo_filter(
    tomo: Tomography,
    other: Tomography | None = None,
) -> BinComboFilter:
    """Create a bin-combination filter from one or two tomographies."""
    if other is None:
        return BinComboFilter(z=tomo.z, curves=[tomo.bins, tomo.bins])

    if np.array_equal(tomo.z, other.z):
        return BinComboFilter(z=tomo.z, curves=[tomo.bins, other.bins])

    z_common = np.union1d(tomo.z, other.z)

    bins_a = {
        k: np.interp(z_common, tomo.z, curve, left=0.0, right=0.0) for k, curve in tomo.bins.items()
    }
    bins_b = {
        k: np.interp(z_common, other.z, curve, left=0.0, right=0.0)
        for k, curve in other.bins.items()
    }

    return BinComboFilter(z=z_common, curves=[bins_a, bins_b])
