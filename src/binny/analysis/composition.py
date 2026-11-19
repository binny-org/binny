"""Functional utilities for composing tomographic samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

import numpy as np

from binny.tomography.core import Tomography
from binny.utils.validators import validate_axis_and_weights

__all__ = [
    "combine_parent_nz",
    "interpolate_tomography",
    "align_bin_grids",
    "combine_tomographies",
    "sample_bin_labels",
    "sample_bins",
    "sample_combinations",
]


def combine_parent_nz(
    samples: Sequence[Mapping[str, np.ndarray]],
    *,
    interpolate: bool = False,
    z_target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine parent redshift distributions on a shared grid."""
    if not samples:
        raise ValueError("At least one sample is required.")

    z0 = np.asarray(
        samples[0]["z"] if z_target is None else z_target,
        dtype=float,
    )
    nz_total = np.zeros_like(z0)

    for sample in samples:
        z, nz = validate_axis_and_weights(sample["z"], sample["nz"])

        if interpolate:
            nz = np.interp(z0, z, nz, left=0.0, right=0.0)
        elif z.shape != z0.shape or not np.allclose(z, z0):
            raise ValueError(
                "All samples must use the same z grid. "
                "Use interpolate=True to combine samples on a common grid."
            )

        nz_total += nz

    return z0, nz_total


def interpolate_tomography(
    sample: Tomography,
    z_target: np.ndarray,
) -> Tomography:
    """Interpolate a tomography result onto a target redshift grid."""
    z_target = np.asarray(z_target, dtype=float)
    z = np.asarray(sample.z, dtype=float)

    return Tomography(
        z=z_target,
        nz=np.interp(z_target, z, sample.nz, left=0.0, right=0.0),
        bins={
            i: np.interp(z_target, z, curve, left=0.0, right=0.0)
            for i, curve in sample.bins.items()
        },
        metadata=dict(sample.metadata),
    )


def align_bin_grids(
    z_a: np.ndarray,
    bins_a: Mapping[int, np.ndarray],
    z_b: np.ndarray,
    bins_b: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Align two bin collections onto their union redshift grid."""
    z_a = np.asarray(z_a, dtype=float)
    z_b = np.asarray(z_b, dtype=float)

    if np.array_equal(z_a, z_b):
        return z_a, dict(bins_a), dict(bins_b)

    z_common = np.union1d(z_a, z_b)

    aligned_a = {
        i: np.interp(z_common, z_a, curve, left=0.0, right=0.0) for i, curve in bins_a.items()
    }
    aligned_b = {
        i: np.interp(z_common, z_b, curve, left=0.0, right=0.0) for i, curve in bins_b.items()
    }

    return z_common, aligned_a, aligned_b


def combine_tomographies(
    samples: Sequence[Tomography],
    *,
    interpolate: bool = False,
    z_target: np.ndarray | None = None,
) -> Tomography:
    """Combine tomographies with matching bin indices by summing curves."""
    if not samples:
        raise ValueError("At least one tomography is required.")

    z0 = np.asarray(
        samples[0].z if z_target is None else z_target,
        dtype=float,
    )

    if interpolate:
        samples = [interpolate_tomography(sample, z0) for sample in samples]

    keys = list(samples[0].bins)
    combined_nz = np.zeros_like(z0)
    combined_bins = {i: np.zeros_like(z0) for i in keys}

    for sample in samples:
        z = np.asarray(sample.z, dtype=float)

        if z.shape != z0.shape or not np.allclose(z, z0):
            raise ValueError(
                "All samples must use the same z grid. "
                "Use interpolate=True to combine samples on a common grid."
            )

        if list(sample.bins) != keys:
            raise ValueError("All samples must have the same bin keys.")

        combined_nz += sample.nz

        for i in keys:
            combined_bins[i] += sample.bins[i]

    return Tomography(
        z=z0,
        nz=combined_nz,
        bins=combined_bins,
        metadata={"kind": "combined"},
    )


def sample_bin_labels(
    samples: Mapping[str, Tomography],
) -> list[tuple[str, int]]:
    """Return sample-aware labels for all tomographic bins."""
    if not samples:
        raise ValueError("At least one sample is required.")

    return [(name, i) for name, sample in samples.items() for i in sample.bins]


def sample_bins(
    samples: Mapping[str, Tomography],
) -> dict[tuple[str, int], np.ndarray]:
    """Return bin curves indexed by sample name and bin index."""
    if not samples:
        raise ValueError("At least one sample is required.")

    return {
        (name, i): np.asarray(curve, dtype=float)
        for name, sample in samples.items()
        for i, curve in sample.bins.items()
    }


def sample_combinations(
    *collections: Mapping[str, Tomography],
) -> list[tuple[tuple[str, int], ...]]:
    """Return Cartesian products of sample-aware bin labels."""
    if not collections:
        raise ValueError("At least one sample collection is required.")

    return list(product(*(sample_bin_labels(collection) for collection in collections)))
