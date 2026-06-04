"""Utilities for combining and flattening tomographic redshift samples.

This module provides helpers for working with multiple ``TomographyBins``
objects as one logical sample collection. It supports summing parent redshift
distributions, interpolating samples onto a shared grid, combining matching
tomographic bins, flattening sample/bin labels, and building sample-aware bin
combinations for downstream correlation or data-vector construction.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np

from binny.nz_tomo._tomography_bins import TomographyBins


def combine_parent_nz(
    samples: Sequence[Mapping[str, Any]],
    *,
    interpolate: bool = False,
    z_target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine parent redshift distributions into one parent sample."""
    if len(samples) == 0:
        raise ValueError("At least one sample is required.")

    z0 = (
        np.asarray(samples[0]["z"], dtype=float)
        if z_target is None
        else np.asarray(z_target, dtype=float)
    )
    nz_total = np.zeros_like(z0, dtype=float)

    for sample in samples:
        z = np.asarray(sample["z"], dtype=float)
        nz = np.asarray(sample["nz"], dtype=float)

        if interpolate:
            nz = np.interp(z0, z, nz, left=0.0, right=0.0)
        elif z.shape != z0.shape or not np.allclose(z, z0):
            raise ValueError(
                "All samples must use the same z grid. "
                "Use interpolate=True to combine samples on a common grid."
            )

        nz_total += nz

    return z0, nz_total


def interpolate_tomography_bins(
    sample: TomographyBins,
    z_target: np.ndarray,
) -> TomographyBins:
    """Interpolate a tomographic sample onto a target redshift grid."""
    z = np.asarray(sample.z, dtype=float)
    z_target = np.asarray(z_target, dtype=float)

    return TomographyBins(
        z=z_target,
        nz=np.interp(
            z_target,
            z,
            np.asarray(sample.nz, dtype=float),
            left=0.0,
            right=0.0,
        ),
        spec=sample.spec,
        bins={
            k: np.interp(
                z_target,
                z,
                np.asarray(nz_bin, dtype=float),
                left=0.0,
                right=0.0,
            )
            for k, nz_bin in sample.bins.items()
        },
        tomo_meta=sample.tomo_meta,
        survey_meta=sample.survey_meta,
        survey=sample.survey,
    )


def combine_tomography_bins(
    samples: Sequence[TomographyBins],
    *,
    interpolate: bool = False,
    z_target: np.ndarray | None = None,
) -> TomographyBins:
    """Combine matching tomographic samples into one ``TomographyBins`` object."""
    if len(samples) == 0:
        raise ValueError("At least one TomographyBins object is required.")

    z0 = (
        np.asarray(samples[0].z, dtype=float)
        if z_target is None
        else np.asarray(z_target, dtype=float)
    )

    if interpolate:
        samples = [interpolate_tomography_bins(sample, z0) for sample in samples]

    first = samples[0]
    keys0 = list(first.bins.keys())

    combined_bins = {k: np.zeros_like(z0, dtype=float) for k in keys0}
    combined_nz = np.zeros_like(z0, dtype=float)

    for sample in samples:
        z = np.asarray(sample.z, dtype=float)

        if z.shape != z0.shape or not np.allclose(z, z0):
            raise ValueError(
                "All samples must use the same z grid. "
                "Use interpolate=True to combine samples on a common grid."
            )

        if list(sample.bins.keys()) != keys0:
            raise ValueError("All samples must have the same bin keys.")

        combined_nz += np.asarray(sample.nz, dtype=float)

        for k in keys0:
            combined_bins[k] += np.asarray(sample.bins[k], dtype=float)

    return TomographyBins(
        z=z0,
        nz=combined_nz,
        spec={"kind": "combined", "source": "combined_tomography_bins"},
        bins=combined_bins,
        tomo_meta=None,
        survey_meta=None,
        survey=None,
    )


def sample_bin_labels(
    samples: Mapping[str, TomographyBins],
) -> list[tuple[str, int]]:
    """Return sample-aware labels for all tomographic bins."""
    _check_shared_z_grid(samples)

    return [
        (sample_name, bin_index)
        for sample_name, sample in samples.items()
        for bin_index in sample.bins
    ]


def sample_bins(
    samples: Mapping[str, TomographyBins],
) -> dict[tuple[str, int], np.ndarray]:
    """Return tomographic bins keyed by sample-aware bin labels."""
    _check_shared_z_grid(samples)

    return {
        (sample_name, bin_index): np.asarray(nz_bin, dtype=float)
        for sample_name, sample in samples.items()
        for bin_index, nz_bin in sample.bins.items()
    }


def sample_combinations(
    *collections: Mapping[str, TomographyBins],
) -> list[tuple[tuple[str, int], ...]]:
    """Return sample-aware bin-label combinations across sample collections."""
    if len(collections) == 0:
        raise ValueError("At least one sample collection is required.")

    return [
        combo for combo in product(*(sample_bin_labels(collection) for collection in collections))
    ]


def _check_shared_z_grid(samples: Mapping[str, TomographyBins]) -> None:
    """Check that all samples in a collection share one redshift grid."""
    if not samples:
        raise ValueError("At least one sample is required.")

    first = next(iter(samples.values()))
    z0 = np.asarray(first.z, dtype=float)

    for name, sample in samples.items():
        z = np.asarray(sample.z, dtype=float)

        if z.shape != z0.shape or not np.allclose(z, z0):
            raise ValueError(f"Sample {name!r} does not share the same z grid.")
