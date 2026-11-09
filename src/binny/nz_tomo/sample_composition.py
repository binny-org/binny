"""Backward-compatible composition utilities for legacy tomography samples."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from binny.nz_tomo._tomography_bins import TomographyBins
from binny.tomography.composition import (
    align_bin_grids as _align_bin_grids,
)
from binny.tomography.composition import (
    combine_parent_nz,
)
from binny.tomography.composition import (
    combine_tomographies as _combine_tomographies,
)
from binny.tomography.composition import (
    interpolate_tomography as _interpolate_tomography,
)
from binny.tomography.composition import (
    sample_bin_labels as _sample_bin_labels,
)
from binny.tomography.composition import (
    sample_bins as _sample_bins,
)
from binny.tomography.composition import (
    sample_combinations as _sample_combinations,
)
from binny.tomography.core import Tomography

__all__ = [
    "combine_parent_nz",
    "interpolate_tomography_bins",
    "align_bin_grids",
    "combine_tomography_bins",
    "sample_bin_labels",
    "sample_bins",
    "sample_combinations",
]


def _to_tomography(sample: TomographyBins) -> Tomography:
    """Convert a legacy tomography container to the functional representation."""
    return Tomography(
        z=sample.z,
        nz=sample.nz,
        bins=dict(sample.bins),
    )


def _to_legacy(
    sample: Tomography,
    *,
    spec: Mapping[str, Any],
    tomo_meta: Any = None,
    survey_meta: Any = None,
    survey: str | None = None,
) -> TomographyBins:
    """Convert a functional tomography result to a legacy container."""
    return TomographyBins(
        z=sample.z,
        nz=sample.nz,
        bins=sample.bins,
        spec=spec,
        tomo_meta=tomo_meta,
        survey_meta=survey_meta,
        survey=survey,
    )


def interpolate_tomography_bins(
    sample: TomographyBins,
    z_target: np.ndarray,
) -> TomographyBins:
    """Interpolate a legacy tomography sample onto a target redshift grid."""
    result = _interpolate_tomography(_to_tomography(sample), z_target)

    return _to_legacy(
        result,
        spec=sample.spec,
        tomo_meta=sample.tomo_meta,
        survey_meta=sample.survey_meta,
        survey=sample.survey,
    )


def align_bin_grids(
    z_a: np.ndarray,
    bins_a: Mapping[int, Any],
    z_b: np.ndarray,
    bins_b: Mapping[int, Any],
) -> tuple[np.ndarray, Mapping[int, Any], Mapping[int, Any]]:
    """Align two bin collections, preserving legacy mapping behavior."""
    z_a_arr = np.asarray(z_a, dtype=float)
    z_b_arr = np.asarray(z_b, dtype=float)

    if np.array_equal(z_a_arr, z_b_arr):
        return z_a_arr, bins_a, bins_b

    return _align_bin_grids(z_a_arr, bins_a, z_b_arr, bins_b)


def combine_tomography_bins(
    samples: Sequence[TomographyBins],
    *,
    interpolate: bool = False,
    z_target: np.ndarray | None = None,
) -> TomographyBins:
    """Combine legacy tomography samples using the functional implementation."""
    if len(samples) == 0:
        raise ValueError("At least one TomographyBins object is required.")

    result = _combine_tomographies(
        [_to_tomography(sample) for sample in samples],
        interpolate=interpolate,
        z_target=z_target,
    )

    return _to_legacy(
        result,
        spec={"kind": "combined", "source": "combined_tomography_bins"},
    )


def _check_shared_z_grid(samples: Mapping[str, TomographyBins]) -> None:
    """Require a shared redshift grid within a legacy sample collection."""
    if not samples:
        raise ValueError("At least one sample is required.")

    first = next(iter(samples.values()))
    z0 = np.asarray(first.z, dtype=float)

    for name, sample in samples.items():
        z = np.asarray(sample.z, dtype=float)
        if z.shape != z0.shape or not np.allclose(z, z0):
            raise ValueError(f"Sample {name!r} does not share the same z grid.")


def sample_bin_labels(
    samples: Mapping[str, TomographyBins],
) -> list[tuple[str, int]]:
    """Return legacy sample-aware bin labels."""
    _check_shared_z_grid(samples)

    return _sample_bin_labels({name: _to_tomography(sample) for name, sample in samples.items()})


def sample_bins(
    samples: Mapping[str, TomographyBins],
) -> dict[tuple[str, int], np.ndarray]:
    """Return legacy sample-aware bin curves."""
    _check_shared_z_grid(samples)

    return _sample_bins({name: _to_tomography(sample) for name, sample in samples.items()})


def sample_combinations(
    *collections: Mapping[str, TomographyBins],
) -> list[tuple[tuple[str, int], ...]]:
    """Return sample-aware combinations, preserving legacy grid validation."""
    if not collections:
        raise ValueError("At least one sample collection is required.")

    converted = []
    for collection in collections:
        _check_shared_z_grid(collection)
        converted.append({name: _to_tomography(sample) for name, sample in collection.items()})

    return _sample_combinations(*converted)
