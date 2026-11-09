"""Consistency tests for functional and legacy predefined tomography."""

import numpy as np
import pytest

from binny import NZTomography
from binny.nz_tomo.predefined import prepare_predefined_bins
from binny.tomography import build_predefined_tomography


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("provide_parent", [False, True])
def test_predefined_functional_matches_legacy(normalize, provide_parent):
    z = np.linspace(0.0, 3.0, 301)

    bins = {
        0: np.exp(-0.5 * ((z - 0.6) / 0.2) ** 2),
        1: np.exp(-0.5 * ((z - 1.2) / 0.3) ** 2),
        2: np.exp(-0.5 * ((z - 1.9) / 0.4) ** 2),
    }

    parent = 2.0 * sum(bins.values()) if provide_parent else None

    functional = build_predefined_tomography(
        z,
        bins,
        nz=parent,
        normalize=normalize,
        include_metadata=True,
    )

    z_legacy, nz_legacy, bins_legacy, meta_legacy = prepare_predefined_bins(
        z,
        bins,
        nz=parent,
        normalize_bins=normalize,
        include_metadata=True,
    )

    np.testing.assert_allclose(functional.z, z_legacy)
    np.testing.assert_allclose(functional.nz, nz_legacy)

    for i in bins:
        np.testing.assert_allclose(functional.bins[i], bins_legacy[i])

    assert functional.metadata is not None
    assert meta_legacy is not None


@pytest.mark.parametrize("normalize", [False, True])
def test_predefined_functional_matches_nztomography(normalize):
    z = np.linspace(0.0, 3.0, 301)

    bins = {
        0: np.exp(-0.5 * ((z - 0.6) / 0.2) ** 2),
        1: np.exp(-0.5 * ((z - 1.2) / 0.3) ** 2),
    }

    functional = build_predefined_tomography(
        z,
        bins,
        normalize=normalize,
    )

    legacy = NZTomography().build_predefined_bins(
        z=z,
        bins=bins,
        normalize_bins=normalize,
    )

    np.testing.assert_allclose(functional.nz, legacy.nz)

    for i in bins:
        np.testing.assert_allclose(functional.bins[i], legacy.bins[i])
