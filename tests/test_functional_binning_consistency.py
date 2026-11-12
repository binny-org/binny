"""Numerical consistency between functional and specialized bin builders."""

import numpy as np
import pytest

from binny.binning.photoz import build_photoz_bins
from binny.binning.specz import build_specz_bins
from binny.tomography.core import build_tomography


@pytest.fixture
def sample():
    z = np.linspace(0.0, 3.0, 1001)
    nz = z**2 * np.exp(-z / 0.5)
    edges = np.linspace(0.2, 2.0, 6)
    return z, nz, edges


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"scatter_scale": 0.05, "mean_offset": 0.01},
        {
            "scatter_scale": [0.03, 0.04, 0.05, 0.06, 0.07],
            "mean_offset": 0.01,
            "outlier_frac": 0.2,
            "outlier_scatter_scale": 0.15,
            "outlier_mean_offset": 0.1,
        },
    ],
)
@pytest.mark.parametrize("normalize", [False, True])
def test_photoz_consistency(sample, params, normalize):
    z, nz, edges = sample

    functional = build_tomography(
        z,
        nz,
        edges,
        method="photoz",
        params=params,
        normalize_input=True,
        normalize=normalize,
    )

    specialized = build_photoz_bins(
        z,
        nz,
        bins={"edges": edges},
        normalize_input=True,
        normalize_bins=normalize,
        **params,
    )

    for i in functional.bin_indices:
        np.testing.assert_allclose(
            functional.bins[i],
            specialized[i],
            rtol=1e-12,
            atol=1e-14,
        )


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"completeness": 0.8},
        {"catastrophic_frac": 0.15, "leakage_model": "neighbor"},
        {"catastrophic_frac": 0.1, "specz_scatter": 0.02},
        {
            "completeness": [0.9, 0.8, 0.95, 0.85, 0.9],
            "catastrophic_frac": 0.1,
            "leakage_model": "gaussian",
            "leakage_sigma": 1.5,
        },
    ],
)
@pytest.mark.parametrize("normalize", [False, True])
def test_specz_consistency(sample, params, normalize):
    z, nz, edges = sample

    functional = build_tomography(
        z,
        nz,
        edges,
        method="specz",
        params=params,
        normalize_input=True,
        normalize=normalize,
    )

    specialized = build_specz_bins(
        z,
        nz,
        bins={"edges": edges},
        normalize_input=True,
        normalize_bins=normalize,
        **params,
    )

    for i in functional.bin_indices:
        np.testing.assert_allclose(
            functional.bins[i],
            specialized[i],
            rtol=1e-12,
            atol=1e-14,
        )
