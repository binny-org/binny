"""Tests for functional tomography construction."""

import numpy as np

from binny.binning.assignments import (
    deterministic_assignment,
)
from binny.binning.photoz import photoz_assignment
from binny.tomography.core import Tomography, build_tomography


def test_build_deterministic_tomography():
    z = np.linspace(0.0, 2.0, 1001)
    nz = z**2 * np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)

    tomo = build_tomography(
        z,
        nz,
        edges,
        deterministic_assignment,
    )

    assert isinstance(tomo, Tomography)
    assert tomo.n_bins == 4
    assert tomo.bin_indices == [0, 1, 2, 3]

    np.testing.assert_allclose(
        tomo.metadata["parent_integral"],
        np.trapezoid(nz, x=z),
    )

    for i in tomo.bin_indices:
        np.testing.assert_allclose(
            np.trapezoid(tomo[i], x=z),
            1.0,
        )

    np.testing.assert_allclose(
        sum(tomo.metadata["frac_per_bin"].values()),
        1.0,
        atol=1e-3,
    )


def test_build_photoz_tomography():
    z = np.linspace(0.0, 3.0, 1001)
    nz = z**2 * np.exp(-z)
    edges = np.linspace(0.2, 2.0, 6)

    tomo = build_tomography(
        z,
        nz,
        edges,
        lambda z, edges: photoz_assignment(
            z,
            edges,
            scatter_scale=0.05,
            outlier_frac=0.2,
            outlier_scatter_scale=0.2,
        ),
    )

    assert tomo.n_bins == 5

    for i in tomo.bin_indices:
        assert tomo.metadata["bin_integrals"][i] > 0.0
        assert 0.0 <= tomo.metadata["frac_per_bin"][i] <= 1.0

        np.testing.assert_allclose(
            np.trapezoid(tomo[i], x=z),
            1.0,
        )


def test_population_fractions_independent_of_bin_normalization():
    z = np.linspace(0.0, 2.0, 1001)
    nz = np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)

    raw = build_tomography(
        z,
        nz,
        edges,
        deterministic_assignment,
        normalize=False,
    )

    normalized = build_tomography(
        z,
        nz,
        edges,
        deterministic_assignment,
        normalize=True,
    )

    assert raw.metadata["frac_per_bin"] == normalized.metadata["frac_per_bin"]

    for i in raw.bin_indices:
        np.testing.assert_allclose(
            normalized.metadata["bin_integrals"][i],
            np.trapezoid(raw[i], x=z),
        )


def test_photoz_population_fractions_match_legacy():
    """Check photo-z population fractions against the legacy builder."""
    from binny.nz_tomo.photoz import build_photoz_bins

    z = np.linspace(0.0, 3.0, 1001)
    nz = 5.0 * z**2 * np.exp(-z)
    edges = np.linspace(0.2, 2.0, 6)

    params = {
        "scatter_scale": 0.05,
        "mean_offset": 0.01,
        "outlier_frac": 0.2,
        "outlier_scatter_scale": 0.2,
    }

    legacy_bins, legacy_meta = build_photoz_bins(
        z,
        nz,
        bin_edges=edges,
        normalize_input=False,
        normalize_bins=True,
        include_metadata=True,
        **params,
    )

    functional = build_tomography(
        z,
        nz,
        edges,
        lambda z, edges: photoz_assignment(z, edges, **params),
    )

    for i in functional.bin_indices:
        np.testing.assert_allclose(functional[i], legacy_bins[i])
        np.testing.assert_allclose(
            functional.metadata["frac_per_bin"][i],
            legacy_meta["bins"]["frac_per_bin"][i],
        )
