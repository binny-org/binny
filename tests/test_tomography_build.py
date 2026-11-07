"""Tests for functional tomography construction."""

import numpy as np

from binny.binning.assignments import (
    deterministic_assignment,
    photoz_assignment,
)
from binny.tomography.build import build_tomography
from binny.tomography.result import Tomography


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
