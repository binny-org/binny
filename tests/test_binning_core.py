"""Tests for the new functional binning API."""

import numpy as np
import pytest

from binny.binning.assignments import (
    deterministic_assignment,
    photoz_assignment,
)
from binny.binning.core import build_bins
from binny.nz_tomo.photoz import build_photoz_bins
from binny.nz_tomo.specz import (
    build_specz_bins,
    build_specz_response_matrix,
    specz_gaussian_response_matrix,
)


def assert_bins_equal(actual, expected):
    """Compare two mappings of tomographic bin distributions."""
    assert actual.keys() == expected.keys()
    for i in expected:
        np.testing.assert_allclose(actual[i], expected[i])


def test_deterministic_matches_existing_specz():
    z = np.linspace(0.0, 2.0, 1001)
    nz = z**2 * np.exp(-z)
    edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    expected = build_specz_bins(
        z,
        nz,
        bin_edges=edges,
        normalize_input=False,
        normalize_bins=False,
    )

    actual = build_bins(
        z,
        nz,
        edges,
        deterministic_assignment,
    )

    assert_bins_equal(actual, expected)


def test_response_matrix_matches_existing_specz():
    z = np.linspace(0.0, 2.0, 1001)
    nz = z**2 * np.exp(-z)
    edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    matrix = np.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0],
            [0.0, 0.1, 0.8, 0.1],
            [0.0, 0.0, 0.1, 0.9],
        ]
    )

    expected = build_specz_bins(
        z,
        nz,
        bin_edges=edges,
        response_matrix=matrix,
        normalize_input=False,
        normalize_bins=False,
    )

    actual = build_bins(
        z,
        nz,
        edges,
        lambda z, edges: deterministic_assignment(
            z,
            edges,
            response_matrix=matrix,
        ),
    )

    assert_bins_equal(actual, expected)


def test_specz_completeness_matches_existing():
    z = np.linspace(0.0, 2.0, 1001)
    nz = z**2 * np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)
    completeness = [1.0, 0.8, 0.5, 0.9]

    expected = build_specz_bins(
        z,
        nz,
        bin_edges=edges,
        completeness=completeness,
        normalize_input=False,
        normalize_bins=False,
    )

    actual = build_bins(
        z,
        nz,
        edges,
        lambda z, edges: deterministic_assignment(
            z,
            edges,
            completeness=completeness,
        ),
    )

    assert_bins_equal(actual, expected)


def test_specz_catastrophic_and_scatter_matches_existing():
    z = np.linspace(0.0, 2.0, 1001)
    nz = z**2 * np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)
    completeness = [1.0, 0.8, 0.6, 0.9]

    catastrophic_frac = [0.0, 0.1, 0.2, 0.15]
    specz_scatter = 0.03

    matrix_cat = build_specz_response_matrix(
        4,
        catastrophic_frac=catastrophic_frac,
        leakage_model="neighbor",
    )

    matrix_scatter = specz_gaussian_response_matrix(
        z_arr=z,
        bin_edges=edges,
        specz_scatter=specz_scatter,
    )

    matrix_total = matrix_scatter @ matrix_cat

    expected = build_specz_bins(
        z,
        nz,
        bin_edges=edges,
        completeness=completeness,
        catastrophic_frac=catastrophic_frac,
        leakage_model="neighbor",
        specz_scatter=specz_scatter,
        normalize_input=False,
        normalize_bins=False,
    )

    actual = build_bins(
        z,
        nz,
        edges,
        lambda z, edges: deterministic_assignment(
            z,
            edges,
            completeness=completeness,
            response_matrix=matrix_total,
        ),
    )

    assert_bins_equal(actual, expected)


def test_photoz_matches_existing():
    z = np.linspace(0.0, 3.0, 1001)
    nz = z**2 * np.exp(-z)
    edges = np.linspace(0.2, 2.0, 6)

    params = {
        "scatter_scale": [0.03, 0.04, 0.05, 0.06, 0.07],
        "mean_offset": [0.0, 0.01, -0.01, 0.02, 0.0],
        "mean_scale": [1.0, 1.01, 0.99, 1.02, 1.0],
        "outlier_frac": [0.0, 0.1, 0.2, 0.3, 0.15],
        "outlier_scatter_scale": 0.2,
        "outlier_mean_offset": 0.1,
        "outlier_mean_scale": 1.0,
    }

    expected = build_photoz_bins(
        z,
        nz,
        bin_edges=edges,
        normalize_input=False,
        normalize_bins=False,
        **params,
    )

    actual = build_bins(
        z,
        nz,
        edges,
        lambda z, edges: photoz_assignment(
            z,
            edges,
            **params,
        ),
    )

    assert_bins_equal(actual, expected)


def test_photoz_normalized_matches_existing():
    z = np.linspace(0.0, 3.0, 1001)
    nz = z**2 * np.exp(-z)
    edges = np.linspace(0.2, 2.0, 6)

    params = {
        "scatter_scale": 0.05,
        "mean_offset": 0.01,
        "outlier_frac": 0.2,
        "outlier_scatter_scale": 0.2,
    }

    expected = build_photoz_bins(
        z,
        nz,
        bin_edges=edges,
        normalize_input=False,
        normalize_bins=True,
        **params,
    )

    actual = build_bins(
        z,
        nz,
        edges,
        lambda z, edges: photoz_assignment(
            z,
            edges,
            **params,
        ),
        normalize=True,
    )

    assert_bins_equal(actual, expected)

    for curve in actual.values():
        np.testing.assert_allclose(
            np.trapezoid(curve, x=z),
            1.0,
            atol=1e-12,
        )


def test_photoz_zero_scatter_matches_deterministic():
    z = np.linspace(0.0, 2.0, 1001)
    edges = np.linspace(0.0, 2.0, 5)

    actual = photoz_assignment(
        z,
        edges,
        scatter_scale=0.0,
    )

    expected = deterministic_assignment(z, edges)

    np.testing.assert_allclose(actual, expected)


def test_probability_and_weight_selection():
    z = np.linspace(0.0, 2.0, 1001)
    nz = np.exp(-z)
    edges = np.array([0.0, 1.0, 2.0])

    def weighted_assignment(z, edges):
        return 2.0 * deterministic_assignment(z, edges)

    with pytest.raises(ValueError, match="probabilities must not exceed"):
        build_bins(
            z,
            nz,
            edges,
            weighted_assignment,
        )

    actual = build_bins(
        z,
        nz,
        edges,
        weighted_assignment,
        selection_kind="weight",
    )

    expected = build_bins(
        z,
        nz,
        edges,
        deterministic_assignment,
    )

    for i in expected:
        np.testing.assert_allclose(actual[i], 2.0 * expected[i])


@pytest.mark.parametrize(
    "selection",
    [
        lambda z, edges: np.zeros((1, z.size)),
        lambda z, edges: np.full((2, z.size), np.nan),
        lambda z, edges: np.full((2, z.size), -0.1),
    ],
)
def test_invalid_selection_raises(selection):
    z = np.linspace(0.0, 2.0, 101)
    nz = np.exp(-z)
    edges = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError):
        build_bins(
            z,
            nz,
            edges,
            selection,
        )
