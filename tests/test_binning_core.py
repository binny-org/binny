"""Tests for ``binny.binning.core``."""

import numpy as np
import pytest
from scipy.special import ndtr

from binny.binning.assignments import deterministic_assignment
from binny.binning.core import build_bins
from binny.binning.photoz import photoz_assignment
from binny.binning.specz import (
    build_specz_response_matrix,
    specz_gaussian_response_matrix,
)
from binny.tomography.core import Tomography


def assert_bins_equal(actual, expected):
    """Compare mappings of tomographic bin distributions."""
    assert actual.keys() == expected.keys()
    for i in expected:
        np.testing.assert_allclose(actual[i], expected[i], rtol=1e-12, atol=1e-12)


def make_parent(z):
    """Return a non-normalized parent redshift distribution."""
    return z**2 * np.exp(-z)


def independent_top_hat(z, edges):
    """Construct left-inclusive, right-exclusive true-redshift bins."""
    return np.stack(
        [
            ((z >= lower) & (z < upper)).astype(float)
            for lower, upper in zip(edges[:-1], edges[1:], strict=True)
        ]
    )


def independent_photoz_probability(z, lower, upper, scatter, offset, scale):
    """Evaluate a Gaussian photo-z bin probability independently."""
    mu = scale * z - offset
    sigma = scatter * (1.0 + z)
    return ndtr((upper - mu) / sigma) - ndtr((lower - mu) / sigma)


def test_deterministic_matches_analytic_top_hat():
    """Tests that deterministic binning matches analytic top-hat selection."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = make_parent(z)
    edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    actual = build_bins(z, nz, edges, deterministic_assignment)
    expected_selection = independent_top_hat(z, edges)

    expected = {i: nz * expected_selection[i] for i in range(len(edges) - 1)}

    assert_bins_equal(actual, expected)


def test_response_matrix_matches_independent_matrix_multiplication():
    """Tests that response mixing matches independent matrix multiplication."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = make_parent(z)
    edges = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    matrix = np.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.1, 0.8, 0.1, 0.0],
            [0.0, 0.1, 0.8, 0.1],
            [0.0, 0.0, 0.1, 0.9],
        ]
    )

    actual = build_bins(
        z,
        nz,
        edges,
        lambda z, edges: deterministic_assignment(z, edges, response_matrix=matrix),
    )

    true_selection = independent_top_hat(z, edges)
    expected_curves = matrix @ (true_selection * nz[None, :])
    expected = {i: expected_curves[i] for i in range(4)}

    assert_bins_equal(actual, expected)


def test_specz_completeness_matches_independent_weights():
    """Tests that spectroscopic completeness scales true-bin populations."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = make_parent(z)
    edges = np.linspace(0.0, 2.0, 5)
    completeness = np.array([1.0, 0.8, 0.5, 0.9])

    actual = build_bins(
        z,
        nz,
        edges,
        lambda z, edges: deterministic_assignment(z, edges, completeness=completeness),
    )

    true_selection = independent_top_hat(z, edges)
    expected = {i: nz * true_selection[i] * completeness[i] for i in range(4)}

    assert_bins_equal(actual, expected)


def test_specz_catastrophic_and_scatter_matches_matrix_composition():
    """Tests that catastrophic leakage and scatter preserve matrix composition."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = make_parent(z)
    edges = np.linspace(0.0, 2.0, 5)

    completeness = np.array([1.0, 0.8, 0.6, 0.9])
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

    true_selection = independent_top_hat(z, edges)
    true_curves = true_selection * completeness[:, None] * nz[None, :]
    expected_curves = matrix_total @ true_curves
    expected = {i: expected_curves[i] for i in range(4)}

    assert_bins_equal(actual, expected)

    np.testing.assert_allclose(matrix_total.sum(axis=0), np.ones(4), atol=1e-12)
    np.testing.assert_allclose(
        np.sum(expected_curves, axis=0),
        np.sum(true_curves, axis=0),
        atol=1e-12,
    )


def test_photoz_matches_independent_gaussian_mixture():
    """Tests that photo-z binning matches an independent Gaussian mixture."""
    z = np.linspace(0.0, 3.0, 1001)
    nz = make_parent(z)
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

    actual = build_bins(
        z,
        nz,
        edges,
        lambda z, edges: photoz_assignment(z, edges, **params),
    )

    expected = {}
    for i, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        core = independent_photoz_probability(
            z,
            lower,
            upper,
            params["scatter_scale"][i],
            params["mean_offset"][i],
            params["mean_scale"][i],
        )
        tail = independent_photoz_probability(
            z,
            lower,
            upper,
            params["outlier_scatter_scale"],
            params["outlier_mean_offset"],
            params["outlier_mean_scale"],
        )
        fraction = params["outlier_frac"][i]
        expected[i] = nz * ((1.0 - fraction) * core + fraction * tail)

    assert_bins_equal(actual, expected)


def test_photoz_normalized_matches_independent_integrals():
    """Tests that normalized photo-z bins match independent integrals."""
    z = np.linspace(0.0, 3.0, 1001)
    nz = make_parent(z)
    edges = np.linspace(0.2, 2.0, 6)

    params = {
        "scatter_scale": 0.05,
        "mean_offset": 0.01,
        "outlier_frac": 0.2,
        "outlier_scatter_scale": 0.2,
    }

    actual = build_bins(
        z,
        nz,
        edges,
        lambda z, edges: photoz_assignment(z, edges, **params),
        normalize=True,
    )

    expected = {}
    for i, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        core = independent_photoz_probability(z, lower, upper, 0.05, 0.01, 1.0)
        tail = independent_photoz_probability(z, lower, upper, 0.2, 0.0, 1.0)
        raw = nz * (0.8 * core + 0.2 * tail)
        expected[i] = raw / np.trapezoid(raw, x=z)

    assert_bins_equal(actual, expected)

    for curve in actual.values():
        np.testing.assert_allclose(np.trapezoid(curve, x=z), 1.0, atol=1e-12)


def test_photoz_zero_scatter_matches_deterministic():
    """Tests that zero photo-z scatter reproduces deterministic binning."""
    z = np.linspace(0.0, 2.0, 1001)
    edges = np.linspace(0.0, 2.0, 5)

    actual = photoz_assignment(z, edges, scatter_scale=0.0)
    expected = deterministic_assignment(z, edges)

    np.testing.assert_allclose(actual, expected)


def test_probability_and_weight_selection():
    """Tests that probability and weight selections enforce their conventions."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = np.exp(-z)
    edges = np.array([0.0, 1.0, 2.0])

    def weighted_assignment(z, edges):
        return 2.0 * deterministic_assignment(z, edges)

    with pytest.raises(ValueError, match="probabilities must not exceed"):
        build_bins(z, nz, edges, weighted_assignment)

    actual = build_bins(
        z,
        nz,
        edges,
        weighted_assignment,
        selection_kind="weight",
    )
    expected = build_bins(z, nz, edges, deterministic_assignment)

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
    """Tests that invalid selection arrays raise ValueError."""
    z = np.linspace(0.0, 2.0, 101)
    nz = np.exp(-z)
    edges = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError):
        build_bins(z, nz, edges, selection)


def test_return_integrals_preserves_population():
    """Tests that returned integrals preserve raw bin populations."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = make_parent(z)
    edges = np.linspace(0.0, 2.0, 5)

    raw_bins = build_bins(z, nz, edges, deterministic_assignment)

    bins, integrals, parent_integral = build_bins(
        z,
        nz,
        edges,
        deterministic_assignment,
        normalize=True,
        return_integrals=True,
    )

    np.testing.assert_allclose(parent_integral, np.trapezoid(nz, x=z))

    for i in bins:
        np.testing.assert_allclose(integrals[i], np.trapezoid(raw_bins[i], x=z))
        np.testing.assert_allclose(np.trapezoid(bins[i], x=z), 1.0)


def test_tomography_result():
    """Tests that constructed bins integrate with the Tomography container."""
    z = np.linspace(0.0, 2.0, 101)
    nz = np.exp(-z)
    edges = np.array([0.0, 1.0, 2.0])

    bins, integrals, parent_integral = build_bins(
        z,
        nz,
        edges,
        deterministic_assignment,
        return_integrals=True,
    )

    result = Tomography(
        z=z,
        nz=nz,
        bins=bins,
        metadata={
            "bin_integrals": integrals,
            "parent_integral": parent_integral,
        },
    )

    assert result.n_bins == 2
    assert result.bin_indices == [0, 1]
    np.testing.assert_array_equal(result[0], bins[0])
    np.testing.assert_array_equal(result[1], bins[1])
