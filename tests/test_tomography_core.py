"""Tests for functional tomography construction."""

import numpy as np
import pytest

from binny.binning.assignments import (
    deterministic_assignment,
    specz_selection_in_bin,
)
from binny.binning.photoz import photoz_assignment
from binny.binning.specz import (
    build_specz_response_matrix,
    specz_gaussian_response_matrix,
)
from binny.nz_tomo.photoz import build_photoz_bins
from binny.tomography.core import Tomography, build_tomography


def test_build_deterministic_tomography():
    """Test that deterministic tomography constructs normalized bins."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = z**2 * np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)

    tomo = build_tomography(
        z,
        nz,
        edges,
        method="deterministic",
    )

    assert isinstance(tomo, Tomography)
    assert tomo.n_bins == 4
    assert tomo.bin_indices == [0, 1, 2, 3]
    assert tomo.metadata["method"] == "deterministic"

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
    """Test that photo-z tomography constructs normalized bins."""
    z = np.linspace(0.0, 3.0, 1001)
    nz = z**2 * np.exp(-z)
    edges = np.linspace(0.2, 2.0, 6)

    tomo = build_tomography(
        z,
        nz,
        edges,
        method="photoz",
        params={
            "scatter_scale": 0.05,
            "outlier_frac": 0.2,
            "outlier_scatter_scale": 0.2,
        },
    )

    assert tomo.n_bins == 5
    assert tomo.metadata["method"] == "photoz"

    for i in tomo.bin_indices:
        assert tomo.metadata["bin_integrals"][i] > 0.0
        assert 0.0 <= tomo.metadata["frac_per_bin"][i] <= 1.0

        np.testing.assert_allclose(
            np.trapezoid(tomo[i], x=z),
            1.0,
        )


def test_population_fractions_independent_of_bin_normalization():
    """Test that bin normalization does not change population fractions."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)

    raw = build_tomography(
        z,
        nz,
        edges,
        method="deterministic",
        normalize=False,
    )

    normalized = build_tomography(
        z,
        nz,
        edges,
        method="deterministic",
        normalize=True,
    )

    assert raw.metadata["frac_per_bin"] == normalized.metadata["frac_per_bin"]

    for i in raw.bin_indices:
        np.testing.assert_allclose(
            normalized.metadata["bin_integrals"][i],
            np.trapezoid(raw[i], x=z),
        )


def test_photoz_population_fractions_match_legacy():
    """Test that photo-z population fractions match the legacy builder."""
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
        method="photoz",
        params=params,
    )

    for i in functional.bin_indices:
        np.testing.assert_allclose(functional[i], legacy_bins[i])
        np.testing.assert_allclose(
            functional.metadata["frac_per_bin"][i],
            legacy_meta["bins"]["frac_per_bin"][i],
        )


def test_specz_generic_construction_with_response():
    """Test that spec-z tomography applies response matrices correctly."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = 5.0 * z**2 * np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)
    n_bins = len(edges) - 1

    completeness = np.array([1.0, 0.8, 0.6, 0.9])

    matrix_cat = build_specz_response_matrix(
        n_bins,
        catastrophic_frac=0.2,
        leakage_model="neighbor",
    )

    matrix_scatter = specz_gaussian_response_matrix(
        z_arr=z,
        bin_edges=edges,
        specz_scatter=0.03,
    )

    matrix_total = matrix_scatter @ matrix_cat

    tomo = build_tomography(
        z,
        nz,
        edges,
        method="specz",
        params={
            "completeness": completeness,
            "catastrophic_frac": 0.2,
            "leakage_model": "neighbor",
            "specz_scatter": 0.03,
        },
        normalize_input=True,
        normalize=False,
    )

    true_bins = np.stack(
        [
            tomo.nz
            * specz_selection_in_bin(
                z,
                edges[i],
                edges[i + 1],
                completeness=float(completeness[i]),
            )
            for i in range(n_bins)
        ]
    )

    expected = matrix_total @ true_bins

    for i in tomo.bin_indices:
        np.testing.assert_allclose(
            tomo[i],
            expected[i],
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            tomo.metadata["bin_integrals"][i],
            np.trapezoid(expected[i], x=z),
        )


def test_specz_population_fractions_with_completeness_and_response():
    """Test that spec-z response mixing preserves selected population."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = 5.0 * np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)
    n_bins = len(edges) - 1

    completeness = np.array([1.0, 0.8, 0.6, 0.9])

    matrix = build_specz_response_matrix(
        n_bins,
        catastrophic_frac=0.2,
        leakage_model="neighbor",
    )

    tomo = build_tomography(
        z,
        nz,
        edges,
        method="specz",
        params={
            "completeness": completeness,
            "catastrophic_frac": 0.2,
            "leakage_model": "neighbor",
        },
        normalize_input=False,
        normalize=True,
    )

    parent_integral = np.trapezoid(nz, x=z)

    true_integrals = np.array(
        [
            np.trapezoid(
                nz
                * specz_selection_in_bin(
                    z,
                    edges[i],
                    edges[i + 1],
                    completeness=float(completeness[i]),
                ),
                x=z,
            )
            for i in range(n_bins)
        ]
    )

    expected_integrals = matrix @ true_integrals
    expected_fractions = expected_integrals / parent_integral

    for i in tomo.bin_indices:
        np.testing.assert_allclose(
            tomo.metadata["bin_integrals"][i],
            expected_integrals[i],
        )
        np.testing.assert_allclose(
            tomo.metadata["frac_per_bin"][i],
            expected_fractions[i],
        )

    np.testing.assert_allclose(
        sum(tomo.metadata["frac_per_bin"].values()),
        sum(true_integrals) / parent_integral,
    )


def test_build_tomography_default_deterministic():
    """Test that deterministic assignment is the default method."""
    z = np.linspace(0.0, 2.0, 501)
    nz = np.ones_like(z)
    edges = np.array([0.0, 1.0, 2.0])

    tomo = build_tomography(
        z,
        nz,
        edges,
        normalize=False,
    )

    expected = deterministic_assignment(z, edges)

    assert tomo.n_bins == 2
    assert tomo.metadata["method"] == "deterministic"
    assert tomo.metadata["params"] == {}

    for i in tomo.bin_indices:
        np.testing.assert_array_equal(
            tomo[i],
            nz * expected[i],
        )


def test_build_tomography_specz_method():
    """Test that built-in spec-z construction matches explicit response mixing."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = z**2 * np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)

    completeness = np.array([1.0, 0.8, 0.6, 0.9])

    params = {
        "completeness": completeness,
        "catastrophic_frac": 0.2,
        "specz_scatter": 0.03,
    }

    tomo = build_tomography(
        z,
        nz,
        edges,
        method="specz",
        params=params,
        normalize=False,
    )

    true_bins = np.stack(
        [
            nz
            * specz_selection_in_bin(
                z,
                edges[i],
                edges[i + 1],
                completeness=float(completeness[i]),
            )
            for i in range(4)
        ]
    )

    matrix_cat = build_specz_response_matrix(
        4,
        catastrophic_frac=0.2,
    )

    matrix_scatter = specz_gaussian_response_matrix(
        z_arr=z,
        bin_edges=edges,
        specz_scatter=0.03,
    )

    expected = (matrix_scatter @ matrix_cat) @ true_bins

    for i in tomo.bin_indices:
        np.testing.assert_allclose(
            tomo[i],
            expected[i],
            rtol=1e-12,
            atol=1e-14,
        )

        np.testing.assert_allclose(
            tomo.metadata["bin_integrals"][i],
            np.trapezoid(expected[i], x=z),
        )

    assert tomo.metadata["method"] == "specz"


def test_build_tomography_photoz_method():
    """Test that built-in photo-z construction matches direct assignment."""
    z = np.linspace(0.0, 3.0, 1001)
    nz = z**2 * np.exp(-z)
    edges = np.linspace(0.2, 2.0, 6)

    params = {
        "scatter_scale": 0.05,
        "mean_offset": 0.01,
        "outlier_frac": 0.2,
        "outlier_scatter_scale": 0.2,
    }

    tomo = build_tomography(
        z,
        nz,
        edges,
        method="photoz",
        params=params,
        normalize=False,
    )

    expected_selection = photoz_assignment(z, edges, **params)

    for i in tomo.bin_indices:
        np.testing.assert_allclose(
            tomo[i],
            nz * expected_selection[i],
        )

    assert tomo.metadata["method"] == "photoz"
    assert tomo.metadata["params"] == params


def test_build_tomography_specz_population_conservation():
    """Test that spec-z response mixing preserves total selected population."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)

    baseline = build_tomography(
        z,
        nz,
        edges,
        method="specz",
        params={"completeness": 0.8},
        normalize=False,
    )

    mixed = build_tomography(
        z,
        nz,
        edges,
        method="specz",
        params={
            "completeness": 0.8,
            "catastrophic_frac": 0.2,
            "specz_scatter": 0.03,
        },
        normalize=False,
    )

    np.testing.assert_allclose(
        sum(mixed.metadata["frac_per_bin"].values()),
        sum(baseline.metadata["frac_per_bin"].values()),
        rtol=1e-12,
        atol=1e-14,
    )


def test_build_tomography_custom_assignment():
    """Test that custom assignment functions remain supported."""
    z = np.linspace(0.0, 2.0, 501)
    nz = np.exp(-z)
    edges = np.array([0.0, 1.0, 2.0])

    custom = build_tomography(
        z,
        nz,
        edges,
        deterministic_assignment,
        normalize=False,
    )

    builtin = build_tomography(
        z,
        nz,
        edges,
        method="deterministic",
        normalize=False,
    )

    assert custom.metadata["method"] == "custom"

    for i in custom.bin_indices:
        np.testing.assert_allclose(custom[i], builtin[i])


def test_build_tomography_rejects_unknown_method():
    """Test that unknown assignment methods raise a ValueError."""
    z = np.linspace(0.0, 2.0, 101)
    nz = np.ones_like(z)
    edges = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="method must be"):
        build_tomography(
            z,
            nz,
            edges,
            method="unknown",
        )


@pytest.mark.parametrize("method", ["photoz", "specz"])
def test_build_tomography_rejects_unknown_params(method):
    """Test that built-in models reject unknown parameter names."""
    z = np.linspace(0.0, 2.0, 101)
    nz = np.ones_like(z)
    edges = np.array([0.0, 1.0, 2.0])

    with pytest.raises(TypeError, match="unknown_parameter"):
        build_tomography(
            z,
            nz,
            edges,
            method=method,
            params={"unknown_parameter": 0.05},
        )


def test_build_tomography_rejects_conflicting_assignment():
    """Test that custom assignment and built-in method cannot be combined."""
    z = np.linspace(0.0, 2.0, 101)
    nz = np.ones_like(z)
    edges = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="either assignment or method"):
        build_tomography(
            z,
            nz,
            edges,
            deterministic_assignment,
            method="specz",
        )


def test_build_tomography_rejects_custom_assignment_params():
    """Test that custom assignments reject model parameters."""
    z = np.linspace(0.0, 2.0, 101)
    nz = np.ones_like(z)
    edges = np.array([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="params cannot be used"):
        build_tomography(
            z,
            nz,
            edges,
            deterministic_assignment,
            params={"scatter_scale": 0.05},
        )


def test_build_tomography_rejects_deterministic_params():
    """Test that deterministic assignment rejects unused parameters."""
    z = np.linspace(0.0, 2.0, 101)
    nz = np.ones_like(z)
    edges = np.array([0.0, 1.0, 2.0])

    with pytest.raises(TypeError, match="does not accept model parameters"):
        build_tomography(
            z,
            nz,
            edges,
            method="deterministic",
            params={"scatter_scale": 0.05},
        )
