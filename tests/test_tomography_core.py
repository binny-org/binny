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
from binny.tomography.core import Tomography, build_tomography


def test_build_deterministic_tomography():
    """Tests that deterministic tomography constructs normalized bins."""
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
    assert tomo.metadata["kind"] == "deterministic"

    np.testing.assert_allclose(
        tomo.metadata["parent_nz"]["norm"],
        1.0,
    )

    for i in tomo.bin_indices:
        np.testing.assert_allclose(
            np.trapezoid(tomo[i], x=z),
            1.0,
        )

    np.testing.assert_allclose(
        sum(tomo.metadata["bins"]["frac_per_bin"].values()),
        1.0,
        atol=1e-3,
    )


def test_build_photoz_tomography():
    """Tests that photo-z tomography constructs normalized bins."""
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
    assert tomo.metadata["kind"] == "photoz"

    for i in tomo.bin_indices:
        assert tomo.metadata["bins"]["bins_norms"][i] > 0.0
        assert 0.0 <= tomo.metadata["bins"]["frac_per_bin"][i] <= 1.0

        np.testing.assert_allclose(
            np.trapezoid(tomo[i], x=z),
            1.0,
        )


def test_population_fractions_independent_of_bin_normalization():
    """Tests that bin normalization does not change population fractions."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)

    raw = build_tomography(
        z,
        nz,
        edges,
        method="deterministic",
        normalize_bins=False,
    )

    normalized = build_tomography(
        z,
        nz,
        edges,
        method="deterministic",
        normalize_bins=True,
    )

    assert raw.metadata["bins"]["frac_per_bin"] == normalized.metadata["bins"]["frac_per_bin"]

    for i in raw.bin_indices:
        np.testing.assert_allclose(
            normalized.metadata["bins"]["bins_norms"][i],
            np.trapezoid(raw[i], x=z),
        )


def test_photoz_population_fractions_match_independent_integrals():
    """Tests that photo-z population fractions match independent integrals."""
    z = np.linspace(0.0, 3.0, 1001)
    nz = 5.0 * z**2 * np.exp(-z)
    edges = np.linspace(0.2, 2.0, 6)

    params = {
        "scatter_scale": 0.05,
        "mean_offset": 0.01,
        "outlier_frac": 0.2,
        "outlier_scatter_scale": 0.2,
    }

    functional = build_tomography(
        z,
        nz,
        edges,
        method="photoz",
        params=params,
        normalize_input=False,
    )

    from scipy.special import ndtr

    mu_core = z - params["mean_offset"]
    sigma_core = params["scatter_scale"] * (1.0 + z)
    sigma_tail = params["outlier_scatter_scale"] * (1.0 + z)
    outlier_frac = params["outlier_frac"]

    parent_integral = np.trapezoid(nz, x=z)

    for i in functional.bin_indices:
        lo, hi = edges[i], edges[i + 1]

        core_probability = ndtr((hi - mu_core) / sigma_core) - ndtr((lo - mu_core) / sigma_core)
        tail_probability = ndtr((hi - z) / sigma_tail) - ndtr((lo - z) / sigma_tail)

        probability = (1.0 - outlier_frac) * core_probability + outlier_frac * tail_probability

        raw_bin = nz * probability
        expected_integral = np.trapezoid(raw_bin, x=z)
        expected_fraction = expected_integral / parent_integral
        expected_normalized_bin = raw_bin / expected_integral

        np.testing.assert_allclose(
            functional[i],
            expected_normalized_bin,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            functional.metadata["bins"]["bins_norms"][i],
            expected_integral,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            functional.metadata["bins"]["frac_per_bin"][i],
            expected_fraction,
            rtol=1e-12,
            atol=1e-12,
        )


def test_specz_generic_construction_with_response():
    """Tests that spec-z tomography applies response matrices correctly."""
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
        normalize_bins=False,
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
            tomo.metadata["bins"]["bins_norms"][i],
            np.trapezoid(expected[i], x=z),
        )


def test_specz_population_fractions_with_completeness_and_response():
    """Tests that spec-z response mixing preserves selected population."""
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
        normalize_bins=True,
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
            tomo.metadata["bins"]["bins_norms"][i],
            expected_integrals[i],
        )
        np.testing.assert_allclose(
            tomo.metadata["bins"]["frac_per_bin"][i],
            expected_fractions[i],
        )

    np.testing.assert_allclose(
        sum(tomo.metadata["bins"]["frac_per_bin"].values()),
        sum(true_integrals) / parent_integral,
    )


def test_build_tomography_default_deterministic():
    """Tests that deterministic assignment is the default method."""
    z = np.linspace(0.0, 2.0, 501)
    nz = np.ones_like(z)
    edges = np.array([0.0, 1.0, 2.0])

    tomo = build_tomography(
        z,
        nz,
        edges,
        normalize_bins=False,
        normalize_input=False,
    )

    expected = deterministic_assignment(z, edges)

    assert tomo.n_bins == 2
    assert tomo.metadata["kind"] == "deterministic"
    assert tomo.metadata["inputs"]["params"] == {}

    for i in tomo.bin_indices:
        np.testing.assert_array_equal(
            tomo[i],
            nz * expected[i],
        )


def test_build_tomography_specz_method():
    """Tests that built-in spec-z construction matches explicit response mixing."""
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
        normalize_bins=False,
        normalize_input=False,
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
            tomo.metadata["bins"]["bins_norms"][i],
            np.trapezoid(expected[i], x=z),
        )

    assert tomo.metadata["kind"] == "specz"


def test_build_tomography_photoz_method():
    """Tests that built-in photo-z construction matches direct assignment."""
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
        normalize_bins=False,
        normalize_input=False,
    )

    expected_selection = photoz_assignment(z, edges, **params)

    for i in tomo.bin_indices:
        np.testing.assert_allclose(
            tomo[i],
            nz * expected_selection[i],
        )

    assert tomo.metadata["kind"] == "photoz"
    assert tomo.metadata["inputs"]["params"] == params


def test_build_tomography_specz_population_conservation():
    """Tests that spec-z response mixing preserves total selected population."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)

    baseline = build_tomography(
        z,
        nz,
        edges,
        method="specz",
        params={"completeness": 0.8},
        normalize_bins=False,
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
        normalize_bins=False,
    )

    np.testing.assert_allclose(
        sum(mixed.metadata["bins"]["frac_per_bin"].values()),
        sum(baseline.metadata["bins"]["frac_per_bin"].values()),
        rtol=1e-12,
        atol=1e-14,
    )


def test_build_tomography_custom_assignment():
    """Tests that custom assignment functions remain supported."""
    z = np.linspace(0.0, 2.0, 501)
    nz = np.exp(-z)
    edges = np.array([0.0, 1.0, 2.0])

    custom = build_tomography(
        z,
        nz,
        edges,
        deterministic_assignment,
        normalize_bins=False,
    )

    builtin = build_tomography(
        z,
        nz,
        edges,
        method="deterministic",
        normalize_bins=False,
    )

    assert custom.metadata["kind"] == "custom"

    for i in custom.bin_indices:
        np.testing.assert_allclose(custom[i], builtin[i])


def test_build_tomography_rejects_unknown_method():
    """Tests that unknown assignment methods raise a ValueError."""
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
    """Tests that built-in models reject unknown parameter names."""
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
    """Tests that custom assignment and built-in method cannot be combined."""
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
    """Tests that custom assignments reject model parameters."""
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
    """Tests that deterministic assignment rejects unused parameters."""
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


@pytest.mark.parametrize("method", ["deterministic", "photoz", "specz"])
@pytest.mark.parametrize("normalize", [False, True])
def test_tomography_canonical_metadata(method, normalize):
    """Tests that built-in assignment methods use the canonical metadata schema."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = 3.0 * np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)

    tomo = build_tomography(
        z,
        nz,
        edges,
        method=method,
        normalize_bins=normalize,
        include_metadata=True,
    )

    metadata = tomo.metadata
    bins_meta = metadata["bins"]

    assert metadata["kind"] == method
    assert bins_meta["indices"] == tomo.bin_indices
    assert bins_meta["n_bins"] == tomo.n_bins
    assert bins_meta["normalize_bins"] is normalize
    assert metadata["inputs"]["normalize_bins"] is normalize

    np.testing.assert_allclose(bins_meta["bin_edges"], edges)
    np.testing.assert_allclose(metadata["parent_nz"]["values"], tomo.nz)

    assert metadata["parent_nz"]["norm"] == pytest.approx(np.trapezoid(tomo.nz, x=z))

    assert set(bins_meta["bins_norms"]) == set(tomo.bin_indices)
    assert set(bins_meta["frac_per_bin"]) == set(tomo.bin_indices)

    for i in tomo.bin_indices:
        assert bins_meta["frac_per_bin"][i] == pytest.approx(
            bins_meta["bins_norms"][i] / metadata["parent_nz"]["norm"]
        )

    assert "bin_integrals" not in metadata
    assert "parent_integral" not in metadata
    assert "frac_per_bin" not in metadata


@pytest.mark.parametrize("method", ["deterministic", "photoz", "specz"])
def test_tomography_without_metadata(method):
    """Tests that disabling metadata does not change numerical tomography results."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)

    with_metadata = build_tomography(
        z,
        nz,
        edges,
        method=method,
        include_metadata=True,
    )

    without_metadata = build_tomography(
        z,
        nz,
        edges,
        method=method,
        include_metadata=False,
    )

    assert without_metadata.metadata == {}

    np.testing.assert_array_equal(without_metadata.z, with_metadata.z)
    np.testing.assert_array_equal(without_metadata.nz, with_metadata.nz)

    for i in with_metadata.bin_indices:
        np.testing.assert_array_equal(
            without_metadata[i],
            with_metadata[i],
        )


@pytest.mark.parametrize("method", ["deterministic", "photoz", "specz"])
def test_build_tomography_normalization_defaults(method):
    """Tests that the parent and individual bins are normalized by default."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = 5.0 * np.exp(-z)
    edges = np.linspace(0.0, 2.0, 5)
    original = nz.copy()

    tomo = build_tomography(
        z,
        nz,
        edges,
        method=method,
    )

    assert np.trapezoid(tomo.nz, x=z) == pytest.approx(1.0)

    for curve in tomo.bins.values():
        assert np.trapezoid(curve, x=z) == pytest.approx(1.0)

    assert tomo.metadata["inputs"]["normalize_input"] is True
    assert tomo.metadata["inputs"]["normalize_bins"] is True

    np.testing.assert_array_equal(nz, original)
