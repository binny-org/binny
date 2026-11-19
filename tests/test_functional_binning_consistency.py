"""Numerical consistency of tomographic construction and assignment models."""

import numpy as np
import pytest

from binny.binning.edges import resolve_bin_edges
from binny.binning.photoz import photoz_assignment
from binny.binning.specz import specz_assignment
from binny.tomography.core import build_tomography
from binny.utils.normalization import integrate_1d, normalize_1d


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
    """Tests that photo-z bins agree with direct assignment calculations."""
    z, nz, edges = sample

    functional = build_tomography(
        z,
        nz,
        edges,
        method="photoz",
        params=params,
        normalize_input=True,
        normalize_bins=normalize,
    )

    parent = normalize_1d(z, nz)
    probabilities = photoz_assignment(z, edges, **params)
    expected = {}

    for i, probability in enumerate(probabilities):
        raw = parent * probability
        area = integrate_1d(z, raw)
        expected[i] = raw / area if normalize and area > 0.0 else raw

    for i in functional.bin_indices:
        np.testing.assert_allclose(
            functional.bins[i],
            expected[i],
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
    """Tests that spec-z bins agree with direct assignment calculations."""
    z, nz, edges = sample

    functional = build_tomography(
        z,
        nz,
        edges,
        method="specz",
        params=params,
        normalize_input=True,
        normalize_bins=normalize,
    )

    parent = normalize_1d(z, nz)
    probabilities = specz_assignment(z, edges, **params)
    expected = {}

    for i, probability in enumerate(probabilities):
        raw = parent * probability
        area = integrate_1d(z, raw)
        expected[i] = raw / area if normalize and area > 0.0 else raw

    for i in functional.bin_indices:
        np.testing.assert_allclose(
            functional.bins[i],
            expected[i],
            rtol=1e-12,
            atol=1e-14,
        )


@pytest.mark.parametrize("method", ["photoz", "specz"])
@pytest.mark.parametrize(
    "bins",
    [
        {"edges": [0.2, 0.6, 1.0, 1.5, 2.0]},
        {
            "scheme": "equidistant",
            "n_bins": 5,
            "range": (0.2, 2.0),
        },
        {
            "scheme": "equal_number",
            "n_bins": 5,
            "range": (0.2, 2.0),
        },
        {
            "scheme": "power",
            "n_bins": 5,
            "range": (0.2, 2.0),
            "params": {"power": 2.0},
        },
        {
            "segments": [
                {
                    "scheme": "equidistant",
                    "n_bins": 2,
                    "x_min": 0.2,
                    "x_max": 1.0,
                },
                {
                    "scheme": "equal_number",
                    "n_bins": 3,
                    "x_min": 1.0,
                    "x_max": 2.0,
                },
            ],
        },
    ],
)
@pytest.mark.parametrize("normalize_input", [False, True])
@pytest.mark.parametrize("normalize_bins", [False, True])
@pytest.mark.parametrize("norm_method", ["trapezoid", "simpson"])
def test_generated_binning_consistency(
    sample,
    method,
    bins,
    normalize_input,
    normalize_bins,
    norm_method,
):
    """Tests that generated bins agree with direct numerical construction."""
    z, nz, _ = sample

    if method == "photoz":
        assignment = photoz_assignment
        params = {
            "scatter_scale": 0.05,
            "mean_offset": 0.01,
            "outlier_frac": 0.15,
            "outlier_scatter_scale": 0.12,
        }
    else:
        assignment = specz_assignment
        params = {
            "completeness": 0.9,
            "catastrophic_frac": 0.1,
            "specz_scatter": 0.02,
        }

    functional = build_tomography(
        z,
        nz,
        bins=bins,
        method=method,
        params=params,
        normalize_input=normalize_input,
        normalize_bins=normalize_bins,
        norm_method=norm_method,
        include_metadata=True,
    )

    parent = normalize_1d(z, nz, method=norm_method) if normalize_input else nz

    edges = resolve_bin_edges(
        axis=z,
        density=parent,
        bins=bins,
        norm_method=norm_method,
        normalize_equal_number_weights=method == "photoz",
    )

    probabilities = assignment(z, edges, **params)
    parent_integral = integrate_1d(z, parent, method=norm_method)

    np.testing.assert_allclose(
        functional.metadata["bins"]["bin_edges"],
        edges,
        rtol=1e-12,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        functional.nz,
        parent,
        rtol=1e-12,
        atol=1e-14,
    )

    np.testing.assert_allclose(
        functional.metadata["parent_nz"]["norm"],
        parent_integral,
        rtol=1e-12,
        atol=1e-14,
    )

    assert functional.bin_indices == list(range(len(edges) - 1))

    for i, probability in enumerate(probabilities):
        raw = parent * probability
        area = integrate_1d(z, raw, method=norm_method)
        expected = raw / area if normalize_bins and area > 0.0 else raw

        np.testing.assert_allclose(
            functional.bins[i],
            expected,
            rtol=1e-12,
            atol=1e-14,
        )

        np.testing.assert_allclose(
            functional.metadata["bins"]["bins_norms"][i],
            area,
            rtol=1e-12,
            atol=1e-14,
        )

        np.testing.assert_allclose(
            functional.metadata["bins"]["frac_per_bin"][i],
            area / parent_integral,
            rtol=1e-12,
            atol=1e-14,
        )


@pytest.mark.parametrize(
    "scheme",
    ["equidistant", "equal_number", "generalized"],
)
def test_independent_photoz_grid_consistency(sample, scheme):
    """Tests that generic binning preserves independent photo-z edge grids."""
    z, nz, _ = sample

    z_ph = np.linspace(0.1, 2.5, 701)
    nz_ph = z_ph**2 * np.exp(-z_ph / 0.35)

    bins = {
        "scheme": scheme,
        "n_bins": 5,
        "range": (0.2, 2.0),
    }

    params = {"scatter_scale": 0.05}

    functional = build_tomography(
        z,
        nz,
        bins=bins,
        method="photoz",
        params=params,
        z_ph=z_ph,
        nz_ph=nz_ph,
        include_metadata=True,
    )

    edges = resolve_bin_edges(
        axis=z_ph,
        density=nz_ph,
        bins=bins,
        equal_number_axis=z_ph,
        equal_number_weights=nz_ph,
        selection_axis=z_ph,
        selection_density=nz_ph,
        normalize_equal_number_weights=True,
    )

    parent = normalize_1d(z, nz)
    probabilities = photoz_assignment(z, edges, **params)

    np.testing.assert_allclose(
        functional.metadata["bins"]["bin_edges"],
        edges,
        rtol=1e-12,
        atol=1e-14,
    )

    np.testing.assert_array_equal(functional.z, z)
    assert functional.bin_indices == list(range(len(edges) - 1))

    for i, probability in enumerate(probabilities):
        raw = parent * probability
        area = integrate_1d(z, raw)
        expected = raw / area if area > 0.0 else raw

        np.testing.assert_allclose(
            functional.bins[i],
            expected,
            rtol=1e-12,
            atol=1e-14,
        )


def test_generic_builder_requires_edge_definition(sample):
    """Tests that generic construction requires exactly one edge definition."""
    z, nz, edges = sample

    with pytest.raises(ValueError, match="exactly one"):
        build_tomography(z, nz)

    with pytest.raises(ValueError, match="exactly one"):
        build_tomography(
            z,
            nz,
            edges=edges,
            bins={"edges": edges},
        )


def test_generic_builder_requires_paired_photoz_inputs(sample):
    """Tests that independent photo-z coordinates and densities are paired."""
    z, nz, edges = sample

    with pytest.raises(ValueError, match="both z_ph and nz_ph"):
        build_tomography(
            z,
            nz,
            edges,
            method="photoz",
            z_ph=z,
        )

    with pytest.raises(ValueError, match="both z_ph and nz_ph"):
        build_tomography(
            z,
            nz,
            edges,
            method="photoz",
            nz_ph=nz,
        )


def test_generic_builder_rejects_photoz_inputs_for_specz(sample):
    """Tests that spec-z construction rejects independent photo-z inputs."""
    z, nz, edges = sample

    with pytest.raises(ValueError, match="require method='photoz'"):
        build_tomography(
            z,
            nz,
            edges,
            method="specz",
            z_ph=z,
            nz_ph=nz,
        )


def test_generic_builder_rejects_outside_specz_edges(sample):
    """Tests that spec-z boundaries must lie within the parent grid."""
    z, nz, _ = sample

    with pytest.raises(ValueError, match="must lie within"):
        build_tomography(
            z,
            nz,
            edges=np.array([-0.1, 1.0, 2.0]),
            method="specz",
        )


def test_generic_builder_allows_outside_photoz_edges(sample):
    """Tests that photo-z boundaries may extend beyond the parent grid."""
    z, nz, _ = sample

    result = build_tomography(
        z,
        nz,
        edges=np.array([-0.1, 1.0, 3.1]),
        method="photoz",
        params={"scatter_scale": 0.05},
    )

    assert result.n_bins == 2


@pytest.mark.parametrize("include_metadata", [False, True])
@pytest.mark.parametrize("include_diagnostics", [False, True])
def test_generic_photoz_diagnostics(
    sample,
    include_metadata,
    include_diagnostics,
):
    """Tests that generic photo-z diagnostics preserve assignment probabilities."""
    z, nz, edges = sample

    params = {
        "scatter_scale": 0.05,
        "mean_offset": 0.01,
        "outlier_frac": 0.15,
        "outlier_scatter_scale": 0.12,
    }

    result = build_tomography(
        z,
        nz,
        edges,
        method="photoz",
        params=params,
        include_metadata=include_metadata,
        include_photoz_diagnostics=include_diagnostics,
    )

    if not include_metadata and not include_diagnostics:
        assert result.metadata == {}
        return

    assert result.metadata["kind"] == "photoz"

    if not include_diagnostics:
        assert "photoz_diagnostics" not in result.metadata
        return

    diagnostics = result.metadata["photoz_diagnostics"]
    probabilities = photoz_assignment(z, edges, **params)

    np.testing.assert_array_equal(
        diagnostics["observed_bin_edges"],
        edges,
    )

    np.testing.assert_array_equal(
        diagnostics["true_z"],
        z,
    )

    for i in result.bin_indices:
        np.testing.assert_allclose(
            diagnostics["assignment_probabilities"][i],
            probabilities[i],
            rtol=1e-12,
            atol=1e-14,
        )


@pytest.mark.parametrize(
    ("params", "response"),
    [
        ({}, "M_total = M_cat"),
        ({"specz_scatter": 0.0}, "M_total = M_cat"),
        ({"specz_scatter": 0.01}, "M_total = M_scatter @ M_cat"),
        ({"sigma0": 0.01}, "M_total = M_scatter @ M_cat"),
        ({"sigma1": 0.01}, "M_total = M_scatter @ M_cat"),
    ],
)
def test_generic_specz_response_metadata(sample, params, response):
    """Tests that generic spec-z metadata identifies the response prescription."""
    z, nz, edges = sample

    result = build_tomography(
        z,
        nz,
        edges,
        method="specz",
        params=params,
    )

    inputs = result.metadata["inputs"]

    assert inputs["response"] == response
    assert inputs["bin_edges_space"] == "truez"
    assert inputs["response_matrix_provided"] is False


def test_generic_specz_per_bin_response_metadata(sample):
    """Tests that metadata identifies nonzero per-bin measurement scatter."""
    z, nz, edges = sample

    scatter = np.zeros(len(edges) - 1)
    scatter[len(scatter) // 2] = 0.01

    result = build_tomography(
        z,
        nz,
        edges,
        method="specz",
        params={"specz_scatter": scatter},
    )

    assert result.metadata["inputs"]["response"] == "M_total = M_scatter @ M_cat"


def test_generic_specz_explicit_response_metadata(sample):
    """Tests that metadata identifies an explicitly supplied response matrix."""
    z, nz, edges = sample

    result = build_tomography(
        z,
        nz,
        edges,
        method="specz",
        params={"response_matrix": np.eye(len(edges) - 1)},
    )

    assert result.metadata["inputs"]["response_matrix_provided"] is True


@pytest.mark.parametrize("method", ["deterministic", "photoz", "specz"])
@pytest.mark.parametrize("include_metadata", [False, True])
def test_generic_builder_saves_metadata(
    sample,
    tmp_path,
    method,
    include_metadata,
):
    """Tests that metadata saving is independent of returning metadata."""
    z, nz, edges = sample
    path = tmp_path / "metadata.txt"

    result = build_tomography(
        z,
        nz,
        edges,
        method=method,
        include_metadata=include_metadata,
        save_metadata_path=str(path),
    )

    assert path.is_file()
    assert "parent_nz:" in path.read_text(encoding="utf-8")
    assert bool(result.metadata) is include_metadata


def test_generic_builder_saves_photoz_diagnostics(sample, tmp_path):
    """Tests that saved metadata includes requested photo-z diagnostics."""
    z, nz, edges = sample
    path = tmp_path / "metadata.txt"

    result = build_tomography(
        z,
        nz,
        edges,
        method="photoz",
        params={"scatter_scale": 0.05},
        include_metadata=False,
        include_photoz_diagnostics=True,
        save_metadata_path=str(path),
    )

    assert "photoz_diagnostics" in result.metadata
    assert "photoz_diagnostics:" in path.read_text(encoding="utf-8")
    assert "assignment_probabilities:" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("method", ["deterministic", "specz"])
def test_generic_builder_rejects_incompatible_diagnostics(sample, method):
    """Tests that photo-z diagnostics require photometric-redshift assignment."""
    z, nz, edges = sample

    with pytest.raises(ValueError, match="Photo-z diagnostics require"):
        build_tomography(
            z,
            nz,
            edges,
            method=method,
            include_photoz_diagnostics=True,
        )
