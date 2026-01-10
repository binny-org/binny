"""Unit tests for ``binny.ztomo.specz`` module."""

from __future__ import annotations

import numpy as np
import pytest

from binny.ztomo.specz import (
    apply_misassignment_matrix,
    build_specz_bins,
    build_specz_misassignment_matrix,
    specz_gaussian_misassignment_matrix,
    specz_selection_in_bin,
)


def _toy_z_nz(nz_kind: str = "smooth"):
    """Tests that helper returns a valid (z, nz) pair for parametrized inputs."""
    z = np.linspace(0.0, 2.0, 501)
    if nz_kind == "smooth":
        nz = z**2 * np.exp(-z)
    elif nz_kind == "flat":
        nz = np.ones_like(z)
    else:
        raise ValueError("unknown nz_kind")
    return z, nz


def _edges_4bins():
    """Tests that helper returns strictly increasing 4-bin edges spanning
    [0, 2]."""
    return np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)


def _assert_col_stochastic(
    matrix: np.ndarray, *, rtol: float = 1e-6, atol: float = 1e-10
):
    """Tests that helper asserts a matrix is finite, non-negative,
    and column-stochastic."""
    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert np.all(np.isfinite(matrix))
    assert np.all(matrix >= -1e-15)
    col_sums = matrix.sum(axis=0)
    assert np.allclose(col_sums, 1.0, rtol=rtol, atol=atol)


def test_specz_selection_basic_left_inclusive_right_exclusive():
    """Tests that selection includes left edge and excludes right edge by default."""
    z = np.array([0.0, 0.49, 0.5, 0.5001, 1.0])
    sel = specz_selection_in_bin(z, 0.5, 1.0, completeness=1.0, inclusive_right=False)
    assert np.allclose(sel, [0, 0, 1, 1, 0], atol=0, rtol=0)


def test_specz_selection_inclusive_right():
    """Tests that inclusive_right includes the right edge of the interval."""
    z = np.array([0.5, 1.0, 1.0001])
    sel = specz_selection_in_bin(z, 0.5, 1.0, completeness=1.0, inclusive_right=True)
    assert np.allclose(sel, [1, 1, 0], atol=0, rtol=0)


def test_specz_selection_completeness_range():
    """Tests that completeness outside [0, 1] raises a ValueError."""
    z = np.linspace(0, 1, 5)
    with pytest.raises(ValueError, match="completeness must be in \\[0, 1\\]"):
        specz_selection_in_bin(z, 0.0, 1.0, completeness=-0.1)
    with pytest.raises(ValueError, match="completeness must be in \\[0, 1\\]"):
        specz_selection_in_bin(z, 0.0, 1.0, completeness=1.1)


def test_misassignment_identity_when_all_zero_catastrophic():
    """Tests that zero catastrophic fraction returns the identity response."""
    matrix = build_specz_misassignment_matrix(4, catastrophic_frac=0.0)
    assert np.allclose(matrix, np.eye(4))
    _assert_col_stochastic(matrix)


def test_misassignment_explicit_matrix_valid():
    """Tests that a valid explicit response matrix is accepted unchanged."""
    matrix0 = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
    matrix = build_specz_misassignment_matrix(2, misassignment_matrix=matrix0)
    assert matrix.dtype == np.float64
    assert np.allclose(matrix, matrix0)
    _assert_col_stochastic(matrix)


def test_misassignment_explicit_matrix_invalid_shape():
    """Tests that an explicit response matrix with wrong shape raises
    a ValueError."""
    matrix0 = np.eye(3)
    with pytest.raises(ValueError):
        build_specz_misassignment_matrix(2, misassignment_matrix=matrix0)


def test_misassignment_explicit_matrix_invalid_column_sums():
    """Tests that an explicit response matrix with invalid column sums
    raises a ValueError."""
    matrix0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError):
        build_specz_misassignment_matrix(2, misassignment_matrix=matrix0)


def test_misassignment_catastrophic_frac_bounds():
    """Tests that catastrophic_frac outside [0, 1] raises a ValueError."""
    with pytest.raises(ValueError, match="catastrophic_frac must be in \\[0, 1\\]"):
        build_specz_misassignment_matrix(3, catastrophic_frac=-0.1)
    with pytest.raises(ValueError, match="catastrophic_frac must be in \\[0, 1\\]"):
        build_specz_misassignment_matrix(3, catastrophic_frac=1.1)


def test_misassignment_uniform_model_properties():
    """Tests that uniform leakage distributes catastrophes equally to all
    other bins."""
    matrix = build_specz_misassignment_matrix(
        4, catastrophic_frac=0.2, leakage_model="uniform"
    )
    _assert_col_stochastic(matrix)
    assert np.allclose(np.diag(matrix), 0.8)
    off = 0.2 / 3
    for j in range(4):
        for i in range(4):
            if i != j:
                assert np.isclose(matrix[i, j], off)


def test_misassignment_neighbor_model_properties_middle_column():
    """Tests that neighbor leakage splits catastrophes across adjacent bins
    in the interior."""
    matrix = build_specz_misassignment_matrix(
        4, catastrophic_frac=0.2, leakage_model="neighbor"
    )
    _assert_col_stochastic(matrix)
    assert np.isclose(matrix[1, 1], 0.8)
    assert np.isclose(matrix[0, 1], 0.1)
    assert np.isclose(matrix[2, 1], 0.1)
    assert np.isclose(matrix[3, 1], 0.0)


def test_misassignment_neighbor_model_edge_column():
    """Tests that neighbor leakage sends catastrophes to the
    single adjacent bin at edges."""
    matrix = build_specz_misassignment_matrix(
        4, catastrophic_frac=0.2, leakage_model="neighbor"
    )
    _assert_col_stochastic(matrix)
    assert np.isclose(matrix[0, 0], 0.8)
    assert np.isclose(matrix[1, 0], 0.2)
    assert np.isclose(matrix[2, 0], 0.0)
    assert np.isclose(matrix[3, 0], 0.0)


def test_misassignment_gaussian_requires_positive_sigma():
    """Tests that gaussian leakage rejects non-positive leakage_sigma."""
    with pytest.raises(ValueError, match="leakage_sigma must be > 0"):
        build_specz_misassignment_matrix(
            3,
            catastrophic_frac=0.2,
            leakage_model="gaussian",
            leakage_sigma=0.0,
        )


def test_misassignment_gaussian_is_col_stochastic_and_nonnegative():
    """Tests that gaussian leakage produces a non-negative
    column-stochastic response."""
    matrix = build_specz_misassignment_matrix(
        4,
        catastrophic_frac=[0.0, 0.1, 0.2, 0.3],
        leakage_model="gaussian",
        leakage_sigma=1.2,
    )
    _assert_col_stochastic(matrix)
    assert np.all(matrix >= -1e-15)


def test_misassignment_n_bins_1():
    """Tests that the n_bins=1 case returns the 1x1 identity matrix."""
    matrix = build_specz_misassignment_matrix(
        1, catastrophic_frac=0.5, leakage_model="uniform"
    )
    assert matrix.shape == (1, 1)
    assert np.allclose(matrix, [[1.0]])
    _assert_col_stochastic(matrix)


def test_apply_misassignment_matrix_identity_no_change():
    """Tests that applying the identity response matrix leaves bins unchanged."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    true_bins: dict[int, np.ndarray] = {}
    for j in range(4):
        sel = specz_selection_in_bin(z, edges[j], edges[j + 1], completeness=1.0)
        true_bins[j] = nz * sel

    matrix = np.eye(4)
    obs = apply_misassignment_matrix(true_bins, matrix)
    for j in range(4):
        assert np.allclose(obs[j], true_bins[j])


def test_apply_misassignment_matrix_requires_contiguous_keys():
    """Tests that missing a required bin key raises an error during stacking."""
    z, nz = _toy_z_nz()
    bins = {0: nz, 2: nz}
    matrix = np.eye(2)
    with pytest.raises(KeyError):
        apply_misassignment_matrix(bins, matrix)


def test_apply_misassignment_matrix_shape_mismatch_errors():
    """Tests that a matrix shape mismatch raises a ValueError."""
    z, nz = _toy_z_nz()
    bins = {0: nz, 1: nz}
    matrix = np.eye(3)
    with pytest.raises(ValueError):
        apply_misassignment_matrix(bins, matrix)


def test_gaussian_scatter_identity_when_sigma0_sigma1_zero():
    """Tests that sigma0=sigma1=0 returns identity when using
    sigma0_plus_sigma1_1pz."""
    z, _ = _toy_z_nz()
    edges = _edges_4bins()
    matrix = specz_gaussian_misassignment_matrix(
        z_arr=z,
        bin_edges=edges,
        specz_scatter=None,
        model="sigma0_plus_sigma1_1pz",
        sigma0=0.0,
        sigma1=0.0,
    )
    assert np.allclose(matrix, np.eye(4))
    _assert_col_stochastic(matrix)


def test_gaussian_scatter_const_requires_positive_scatter():
    """Tests that const scatter rejects non-positive specz_scatter."""
    z, _ = _toy_z_nz()
    edges = _edges_4bins()
    with pytest.raises(ValueError, match="specz_scatter must be > 0"):
        specz_gaussian_misassignment_matrix(
            z_arr=z,
            bin_edges=edges,
            specz_scatter=0.0,
            model="const",
        )


def test_gaussian_scatter_model_requires_sigma0_plus_sigma1_when_none():
    """Tests that specz_scatter=None requires model='sigma0_plus_sigma1_1pz'."""
    z, _ = _toy_z_nz()
    edges = _edges_4bins()
    with pytest.raises(ValueError, match="model must be 'sigma0_plus_sigma1_1pz'"):
        specz_gaussian_misassignment_matrix(
            z_arr=z,
            bin_edges=edges,
            specz_scatter=None,
            model="const",
            sigma0=1e-4,
            sigma1=0.0,
        )


def test_gaussian_scatter_returns_col_stochastic_matrix():
    """Tests that gaussian measurement scatter returns a finite
    column-stochastic matrix."""
    z, _ = _toy_z_nz()
    edges = _edges_4bins()
    matrix = specz_gaussian_misassignment_matrix(
        z_arr=z,
        bin_edges=edges,
        specz_scatter=[5e-4, 5e-4, 8e-4, 1e-3],
        model="const",
    )
    assert matrix.shape == (4, 4)
    _assert_col_stochastic(matrix)
    assert np.all(matrix >= -1e-15)


def test_gaussian_scatter_empty_bin_support_falls_back_to_identity_column():
    """Tests that missing grid support for a true bin yields an identity
    fallback column."""
    z = np.concatenate([np.linspace(0.0, 0.99, 200), np.linspace(1.5, 2.0, 200)])
    edges = _edges_4bins()
    matrix = specz_gaussian_misassignment_matrix(
        z_arr=z,
        bin_edges=edges,
        specz_scatter=[5e-4, 5e-4, 5e-4, 5e-4],
        model="const",
    )
    _assert_col_stochastic(matrix)
    assert np.allclose(matrix[:, 2], np.array([0.0, 0.0, 1.0, 0.0]))


def test_build_specz_bins_keys_shapes_and_dtype():
    """Tests that build_specz_bins returns contiguous integer keys with correct
    shapes and dtype."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    bins = build_specz_bins(z, nz, edges)
    assert list(bins.keys()) == [0, 1, 2, 3]
    for i in range(4):
        assert bins[i].shape == z.shape
        assert bins[i].dtype == np.float64


def test_build_specz_bins_normalize_input_true_normalize_bins_true_integrals_are_one():
    """Tests that normalize_bins=True yields unit-normalized non-empty bin shapes."""
    z, nz = _toy_z_nz("smooth")
    edges = _edges_4bins()
    bins = build_specz_bins(z, nz, edges, normalize_input=True, normalize_bins=True)
    for i in range(4):
        integ = np.trapezoid(bins[i], x=z)
        assert np.isclose(integ, 1.0, rtol=1e-6, atol=1e-10)


def test_build_specz_bins_no_bin_normalization_respects_completeness():
    """Tests that normalize_bins=False preserves relative normalizations across
    bins."""
    z, nz = _toy_z_nz("smooth")
    edges = _edges_4bins()
    completeness = [1.0, 0.8, 0.6, 0.4]
    bins = build_specz_bins(
        z,
        nz,
        edges,
        completeness=completeness,
        normalize_input=True,
        normalize_bins=False,
    )
    integrals = np.array([np.trapezoid(bins[i], x=z) for i in range(4)])
    assert np.all(integrals >= 0.0)
    assert integrals.sum() <= 1.0 + 1e-8


def test_build_specz_bins_invalid_edges_outside_range():
    """Tests that bin_edges outside the z grid range raises a ValueError."""
    z, nz = _toy_z_nz()
    edges = np.array([-0.1, 0.5, 1.0])
    with pytest.raises(ValueError, match="bin_edges must lie within z-range"):
        build_specz_bins(z, nz, edges)


def test_build_specz_bins_invalid_edges_not_increasing():
    """Tests that non-increasing bin_edges raises a ValueError."""
    z, nz = _toy_z_nz()
    edges = np.array([0.0, 1.0, 0.9, 2.0])
    with pytest.raises(ValueError, match="bin_edges must be strictly increasing"):
        build_specz_bins(z, nz, edges)


def test_build_specz_bins_catastrophic_neighbor_changes_bins():
    """Tests that enabling catastrophic leakage changes observed-bin
    distributions."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    bins0 = build_specz_bins(
        z,
        nz,
        edges,
        catastrophic_frac=0.0,
        leakage_model="neighbor",
        normalize_input=True,
        normalize_bins=False,
    )
    bins1 = build_specz_bins(
        z,
        nz,
        edges,
        catastrophic_frac=0.2,
        leakage_model="neighbor",
        normalize_input=True,
        normalize_bins=False,
    )
    diffs = [np.linalg.norm(bins1[i] - bins0[i]) for i in range(4)]
    assert any(d > 0 for d in diffs)


def test_build_specz_bins_explicit_matrix_overrides_catastrophic_frac():
    """Tests that an explicit response matrix overrides catastrophic_frac
    and leakage_model."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    matrix = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    bins = build_specz_bins(
        z,
        nz,
        edges,
        catastrophic_frac=0.0,
        leakage_model="neighbor",
        misassignment_matrix=matrix,
        normalize_input=True,
        normalize_bins=False,
    )
    mask_01 = (z >= 0.5) & (z < 1.0)
    mask_00 = (z >= 0.0) & (z < 0.5)
    i0 = np.trapezoid(bins[0][mask_01], x=z[mask_01])
    i1 = np.trapezoid(bins[1][mask_00], x=z[mask_00])
    assert i0 > 0.0
    assert i1 > 0.0


def test_build_specz_bins_adds_measurement_scatter_when_enabled_const():
    """Tests that enabling constant measurement scatter changes distributions."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    bins_no = build_specz_bins(
        z,
        nz,
        edges,
        catastrophic_frac=0.05,
        leakage_model="neighbor",
        normalize_bins=False,
    )
    bins_sc = build_specz_bins(
        z,
        nz,
        edges,
        catastrophic_frac=0.05,
        leakage_model="neighbor",
        specz_scatter=[5e-4, 5e-4, 8e-4, 1e-3],
        specz_scatter_model="const",
        normalize_bins=False,
    )
    dist = sum(np.linalg.norm(bins_sc[i] - bins_no[i]) for i in range(4))
    assert dist > 0.0


def test_build_specz_bins_adds_measurement_scatter_when_enabled_sigma0_sigma1():
    """Tests that enabling sigma0_plus_sigma1_1pz measurement scatter
    changes distributions."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    bins_no = build_specz_bins(
        z,
        nz,
        edges,
        catastrophic_frac=0.02,
        leakage_model="neighbor",
        normalize_bins=False,
    )
    bins_sc = build_specz_bins(
        z,
        nz,
        edges,
        catastrophic_frac=0.02,
        leakage_model="neighbor",
        specz_scatter=None,
        specz_scatter_model="sigma0_plus_sigma1_1pz",
        sigma0=1e-4,
        sigma1=2e-4,
        normalize_bins=False,
    )
    dist = sum(np.linalg.norm(bins_sc[i] - bins_no[i]) for i in range(4))
    assert dist > 0.0
