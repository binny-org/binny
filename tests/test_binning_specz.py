"""Unit tests for spectroscopic-redshift binning and response models."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from binny.binning.assignments import specz_selection_in_bin
from binny.binning.specz import (
    apply_response_matrix,
    build_specz_response_matrix,
    specz_gaussian_response_matrix,
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
    """Tests that helper returns strictly increasing 4-bin edges spanning [0, 2]."""
    return np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=float)


def _assert_col_stochastic(matrix: np.ndarray, *, rtol: float = 1e-6, atol: float = 1e-10):
    """Tests that helper asserts a matrix is finite, non-negative, and column-stochastic."""
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
    with pytest.raises(ValueError, match=r"completeness must be in \[0, 1\]"):
        specz_selection_in_bin(z, 0.0, 1.0, completeness=-0.1)
    with pytest.raises(ValueError, match=r"completeness must be in \[0, 1\]"):
        specz_selection_in_bin(z, 0.0, 1.0, completeness=1.1)


def test_response_identity_when_all_zero_catastrophic():
    """Tests that zero catastrophic fraction returns the identity response."""
    matrix = build_specz_response_matrix(4, catastrophic_frac=0.0)
    assert np.allclose(matrix, np.eye(4))
    _assert_col_stochastic(matrix)


def test_response_explicit_matrix_valid():
    """Tests that a valid explicit response matrix is accepted unchanged."""
    matrix0 = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=float)
    matrix = build_specz_response_matrix(2, response_matrix=matrix0)
    assert matrix.dtype == np.float64
    assert np.allclose(matrix, matrix0)
    _assert_col_stochastic(matrix)


def test_response_explicit_matrix_invalid_shape():
    """Tests that an explicit response matrix with wrong shape raises ValueError."""
    matrix0 = np.eye(3)
    with pytest.raises(ValueError):
        build_specz_response_matrix(2, response_matrix=matrix0)


def test_response_explicit_matrix_invalid_column_sums():
    """Tests that an explicit response matrix with invalid column sums
    raises."""
    matrix0 = np.array([[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError):
        build_specz_response_matrix(2, response_matrix=matrix0)


def test_response_catastrophic_frac_bounds():
    """Tests that catastrophic_frac outside [0, 1] raises a ValueError."""
    with pytest.raises(ValueError, match=r"catastrophic_frac must be in \[0, 1\]"):
        build_specz_response_matrix(3, catastrophic_frac=-0.1)
    with pytest.raises(ValueError, match=r"catastrophic_frac must be in \[0, 1\]"):
        build_specz_response_matrix(3, catastrophic_frac=1.1)


def test_response_uniform_model_properties():
    """Tests that uniform leakage distributes catastrophes equally to all
    bins."""
    matrix = build_specz_response_matrix(4, catastrophic_frac=0.2, leakage_model="uniform")
    _assert_col_stochastic(matrix)
    assert np.allclose(np.diag(matrix), 0.8)
    off = 0.2 / 3
    for j in range(4):
        for i in range(4):
            if i != j:
                assert np.isclose(matrix[i, j], off)


def test_response_neighbor_model_properties_middle_column():
    """Tests that neighbor leakage splits catastrophes across adjacent bins
    in the interior."""
    matrix = build_specz_response_matrix(4, catastrophic_frac=0.2, leakage_model="neighbor")
    _assert_col_stochastic(matrix)
    assert np.isclose(matrix[1, 1], 0.8)
    assert np.isclose(matrix[0, 1], 0.1)
    assert np.isclose(matrix[2, 1], 0.1)
    assert np.isclose(matrix[3, 1], 0.0)


def test_response_neighbor_model_edge_column():
    """Tests that neighbor leakage sends catastrophes to the single adjacent
    bin at edges."""
    matrix = build_specz_response_matrix(4, catastrophic_frac=0.2, leakage_model="neighbor")
    _assert_col_stochastic(matrix)
    assert np.isclose(matrix[0, 0], 0.8)
    assert np.isclose(matrix[1, 0], 0.2)
    assert np.isclose(matrix[2, 0], 0.0)
    assert np.isclose(matrix[3, 0], 0.0)


def test_response_gaussian_requires_positive_sigma():
    """Tests that gaussian leakage rejects non-positive leakage_sigma."""
    with pytest.raises(ValueError, match=r"leakage_sigma must be > 0"):
        build_specz_response_matrix(
            3,
            catastrophic_frac=0.2,
            leakage_model="gaussian",
            leakage_sigma=0.0,
        )


def test_response_gaussian_is_col_stochastic_and_nonnegative():
    """Tests that gaussian leakage produces a non-negative column-stochastic response."""
    matrix = build_specz_response_matrix(
        4,
        catastrophic_frac=[0.0, 0.1, 0.2, 0.3],
        leakage_model="gaussian",
        leakage_sigma=1.2,
    )
    _assert_col_stochastic(matrix)
    assert np.all(matrix >= -1e-15)


def test_response_n_bins_1():
    """Tests that the n_bins=1 case returns the 1x1 identity matrix."""
    matrix = build_specz_response_matrix(1, catastrophic_frac=0.5, leakage_model="uniform")
    assert matrix.shape == (1, 1)
    assert np.allclose(matrix, [[1.0]])
    _assert_col_stochastic(matrix)


def test_apply_response_matrix_identity_no_change():
    """Tests that applying the identity response matrix leaves bins unchanged."""
    z, nz = _toy_z_nz()
    edges = _edges_4bins()
    true_bins: dict[int, np.ndarray] = {}
    for j in range(4):
        sel = specz_selection_in_bin(z, edges[j], edges[j + 1], completeness=1.0)
        true_bins[j] = nz * sel

    matrix = np.eye(4)
    obs = apply_response_matrix(true_bins, matrix)
    for j in range(4):
        assert np.allclose(obs[j], true_bins[j])


def test_apply_response_matrix_requires_contiguous_keys():
    """Tests that bins must contain exactly keys 0..n_bins-1."""
    z, nz = _toy_z_nz()
    bins = {0: nz, 2: nz}
    matrix = np.eye(2)
    with pytest.raises(ValueError, match=r"bins must contain exactly keys 0\.\.1"):
        apply_response_matrix(bins, matrix)


def test_apply_response_matrix_shape_mismatch_errors():
    """Tests that a matrix shape mismatch raises a ValueError."""
    z, nz = _toy_z_nz()
    bins = {0: nz, 1: nz}
    matrix = np.eye(3)
    with pytest.raises(ValueError):
        apply_response_matrix(bins, matrix)


def test_gaussian_scatter_identity_when_sigma0_sigma1_zero():
    """Tests that sigma0=sigma1=0 returns identity for sigma0_plus_sigma1_1pz."""
    z, _ = _toy_z_nz()
    edges = _edges_4bins()
    matrix = specz_gaussian_response_matrix(
        z_arr=z,
        bin_edges=edges,
        specz_scatter=None,
        model="sigma0_plus_sigma1_1pz",
        sigma0=0.0,
        sigma1=0.0,
    )
    assert np.allclose(matrix, np.eye(4))
    _assert_col_stochastic(matrix)


def test_gaussian_scatter_requires_sigma0_plus_sigma1_when_none():
    """Tests that model must be supported when specz_scatter is None."""
    z, _ = _toy_z_nz()
    edges = _edges_4bins()

    matrix = specz_gaussian_response_matrix(
        z_arr=z,
        bin_edges=edges,
        specz_scatter=None,
        model="const",
        sigma0=1e-4,
        sigma1=0.0,
    )
    assert matrix.shape == (4, 4)
    _assert_col_stochastic(matrix)


def test_gaussian_scatter_returns_col_stochastic_matrix():
    """Tests that gaussian measurement scatter returns a finite
    column-stochastic matrix."""
    z, _ = _toy_z_nz()
    edges = _edges_4bins()
    matrix = specz_gaussian_response_matrix(
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
    matrix = specz_gaussian_response_matrix(
        z_arr=z,
        bin_edges=edges,
        specz_scatter=[5e-4, 5e-4, 5e-4, 5e-4],
        model="const",
    )
    _assert_col_stochastic(matrix)
    assert np.allclose(matrix[:, 2], np.array([0.0, 0.0, 1.0, 0.0]))


def test_apply_response_matrix_requires_same_bin_shapes():
    """Tests that all bins must have identical array shapes."""
    z, nz = _toy_z_nz()
    bins = {0: nz, 1: nz[:-1]}
    matrix = np.eye(2)
    with pytest.raises(ValueError, match=r"All bin arrays must have the same shape"):
        apply_response_matrix(bins, matrix)


def test_response_rejects_unknown_leakage_model() -> None:
    """Tests that build_specz_response_matrix rejects unknown leakage_model."""
    with pytest.raises(ValueError, match=r"leakage_model must be"):
        build_specz_response_matrix(3, catastrophic_frac=0.2, leakage_model="nope")  # type: ignore[arg-type]


def test_gaussian_leakage_sigma_allows_fallback_when_weights_underflow() -> None:
    """Tests that gaussian leakage can fall back to neighbor-like behavior."""
    matrix = build_specz_response_matrix(
        4,
        catastrophic_frac=0.3,
        leakage_model="gaussian",
        leakage_sigma=1e-300,
    )
    _assert_col_stochastic(matrix)
    # For interior column j=1, fallback should split to neighbors (0,2)
    assert matrix[0, 1] > 0.0
    assert matrix[2, 1] > 0.0


def test_gaussian_scatter_rejects_nonpositive_specz_scatter() -> None:
    """Tests that negative specz_scatter is rejected (zero is allowed)."""
    z, _ = _toy_z_nz()
    edges = _edges_4bins()

    m0 = specz_gaussian_response_matrix(
        z_arr=z,
        bin_edges=edges,
        specz_scatter=[1e-3, 0.0, 1e-3, 1e-3],
        model="const",
    )
    assert m0.shape == (4, 4)
    _assert_col_stochastic(m0)

    with pytest.raises(ValueError, match=r"specz_scatter must be >= 0"):
        specz_gaussian_response_matrix(
            z_arr=z,
            bin_edges=edges,
            specz_scatter=[1e-3, -1e-6, 1e-3, 1e-3],
            model="const",
        )


def test_gaussian_scatter_rejects_negative_sigma0_sigma1() -> None:
    """Tests that specz_gaussian_response_matrix rejects negative sigma0/sigma1."""
    z, _ = _toy_z_nz()
    edges = _edges_4bins()
    with pytest.raises(ValueError, match=r"sigma0 and sigma1 must be >="):
        specz_gaussian_response_matrix(
            z_arr=z,
            bin_edges=edges,
            specz_scatter=None,
            model="sigma0_plus_sigma1_1pz",
            sigma0=-1e-4,
            sigma1=0.0,
        )


def test_response_warns_when_zero_catastrophic_frac_and_nondefault_leakage_params() -> None:
    """Tests that zero catastrophic_frac warns when leakage parameters are also provided."""
    with pytest.warns(
        RuntimeWarning,
        match=r"catastrophic_frac is zero for all bins, so catastrophic leakage parameters",
    ):
        matrix = build_specz_response_matrix(
            4,
            catastrophic_frac=0.0,
            leakage_model="gaussian",
            leakage_sigma=2.5,
        )

    assert np.allclose(matrix, np.eye(4))
    _assert_col_stochastic(matrix)


def test_response_does_not_warn_when_zero_catastrophic_frac_and_default_leakage_params() -> None:
    """Tests that zero catastrophic_frac does not warn for default leakage settings."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        matrix = build_specz_response_matrix(
            4,
            catastrophic_frac=0.0,
            leakage_model="neighbor",
            leakage_sigma=1.0,
        )

    assert len(record) == 0
    assert np.allclose(matrix, np.eye(4))
    _assert_col_stochastic(matrix)


def test_response_warns_when_zero_catastrophic_frac_and_per_bin_nondefault_leakage_sigma() -> None:
    """Tests that zero catastrophic_frac warns when per-bin leakage_sigma is non-default."""
    with pytest.warns(
        RuntimeWarning,
        match=r"catastrophic_frac is zero for all bins, so catastrophic leakage parameters",
    ):
        matrix = build_specz_response_matrix(
            4,
            catastrophic_frac=[0.0, 0.0, 0.0, 0.0],
            leakage_model="neighbor",
            leakage_sigma=[1.0, 1.0, 2.0, 1.0],
        )

    assert np.allclose(matrix, np.eye(4))
    _assert_col_stochastic(matrix)
