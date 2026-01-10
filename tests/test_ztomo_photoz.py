"""Unit tests for ``binny.ztomo.photoz`` module."""

import numpy as np
import pytest

from binny.ztomo.photoz import build_photoz_bins, true_redshift_distribution


@pytest.fixture
def z_nz():
    """Provides a sample redshift grid and unnormalized n(z) for testing."""
    z = np.linspace(0.0, 3.0, 501)
    nz = z**2 * np.exp(-z)  # deliberately NOT normalized on [0, 3]
    return z, nz


def test_true_redshift_distribution_full_range_returns_parent(z_nz):
    """Tests that true_redshift_distribution with huge bin_min/bin_max"""
    z, nz = z_nz
    nb = true_redshift_distribution(
        z,
        nz,
        bin_min=-1e6,
        bin_max=+1e6,
        scatter_scale=0.05,
        mean_offset=0.0,
        mean_scale=1.0,
        outlier_frac=0.0,
        outlier_scatter_scale=None,
    )
    # Relative error should be tiny; edges are numerically safe with huge bounds.
    rel = np.max(np.abs(nb - nz) / np.maximum(np.abs(nz), 1e-30))
    assert rel < 1e-10


def test_true_redshift_distribution_outlier_disabled_matches_core(z_nz):
    """Tests that true_redshift_distribution with outlier_frac>0 but
    outlier_scatter_scale=None matches the core-only case (outlier_frac=0)."""
    z, nz = z_nz

    core_only = true_redshift_distribution(
        z,
        nz,
        bin_min=0.5,
        bin_max=1.0,
        scatter_scale=0.05,
        mean_offset=0.01,
        mean_scale=1.0,
        outlier_frac=0.0,
        outlier_scatter_scale=None,
    )

    outlier_disabled = true_redshift_distribution(
        z,
        nz,
        bin_min=0.5,
        bin_max=1.0,
        scatter_scale=0.05,
        mean_offset=0.01,
        mean_scale=1.0,
        outlier_frac=0.5,  # nonzero
        outlier_scatter_scale=None,  # disabled
        outlier_mean_offset=0.2,
        outlier_mean_scale=0.9,
    )

    assert np.allclose(outlier_disabled, core_only, rtol=0.0, atol=0.0)


def test_true_redshift_distribution_validates_parameters(z_nz):
    """Tests that true_redshift_distribution raises ValueError."""
    z, nz = z_nz

    with pytest.raises(ValueError, match=r"outlier_frac must lie"):
        true_redshift_distribution(
            z, nz, 0.0, 1.0, scatter_scale=0.1, mean_offset=0.0, outlier_frac=1.1
        )

    with pytest.raises(ValueError, match=r"mean_scale must be > 0"):
        true_redshift_distribution(
            z, nz, 0.0, 1.0, scatter_scale=0.1, mean_offset=0.0, mean_scale=0.0
        )

    with pytest.raises(ValueError, match=r"scatter_scale must be > 0"):
        true_redshift_distribution(
            z, nz, 0.0, 1.0, scatter_scale=0.0, mean_offset=0.0, mean_scale=1.0
        )

    # Outlier validation only triggered when outliers are enabled
    with pytest.raises(ValueError, match=r"outlier_mean_scale must be > 0"):
        true_redshift_distribution(
            z,
            nz,
            0.0,
            1.0,
            scatter_scale=0.1,
            mean_offset=0.0,
            mean_scale=1.0,
            outlier_frac=0.2,
            outlier_scatter_scale=0.2,
            outlier_mean_scale=0.0,
        )

    with pytest.raises(ValueError, match=r"outlier_scatter_scale must be > 0"):
        true_redshift_distribution(
            z,
            nz,
            0.0,
            1.0,
            scatter_scale=0.1,
            mean_offset=0.0,
            mean_scale=1.0,
            outlier_frac=0.2,
            outlier_scatter_scale=0.0,
        )


def test_build_photoz_bins_basic_shapes_and_keys(z_nz):
    """Tests that build_photoz_bins returns correct keys and shapes."""
    z, nz = z_nz
    edges = [0.0, 0.5, 1.0, 1.5]

    bins = build_photoz_bins(
        z,
        nz,
        edges,
        scatter_scale=0.05,
        mean_offset=0.01,
        mean_scale=1.0,
        outlier_frac=0.0,
        outlier_scatter_scale=None,
        normalize_input=True,
        normalize_bins=True,
    )

    assert list(bins.keys()) == [0, 1, 2]
    for k, arr in bins.items():
        assert isinstance(k, int)
        assert arr.shape == z.shape
        assert np.all(np.isfinite(arr))


def test_build_photoz_bins_normalize_bins_integrates_to_one(z_nz):
    """Tests that build_photoz_bins with normalize_bins=True."""
    z, nz = z_nz
    edges = [0.0, 0.5, 1.0, 1.5]

    bins = build_photoz_bins(
        z,
        nz,
        edges,
        scatter_scale=0.05,
        mean_offset=0.01,
        mean_scale=1.0,
        outlier_frac=0.02,
        outlier_scatter_scale=0.15,
        normalize_input=True,
        normalize_bins=True,
    )

    for arr in bins.values():
        integral = np.trapezoid(arr, z)
        assert np.isclose(integral, 1.0, rtol=1e-6, atol=1e-8)


def test_build_photoz_bins_normalize_input_allows_already_normalized():
    """Tests that build_photoz_bins with normalize_input=True
    accepts already-normalized input n(z)."""
    z = np.linspace(0.0, 3.0, 501)
    nz = np.ones_like(z)
    nz = nz / np.trapezoid(nz, z)  # explicitly normalized

    bins = build_photoz_bins(
        z,
        nz,
        bin_edges=[0.0, 0.5, 1.0],
        scatter_scale=0.05,
        mean_offset=0.01,
        normalize_input=True,
        normalize_bins=False,
    )

    assert list(bins.keys()) == [0, 1]
    for arr in bins.values():
        assert arr.shape == z.shape
        assert np.all(np.isfinite(arr))


def test_build_photoz_bins_per_bin_parameters(z_nz):
    """Tests that build_photoz_bins handles per-bin parameter broadcasting."""
    z, nz = z_nz
    edges = [0.0, 0.4, 0.8, 1.2, 1.6]

    bins = build_photoz_bins(
        z,
        nz,
        edges,
        scatter_scale=[0.04, 0.05, 0.06, 0.05],
        mean_offset=[0.00, 0.01, 0.02, 0.02],
        mean_scale=[1.00, 1.00, 0.98, 0.98],
        outlier_frac=[0.00, 0.01, 0.02, 0.01],
        outlier_scatter_scale=[None, 0.20, 0.20, 0.15],
        normalize_input=True,
        normalize_bins=True,
    )

    assert list(bins.keys()) == [0, 1, 2, 3]
    for arr in bins.values():
        assert arr.shape == z.shape


def test_build_photoz_bins_broadcasting_wrong_length_raises(z_nz):
    """Tests that build_photoz_bins raises ValueError for wrong-length
    per-bin params."""
    z, nz = z_nz
    edges = [0.0, 0.5, 1.0, 1.5]

    # scatter_scale has wrong length (2 instead of 3)
    with pytest.raises(ValueError):
        build_photoz_bins(
            z,
            nz,
            edges,
            scatter_scale=[0.05, 0.06],
            mean_offset=0.01,
            normalize_input=True,
        )


def test_build_photoz_bins_no_normalize_bins_sums_to_parent_when_bins_cover_range(z_nz):
    """Test that build_photoz_bins with normalize_bins=False and
    wide bin coverage sums to the normalized parent n(z)."""
    z, nz = z_nz

    # Use wide bin coverage so the selection partitions observed-z space well.
    # This makes sum over bins approximate the parent distribution
    # (up to edge effects).
    edges = [-5.0, -1.0, 0.0, 1.0, 2.0, 5.0]

    bins = build_photoz_bins(
        z,
        nz,
        edges,
        scatter_scale=0.05,
        mean_offset=0.0,
        mean_scale=1.0,
        outlier_frac=0.0,
        outlier_scatter_scale=None,
        normalize_input=True,
        normalize_bins=False,
    )

    # Each individual bin should integrate to <= 1 (they're fractional slices)
    integrals = [np.trapezoid(arr, z) for arr in bins.values()]
    assert all(0.0 <= integ <= 1.0 + 1e-10 for integ in integrals)

    # Their sum should be close to 1 (since parent was normalized)
    total_integral = float(np.sum(integrals))
    assert np.isclose(total_integral, 1.0, rtol=5e-3, atol=5e-4)

    # Pointwise sum over bins should approximate the normalized parent n(z)
    summed = np.zeros_like(z, dtype=float)
    for arr in bins.values():
        summed += arr

    # Compare to the normalized parent (what build_photoz_bins uses internally)
    nz_norm = nz / np.trapezoid(nz, z)

    rel_err = np.max(np.abs(summed - nz_norm) / np.maximum(nz_norm, 1e-12))
    assert rel_err < 5e-2


def test_build_photoz_bins_accepts_lists():
    """Tests that build_photoz_bins accepts list inputs."""
    z = [0.0, 0.5, 1.0, 1.5]
    nz = [1.0, 2.0, 1.0, 0.5]
    bin_edges = [0.0, 0.75, 1.5]

    bins = build_photoz_bins(
        z,
        nz,
        bin_edges,
        scatter_scale=0.05,
        mean_offset=0.01,
    )

    assert list(bins.keys()) == [0, 1]
    assert bins[0].shape == (len(z),)
    assert np.all(np.isfinite(bins[0]))


def test_build_photoz_bins_list_vs_array_same():
    """Tests that build_photoz_bins gives same result for list vs. array inputs."""
    z_list = [0.0, 0.5, 1.0, 1.5]
    nz_list = [1.0, 2.0, 1.0, 0.5]
    edges_list = [0.0, 0.75, 1.5]

    z = np.asarray(z_list)
    nz = np.asarray(nz_list)
    edges = np.asarray(edges_list)

    bins_list = build_photoz_bins(
        z_list, nz_list, edges_list, scatter_scale=0.05, mean_offset=0.01
    )
    bins_arr = build_photoz_bins(z, nz, edges, scatter_scale=0.05, mean_offset=0.01)

    assert np.allclose(bins_list[0], bins_arr[0])
    assert np.allclose(bins_list[1], bins_arr[1])


def test_build_photoz_bins_invalid_bin_edges_raises(z_nz):
    """Tests that build_photoz_bins raises ValueError for invalid bin_edges."""
    z, nz = z_nz

    # Not enough edges
    with pytest.raises(ValueError, match=r"at least two entries"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=[0.0],
            scatter_scale=0.05,
            mean_offset=0.0,
        )

    # Edges outside z range
    with pytest.raises(ValueError, match=r"strictly increasing"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=[0.0, 1.0, 1.0],
            scatter_scale=0.05,
            mean_offset=0.0,
        )


def test_true_redshift_distribution_accepts_lists():
    z = [0.0, 0.5, 1.0]
    nz = [1.0, 2.0, 1.0]
    out = true_redshift_distribution(
        z,
        nz,
        bin_min=0.0,
        bin_max=1.0,
        scatter_scale=0.05,
        mean_offset=0.0,
    )
    assert out.shape == (3,)
    assert np.all(np.isfinite(out))
