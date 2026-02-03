"""Unit tests for ``binny.nz_tomo.photoz`` module."""

from __future__ import annotations

import numpy as np
import pytest

from binny.nz_tomo.photoz import build_photoz_bins, true_redshift_distribution


@pytest.fixture
def z_nz():
    """Provides a sample redshift grid and unnormalized n(z) for testing."""
    z = np.linspace(0.0, 3.0, 501)
    nz = z**2 * np.exp(-z)  # deliberately NOT normalized on [0, 3]
    return z, nz


def test_true_redshift_distribution_full_range_returns_parent(z_nz):
    """Tests that huge bin bounds yield P(bin|z)≈1 and returns the parent."""
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
    rel = np.max(np.abs(nb - nz) / np.maximum(np.abs(nz), 1e-30))
    assert rel < 1e-10


def test_true_redshift_distribution_outlier_disabled_matches_core(z_nz):
    """Tests that outliers require outlier_scatter_scale when outlier_frac > 0."""
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

    with pytest.raises(ValueError, match=r"outlier_scatter_scale must be set"):
        true_redshift_distribution(
            z,
            nz,
            bin_min=0.5,
            bin_max=1.0,
            scatter_scale=0.05,
            mean_offset=0.01,
            mean_scale=1.0,
            outlier_frac=0.5,
            outlier_scatter_scale=None,
            outlier_mean_offset=0.2,
            outlier_mean_scale=0.9,
        )

    assert np.all(np.isfinite(core_only))


def test_true_redshift_distribution_outliers_enabled_changes_output(z_nz):
    """Tests that enabling outliers changes the selected distribution."""
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

    with_outliers = true_redshift_distribution(
        z,
        nz,
        bin_min=0.5,
        bin_max=1.0,
        scatter_scale=0.05,
        mean_offset=0.01,
        mean_scale=1.0,
        outlier_frac=0.2,
        outlier_scatter_scale=0.2,
        outlier_mean_offset=0.2,
        outlier_mean_scale=0.9,
    )

    assert not np.allclose(with_outliers, core_only)


def test_true_redshift_distribution_validates_parameters(z_nz):
    """Tests that invalid model parameters raise ValueError."""
    z, nz = z_nz

    with pytest.raises(ValueError, match=r"outlier_frac must lie"):
        true_redshift_distribution(
            z,
            nz,
            0.0,
            1.0,
            scatter_scale=0.1,
            mean_offset=0.0,
            outlier_frac=1.1,
        )

    with pytest.raises(ValueError, match=r"mean_scale must be > 0"):
        true_redshift_distribution(
            z,
            nz,
            0.0,
            1.0,
            scatter_scale=0.1,
            mean_offset=0.0,
            mean_scale=0.0,
        )

    with pytest.raises(ValueError, match=r"scatter_scale must be >="):
        true_redshift_distribution(
            z,
            nz,
            0.0,
            1.0,
            scatter_scale=-1.0,
            mean_offset=0.0,
            mean_scale=1.0,
        )

    # scatter_scale == 0.0 is allowed (deterministic assignment limit)
    out0 = true_redshift_distribution(
        z,
        nz,
        0.0,
        1.0,
        scatter_scale=0.0,
        mean_offset=0.0,
        mean_scale=1.0,
        outlier_frac=0.0,
        outlier_scatter_scale=None,
    )
    assert out0.shape == z.shape
    assert np.all(np.isfinite(out0))

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

    # outlier_scatter_scale == 0.0 is allowed when outliers are enabled
    out1 = true_redshift_distribution(
        z,
        nz,
        0.0,
        1.0,
        scatter_scale=0.1,
        mean_offset=0.0,
        mean_scale=1.0,
        outlier_frac=0.2,
        outlier_scatter_scale=0.0,
        outlier_mean_offset=0.0,
        outlier_mean_scale=1.0,
    )
    assert out1.shape == z.shape
    assert np.all(np.isfinite(out1))


def test_true_redshift_distribution_accepts_lists():
    """Tests that list inputs are accepted and return finite outputs."""
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


def test_true_redshift_distribution_outlier_frac_zero_ignores_scatter(z_nz):
    """Tests that outlier scatter is ignored when outlier_frac is zero."""
    z, nz = z_nz
    a = true_redshift_distribution(
        z,
        nz,
        bin_min=0.5,
        bin_max=1.0,
        scatter_scale=0.05,
        mean_offset=0.01,
        outlier_frac=0.0,
        outlier_scatter_scale=None,
        outlier_mean_offset=0.2,
        outlier_mean_scale=0.9,
    )
    b = true_redshift_distribution(
        z,
        nz,
        bin_min=0.5,
        bin_max=1.0,
        scatter_scale=0.05,
        mean_offset=0.01,
        outlier_frac=0.0,
        outlier_scatter_scale=0.2,  # should be ignored
        outlier_mean_offset=0.2,
        outlier_mean_scale=0.9,
    )
    assert np.allclose(a, b, rtol=0.0, atol=0.0)


def test_build_photoz_bins_basic_shapes_and_keys(z_nz):
    """Tests that explicit edges return the expected keys and shapes."""
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
    )

    assert list(bins.keys()) == [0, 1, 2]
    for k, arr in bins.items():
        assert isinstance(k, int)
        assert arr.shape == z.shape
        assert np.all(np.isfinite(arr))


def test_build_photoz_bins_per_bin_parameters(z_nz):
    """Tests that per-bin params broadcast and produce the right bin count."""
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
        outlier_scatter_scale=[0.20, 0.20, 0.20, 0.15],
        outlier_mean_offset=[0.0, 0.1, 0.0, 0.0],
        outlier_mean_scale=[1.0, 0.9, 1.0, 1.0],
    )

    assert list(bins.keys()) == [0, 1, 2, 3]
    for arr in bins.values():
        assert arr.shape == z.shape
        assert np.all(np.isfinite(arr))


def test_build_photoz_bins_broadcasting_wrong_length_raises(z_nz):
    """Tests that wrong-length per-bin params raise ValueError."""
    z, nz = z_nz
    edges = [0.0, 0.5, 1.0, 1.5]

    with pytest.raises(ValueError):
        build_photoz_bins(
            z,
            nz,
            edges,
            scatter_scale=[0.05, 0.06],  # should be length 3
            mean_offset=0.01,
        )


def test_build_photoz_bins_accepts_lists():
    """Tests that list inputs are accepted and return finite outputs."""
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
    """Tests that list vs array inputs produce identical results."""
    z_list = [0.0, 0.5, 1.0, 1.5]
    nz_list = [1.0, 2.0, 1.0, 0.5]
    edges_list = [0.0, 0.75, 1.5]

    z = np.asarray(z_list)
    nz = np.asarray(nz_list)
    edges = np.asarray(edges_list)

    bins_list = build_photoz_bins(
        z_list,
        nz_list,
        edges_list,
        scatter_scale=0.05,
        mean_offset=0.01,
    )
    bins_arr = build_photoz_bins(z, nz, edges, scatter_scale=0.05, mean_offset=0.01)

    assert np.allclose(bins_list[0], bins_arr[0])
    assert np.allclose(bins_list[1], bins_arr[1])


def test_build_photoz_bins_invalid_bin_edges_raises(z_nz):
    """Tests that invalid bin_edges raise ValueError."""
    z, nz = z_nz

    with pytest.raises(ValueError, match=r"at least two"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=[0.0],
            scatter_scale=0.05,
            mean_offset=0.0,
        )

    with pytest.raises(ValueError, match=r"increasing"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=[0.0, 1.0, 1.0],
            scatter_scale=0.05,
            mean_offset=0.0,
        )


def test_build_photoz_bins_scheme_requires_n_bins(z_nz):
    """Tests that string binning_scheme requires n_bins."""
    z, nz = z_nz
    with pytest.raises(ValueError, match=r"n_bins"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=None,
            binning_scheme="equidistant",
            n_bins=None,
            scatter_scale=0.05,
            mean_offset=0.0,
        )


def test_build_photoz_bins_requires_binning_scheme_when_no_edges(z_nz):
    """Tests that missing binning_scheme raises when bin_edges is None."""
    z, nz = z_nz
    with pytest.raises(ValueError, match=r"binning_scheme"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=None,
            scatter_scale=0.05,
            mean_offset=0.0,
        )


def test_build_photoz_bins_bin_edges_mutually_exclusive_with_scheme(z_nz):
    """Tests that bin_edges conflicts with (binning_scheme, n_bins)."""
    z, nz = z_nz
    with pytest.raises(ValueError, match=r"either.*bin_edges|bin_edges.*either"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=[0.0, 1.0],
            scatter_scale=0.05,
            mean_offset=0.0,
            binning_scheme="equidistant",
            n_bins=2,
        )


def test_build_photoz_bins_equal_number_requires_both_zph_and_nzph(z_nz):
    """Tests that equal_number forbids providing only one of (z_ph, nz_ph)."""
    z, nz = z_nz
    with pytest.raises(ValueError, match=r"z_ph.*nz_ph|nz_ph.*z_ph|both"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=None,
            binning_scheme="equal_number",
            n_bins=3,
            z_ph=z,  # nz_ph missing
            scatter_scale=0.05,
            mean_offset=0.0,
        )


def test_build_photoz_bins_normalize_bins_true_integrates_to_one(z_nz):
    """Tests that normalize_bins=True returns per-bin unit integrals."""
    z, nz = z_nz
    edges = [0.0, 0.5, 1.0, 1.5]

    bins = build_photoz_bins(
        z,
        nz,
        edges,
        scatter_scale=0.05,
        mean_offset=0.01,
        normalize_bins=True,
    )

    for arr in bins.values():
        integ = np.trapezoid(arr, x=z)
        assert np.isclose(integ, 1.0, rtol=1e-6, atol=1e-8)


def test_build_photoz_bins_normalize_bins_false_scales_with_parent(z_nz):
    """Tests that normalize_bins=False scales with parent when normalize_input=False."""
    z, nz = z_nz
    edges = [0.0, 0.5, 1.0, 1.5]

    bins_a = build_photoz_bins(
        z,
        nz,
        edges,
        scatter_scale=0.05,
        mean_offset=0.01,
        normalize_input=False,
        normalize_bins=False,
    )
    bins_b = build_photoz_bins(
        z,
        3.0 * nz,
        edges,
        scatter_scale=0.05,
        mean_offset=0.01,
        normalize_input=False,
        normalize_bins=False,
    )

    for k in bins_a:
        assert np.allclose(bins_b[k], 3.0 * bins_a[k], rtol=1e-12, atol=0.0)


def test_build_photoz_bins_equidistant_scheme_counts_bins(z_nz):
    """Tests that equidistant scheme returns exactly n_bins bins."""
    z, nz = z_nz
    bins = build_photoz_bins(
        z,
        nz,
        bin_edges=None,
        binning_scheme="equidistant",
        n_bins=4,
        scatter_scale=0.05,
        mean_offset=0.0,
    )
    assert list(bins.keys()) == [0, 1, 2, 3]


def test_build_photoz_bins_equidistant_scheme_shapes(z_nz):
    """Tests that equidistant scheme outputs are finite and match z shape."""
    z, nz = z_nz
    bins = build_photoz_bins(
        z,
        nz,
        bin_edges=None,
        binning_scheme="equidistant",
        n_bins=3,
        scatter_scale=0.05,
        mean_offset=0.01,
    )
    for arr in bins.values():
        assert arr.shape == z.shape
        assert np.all(np.isfinite(arr))


def test_build_photoz_bins_equal_number_with_observed_inputs(z_nz):
    """Tests that equal_number accepts explicit (z_ph, nz_ph) inputs."""
    z, nz = z_nz
    bins = build_photoz_bins(
        z,
        nz,
        bin_edges=None,
        binning_scheme="equal_number",
        n_bins=3,
        z_ph=z,
        nz_ph=nz,
        scatter_scale=0.05,
        mean_offset=0.0,
    )
    assert list(bins.keys()) == [0, 1, 2]


def test_build_photoz_bins_unsupported_scheme_raises(z_nz):
    """Tests that unsupported string scheme raises ValueError."""
    z, nz = z_nz
    with pytest.raises(ValueError, match=r"Unsupported|unsupported"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=None,
            binning_scheme="banana",
            n_bins=3,
            scatter_scale=0.05,
            mean_offset=0.0,
        )


def test_build_photoz_bins_mixed_requires_no_top_level_n_bins(z_nz):
    """Tests that mixed mode rejects top-level n_bins."""
    z, nz = z_nz
    segs = [
        {"z_min": 0.0, "z_max": 1.0, "scheme": "eq", "n_bins": 2},
        {"z_min": 1.0, "z_max": 2.0, "scheme": "eq", "n_bins": 2},
    ]
    with pytest.raises(ValueError, match=r"mixed|segments"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=None,
            binning_scheme=segs,
            n_bins=4,
            scatter_scale=0.05,
            mean_offset=0.0,
        )


def test_build_photoz_bins_mixed_dict_requires_segments_key(z_nz):
    """Tests that mixed dict form must include a 'segments' key."""
    z, nz = z_nz
    with pytest.raises(ValueError, match=r"segments"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=None,
            binning_scheme={"nope": []},
            scatter_scale=0.05,
            mean_offset=0.0,
        )


def test_build_photoz_bins_mixed_sequence_builds_bins(z_nz):
    """Tests that mixed segments build the correct total bin count."""
    z, nz = z_nz
    segs = [
        {"z_min": 0.0, "z_max": 1.0, "scheme": "eq", "n_bins": 2},
        {"z_min": 1.0, "z_max": 2.0, "scheme": "eq", "n_bins": 3},
    ]
    bins = build_photoz_bins(
        z,
        nz,
        bin_edges=None,
        binning_scheme=segs,
        scatter_scale=0.05,
        mean_offset=0.0,
    )
    assert list(bins.keys()) == [0, 1, 2, 3, 4]


def test_build_photoz_bins_mixed_dict_builds_bins(z_nz):
    """Tests that mixed dict form builds bins."""
    z, nz = z_nz
    segs = [
        {"z_min": 0.0, "z_max": 1.0, "scheme": "eq", "n_bins": 2},
        {"z_min": 1.0, "z_max": 2.0, "scheme": "eq", "n_bins": 1},
    ]
    bins = build_photoz_bins(
        z,
        nz,
        bin_edges=None,
        binning_scheme={"segments": segs},
        scatter_scale=0.05,
        mean_offset=0.0,
    )
    assert list(bins.keys()) == [0, 1, 2]


def test_build_photoz_bins_metadata_round_trip(z_nz, tmp_path):
    """Tests that include_metadata returns expected fields and save works."""
    z, nz = z_nz
    edges = [0.0, 0.5, 1.0, 1.5]
    outpath = tmp_path / "photoz_meta.txt"

    bins, meta = build_photoz_bins(
        z,
        nz,
        edges,
        scatter_scale=0.05,
        mean_offset=0.01,
        normalize_bins=True,
        include_metadata=True,
        save_metadata_path=str(outpath),
    )

    assert isinstance(bins, dict)
    assert isinstance(meta, dict)
    assert outpath.exists()

    assert meta.get("kind") == "photoz"

    # Be tolerant to schema evolution, but require edges + norms exist somewhere.
    meta_str = repr(meta)
    assert "bin_edges" in meta_str
    assert "bins_norms" in meta_str

    # Returned bins are unit-normalized when normalize_bins=True.
    for arr in bins.values():
        assert np.isclose(np.trapezoid(arr, x=z), 1.0, rtol=1e-6, atol=1e-8)
