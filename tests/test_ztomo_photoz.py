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
    )

    assert list(bins.keys()) == [0, 1, 2]
    for k, arr in bins.items():
        assert isinstance(k, int)
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
    """Tests that true_redshift_distribution accepts list inputs."""
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


def test_build_photoz_bins_scheme_requires_n_bins(z_nz):
    """Tests that build_photoz_bins raises when binning_scheme is
    set but n_bins is not."""
    z, nz = z_nz
    with pytest.raises(ValueError, match=r"You must provide n_bins"):
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
    """Tests that build_photoz_bins raises when bin_edges is not set but
    binning_scheme"""
    z, nz = z_nz
    with pytest.raises(ValueError, match=r"must provide binning_scheme"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=None,
            scatter_scale=0.05,
            mean_offset=0.0,
        )


def test_build_photoz_bins_bin_edges_mutually_exclusive_with_scheme(z_nz):
    """Tests that build_photoz_bins raises when both bin_edges and binning_scheme are
    provided."""
    z, nz = z_nz
    with pytest.raises(ValueError, match=r"either bin_edges or"):
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
    """Tests that build_photoz_bins raises when equal_number is set but nz_ph is not
    provided."""
    z, nz = z_nz
    with pytest.raises(ValueError, match=r"Provide both z_ph and nz_ph"):
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


def test_build_photoz_bins_equal_number_proxy_path_matches_explicit_proxy(z_nz):
    """Tests that build_photoz_bins with equal_number binning scheme gives same result
    as explicit proxy."""
    z, nz = z_nz

    bins_proxy = build_photoz_bins(
        z,
        nz,
        bin_edges=None,
        binning_scheme="equal_number",
        n_bins=3,
        scatter_scale=0.05,
        mean_offset=0.01,
        normalize_input=True,
    )

    bins_explicit = build_photoz_bins(
        z,
        nz,
        bin_edges=None,
        binning_scheme="equal_number",
        n_bins=3,
        z_ph=z,
        nz_ph=nz,
        scatter_scale=0.05,
        mean_offset=0.01,
        normalize_input=True,
    )

    for k in bins_proxy:
        assert np.allclose(bins_proxy[k], bins_explicit[k], rtol=1e-12, atol=0.0)


def test_build_photoz_bins_normalize_bins_true_integrates_to_one(z_nz):
    """Tests that build_photoz_bins with normalize_bins=True returns bins that
    normalize to 1.0."""
    z, nz = z_nz
    edges = [0.0, 0.5, 1.0, 1.5]

    bins = build_photoz_bins(
        z,
        nz,
        edges,
        scatter_scale=0.05,
        mean_offset=0.01,
        normalize_input=True,
        normalize_bins=True,
    )

    for arr in bins.values():
        integ = np.trapezoid(arr, x=z)
        assert np.isclose(integ, 1.0, rtol=1e-6, atol=1e-8)


def test_true_redshift_distribution_outliers_enabled_changes_output(z_nz):
    """Tests that true_redshift_distribution with outlier_frac>0 returns a different
    distribution than core-only case."""
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


def test_build_photoz_bins_equidistant_scheme_counts_bins(z_nz):
    """Tests that equidistant scheme returns n_bins bins."""
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
        normalize_input=True,
    )
    for arr in bins.values():
        assert arr.shape == z.shape
        assert np.all(np.isfinite(arr))


def test_build_photoz_bins_equal_number_with_observed_inputs(z_nz):
    """Tests that equal-number scheme accepts (z_ph, nz_ph) inputs."""
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
        normalize_input=True,
    )
    assert list(bins.keys()) == [0, 1, 2]


def test_build_photoz_bins_unsupported_scheme_raises(z_nz):
    """Tests that unsupported string scheme raises."""
    z, nz = z_nz
    with pytest.raises(ValueError, match=r"Unsupported binning_scheme"):
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
    with pytest.raises(ValueError, match=r"mixed binning mode"):
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
    """Tests that mixed dict must include 'segments'."""
    z, nz = z_nz
    with pytest.raises(ValueError, match=r"must contain key 'segments'"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=None,
            binning_scheme={"nope": []},
            scatter_scale=0.05,
            mean_offset=0.0,
        )


def test_build_photoz_bins_mixed_sequence_builds_bins(z_nz):
    """Tests that mixed segments sequence builds correct bin count."""
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
    """Tests that mixed segments dict form works."""
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


def test_build_photoz_bins_mixed_noncontiguous_raises(z_nz):
    """Tests that mixed segments must be contiguous."""
    z, nz = z_nz
    segs = [
        {"z_min": 0.0, "z_max": 1.0, "scheme": "eq", "n_bins": 2},
        {"z_min": 1.1, "z_max": 2.0, "scheme": "eq", "n_bins": 2},
    ]
    with pytest.raises(ValueError, match=r"contiguous"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=None,
            binning_scheme=segs,
            scatter_scale=0.05,
            mean_offset=0.0,
        )


def test_build_photoz_bins_mixed_overlap_raises(z_nz):
    """Tests that mixed segments cannot overlap."""
    z, nz = z_nz
    segs = [
        {"z_min": 0.0, "z_max": 1.0, "scheme": "eq", "n_bins": 2},
        {"z_min": 0.9, "z_max": 2.0, "scheme": "eq", "n_bins": 2},
    ]
    with pytest.raises(ValueError, match=r"non-overlapping"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=None,
            binning_scheme=segs,
            scatter_scale=0.05,
            mean_offset=0.0,
        )


def test_build_photoz_bins_mixed_bad_segment_scheme_raises(z_nz):
    """Tests that mixed segment scheme must be supported."""
    z, nz = z_nz
    segs = [
        {"z_min": 0.0, "z_max": 1.0, "scheme": "nope", "n_bins": 2},
    ]
    with pytest.raises(ValueError, match=r"Unsupported segment scheme"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=None,
            binning_scheme=segs,
            scatter_scale=0.05,
            mean_offset=0.0,
        )


def test_build_photoz_bins_mixed_equal_number_needs_data(z_nz):
    """Tests that mixed equal-number needs enough points in segment."""
    z, nz = z_nz
    segs = [
        {"z_min": 9.0, "z_max": 10.0, "scheme": "en", "n_bins": 2},
    ]
    with pytest.raises(ValueError, match=r"too few points"):
        build_photoz_bins(
            z,
            nz,
            bin_edges=None,
            binning_scheme=segs,
            scatter_scale=0.05,
            mean_offset=0.0,
        )


def test_build_photoz_bins_mixed_equal_number_uses_proxy(z_nz):
    """Tests that mixed equal-number works without (z_ph, nz_ph)."""
    z, nz = z_nz
    segs = [
        {"z_min": 0.0, "z_max": 1.5, "scheme": "en", "n_bins": 3},
        {"z_min": 1.5, "z_max": 3.0, "scheme": "eq", "n_bins": 2},
    ]
    bins = build_photoz_bins(
        z,
        nz,
        bin_edges=None,
        binning_scheme=segs,
        scatter_scale=0.05,
        mean_offset=0.0,
        normalize_input=True,
    )
    assert list(bins.keys()) == [0, 1, 2, 3, 4]


def test_build_photoz_bins_normalize_input_false_changes_outputs(z_nz):
    """Tests that disabling input normalization changes at least one bin."""
    z, nz = z_nz
    edges = [0.0, 0.5, 1.0, 1.5]

    bins_norm = build_photoz_bins(
        z,
        nz,
        edges,
        scatter_scale=0.05,
        mean_offset=0.01,
        normalize_input=True,
        normalize_bins=False,
    )
    bins_raw = build_photoz_bins(
        z,
        nz,
        edges,
        scatter_scale=0.05,
        mean_offset=0.01,
        normalize_input=False,
        normalize_bins=False,
    )

    diffs = []
    for k in bins_norm:
        diffs.append(np.max(np.abs(bins_norm[k] - bins_raw[k])))
    assert max(diffs) > 0.0


def test_build_photoz_bins_normalize_bins_false_not_unit_integral(z_nz):
    """Tests that without bin normalization, integrals are not forced to 1."""
    z, nz = z_nz
    edges = [0.0, 0.5, 1.0, 1.5]
    bins = build_photoz_bins(
        z,
        nz,
        edges,
        scatter_scale=0.05,
        mean_offset=0.01,
        normalize_input=True,
        normalize_bins=False,
    )
    ints = [float(np.trapezoid(arr, x=z)) for arr in bins.values()]
    assert any(not np.isclose(v, 1.0, rtol=1e-6, atol=1e-8) for v in ints)


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
        outlier_scatter_scale=0.2,
        outlier_mean_offset=0.2,
        outlier_mean_scale=0.9,
    )
    assert np.allclose(a, b, rtol=0.0, atol=0.0)
