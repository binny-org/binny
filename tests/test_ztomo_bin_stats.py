"""Unit tests for binny.ztomo.bin_stats functions."""

import numpy as np
import pytest

from binny.ztomo.bin_stats import (
    bin_fractions,
    bin_integrals,
    bin_moments,
    bin_overlap_percent,
    bin_overlaps,
    n_eff_per_bin,
    overlapping_bin_pairs,
    summarize_bins,
)


def _toy_grid():
    """Creates a toy redshift grid for testing."""
    z = np.linspace(0.0, 2.0, 2001)
    return z


def _gaussian(z, mu, sig):
    """Creates a Gaussian n(z) on grid z with mean mu and std sig."""
    return np.exp(-0.5 * ((z - mu) / sig) ** 2)


def test_bin_moments_gaussian_mean_std_close():
    """Tests that bin_moments recovers mean and std of a Gaussian n(z)."""
    z = _toy_grid()
    mu, sig = 0.8, 0.15
    nz = _gaussian(z, mu, sig)

    mean, std = bin_moments(z, nz)

    assert np.isclose(mean, mu, rtol=0.0, atol=2e-3)
    assert np.isclose(std, sig, rtol=0.0, atol=2e-3)


def test_bin_moments_raises_on_nonpositive_norm():
    """Tests that bin_moments raises ValueError for non-positive normalization."""
    z = _toy_grid()
    nz = np.zeros_like(z)

    with pytest.raises(ValueError, match="normalization must be positive"):
        bin_moments(z, nz)


def test_summarize_bins_includes_sigma_mean_from_scalar():
    """Tests that summarize_bins includes sigma_mean from scalar input."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    out = summarize_bins(z, bins, sigma_mean=0.03)

    assert set(out.keys()) == {0, 1}
    assert out[0]["sigma_mean"] == pytest.approx(0.03)
    assert out[1]["sigma_mean"] == pytest.approx(0.03)
    assert "mean" in out[0] and "std" in out[0]


def test_summarize_bins_sigma_mean_sequence_length_mismatch_raises():
    """Tests that summarize_bins raises ValueError
    for sigma_mean sequence length mismatch."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    with pytest.raises(ValueError, match="sigma_mean sequence must have length"):
        summarize_bins(z, bins, sigma_mean=[0.1])  # wrong length


def test_summarize_bins_sigma_mean_mapping_missing_key_raises():
    """Tests that summarize_bins raises ValueError"""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    with pytest.raises(ValueError, match=r"sigma_mean is missing bin index 1"):
        summarize_bins(z, bins, sigma_mean={0: 0.01})


def test_summarize_bins_neff_mapping_missing_key_raises():
    """Tests that summarize_bins raises ValueError for neff_per_bin missing key."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    with pytest.raises(ValueError, match=r"neff_per_bin is missing bin index 1"):
        summarize_bins(z, bins, neff_per_bin={0: 100.0})  # missing 1


def test_summarize_bins_neff_nonpositive_raises():
    """Tests that summarize_bins raises ValueError for non-positive neff_per_bin."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1)}

    with pytest.raises(ValueError, match=r"neff_per_bin\[0\] must be positive"):
        summarize_bins(z, bins, neff_per_bin={0: 0.0})


def test_summarize_bins_sigma_mean_from_neff_matches_std_over_sqrt_neff():
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.8, 0.2)}
    neff = {0: 400.0}

    out = summarize_bins(z, bins, neff_per_bin=neff)
    mean, std = bin_moments(z, bins[0])

    assert out[0]["mean"] == pytest.approx(mean)
    assert out[0]["std"] == pytest.approx(std)
    assert out[0]["sigma_mean"] == pytest.approx(std / np.sqrt(neff[0]))


def test_bin_integrals_and_fractions_sum_to_one():
    """Tests that bin_integrals and bin_fractions behave as expected."""
    z = _toy_grid()
    b0 = _gaussian(z, 0.5, 0.1)
    b1 = 2.0 * _gaussian(z, 1.2, 0.2)  # deliberately different normalization
    bins = {0: b0, 1: b1}

    integrals = bin_integrals(z, bins)
    fracs = bin_fractions(z, bins)

    assert integrals[0] > 0
    assert integrals[1] > 0
    assert np.isclose(sum(fracs.values()), 1.0, atol=1e-12)
    assert fracs[1] > fracs[0]


def test_bin_fractions_raises_on_nonpositive_total():
    """Tests that bin_fractions raises ValueError for non-positive total."""
    z = _toy_grid()
    bins = {0: np.zeros_like(z), 1: np.zeros_like(z)}

    with pytest.raises(
        ValueError, match="Total integrated n\\(z\\) over all bins must be positive"
    ):
        bin_fractions(z, bins)


def test_n_eff_per_bin_allocates_proportionally():
    """Tests that n_eff_per_bin allocates neff proportionally to bin integrals."""
    z = _toy_grid()
    b0 = _gaussian(z, 0.5, 0.1)
    b1 = 3.0 * _gaussian(z, 1.2, 0.2)
    bins = {0: b0, 1: b1}

    neff_total = 1000.0
    neff, frac = n_eff_per_bin(z, bins, neff_total, expect_unnormalized=True)

    assert np.isclose(sum(frac.values()), 1.0, atol=1e-12)
    assert np.isclose(sum(neff.values()), neff_total, atol=1e-12)
    assert neff[1] > neff[0]


def test_n_eff_per_bin_raises_if_all_bins_look_normalized():
    """Tests that n_eff_per_bin raises ValueError if all bins look normalized."""
    z = _toy_grid()
    # Make two bins that integrate approx 1 each
    p0 = _gaussian(z, 0.6, 0.1)
    p0 /= np.trapezoid(p0, z)
    p1 = _gaussian(z, 1.2, 0.2)
    p1 /= np.trapezoid(p1, z)

    bins = {0: p0, 1: p1}

    with pytest.raises(ValueError, match="All bins appear normalized"):
        n_eff_per_bin(z, bins, n_eff_total=100.0, expect_unnormalized=True)


def test_n_eff_per_bin_allows_normalized_if_expect_unnormalized_false():
    """Tests that n_eff_per_bin works with normalized bins if
    expect_unnormalized=False."""
    z = _toy_grid()
    p0 = _gaussian(z, 0.6, 0.1)
    p0 /= np.trapezoid(p0, z)
    p1 = _gaussian(z, 1.2, 0.2)
    p1 /= np.trapezoid(p1, z)
    bins = {0: p0, 1: p1}

    neff, frac = n_eff_per_bin(z, bins, n_eff_total=100.0, expect_unnormalized=False)

    assert np.isclose(sum(frac.values()), 1.0, atol=1e-12)
    assert np.isclose(sum(neff.values()), 100.0, atol=1e-12)


def test_bin_overlaps_min_diagonal_is_one_when_normalized():
    """Tests that bin_overlaps with method='min' has 1.0 on diagonal
    when bins are normalized."""
    z = _toy_grid()
    p0 = _gaussian(z, 0.6, 0.1)
    p0 /= np.trapezoid(p0, z)
    p1 = _gaussian(z, 1.2, 0.2)
    p1 /= np.trapezoid(p1, z)
    bins = {0: p0, 1: p1}

    ov = bin_overlaps(
        z, bins, method="min", assume_normalized=True, normalize_if_needed=False
    )

    assert ov[0][0] == pytest.approx(1.0)
    assert ov[1][1] == pytest.approx(1.0)
    assert 0.0 <= ov[0][1] <= 1.0
    assert ov[0][1] == pytest.approx(ov[1][0])


def test_bin_overlaps_min_raises_if_assume_normalized_and_not_normalized():
    """Tests that bin_overlaps with method='min' raises ValueError
    if assume_normalized=True"""
    z = _toy_grid()
    # Not normalized on purpose
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    with pytest.raises(ValueError, match="does not appear normalized"):
        bin_overlaps(
            z,
            bins,
            method="min",
            assume_normalized=True,
            normalize_if_needed=False,
        )


def test_bin_overlaps_min_can_normalize_if_needed():
    """Tests that bin_overlaps with method='min' can normalize bins if needed."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    ov = bin_overlaps(
        z,
        bins,
        method="min",
        assume_normalized=True,
        normalize_if_needed=True,
    )

    assert ov[0][0] == pytest.approx(1.0)
    assert ov[1][1] == pytest.approx(1.0)


def test_bin_overlaps_cosine_bounds_and_symmetry():
    """Tests that bin_overlaps with method='cosine' returns values in [0, 1]
    and is symmetric."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    ov = bin_overlaps(z, bins, method="cosine")

    assert 0.0 <= ov[0][1] <= 1.0
    assert ov[0][1] == pytest.approx(ov[1][0])
    assert ov[0][0] == pytest.approx(1.0)
    assert ov[1][1] == pytest.approx(1.0)


def test_bin_overlap_percent_is_100_on_diagonal_when_normalized():
    """Tests that bin_overlap_percent is 100% on diagonal when bins are normalized."""
    z = _toy_grid()
    p0 = _gaussian(z, 0.6, 0.1)
    p0 /= np.trapezoid(p0, z)
    p1 = _gaussian(z, 1.2, 0.2)
    p1 /= np.trapezoid(p1, z)
    bins = {0: p0, 1: p1}

    pct = bin_overlap_percent(
        z, bins, assume_normalized=True, normalize_if_needed=False
    )

    assert pct[0][0] == pytest.approx(100.0)
    assert pct[1][1] == pytest.approx(100.0)
    assert 0.0 <= pct[0][1] <= 100.0


def test_overlapping_bin_pairs_threshold_filters_and_sorts():
    """Tests that overlapping_bin_pairs filters by threshold and sorts output."""
    z = _toy_grid()
    # Make bins overlap strongly by using close means
    p0 = _gaussian(z, 0.8, 0.2)
    p0 /= np.trapezoid(p0, z)
    p1 = _gaussian(z, 0.9, 0.2)
    p1 /= np.trapezoid(p1, z)
    bins = {0: p0, 1: p1}

    pairs = overlapping_bin_pairs(
        z,
        bins,
        threshold_percent=10.0,
        assume_normalized=True,
        normalize_if_needed=False,
    )

    assert pairs
    i, j, val = pairs[0]
    assert (i, j) == (0, 1)
    assert val >= 10.0
