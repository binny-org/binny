"""Unit tests for ``binny.ztomo.bin_stats`` module."""

import numpy as np
import pytest

from binny.ztomo.bin_stats import (
    bin_centers,
    bin_moments,
    bin_quantiles,
    galaxy_count_per_bin,
    galaxy_density_per_bin,
    galaxy_fraction_per_bin,
    in_range_fraction,
    in_range_fraction_per_bin,
    peak_flags,
    peak_flags_per_bin,
    summarize_bins,
)


def _toy_grid():
    """Creates a toy redshift grid for testing."""
    return np.linspace(0.0, 2.0, 2001)


def _gaussian(z, mu, sig):
    """Creates a Gaussian n(z) on grid z with mean mu and std sig."""
    return np.exp(-0.5 * ((z - mu) / sig) ** 2)


def _normalize_pdf(z, nz):
    """Normalizes a nonnegative curve nz(z) into a PDF on z."""
    norm = np.trapezoid(nz, x=z)
    if norm <= 0:
        raise ValueError("Cannot normalize: non-positive integral.")
    return nz / norm


def test_bin_moments_gaussian_mean_std_close():
    """Tests that bin_moments recovers mean and std for a Gaussian n(z)."""
    z = _toy_grid()
    mu, sig = 0.8, 0.15
    nz = _gaussian(z, mu, sig)

    stats = bin_moments(z, nz)

    assert {
        "mean",
        "median",
        "mode",
        "std",
        "skewness",
        "kurtosis",
        "iqr",
        "width_68",
    } <= set(stats.keys())

    assert np.isclose(stats["mean"], mu, rtol=0.0, atol=2e-3)
    assert np.isclose(stats["std"], sig, rtol=0.0, atol=2e-3)

    # Symmetric Gaussian => median ~ mean and mode ~ mean on a fine grid
    assert np.isclose(stats["median"], mu, rtol=0.0, atol=2e-3)
    assert np.isclose(stats["mode"], mu, rtol=0.0, atol=2e-3)

    # Sanity on derived summaries
    assert stats["width_68"] > 0.0
    assert stats["iqr"] > 0.0


def test_bin_moments_raises_on_nonpositive_total_weight():
    """Tests that bin_moments raises ValueError for non-positive total weight."""
    z = _toy_grid()
    nz = np.zeros_like(z)

    with pytest.raises(ValueError, match="Total weight must be positive"):
        bin_moments(z, nz)


def test_bin_centers_methods_and_rounding():
    """Tests that bin_centers works for different methods and rounding."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    centers = bin_centers(z, bins, method="median", decimal_places=2)
    assert set(centers.keys()) == {0, 1}
    assert isinstance(centers[0], float)

    centers_full = bin_centers(z, bins, method="median", decimal_places=None)
    assert centers[0] == pytest.approx(round(centers_full[0], 2))
    assert centers[1] == pytest.approx(round(centers_full[1], 2))


def test_bin_centers_percentile_method_p50_matches_median():
    """Tests that bin_centers(method='p50') matches bin_centers(method='median')."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    c_med = bin_centers(z, bins, method="median", decimal_places=None)
    c_p50 = bin_centers(z, bins, method="p50", decimal_places=None)

    assert c_p50[0] == pytest.approx(c_med[0])
    assert c_p50[1] == pytest.approx(c_med[1])


def test_bin_centers_percentile_invalid_raises():
    """Tests that bin_centers raises ValueError for invalid percentile methods."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1)}

    with pytest.raises(ValueError, match="percentile methods must look like"):
        bin_centers(z, bins, method="pXX")

    with pytest.raises(
        ValueError, match="percentile in method='pXX' must be between 0 and 100"
    ):
        bin_centers(z, bins, method="p120")


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
    """Tests that summarize_bins raises ValueError for
    sigma_mean sequence length mismatch."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    with pytest.raises(ValueError, match="sigma_mean sequence must have length"):
        summarize_bins(z, bins, sigma_mean=[0.1])  # wrong length


def test_summarize_bins_sigma_mean_mapping_missing_key_raises():
    """Tests that summarize_bins raises ValueError if sigma_mean
    mapping misses a bin."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    with pytest.raises(ValueError, match=r"sigma_mean is missing bin index 1"):
        summarize_bins(z, bins, sigma_mean={0: 0.01})


def test_summarize_bins_sigma_mean_from_count_matches_std_over_sqrt_count():
    """Tests that summarize_bins computes sigma_mean when count_per_bin is
    provided."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.8, 0.2)}
    counts = {0: 400.0}

    out = summarize_bins(z, bins, count_per_bin=counts)
    stats = bin_moments(z, bins[0])

    assert out[0]["mean"] == pytest.approx(stats["mean"])
    assert out[0]["std"] == pytest.approx(stats["std"])
    divi = stats["std"] / np.sqrt(counts[0])
    assert out[0]["sigma_mean"] == divi


def test_summarize_bins_sigma_mean_from_density_and_area():
    """Tests that summarize_bins infers counts from density_per_bin
    and survey_area."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.8, 0.2)}
    dens = {0: 10.0}  # gal/arcmin^2
    area = 100.0  # arcmin^2 => count=1000

    out = summarize_bins(z, bins, density_per_bin=dens, survey_area=area)
    stats = bin_moments(z, bins[0])

    count = dens[0] * area
    assert out[0]["sigma_mean"] == pytest.approx(stats["std"] / np.sqrt(count))


def test_summarize_bins_density_requires_survey_area():
    """Tests that summarize_bins requires survey_area when density_per_bin
    is used."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.8, 0.2)}
    dens = {0: 10.0}

    with pytest.raises(ValueError, match="survey_area must be provided"):
        summarize_bins(z, bins, density_per_bin=dens)


def test_galaxy_fraction_per_bin_sums_to_one_for_unnormalized_curves():
    """Tests that galaxy_fraction_per_bin returns fractions summing to 1
    for unnormalized bins."""
    z = _toy_grid()
    b0 = _gaussian(z, 0.5, 0.1)
    b1 = 2.0 * _gaussian(z, 1.2, 0.2)  # heavier population
    bins = {0: b0, 1: b1}

    fracs = galaxy_fraction_per_bin(z, bins)

    assert fracs[0] > 0.0
    assert fracs[1] > 0.0
    assert np.isclose(sum(fracs.values()), 1.0, atol=1e-12)
    assert fracs[1] > fracs[0]


def test_galaxy_fraction_per_bin_raises_on_nonpositive_total():
    """Tests that galaxy_fraction_per_bin raises ValueError for non-positive total."""
    z = _toy_grid()
    bins = {0: np.zeros_like(z), 1: np.zeros_like(z)}

    with pytest.raises(
        ValueError, match=r"Total integrated n\(z\) over all bins must be positive"
    ):
        galaxy_fraction_per_bin(z, bins)


def test_galaxy_fraction_per_bin_raises_if_bins_look_normalized():
    """Tests that galaxy_fraction_per_bin raises if bins look
    individually normalized."""
    z = _toy_grid()
    p0 = _normalize_pdf(z, _gaussian(z, 0.6, 0.1))
    p1 = _normalize_pdf(z, _gaussian(z, 1.2, 0.2))
    bins = {0: p0, 1: p1}

    with pytest.raises(ValueError, match="Bin curves appear normalized"):
        galaxy_fraction_per_bin(z, bins)


def test_galaxy_density_per_bin_allocates_proportionally_from_integrals():
    """Tests that galaxy_density_per_bin allocates density_total
    proportionally to bin integrals."""
    z = _toy_grid()
    b0 = _gaussian(z, 0.5, 0.1)
    b1 = 3.0 * _gaussian(z, 1.2, 0.2)
    bins = {0: b0, 1: b1}

    density_total = 40.0
    dens, frac = galaxy_density_per_bin(z, bins, density_total)

    assert np.isclose(sum(frac.values()), 1.0, atol=1e-12)
    assert np.isclose(sum(dens.values()), density_total, atol=1e-12)
    assert dens[1] > dens[0]


def test_galaxy_density_per_bin_raises_if_bins_look_normalized_without_frac():
    """Tests that galaxy_density_per_bin raises if bins look normalized
    and frac_per_bin is not provided."""
    z = _toy_grid()
    p0 = _normalize_pdf(z, _gaussian(z, 0.6, 0.1))
    p1 = _normalize_pdf(z, _gaussian(z, 1.2, 0.2))
    bins = {0: p0, 1: p1}

    with pytest.raises(ValueError, match="Bin curves appear normalized"):
        galaxy_density_per_bin(z, bins, density_total=10.0)


def test_galaxy_density_per_bin_allows_normalized_with_explicit_frac():
    """Tests that galaxy_density_per_bin works with normalized bins
    when frac_per_bin is provided."""
    z = _toy_grid()
    p0 = _normalize_pdf(z, _gaussian(z, 0.6, 0.1))
    p1 = _normalize_pdf(z, _gaussian(z, 1.2, 0.2))
    bins = {0: p0, 1: p1}

    dens, frac = galaxy_density_per_bin(
        z,
        bins,
        density_total=10.0,
        frac_per_bin={0: 0.25, 1: 0.75},
    )

    assert frac[0] == pytest.approx(0.25 / (0.25 + 0.75))
    assert frac[1] == pytest.approx(0.75 / (0.25 + 0.75))
    assert np.isclose(sum(dens.values()), 10.0, atol=1e-12)


def test_galaxy_density_per_bin_normalize_false_requires_sum_one():
    """Tests that galaxy_density_per_bin(normalize=False) requires
    frac_per_bin to sum to 1."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    with pytest.raises(
        ValueError, match="normalize=False requires frac_per_bin weights to sum to 1"
    ):
        galaxy_density_per_bin(
            z,
            bins,
            density_total=10.0,
            frac_per_bin={0: 0.2, 1: 0.2},
            normalize=False,
        )


def test_galaxy_count_per_bin_converts_with_area():
    """Tests that galaxy_count_per_bin converts gal/arcmin^2 * arcmin^2
    to counts."""
    dens = {0: 2.5, 1: 10.0}
    area = 100.0
    counts = galaxy_count_per_bin(dens, area)

    assert counts[0] == pytest.approx(250.0)
    assert counts[1] == pytest.approx(1000.0)


def test_galaxy_count_per_bin_raises_on_nonpositive_area():
    """Tests that galaxy_count_per_bin raises ValueError for non-positive
    survey_area."""
    dens = {0: 2.5}
    with pytest.raises(ValueError, match="survey_area must be positive"):
        galaxy_count_per_bin(dens, 0.0)


def test_bin_quantiles_returns_expected_keys_and_ordering_for_symmetric_gaussian():
    """Tests that bin_quantiles returns requested quantiles and they are
    ordered for a symmetric Gaussian."""
    z = _toy_grid()
    mu, sig = 1.0, 0.2
    nz = _gaussian(z, mu, sig)

    qs = bin_quantiles(z, nz, [0.16, 0.5, 0.84])

    assert set(qs.keys()) == {0.16, 0.5, 0.84}
    assert qs[0.16] < qs[0.5] < qs[0.84]
    assert np.isclose(qs[0.5], mu, atol=2e-3)


def test_in_range_fraction_basic_behaviour():
    """Tests that in_range_fraction is in [0, 1] and increases for
    wider intervals."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.2)

    f_narrow = in_range_fraction(z, nz, 0.9, 1.1)
    f_wide = in_range_fraction(z, nz, 0.6, 1.4)

    assert 0.0 <= f_narrow <= 1.0
    assert 0.0 <= f_wide <= 1.0
    assert f_wide > f_narrow


def test_in_range_fraction_raises_on_invalid_bounds():
    """Tests that in_range_fraction raises ValueError when z_max edge is not
    greater than z_min."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.2)

    with pytest.raises(ValueError, match="z_max must be greater than z_min"):
        in_range_fraction(z, nz, 1.0, 1.0)


def test_in_range_fraction_per_bin_accepts_mapping_edges():
    """Tests that in_range_fraction_per_bin works when bin_edges is a mapping."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    edges = {0: (0.3, 0.9), 1: (0.9, 1.6)}

    out = in_range_fraction_per_bin(z, bins, edges)

    assert set(out.keys()) == {0, 1}
    assert 0.0 <= out[0] <= 1.0
    assert 0.0 <= out[1] <= 1.0


def test_in_range_fraction_per_bin_accepts_sequence_edges():
    """Tests that in_range_fraction_per_bin works when bin_edges
    is a sequence."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    edges = [0.0, 1.0, 2.0]

    out = in_range_fraction_per_bin(z, bins, edges)

    assert set(out.keys()) == {0, 1}
    assert 0.0 <= out[0] <= 1.0
    assert 0.0 <= out[1] <= 1.0


def test_peak_flags_single_peak_has_one_peak_and_second_ratio_zero():
    """Tests that peak_flags reports a single Gaussian as single-peaked
    with no second peak."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.2)

    out = peak_flags(z, nz, min_rel_height=0.1)

    assert out["num_peaks"] == pytest.approx(1.0)
    assert out["second_peak_ratio"] == pytest.approx(0.0)
    assert np.isclose(out["mode"], 1.0, atol=2e-3)


def test_peak_flags_detects_two_peaks_and_second_ratio_positive():
    """Tests that peak_flags detects two separated peaks."""
    z = _toy_grid()
    nz = _gaussian(z, 0.7, 0.06) + 0.5 * _gaussian(z, 1.3, 0.06)

    out = peak_flags(z, nz, min_rel_height=0.1)

    assert out["num_peaks"] >= 2.0
    assert 0.0 < out["second_peak_ratio"] < 1.0


def test_peak_flags_per_bin_returns_mapping():
    """Tests that peak_flags_per_bin returns per-bin peak dictionaries."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    out = peak_flags_per_bin(z, bins)

    assert set(out.keys()) == {0, 1}
    assert "mode" in out[0]
    assert "num_peaks" in out[1]
