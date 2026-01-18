"""Unit tests for ``binny.nz_tomo.bin_stats`` module."""

from __future__ import annotations

import numpy as np
import pytest

from binny.nz_tomo.bin_stats import (
    bin_centers,
    bin_moments,
    bin_quantiles,
    galaxy_count_per_bin,
    galaxy_fraction_per_bin,
    in_range_fraction,
    in_range_fraction_per_bin,
    peak_flags,
    peak_flags_per_bin,
    population_stats,
    shape_stats,
)


def _toy_grid() -> np.ndarray:
    """Creates a toy redshift grid for testing."""
    return np.linspace(0.0, 2.0, 2001)


def _gaussian(z: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """Creates a Gaussian curve on grid z with mean mu and std sig."""
    return np.exp(-0.5 * ((z - mu) / sig) ** 2)


def _normalize_pdf(z: np.ndarray, nz: np.ndarray) -> np.ndarray:
    """Normalizes a nonnegative curve nz(z) into a PDF on z."""
    norm = np.trapezoid(nz, x=z)
    if norm <= 0.0:
        raise ValueError("Cannot normalize: non-positive integral.")
    return nz / norm


def test_bin_moments_gaussian_mean_std_close() -> None:
    """Tests that bin_moments recovers mean and std for a Gaussian curve."""
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
    assert np.isclose(stats["median"], mu, rtol=0.0, atol=2e-3)
    assert np.isclose(stats["mode"], mu, rtol=0.0, atol=2e-3)

    assert stats["width_68"] > 0.0
    assert stats["iqr"] > 0.0


def test_bin_moments_raises_on_nonpositive_total_weight() -> None:
    """Tests that bin_moments raises ValueError for non-positive total weight."""
    z = _toy_grid()
    nz = np.zeros_like(z)

    with pytest.raises(ValueError, match="Total weight must be positive"):
        bin_moments(z, nz)


def test_bin_centers_methods_and_rounding() -> None:
    """Tests that bin_centers supports methods and rounding controls."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    centers = bin_centers(z, bins, method="median", decimal_places=2)
    assert set(centers.keys()) == {0, 1}
    assert isinstance(centers[0], float)

    centers_full = bin_centers(z, bins, method="median", decimal_places=None)
    assert centers[0] == pytest.approx(round(centers_full[0], 2))
    assert centers[1] == pytest.approx(round(centers_full[1], 2))


def test_bin_centers_percentile_method_p50_matches_median() -> None:
    """Tests that bin_centers(method='p50') matches method='median'."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    c_med = bin_centers(z, bins, method="median", decimal_places=None)
    c_p50 = bin_centers(z, bins, method="p50", decimal_places=None)

    assert c_p50[0] == pytest.approx(c_med[0])
    assert c_p50[1] == pytest.approx(c_med[1])


def test_bin_centers_percentile_invalid_raises() -> None:
    """Tests that bin_centers raises ValueError for invalid percentile methods."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1)}

    with pytest.raises(ValueError, match="percentile methods must look like"):
        bin_centers(z, bins, method="pXX")

    msg = "percentile in method='pXX' must be between 0 and 100"
    with pytest.raises(ValueError, match=msg):
        bin_centers(z, bins, method="p120")


def test_shape_stats_returns_expected_top_level_keys() -> None:
    """Tests that shape_stats returns the expected top-level structure."""
    z = _toy_grid()
    p0 = _normalize_pdf(z, _gaussian(z, 0.6, 0.1))
    p1 = _normalize_pdf(z, _gaussian(z, 1.2, 0.2))
    bins = {0: p0, 1: p1}

    out = shape_stats(z, bins)

    assert set(out.keys()) == {"centers", "peaks", "per_bin"}
    assert set(out["per_bin"].keys()) == {0, 1}
    assert set(out["centers"].keys()) == {0, 1}
    assert set(out["peaks"].keys()) == {0, 1}


def test_shape_stats_per_bin_entries_have_expected_fields() -> None:
    """Tests that shape_stats(per_bin) entries contain expected fields."""
    z = _toy_grid()
    b0 = _gaussian(z, 0.6, 0.1)
    b1 = 2.0 * _gaussian(z, 1.3, 0.15)
    bins = {0: b0, 1: b1}

    out = shape_stats(
        z,
        bins,
        center_method="median",
        decimal_places=None,
        quantiles=(0.16, 0.5, 0.84),
        min_rel_height=0.1,
    )

    for i in (0, 1):
        entry = out["per_bin"][i]
        assert set(entry.keys()) == {"moments", "center", "quantiles", "peaks"}
        assert set(entry["moments"].keys()) >= {"mean", "median", "mode", "std"}
        assert isinstance(entry["center"], float)
        assert set(entry["quantiles"].keys()) == {0.16, 0.5, 0.84}
        assert set(entry["peaks"].keys()) == {
            "mode",
            "mode_height",
            "num_peaks",
            "second_peak_ratio",
        }


def test_shape_stats_includes_in_range_fraction_when_edges_given() -> None:
    """Tests that shape_stats includes in_range_fraction when bin_edges is passed."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    edges = {0: (0.3, 0.9), 1: (0.9, 1.6)}

    out = shape_stats(z, bins, bin_edges=edges)

    assert "in_range_fraction" in out
    assert set(out["in_range_fraction"].keys()) == {0, 1}
    assert 0.0 <= out["in_range_fraction"][0] <= 1.0
    assert 0.0 <= out["in_range_fraction"][1] <= 1.0


def test_galaxy_fraction_per_bin_normalizes_and_casts_keys() -> None:
    """Tests that galaxy_fraction_per_bin normalizes and casts keys to int."""
    meta = {"frac_per_bin": {"0": 2.0, 1: 6.0}}

    fracs = galaxy_fraction_per_bin(meta)

    assert set(fracs.keys()) == {0, 1}
    assert np.isclose(sum(fracs.values()), 1.0, atol=1e-12)
    assert fracs[1] > fracs[0]


def test_galaxy_fraction_per_bin_raises_if_missing_or_bad() -> None:
    """Tests that galaxy_fraction_per_bin raises on missing/invalid metadata."""
    with pytest.raises(ValueError, match="must contain a mapping 'frac_per_bin'"):
        galaxy_fraction_per_bin({})

    with pytest.raises(ValueError, match="must contain a mapping 'frac_per_bin'"):
        galaxy_fraction_per_bin({"frac_per_bin": 123})

    msg = "Sum of metadata frac_per_bin must be positive"
    with pytest.raises(ValueError, match=msg):
        galaxy_fraction_per_bin({"frac_per_bin": {0: 0.0, 1: 0.0}})


def test_population_stats_uses_metadata_fractions_and_normalizes_default() -> None:
    """Tests that population_stats uses metadata fractions and normalizes by default."""
    z = _toy_grid()
    bins = {
        0: _normalize_pdf(z, _gaussian(z, 0.6, 0.1)),
        1: _normalize_pdf(z, _gaussian(z, 1.2, 0.2)),
    }
    meta = {"frac_per_bin": {0: 0.25, 1: 0.75}}

    out = population_stats(bins, meta)

    assert set(out.keys()) == {"fractions"}
    assert np.isclose(sum(out["fractions"].values()), 1.0, atol=1e-12)
    assert out["fractions"][1] > out["fractions"][0]


def test_population_stats_density_and_counts_allocation() -> None:
    """Tests that population_stats allocates densities and counts from fractions."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    meta = {"frac_per_bin": {0: 1.0, 1: 3.0}}

    out = population_stats(bins, meta, density_total=40.0, survey_area=100.0)

    assert np.isclose(sum(out["fractions"].values()), 1.0, atol=1e-12)
    assert np.isclose(sum(out["density_per_bin"].values()), 40.0, atol=1e-12)
    total_counts = sum(out["count_per_bin"].values())
    assert np.isclose(total_counts, 40.0 * 100.0, atol=1e-12)
    assert out["density_per_bin"][1] > out["density_per_bin"][0]
    assert out["count_per_bin"][1] > out["count_per_bin"][0]


def test_population_stats_survey_area_requires_density_total() -> None:
    """Tests that population_stats rejects survey_area without density_total."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1)}
    meta = {"frac_per_bin": {0: 1.0}}

    with pytest.raises(ValueError, match="survey_area requires density_total"):
        population_stats(bins, meta, survey_area=100.0)


def test_population_stats_missing_bin_index_raises() -> None:
    """Tests that population_stats raises if metadata is missing a bin index."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    meta = {"frac_per_bin": {0: 1.0}}

    with pytest.raises(ValueError, match="missing bin index 1"):
        population_stats(bins, meta)


def test_galaxy_count_per_bin_converts_with_area() -> None:
    """Tests that galaxy_count_per_bin converts density * area to counts."""
    dens = {0: 2.5, 1: 10.0}
    area = 100.0

    counts = galaxy_count_per_bin(dens, area)

    assert counts[0] == pytest.approx(250.0)
    assert counts[1] == pytest.approx(1000.0)


def test_galaxy_count_per_bin_raises_on_nonpositive_area() -> None:
    """Tests that galaxy_count_per_bin raises ValueError for non-positive area."""
    dens = {0: 2.5}

    with pytest.raises(ValueError, match="survey_area must be positive"):
        galaxy_count_per_bin(dens, 0.0)


def test_bin_quantiles_returns_expected_keys_and_ordering() -> None:
    """Tests that bin_quantiles returns requested quantiles in correct order."""
    z = _toy_grid()
    mu, sig = 1.0, 0.2
    nz = _gaussian(z, mu, sig)

    qs = bin_quantiles(z, nz, [0.16, 0.5, 0.84])

    assert set(qs.keys()) == {0.16, 0.5, 0.84}
    assert qs[0.16] < qs[0.5] < qs[0.84]
    assert np.isclose(qs[0.5], mu, atol=2e-3)


def test_in_range_fraction_basic_behaviour() -> None:
    """Tests that in_range_fraction is in [0, 1] and increases with width."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.2)

    f_narrow = in_range_fraction(z, nz, 0.9, 1.1)
    f_wide = in_range_fraction(z, nz, 0.6, 1.4)

    assert 0.0 <= f_narrow <= 1.0
    assert 0.0 <= f_wide <= 1.0
    assert f_wide > f_narrow


def test_in_range_fraction_raises_on_invalid_bounds() -> None:
    """Tests that in_range_fraction raises ValueError for invalid bounds."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.2)

    with pytest.raises(ValueError, match="z_max must be greater than z_min"):
        in_range_fraction(z, nz, 1.0, 1.0)


def test_in_range_fraction_per_bin_accepts_mapping_edges() -> None:
    """Tests that in_range_fraction_per_bin accepts mapping bin_edges."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    edges = {0: (0.3, 0.9), 1: (0.9, 1.6)}

    out = in_range_fraction_per_bin(z, bins, edges)

    assert set(out.keys()) == {0, 1}
    assert 0.0 <= out[0] <= 1.0
    assert 0.0 <= out[1] <= 1.0


def test_in_range_fraction_per_bin_accepts_sequence_edges() -> None:
    """Tests that in_range_fraction_per_bin accepts sequence bin_edges."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    edges = [0.0, 1.0, 2.0]

    out = in_range_fraction_per_bin(z, bins, edges)

    assert set(out.keys()) == {0, 1}
    assert 0.0 <= out[0] <= 1.0
    assert 0.0 <= out[1] <= 1.0


def test_peak_flags_single_peak_has_one_peak_and_second_ratio_zero() -> None:
    """Tests that peak_flags reports a Gaussian as single-peaked."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.2)

    out = peak_flags(z, nz, min_rel_height=0.1)

    assert out["num_peaks"] == pytest.approx(1.0)
    assert out["second_peak_ratio"] == pytest.approx(0.0)
    assert np.isclose(out["mode"], 1.0, atol=2e-3)


def test_peak_flags_detects_two_peaks_and_second_ratio_positive() -> None:
    """Tests that peak_flags detects two separated peaks."""
    z = _toy_grid()
    nz = _gaussian(z, 0.7, 0.06) + 0.5 * _gaussian(z, 1.3, 0.06)

    out = peak_flags(z, nz, min_rel_height=0.1)

    assert out["num_peaks"] >= 2.0
    assert 0.0 < out["second_peak_ratio"] < 1.0


def test_peak_flags_per_bin_returns_mapping() -> None:
    """Tests that peak_flags_per_bin returns per-bin peak dictionaries."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    out = peak_flags_per_bin(z, bins)

    assert set(out.keys()) == {0, 1}
    assert "mode" in out[0]
    assert "num_peaks" in out[1]


def test_peak_flags_per_bin_respects_sorted_bin_indices() -> None:
    """Tests that peak_flags_per_bin uses sorted bin keys in its output."""
    z = _toy_grid()
    bins = {2: _gaussian(z, 1.4, 0.1), 0: _gaussian(z, 0.6, 0.1)}

    out = peak_flags_per_bin(z, bins)

    assert list(out.keys()) == [0, 2]


def test_shape_stats_raises_on_empty_bins() -> None:
    """Tests that shape_stats raises ValueError for empty bins mapping."""
    z = _toy_grid()
    with pytest.raises(ValueError, match=r"bins must not be empty"):
        shape_stats(z, {})


def test_shape_stats_propagates_invalid_center_method() -> None:
    """Tests that shape_stats propagates errors for invalid center_method."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1)}
    with pytest.raises(ValueError, match=r'method must be "mean", "median", "mode", or "pXX"'):
        shape_stats(z, bins, center_method="nope")


def test_shape_stats_rejects_invalid_quantiles() -> None:
    """Tests that shape_stats rejects quantiles outside [0, 1]."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1)}
    with pytest.raises(ValueError, match=r"q must be between 0 and 1"):
        shape_stats(z, bins, quantiles=(0.5, 1.1))


def test_in_range_fraction_per_bin_missing_index_in_mapping_raises() -> None:
    """Tests that in_range_fraction_per_bin raises when mapping edges miss a bin."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    edges = {0: (0.3, 0.9)}  # missing bin 1
    with pytest.raises(ValueError, match=r"bin_edges is missing bin index 1"):
        in_range_fraction_per_bin(z, bins, edges)


def test_in_range_fraction_per_bin_sequence_too_short_raises() -> None:
    """Tests that in_range_fraction_per_bin raises when edge sequence is too short."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    edges = [0.0, 1.0]  # cannot cover bin 1
    with pytest.raises(ValueError, match=r"bin_edges sequence is too short"):
        in_range_fraction_per_bin(z, bins, edges)


def test_peak_flags_short_curve_branch_returns_one_peak() -> None:
    """Tests that peak_flags handles curves with fewer than 3 points."""
    z = np.array([0.0, 1.0], dtype=float)
    nz = np.array([0.0, 2.0], dtype=float)
    out = peak_flags(z, nz, min_rel_height=0.1)
    assert out["num_peaks"] == pytest.approx(1.0)
    assert out["second_peak_ratio"] == pytest.approx(0.0)
    assert out["mode"] == pytest.approx(1.0)


def test_peak_flags_nonpositive_max_returns_zero_peaks() -> None:
    """Tests that peak_flags returns zero peaks when global max is non-positive."""
    z = _toy_grid()
    nz = np.zeros_like(z)
    out = peak_flags(z, nz, min_rel_height=0.1)
    assert out["num_peaks"] == pytest.approx(0.0)
    assert out["second_peak_ratio"] == pytest.approx(0.0)


def test_peak_flags_height_filter_can_remove_all_peaks() -> None:
    """Tests that peak_flags returns zero peaks if all detected peaks fail height cut."""
    z = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    # Big global max at endpoint (not counted as a peak), tiny interior bump at z=2.
    nz = np.array([10.0, 0.0, 0.1, 0.0, 0.0], dtype=float)

    out = peak_flags(z, nz, min_rel_height=0.5)  # threshold = 5.0

    assert out["num_peaks"] == pytest.approx(0.0)
    assert out["second_peak_ratio"] == pytest.approx(0.0)


def test_population_stats_normalize_frac_false_rejects_not_sum_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that population_stats rejects non-unit sums when normalize_frac is False."""
    import binny.nz_tomo.bin_stats as bsmod

    def _fake_frac(_meta):
        return {0: 0.2, 1: 0.2}  # sum = 0.4

    monkeypatch.setattr(bsmod, "galaxy_fraction_per_bin", _fake_frac)

    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    meta = {"frac_per_bin": {0: 1.0, 1: 1.0}}  # doesn't matter now

    with pytest.raises(ValueError, match=r"normalize_frac=False requires fractions to sum to 1"):
        population_stats(bins, meta, normalize_frac=False, rtol=0.0, atol=0.0)
