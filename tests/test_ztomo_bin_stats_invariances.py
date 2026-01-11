"""Unit tests for ``binny.ztomo.bin_stats`` module.

This suite includes:
- correctness checks on Gaussian toy cases
- invariance checks (amplitude scaling, translation, zmin=0 vs eps)
- conservation / consistency checks for bin partitioning
- input validation checks (nonpositive total weight,
 negative weights policy, non-monotonic z)
"""

from __future__ import annotations

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


def _toy_grid() -> np.ndarray:
    """Generates a toy grid for testing."""
    return np.linspace(0.0, 2.0, 2001)


def _gaussian(z: np.ndarray, mu: float, sig: float) -> np.ndarray:
    """Returns an unnormalized Gaussian-shaped curve."""
    return np.exp(-0.5 * ((z - mu) / sig) ** 2)


def _normalize_pdf(z: np.ndarray, nz: np.ndarray) -> np.ndarray:
    """Normalizes a nonnegative curve nz(z) into a PDF on z."""
    norm = float(np.trapezoid(nz, x=z))
    if norm <= 0:
        raise ValueError("Cannot normalize: non-positive integral.")
    return nz / norm


def _bins_from_edges(
    z: np.ndarray, nz: np.ndarray, edges: list[float]
) -> dict[int, np.ndarray]:
    """Partitions nz(z) into bin curves using fixed edges."""
    bins: dict[int, np.ndarray] = {}
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        m = (z >= lo) & (z < hi) if i < len(edges) - 2 else (z >= lo) & (z <= hi)
        bins[i] = nz * m.astype(float)
    return bins


def _make_nonmonotonic_z() -> np.ndarray:
    """Returns a nearly-monotonic grid with a tiny non-monotonic swap."""
    z = _toy_grid().copy()
    z[1000], z[1001] = z[1001], z[1000]
    return z


def _as_clipped_nonnegative(y: np.ndarray) -> np.ndarray:
    """Clips negative values to zero (used for policy-agnostic tests)."""
    return np.clip(y, 0.0, None)


def test_bin_moments_gaussian_mean_std_close() -> None:
    """bin_moments recovers mean/std for a Gaussian n(z)."""
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
    """Tests that bin_moments raises on nonpositive total weight."""
    z = _toy_grid()
    nz = np.zeros_like(z)

    with pytest.raises(ValueError, match="Total weight must be positive"):
        bin_moments(z, nz)


def test_bin_centers_methods_and_rounding() -> None:
    """Tests that bin_centers methods round to the correct precision."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    centers = bin_centers(z, bins, method="median", decimal_places=2)
    assert set(centers.keys()) == {0, 1}
    assert isinstance(centers[0], float)

    centers_full = bin_centers(z, bins, method="median", decimal_places=None)
    assert centers[0] == pytest.approx(round(centers_full[0], 2))
    assert centers[1] == pytest.approx(round(centers_full[1], 2))


def test_bin_centers_percentile_method_p50_matches_median() -> None:
    """Tests that bin_centers p50 method matches median."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    c_med = bin_centers(z, bins, method="median", decimal_places=None)
    c_p50 = bin_centers(z, bins, method="p50", decimal_places=None)

    assert c_p50[0] == pytest.approx(c_med[0])
    assert c_p50[1] == pytest.approx(c_med[1])


def test_bin_centers_percentile_invalid_raises() -> None:
    """Tests that bin_centers pXX methods raise on invalid percentile method."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1)}

    with pytest.raises(ValueError, match="percentile methods must look like"):
        bin_centers(z, bins, method="pXX")

    with pytest.raises(
        ValueError, match="percentile in method='pXX' must be between 0 and 100"
    ):
        bin_centers(z, bins, method="p120")


def test_summarize_bins_includes_sigma_mean_from_scalar() -> None:
    """Tests that summarize_bins includes sigma_mean from scalar sigma_mean_factor."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    out = summarize_bins(z, bins, sigma_mean=0.03)

    assert set(out.keys()) == {0, 1}
    assert out[0]["sigma_mean"] == pytest.approx(0.03)
    assert out[1]["sigma_mean"] == pytest.approx(0.03)
    assert "mean" in out[0] and "std" in out[0]


def test_summarize_bins_sigma_mean_sequence_length_mismatch_raises() -> None:
    """Tests that summarize_bins raises on sigma_mean sequence length mismatch."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    with pytest.raises(ValueError, match="sigma_mean sequence must have length"):
        summarize_bins(z, bins, sigma_mean=[0.1])


def test_summarize_bins_sigma_mean_mapping_missing_key_raises() -> None:
    """Tests that summarize_bins raises on sigma_mean mapping missing key."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.5, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    with pytest.raises(ValueError, match=r"sigma_mean is missing bin index 1"):
        summarize_bins(z, bins, sigma_mean={0: 0.01})


def test_summarize_bins_sigma_mean_from_count_matches_std_over_sqrt_count() -> None:
    """Tests that summarize_bins sigma_mean_factor=1.0 matches std/sqrt(count)."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.8, 0.2)}
    counts = {0: 400.0}

    out = summarize_bins(z, bins, count_per_bin=counts)
    stats = bin_moments(z, bins[0])

    assert out[0]["mean"] == pytest.approx(stats["mean"])
    assert out[0]["std"] == pytest.approx(stats["std"])
    assert out[0]["sigma_mean"] == pytest.approx(stats["std"] / np.sqrt(counts[0]))


def test_summarize_bins_sigma_mean_from_density_and_area() -> None:
    """Tests that summarize_bins sigma_mean_factor=1.0 matches std/sqrt(count)."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.8, 0.2)}
    dens = {0: 10.0}
    area = 100.0

    out = summarize_bins(z, bins, density_per_bin=dens, survey_area=area)
    stats = bin_moments(z, bins[0])

    count = dens[0] * area
    assert out[0]["sigma_mean"] == pytest.approx(stats["std"] / np.sqrt(count))


def test_summarize_bins_density_requires_survey_area() -> None:
    """Tests that summarize_bins raises on density_per_bin without survey_area."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.8, 0.2)}
    dens = {0: 10.0}

    with pytest.raises(ValueError, match="survey_area must be provided"):
        summarize_bins(z, bins, density_per_bin=dens)


def test_galaxy_fraction_per_bin_sums_to_one_for_unnormalized_curves() -> None:
    """Tests that galaxy_fraction_per_bin sums to one for unnormalized curves."""
    z = _toy_grid()
    b0 = _gaussian(z, 0.5, 0.1)
    b1 = 2.0 * _gaussian(z, 1.2, 0.2)
    bins = {0: b0, 1: b1}

    fracs = galaxy_fraction_per_bin(z, bins)

    assert fracs[0] > 0.0
    assert fracs[1] > 0.0
    assert np.isclose(sum(fracs.values()), 1.0, atol=1e-12)
    assert fracs[1] > fracs[0]


def test_galaxy_fraction_per_bin_raises_on_nonpositive_total() -> None:
    """Tests that galaxy_fraction_per_bin raises on nonpositive total integral."""
    z = _toy_grid()
    bins = {0: np.zeros_like(z), 1: np.zeros_like(z)}

    with pytest.raises(
        ValueError, match=r"Total integrated n\(z\) over all bins must be positive"
    ):
        galaxy_fraction_per_bin(z, bins)


def test_galaxy_fraction_per_bin_raises_if_bins_look_normalized() -> None:
    """Tests that galaxy_fraction_per_bin raises if bins look normalized."""
    z = _toy_grid()
    p0 = _normalize_pdf(z, _gaussian(z, 0.6, 0.1))
    p1 = _normalize_pdf(z, _gaussian(z, 1.2, 0.2))
    bins = {0: p0, 1: p1}

    with pytest.raises(ValueError, match="Bin curves appear normalized"):
        galaxy_fraction_per_bin(z, bins)


def test_galaxy_density_per_bin_allocates_proportionally_from_integrals() -> None:
    """Tests that galaxy_density_per_bin allocates proportional to bin integrals."""
    z = _toy_grid()
    b0 = _gaussian(z, 0.5, 0.1)
    b1 = 3.0 * _gaussian(z, 1.2, 0.2)
    bins = {0: b0, 1: b1}

    density_total = 40.0
    dens, frac = galaxy_density_per_bin(z, bins, density_total)

    assert np.isclose(sum(frac.values()), 1.0, atol=1e-12)
    assert np.isclose(sum(dens.values()), density_total, atol=1e-12)
    assert dens[1] > dens[0]


def test_galaxy_density_per_bin_raises_if_bins_look_normalized_without_frac() -> None:
    """Tests that galaxy_density_per_bin raises if bins look normalized."""
    z = _toy_grid()
    p0 = _normalize_pdf(z, _gaussian(z, 0.6, 0.1))
    p1 = _normalize_pdf(z, _gaussian(z, 1.2, 0.2))
    bins = {0: p0, 1: p1}

    with pytest.raises(ValueError, match="Bin curves appear normalized"):
        galaxy_density_per_bin(z, bins, density_total=10.0)


def test_galaxy_density_per_bin_allows_normalized_with_explicit_frac() -> None:
    """Tests that galaxy_density_per_bin allows normalized curves."""
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


def test_galaxy_density_per_bin_normalize_false_requires_sum_one() -> None:
    """Tests that galaxy_density_per_bin raises if frac_per_bin does not sum to 1.0."""
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


def test_galaxy_count_per_bin_converts_with_area() -> None:
    """Tests that galaxy_count_per_bin converts counts to galaxies per bin area."""
    dens = {0: 2.5, 1: 10.0}
    area = 100.0
    counts = galaxy_count_per_bin(dens, area)

    assert counts[0] == pytest.approx(250.0)
    assert counts[1] == pytest.approx(1000.0)


def test_galaxy_count_per_bin_raises_on_nonpositive_area() -> None:
    """Tests that galaxy_count_per_bin raises on nonpositive area."""
    dens = {0: 2.5}
    with pytest.raises(ValueError, match="survey_area must be positive"):
        galaxy_count_per_bin(dens, 0.0)


def test_bin_quantiles_returns_expected_keys_and_ordering_for_symmetric_gaussian() -> (
    None
):
    """Tests that bin_quantiles returns expected keys and ordering for
    symmetric Gaussian."""
    z = _toy_grid()
    mu, sig = 1.0, 0.2
    nz = _gaussian(z, mu, sig)

    qs = bin_quantiles(z, nz, [0.16, 0.5, 0.84])

    assert set(qs.keys()) == {0.16, 0.5, 0.84}
    assert qs[0.16] < qs[0.5] < qs[0.84]
    assert np.isclose(qs[0.5], mu, atol=2e-3)


def test_in_range_fraction_basic_behaviour() -> None:
    """Test that in_range_fraction returns fraction of nz in range [z_min, z_max]."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.2)

    f_narrow = in_range_fraction(z, nz, 0.9, 1.1)
    f_wide = in_range_fraction(z, nz, 0.6, 1.4)

    assert 0.0 <= f_narrow <= 1.0
    assert 0.0 <= f_wide <= 1.0
    assert f_wide > f_narrow


def test_in_range_fraction_raises_on_invalid_bounds() -> None:
    """Test that in_range_fraction raises on invalid bounds."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.2)

    with pytest.raises(ValueError, match="z_max must be greater than z_min"):
        in_range_fraction(z, nz, 1.0, 1.0)


def test_in_range_fraction_per_bin_accepts_mapping_edges() -> None:
    """Tests that in_range_fraction_per_bin accepts mapping for bin_edges."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    edges = {0: (0.3, 0.9), 1: (0.9, 1.6)}

    out = in_range_fraction_per_bin(z, bins, edges)

    assert set(out.keys()) == {0, 1}
    assert 0.0 <= out[0] <= 1.0
    assert 0.0 <= out[1] <= 1.0


def test_in_range_fraction_per_bin_accepts_sequence_edges() -> None:
    """Tests that in_range_fraction_per_bin accepts sequence for bin_edges."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    edges = [0.0, 1.0, 2.0]

    out = in_range_fraction_per_bin(z, bins, edges)

    assert set(out.keys()) == {0, 1}
    assert 0.0 <= out[0] <= 1.0
    assert 0.0 <= out[1] <= 1.0


def test_peak_flags_single_peak_has_one_peak_and_second_ratio_zero() -> None:
    """Tests that peak_flags detects a single peak and second_peak_ratio=0.0."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.2)

    out = peak_flags(z, nz, min_rel_height=0.1)

    assert out["num_peaks"] == pytest.approx(1.0)
    assert out["second_peak_ratio"] == pytest.approx(0.0)
    assert np.isclose(out["mode"], 1.0, atol=2e-3)


def test_peak_flags_detects_two_peaks_and_second_ratio_positive() -> None:
    """Tests that peak_flags detects two peaks and positive second_peak_ratio."""
    z = _toy_grid()
    nz = _gaussian(z, 0.7, 0.06) + 0.5 * _gaussian(z, 1.3, 0.06)

    out = peak_flags(z, nz, min_rel_height=0.1)

    assert out["num_peaks"] >= 2.0
    assert 0.0 < out["second_peak_ratio"] < 1.0


def test_peak_flags_per_bin_returns_mapping() -> None:
    """Tests that peak_flags_per_bin returns a mapping of peak flags per bin."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    out = peak_flags_per_bin(z, bins)

    assert set(out.keys()) == {0, 1}
    assert "mode" in out[0]
    assert "num_peaks" in out[1]


def test_amplitude_scale_invariance_core_stats() -> None:
    """Tests that amplitude scale invariance holds for core stats."""
    z = _toy_grid()
    nz = _gaussian(z, 0.9, 0.2)
    nz2 = 7.3 * nz

    m1 = bin_moments(z, nz)
    m2 = bin_moments(z, nz2)
    for k in ("mean", "median", "std"):
        assert m2[k] == pytest.approx(m1[k], rel=0.0, abs=2e-3)

    q1 = bin_quantiles(z, nz, [0.16, 0.5, 0.84])
    q2 = bin_quantiles(z, nz2, [0.16, 0.5, 0.84])
    for p in q1:
        assert q2[p] == pytest.approx(q1[p], rel=0.0, abs=2e-3)

    f1 = in_range_fraction(z, nz, 0.6, 1.2)
    f2 = in_range_fraction(z, nz2, 0.6, 1.2)
    assert f2 == pytest.approx(f1, rel=0.0, abs=1e-12)


def test_partition_integral_conservation_and_fraction_consistency() -> None:
    """Tests that partition integral conservation and fraction consistency
    hold."""
    z = np.linspace(0.0, 2.0, 4001)
    nz = _gaussian(z, 0.9, 0.25)

    edges = [0.0, 0.7, 1.1, 2.0]
    bins = _bins_from_edges(z, nz, edges)

    total = float(np.trapezoid(nz, x=z))
    total_bins = float(np.trapezoid(sum(bins.values()), x=z))
    assert total_bins == pytest.approx(total, rel=0.0, abs=1e-10)

    fracs = galaxy_fraction_per_bin(z, bins)
    assert sum(fracs.values()) == pytest.approx(1.0, rel=0.0, abs=1e-12)

    ints = {i: float(np.trapezoid(bins[i], x=z)) for i in bins}
    denom = float(sum(ints.values()))
    for i in bins:
        assert fracs[i] == pytest.approx(ints[i] / denom, rel=0.0, abs=1e-12)


def test_zmin_zero_vs_eps_robust_for_smooth_distribution() -> None:
    """Tests that zmin=0.0 is robust to smooth distribution."""
    mu, sig = 0.9, 0.2
    z0 = np.linspace(0.0, 2.0, 3001)
    z1 = np.linspace(1e-4, 2.0, 3001)

    nz0 = _gaussian(z0, mu, sig)
    nz1 = _gaussian(z1, mu, sig)

    m0 = bin_moments(z0, nz0)
    m1 = bin_moments(z1, nz1)

    assert m1["mean"] == pytest.approx(m0["mean"], rel=0.0, abs=2e-3)
    assert m1["median"] == pytest.approx(m0["median"], rel=0.0, abs=2e-3)


def test_translation_invariance_for_centers_and_quantiles() -> None:
    """Tests that translation invariance holds for centers and quantiles."""
    z = np.linspace(0.0, 2.0, 3001)
    nz = _gaussian(z, 0.9, 0.2)

    shift = 0.3
    zs = z + shift
    nzs = _gaussian(zs - shift, 0.9, 0.2)

    q = bin_quantiles(z, nz, [0.16, 0.5, 0.84])
    qs = bin_quantiles(zs, nzs, [0.16, 0.5, 0.84])
    for p in q:
        assert qs[p] == pytest.approx(q[p] + shift, rel=0.0, abs=2e-3)

    c = bin_centers(z, {0: nz}, method="median", decimal_places=None)[0]
    cs = bin_centers(zs, {0: nzs}, method="median", decimal_places=None)[0]
    assert cs == pytest.approx(c + shift, rel=0.0, abs=2e-3)


def test_negative_weights_policy_bin_moments_is_reasonable() -> None:
    """Tests that negative weights policy for bin_moments is reasonable."""
    z = _toy_grid()
    nz = _gaussian(z, 0.9, 0.2)
    nz_bad = nz.copy()
    nz_bad[800:820] *= -1.0

    try:
        m_bad = bin_moments(z, nz_bad)
    except ValueError:
        return

    m_clip = bin_moments(z, _as_clipped_nonnegative(nz_bad))
    for k in ("mean", "median", "std"):
        assert m_bad[k] == pytest.approx(m_clip[k], rel=0.0, abs=2e-3)


def test_negative_weights_policy_bin_quantiles_is_reasonable() -> None:
    """Tests that negative weights policy for bin_quantiles is reasonable."""
    z = _toy_grid()
    nz = _gaussian(z, 0.9, 0.2)
    nz_bad = nz.copy()
    nz_bad[800:820] *= -1.0

    ps = [0.16, 0.5, 0.84]

    try:
        q_bad = bin_quantiles(z, nz_bad, ps)
    except ValueError:
        return

    q_clip = bin_quantiles(z, _as_clipped_nonnegative(nz_bad), ps)
    for p in ps:
        assert q_bad[p] == pytest.approx(q_clip[p], rel=0.0, abs=2e-3)


def test_negative_weights_policy_bin_centers_is_reasonable() -> None:
    """Tests that negative weights policy for bin_centers is reasonable."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    bins_bad = {k: v.copy() for k, v in bins.items()}
    bins_bad[1][900:920] *= -1.0

    try:
        c_bad = bin_centers(z, bins_bad, method="median", decimal_places=None)
    except ValueError:
        return

    c_clip = bin_centers(
        z,
        {k: _as_clipped_nonnegative(v) for k, v in bins_bad.items()},
        method="median",
        decimal_places=None,
    )
    assert c_bad[0] == pytest.approx(c_clip[0], rel=0.0, abs=2e-3)
    assert c_bad[1] == pytest.approx(c_clip[1], rel=0.0, abs=2e-3)


def test_negative_weights_policy_peak_flags_is_reasonable() -> None:
    """Tests that negative weights policy for peak_flags is reasonable."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.2)
    nz_bad = nz.copy()
    nz_bad[800:820] *= -1.0

    try:
        p_bad = peak_flags(z, nz_bad, min_rel_height=0.1)
    except ValueError:
        return

    p_clip = peak_flags(z, _as_clipped_nonnegative(nz_bad), min_rel_height=0.1)
    assert p_bad["num_peaks"] == pytest.approx(p_clip["num_peaks"])
    assert p_bad["second_peak_ratio"] == pytest.approx(p_clip["second_peak_ratio"])
    assert p_bad["mode"] == pytest.approx(p_clip["mode"], rel=0.0, abs=2e-3)


def test_negative_weights_policy_in_range_fraction_is_reasonable() -> None:
    """Tests that negative weights policy for in_range_fraction is reasonable."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.2)
    nz_bad = nz.copy()
    nz_bad[800:820] *= -1.0

    # Either: raise OR return something sensible
    try:
        f_bad = in_range_fraction(z, nz_bad, 0.6, 1.4)
    except ValueError:
        return

    assert np.isfinite(f_bad)
    assert 0.0 <= f_bad <= 1.0

    f_clip = in_range_fraction(z, _as_clipped_nonnegative(nz_bad), 0.6, 1.4)
    assert f_bad == pytest.approx(f_clip, rel=0.0, abs=5e-3)


def test_negative_weights_policy_galaxy_fraction_is_reasonable() -> None:
    """Tests that negative weights policy for galaxy_fraction_per_bin is reasonable."""
    z = _toy_grid()
    b0 = _gaussian(z, 0.5, 0.1)
    b1 = _gaussian(z, 1.2, 0.2)
    b1_bad = b1.copy()
    b1_bad[900:920] *= -1.0
    bins_bad = {0: b0, 1: b1_bad}

    try:
        fr_bad = galaxy_fraction_per_bin(z, bins_bad)
    except ValueError:
        return

    assert all(np.isfinite(v) for v in fr_bad.values())
    assert all(v >= 0.0 for v in fr_bad.values())
    assert sum(fr_bad.values()) == pytest.approx(1.0, rel=0.0, abs=1e-12)

    # Optional: compare to clipped baseline with a loose tolerance
    fr_clip = galaxy_fraction_per_bin(
        z, {k: _as_clipped_nonnegative(v) for k, v in bins_bad.items()}
    )
    for k in fr_bad:
        assert fr_bad[k] == pytest.approx(fr_clip[k], rel=0.0, abs=5e-3)


def test_negative_weights_policy_galaxy_density_is_reasonable() -> None:
    """Tests that negative weights policy for galaxy_density_per_bin is reasonable."""
    z = _toy_grid()
    b0 = _gaussian(z, 0.5, 0.1)
    b1 = _gaussian(z, 1.2, 0.2)
    b1_bad = b1.copy()
    b1_bad[900:920] *= -1.0
    bins_bad = {0: b0, 1: b1_bad}

    try:
        dens_bad, frac_bad = galaxy_density_per_bin(z, bins_bad, density_total=10.0)
    except ValueError:
        return

    assert all(np.isfinite(v) for v in dens_bad.values())
    assert all(v >= 0.0 for v in dens_bad.values())
    assert np.isclose(sum(dens_bad.values()), 10.0, atol=1e-10)

    assert all(np.isfinite(v) for v in frac_bad.values())
    assert all(v >= 0.0 for v in frac_bad.values())
    assert sum(frac_bad.values()) == pytest.approx(1.0, rel=0.0, abs=1e-12)

    dens_clip, frac_clip = galaxy_density_per_bin(
        z,
        {k: _as_clipped_nonnegative(v) for k, v in bins_bad.items()},
        density_total=10.0,
    )
    for k in frac_bad:
        assert frac_bad[k] == pytest.approx(frac_clip[k], rel=0.0, abs=5e-3)


def test_bin_moments_raises_on_nonmonotonic_z() -> None:
    """Tests that bin_moments raises on nonmonotonic z."""
    z = _make_nonmonotonic_z()
    nz = _gaussian(z, 0.9, 0.2)

    with pytest.raises(ValueError, match="monotonic|sorted|increasing"):
        bin_moments(z, nz)


def test_bin_quantiles_raises_on_nonmonotonic_z() -> None:
    """Tests that bin_quantiles raises on nonmonotonic z."""
    z = _make_nonmonotonic_z()
    nz = _gaussian(z, 0.9, 0.2)

    with pytest.raises(ValueError, match="monotonic|sorted|increasing"):
        bin_quantiles(z, nz, [0.16, 0.5, 0.84])


def test_bin_centers_raises_on_nonmonotonic_z() -> None:
    """Tests that bin_centers raises on nonmonotonic z."""
    z = _make_nonmonotonic_z()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    with pytest.raises(ValueError, match="monotonic|sorted|increasing"):
        bin_centers(z, bins, method="median", decimal_places=None)
