"""Unit tests for ``binny.nz_tomo.bin_stats`` invariances and API behavior."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from binny.nz_tomo.bin_stats import (
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
    population_stats,
    shape_stats,
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
    if norm <= 0.0:
        raise ValueError("Cannot normalize: non-positive integral.")
    return nz / norm


def _make_nonmonotonic_z() -> np.ndarray:
    """Returns a nearly-monotonic grid with a tiny non-monotonic swap."""
    z = _toy_grid().copy()
    z[1000], z[1001] = z[1001], z[1000]
    return z


@pytest.mark.parametrize("scale", [0.1, 2.0, 7.3])
def test_invariance_amplitude_scaling_bin_moments(scale: float) -> None:
    """Tests that bin_moments is invariant under amplitude scaling."""
    z = _toy_grid()
    nz = _gaussian(z, 0.9, 0.2)

    m1 = bin_moments(z, nz)
    m2 = bin_moments(z, scale * nz)

    for k in (
        "mean",
        "median",
        "mode",
        "std",
        "skewness",
        "kurtosis",
        "iqr",
        "width_68",
    ):
        assert m2[k] == pytest.approx(m1[k], rel=0.0, abs=2e-3)


@pytest.mark.parametrize("scale", [0.2, 5.0])
def test_invariance_amplitude_scaling_bin_quantiles(scale: float) -> None:
    """Tests that bin_quantiles is invariant under amplitude scaling."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.25)
    qs = [0.16, 0.5, 0.84]

    q1 = bin_quantiles(z, nz, qs)
    q2 = bin_quantiles(z, scale * nz, qs)

    for q in qs:
        assert q2[q] == pytest.approx(q1[q], rel=0.0, abs=2e-3)


@pytest.mark.parametrize("scale", [0.3, 9.0])
def test_invariance_amplitude_scaling_in_range_fraction(scale: float) -> None:
    """Tests that in_range_fraction is invariant under amplitude scaling."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.25)

    f1 = in_range_fraction(z, nz, 0.7, 1.3)
    f2 = in_range_fraction(z, scale * nz, 0.7, 1.3)

    assert f2 == pytest.approx(f1, rel=0.0, abs=1e-12)


@pytest.mark.parametrize("scale", [0.5, 10.0])
def test_invariance_amplitude_scaling_peak_flags(scale: float) -> None:
    """Tests that peak_flags is invariant under amplitude scaling."""
    z = _toy_grid()
    nz = _gaussian(z, 1.0, 0.2) + 0.4 * _gaussian(z, 1.35, 0.06)

    p1 = peak_flags(z, nz, min_rel_height=0.1)
    p2 = peak_flags(z, scale * nz, min_rel_height=0.1)

    assert p2["mode"] == pytest.approx(p1["mode"], rel=0.0, abs=2e-3)
    assert p2["num_peaks"] == pytest.approx(p1["num_peaks"], rel=0.0, abs=1e-12)
    assert p2["second_peak_ratio"] == pytest.approx(p1["second_peak_ratio"], rel=0.0, abs=1e-12)


@pytest.mark.parametrize("shift", [0.05, 0.3])
def test_equivariance_translation_bin_moments(shift: float) -> None:
    """Tests that bin_moments is equivariant under z translation."""
    z = np.linspace(0.0, 2.0, 3001)
    nz = _gaussian(z, 0.9, 0.2)

    zs = z + shift
    nzs = nz.copy()

    m = bin_moments(z, nz)
    ms = bin_moments(zs, nzs)

    for k in ("mean", "median", "mode"):
        assert ms[k] == pytest.approx(m[k] + shift, rel=0.0, abs=2e-3)

    for k in ("std", "skewness", "kurtosis", "iqr", "width_68"):
        assert ms[k] == pytest.approx(m[k], rel=0.0, abs=2e-3)


@pytest.mark.parametrize("shift", [0.1, 0.27])
def test_equivariance_translation_bin_quantiles(shift: float) -> None:
    """Tests that bin_quantiles is equivariant under z translation."""
    z = np.linspace(0.0, 2.0, 3001)
    nz = _gaussian(z, 0.9, 0.2)

    zs = z + shift
    nzs = nz.copy()

    qs = [0.16, 0.5, 0.84]
    q = bin_quantiles(z, nz, qs)
    qs_out = bin_quantiles(zs, nzs, qs)

    for p in qs:
        assert qs_out[p] == pytest.approx(q[p] + shift, rel=0.0, abs=2e-3)


@pytest.mark.parametrize("shift", [0.12, 0.31])
def test_equivariance_translation_bin_centers(shift: float) -> None:
    """Tests that bin_centers is equivariant under z translation."""
    z = np.linspace(0.0, 2.0, 3001)
    b0 = _gaussian(z, 0.6, 0.1)
    b1 = _gaussian(z, 1.2, 0.2)
    bins = {0: b0, 1: b1}

    zs = z + shift
    bins_s = {0: b0.copy(), 1: b1.copy()}

    c = bin_centers(z, bins, method="median", decimal_places=None)
    cs = bin_centers(zs, bins_s, method="median", decimal_places=None)

    for k in c:
        assert cs[k] == pytest.approx(c[k] + shift, rel=0.0, abs=2e-3)


def test_shape_stats_allows_individually_normalized_bins() -> None:
    """Tests that shape_stats accepts individually normalized per-bin PDFs."""
    z = _toy_grid()
    p0 = _normalize_pdf(z, _gaussian(z, 0.6, 0.1))
    p1 = _normalize_pdf(z, _gaussian(z, 1.2, 0.2))
    bins = {0: p0, 1: p1}

    out = shape_stats(
        z,
        bins,
        center_method="median",
        decimal_places=None,
        quantiles=(0.16, 0.5, 0.84),
        min_rel_height=0.1,
    )

    assert set(out.keys()) >= {"centers", "peaks", "per_bin"}
    assert set(out["centers"].keys()) == {0, 1}
    assert "moments" in out["per_bin"][0]
    assert "quantiles" in out["per_bin"][1]


def test_shape_stats_returns_in_range_fraction_when_edges_given() -> None:
    """Tests that shape_stats returns in-range fractions when edges are provided."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    edges = [0.0, 1.0, 2.0]

    out = shape_stats(z, bins, bin_edges=edges)

    assert "in_range_fraction" in out
    assert set(out["in_range_fraction"].keys()) == {0, 1}
    assert 0.0 <= out["in_range_fraction"][0] <= 1.0
    assert 0.0 <= out["in_range_fraction"][1] <= 1.0


def test_population_stats_uses_metadata_not_bin_integrals() -> None:
    """Tests that population_stats fractions come from metadata, not bins."""
    z = _toy_grid()
    b0 = _gaussian(z, 0.6, 0.1)
    b1 = 10.0 * _gaussian(z, 1.2, 0.2)
    bins = {0: b0, 1: b1}

    meta: dict[str, Any] = {"frac_per_bin": {0: 0.8, 1: 0.2}}
    out = population_stats(bins, meta, normalize_frac=False, decimal_places=16)

    assert out["fractions"][0] == pytest.approx(0.8, rel=0.0, abs=1e-12)
    assert out["fractions"][1] == pytest.approx(0.2, rel=0.0, abs=1e-12)


def test_population_stats_normalizes_frac_when_requested() -> None:
    """Tests that population_stats renormalizes metadata fractions when requested."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    meta: dict[str, Any] = {"frac_per_bin": {0: 2.0, 1: 1.0}}
    out = population_stats(bins, meta, normalize_frac=True, decimal_places=16)

    assert out["fractions"][0] == pytest.approx(2.0 / 3.0, rel=0.0, abs=1e-12)
    assert out["fractions"][1] == pytest.approx(1.0 / 3.0, rel=0.0, abs=1e-12)
    assert sum(out["fractions"].values()) == pytest.approx(1.0, rel=0.0, abs=1e-12)


def test_population_stats_normalize_false_still_returns_normalized() -> None:
    """Tests that population_stats returns normalized fractions even if
    normalize_frac=False."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}

    meta: dict[str, Any] = {"frac_per_bin": {0: 0.2, 1: 0.2}}
    out = population_stats(bins, meta, normalize_frac=False, rtol=0.0, atol=0.0, decimal_places=16)

    assert sum(out["fractions"].values()) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert out["fractions"][0] == pytest.approx(0.5, rel=0.0, abs=1e-12)
    assert out["fractions"][1] == pytest.approx(0.5, rel=0.0, abs=1e-12)


def test_population_stats_density_allocation_matches_fractions() -> None:
    """Tests that population_stats allocates density per bin using fractions."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: 10.0 * _gaussian(z, 1.2, 0.2)}
    meta: dict[str, Any] = {"frac_per_bin": {0: 0.25, 1: 0.75}}

    out = population_stats(bins, meta, density_total=40.0, normalize_frac=False, decimal_places=16)

    assert out["density_total"] == pytest.approx(40.0, rel=0.0, abs=0.0)
    assert out["density_per_bin"][0] == pytest.approx(10.0, rel=0.0, abs=1e-12)
    assert out["density_per_bin"][1] == pytest.approx(30.0, rel=0.0, abs=1e-12)


def test_population_stats_count_allocation_matches_density_and_area() -> None:
    """Tests that population_stats count_per_bin matches density times area."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    meta: dict[str, Any] = {"frac_per_bin": {"0": 0.4, "1": 0.6}}

    out = population_stats(
        bins, meta, density_total=10.0, survey_area=100.0, normalize_frac=False, decimal_places=16
    )

    assert out["count_per_bin"][0] == pytest.approx(10.0 * 0.4 * 100.0, abs=1e-12)
    assert out["count_per_bin"][1] == pytest.approx(10.0 * 0.6 * 100.0, abs=1e-12)


def test_population_stats_raises_on_missing_bin_index_in_metadata() -> None:
    """Tests that population_stats raises if metadata is missing a bin index."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    meta: dict[str, Any] = {"frac_per_bin": {0: 1.0}}

    with pytest.raises(ValueError, match="missing bin index"):
        population_stats(bins, meta)


def test_population_stats_raises_if_survey_area_without_density_total() -> None:
    """Tests that population_stats raises if survey_area is set without
    density_total."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1)}
    meta: dict[str, Any] = {"frac_per_bin": {0: 1.0}}

    with pytest.raises(ValueError, match="survey_area requires density_total"):
        population_stats(bins, meta, survey_area=100.0, decimal_places=16)


def test_galaxy_fraction_per_bin_normalizes_and_casts_keys() -> None:
    """Tests that galaxy_fraction_per_bin normalizes and casts str keys to int."""
    meta: dict[str, Any] = {"frac_per_bin": {"0": 2.0, 1: 1.0}}
    out = galaxy_fraction_per_bin(meta)

    assert set(out.keys()) == {0, 1}
    assert sum(out.values()) == pytest.approx(1.0, rel=0.0, abs=1e-12)
    assert out[0] == pytest.approx(2.0 / 3.0, rel=0.0, abs=1e-12)


def test_galaxy_fraction_per_bin_raises_on_missing_frac_per_bin() -> None:
    """Tests that galaxy_fraction_per_bin raises if frac_per_bin is missing."""
    meta: dict[str, Any] = {}
    with pytest.raises(ValueError, match="must contain a mapping 'frac_per_bin'"):
        galaxy_fraction_per_bin(meta)


def test_galaxy_density_per_bin_raises_on_negative_total_density() -> None:
    """Tests that galaxy_density_per_bin raises when density_total is negative."""
    meta: dict[str, Any] = {"frac_per_bin": {0: 1.0}}
    with pytest.raises(ValueError, match="density_total must be non-negative"):
        galaxy_density_per_bin(meta, density_total=-1.0)


def test_galaxy_count_per_bin_raises_on_negative_density() -> None:
    """Tests that galaxy_count_per_bin raises when any density is negative."""
    with pytest.raises(ValueError, match="must be non-negative"):
        galaxy_count_per_bin({0: -0.1}, survey_area=10.0)


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


def test_peak_flags_per_bin_returns_sorted_int_keys() -> None:
    """Tests that peak_flags_per_bin returns int keys in sorted order."""
    z = _toy_grid()
    bins = {
        2: _gaussian(z, 1.4, 0.1),
        0: _gaussian(z, 0.6, 0.1),
        1: _gaussian(z, 1.0, 0.1),
    }

    out = peak_flags_per_bin(z, bins)

    assert list(out.keys()) == [0, 1, 2]
    assert all(isinstance(k, int) for k in out.keys())


def test_in_range_fraction_per_bin_mapping_missing_key_raises() -> None:
    """Tests that in_range_fraction_per_bin raises if mapping edges are missing."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    edges = {0: (0.3, 0.9)}

    with pytest.raises(ValueError, match="missing bin index"):
        in_range_fraction_per_bin(z, bins, edges)


def test_in_range_fraction_per_bin_sequence_too_short_raises() -> None:
    """Tests that in_range_fraction_per_bin raises if edge sequence is too short."""
    z = _toy_grid()
    bins = {0: _gaussian(z, 0.6, 0.1), 1: _gaussian(z, 1.2, 0.2)}
    edges = [0.0, 1.0]

    with pytest.raises(ValueError, match="sequence is too short"):
        in_range_fraction_per_bin(z, bins, edges)


def test_shape_stats_raises_on_empty_bins() -> None:
    """Tests that shape_stats raises on empty bins."""
    z = _toy_grid()
    with pytest.raises(ValueError, match="bins must not be empty"):
        shape_stats(z, {})


def test_population_stats_raises_on_empty_bins() -> None:
    """Tests that population_stats raises on empty bins."""
    with pytest.raises(ValueError, match="bins must not be empty"):
        population_stats({}, {"frac_per_bin": {}})
