"""Unit tests for binny.correlations.scores."""

from __future__ import annotations

import numpy as np
import pytest

from binny.correlations import scores as sc


def _tri(z: np.ndarray, center: float, width: float) -> np.ndarray:
    """Simple nonnegative triangular bump for deterministic tests."""
    x = np.abs(z - center)
    return np.maximum(0.0, 1.0 - x / float(width))


def test_score_peak_location_returns_peak_grid_coordinate():
    """Tests that score_peak_location returns the argmax z location."""
    z = np.linspace(0.0, 1.0, 11)
    c0 = _tri(z, center=0.2, width=0.2)
    c1 = _tri(z, center=0.7, width=0.2)

    out = sc.score_peak_location(z=z, curves={0: c0, 1: c1})

    assert out[0] == pytest.approx(z[int(np.argmax(c0))])
    assert out[1] == pytest.approx(z[int(np.argmax(c1))])


def test_score_peak_location_raises_on_shape_mismatch():
    """Tests that score_peak_location raises when curve shape mismatches z."""
    z = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError, match=r"must have same shape as z"):
        _ = sc.score_peak_location(z=z, curves={0: np.ones(10)})


def test_score_mean_location_matches_reference_integral():
    """Tests that score_mean_location matches trapezoid reference."""
    z = np.linspace(0.0, 1.0, 501)
    c = _tri(z, center=0.6, width=0.3)

    out = sc.score_mean_location(z=z, curves={7: c})

    num = float(np.trapezoid(z * c, z))
    den = float(np.trapezoid(c, z))
    assert out[7] == pytest.approx(num / den)


def test_score_mean_location_returns_nan_on_zero_norm():
    """Tests that score_mean_location returns nan for zero-area curves."""
    z = np.linspace(0.0, 1.0, 11)
    out = sc.score_mean_location(z=z, curves={0: np.zeros_like(z)})
    assert np.isnan(out[0])


def test_score_mean_location_raises_on_shape_mismatch():
    """Tests that score_mean_location raises when curve shape mismatches z."""
    z = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError, match=r"must have same shape as z"):
        _ = sc.score_mean_location(z=z, curves={0: np.ones(12)})


def test_score_median_location_returns_midpoint_for_symmetric_curve():
    """Tests that score_median_location returns center for symmetric curve."""
    z = np.linspace(0.0, 1.0, 2001)
    c = _tri(z, center=0.5, width=0.4)

    out = sc.score_median_location(z=z, curves={3: c})

    assert out[3] == pytest.approx(0.5, abs=1e-3)


def test_score_median_location_returns_nan_on_zero_norm():
    """Tests that score_median_location returns nan for zero-area curves."""
    z = np.linspace(0.0, 1.0, 11)
    out = sc.score_median_location(z=z, curves={0: np.zeros_like(z)})
    assert np.isnan(out[0])


def test_score_median_location_raises_when_z_has_fewer_than_two_points():
    """Tests that score_median_location raises when z has < 2 points."""
    z = np.array([0.0])
    c = np.array([1.0])
    with pytest.raises(ValueError, match=r"at least 2 points"):
        _ = sc.score_median_location(z=z, curves={0: c})


def test_score_median_location_raises_on_shape_mismatch():
    """Tests that score_median_location raises when curve shape mismatches z."""
    z = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError, match=r"must have same shape as z"):
        _ = sc.score_median_location(z=z, curves={0: np.ones(10)})


def test_score_credible_width_raises_on_invalid_mass():
    """Tests that score_credible_width raises for mass outside (0, 1)."""
    z = np.linspace(0.0, 1.0, 11)
    c = np.ones_like(z)

    with pytest.raises(ValueError, match=r"mass must be in"):
        _ = sc.score_credible_width(z=z, curves={0: c}, mass=0.0)

    with pytest.raises(ValueError, match=r"mass must be in"):
        _ = sc.score_credible_width(z=z, curves={0: c}, mass=1.0)


def test_score_credible_width_returns_nan_on_zero_norm():
    """Tests that score_credible_width returns nan for zero-area curves."""
    z = np.linspace(0.0, 1.0, 11)
    out = sc.score_credible_width(z=z, curves={0: np.zeros_like(z)}, mass=0.68)
    assert np.isnan(out[0])


def test_score_credible_width_matches_reference_quantiles():
    """Tests that score_credible_width matches quantiles from the CDF."""
    z = np.linspace(0.0, 1.0, 2001)
    c = _tri(z, center=0.5, width=0.4)
    mass = 0.68

    out = sc.score_credible_width(z=z, curves={2: c}, mass=mass)

    dz = np.diff(z)
    area = 0.5 * (c[:-1] + c[1:]) * dz
    total = float(np.sum(area))
    cdf = np.concatenate([[0.0], np.cumsum(area) / total])

    lo_q = (1.0 - mass) / 2.0
    hi_q = 1.0 - lo_q
    z_lo = float(np.interp(lo_q, cdf, z))
    z_hi = float(np.interp(hi_q, cdf, z))

    assert out[2] == pytest.approx(z_hi - z_lo)


def test_score_credible_width_raises_on_shape_mismatch():
    """Tests that score_credible_width raises when curve shape mismatches z."""
    z = np.linspace(0.0, 1.0, 11)
    with pytest.raises(ValueError, match=r"must have same shape as z"):
        _ = sc.score_credible_width(z=z, curves={0: np.ones(10)}, mass=0.68)


def test_all_score_functions_cast_keys_to_int():
    """Tests that score functions cast mapping keys to int."""
    z = np.linspace(0.0, 1.0, 11)
    c = _tri(z, center=0.5, width=0.4)
    curves = {"7": c}

    out_peak = sc.score_peak_location(z=z, curves=curves)
    out_mean = sc.score_mean_location(z=z, curves=curves)
    out_med = sc.score_median_location(z=z, curves=curves)
    out_w = sc.score_credible_width(z=z, curves=curves, mass=0.68)

    assert 7 in out_peak
    assert 7 in out_mean
    assert 7 in out_med
    assert 7 in out_w
