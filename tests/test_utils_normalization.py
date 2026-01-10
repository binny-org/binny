"""Unit tests for binny.utils.normalization."""

import numpy as np
import pytest

from binny.utils.normalization import (
    cdf_from_curve,
    curve_norm_mode,
    integrate_bins,
    mass_per_segment,
    normalize_1d,
    normalize_edges,
    normalize_or_check_curves,
    prepare_metric_inputs,
    trapz_weights,
    weighted_quantile_from_cdf,
)


def test_normalize_1d_trapezoid_returns_unit_integral():
    """Tests that normalize_1d returns y with unit trapezoid integral."""
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    out = normalize_1d(x, y, method="trapezoid")
    assert out.dtype == np.float64
    assert np.isclose(np.trapezoid(out, x=x), 1.0)


def test_normalize_1d_simpson_returns_unit_integral():
    """Tests that normalize_1d returns y with unit simpson integral."""
    x = np.linspace(0.0, 1.0, 11)
    y = x**2 + 1.0
    out = normalize_1d(x, y, method="simpson")
    assert out.dtype == np.float64
    assert np.isclose(np.trapezoid(out, x=x), 1.0, rtol=2e-3, atol=1e-6)


def test_normalize_1d_rejects_unknown_method():
    """Tests that normalize_1d rejects unknown integration methods."""
    with pytest.raises(ValueError, match=r"method must be"):
        normalize_1d([0, 1], [1, 1], method="nope")  # type: ignore[arg-type]


def test_normalize_1d_rejects_non_positive_norm():
    """Tests that normalize_1d rejects non-positive normalization factor."""
    with pytest.raises(ValueError, match=r"Normalization factor must be positive"):
        normalize_1d([0, 1], [0, 0], method="trapezoid")


def test_integrate_bins_happy_path_returns_integrals():
    """Tests that integrate_bins returns trapezoid integrals per bin."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {0: np.array([0.0, 1.0, 0.0]), 1: np.array([1.0, 1.0, 1.0])}
    out = integrate_bins(z, bins)
    assert set(out) == {0, 1}
    assert np.isclose(out[0], 1.0)
    assert np.isclose(out[1], 2.0)


def test_integrate_bins_rejects_empty():
    """Tests that integrate_bins rejects empty bins mapping."""
    with pytest.raises(ValueError, match=r"bins must not be empty"):
        integrate_bins([0, 1], {})


def test_integrate_bins_annotates_invalid_bin_index():
    """Tests that integrate_bins annotates the offending bin index in errors."""
    z = np.array([0.0, 1.0, 2.0])
    bins = {7: np.array([1.0, np.nan, 1.0])}
    with pytest.raises(ValueError, match=r"Invalid bin 7:"):
        integrate_bins(z, bins)


def test_cdf_from_curve_returns_cdf_and_norm():
    """Tests that cdf_from_curve returns cdf starting at 0 and correct norm."""
    z = np.array([0.0, 1.0, 2.0])
    nz = np.array([0.0, 1.0, 0.0])
    cdf, norm = cdf_from_curve(z, nz)
    assert cdf.dtype == np.float64
    assert cdf[0] == 0.0
    assert np.isclose(norm, 1.0)
    assert np.isclose(cdf[-1], norm)


def test_cdf_from_curve_rejects_negative_values():
    """Tests that cdf_from_curve rejects curves with negative values."""
    z = np.array([0.0, 1.0, 2.0])
    nz = np.array([0.0, -1.0, 0.0])
    with pytest.raises(ValueError, match=r"must be nonnegative"):
        cdf_from_curve(z, nz)


def test_cdf_from_curve_rejects_non_positive_total_mass():
    """Tests that cdf_from_curve rejects curves with non-positive total mass."""
    z = np.array([0.0, 1.0, 2.0])
    nz = np.zeros_like(z)
    with pytest.raises(ValueError, match=r"Total weight must be positive"):
        cdf_from_curve(z, nz)


def test_weighted_quantile_from_cdf_endpoints():
    """Tests that weighted_quantile_from_cdf returns endpoints at q=0 and q=1."""
    z = np.array([0.0, 1.0, 2.0])
    cdf = np.array([0.0, 0.5, 1.0])
    assert weighted_quantile_from_cdf(z, cdf, norm=1.0, q=0.0) == 0.0
    assert weighted_quantile_from_cdf(z, cdf, norm=1.0, q=1.0) == 2.0


def test_weighted_quantile_from_cdf_linear_interp():
    """Tests that weighted_quantile_from_cdf interpolates between nodes."""
    z = np.array([0.0, 1.0, 2.0])
    cdf = np.array([0.0, 0.5, 1.0])
    q = weighted_quantile_from_cdf(z, cdf, norm=1.0, q=0.25)
    assert np.isclose(q, 0.5)


def test_weighted_quantile_from_cdf_rejects_bad_inputs():
    """Tests that weighted_quantile_from_cdf rejects invalid q and norm."""
    z = np.array([0.0, 1.0])
    cdf = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match=r"q must be between 0 and 1"):
        weighted_quantile_from_cdf(z, cdf, norm=1.0, q=1.1)
    with pytest.raises(ValueError, match=r"norm must be positive"):
        weighted_quantile_from_cdf(z, cdf, norm=0.0, q=0.5)


def test_trapz_weights_matches_np_trapezoid():
    """Tests that trapz_weights reproduces np.trapezoid via dot product."""
    z = np.array([0.0, 1.0, 3.0])
    f = np.array([2.0, 4.0, 6.0])
    w = trapz_weights(z)
    assert np.isclose(np.sum(w * f), np.trapezoid(f, x=z))


def test_trapz_weights_small_grids_return_zeros():
    """Tests that trapz_weights returns zeros for grids with fewer than 2 points."""
    z = np.array([1.0])
    w = trapz_weights(z)
    assert w.dtype == np.float64
    assert np.allclose(w, 0.0)


def test_trapz_weights_rejects_not_increasing():
    """Tests that trapz_weights rejects non-increasing grids."""
    with pytest.raises(ValueError, match=r"strictly increasing"):
        trapz_weights([0.0, 0.0, 1.0])


def test_mass_per_segment_happy_path():
    """Tests that mass_per_segment returns segment masses with correct length."""
    z = np.array([0.0, 1.0, 3.0])
    p = np.array([1.0, 1.0, 1.0])
    m = mass_per_segment(z, p)
    assert m.dtype == np.float64
    assert m.shape == (2,)
    np.testing.assert_allclose(m, [1.0, 2.0])


def test_mass_per_segment_rejects_shape_mismatch():
    """Tests that mass_per_segment rejects mismatched input lengths."""
    with pytest.raises(ValueError, match=r"same length"):
        mass_per_segment([0, 1, 2], [1, 1])


def test_mass_per_segment_rejects_not_increasing():
    """Tests that mass_per_segment rejects non-increasing z grid."""
    with pytest.raises(ValueError, match=r"strictly increasing"):
        mass_per_segment([0, 1, 1], [1, 1, 1])


def test_normalize_or_check_curves_normalizes_when_requested():
    """Tests that normalize_or_check_curves normalizes curves when normalize=True."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 2.0, 0.0])}
    out = normalize_or_check_curves(z, p, normalize=True, check_normalized=False)
    assert np.isclose(np.trapezoid(out[0], x=z), 1.0)


def test_normalize_or_check_curves_check_raises_if_not_normalized():
    """Tests that normalize_or_check_curves raises if check_normalized fails."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 2.0, 0.0])}
    with pytest.raises(ValueError, match=r"does not appear normalized"):
        normalize_or_check_curves(z, p, normalize=False, check_normalized=True)


def test_normalize_or_check_curves_warns_if_already_normalized():
    """Tests that normalize_or_check_curves warns when renormalizing unit curves."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 1.0, 0.0])}
    with pytest.warns(UserWarning, match=r"appears already normalized"):
        normalize_or_check_curves(
            z,
            p,
            normalize=True,
            check_normalized=False,
            warn_if_already_normalized=True,
        )


def test_normalize_edges_from_mapping_happy_path():
    """Tests that normalize_edges returns edges for requested indices from mapping."""
    edges = {0: (0.0, 1.0), 2: (2.0, 3.0)}
    out = normalize_edges([0, 2], edges)
    assert out == {0: (0.0, 1.0), 2: (2.0, 3.0)}


def test_normalize_edges_mapping_missing_index_raises():
    """Tests that normalize_edges raises if mapping misses a requested index."""
    edges = {0: (0.0, 1.0)}
    with pytest.raises(ValueError, match=r"missing bin index"):
        normalize_edges([0, 1], edges)


def test_normalize_edges_from_array_happy_path():
    """Tests that normalize_edges converts a strictly increasing edge array."""
    out = normalize_edges([0, 1], [0.0, 1.0, 3.0])
    assert out == {0: (0.0, 1.0), 1: (1.0, 3.0)}


def test_normalize_edges_array_rejects_out_of_range_bin():
    """Tests that normalize_edges rejects bin indices out of range for edge array."""
    with pytest.raises(ValueError, match=r"out of range"):
        normalize_edges([2], [0.0, 1.0, 2.0])


def test_prepare_metric_inputs_curves_mode_none():
    """Tests that prepare_metric_inputs returns validated curves in curves mode."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 2.0, 0.0])}
    z_out, out = prepare_metric_inputs(z, p, mode="curves", curve_norm="none")
    assert z_out.dtype == np.float64
    assert out[0].shape == z.shape


def test_prepare_metric_inputs_segments_prob_sums_to_one():
    """Tests that prepare_metric_inputs returns segment probabilities summing to 1."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 2.0, 0.0])}
    _, out = prepare_metric_inputs(
        z,
        p,
        mode="segments_prob",
        curve_norm="normalize",
    )
    assert out[0].shape == (z.size - 1,)
    assert np.isclose(np.sum(out[0]), 1.0)


def test_prepare_metric_inputs_check_mode_raises_when_not_normalized():
    """Tests that prepare_metric_inputs raises if curve_norm='check' fails."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 2.0, 0.0])}
    with pytest.raises(ValueError, match=r"does not appear normalized"):
        prepare_metric_inputs(z, p, mode="curves", curve_norm="check")


def test_prepare_metric_inputs_rejects_unknown_mode():
    """Tests that prepare_metric_inputs rejects unknown modes."""
    z = np.array([0.0, 1.0])
    p = {0: np.array([1.0, 1.0])}
    with pytest.raises(ValueError, match=r"mode must be"):
        prepare_metric_inputs(z, p, mode="nope")  # type: ignore[arg-type]


def test_curve_norm_mode_returns_expected_values():
    """Tests that curve_norm_mode returns the correct normalization mode."""
    assert (
        curve_norm_mode(
            required=False,
            assume_normalized=True,
            normalize_if_needed=True,
        )
        == "none"
    )
    assert (
        curve_norm_mode(
            required=True,
            assume_normalized=False,
            normalize_if_needed=True,
        )
        == "none"
    )
    assert (
        curve_norm_mode(
            required=True,
            assume_normalized=True,
            normalize_if_needed=True,
        )
        == "normalize"
    )
    assert (
        curve_norm_mode(
            required=True,
            assume_normalized=True,
            normalize_if_needed=False,
        )
        == "check"
    )
