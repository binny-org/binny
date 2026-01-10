"""Unit tests for binny.utils.pairwise_metrics."""

import numpy as np
import pytest

from binny.utils.pairwise_metrics import (
    apply_unit,
    fill_symmetric,
    pair_cosine,
    pair_hellinger,
    pair_js,
    pair_min,
    pair_tv,
    segment_mass_probs,
)


def _simple_curves():
    z = np.array([0.0, 1.0, 2.0])
    p = {
        0: np.array([0.0, 1.0, 0.0]),
        1: np.array([0.0, 1.0, 0.0]),
        2: np.array([1.0, 0.0, 0.0]),
    }
    return z, p


def test_pair_min_returns_expected_overlap():
    """Tests that pair_min returns the trapezoid integral of pointwise minimum."""
    z, p = _simple_curves()
    f = pair_min(z, p)
    assert np.isclose(f(0, 1), 1.0)
    assert np.isclose(f(0, 2), 0.0)


def test_pair_cosine_returns_one_for_identical_curves():
    """Tests that pair_cosine returns 1 for identical nonzero curves."""
    z, p = _simple_curves()
    f = pair_cosine(z, p)
    assert np.isclose(f(0, 1), 1.0)


def test_pair_cosine_returns_zero_for_zero_norm_curve():
    """Tests that pair_cosine returns 0 when one curve has zero trapezoid L2 norm."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.zeros_like(z), 1: np.array([0.0, 1.0, 0.0])}
    f = pair_cosine(z, p)
    assert f(0, 1) == 0.0
    assert f(1, 0) == 0.0


def test_pair_js_zero_for_identical_probs():
    """Tests that pair_js returns 0 for identical probability vectors."""
    masses = {0: np.array([0.2, 0.8]), 1: np.array([0.2, 0.8])}
    f = pair_js(masses)
    assert np.isclose(f(0, 1), 0.0)


def test_pair_js_positive_for_different_probs():
    """Tests that pair_js returns a positive distance for different vectors."""
    masses = {0: np.array([1.0, 0.0]), 1: np.array([0.0, 1.0])}
    f = pair_js(masses)
    v = f(0, 1)
    assert 0.0 <= v <= 1.0
    assert v > 0.0


def test_pair_js_raises_for_non_probability_vector():
    """Tests that pair_js raises for inputs that are not probability vectors."""
    masses = {0: np.array([0.2, 0.2]), 1: np.array([0.5, 0.5])}
    f = pair_js(masses)
    with pytest.raises(ValueError, match=r"masses\[0\] must sum to 1"):
        f(0, 1)


def test_pair_js_raises_for_shape_mismatch():
    """Tests that pair_js raises when vectors have different shapes."""
    masses = {0: np.array([0.5, 0.5]), 1: np.array([1.0, 0.0, 0.0])}
    f = pair_js(masses)
    with pytest.raises(ValueError, match=r"must have the same shape"):
        f(0, 1)


def test_pair_hellinger_zero_for_identical_probs():
    """Tests that pair_hellinger returns 0 for identical probability vectors."""
    masses = {0: np.array([0.1, 0.9]), 1: np.array([0.1, 0.9])}
    f = pair_hellinger(masses)
    assert np.isclose(f(0, 1), 0.0)


def test_pair_tv_expected_value():
    """Tests that pair_tv returns half the L1 distance between probabilities."""
    masses = {0: np.array([0.0, 1.0]), 1: np.array([1.0, 0.0])}
    f = pair_tv(masses)
    assert np.isclose(f(0, 1), 1.0)


def test_fill_symmetric_fills_diagonal_and_mirrors():
    """Tests that fill_symmetric mirrors upper triangle to lower triangle."""

    def pv(i, j):
        return float(i + j)

    out = fill_symmetric([0, 1, 2], pv)
    assert out[0][0] == 0.0
    assert out[0][2] == out[2][0]
    assert out[1][2] == out[2][1]


def test_segment_mass_probs_returns_probability_vectors():
    """Tests that segment_mass_probs returns per-segment probabilities summing to 1."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.array([0.0, 2.0, 0.0])}
    out = segment_mass_probs(z, p)
    assert out[0].dtype == np.float64
    assert out[0].shape == (z.size - 1,)
    assert np.isclose(np.sum(out[0]), 1.0)


def test_segment_mass_probs_raises_for_non_positive_segment_mass():
    """Tests that segment_mass_probs raises when total segment mass is non-positive."""
    z = np.array([0.0, 1.0, 2.0])
    p = {0: np.zeros_like(z)}
    with pytest.raises(ValueError, match=r"non-positive mass on segments"):
        segment_mass_probs(z, p)


def test_apply_unit_fraction_returns_same_object():
    """Tests that apply_unit returns input unchanged when unit is fraction."""
    mat = {0: {0: 0.5, 1: 0.1}, 1: {0: 0.1, 1: 0.2}}
    out = apply_unit(mat, unit="fraction")
    assert out is mat


def test_apply_unit_percent_scales_values():
    """Tests that apply_unit converts values to percent when unit is percent."""
    mat = {0: {0: 0.5, 1: 0.1}}
    out = apply_unit(mat, unit="percent")
    assert np.isclose(out[0][0], 50.0)
    assert np.isclose(out[0][1], 10.0)


def test_apply_unit_rejects_unknown_unit():
    """Tests that apply_unit rejects unknown unit strings."""
    with pytest.raises(ValueError, match=r'unit must be "fraction" or "percent"'):
        apply_unit({0: {0: 1.0}}, unit="nope")  # type: ignore[arg-type]
