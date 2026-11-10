"""Tests for internal helpers in ``binny.tomography.comparison``."""

from __future__ import annotations

import numpy as np
import pytest

from binny.tomography.comparison import (
    _curve_norm_mode,
    _pair_cosine_between,
    _pair_hellinger_between,
    _pair_js_between,
    _pair_min_between,
    _pair_tv_between,
    _prepare_curve_inputs,
    _prepare_mass_inputs,
    _rectangular_from_pair_value,
    _validate_method,
)


def _example_z() -> np.ndarray:
    """Tests that helper z grid creation is deterministic."""
    return np.linspace(0.0, 1.0, 5)


def _example_bins() -> dict[int, np.ndarray]:
    """Tests that helper bins are deterministic and nontrivial."""
    return {
        0: np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=float),
        1: np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float),
        2: np.array([2.0, 1.0, 0.0, 1.0, 2.0], dtype=float),
    }


def _normalized_curve(values: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Tests that helper curve normalization returns unit-integral curves."""
    area = float(np.trapezoid(values, x=z))
    return np.asarray(values, dtype=float) / area


def test_validate_method_accepts_supported_methods():
    """Tests that _validate_method accepts supported methods and lowercases them."""
    assert _validate_method("min") == "min"
    assert _validate_method("COSINE") == "cosine"
    assert _validate_method("Js") == "js"
    assert _validate_method("hellinger") == "hellinger"
    assert _validate_method("tv") == "tv"


def test_validate_method_raises_on_unknown():
    """Tests that _validate_method raises on unsupported methods."""
    with pytest.raises(
        ValueError, match=r'method must be "min", "cosine", "js", "hellinger", or "tv"'
    ):
        _validate_method("pearson")


def test_curve_norm_mode_returns_normalize_only_when_required_and_requested():
    """Tests that _curve_norm_mode returns normalize only when both flags are True."""
    assert _curve_norm_mode(normalize=True, requires_norm=True) == "normalize"
    assert _curve_norm_mode(normalize=False, requires_norm=True) == "none"
    assert _curve_norm_mode(normalize=True, requires_norm=False) == "none"
    assert _curve_norm_mode(normalize=False, requires_norm=False) == "none"


def test_prepare_curve_inputs_returns_sorted_indices_and_curves():
    """Tests that _prepare_curve_inputs returns sorted indices and prepared curves."""
    z = _example_z()
    bins = {2: _example_bins()[2], 0: _example_bins()[0]}

    z_m, bin_indices, curves = _prepare_curve_inputs(
        z,
        bins,
        normalize=False,
        requires_norm=False,
        rtol=1e-3,
        atol=1e-6,
    )

    assert np.allclose(z_m, z)
    assert bin_indices == [0, 2]
    assert set(curves.keys()) == {0, 2}
    assert np.allclose(curves[0], bins[0])
    assert np.allclose(curves[2], bins[2])


def test_prepare_mass_inputs_returns_sorted_indices_and_probabilities():
    """Tests that _prepare_mass_inputs returns sorted indices and segment-mass vectors."""
    z = _example_z()
    bins = _example_bins()

    bin_indices, masses = _prepare_mass_inputs(
        z,
        bins,
        normalize=False,
        requires_norm=True,
        rtol=1e-3,
        atol=1e-6,
    )

    assert bin_indices == [0, 1, 2]
    assert set(masses.keys()) == {0, 1, 2}
    for key in bin_indices:
        assert masses[key].ndim == 1
        assert np.isclose(np.sum(masses[key]), 1.0)


def test_rectangular_from_pair_value_builds_expected_nested_mapping():
    """Tests that _rectangular_from_pair_value builds a rectangular nested mapping."""
    rows = [0, 2]
    cols = [1, 3]

    out = _rectangular_from_pair_value(rows, cols, lambda i, j: 10 * i + j)

    assert out == {
        0: {1: 1.0, 3: 3.0},
        2: {1: 21.0, 3: 23.0},
    }


def test_pair_min_between_returns_expected_overlap():
    """Tests that _pair_min_between returns the pointwise-minimum overlap integral."""
    z = _example_z()
    curves_a = {0: np.array([0.0, 1.0, 1.0, 0.0, 0.0])}
    curves_b = {5: np.array([0.0, 0.0, 1.0, 1.0, 0.0])}

    evaluator = _pair_min_between(z, curves_a, curves_b)
    expected = float(np.trapezoid(np.minimum(curves_a[0], curves_b[5]), x=z))

    assert np.isclose(evaluator(0, 5), expected)


def test_pair_cosine_between_returns_one_for_identical_curves():
    """Tests that _pair_cosine_between returns one for identical curves."""
    z = _example_z()
    curve = np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=float)
    curves_a = {0: curve}
    curves_b = {7: curve.copy()}

    evaluator = _pair_cosine_between(z, curves_a, curves_b)
    assert np.isclose(evaluator(0, 7), 1.0)


def test_pair_cosine_between_returns_zero_when_denominator_is_zero():
    """Tests that _pair_cosine_between returns zero for zero-norm curves."""
    z = _example_z()
    curves_a = {0: np.zeros_like(z)}
    curves_b = {1: np.ones_like(z)}

    evaluator = _pair_cosine_between(z, curves_a, curves_b)
    assert evaluator(0, 1) == 0.0


def test_pair_js_between_returns_zero_for_identical_probabilities():
    """Tests that _pair_js_between returns zero for identical probability vectors."""
    p = np.array([0.2, 0.3, 0.5], dtype=float)
    masses_a = {0: p}
    masses_b = {1: p.copy()}

    evaluator = _pair_js_between(masses_a, masses_b)
    assert np.isclose(evaluator(0, 1), 0.0)


def test_pair_hellinger_between_returns_zero_for_identical_probabilities():
    """Tests that _pair_hellinger_between returns zero for identical probability vectors."""
    p = np.array([0.2, 0.3, 0.5], dtype=float)
    masses_a = {0: p}
    masses_b = {1: p.copy()}

    evaluator = _pair_hellinger_between(masses_a, masses_b)
    assert np.isclose(evaluator(0, 1), 0.0)


def test_pair_tv_between_returns_zero_for_identical_probabilities():
    """Tests that _pair_tv_between returns zero for identical probability vectors."""
    p = np.array([0.2, 0.3, 0.5], dtype=float)
    masses_a = {0: p}
    masses_b = {1: p.copy()}

    evaluator = _pair_tv_between(masses_a, masses_b)
    assert np.isclose(evaluator(0, 1), 0.0)
