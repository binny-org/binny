"""Unit tests for binny.correlations.metrics."""

from __future__ import annotations

import numpy as np
import pytest

from binny.correlations import metrics as m


def _tri(z: np.ndarray, center: float, width: float) -> np.ndarray:
    """Simple nonnegative triangular bump for deterministic overlap tests."""
    x = np.abs(z - center)
    y = np.maximum(0.0, 1.0 - x / float(width))
    return y


def test_prepare_indexed_curves_and_norms_raises_on_empty_curves():
    """Tests that _prepare_indexed_curves_and_norms raises on empty curves."""
    z = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match=r"must contain at least one mapping"):
        m._prepare_indexed_curves_and_norms(z=z, curves=[])


def test_prepare_indexed_curves_and_norms_raises_on_shape_mismatch():
    """Tests that _prepare_indexed_curves_and_norms raises on shape mismatch."""
    z = np.linspace(0.0, 1.0, 5)
    curves = [{0: np.ones(4)}]
    with pytest.raises(ValueError, match=r"must have same shape as z"):
        m._prepare_indexed_curves_and_norms(z=z, curves=curves, label="curves")


def test_prepare_indexed_curves_and_norms_casts_indices_to_int_and_computes_norms():
    """Tests that _prepare_indexed_curves_and_norms casts keys and computes norms."""
    z = np.linspace(0.0, 1.0, 5)
    c0 = np.ones_like(z)
    c1 = 2.0 * np.ones_like(z)

    zz, arrs, norms = m._prepare_indexed_curves_and_norms(
        z=z,
        curves=[{"0": c0, 1.0: c1}],  # mixed key types
    )

    assert zz.dtype == float
    assert set(arrs[0].keys()) == {0, 1}

    n0 = float(np.trapezoid(c0, z))
    n1 = float(np.trapezoid(c1, z))
    assert norms[0][0] == pytest.approx(n0)
    assert norms[0][1] == pytest.approx(n1)


def test_metric_from_curves_n_raises_on_wrong_arity():
    """Tests that _metric_from_curves_n raises on wrong number of indices."""
    z = np.linspace(0.0, 1.0, 5)
    c = np.ones_like(z)
    arrs = [{0: c}, {0: c}]

    def kernel(a, b):
        return float(np.trapezoid(a + b, z))

    f = m._metric_from_curves_n(arrs, kernel)
    with pytest.raises(TypeError, match=r"Expected 2 indices, got 1"):
        _ = f(0)


def test_metric_from_curves_n_raises_on_missing_index():
    """Tests that _metric_from_curves_n raises on missing index in a slot."""
    z = np.linspace(0.0, 1.0, 5)
    c = np.ones_like(z)
    arrs = [{0: c}, {0: c}]

    def kernel(a, b):
        _, _ = a, b
        return 0.0

    f = m._metric_from_curves_n(arrs, kernel)
    with pytest.raises(KeyError, match=r"Missing curves for slot 1 index 7"):
        _ = f(0, 7)


def test_metric_from_curves_n_delegates_to_kernel_with_selected_curves():
    """Tests that _metric_from_curves_n selects curves and calls kernel."""
    z = np.linspace(0.0, 1.0, 5)
    c0 = np.ones_like(z)
    c1 = 2.0 * np.ones_like(z)

    arrs = [{0: c0}, {5: c1}]
    called = {"ok": False}

    def kernel(a, b):
        called["ok"] = True
        assert np.allclose(a, c0)
        assert np.allclose(b, c1)
        return 3.0

    f = m._metric_from_curves_n(arrs, kernel)
    out = f(0, 5)

    assert called["ok"] is True
    assert out == pytest.approx(3.0)


def test_metric_min_overlap_fraction_basic_pairwise():
    """Tests that metric_min_overlap_fraction returns expected normalized overlap."""
    z = np.linspace(0.0, 1.0, 501)

    # Two overlapping triangles.
    a = _tri(z, center=0.4, width=0.2)
    b = _tri(z, center=0.5, width=0.2)

    f = m.metric_min_overlap_fraction(z=z, curves=[{0: a}, {0: b}])
    out = f(0, 0)

    # Reference: integral(min) / (int(a)*int(b))
    num = float(np.trapezoid(np.minimum(a, b), z))
    den = float(np.trapezoid(a, z)) * float(np.trapezoid(b, z))
    assert out == pytest.approx(num / den)


def test_metric_min_overlap_fraction_returns_zero_if_any_norm_is_zero():
    """Tests that metric_min_overlap_fraction returns 0 when any curve norm is zero."""
    z = np.linspace(0.0, 1.0, 11)
    a = np.zeros_like(z)
    b = np.ones_like(z)

    f = m.metric_min_overlap_fraction(z=z, curves=[{0: a}, {0: b}])
    assert f(0, 0) == 0.0


def test_metric_min_overlap_fraction_raises_on_missing_index():
    """Tests that metric_min_overlap_fraction raises on missing index."""
    z = np.linspace(0.0, 1.0, 11)
    a = np.ones_like(z)
    b = np.ones_like(z)

    f = m.metric_min_overlap_fraction(z=z, curves=[{0: a}, {0: b}])
    with pytest.raises(KeyError):
        _ = f(0, 7)


def test_metric_min_overlap_fraction_raises_on_wrong_arity():
    """Tests that metric_min_overlap_fraction raises on wrong arity."""
    z = np.linspace(0.0, 1.0, 11)
    a = np.ones_like(z)
    b = np.ones_like(z)

    f = m.metric_min_overlap_fraction(z=z, curves=[{0: a}, {0: b}])
    with pytest.raises(TypeError, match=r"Expected 2 indices, got 1"):
        _ = f(0)


def test_metric_overlap_coefficient_basic_pairwise():
    """Tests that metric_overlap_coefficient returns expected overlap coefficient."""
    z = np.linspace(0.0, 1.0, 501)
    a = _tri(z, center=0.4, width=0.2)
    b = _tri(z, center=0.5, width=0.2)

    f = m.metric_overlap_coefficient(z=z, curves=[{0: a}, {0: b}])
    out = f(0, 0)

    num = float(np.trapezoid(np.minimum(a, b), z))
    denom = min(float(np.trapezoid(a, z)), float(np.trapezoid(b, z)))
    assert out == pytest.approx(num / denom)


def test_metric_overlap_coefficient_returns_zero_if_min_norm_is_zero():
    """Tests that metric_overlap_coefficient returns 0 when min norm is 0."""
    z = np.linspace(0.0, 1.0, 11)
    a = np.zeros_like(z)
    b = np.ones_like(z)

    f = m.metric_overlap_coefficient(z=z, curves=[{0: a}, {0: b}])
    assert f(0, 0) == 0.0


def test_metric_overlap_coefficient_raises_on_wrong_arity():
    """Tests that metric_overlap_coefficient raises on wrong arity."""
    z = np.linspace(0.0, 1.0, 11)
    a = np.ones_like(z)
    b = np.ones_like(z)

    f = m.metric_overlap_coefficient(z=z, curves=[{0: a}, {0: b}])
    with pytest.raises(IndexError):
        _ = f(0, 0, 0)


def test_metric_from_curves_builds_metric_and_casts_indices_to_int():
    """Tests that metric_from_curves builds a metric and casts keys to int."""
    z = np.linspace(0.0, 1.0, 5)
    a = np.ones_like(z)
    b = 2.0 * np.ones_like(z)

    def kernel(x, y):
        return float(np.sum(x - y))

    f = m.metric_from_curves(curves=[{"0": a}, {1.0: b}], kernel=kernel)
    # keys should be int-cast: "0"->0, 1.0->1
    out = f(0, 1)
    assert out == pytest.approx(float(np.sum(a - b)))


def test_metric_from_curves_raises_on_missing_index():
    """Tests that metric_from_curves raises on missing index."""
    z = np.linspace(0.0, 1.0, 5)
    a = np.ones_like(z)
    b = np.ones_like(z)

    def kernel(x, y):
        _, _ = x, y
        return 0.0

    f = m.metric_from_curves(curves=[{0: a}, {0: b}], kernel=kernel)
    with pytest.raises(KeyError, match=r"Missing curves for slot 1 index 2"):
        _ = f(0, 2)


def test_metric_from_curves_raises_on_wrong_arity():
    """Tests that metric_from_curves raises on wrong arity."""
    z = np.linspace(0.0, 1.0, 5)
    a = np.ones_like(z)
    b = np.ones_like(z)

    def kernel(x, y):
        _, _ = x, y
        return 0.0

    f = m.metric_from_curves(curves=[{0: a}, {0: b}], kernel=kernel)
    with pytest.raises(TypeError, match=r"Expected 2 indices, got 1"):
        _ = f(0)
