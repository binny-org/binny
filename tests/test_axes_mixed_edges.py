"""Unit tests for binny.axes.mixed_edges module."""

import numpy as np
import pytest

from binny.axes.mixed_edges import _call_with, _get, mixed_edges


def test_get_uses_params_over_fallback():
    """Tests that _get returns the value from params when present."""
    params = {"x_min": 0.5}
    assert _get(0, params, "x_min", 0.0) == 0.5


def test_get_uses_fallback_when_missing_in_params():
    """Tests that _get returns the fallback value when key is not in params."""
    params = {}
    assert _get(0, params, "x_min", 0.0) == 0.0


def test_get_raises_when_missing_and_fallback_none():
    """Tests that _get raises ValueError when neither params
    nor fallback provide a value."""
    with pytest.raises(ValueError, match="requires 'x_min'"):
        _get(2, {}, "x_min", None)


def test_call_with_applies_casts_and_calls_func():
    """Tests that _call_with applies casts and forwards kwargs to the function."""

    def dummy_func(*, x_min: float, x_max: float, n_bins: int) -> np.ndarray:
        assert isinstance(x_min, float)
        assert isinstance(x_max, float)
        return np.linspace(x_min, x_max, n_bins + 1)

    params = {"x_min": "0", "x_max": "10"}
    g = {}
    edges = _call_with(
        0,
        params,
        4,
        g,
        func=dummy_func,
        required=("x_min", "x_max"),
        casts={"x_min": float, "x_max": float},
    )

    assert edges.shape == (5,)
    assert np.allclose(edges, np.array([0.0, 2.5, 5.0, 7.5, 10.0]))


def test_mixed_edges_concatenates_segments_without_duplicate_boundary():
    """Tests that mixed_edges concatenates edges
    and drops the first edge of subsequent segments."""
    segments = [
        {"method": "equidistant", "n_bins": 2, "params": {"x_min": 0.0, "x_max": 2.0}},
        {"method": "equidistant", "n_bins": 2, "params": {"x_min": 2.0, "x_max": 4.0}},
    ]
    edges = mixed_edges(segments)

    # Each segment yields 3 edges; second segment drops its first -> total 3 + 2 = 5
    assert edges.shape == (5,)
    assert np.allclose(edges, np.array([0.0, 1.0, 2.0, 3.0, 4.0]))


def test_mixed_edges_uses_globals_when_params_missing():
    """Tests that mixed_edges uses global x/weights
    when segment params do not provide them."""
    x = np.linspace(0.0, 10.0, 1001)
    w = np.ones_like(x)

    segments = [{"method": "equal_number", "n_bins": 5}]  # no params
    edges = mixed_edges(segments, x=x, weights=w)

    assert edges.shape == (6,)
    assert np.isclose(edges[0], x[0])
    assert np.isclose(edges[-1], x[-1])
    assert np.all(np.diff(edges) >= 0)


def test_mixed_edges_equal_number_then_equidistant():
    """Tests that mixed_edges works with equal_number
    followed by equidistant segment."""
    x = np.linspace(0.0, 10.0, 2001)
    w = 1.0 + x

    segments = [
        {"method": "equal_number",
         "n_bins": 3},
        {"method": "equidistant",
         "n_bins": 2,
         "params": {"x_min": 10.0, "x_max": 12.0}},
    ]
    edges = mixed_edges(segments, x=x, weights=w)

    assert edges.shape == (6,)
    assert np.isclose(edges[0], 0.0)
    assert np.isclose(edges[-1], 12.0)
    assert np.all(np.diff(edges) >= 0)

    assert np.count_nonzero(np.isclose(edges, 10.0)) == 1


def test_mixed_edges_equidistant_chi_matches_linear_chi_case():
    """Tests that mixed_edges with equidistant_chi matches equidistant
    for linear chi(z)."""
    z = np.linspace(0.0, 2.0, 2001)
    chi = 10.0 * z

    segments = [{"method": "equidistant_chi", "n_bins": 4}]
    edges = mixed_edges(segments, z=z, chi=chi)

    assert edges.shape == (5,)
    assert np.allclose(edges, np.linspace(0.0, 2.0, 5), rtol=0, atol=1e-12)


def test_mixed_edges_raises_for_unhandled_method():
    """Tests that mixed_edges raises for an unknown method name."""
    segments = [{"method": "not_a_method", "n_bins": 2}]
    with pytest.raises(ValueError, match="Unknown binning method"):
        mixed_edges(segments)


def test_mixed_edges_raises_when_required_global_missing():
    """Tests that mixed_edges raises when required global args are missing."""
    segments = [{"method": "equal_number", "n_bins": 3}]  # requires x, weights
    with pytest.raises(ValueError, match="requires 'x'|requires 'weights'"):
        mixed_edges(segments)


def test_mixed_edges_log_and_geometric_are_monotonic():
    """Tests that log and geometric segments produce monotonic increasing edges."""
    segments = [
        {"method": "log",
         "n_bins": 3,
         "params": {"x_min": 1.0, "x_max": 100.0}},
        {"method": "geometric",
         "n_bins": 2,
         "params": {"x_min": 100.0, "x_max": 400.0}},
    ]
    edges = mixed_edges(segments)

    assert edges.shape == (6,)
    assert np.all(np.diff(edges) > 0)
    assert np.isclose(edges[0], 1.0)
    assert np.isclose(edges[-1], 400.0)


def test_mixed_edges_runtimeerror_if_method_not_in_dispatch(monkeypatch):
    """Tests RuntimeError when a resolved method is not
     handled by mixed_edges dispatch."""
    import binny.axes.mixed_edges as mm

    monkeypatch.setattr(mm, "validate_mixed_segments", lambda *args, **kwargs: None)
    monkeypatch.setattr(mm, "resolve_binning_method", lambda _: "some_new_method")

    segments = [{"method": "anything", "n_bins": 2}]
    with pytest.raises(RuntimeError, match="Unhandled binning method"):
        mm.mixed_edges(segments)
