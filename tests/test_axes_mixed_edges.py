"""Unit tests for ``binny.axes.mixed_edges`` module."""

import numpy as np
import pytest

import binny.axes.mixed_edges as memod
from binny.axes.mixed_edges import (
    _call_with,
    _get,
    _validate_segment_edges,
    mixed_edges,
)


def test_get_uses_params_over_fallback():
    """Tests that _get returns the value from params when present."""
    params = {"x_min": 0.5}
    assert _get(0, params, "x_min", 0.0) == 0.5


def test_get_uses_fallback_when_missing_in_params():
    """Tests that _get returns the fallback when params lack the key."""
    params = {}
    assert _get(0, params, "x_min", 0.0) == 0.0


def test_get_raises_when_missing_and_fallback_none():
    """Tests that _get raises ValueError when both params and fallback lack
    the key."""
    with pytest.raises(ValueError, match=r"Segment 2 requires 'x_min'"):
        _get(2, {}, "x_min", None)


def test_call_with_applies_casts_and_calls_func():
    """Tests that _call_with casts params and calls the function correctly."""

    def dummy_func(*, x_min: float, x_max: float, n_bins: int) -> np.ndarray:
        """Dummy function that returns a linspace."""
        assert isinstance(x_min, float)
        assert isinstance(x_max, float)
        return np.linspace(x_min, x_max, n_bins + 1)

    params = {"x_min": "0", "x_max": "10"}
    edges = _call_with(
        0,
        params,
        4,
        {},
        func=dummy_func,
        required=("x_min", "x_max"),
        casts={"x_min": float, "x_max": float},
    )

    assert edges.shape == (5,)
    assert np.allclose(edges, np.array([0.0, 2.5, 5.0, 7.5, 10.0]))


def test_mixed_edges_concatenates_segments_without_duplicate_boundary():
    """Tests that mixed_edges concatenates segments and removes duplicate
    boundaries."""
    segments = [
        {
            "method": "equidistant",
            "n_bins": 2,
            "params": {"x_min": 0.0, "x_max": 2.0},
        },
        {
            "method": "equidistant",
            "n_bins": 2,
            "params": {"x_min": 2.0, "x_max": 4.0},
        },
    ]
    edges = mixed_edges(segments)

    assert edges.shape == (5,)
    assert np.allclose(edges, np.array([0.0, 1.0, 2.0, 3.0, 4.0]))
    assert np.all(np.diff(edges) > 0)


def test_mixed_edges_raises_when_segment_boundaries_do_not_match():
    """Tests that mixed_edges raises ValueError when segment boundaries
    do not match."""
    segments = [
        {
            "method": "equidistant",
            "n_bins": 2,
            "params": {"x_min": 0.0, "x_max": 2.0},
        },
        # starts at 2.1 instead of 2.0
        {
            "method": "equidistant",
            "n_bins": 2,
            "params": {"x_min": 2.1, "x_max": 4.0},
        },
    ]
    with pytest.raises(ValueError, match=r"does not match previous right edge"):
        mixed_edges(segments)


def test_mixed_edges_uses_globals_when_params_missing():
    """Tests that mixed_edges uses global inputs when segment params
    are missing."""
    x = np.linspace(0.0, 10.0, 1001)
    w = np.ones_like(x)

    segments = [{"method": "equal_number", "n_bins": 5}]  # no params
    edges = mixed_edges(segments, x=x, weights=w)

    assert edges.shape == (6,)
    assert np.isclose(edges[0], x[0])
    assert np.isclose(edges[-1], x[-1])
    assert np.all(np.diff(edges) > 0)


def test_mixed_edges_equal_number_then_equidistant():
    """Tests that mixed_edges works for equal_number followed by equidistant."""
    x = np.linspace(0.0, 10.0, 2001)
    w = 1.0 + x

    segments = [
        {"method": "equal_number", "n_bins": 3},
        {
            "method": "equidistant",
            "n_bins": 2,
            "params": {"x_min": 10.0, "x_max": 12.0},
        },
    ]
    edges = mixed_edges(segments, x=x, weights=w)

    assert edges.shape == (6,)
    assert np.isclose(edges[0], 0.0)
    assert np.isclose(edges[-1], 12.0)
    assert np.all(np.diff(edges) > 0)
    assert np.count_nonzero(np.isclose(edges, 10.0)) == 1


def test_mixed_edges_equidistant_chi_matches_linear_chi_case():
    """Tests that mixed_edges with equidistant_chi matches
    equidistant_edges for linear chi(z)."""
    z = np.linspace(0.0, 2.0, 2001)
    chi = 10.0 * z

    segments = [{"method": "equidistant_chi", "n_bins": 4}]
    edges = mixed_edges(segments, z=z, chi=chi)

    assert edges.shape == (5,)
    assert np.allclose(edges, np.linspace(0.0, 2.0, 5), rtol=0, atol=1e-12)


def test_mixed_edges_raises_for_unknown_method():
    """Tests that mixed_edges raises ValueError for unknown method."""
    segments = [{"method": "not_a_method", "n_bins": 2}]
    with pytest.raises(ValueError, match=r"Unknown binning method"):
        mixed_edges(segments)


def test_mixed_edges_raises_when_required_global_missing():
    """Tests that mixed_edges raises ValueError when a required global input
    is missing."""
    segments = [{"method": "equal_number", "n_bins": 3}]
    with pytest.raises(ValueError, match=r"requires 'x'|requires 'weights'"):
        mixed_edges(segments)


def test_mixed_edges_log_and_geometric_are_monotonic():
    """Tests that mixed_edges with log and geometric segments
    produces strictly increasing edges."""
    segments = [
        {
            "method": "log",
            "n_bins": 3,
            "params": {"x_min": 1.0, "x_max": 100.0},
        },
        {
            "method": "geometric",
            "n_bins": 2,
            "params": {"x_min": 100.0, "x_max": 400.0},
        },
    ]
    edges = mixed_edges(segments)

    assert edges.shape == (6,)
    assert np.all(np.diff(edges) > 0)
    assert np.isclose(edges[0], 1.0)
    assert np.isclose(edges[-1], 400.0)


def test_validate_segment_edges_happy_path_returns_right_endpoint():
    """Tests that _validate_segment_edges returns the right endpoint
    when given valid edges."""
    edges = np.array([0.0, 1.0, 2.0])
    right = _validate_segment_edges(0, edges, n_bins=2, prev_right=None)
    assert right == 2.0


def test_validate_segment_edges_rejects_non_1d():
    """Tests that _validate_segment_edges raises ValueError when edges are
    not 1D."""
    edges = np.array([[0.0, 1.0, 2.0]])
    with pytest.raises(ValueError, match=r"edges must be 1D"):
        _validate_segment_edges(0, edges, n_bins=2, prev_right=None)


def test_validate_segment_edges_rejects_wrong_length():
    """Tests that _validate_segment_edges raises ValueError when edges
    length does not match n_bins + 1."""
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"expected 3 edges"):
        _validate_segment_edges(0, edges, n_bins=2, prev_right=None)


def test_validate_segment_edges_rejects_nonfinite():
    """Tests that _validate_segment_edges raises ValueError when edges
    contain non-finite values."""
    edges = np.array([0.0, np.nan, 2.0])
    with pytest.raises(ValueError, match=r"must be finite"):
        _validate_segment_edges(0, edges, n_bins=2, prev_right=None)


def test_validate_segment_edges_rejects_not_strictly_increasing():
    """Tests that _validate_segment_edges raises ValueError when edges
    are not strictly increasing."""
    edges = np.array([0.0, 1.0, 1.0])
    with pytest.raises(ValueError, match=r"strictly increasing"):
        _validate_segment_edges(0, edges, n_bins=2, prev_right=None)


def test_validate_segment_edges_rejects_boundary_mismatch():
    """Tests that _validate_segment_edges raises ValueError when the left
    edge does not match the previous right edge."""
    edges = np.array([2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match=r"does not match previous right edge"):
        _validate_segment_edges(1, edges, n_bins=2, prev_right=2.1, atol=1e-12)


def test_validate_segment_edges_allows_boundary_match_within_tolerance():
    """Tests that _validate_segment_edges allows left edge to match previous
    right edge within tolerance."""
    edges = np.array([2.0, 3.0, 4.0])
    right = _validate_segment_edges(1, edges, n_bins=2, prev_right=2.0 + 1e-13, atol=1e-12)
    assert right == 4.0


def test_mixed_edges_final_sanity_check_catches_nonincreasing_combined_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that mixed_edges raises if the combined edge array is not increasing."""

    def fake_call_with(
        seg_i: int,
        params,
        n_bins: int,
        g,
        *,
        func,
        required,
        casts=None,
    ) -> np.ndarray:
        """Fake calls to _call_with that return different edges for each segment."""
        _ = params, n_bins, g, func, required, casts
        if seg_i == 0:
            return np.array([0.0, 1.0, 2.0], dtype=float)
        return np.array([2.0, 1.5, 3.0], dtype=float)

    def fake_validate_segment_edges(
        seg_i: int,
        edges: np.ndarray,
        *,
        n_bins: int,
        prev_right: float | None,
        atol: float = 1e-12,
    ) -> float:
        """Fake validation that rejects non-increasing edges for the second segment."""
        _ = seg_i, n_bins
        edges = np.asarray(edges, dtype=float)
        if prev_right is not None and not np.isclose(edges[0], prev_right, rtol=0, atol=atol):
            raise ValueError("boundary mismatch in test")
        return float(edges[-1])

    monkeypatch.setattr(memod, "_call_with", fake_call_with)
    monkeypatch.setattr(memod, "_validate_segment_edges", fake_validate_segment_edges)

    segments = [
        {
            "method": "equidistant",
            "n_bins": 2,
            "params": {"x_min": 0.0, "x_max": 2.0},
        },
        {
            "method": "equidistant",
            "n_bins": 2,
            "params": {"x_min": 2.0, "x_max": 4.0},
        },
    ]

    with pytest.raises(ValueError, match=r"Combined mixed edges are not strictly increasing"):
        memod.mixed_edges(segments)
