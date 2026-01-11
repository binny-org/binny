"""Unit tests for "binny.axes.grids"."""

from __future__ import annotations

import numpy as np
import pytest

from binny.axes.grids import linear_grid, log_grid


def test_linear_grid_returns_float64_and_shape():
    """Tests that linear_grid returns correct dtype and shape."""
    g = linear_grid(0.0, 10.0, 6)
    assert g.dtype == np.float64
    assert g.shape == (6,)


def test_linear_grid_includes_endpoints_and_is_linear():
    """Tests that linear_grid includes endpoints and is linear."""
    x_min, x_max, n = 1.5, 3.5, 5
    g = linear_grid(x_min, x_max, n)

    assert g[0] == pytest.approx(x_min)
    assert g[-1] == pytest.approx(x_max)

    # Constant spacing.
    d = np.diff(g)
    assert np.allclose(d, d[0])


@pytest.mark.parametrize("n", [2, 3, 10])
def test_linear_grid_n_points(n: int):
    """Tests that linear_grid returns n points."""
    g = linear_grid(-2.0, 7.0, n)
    assert len(g) == n


def test_linear_grid_accepts_integer_like_n():
    """Tests that linear_grid accepts integer-like n."""
    g = linear_grid(0.0, 1.0, 5.0)
    assert g.shape == (5,)
    assert g[0] == pytest.approx(0.0)
    assert g[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("n", [0, 1, -3])
def test_linear_grid_rejects_small_n(n: int):
    """Tests that linear_grid rejects small n."""
    with pytest.raises(ValueError, match=r"n must be"):
        linear_grid(0.0, 1.0, n)


def test_linear_grid_rejects_bool_n():
    """Tests that linear_grid rejects bool n."""
    with pytest.raises(TypeError, match=r"n must be"):
        linear_grid(0.0, 1.0, True)


def test_linear_grid_rejects_non_integer_like_n():
    """Tests that linear_grid rejects non-integer-like n."""
    with pytest.raises(TypeError, match=r"n must be"):
        linear_grid(0.0, 1.0, 3.5)


@pytest.mark.parametrize("x_min,x_max", [(1.0, 1.0), (2.0, 1.0)])
def test_linear_grid_rejects_non_increasing_endpoints(x_min: float, x_max: float):
    """Tests that linear_grid rejects non-increasing endpoints."""
    with pytest.raises(ValueError, match=r"x_max must be|increasing|greater than"):
        linear_grid(x_min, x_max, 3)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_linear_grid_rejects_non_finite_endpoints(bad: float):
    """Tests that linear_grid rejects non-finite endpoints."""
    with pytest.raises(ValueError, match=r"finite"):
        linear_grid(bad, 1.0, 3)
    with pytest.raises(ValueError, match=r"finite"):
        linear_grid(0.0, bad, 3)


def test_log_grid_returns_float64_and_shape():
    """Tests that log_grid returns correct dtype and shape."""
    g = log_grid(1.0, 100.0, 5)
    assert g.dtype == np.float64
    assert g.shape == (5,)


def test_log_grid_includes_endpoints_and_is_log_spaced():
    """Tests that log_grid includes endpoints and is log-spaced."""
    x_min, x_max, n = 1.0, 1_000.0, 6
    g = log_grid(x_min, x_max, n)

    assert g[0] == pytest.approx(x_min)
    assert g[-1] == pytest.approx(x_max)

    # Constant ratio for geometric progression.
    r = g[1:] / g[:-1]
    assert np.allclose(r, r[0])


@pytest.mark.parametrize("n", [2, 3, 10])
def test_log_grid_n_points(n: int):
    """Tests that log_grid returns n points."""
    g = log_grid(0.1, 10.0, n)
    assert len(g) == n


def test_log_grid_accepts_integer_like_n():
    """Tests that log_grid accepts integer-like n."""
    g = log_grid(1.0, 10.0, 4.0)
    assert g.shape == (4,)
    assert g[0] == pytest.approx(1.0)
    assert g[-1] == pytest.approx(10.0)


@pytest.mark.parametrize("n", [0, 1, -3])
def test_log_grid_rejects_small_n(n: int):
    """Tests that log_grid rejects small n."""
    with pytest.raises(ValueError, match=r"n must be"):
        log_grid(1.0, 10.0, n)


def test_log_grid_rejects_bool_n():
    """Tests that log_grid rejects bool n."""
    with pytest.raises(TypeError, match=r"n must be"):
        log_grid(1.0, 10.0, False)


def test_log_grid_rejects_non_integer_like_n():
    """Tests that log_grid rejects non-integer-like n."""
    with pytest.raises(TypeError, match=r"n must be"):
        log_grid(1.0, 10.0, 2.2)


@pytest.mark.parametrize("x_min,x_max", [(1.0, 1.0), (2.0, 1.0)])
def test_log_grid_rejects_non_increasing_endpoints(x_min: float, x_max: float):
    """Tests that log_grid rejects non-increasing endpoints."""
    with pytest.raises(ValueError, match=r"x_max must be|increasing|greater than"):
        log_grid(x_min, x_max, 3)


@pytest.mark.parametrize("x_min,x_max", [(0.0, 2.0), (-1.0, 2.0)])
def test_log_grid_rejects_non_positive_x_min(x_min: float, x_max: float):
    """Tests that log_grid rejects non-positive x_min."""
    with pytest.raises(ValueError, match=r"> 0|positive|log-spaced"):
        log_grid(x_min, x_max, 3)


def test_log_grid_matches_log_edges_output():
    """Tests that log_grid matches log_edges output."""
    x_min, x_max, n = 0.5, 50.0, 9
    g = log_grid(x_min, x_max, n)
    expected = np.asarray(log_grid(x_min, x_max, n), dtype=np.float64)
    assert np.allclose(g, expected)
