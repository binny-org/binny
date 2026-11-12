"""Unit tests for "binny.axes.grids"."""

from __future__ import annotations

import numpy as np
import pytest

from binny.axes.grids import (
    chebyshev_grid,
    grid,
    linear_grid,
    log_grid,
    power_grid,
    register_grid,
    sinh_grid,
)


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


@pytest.mark.parametrize("kind", ["linear", "lin", "uniform", "LiNeAr"])
def test_grid_dispatches_to_linear(kind: str):
    """Tests that grid dispatches linear aliases to linear_grid."""
    g = grid(kind, start=0.0, stop=1.0, n=5)
    expected = linear_grid(0.0, 1.0, 5)
    assert np.allclose(g, expected)
    assert g.dtype == np.float64


@pytest.mark.parametrize("kind", ["log", "log_grid", "logarithmic", "geom", "geometric", "LoG"])
def test_grid_dispatches_to_log(kind: str):
    """Tests that grid dispatches log aliases to log_grid."""
    g = grid(kind, start=1.0, stop=100.0, n=6)
    expected = log_grid(1.0, 100.0, 6)
    assert np.allclose(g, expected)
    assert g.dtype == np.float64


def test_grid_rejects_unknown_kind():
    """Tests that grid raises for unknown kind."""
    with pytest.raises(ValueError, match=r"Unknown grid kind"):
        grid("banana", start=0.0, stop=1.0, n=3)


def test_log_grid_matches_log_edges_output():
    """Tests that log_grid matches log_edges output."""
    x_min, x_max, n = 0.5, 50.0, 9
    g = log_grid(x_min, x_max, n)
    expected = np.asarray(log_grid(x_min, x_max, n), dtype=np.float64)
    assert np.allclose(g, expected)


@pytest.mark.parametrize("power", [0.5, 1.0, 2.0, 3.0])
def test_power_grid_properties(power):
    """Test shape, dtype, endpoints, and monotonicity."""
    g = power_grid(0.0, 10.0, 50, power=power)

    assert g.shape == (50,)
    assert g.dtype == np.float64
    assert g[0] == pytest.approx(0.0)
    assert g[-1] == pytest.approx(10.0)
    assert np.all(np.diff(g) > 0)


def test_power_grid_matches_analytic_formula():
    """Test the power-law coordinate transformation."""
    g = power_grid(2.0, 10.0, 5, power=2.0)
    t = np.linspace(0.0, 1.0, 5)

    np.testing.assert_allclose(g, 2.0 + 8.0 * t**2)


def test_power_grid_one_matches_linear():
    """Test that power=1 reproduces linear spacing."""
    actual = power_grid(-2.0, 7.0, 20, power=1.0)
    expected = linear_grid(-2.0, 7.0, 20)

    np.testing.assert_allclose(actual, expected)


def test_power_grid_stretching_direction():
    """Test that power controls the clustering direction."""
    lower = power_grid(0.0, 1.0, 11, power=2.0)
    upper = power_grid(0.0, 1.0, 11, power=0.5)

    assert lower[1] < 0.1
    assert upper[1] > 0.1


@pytest.mark.parametrize("power", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_power_grid_rejects_invalid_power(power):
    """Test that power must be positive and finite."""
    with pytest.raises(ValueError, match="power must be positive and finite"):
        power_grid(0.0, 1.0, 10, power=power)


@pytest.mark.parametrize("n", [0, 1, -2])
def test_power_grid_rejects_invalid_n(n):
    """Test that power grids share the standard grid validation."""
    with pytest.raises(ValueError, match="n must be"):
        power_grid(0.0, 1.0, n)


@pytest.mark.parametrize("n", [2, 3, 10, 50])
def test_chebyshev_grid_properties(n):
    """Test shape, dtype, endpoints, and monotonicity."""
    g = chebyshev_grid(-3.0, 7.0, n)

    assert g.shape == (n,)
    assert g.dtype == np.float64
    assert g[0] == pytest.approx(-3.0)
    assert g[-1] == pytest.approx(7.0)
    assert np.all(np.diff(g) > 0)


def test_chebyshev_grid_matches_analytic_formula():
    """Test Chebyshev-Lobatto node positions."""
    n = 11
    theta = np.arange(n) * np.pi / (n - 1)
    expected = 0.5 * (1.0 - np.cos(theta))

    np.testing.assert_allclose(
        chebyshev_grid(0.0, 1.0, n),
        expected,
    )


def test_chebyshev_grid_clusters_near_endpoints():
    """Test that endpoint spacing is smaller than central spacing."""
    g = chebyshev_grid(0.0, 1.0, 21)
    spacing = np.diff(g)

    assert spacing[0] < spacing[len(spacing) // 2]
    assert spacing[-1] < spacing[len(spacing) // 2]


def test_chebyshev_grid_is_symmetric():
    """Test symmetry around the interval midpoint."""
    g = chebyshev_grid(0.0, 10.0, 21)

    np.testing.assert_allclose(g + g[::-1], 10.0)


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0, 3.0, 5.0])
def test_sinh_grid_properties(strength):
    """Test shape, dtype, endpoints, and monotonicity."""
    g = sinh_grid(-2.0, 8.0, 51, strength=strength)

    assert g.shape == (51,)
    assert g.dtype == np.float64
    assert g[0] == pytest.approx(-2.0)
    assert g[-1] == pytest.approx(8.0)
    assert np.all(np.diff(g) > 0)


def test_sinh_grid_zero_matches_linear():
    """Test that zero stretching produces linear spacing."""
    actual = sinh_grid(0.0, 10.0, 31, strength=0.0)
    expected = linear_grid(0.0, 10.0, 31)

    np.testing.assert_allclose(actual, expected)


def test_sinh_grid_is_symmetric():
    """Test symmetry around the interval midpoint."""
    g = sinh_grid(0.0, 10.0, 51, strength=3.0)

    np.testing.assert_allclose(g + g[::-1], 10.0)


def test_sinh_grid_clusters_near_midpoint():
    """Test that central spacing is smaller than endpoint spacing."""
    g = sinh_grid(0.0, 1.0, 51, strength=4.0)
    spacing = np.diff(g)

    assert spacing[len(spacing) // 2] < spacing[0]
    assert spacing[len(spacing) // 2] < spacing[-1]


def test_sinh_grid_strength_controls_concentration():
    """Test that stronger stretching increases central concentration."""
    weak = sinh_grid(0.0, 1.0, 51, strength=1.0)
    strong = sinh_grid(0.0, 1.0, 51, strength=4.0)

    middle = len(weak) // 2

    assert strong[middle + 1] - strong[middle] < (weak[middle + 1] - weak[middle])


@pytest.mark.parametrize("strength", [-1.0, np.nan, np.inf, -np.inf])
def test_sinh_grid_rejects_invalid_strength(strength):
    """Test that strength must be nonnegative and finite."""
    with pytest.raises(ValueError, match="strength must be nonnegative and finite"):
        sinh_grid(0.0, 1.0, 10, strength=strength)


@pytest.mark.parametrize(
    ("kind", "kwargs", "builder"),
    [
        ("power", {"power": 2.5}, power_grid),
        ("POWER", {"power": 2.5}, power_grid),
        ("chebyshev", {}, chebyshev_grid),
        ("CHEBYSHEV", {}, chebyshev_grid),
        ("sinh", {"strength": 3.0}, sinh_grid),
        ("SINH", {"strength": 3.0}, sinh_grid),
    ],
)
def test_grid_dispatches_new_strategies(kind, kwargs, builder):
    """Test registered built-in strategies and case-insensitive lookup."""
    actual = grid(kind, start=0.0, stop=10.0, n=30, **kwargs)
    expected = builder(0.0, 10.0, 30, **kwargs)

    np.testing.assert_allclose(actual, expected)


def test_register_custom_grid():
    """Test registration and dispatch of a custom grid."""

    @register_grid("test_custom_quadratic")
    def custom(start, stop, n):
        t = np.linspace(0.0, 1.0, n)
        return start + (stop - start) * t**2

    actual = grid("test_custom_quadratic", start=0.0, stop=10.0, n=11)
    expected = custom(0.0, 10.0, 11)

    np.testing.assert_allclose(actual, expected)


def test_register_grid_preserves_callable():
    """Test that registration returns the original callable."""

    def custom(start, stop, n):
        return np.linspace(start, stop, n)

    registered = register_grid("test_preserved_callable")(custom)

    assert registered is custom


def test_register_grid_is_case_insensitive():
    """Test case-insensitive registration and lookup."""

    @register_grid("test_case_grid")
    def custom(start, stop, n):
        return np.linspace(start, stop, n)

    actual = grid("TEST_CASE_GRID", start=0.0, stop=1.0, n=5)

    np.testing.assert_allclose(actual, np.linspace(0.0, 1.0, 5))


def test_register_grid_rejects_duplicate_name():
    """Test that a registered name cannot be overwritten."""

    @register_grid("test_duplicate_grid")
    def custom(start, stop, n):
        return np.linspace(start, stop, n)

    with pytest.raises(ValueError, match="already registered"):
        register_grid("test_duplicate_grid")(custom)


@pytest.mark.parametrize("name", ["linear", "log", "power", "chebyshev", "sinh"])
def test_register_grid_rejects_builtin_names(name):
    """Test that custom registration cannot overwrite built-in grids."""
    with pytest.raises(ValueError):
        register_grid(name)(lambda: None)


@pytest.mark.parametrize("name", ["lin", "uniform", "geom", "geometric"])
def test_register_grid_rejects_reserved_aliases(name):
    """Test that existing aliases cannot be registered as custom names."""
    with pytest.raises(ValueError):
        register_grid(name)(lambda: None)


@pytest.mark.parametrize("name", ["", 123, None])
def test_register_grid_rejects_invalid_names(name):
    """Test rejection of empty or non-string registry names."""
    with pytest.raises((TypeError, ValueError)):
        register_grid(name)


def test_register_grid_rejects_non_callable():
    """Test that registered objects must be callable."""
    with pytest.raises(TypeError, match="callable"):
        register_grid("test_not_callable")(42)


def test_grid_rejects_non_1d_custom_output():
    """Test that custom grid output must be one-dimensional."""

    @register_grid("test_invalid_ndim")
    def custom():
        return np.ones((2, 2))

    with pytest.raises(ValueError, match="strictly increasing 1D"):
        grid("test_invalid_ndim")


def test_grid_rejects_non_finite_custom_output():
    """Test that custom grid output must contain finite values."""

    @register_grid("test_invalid_finite")
    def custom():
        return np.array([0.0, np.nan, 1.0])

    with pytest.raises(ValueError, match="finite"):
        grid("test_invalid_finite")


def test_grid_rejects_non_increasing_custom_output():
    """Test that custom grid points must be strictly increasing."""

    @register_grid("test_invalid_order")
    def custom():
        return np.array([0.0, 2.0, 1.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        grid("test_invalid_order")


def test_grid_rejects_duplicate_custom_points():
    """Test that duplicate coordinates are rejected."""

    @register_grid("test_duplicate_points")
    def custom():
        return np.array([0.0, 1.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        grid("test_duplicate_points")


def test_grid_rejects_single_point_custom_output():
    """Test that a grid must contain at least two points."""

    @register_grid("test_single_point")
    def custom():
        return np.array([1.0])

    with pytest.raises(ValueError, match="at least two points"):
        grid("test_single_point")


def test_grid_converts_custom_output_to_float64():
    """Test that custom grid output is converted to float64."""

    @register_grid("test_float32_grid")
    def custom():
        return np.array([0.0, 0.5, 1.0], dtype=np.float32)

    actual = grid("test_float32_grid")

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, [0.0, 0.5, 1.0])


def test_grid_forwards_custom_keyword_arguments():
    """Test that custom grid parameters are forwarded correctly."""

    @register_grid("test_kwargs_grid")
    def custom(start, stop, n, *, exponent=1.0):
        t = np.linspace(0.0, 1.0, n)
        return start + (stop - start) * t**exponent

    actual = grid(
        "test_kwargs_grid",
        start=0.0,
        stop=1.0,
        n=5,
        exponent=2.0,
    )

    np.testing.assert_allclose(actual, np.linspace(0.0, 1.0, 5) ** 2)
