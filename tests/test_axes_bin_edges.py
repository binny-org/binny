"""Unit tests for ``binny.axes.bin_edges`` module."""

import numpy as np
import pytest

import binny.axes.bin_edges as bemod
from binny.axes.bin_edges import (
    _cumulative_trapz,
    _equal_weight_edges,
    equal_information_edges,
    equal_number_edges,
    equidistant_chi_edges,
    equidistant_edges,
    generalized_tomographic_edges,
    geometric_edges,
    log_edges,
)


def _trapz_integral_between(x: np.ndarray, w: np.ndarray, a: float, b: float) -> float:
    """Integrates weights ``w`` over ``x`` between ``a`` and ``b``
    using trapezoidal rule."""
    if b < a:
        a, b = b, a

    a = max(a, float(x[0]))
    b = min(b, float(x[-1]))
    if b <= a:
        return 0.0

    mask = (x > a) & (x < b)
    xg = np.concatenate(([a], x[mask], [b]))
    wg = np.interp(xg, x, w)
    return float(np.trapezoid(wg, xg))


def test_equidistant_edges_shape_endpoints_and_spacing():
    """Tests that equidistant_edges returns correct shape,
    endpoints, and uniform spacing."""
    edges = equidistant_edges(0.0, 10.0, 5)

    assert edges.shape == (6,)
    assert edges[0] == 0.0
    assert edges[-1] == 10.0

    diffs = np.diff(edges)
    assert np.allclose(diffs, diffs[0])


def test_equidistant_edges_n_bins_1_returns_two_edges():
    """Tests that equidistant_edges with n_bins=1 returns just the two endpoints."""
    edges = equidistant_edges(2.5, 3.5, 1)

    assert edges.shape == (2,)
    assert np.allclose(edges, [2.5, 3.5])


def test_equidistant_edges_invalid_interval_raises():
    """Tests that equidistant_edges raises ValueError for invalid intervals."""
    with pytest.raises(ValueError):
        equidistant_edges(1.0, 1.0, 5)  # zero width

    with pytest.raises(ValueError):
        equidistant_edges(2.0, 1.0, 5)  # reversed interval

    with pytest.raises(ValueError):
        equidistant_edges(0.0, 1.0, 0)  # invalid n_bins


def test_log_edges_shape_endpoints_and_monotonic():
    """Tests that log_edges returns correct shape, endpoints, and monotonic spacing."""
    edges = log_edges(1.0, 100.0, 4)

    assert edges.shape == (5,)
    assert edges[0] == 1.0
    assert edges[-1] == 100.0
    assert np.all(np.diff(edges) > 0)


def test_log_edges_n_bins_1_returns_two_edges():
    """Tests that log_edges with ``n_bins=1`` returns just the two endpoints."""
    edges = log_edges(0.1, 10.0, 1)

    assert np.allclose(edges, [0.1, 10.0])


def test_log_edges_invalid_min_raises():
    """Tests that log_edges raises ValueError for invalid x_min."""
    with pytest.raises(ValueError):
        log_edges(0.0, 10.0, 4)

    with pytest.raises(ValueError):
        log_edges(-1.0, 10.0, 4)


def test_geometric_edges_n_matches_geomspace_behavior():
    """Tests that geometric_edges_n matches numpy.geomspace behavior."""
    edges = geometric_edges(1.0, 1000.0, 3)
    ref = np.geomspace(1.0, 1000.0, 4, dtype=float)

    assert np.allclose(edges, ref)


def test_cumulative_trapz_constant_weights_linear_x():
    """Tests that _cumulative_trapz computes correct integral
    for constant weights and linear x."""
    x = np.array([0.0, 1.0, 2.0, 3.0])
    w = np.array([2.0, 2.0, 2.0, 2.0])
    cumul = _cumulative_trapz(x, w)

    assert np.allclose(cumul, [0.0, 2.0, 4.0, 6.0])


def test_cumulative_trapz_linear_weights_linear_x():
    """Tests that _cumulative_trapz computes correct integral
    for linear weights and linear x."""
    x = np.array([0.0, 1.0, 2.0])
    w = np.array([0.0, 1.0, 2.0])  # w=x
    cumul = _cumulative_trapz(x, w)

    assert np.allclose(cumul, [0.0, 0.5, 2.0])


def test_equal_weight_edges_constant_weights_reduces_to_equidistant():
    """Tests that _equal_weight_edges with constant weights reduces
    to equidistant edges."""
    x = np.linspace(0.0, 10.0, 501)
    w = np.ones_like(x)
    edges = _equal_weight_edges(x, w, 5)
    ref = equidistant_edges(0.0, 10.0, 5)

    assert np.allclose(edges, ref, rtol=0, atol=1e-12)


def test_equal_weight_edges_n_bins_1_returns_two_endpoints():
    """Tests that _equal_weight_edges with n_bins=1 returns just the two endpoints."""
    x = np.linspace(0.0, 10.0, 101)
    w = 1.0 + x  # any positive weights
    edges = _equal_weight_edges(x, w, 1)

    assert edges.shape == (2,)
    assert np.isclose(edges[0], x[0])
    assert np.isclose(edges[-1], x[-1])


def test_equal_number_edges_returns_correct_shape_and_endpoints():
    """Tests that equal_number_edges returns correct shape and endpoints."""
    x = np.linspace(0.0, 1.0, 101)
    weights = np.ones_like(x)
    edges = equal_number_edges(x, weights, 4)

    assert edges.shape == (5,)
    assert np.isclose(edges[0], x[0])
    assert np.isclose(edges[-1], x[-1])
    assert np.all(np.diff(edges) >= 0)


def test_equal_information_edges_behaves_same_as_equal_number_edges():
    """Tests that equal_information_edges behaves the same as equal_number_edges"""
    x = np.linspace(0.0, 2.0, 201)
    info = 1.0 + x**2
    e1 = equal_number_edges(x, info, 5)
    e2 = equal_information_edges(x, info, 5)

    assert np.allclose(e1, e2)


def test_equal_weight_edges_equalizes_integrated_weight_approximately():
    """Tests that _equal_weight_edges approximately equalizes
    integrated weight per bin."""
    x = np.linspace(0.0, 10.0, 2001)
    w = 1.0 + x

    n_bins = 5
    edges = _equal_weight_edges(x, w, n_bins)

    assert edges.shape == (n_bins + 1,)
    assert np.isclose(edges[0], x[0])
    assert np.isclose(edges[-1], x[-1])

    total = _trapz_integral_between(x, w, edges[0], edges[-1])
    target = total / n_bins

    bin_ints = [_trapz_integral_between(x, w, edges[i], edges[i + 1]) for i in range(n_bins)]

    rel_err = np.max(np.abs(np.array(bin_ints) - target) / target)
    assert rel_err < 5e-3


def test_equal_weight_edges_total_weight_nonpositive_raises():
    """Tests that _equal_weight_edges raises ValueError for
    non-positive total weight."""
    x = np.linspace(0.0, 1.0, 11)

    w0 = np.zeros_like(x)
    with pytest.raises(ValueError):
        _equal_weight_edges(x, w0, 3)

    wneg = -np.ones_like(x)
    with pytest.raises(ValueError):
        _equal_weight_edges(x, wneg, 3)


def test_equal_number_edges_invalid_n_bins_raises():
    """Tests that equal_number_edges raises ValueError for invalid n_bins."""
    x = np.linspace(0.0, 1.0, 11)
    w = np.ones_like(x)

    with pytest.raises(ValueError):
        equal_number_edges(x, w, 0)
    with pytest.raises(ValueError):
        equal_number_edges(x, w, -3)


def test_equidistant_chi_edges_matches_linear_chi_case():
    """Tests that equidistant_chi_edges matches equidistant_edges for linear chi(z)."""
    z = np.linspace(0.0, 2.0, 2001)
    chi = 10.0 * z  # linear, strictly increasing

    n_bins = 4
    z_edges = equidistant_chi_edges(z, chi, n_bins)
    ref = equidistant_edges(z[0], z[-1], n_bins)

    assert z_edges.shape == (n_bins + 1,)
    assert np.allclose(z_edges, ref, rtol=0, atol=1e-12)


def test_equidistant_chi_edges_monotonic_for_monotonic_chi():
    """Tests that equidistant_chi_edges returns monotonic edges for monotonic chi(z)."""
    z = np.linspace(0.0, 3.0, 1001)
    chi = z**2 + 1.0

    edges = equidistant_chi_edges(z, chi, 6)
    assert np.all(np.diff(edges) >= 0)
    assert np.isclose(edges[0], z[0])
    assert np.isclose(edges[-1], z[-1])


def test_equidistant_chi_edges_invalid_n_bins_raises():
    """Tests that equidistant_chi_edges raises ValueError for invalid n_bins."""
    z = np.linspace(0.0, 1.0, 11)
    chi = np.linspace(0.0, 10.0, 11)
    with pytest.raises(ValueError):
        equidistant_chi_edges(z, chi, 0)


def test_equal_weight_edges_nonfinite_total_raises():
    """Tests that _equal_weight_edges raises ValueError
    for non-finite total weight."""
    x = np.linspace(0.0, 1.0, 11)
    w = np.full_like(x, np.inf)
    with pytest.raises(ValueError):
        _equal_weight_edges(x, w, 3)


def test_equidistant_chi_edges_requires_strictly_increasing_chi():
    """Tests that equidistant_chi_edges raises when chi is not strictly increasing."""
    z = np.linspace(0.0, 1.0, 6)
    chi = np.array([0.0, 1.0, 1.0, 2.0, 3.0, 4.0])  # plateau -> not strictly inc

    with pytest.raises(ValueError, match=r"chi must be strictly increasing"):
        _ = equidistant_chi_edges(z, chi, 3)


def test_equal_weight_edges_repeated_edges_guard_triggers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that _equal_weight_edges raises if interpolation yields non-increasing edges."""

    def _bad_interp(_x, _xp, fp):
        """Returns first value in fp array, which is repeated for all x values."""
        return float(fp[0])

    monkeypatch.setattr(bemod.np, "interp", _bad_interp)

    x = np.linspace(0.0, 1.0, 11)
    w = np.ones_like(x)

    with pytest.raises(ValueError, match=r"Cannot construct strictly increasing bin edges"):
        bemod._equal_weight_edges(x, w, 5)


@pytest.fixture
def generalized_sample():
    """Return a redshift distribution and comoving-distance derivative."""
    z = np.linspace(0.0, 2.0, 1001)
    nz = np.exp(-0.5 * ((z - 1.0) / 0.35) ** 2)
    dchi_dz = 3000.0 / np.sqrt(0.3 * (1.0 + z) ** 3 + 0.7)
    return z, nz, dchi_dz


def test_generalized_edges_equal_redshift(generalized_sample):
    """Recover equidistant redshift binning."""
    z, nz, _ = generalized_sample

    edges = generalized_tomographic_edges(z, nz, 5)
    expected = equidistant_edges(z[0], z[-1], 5)

    np.testing.assert_allclose(edges, expected, rtol=0, atol=1e-12)


def test_generalized_edges_equal_number(generalized_sample):
    """Recover equal-number binning."""
    z, nz, _ = generalized_sample

    edges = generalized_tomographic_edges(z, nz, 5, alpha=1.0)
    expected = equal_number_edges(z, nz, 5)

    np.testing.assert_allclose(edges, expected, rtol=0, atol=1e-12)


def test_generalized_edges_equal_comoving(generalized_sample):
    """Recover equal-comoving-distance binning."""
    z, nz, dchi_dz = generalized_sample

    chi = _cumulative_trapz(z, dchi_dz)

    edges = generalized_tomographic_edges(z, nz, 5, beta=1.0, dchi_dz=dchi_dz)
    expected = equidistant_chi_edges(z, chi, 5)

    np.testing.assert_allclose(edges, expected, rtol=0, atol=1e-10)


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5)],
)
def test_generalized_edges_valid(generalized_sample, alpha, beta):
    """Verify edge count, monotonicity, and endpoints."""
    z, nz, dchi_dz = generalized_sample

    edges = generalized_tomographic_edges(
        z,
        nz,
        5,
        alpha=alpha,
        beta=beta,
        dchi_dz=dchi_dz,
    )

    assert edges.shape == (6,)
    assert np.all(np.diff(edges) > 0)
    assert edges[0] == z[0]
    assert edges[-1] == z[-1]


def test_generalized_edges_single_bin(generalized_sample):
    """One bin must span the full redshift interval."""
    z, nz, dchi_dz = generalized_sample

    edges = generalized_tomographic_edges(z, nz, 1, alpha=0.5, beta=0.5, dchi_dz=dchi_dz)

    np.testing.assert_array_equal(edges, [z[0], z[-1]])


def test_generalized_edges_missing_dchi_dz(generalized_sample):
    """Require the distance derivative for nonzero beta."""
    z, nz, _ = generalized_sample

    with pytest.raises(ValueError, match="dchi_dz is required"):
        generalized_tomographic_edges(z, nz, 5, beta=1.0)


def test_generalized_edges_negative_nz(generalized_sample):
    """Reject negative redshift distributions."""
    z, nz, _ = generalized_sample
    nz = nz.copy()
    nz[10] = -1.0

    with pytest.raises(ValueError, match="nz must be non-negative"):
        generalized_tomographic_edges(z, nz, 5)


def test_generalized_edges_negative_alpha_zero_nz(generalized_sample):
    """Reject zero densities raised to negative powers."""
    z, nz, _ = generalized_sample
    nz = nz.copy()
    nz[0] = 0.0

    with pytest.raises(ValueError, match="nz must be strictly positive"):
        generalized_tomographic_edges(z, nz, 5, alpha=-0.5)


def test_generalized_edges_nonpositive_dchi_dz(generalized_sample):
    """Reject nonpositive comoving-distance derivatives."""
    z, nz, dchi_dz = generalized_sample
    dchi_dz = dchi_dz.copy()
    dchi_dz[10] = 0.0

    with pytest.raises(ValueError, match="dchi_dz must be strictly positive"):
        generalized_tomographic_edges(z, nz, 5, beta=1.0, dchi_dz=dchi_dz)


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [(np.nan, 0.0), (0.0, np.inf)],
)
def test_generalized_edges_invalid_exponents(generalized_sample, alpha, beta):
    """Reject non-finite alpha and beta."""
    z, nz, _ = generalized_sample

    with pytest.raises(ValueError, match="alpha and beta must be finite"):
        generalized_tomographic_edges(z, nz, 5, alpha=alpha, beta=beta)


def test_generalized_edges_zero_distribution():
    """Reject a distribution with zero integrated weight."""
    z = np.linspace(0.0, 2.0, 101)
    nz = np.zeros_like(z)

    with pytest.raises(ValueError):
        generalized_tomographic_edges(z, nz, 5, alpha=1.0)


def test_generalized_edges_invalid_n_bins(generalized_sample):
    """Reject invalid numbers of bins."""
    z, nz, _ = generalized_sample

    with pytest.raises(ValueError):
        generalized_tomographic_edges(z, nz, 0)
