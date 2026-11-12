"""Tests for the tomographic binning scheme registry."""

from __future__ import annotations

import numpy as np
import pytest

from binny.binning.scheme_registry import (
    available_binning_schemes,
    get_binning_scheme,
    register_binning_scheme,
)


@pytest.fixture
def isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that registry changes are isolated between tests."""
    monkeypatch.setattr("binny.binning.scheme_registry._SCHEMES", {})


def test_register_binning_scheme(isolated_registry: None) -> None:
    """Test that a binning scheme can be registered."""

    @register_binning_scheme("custom")
    def custom(z, nz, n_bins):
        return np.linspace(z[0], z[-1], n_bins + 1)

    assert get_binning_scheme("custom") is custom


def test_register_binning_scheme_aliases(isolated_registry: None) -> None:
    """Test that aliases resolve to the same registered function."""

    @register_binning_scheme("custom", aliases=("alias1", "alias2"))
    def custom(z, nz, n_bins):
        return np.linspace(z[0], z[-1], n_bins + 1)

    assert get_binning_scheme("custom") is custom
    assert get_binning_scheme("alias1") is custom
    assert get_binning_scheme("alias2") is custom


def test_register_binning_scheme_normalizes_names(isolated_registry: None) -> None:
    """Test that scheme names are stripped and lowercased."""

    @register_binning_scheme("  Custom  ", aliases=("  ALIAS  ",))
    def custom(z, nz, n_bins):
        return np.linspace(z[0], z[-1], n_bins + 1)

    assert get_binning_scheme("custom") is custom
    assert get_binning_scheme("CUSTOM") is custom
    assert get_binning_scheme(" alias ") is custom


def test_register_binning_scheme_rejects_empty_name(isolated_registry: None) -> None:
    """Test that empty scheme names are rejected."""

    with pytest.raises(ValueError, match="must not be empty"):

        @register_binning_scheme("  ")
        def custom(z, nz, n_bins):
            return np.linspace(z[0], z[-1], n_bins + 1)


def test_register_binning_scheme_rejects_empty_alias(isolated_registry: None) -> None:
    """Test that empty aliases are rejected."""

    with pytest.raises(ValueError, match="must not be empty"):

        @register_binning_scheme("custom", aliases=("  ",))
        def custom(z, nz, n_bins):
            return np.linspace(z[0], z[-1], n_bins + 1)


def test_register_binning_scheme_rejects_duplicate_name(isolated_registry: None) -> None:
    """Test that duplicate scheme registrations are rejected."""

    @register_binning_scheme("custom")
    def first(z, nz, n_bins):
        return np.linspace(z[0], z[-1], n_bins + 1)

    with pytest.raises(ValueError, match="already registered"):

        @register_binning_scheme("custom")
        def second(z, nz, n_bins):
            return np.linspace(z[0], z[-1], n_bins + 1)


def test_register_binning_scheme_rejects_duplicate_alias(isolated_registry: None) -> None:
    """Test that aliases cannot overwrite existing registrations."""

    @register_binning_scheme("first", aliases=("shared",))
    def first(z, nz, n_bins):
        return np.linspace(z[0], z[-1], n_bins + 1)

    with pytest.raises(ValueError, match="already registered"):

        @register_binning_scheme("second", aliases=("shared",))
        def second(z, nz, n_bins):
            return np.linspace(z[0], z[-1], n_bins + 1)


def test_register_binning_scheme_is_atomic(isolated_registry: None) -> None:
    """Test that failed registrations do not partially modify the registry."""

    @register_binning_scheme("existing")
    def existing(z, nz, n_bins):
        return np.linspace(z[0], z[-1], n_bins + 1)

    with pytest.raises(ValueError, match="already registered"):

        @register_binning_scheme("new", aliases=("existing",))
        def custom(z, nz, n_bins):
            return np.linspace(z[0], z[-1], n_bins + 1)

    assert available_binning_schemes() == ("existing",)


def test_get_binning_scheme_unknown(isolated_registry: None) -> None:
    """Test that retrieving an unknown scheme raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported binning_scheme"):
        get_binning_scheme("unknown")


def test_get_binning_scheme_normalizes_lookup(isolated_registry: None) -> None:
    """Test that scheme lookup ignores whitespace and case."""

    @register_binning_scheme("custom")
    def custom(z, nz, n_bins):
        return np.linspace(z[0], z[-1], n_bins + 1)

    assert get_binning_scheme("  CUSTOM  ") is custom


def test_available_binning_schemes_empty(isolated_registry: None) -> None:
    """Test that an empty registry returns an empty tuple."""
    assert available_binning_schemes() == ()


def test_available_binning_schemes_sorted(isolated_registry: None) -> None:
    """Test that registered names and aliases are returned sorted."""

    @register_binning_scheme("zeta", aliases=("gamma",))
    def zeta(z, nz, n_bins):
        return np.linspace(z[0], z[-1], n_bins + 1)

    @register_binning_scheme("alpha")
    def alpha(z, nz, n_bins):
        return np.linspace(z[0], z[-1], n_bins + 1)

    assert available_binning_schemes() == ("alpha", "gamma", "zeta")


def test_registered_binning_scheme_callable(isolated_registry: None) -> None:
    """Test that a retrieved scheme generates the expected bin edges."""

    @register_binning_scheme("custom")
    def custom(z, nz, n_bins, *, power=1.0):
        t = np.linspace(0.0, 1.0, n_bins + 1)
        return z[0] + (z[-1] - z[0]) * t**power

    z = np.linspace(0.0, 2.0, 101)
    nz = np.ones_like(z)

    scheme = get_binning_scheme("custom")
    edges = scheme(z, nz, 4, power=2.0)

    np.testing.assert_allclose(edges, [0.0, 0.125, 0.5, 1.125, 2.0])


def test_generalized_reproduces_equidistant_edges() -> None:
    """Test that alpha=0 and beta=0 reproduce equidistant redshift bins."""
    from binny.axes.bin_edges import equidistant_edges, generalized_tomographic_edges

    z = np.linspace(0.0, 3.0, 501)
    nz = z**2 * np.exp(-z)
    n_bins = 5

    expected = equidistant_edges(z[0], z[-1], n_bins)
    actual = generalized_tomographic_edges(
        z,
        nz,
        n_bins,
        alpha=0.0,
        beta=0.0,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_generalized_reproduces_equal_number_edges() -> None:
    """Test that alpha=1 and beta=0 reproduce equipopulated bins."""
    from binny.axes.bin_edges import equal_number_edges, generalized_tomographic_edges

    z = np.linspace(0.0, 3.0, 501)
    nz = z**2 * np.exp(-z)
    n_bins = 5

    expected = equal_number_edges(z, nz, n_bins)
    actual = generalized_tomographic_edges(
        z,
        nz,
        n_bins,
        alpha=1.0,
        beta=0.0,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)


def test_generalized_reproduces_equidistant_chi_edges() -> None:
    """Test that alpha=0 and beta=1 reproduce equidistant comoving-distance bins."""
    from binny.axes.bin_edges import (
        equidistant_chi_edges,
        generalized_tomographic_edges,
    )

    z = np.linspace(0.0, 3.0, 501)
    nz = z**2 * np.exp(-z)
    n_bins = 5

    dchi_dz = 1.0 / np.sqrt(0.3 * (1.0 + z) ** 3 + 0.7)
    chi = np.zeros_like(z)
    chi[1:] = np.cumsum(0.5 * (dchi_dz[1:] + dchi_dz[:-1]) * np.diff(z))

    expected = equidistant_chi_edges(z, chi, n_bins)
    actual = generalized_tomographic_edges(
        z,
        nz,
        n_bins,
        alpha=0.0,
        beta=1.0,
        comoving_distance=chi,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
