"""Unit tests for ``binny.nz.registry`` module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import binny.nz.registry as reg


@pytest.fixture
def z() -> np.ndarray:
    """Tests that a standard z grid is available for nz registry tests."""
    return np.linspace(0.0, 3.0, 301, dtype=np.float64)


def test_available_models_returns_sorted_list() -> None:
    """Tests that available_models returns a sorted list of strings."""
    names = reg.available_models()
    assert isinstance(names, list)
    assert all(isinstance(x, str) for x in names)
    assert names == sorted(names)
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    "name",
    [
        "smail",
        "gaussian",
        "gaussian_mixture",
        "gamma",
        "schechter",
        "lognormal",
        "tophat",
        "shifted_smail",
        "skew_normal",
        "student_t",
        "tabulated",
    ],
)
def test_get_model_returns_callable_for_known_names(name: str) -> None:
    """Tests that get_model returns a callable for all known names."""
    fn = reg.get_model(name)
    assert callable(fn)


@pytest.mark.parametrize("name", ["SMAIL", "SmAiL", "GAUSSIAN", "Schechter", "TABULATED"])
def test_get_model_is_case_insensitive(name: str) -> None:
    """Tests that get_model is case-insensitive for model names."""
    fn1 = reg.get_model(name)
    fn2 = reg.get_model(str(name).lower())
    assert fn1 is fn2


def test_get_model_does_not_strip_whitespace() -> None:
    """Tests that get_model does not strip whitespace from model names."""
    with pytest.raises(ValueError, match=r"Unknown redshift distribution model"):
        reg.get_model("  smail  ")


def test_get_model_unknown_raises_value_error() -> None:
    """Tests that get_model raises ValueError with available names included."""
    with pytest.raises(ValueError, match=r"Unknown redshift distribution model"):
        reg.get_model("definitely_not_a_model")

    try:
        reg.get_model("definitely_not_a_model")
    except ValueError as e:
        msg = str(e)
        assert "Available:" in msg
        assert "smail" in msg
        assert "tabulated" in msg


def test_nz_model_returns_float64_array(z: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that nz_model returns a float64 ndarray with same shape as z."""

    def _fake_model(z_in: np.ndarray, /, **params: Any) -> np.ndarray:
        _ = params
        return np.ones_like(z_in, dtype=np.float64)

    monkeypatch.setattr(reg, "_MODELS", {"fake": _fake_model})

    out = reg.nz_model("fake", z)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == z.shape


def test_nz_model_accepts_list_like_z(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that nz_model accepts list-like z and returns an ndarray."""

    def _fake_model(z_in: np.ndarray, /, **params: Any) -> np.ndarray:
        _ = params
        return np.arange(z_in.size, dtype=np.float64)

    monkeypatch.setattr(reg, "_MODELS", {"fake": _fake_model})

    z_list = [0.0, 0.5, 1.0]
    out = reg.nz_model("fake", z_list)
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == (3,)
    assert np.all(out == np.array([0.0, 1.0, 2.0], dtype=np.float64))


def test_nz_model_forwards_params(z: np.ndarray, monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that nz_model forwards params to the underlying model callable."""
    seen: dict[str, Any] = {}

    def _fake_model(z_in: np.ndarray, /, **params: Any) -> np.ndarray:
        seen["z"] = z_in
        seen["params"] = params
        return np.ones_like(z_in, dtype=np.float64)

    monkeypatch.setattr(reg, "_MODELS", {"fake": _fake_model})

    out = reg.nz_model("fake", z, a=1, b="x")
    assert np.all(out == 1.0)
    assert isinstance(seen["z"], np.ndarray)
    assert dict(seen["params"]) == {"a": 1, "b": "x"}


def test_nz_model_unknown_raises_value_error(z: np.ndarray) -> None:
    """Tests that nz_model raises ValueError for unknown model names."""
    with pytest.raises(ValueError, match=r"Unknown redshift distribution model"):
        _ = reg.nz_model("nope", z)


def test_nz_model_evaluates_tabulated_model(z: np.ndarray) -> None:
    """Tests that nz_model evaluates the tabulated model."""
    z_input = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    nz_input = np.array([0.0, 1.0, 0.5, 0.0], dtype=np.float64)

    out = reg.nz_model("tabulated", z, z_input=z_input, nz_input=nz_input)

    assert isinstance(out, np.ndarray)
    assert out.dtype == np.float64
    assert out.shape == z.shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)


def test_available_models_includes_tabulated() -> None:
    """Tests that available_models includes the tabulated model."""
    assert "tabulated" in reg.available_models()


def test_get_model_returns_tabulated_callable() -> None:
    """Tests that get_model returns the tabulated model callable."""
    fn = reg.get_model("tabulated")
    assert callable(fn)


def test_nz_model_tabulated_interpolates_expected_values() -> None:
    """Tests that nz_model interpolates expected values for tabulated input."""
    z_eval = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    z_input = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    nz_input = np.array([0.0, 2.0, 0.0], dtype=np.float64)

    out = reg.nz_model("tabulated", z_eval, z_input=z_input, nz_input=nz_input)

    expected = np.array([0.0, 1.0, 2.0, 1.0, 0.0], dtype=np.float64)

    assert np.allclose(out, expected)


def test_nz_model_tabulated_returns_zero_outside_table_range() -> None:
    """Tests that tabulated model returns zero outside the table range."""
    z_eval = np.array([-0.5, 0.0, 1.0, 2.0, 2.5], dtype=np.float64)
    z_input = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    nz_input = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    out = reg.nz_model("tabulated", z_eval, z_input=z_input, nz_input=nz_input)

    expected = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)

    assert np.allclose(out, expected)
