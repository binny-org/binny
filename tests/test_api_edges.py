"""Unit tests for ``binny.api.edges`` module."""

from __future__ import annotations

import numpy as np
import pytest

import binny.api.edges as api


def _stubs(name: str):
    """Creates a function that records its calls and returns a phnatom
    value."""
    sentinel = np.array([hash(name) % 97], dtype=float)

    def _fn(**kwargs):
        """Phantom function that records its calls and returns a phantom
        value."""
        _fn.called = True
        _fn.kwargs = dict(kwargs)
        return sentinel

    _fn.called = False
    _fn.kwargs = None
    _fn.sentinel = sentinel
    return _fn


def _assert_kwargs_equal(actual: dict, expected: dict) -> None:
    """Compares kwargs dicts safely when they may contain numpy arrays."""
    assert actual.keys() == expected.keys()
    for k, v_exp in expected.items():
        v_act = actual[k]
        if isinstance(v_exp, np.ndarray) or isinstance(v_act, np.ndarray):
            np.testing.assert_array_equal(np.asarray(v_act), np.asarray(v_exp))
        else:
            assert v_act == v_exp


@pytest.mark.parametrize(
    "method, target_attr",
    [
        ("equidistant", "equidistant_edges"),
        ("linear", "equidistant_edges"),
        ("eq", "equidistant_edges"),
        ("EQ", "equidistant_edges"),
        ("LoG", "log_edges"),
        ("log", "log_edges"),
        ("log_edges", "log_edges"),
        ("logarithmic", "log_edges"),
        ("geometric", "geometric_edges"),
        ("geom", "geometric_edges"),
        ("geomspace", "geometric_edges"),
        ("equal_number", "equal_number_edges"),
        ("equipopulated", "equal_number_edges"),
        ("en", "equal_number_edges"),
        ("equal_information", "equal_information_edges"),
        ("ei", "equal_information_edges"),
        ("equidistant_chi", "equidistant_chi_edges"),
        ("chi", "equidistant_chi_edges"),
    ],
)
def test_bin_edges_routes_aliases_and_is_case_insensitive(
    method: str, target_attr: str, monkeypatch
):
    """Tests that bin_edges routes aliases and is case-insensitive."""
    stubs = {}
    for attr in [
        "equidistant_edges",
        "log_edges",
        "geometric_edges",
        "equal_number_edges",
        "equal_information_edges",
        "equidistant_chi_edges",
    ]:
        stub = _stubs(attr)
        stubs[attr] = stub
        monkeypatch.setattr(api, attr, stub)

    x = np.array([1, 2, 3])
    expected_kwargs = {"a": 1, "b": "two", "x": x}

    out = api.bin_edges(method, **expected_kwargs)

    for attr, stub in stubs.items():
        if attr == target_attr:
            assert stub.called is True, f"Expected {attr} for method={method!r}"
            _assert_kwargs_equal(stub.kwargs, expected_kwargs)
            np.testing.assert_array_equal(out, stub.sentinel)
        else:
            assert stub.called is False, f"Did not expect {attr} for method={method!r}"


def test_bin_edges_unknown_method_raises_value_error():
    """Tests that bin_edges raises ValueError for unknown method."""
    with pytest.raises(ValueError, match=r"Unknown bin edge method"):
        api.bin_edges("definitely_not_a_method", x_min=0.0, x_max=1.0, n_bins=3)


def test_bin_edges_equdistant_smoke_real_output():
    """Tests that bin_edges returns a valid array for equidistant method."""
    edges = api.bin_edges("equidistant", x_min=0.0, x_max=3.0, n_bins=5)

    edges = np.asarray(edges)
    assert edges.ndim == 1
    assert edges.size == 6  # n_bins + 1
    assert np.isclose(edges[0], 0.0)
    assert np.isclose(edges[-1], 3.0)
    assert np.all(np.diff(edges) > 0)


def test_bin_edges_log_smoke_real_output():
    """Tests that bin_edges returns a valid array for logarithmic method."""
    edges = api.bin_edges("log", x_min=10.0, x_max=2000.0, n_bins=4)

    edges = np.asarray(edges)
    assert edges.ndim == 1
    assert edges.size == 5
    assert np.isclose(edges[0], 10.0)
    assert np.isclose(edges[-1], 2000.0)
    assert np.all(np.diff(edges) > 0)


def test_bin_edges_geometric_smoke_real_output():
    """Tests that bin_edges returns a valid array for geometric method."""
    edges = api.bin_edges("geometric", x_min=2.0, x_max=162.0, n_bins=4)

    edges = np.asarray(edges)
    assert edges.ndim == 1
    assert edges.size == 5
    assert np.isclose(edges[0], 2.0)
    assert np.isclose(edges[-1], 162.0)
    assert np.all(np.diff(edges) > 0)


def test_bin_edges_equal_number_smoke_real_output():
    """Tests that bin_edges returns a valid array for equal_number method."""
    x = np.linspace(0.0, 10.0, 501)
    w = np.ones_like(x)

    edges = api.bin_edges("equal_number", x=x, weights=w, n_bins=5)

    edges = np.asarray(edges)
    assert edges.ndim == 1
    assert edges.size == 6
    assert np.isclose(edges[0], x.min())
    assert np.isclose(edges[-1], x.max())
    assert np.all(np.diff(edges) >= 0)
