"""Unit tests for ``binny.api.metrics`` module."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

import binny.api.metrics as api


def _stub(name: str):
    """Creates a function that records its calls and returns a phnatomic value."""

    def _fn(*args, **kwargs):
        """Phantom function that records its calls and returns a phantom value."""
        _fn.called = True
        _fn.args = args
        _fn.kwargs = dict(kwargs)
        return _fn.return_value

    _fn.name = name
    _fn.called = False
    _fn.args = None
    _fn.kwargs = None
    _fn.return_value = None
    return _fn


def test_bin_summary_centers_only_calls_centers_and_summarize(monkeypatch):
    """Tests that bin_summary calls bin_centers and summarize_bins."""
    z = np.linspace(0.0, 1.0, 11)
    bins: Mapping[int, Any] = {0: np.ones_like(z), 1: np.ones_like(z) * 2}

    stub_centers = _stub("bin_centers")
    stub_density = _stub("galaxy_density_per_bin")
    stub_count = _stub("galaxy_count_per_bin")
    stub_summarize = _stub("summarize_bins")
    stub_fraction = _stub("galaxy_fraction_per_bin")

    stub_centers.return_value = {"0": 0.1, "1": 0.2}
    stub_summarize.return_value = {"anything": "ok"}
    stub_fraction.return_value = (0, None)

    monkeypatch.setattr(api, "bin_centers", stub_centers)
    monkeypatch.setattr(api, "galaxy_density_per_bin", stub_density)
    monkeypatch.setattr(api, "galaxy_count_per_bin", stub_count)
    monkeypatch.setattr(api, "summarize_bins", stub_summarize)
    monkeypatch.setattr(api, "galaxy_fraction_per_bin", stub_fraction)

    out = api.bin_summary(z, bins, center_method="mean", decimal_places=3)

    assert out == {
        "centers": {"0": 0.1, "1": 0.2},
        "fraction_per_bin": 0,
        "density_per_bin": None,
        "count_per_bin": None,
        "summary": {"anything": "ok"},
    }

    assert stub_centers.called is True
    assert stub_centers.args == (z, bins)
    assert stub_centers.kwargs == {"method": "mean", "decimal_places": 3}

    assert stub_density.called is False
    assert stub_count.called is False

    assert stub_summarize.called is True
    assert stub_summarize.args == (z, bins)
    assert stub_summarize.kwargs == {
        "count_per_bin": None,
        "density_per_bin": None,
        "survey_area": None,
    }

    assert stub_fraction.called is True
    assert stub_fraction.args == (z, bins)
    assert stub_fraction.kwargs == {}


def test_bin_summary_with_density_calls_density_and_summarize(monkeypatch):
    """Tests that bin_summary calls galaxy_density_per_bin and summarize_bins."""
    z = np.linspace(0.0, 1.0, 11)
    bins: Mapping[int, Any] = {0: np.ones_like(z), 1: np.ones_like(z) * 2}

    stub_centers = _stub("bin_centers")
    stub_density = _stub("galaxy_density_per_bin")
    stub_count = _stub("galaxy_count_per_bin")
    stub_summarize = _stub("summarize_bins")
    stub_fraction = _stub("galaxy_fraction_per_bin")

    centers_ret = {"0": 0.11, "1": 0.22}
    density_ret = {"0": 5.0, "1": 7.0}
    frac_ret = {"0": 0.4, "1": 0.6}

    stub_centers.return_value = centers_ret
    stub_density.return_value = (density_ret, {"0": 0.4, "1": 0.6})
    stub_summarize.return_value = {"summary": True}
    stub_fraction.return_value = (frac_ret, None)

    monkeypatch.setattr(api, "bin_centers", stub_centers)
    monkeypatch.setattr(api, "galaxy_density_per_bin", stub_density)
    monkeypatch.setattr(api, "galaxy_count_per_bin", stub_count)
    monkeypatch.setattr(api, "summarize_bins", stub_summarize)
    monkeypatch.setattr(api, "galaxy_fraction_per_bin", stub_fraction)

    out = api.bin_summary(
        z,
        bins,
        center_method="median",
        decimal_places=None,
        density_total=12.0,
        survey_area=None,
    )

    assert out["centers"] == centers_ret
    assert out["density_per_bin"] == density_ret
    assert out["count_per_bin"] is None
    assert out["summary"] == {"summary": True}
    assert out["fraction_per_bin"] == frac_ret

    assert stub_centers.called is True
    assert stub_centers.kwargs == {"method": "median", "decimal_places": None}

    assert stub_density.called is True
    assert stub_density.args == (z, bins)
    assert stub_density.kwargs == {"density_total": 12.0}

    assert stub_count.called is False

    assert stub_summarize.called is True
    assert stub_summarize.kwargs == {
        "count_per_bin": None,
        "density_per_bin": density_ret,
        "survey_area": None,
    }
    assert stub_fraction.called is True
    assert stub_fraction.args == (z, bins)
    assert stub_fraction.kwargs == {}


def test_bin_summary_with_density_and_area_calls_count(monkeypatch):
    z = np.linspace(0.0, 1.0, 11)
    bins: Mapping[int, Any] = {0: np.ones_like(z), 1: np.ones_like(z) * 2}

    stub_centers = _stub("bin_centers")
    stub_density = _stub("galaxy_density_per_bin")
    stub_count = _stub("galaxy_count_per_bin")
    stub_summarize = _stub("summarize_bins")
    stub_fraction = _stub("galaxy_fraction_per_bin")

    centers_ret = {"0": 0.1, "1": 0.2}
    density_ret = {"0": 1.5, "1": 2.5}
    count_ret = {"0": 150.0, "1": 250.0}
    frac_ret = {"0": 0.3, "1": 0.7}

    stub_centers.return_value = centers_ret
    stub_density.return_value = (density_ret, {"0": 0.3, "1": 0.7})
    stub_count.return_value = count_ret
    stub_summarize.return_value = {"ok": 1}
    stub_fraction.return_value = (frac_ret, None)

    monkeypatch.setattr(api, "bin_centers", stub_centers)
    monkeypatch.setattr(api, "galaxy_density_per_bin", stub_density)
    monkeypatch.setattr(api, "galaxy_count_per_bin", stub_count)
    monkeypatch.setattr(api, "summarize_bins", stub_summarize)
    monkeypatch.setattr(api, "galaxy_fraction_per_bin", stub_fraction)

    out = api.bin_summary(
        z,
        bins,
        density_total=4.0,
        survey_area=100.0,
    )

    assert out == {
        "centers": centers_ret,
        "density_per_bin": density_ret,
        "count_per_bin": count_ret,
        "summary": {"ok": 1},
        "fraction_per_bin": frac_ret,
    }

    assert stub_density.called is True
    assert stub_count.called is True
    assert stub_count.args == (density_ret, 100.0)
    assert stub_count.kwargs == {}

    assert stub_summarize.called is True
    assert stub_summarize.kwargs == {
        "count_per_bin": count_ret,
        "density_per_bin": density_ret,
        "survey_area": 100.0,
    }
    assert stub_fraction.called is True
    assert stub_fraction.args == (z, bins)
    assert stub_fraction.kwargs == {}
