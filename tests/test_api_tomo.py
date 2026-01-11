"""Unit tests for ``api.tomo` module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

import binny.api.tomo as api


@dataclass
class Recorder:
    """Records calls to a function and returns its arguments."""

    return_value: Any = None
    called: bool = False
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Calls the underlying function and records its arguments."""
        self.called = True
        self.args = args
        self.kwargs = dict(kwargs)
        return self.return_value


def test_photoz_bins_forwards_to_build_photoz_bins(monkeypatch):
    """Tests that photoz_bins forwards its arguments to build_photoz_bins."""
    z = np.linspace(0.0, 3.0, 7)
    nz = np.exp(-z)
    edges = np.array([0.0, 1.0, 2.0, 3.0])

    expected = {0: np.ones_like(z), 1: np.ones_like(z) * 2}
    rec = Recorder(return_value=expected)
    monkeypatch.setattr(api, "build_photoz_bins", rec)

    out = api.photoz_bins(z, nz, edges, sigma0=0.05, f_cat=0.02)

    assert rec.called is True
    assert rec.kwargs == {
        "z": z,
        "nz": nz,
        "bin_edges": edges,
        "sigma0": 0.05,
        "f_cat": 0.02,
    }
    assert out is expected


def test_specz_bins_forwards_to_build_specz_bins(monkeypatch):
    """Tests that specz_bins forwards its arguments to build_specz_bins."""
    z = np.linspace(0.0, 3.0, 7)
    nz = np.exp(-z)
    edges = np.array([0.0, 1.0, 2.0, 3.0])

    expected = {0: np.ones_like(z)}
    rec = Recorder(return_value=expected)
    monkeypatch.setattr(api, "build_specz_bins", rec)

    out = api.specz_bins(z, nz, edges, completeness=0.8, apply_specz_errors=True)

    assert rec.called is True
    assert rec.kwargs == {
        "z": z,
        "nz": nz,
        "bin_edges": edges,
        "completeness": 0.8,
        "apply_specz_errors": True,
    }
    assert out is expected


@pytest.mark.parametrize(
    "kind, expected_target",
    [
        ("photoz", "photoz_bins"),
        ("photo", "photoz_bins"),
        ("PHOTOZ", "photoz_bins"),
        ("PhOtO", "photoz_bins"),
        ("specz", "specz_bins"),
        ("spec", "specz_bins"),
        ("SPECZ", "specz_bins"),
    ],
)
def test_tomo_bins_routes_by_kind_alias_case_insensitive(
    monkeypatch, kind: str, expected_target: str
):
    """Tests that tomo_bins routes to photoz or specz bins based on kind
    alias."""
    z = np.linspace(0.0, 3.0, 7)
    nz = np.exp(-z)
    edges = np.array([0.0, 1.0, 2.0, 3.0])

    rec_photo = Recorder(return_value={"photo": True})
    rec_spec = Recorder(return_value={"spec": True})

    monkeypatch.setattr(api, "photoz_bins", rec_photo)
    monkeypatch.setattr(api, "specz_bins", rec_spec)

    params = {"a": 1, "b": "two"}
    out = api.tomo_bins(kind, z, nz, edges, params=params)

    if expected_target == "photoz_bins":
        assert rec_photo.called is True
        assert rec_spec.called is False
        assert rec_photo.args == (z, nz, edges)
        assert rec_photo.kwargs == params
        assert out == {"photo": True}
    else:
        assert rec_spec.called is True
        assert rec_photo.called is False
        assert rec_spec.args == (z, nz, edges)
        assert rec_spec.kwargs == params
        assert out == {"spec": True}


def test_tomo_bins_params_none_is_treated_as_empty_dict(monkeypatch):
    """Tests that tomo_bins passes an empty dict as params if params=None."""
    z = np.linspace(0.0, 1.0, 3)
    nz = np.exp(-z)
    edges = np.array([0.0, 1.0])

    rec_photo = Recorder(return_value={"ok": True})
    monkeypatch.setattr(api, "photoz_bins", rec_photo)

    out = api.tomo_bins("photoz", z, nz, edges, params=None)

    assert rec_photo.called is True
    assert rec_photo.args == (z, nz, edges)
    assert rec_photo.kwargs == {}
    assert out == {"ok": True}


def test_tomo_bins_unknown_kind_raises_value_error():
    """Tests that tomo_bins raises ValueError for unknown kind."""
    with pytest.raises(ValueError, match=r"kind must be 'photoz' or 'specz'"):
        api.tomo_bins(
            "nope",
            z=np.array([0.0, 1.0]),
            nz=np.array([1.0, 1.0]),
            bin_edges=np.array([0.0, 1.0]),
        )
