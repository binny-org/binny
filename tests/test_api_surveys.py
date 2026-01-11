"""Unit tests for ``binny.api.surveys`` module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

import binny.api.surveys as api


@dataclass
class Recorder:
    """Class that records calls to a function and returns its arguments."""

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


def test_lsst_tomography_forwards_all_kwargs_and_returns(monkeypatch):
    """Tests that the wrapper forwards all kwargs and returns the result."""
    z = np.linspace(0.0, 3.0, 7)

    expected = {
        0: np.array([1.0, 2.0, 3.0]),
        1: np.array([4.0, 5.0, 6.0]),
    }

    rec = Recorder(return_value=expected)
    monkeypatch.setattr(api, "_lsst_tomography", rec)

    out = api.lsst_tomography(
        z=z,
        year=10,
        sample="source",
        config_file="my_config.yaml",
        normalize_input=False,
        normalize_bins=True,
    )

    assert rec.called is True
    assert rec.args == ()
    assert rec.kwargs == {
        "z": z,
        "year": 10,
        "sample": "source",
        "config_file": "my_config.yaml",
        "normalize_input": False,
        "normalize_bins": True,
    }

    assert out is expected


def test_lsst_tomography_uses_default_kwargs(monkeypatch):
    """Tests that the wrapper uses default kwargs for missing ones."""
    z = np.linspace(0.0, 3.0, 7)

    expected = {0: np.array([1.0])}
    rec = Recorder(return_value=expected)
    monkeypatch.setattr(api, "_lsst_tomography", rec)

    out = api.lsst_tomography(z=z, year=1, sample="lens")

    assert rec.called is True
    assert rec.kwargs == {
        "z": z,
        "year": 1,
        "sample": "lens",
        "config_file": "lsst_survey_specs.yaml",
        "normalize_input": True,
        "normalize_bins": True,
    }
    assert out is expected


def test_lsst_tomography_propagates_exceptions(monkeypatch):
    """Tests that the wrapper propagates exceptions from the wrapped function."""

    def _boom(**kwargs: Any):
        """Function that always raises an exception."""
        raise RuntimeError("nope")

    monkeypatch.setattr(api, "_lsst_tomography", _boom)

    with pytest.raises(RuntimeError, match="nope"):
        api.lsst_tomography(z=np.linspace(0.0, 1.0, 3), year=1, sample="lens")


@pytest.mark.parametrize("sample", ["bad", "SOURCE", "", None])
def test_lsst_tomo_invalid_sample_is_not_validated_by_wrapper(monkeypatch, sample):
    """Tests that the wrapper does not validate the sample argument."""
    z = np.linspace(0.0, 1.0, 3)

    rec = Recorder(return_value={})
    monkeypatch.setattr(api, "_lsst_tomography", rec)

    api.lsst_tomography(z=z, year=1, sample=sample)

    assert rec.called is True
    assert rec.kwargs["sample"] is sample
