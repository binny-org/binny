"""Unit tests for ``binny.api.grids``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

import binny.api.grids as api


@dataclass
class Recorder:
    """Records calls to a function and returns a preset value."""

    return_value: Any = None
    called: bool = False
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __call__(self, **kwargs: Any) -> Any:
        self.called = True
        self.kwargs = dict(kwargs)
        return self.return_value


@pytest.mark.parametrize("kind", ["linear", "lin", "uniform", "LINEAR", "LiN"])
def test_grid_routes_to_linear_grid(monkeypatch, kind: str) -> None:
    """Tests that linear aliases route to linear_grid and forward kwargs."""
    rec_lin = Recorder(return_value="lin-ok")
    rec_log = Recorder(return_value="log-ok")

    monkeypatch.setattr(api, "linear_grid", rec_lin)
    monkeypatch.setattr(api, "log_grid", rec_log)

    out = api.grid(kind, x_min=0.0, x_max=1.0, n=3)

    assert out == "lin-ok"
    assert rec_lin.called is True
    assert rec_log.called is False
    assert rec_lin.kwargs == {"x_min": 0.0, "x_max": 1.0, "n": 3}


@pytest.mark.parametrize(
    "kind", ["log", "log_grid", "logarithmic", "geom", "geometric", "LOG", "GeOm"]
)
def test_grid_routes_to_log_grid(monkeypatch, kind: str) -> None:
    """Tests that log aliases route to log_grid and forward kwargs."""
    rec_lin = Recorder(return_value="lin-ok")
    rec_log = Recorder(return_value="log-ok")

    monkeypatch.setattr(api, "linear_grid", rec_lin)
    monkeypatch.setattr(api, "log_grid", rec_log)

    out = api.grid(kind, x_min=1.0, x_max=10.0, n=4)

    assert out == "log-ok"
    assert rec_log.called is True
    assert rec_lin.called is False
    assert rec_log.kwargs == {"x_min": 1.0, "x_max": 10.0, "n": 4}


@pytest.mark.parametrize("kind", ["", "unknown", "logg", "linear_gridder"])
def test_grid_unknown_kind_raises_value_error(kind: str) -> None:
    """Tests that unknown kinds raise a ValueError with a helpful message."""
    with pytest.raises(ValueError, match=r"Unknown grid kind"):
        api.grid(kind, x_min=0.0, x_max=1.0, n=3)
