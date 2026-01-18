"""Tests that :class:`NZTomography` LSST YAML configs match reference .npy datasets."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from binny.api.nz_tomography import NZTomography

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REF_IO_PATH = _REPO_ROOT / "benchmarks" / "reference_io.py"
_LSST_YAML = _REPO_ROOT / "src" / "binny" / "surveys" / "configs" / "lsst_survey_specs.yaml"
_REF_BASE = _REPO_ROOT / "tests" / "reference" / "data"

_spec = importlib.util.spec_from_file_location("benchmarks_reference_io", _REF_IO_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load reference_io from {_REF_IO_PATH}")
_ref_io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref_io)
load_benchmark = _ref_io.load_benchmark


_RANGE_TYPES: dict[str, tuple[float, float]] = {
    "narrow": (0.0, 2.0),
    "default": (0.0, 3.5),
    "wide": (0.0, 5.0),
}

_GRID_RESOLUTIONS: dict[str, int] = {
    "coarse": 200,
    "default": 500,
    "fine": 1000,
    "superfine": 2000,
}


def _available_presets() -> list[str]:
    """Lists all available presets in the reference directory."""
    return sorted([p.name for p in _REF_BASE.iterdir() if p.is_dir()])


def _parse_preset(preset: str) -> tuple[float, float, int]:
    """Parses preset name into (zmin, zmax, n) tuple."""
    r, g = preset.split("__", 1)
    zmin, zmax = _RANGE_TYPES[r]
    n = _GRID_RESOLUTIONS[g]
    return zmin, zmax, n


def _stack_bins(
    bins: dict[int, np.ndarray] | dict[str, np.ndarray],
) -> np.ndarray:
    """Stacks bins into a single array."""
    keys = sorted(bins, key=lambda k: int(k))
    return np.stack([np.asarray(bins[k], dtype=float) for k in keys], axis=0)


@pytest.mark.parametrize("preset", _available_presets())
@pytest.mark.parametrize("year", [1, 10])
@pytest.mark.parametrize("role", ["lens", "source"])
def test_lsst_nztomography_matches_reference(preset: str, year: int, role: str) -> None:
    """Tests that LSST YAML config matches reference .npy dataset."""
    ref = load_benchmark(preset=preset, sample=role, year=year)
    ref_bins = np.asarray(ref["bins"], dtype=float)
    ref_z = np.asarray(ref["z"], dtype=float)

    zmin, zmax, n = _parse_preset(preset)
    z = np.linspace(zmin, zmax, n, dtype=float)
    np.testing.assert_allclose(z, ref_z, rtol=0.0, atol=0.0)

    t = NZTomography.from_config(
        _LSST_YAML,
        key="survey",
        role=role,
        year=str(year),
        z=z,
        include_survey_metadata=False,
    )

    t.build(include_metadata=False)

    got_bins = _stack_bins(dict(t.bins()))
    assert got_bins.shape == ref_bins.shape

    np.testing.assert_allclose(got_bins, ref_bins, rtol=1e-10, atol=1e-12)
