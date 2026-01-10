"""Tests for the LSST tomography API against reference data."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import binny as bn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REF_IO_PATH = _REPO_ROOT / "benchmarks" / "reference_io.py"

_spec = importlib.util.spec_from_file_location("benchmarks_reference_io", _REF_IO_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load reference_io from {_REF_IO_PATH}")

_ref_io = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref_io)

load_benchmark = _ref_io.load_benchmark


def _available_presets() -> list[str]:
    base = _REPO_ROOT / "tests" / "reference" / "data"
    return sorted([p.name for p in base.iterdir() if p.is_dir()])


@pytest.mark.parametrize("preset", _available_presets())
@pytest.mark.parametrize("year", [1, 10])
@pytest.mark.parametrize("sample", ["lens", "source"])
def test_lsst_tomography_matches_reference(preset: str, year: int, sample: str) -> None:
    ref = load_benchmark(preset=preset, sample=sample, year=year)
    z = ref["z"]

    got_dict = bn.lsst_tomography(z=z, year=year, sample=sample)
    got_bins = np.stack([got_dict[i] for i in sorted(got_dict)], axis=0)

    assert got_bins.shape == ref["bins"].shape

    try:
        np.testing.assert_allclose(got_bins, ref["bins"], rtol=1e-10, atol=1e-12)
    except AssertionError:
        print(f"\nPRESET={preset} YEAR={year} SAMPLE={sample}")

        ref_integrals = [float(np.trapezoid(b, x=z)) for b in ref["bins"]]
        got_integrals = [float(np.trapezoid(b, x=z)) for b in got_bins]

        print("REF per-bin integrals:", ref_integrals)
        print("GOT per-bin integrals:", got_integrals)

        print("REF sum integral:", float(np.trapezoid(ref["bins"].sum(axis=0), x=z)))
        print("GOT sum integral:", float(np.trapezoid(got_bins.sum(axis=0), x=z)))

        print("REF max:", float(ref["bins"].max()), "GOT max:", float(got_bins.max()))
        raise
