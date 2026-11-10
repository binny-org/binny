"""Unit tests for ``binny.surveys.survey_presets``."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from binny.surveys import survey_presets
from binny.surveys.survey_presets import (
    build_survey_tomography,
    list_survey_configs,
    load_survey_config,
    show_survey_config,
)
from binny.tomography import population_stats, shape_stats


def test_load_survey_config_normalizes_name(monkeypatch) -> None:
    """Tests that load_survey_config normalizes survey names."""
    called = {"path": None}

    def fake_load_config(path: str) -> dict[str, Any]:
        called["path"] = path
        return {"name": "lsst", "tomography": []}

    monkeypatch.setattr(
        survey_presets,
        "load_config",
        fake_load_config,
        raising=True,
    )

    cfg = load_survey_config(" LSST ")

    assert cfg == {"name": "lsst", "tomography": []}
    assert called["path"] == "lsst_survey_specs.yaml"


def test_load_survey_config_returns_plain_dict(monkeypatch) -> None:
    """Tests that load_survey_config returns a plain dictionary."""

    class MappingLike(dict):
        """Small dict subclass used to check dict conversion."""

    monkeypatch.setattr(
        survey_presets,
        "load_config",
        lambda path: MappingLike({"name": "desi", "tomography": []}),
        raising=True,
    )

    cfg = load_survey_config("desi")

    assert type(cfg) is dict
    assert cfg["name"] == "desi"
    assert cfg["tomography"] == []


def test_load_survey_config_propagates_missing_config(monkeypatch) -> None:
    """Tests that load_survey_config propagates missing-config errors."""

    def fake_load_config(path: str) -> dict[str, Any]:
        raise FileNotFoundError("missing config")

    monkeypatch.setattr(
        survey_presets,
        "load_config",
        fake_load_config,
        raising=True,
    )

    with pytest.raises(FileNotFoundError, match="missing config"):
        load_survey_config("not-a-survey")


def test_list_survey_configs_filters_suffix(monkeypatch) -> None:
    """Tests that list_survey_configs filters survey config filenames."""
    monkeypatch.setattr(
        survey_presets,
        "list_configs",
        lambda: [
            "lsst_survey_specs.yaml",
            "README.txt",
            "hsc_survey_specs.yaml",
            "foo.yaml",
            "roman_survey_specs.yml",
            "desi_survey_specs.yaml",
        ],
        raising=True,
    )

    assert list_survey_configs() == ["desi", "hsc", "lsst"]


def test_list_survey_configs_returns_unique_sorted_names(monkeypatch) -> None:
    """Tests that list_survey_configs returns sorted unique survey names."""
    monkeypatch.setattr(
        survey_presets,
        "list_configs",
        lambda: [
            "roman_survey_specs.yaml",
            "lsst_survey_specs.yaml",
            "roman_survey_specs.yaml",
            "desi_survey_specs.yaml",
        ],
        raising=True,
    )

    assert list_survey_configs() == ["desi", "lsst", "roman"]


def test_list_survey_configs_returns_empty_list_when_no_presets(monkeypatch) -> None:
    """Tests that list_survey_configs returns an empty list when no presets exist."""
    monkeypatch.setattr(
        survey_presets,
        "list_configs",
        lambda: [
            "README.txt",
            "example.yaml",
            "example_survey_specs.yml",
        ],
        raising=True,
    )

    assert list_survey_configs() == []


def test_show_survey_config_returns_yaml_text(monkeypatch) -> None:
    """Tests that show_survey_config returns YAML text."""
    cfg = {
        "name": "lsst",
        "tomography": [
            {
                "role": "source",
                "kind": "photoz",
            }
        ],
    }

    monkeypatch.setattr(
        survey_presets,
        "load_survey_config",
        lambda survey: cfg,
        raising=True,
    )

    text = show_survey_config("lsst", print_output=False)

    assert isinstance(text, str)
    assert yaml.safe_load(text) == cfg
    assert "name: lsst" in text
    assert "tomography:" in text


def test_show_survey_config_does_not_print_when_disabled(
    monkeypatch,
    capsys,
) -> None:
    """Tests that show_survey_config can suppress printing."""
    monkeypatch.setattr(
        survey_presets,
        "load_survey_config",
        lambda survey: {"name": "lsst", "tomography": []},
        raising=True,
    )

    text = show_survey_config("lsst", print_output=False)
    captured = capsys.readouterr()

    assert text
    assert captured.out == ""


def test_show_survey_config_prints_by_default(
    monkeypatch,
    capsys,
) -> None:
    """Tests that show_survey_config prints YAML by default."""
    monkeypatch.setattr(
        survey_presets,
        "load_survey_config",
        lambda survey: {"name": "roman", "tomography": []},
        raising=True,
    )

    text = show_survey_config("roman")
    captured = capsys.readouterr()

    assert captured.out == text + "\n"
    assert "name: roman" in captured.out
    assert "tomography:" in captured.out


def test_show_survey_config_forwards_survey_name(monkeypatch) -> None:
    """Tests that show_survey_config forwards the survey name to the loader."""
    called = {"survey": None}

    def fake_load_survey_config(survey: str) -> dict[str, Any]:
        called["survey"] = survey
        return {"name": "desi", "tomography": []}

    monkeypatch.setattr(
        survey_presets,
        "load_survey_config",
        fake_load_survey_config,
        raising=True,
    )

    show_survey_config("DESI", print_output=False)

    assert called["survey"] == "DESI"


# ---------------------------------------------------------------------------
# LSST reference regression tests
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REF_IO_PATH = _REPO_ROOT / "benchmarks" / "reference_io.py"
_REF_BASE = _REPO_ROOT / "tests" / "reference" / "data"

_spec = importlib.util.spec_from_file_location(
    "benchmarks_reference_io",
    _REF_IO_PATH,
)
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
    """List all available presets in the reference directory."""
    return sorted(p.name for p in _REF_BASE.iterdir() if p.is_dir())


def _parse_preset(preset: str) -> tuple[float, float, int]:
    """Parse preset name into (zmin, zmax, n)."""
    r, g = preset.split("__", 1)
    zmin, zmax = _RANGE_TYPES[r]
    n = _GRID_RESOLUTIONS[g]
    return zmin, zmax, n


def _stack_bins(bins: dict[int, np.ndarray] | dict[str, np.ndarray]) -> np.ndarray:
    """Stack bins into a single array."""
    keys = sorted(bins, key=lambda k: int(k))
    return np.stack(
        [np.asarray(bins[k], dtype=float) for k in keys],
        axis=0,
    )


@pytest.mark.parametrize("preset", _available_presets())
@pytest.mark.parametrize("year", [1, 10])
@pytest.mark.parametrize("role", ["lens", "source"])
def test_lsst_tomography_matches_reference(
    preset: str,
    year: int,
    role: str,
) -> None:
    """Test that LSST survey tomography matches reference datasets."""
    ref = load_benchmark(preset=preset, sample=role, year=year)
    ref_bins = np.asarray(ref["bins"], dtype=float)
    ref_z = np.asarray(ref["z"], dtype=float)

    zmin, zmax, n = _parse_preset(preset)
    z = np.linspace(zmin, zmax, n, dtype=float)

    np.testing.assert_allclose(z, ref_z, rtol=0.0, atol=0.0)

    result = build_survey_tomography(
        "lsst",
        role=role,
        year=str(year),
        z=z,
        include_survey_metadata=False,
        include_tomo_metadata=False,
    )

    got_bins = _stack_bins(dict(result.bins))

    assert got_bins.shape == ref_bins.shape

    np.testing.assert_allclose(
        got_bins,
        ref_bins,
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "fname",
    [
        "example_minimal_photoz.yaml",
        "example_full_photoz.yaml",
        "example_minimal_specz.yaml",
        "example_full_specz.yaml",
        "example_tabulated.yaml",
        "mock_predefined_config.yaml",
    ],
)
def test_example_configs_build_tomography(fname: str) -> None:
    """Test that shipped YAML examples build valid tomography."""
    config_resource = files("binny.surveys.configs").joinpath(fname)

    with as_file(config_resource) as path:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert isinstance(raw, Mapping)

        root = raw["survey"] if isinstance(raw.get("survey"), Mapping) else raw

        assert "tomography" in root
        assert isinstance(root["tomography"], list)
        assert root["tomography"]

        for entry in root["tomography"]:
            assert isinstance(entry, Mapping)

            kind = str(entry.get("kind", "photoz")).strip().lower()
            assert kind in {"photoz", "specz", "predefined"}

            if kind != "predefined":
                assert "z_grid" in root
                assert isinstance(entry.get("nz"), Mapping)

            assert isinstance(entry.get("bins"), Mapping)

            uncertainties = entry.get("uncertainties")
            if uncertainties is not None:
                assert isinstance(uncertainties, Mapping)

            selectors: dict[str, Any] = {}
            for selector in ("role", "year", "scenario", "sample"):
                value = entry.get(selector)
                if value is not None:
                    selectors[selector] = str(value)

            result = build_survey_tomography(
                config_file=path,
                include_survey_metadata=True,
                include_tomo_metadata=True,
                **selectors,
            )

            assert result.spec is not None
            assert result.spec["kind"] == kind

            z = result.z
            nz = result.nz
            built_bins = result.bins

            assert isinstance(z, np.ndarray)
            assert z.ndim == 1
            assert z.size >= 2
            assert np.all(np.isfinite(z))
            assert np.all(np.diff(z) > 0.0)

            assert isinstance(nz, np.ndarray)
            assert nz.shape == z.shape
            assert np.all(np.isfinite(nz))
            assert np.all(nz >= 0.0)

            assert isinstance(built_bins, Mapping)
            assert len(built_bins) >= 1

            for bin_index, curve in built_bins.items():
                assert isinstance(bin_index, int)
                assert isinstance(curve, np.ndarray)
                assert curve.shape == z.shape
                assert np.all(np.isfinite(curve))
                assert np.all(curve >= 0.0)

            shape_statistics = shape_stats(z, built_bins)
            assert isinstance(shape_statistics, dict)
            assert set(shape_statistics["per_bin"]) == set(built_bins)

            population_statistics = population_stats(
                built_bins,
                result.metadata,
            )
            assert isinstance(population_statistics, dict)
            assert set(population_statistics["fractions"]) == set(built_bins)


def test_build_survey_tomography_merges_nested_overrides():
    """Tests that nested overrides preserve other bin configuration fields."""
    base = build_survey_tomography(
        "lsst",
        role="lens",
        year="1",
    )

    assert base.spec is not None

    original_bins = base.spec["bins"]
    assert "n_bins" in original_bins

    result = build_survey_tomography(
        "lsst",
        role="lens",
        year="1",
        overrides={"bins": {"n_bins": original_bins["n_bins"]}},
    )

    assert result.spec is not None
    assert result.spec["bins"]["n_bins"] == original_bins["n_bins"]

    for key, value in original_bins.items():
        if key != "n_bins":
            assert result.spec["bins"][key] == value
