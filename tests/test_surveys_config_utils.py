"""Unit tests for binny.surveys.config_utils."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from binny.surveys.config_utils import (
    _build_parent_nz,
    _extract_survey_meta,
    _extract_z_grid,
    _iter_tomography_entries,
    _load_yaml_mapping,
    _parse_bins,
    _parse_entry,
    _require_mapping,
    _require_single,
    _resolve_config_entry,
    _select_entries,
    _survey_meta,
    _tabulated_params_from_config,
    config_path,
    list_configs,
)


@pytest.fixture
def minimal_cfg() -> dict[str, Any]:
    """Tests that a minimal valid config mapping is available."""
    return {
        "name": "test",
        "z_grid": {"start": 0.0, "stop": 1.0, "n": 11},
        "survey_meta": {"a": 1},
        "tomography": [
            {
                "role": "lens",
                "year": "1",
                "kind": "photoz",
                "nz": {"model": "smail", "params": {}},
                "bins": {"scheme": "equidistant", "n_bins": 2},
            }
        ],
    }


def test_list_configs_returns_yaml_filenames() -> None:
    """Tests that list_configs returns YAML filenames."""
    names = list_configs()
    assert isinstance(names, list)
    assert all(name.endswith((".yaml", ".yml")) for name in names)


def test_config_path_resolves_existing_file() -> None:
    """Tests that config_path resolves shipped configs."""
    name = list_configs()[0]
    path = config_path(name)
    assert isinstance(path, Path)
    assert path.exists()


def test_load_yaml_mapping_requires_mapping(tmp_path) -> None:
    """Tests that _load_yaml_mapping rejects non-mapping YAML."""
    p = tmp_path / "bad.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Config root must be a mapping"):
        _load_yaml_mapping(p)


def test_require_mapping_accepts_mapping() -> None:
    """Tests that _require_mapping returns mappings unchanged."""
    m = {"a": 1}
    out = _require_mapping(m, what="x")
    assert out is m


def test_require_mapping_rejects_non_mapping() -> None:
    """Tests that _require_mapping raises for non-mappings."""
    with pytest.raises(ValueError, match="must be a mapping"):
        _require_mapping(3, what="x")


def test_resolve_config_entry_loads_yaml(tmp_path) -> None:
    """Tests that _resolve_config_entry loads YAML configs."""
    cfg = {"name": "x", "tomography": []}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    out, key = _resolve_config_entry(config_file=p, key=None)
    assert isinstance(out, Mapping)
    assert key == "x"


def test_resolve_config_entry_rejects_key_argument(tmp_path) -> None:
    """Tests that _resolve_config_entry rejects top-level key usage."""
    cfg = {"tomography": []}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="does not support top-level keys"):
        _resolve_config_entry(config_file=p, key="x")


def test_extract_z_grid_uses_config(minimal_cfg: Mapping[str, Any]) -> None:
    """Tests that _extract_z_grid uses cfg.z_grid when present."""
    z = _extract_z_grid(minimal_cfg, z=None)
    assert isinstance(z, np.ndarray)
    assert z.size == 11


def test_extract_z_grid_uses_override(minimal_cfg: Mapping[str, Any]) -> None:
    """Tests that _extract_z_grid prefers override z."""
    z0 = np.linspace(0.0, 2.0, 5)
    z = _extract_z_grid(minimal_cfg, z=z0)
    assert np.all(z == z0)


def test_extract_survey_meta_returns_mapping(
    minimal_cfg: Mapping[str, Any],
) -> None:
    """Tests that _extract_survey_meta returns a dict when present."""
    meta = _extract_survey_meta(minimal_cfg)
    assert isinstance(meta, dict)
    assert meta["a"] == 1


def test_iter_tomography_entries_returns_list(
    minimal_cfg: Mapping[str, Any],
) -> None:
    """Tests that _iter_tomography_entries returns a list of mappings."""
    entries = _iter_tomography_entries(minimal_cfg)
    assert isinstance(entries, list)
    assert len(entries) == 1
    assert isinstance(entries[0], Mapping)


def test_select_entries_filters_by_role_and_year(
    minimal_cfg: Mapping[str, Any],
) -> None:
    """Tests that _select_entries filters entries correctly."""
    entries = _iter_tomography_entries(minimal_cfg)

    sel = _select_entries(entries, role="lens", year="1")
    assert len(sel) == 1

    sel = _select_entries(entries, role="source", year="1")
    assert sel == []


def test_require_single_rejects_ambiguity(
    minimal_cfg: Mapping[str, Any],
) -> None:
    """Tests that _require_single raises when entries are ambiguous."""
    entries = _iter_tomography_entries(minimal_cfg) * 2
    with pytest.raises(ValueError, match="ambiguous"):
        _require_single(entries, what="entry")


def test_parse_bins_with_scheme() -> None:
    """Tests that _parse_bins accepts scheme-based bins."""
    bins = _parse_bins({"scheme": "equidistant", "n_bins": 3})
    assert bins["scheme"] == "equidistant"
    assert bins["n_bins"] == 3


def test_parse_bins_with_edges() -> None:
    """Tests that _parse_bins accepts explicit edges."""
    bins = _parse_bins({"edges": [0.0, 0.5, 1.0]})
    assert "edges" in bins
    assert bins["edges"].size == 3


def test_parse_entry_handles_optional_fields() -> None:
    """Tests that _parse_entry handles optional fields correctly."""
    entry = {
        "kind": "photoz",
        "nz": {"model": "smail", "params": {}},
        "bins": {"scheme": "equidistant", "n_bins": 2},
        "n_gal_arcmin2": None,
    }

    spec = _parse_entry(entry)
    assert spec["kind"] == "photoz"
    assert spec["n_gal_arcmin2"] is None
    assert isinstance(spec["bins"], dict)


def test_survey_meta_builds_expected_mapping(
    minimal_cfg: Mapping[str, Any],
) -> None:
    """Tests that _survey_meta returns standardized metadata."""
    meta = _survey_meta(
        cfg=minimal_cfg,
        resolved_key="k",
        role="lens",
        year="1",
    )

    assert meta["survey"] == "test"
    assert meta["key"] == "k"
    assert meta["role"] == "lens"
    assert meta["year"] == "1"


def test_resolve_config_entry_rejects_missing_tomography(tmp_path) -> None:
    """Tests that _resolve_config_entry rejects configs without tomography."""
    cfg = {"name": "x"}
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain a 'tomography' list"):
        _resolve_config_entry(config_file=p, key=None)


def test_extract_z_grid_uses_package_default_when_missing() -> None:
    """Tests that _extract_z_grid uses the package default when z_grid missing."""
    cfg = {"tomography": []}
    z = _extract_z_grid(cfg, z=None)
    assert isinstance(z, np.ndarray)
    assert z.ndim == 1
    assert z.size == 301
    assert z[0] == pytest.approx(0.0)
    assert z[-1] == pytest.approx(3.0)


def test_extract_z_grid_rejects_missing_keys() -> None:
    """Tests that _extract_z_grid rejects incomplete z_grid mappings."""
    cfg = {"tomography": [], "z_grid": {"start": 0.0, "stop": 1.0}}  # missing n
    with pytest.raises(ValueError, match="z_grid must contain keys"):
        _extract_z_grid(cfg, z=None)


def test_extract_survey_meta_returns_none_when_missing() -> None:
    """Tests that _extract_survey_meta returns None when survey_meta missing."""
    cfg = {"tomography": []}
    assert _extract_survey_meta(cfg) is None


def test_build_parent_nz_rejects_missing_model() -> None:
    """Tests that _build_parent_nz rejects nz blocks missing model."""
    z = np.linspace(0.0, 1.0, 5)
    entry = {"nz": {"params": {}}}

    with pytest.raises(ValueError, match="must contain a 'model'"):
        _build_parent_nz(entry, z)


def test_build_parent_nz_rejects_non_mapping_params() -> None:
    """Tests that _build_parent_nz rejects non-mapping nz.params."""
    z = np.linspace(0.0, 1.0, 5)
    entry = {"nz": {"model": "smail", "params": [1, 2, 3]}}

    with pytest.raises(ValueError, match="nz.params must be a mapping"):
        _build_parent_nz(entry, z)


def test_iter_tomography_entries_rejects_non_list() -> None:
    """Tests that _iter_tomography_entries rejects non-list tomography blocks."""
    cfg = {"tomography": "nope"}
    with pytest.raises(ValueError, match="tomography must be a list of mappings"):
        _iter_tomography_entries(cfg)


def test_parse_bins_rejects_edges_with_scheme() -> None:
    """Tests that _parse_bins rejects edges mixed with scheme/n_bins/range."""
    with pytest.raises(ValueError, match="if 'edges' is provided"):
        _parse_bins({"edges": [0.0, 1.0], "scheme": "equidistant", "n_bins": 2})


def test_parse_bins_rejects_bad_edges_shape() -> None:
    """Tests that _parse_bins rejects non-1D or too-short edges."""
    with pytest.raises(ValueError, match="bins.edges must be a 1D sequence"):
        _parse_bins({"edges": [0.0]})


def test_parse_entry_rejects_bad_kind() -> None:
    """Tests that _parse_entry rejects unsupported kind values."""
    with pytest.raises(ValueError, match="kind must be 'photoz' or 'specz'"):
        _parse_entry(
            {
                "kind": "nope",
                "nz": {"model": "smail", "params": {}},
                "bins": {"scheme": "x", "n_bins": 2},
            }
        )


def test_parse_entry_rejects_non_mapping_uncertainties() -> None:
    """Tests that _parse_entry rejects non-mapping uncertainties."""
    with pytest.raises(ValueError, match="uncertainties must be a mapping"):
        _parse_entry(
            {
                "kind": "photoz",
                "nz": {"model": "smail", "params": {}},
                "bins": {"scheme": "equidistant", "n_bins": 2},
                "uncertainties": [1, 2],
            }
        )


def test_tabulated_params_from_config_accepts_inline_arrays() -> None:
    """Tests that tabulated inline arrays are passed through."""
    z_input = [0.0, 0.5, 1.0]
    nz_input = [0.0, 2.0, 0.0]

    params = _tabulated_params_from_config(
        {
            "model": "tabulated",
            "z_input": z_input,
            "nz_input": nz_input,
        }
    )

    assert params["z_input"] == z_input
    assert params["nz_input"] == nz_input


def test_tabulated_params_from_config_rejects_partial_inline_arrays() -> None:
    """Tests that tabulated inline arrays require both z and n(z)."""
    with pytest.raises(ValueError, match="requires both 'z_input' and 'nz_input'"):
        _tabulated_params_from_config(
            {
                "model": "tabulated",
                "z_input": [0.0, 1.0],
            }
        )


def test_tabulated_params_from_config_reads_absolute_source_file(tmp_path) -> None:
    """Tests that tabulated source files are read from absolute paths."""
    p = tmp_path / "nz.txt"
    np.savetxt(
        p,
        np.array(
            [
                [0.0, 0.0],
                [0.5, 2.0],
                [1.0, 0.0],
            ]
        ),
    )

    params = _tabulated_params_from_config(
        {
            "model": "tabulated",
            "source": {
                "path": str(p),
                "z_col": 0,
                "nz_col": 1,
                "skiprows": 0,
            },
        }
    )

    assert np.allclose(params["z_input"], [0.0, 0.5, 1.0])
    assert np.allclose(params["nz_input"], [0.0, 2.0, 0.0])


def test_tabulated_params_from_config_rejects_source_without_path() -> None:
    """Tests that tabulated source files require a path."""
    with pytest.raises(ValueError, match="source must contain a 'path' field"):
        _tabulated_params_from_config(
            {
                "model": "tabulated",
                "source": {
                    "z_col": 0,
                    "nz_col": 1,
                },
            }
        )


def test_build_parent_nz_accepts_inline_tabulated_nz() -> None:
    """Tests that _build_parent_nz builds inline tabulated n(z)."""
    z = np.linspace(0.0, 1.0, 5)

    entry = {
        "nz": {
            "model": "tabulated",
            "z_input": [0.0, 0.5, 1.0],
            "nz_input": [0.0, 2.0, 0.0],
            "params": {"normalize": False},
        }
    }

    nz = _build_parent_nz(entry, z)

    assert isinstance(nz, np.ndarray)
    assert nz.shape == z.shape
    assert np.all(nz >= 0.0)
    assert nz[0] == pytest.approx(0.0)
    assert nz[-1] == pytest.approx(0.0)


def test_parse_entry_accepts_tabulated_source_block() -> None:
    """Tests that _parse_entry preserves tabulated source blocks."""
    entry = {
        "role": "lens",
        "year": "lrg",
        "kind": "specz",
        "nz": {
            "model": "tabulated",
            "source": {
                "path": "desi_lrg_nz.txt",
                "z_col": 0,
                "nz_col": 1,
                "skiprows": 0,
            },
            "params": {"normalize": True},
        },
        "bins": {"edges": [0.4, 1.0]},
    }

    spec = _parse_entry(entry)

    assert spec["role"] == "lens"
    assert spec["year"] == "lrg"
    assert spec["kind"] == "specz"
    assert spec["nz"]["model"] == "tabulated"
    assert spec["nz"]["source"]["path"] == "desi_lrg_nz.txt"
    assert np.allclose(spec["bins"]["edges"], [0.4, 1.0])
