"""Unit tests for binny.api.nz_tomography."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

import binny.surveys.config_utils as cu
from binny.api.nz_tomography import NZTomography


def _fake_bins(z: np.ndarray, n_bins: int = 2) -> dict[int, np.ndarray]:
    # Simple deterministic bins: alternating masks so bins are nontrivial.
    bins: dict[int, np.ndarray] = {}
    for i in range(n_bins):
        b = np.zeros_like(z, dtype=float)
        b[i::n_bins] = 1.0
        bins[i] = b
    return bins


def _install_fake_builder_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install fake modules so NZTomography._resolve_builder can import them."""
    photoz_mod = types.ModuleType("binny.nz_tomo.photoz")
    specz_mod = types.ModuleType("binny.nz_tomo.specz")

    def build_photoz_bins(*, z, nz, include_metadata: bool = False, **kwargs):
        """Fake build_photoz_bins that returns two bins."""
        bins = _fake_bins(np.asarray(z), n_bins=int(kwargs.get("n_bins", 2)))
        if include_metadata:
            meta = {"builder": "photoz", "kwargs": dict(kwargs), "n_parent": float(np.sum(nz))}
            return bins, meta
        return bins

    def build_specz_bins(*, z, nz, include_metadata: bool = False, **kwargs):
        """Fake build_specz_bins that returns a single bin."""
        bins = _fake_bins(np.asarray(z), n_bins=int(kwargs.get("n_bins", 3)))
        if include_metadata:
            meta = {"builder": "specz", "kwargs": dict(kwargs), "n_parent": float(np.sum(nz))}
            return bins, meta
        return bins

    photoz_mod.build_photoz_bins = build_photoz_bins
    specz_mod.build_specz_bins = build_specz_bins

    monkeypatch.setitem(sys.modules, "binny.nz_tomo.photoz", photoz_mod)
    monkeypatch.setitem(sys.modules, "binny.nz_tomo.specz", specz_mod)


def test_clear_resets_state():
    """Tests that clear resets the state."""
    t = NZTomography()
    t._parent = {"z": np.array([0.0]), "nz": np.array([1.0]), "survey_meta": None}
    t._state = {"tomo_spec": {"kind": "photoz", "bins": {}}, "bins": {}, "tomo_meta": None}

    t.clear()
    assert t._parent is None
    assert t._state is None


def test_shape_stats_raises_before_build():
    """Tests that shape_stats raises before build_bins is called."""
    t = NZTomography()
    with pytest.raises(ValueError, match=r"Call build_bins"):
        t.shape_stats()


def test_population_stats_raises_before_build():
    """Tests that population_stats raises before build_bins is called."""
    t = NZTomography()
    with pytest.raises(ValueError, match=r"Call build_bins"):
        t.population_stats()


def test_cross_bin_stats_raises_before_build():
    """Tests that cross_bin_stats raises before build_bins is called."""
    t = NZTomography()
    with pytest.raises(ValueError, match=r"Call build_bins"):
        t.cross_bin_stats(overlap={})


def test_population_stats_raises_when_no_tomo_meta_cached(monkeypatch):
    """Tests that population_stats raises when no tomo_meta is cached."""
    _install_fake_builder_modules(monkeypatch)

    monkeypatch.setattr(cu, "_parse_entry", lambda e: dict(e), raising=True)
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 2}, raising=True)

    t = NZTomography()
    z = np.linspace(0, 1, 11)
    nz = np.ones_like(z)

    t.build_bins(z=z, nz=nz, tomo_spec={"kind": "photoz", "bins": {"scheme": "x", "n_bins": 2}})
    with pytest.raises(ValueError, match=r"include_tomo_metadata=True"):
        t.population_stats()


def test_build_bins_from_arrays_basic(monkeypatch):
    """Tests that build_bins from arrays works with basic inputs."""
    _install_fake_builder_modules(monkeypatch)

    parse_calls = {"n": 0}

    def fake_parse_entry(e):
        """Fake parse_entry that counts calls."""
        parse_calls["n"] += 1
        out = dict(e)
        out.setdefault("role", "source")
        out.setdefault("year", "1")
        return out

    monkeypatch.setattr(cu, "_parse_entry", fake_parse_entry, raising=True)

    def fake_builder_kwargs_from_spec(spec):
        """Fake builder_kwargs_from_spec that ignores spec."""
        return {"n_bins": int(spec["bins"].get("n_bins", 2))}

    monkeypatch.setattr(
        cu, "_builder_kwargs_from_spec", fake_builder_kwargs_from_spec, raising=True
    )

    t = NZTomography()
    z = np.linspace(0, 1, 11)
    nz = np.linspace(1, 2, 11)

    payload = t.build_bins(
        z=z,
        nz=nz,
        tomo_spec={"kind": "photoz", "bins": {"scheme": "whatever", "n_bins": 2}},
        include_tomo_metadata=False,
        include_survey_metadata=False,
    )

    assert parse_calls["n"] == 1
    assert np.allclose(payload["z"], z)
    assert np.allclose(payload["nz"], nz)
    assert payload["spec"]["kind"] == "photoz"
    assert set(payload["bins"].keys()) == {0, 1}
    assert payload["tomo_meta"] is None
    assert payload["survey_meta"] is None

    assert t._state is not None
    assert t._state["bins"] == payload["bins"]
    assert t._state["tomo_meta"] is None


def test_build_bins_from_arrays_sets_stub_nz_block(monkeypatch):
    """Tests that build_bins from arrays sets a stub nz block."""
    _install_fake_builder_modules(monkeypatch)

    captured_entry = {"value": None}

    def fake_parse_entry(e):
        """Fake parse_entry that captures the entry."""
        captured_entry["value"] = dict(e)
        out = dict(e)
        out.setdefault("role", "source")
        out.setdefault("year", "1")
        return out

    monkeypatch.setattr(cu, "_parse_entry", fake_parse_entry, raising=True)
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 2}, raising=True)

    t = NZTomography()
    z = np.linspace(0, 1, 11)
    nz = np.ones_like(z)

    # tomo_spec intentionally has no "nz" block
    t.build_bins(z=z, nz=nz, tomo_spec={"kind": "photoz", "bins": {"scheme": "x", "n_bins": 2}})

    assert isinstance(captured_entry["value"], dict)
    assert captured_entry["value"].get("nz") == {"model": "arrays"}


def test_build_bins_kind_argument_overrides_spec(monkeypatch):
    """Tests that build_bins kind argument overrides spec."""
    _install_fake_builder_modules(monkeypatch)
    monkeypatch.setattr(
        cu, "_parse_entry", lambda e: {**dict(e), "role": "lens", "year": "10"}, raising=True
    )
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 3}, raising=True)

    t = NZTomography()
    z = np.linspace(0, 1, 11)
    nz = np.ones_like(z)

    payload = t.build_bins(
        z=z,
        nz=nz,
        tomo_spec={"kind": "photoz", "bins": {"scheme": "x", "n_bins": 3}},
        kind="SPEcz",
        include_tomo_metadata=True,
    )

    assert payload["spec"]["kind"] == "specz"
    assert payload["tomo_meta"]["builder"] == "specz"
    assert set(payload["bins"].keys()) == {0, 1, 2}


def test_build_bins_overrides_merge_nested_mappings(monkeypatch):
    """Tests that build_bins merges overrides into nested mappings."""
    _install_fake_builder_modules(monkeypatch)
    monkeypatch.setattr(
        cu, "_parse_entry", lambda e: {**dict(e), "role": "source", "year": "1"}, raising=True
    )

    def fake_builder_kwargs_from_spec(spec):
        """Fake builder_kwargs_from_spec that merges overrides into spec."""
        return {"n_bins": int(spec["bins"]["n_bins"])}

    monkeypatch.setattr(
        cu, "_builder_kwargs_from_spec", fake_builder_kwargs_from_spec, raising=True
    )

    t = NZTomography()
    z = np.linspace(0, 1, 11)
    nz = np.ones_like(z)

    payload = t.build_bins(
        z=z,
        nz=nz,
        tomo_spec={"kind": "photoz", "bins": {"scheme": "x", "n_bins": 2}},
        overrides={"bins": {"n_bins": 4}},
    )
    assert set(payload["bins"].keys()) == {0, 1, 2, 3}


def test_build_bins_requires_bins_mapping(monkeypatch):
    """Tests that build_bins requires a 'bins' mapping."""
    _install_fake_builder_modules(monkeypatch)
    monkeypatch.setattr(cu, "_parse_entry", lambda e: dict(e), raising=True)

    t = NZTomography()
    z = np.linspace(0, 1, 11)
    nz = np.ones_like(z)

    with pytest.raises(ValueError, match=r"must contain a 'bins' mapping"):
        t.build_bins(z=z, nz=nz, tomo_spec={"kind": "photoz"})


def test_build_bins_from_cfg_mapping(monkeypatch):
    """Tests that build_bins works with a cfg mapping."""
    _install_fake_builder_modules(monkeypatch)

    monkeypatch.setattr(cu, "_require_mapping", lambda x, what=None: x, raising=True)
    monkeypatch.setattr(
        cu, "_extract_z_grid", lambda cfg, z: np.asarray(z, dtype=float), raising=True
    )

    entry = {
        "role": "lens",
        "year": "1",
        "kind": "photoz",
        "nz": {"model": "smail"},
        "bins": {"n_bins": 2},
    }
    monkeypatch.setattr(cu, "_iter_tomography_entries", lambda cfg: [entry], raising=True)
    monkeypatch.setattr(
        cu, "_select_entries", lambda entries, role=None, year=None: entries, raising=True
    )
    monkeypatch.setattr(cu, "_require_single", lambda matches, what=None: matches[0], raising=True)
    monkeypatch.setattr(cu, "_parse_entry", lambda e: dict(e), raising=True)
    monkeypatch.setattr(
        cu, "_build_parent_nz", lambda entry, z: np.ones_like(z, dtype=float), raising=True
    )
    monkeypatch.setattr(
        cu,
        "_survey_meta",
        lambda cfg, resolved_key, role, year: {
            "resolved_key": resolved_key,
            "role": role,
            "year": year,
        },
        raising=True,
    )

    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 2}, raising=True)

    t = NZTomography()
    z = np.linspace(0, 1, 11)
    payload = t.build_bins(cfg={"tomography": [entry]}, z=z, include_survey_metadata=True)

    assert np.allclose(payload["z"], z)
    assert np.allclose(payload["nz"], np.ones_like(z))
    assert payload["survey_meta"]["role"] == "lens"
    assert payload["survey_meta"]["year"] == "1"
    assert set(payload["bins"].keys()) == {0, 1}


def test_build_bins_from_config_file_delegates_to_resolver(tmp_path: Path, monkeypatch):
    """Tests that build_bins delegates to _resolve_config_entry."""
    _install_fake_builder_modules(monkeypatch)

    p = tmp_path / "survey.yaml"
    p.write_text("dummy: true\n", encoding="utf-8")

    cfg_map = {
        "tomography": [{"role": "source", "year": "1", "kind": "photoz", "bins": {"n_bins": 2}}]
    }
    monkeypatch.setattr(
        cu, "_resolve_config_entry", lambda config_file, key=None: (cfg_map, "k0"), raising=True
    )

    monkeypatch.setattr(cu, "_require_mapping", lambda x, what=None: x, raising=True)
    monkeypatch.setattr(
        cu, "_extract_z_grid", lambda cfg, z: np.asarray(z, dtype=float), raising=True
    )
    monkeypatch.setattr(
        cu, "_iter_tomography_entries", lambda cfg: list(cfg["tomography"]), raising=True
    )
    monkeypatch.setattr(
        cu, "_select_entries", lambda entries, role=None, year=None: entries, raising=True
    )
    monkeypatch.setattr(cu, "_require_single", lambda matches, what=None: matches[0], raising=True)
    monkeypatch.setattr(cu, "_parse_entry", lambda e: dict(e), raising=True)
    monkeypatch.setattr(
        cu, "_build_parent_nz", lambda entry, z: np.ones_like(z, dtype=float), raising=True
    )
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 2}, raising=True)

    t = NZTomography()
    z = np.linspace(0, 1, 11)
    payload = t.build_bins(config_file=p, z=z)
    assert set(payload["bins"].keys()) == {0, 1}


def test_build_survey_bins_sets_include_tomo_metadata_when_include_stats(monkeypatch):
    """Test build_survey_bins forces include_tomo_metadata when include_stats=True."""
    _install_fake_builder_modules(monkeypatch)

    monkeypatch.setattr(
        cu,
        "_resolve_config_entry",
        lambda config_file, key=None: ({"tomography": [{}]}, "k"),
        raising=True,
    )
    monkeypatch.setattr(cu, "_require_mapping", lambda x, what=None: x, raising=True)
    monkeypatch.setattr(
        cu, "_extract_z_grid", lambda cfg, z: np.asarray(z, dtype=float), raising=True
    )
    monkeypatch.setattr(
        cu,
        "_iter_tomography_entries",
        lambda cfg: [{"role": "source", "year": "1", "kind": "photoz", "bins": {"n_bins": 2}}],
        raising=True,
    )
    monkeypatch.setattr(
        cu, "_select_entries", lambda entries, role=None, year=None: entries, raising=True
    )
    monkeypatch.setattr(cu, "_require_single", lambda matches, what=None: matches[0], raising=True)
    monkeypatch.setattr(cu, "_parse_entry", lambda e: dict(e), raising=True)
    monkeypatch.setattr(
        cu, "_build_parent_nz", lambda entry, z: np.ones_like(z, dtype=float), raising=True
    )
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 2}, raising=True)

    monkeypatch.setattr(
        "binny.api.nz_tomography._shape_stats",
        lambda *, z, bins, **k: {"ok": True, "n_bins": len(bins)},
        raising=True,
    )
    monkeypatch.setattr(
        "binny.api.nz_tomography._population_stats",
        lambda *, bins, metadata, **k: {"ok": True, "builder": metadata.get("builder")},
        raising=True,
    )

    t = NZTomography()
    z = np.linspace(0, 1, 11)
    dummy_file = Path("dummy.yaml")
    payload = t.build_survey_bins(
        "LSST",
        config_file=dummy_file,
        z=z,
        include_stats=True,
        include_tomo_metadata=False,
    )

    assert payload["survey"] == "lsst"
    assert payload["tomo_meta"] is not None
    assert payload["shape_stats"]["ok"] is True
    assert payload["population_stats"]["ok"] is True


def test_cross_bin_stats_skips_all_when_all_none(monkeypatch):
    """Tests that cross_bin_stats returns an empty dict when all inputs are None."""
    _install_fake_builder_modules(monkeypatch)
    monkeypatch.setattr(
        cu, "_parse_entry", lambda e: {**dict(e), "role": "source", "year": "1"}, raising=True
    )
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 2}, raising=True)

    t = NZTomography()
    z = np.linspace(0, 1, 11)
    nz = np.ones_like(z)
    t.build_bins(z=z, nz=nz, tomo_spec={"kind": "photoz", "bins": {"scheme": "x", "n_bins": 2}})

    out = t.cross_bin_stats()
    assert out == {}


def test_cross_bin_stats_leakage_requires_bin_edges(monkeypatch):
    """Tests that cross_bin_stats requires leakage_bin_edges."""
    _install_fake_builder_modules(monkeypatch)
    monkeypatch.setattr(
        cu, "_parse_entry", lambda e: {**dict(e), "role": "source", "year": "1"}, raising=True
    )
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 2}, raising=True)

    t = NZTomography()
    z = np.linspace(0, 1, 11)
    nz = np.ones_like(z)
    t.build_bins(z=z, nz=nz, tomo_spec={"kind": "photoz", "bins": {"scheme": "x", "n_bins": 2}})

    with pytest.raises(ValueError, match=r"leakage requires leakage=\{'bin_edges':"):
        t.cross_bin_stats(leakage={"foo": 1})


def test_cross_bin_stats_delegates_to_bin_similarity(monkeypatch):
    """Tests that cross_bin_stats delegates to bin_similarity."""
    _install_fake_builder_modules(monkeypatch)
    monkeypatch.setattr(
        cu, "_parse_entry", lambda e: {**dict(e), "role": "source", "year": "1"}, raising=True
    )
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 2}, raising=True)

    calls: dict[str, dict] = {"overlap": {}, "pairs": {}, "leakage": {}, "pearson": {}}

    def fake_overlap(z, bins, **kw):
        """Fake overlap matrix that returns a constant matrix."""
        calls["overlap"] = {"z": np.asarray(z), "bins": bins, "kw": dict(kw)}
        return np.eye(len(bins))

    def fake_pairs(z, bins, **kw):
        """Fake pairs matrix that returns a constant matrix."""
        calls["pairs"] = {"z": np.asarray(z), "bins": bins, "kw": dict(kw)}
        return [(0, 1)]

    def fake_leakage(z, bins, bin_edges, **kw):
        """Fake leakage matrix that returns a constant matrix."""
        calls["leakage"] = {
            "z": np.asarray(z),
            "bins": bins,
            "bin_edges": bin_edges,
            "kw": dict(kw),
        }
        return np.ones((len(bins), len(bins)))

    def fake_pearson(z, bins, **kw):
        """Fake pearson matrix that returns a constant matrix."""
        calls["pearson"] = {"z": np.asarray(z), "bins": bins, "kw": dict(kw)}
        return np.zeros((len(bins), len(bins)))

    monkeypatch.setattr("binny.api.nz_tomography._bin_sim.bin_overlap", fake_overlap, raising=True)
    monkeypatch.setattr("binny.api.nz_tomography._bin_sim.overlap_pairs", fake_pairs, raising=True)
    monkeypatch.setattr(
        "binny.api.nz_tomography._bin_sim.leakage_matrix", fake_leakage, raising=True
    )
    monkeypatch.setattr(
        "binny.api.nz_tomography._bin_sim.pearson_matrix", fake_pearson, raising=True
    )

    t = NZTomography()
    z = np.linspace(0, 1, 11)
    nz = np.ones_like(z)
    payload = t.build_bins(
        z=z, nz=nz, tomo_spec={"kind": "photoz", "bins": {"scheme": "x", "n_bins": 2}}
    )

    out = t.cross_bin_stats(
        overlap={"metric": "l1"},
        pairs={"threshold": 0.1},
        leakage={"bin_edges": [0.0, 0.5, 1.0], "normalize": True},
        pearson={"clip": True},
    )

    assert set(out.keys()) == {"overlap", "pairs", "leakage", "pearson"}
    assert out["overlap"].shape == (2, 2)
    assert out["leakage"].shape == (2, 2)

    assert np.allclose(calls["overlap"]["z"], payload["z"])
    assert calls["overlap"]["bins"] == payload["bins"]
    assert calls["overlap"]["kw"] == {"metric": "l1"}

    assert calls["pairs"]["kw"] == {"threshold": 0.1}
    assert calls["leakage"]["bin_edges"] == [0.0, 0.5, 1.0]
    assert calls["leakage"]["kw"] == {"normalize": True}
    assert calls["pearson"]["kw"] == {"clip": True}


def test_resolve_builder_accepts_photoz_and_specz(monkeypatch):
    """Tests that resolve_builder accepts photoz and specz tomography kinds."""
    _install_fake_builder_modules(monkeypatch)

    t = NZTomography()
    b1 = t._resolve_builder("photoz")
    b2 = t._resolve_builder("specz")

    z = np.linspace(0, 1, 11)
    nz = np.ones_like(z)

    out1 = b1(z=z, nz=nz, include_metadata=False, n_bins=2)
    out2 = b2(z=z, nz=nz, include_metadata=False, n_bins=3)

    assert set(out1.keys()) == {0, 1}
    assert set(out2.keys()) == {0, 1, 2}


def test_resolve_builder_raises_on_unknown(monkeypatch):
    """Tests that resolve_builder raises on unknown tomography kind."""
    _install_fake_builder_modules(monkeypatch)
    t = NZTomography()
    with pytest.raises(ValueError, match=r"Unknown tomography kind"):
        t._resolve_builder("wat")


def test_nz_model_delegates(monkeypatch):
    """Tests that nz_model delegates to the real nz_model function."""
    called = {"args": None, "kwargs": None}

    def fake_nz_model(name, z, /, **params):
        called["args"] = (name, np.asarray(z))
        called["kwargs"] = dict(params)
        return np.asarray(z) * 0.0 + 7.0

    monkeypatch.setattr("binny.api.nz_tomography._nz_model", fake_nz_model, raising=True)

    z = np.linspace(0, 1, 3)
    out = NZTomography.nz_model("smail", z, z0=0.5)

    assert np.allclose(out, 7.0)
    assert called["args"][0] == "smail"
    assert np.allclose(called["args"][1], z)
    assert called["kwargs"] == {"z0": 0.5}


def test_list_nz_models_delegates(monkeypatch):
    """Tests that list_nz_models delegates to the real list_nz_models function."""
    monkeypatch.setattr(
        "binny.api.nz_tomography._available_nz_models", lambda: ["a", "b"], raising=True
    )
    assert NZTomography.list_nz_models() == ["a", "b"]


def test_list_survey_presets_filters_suffix(monkeypatch):
    """Tests that list_survey_presets filters out non-survey configs."""
    monkeypatch.setattr(
        cu,
        "list_configs",
        lambda: ["lsst_survey_specs.yaml", "README.txt", "hsc_survey_specs.yaml", "foo.yaml"],
        raising=True,
    )
    assert NZTomography.list_surveys() == ["hsc", "lsst"]


def test_registry_is_callable():
    """Tests that registry objects are callable."""
    from binny.nz.registry import nz_model

    z = np.linspace(0, 1, 5)
    out = nz_model("smail", z, z0=0.5, alpha=2.0, beta=1.0)
    assert out.shape == z.shape
    assert np.all(np.isfinite(out))
