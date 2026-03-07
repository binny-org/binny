"""Unit tests for binny.api.nz_tomography."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

import binny.surveys.config_utils as cu
from binny.api.nz_tomography import NZTomography
from binny.correlations.bin_combo_filter import BinComboFilter


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

    assert np.allclose(payload.z, z)
    assert np.allclose(payload.nz, nz)
    assert payload.spec["kind"] == "photoz"
    assert set(payload.bins.keys()) == {0, 1}
    assert payload.tomo_meta is None
    assert payload.survey_meta is None

    assert t._state is not None
    assert t._state["bins"] == payload.bins
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

    assert payload.spec["kind"] == "specz"
    assert payload.tomo_meta["builder"] == "specz"
    assert set(payload.bins.keys()) == {0, 1, 2}


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
    assert set(payload.bins.keys()) == {0, 1, 2, 3}


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

    assert np.allclose(payload.z, z)
    assert np.allclose(payload.nz, np.ones_like(z))
    assert payload.survey_meta["role"] == "lens"
    assert payload.survey_meta["year"] == "1"
    assert set(payload.bins.keys()) == {0, 1}


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
    assert set(payload.bins.keys()) == {0, 1}


def test_build_survey_bins_returns_tomographybins_and_stats_work(monkeypatch):
    """Test build_survey_bins can build and the returned handle can compute stats."""
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

    # Patch the stats functions where TomographyBins actually calls them
    monkeypatch.setattr(
        "binny.nz_tomo._tomography_bins._shape_stats",
        lambda *, z, bins, **k: {"ok": True, "n_bins": len(bins)},
        raising=True,
    )
    monkeypatch.setattr(
        "binny.nz_tomo._tomography_bins._population_stats",
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
        include_tomo_metadata=True,  # required for population_stats
    )

    assert payload.survey == "lsst"
    assert payload.tomo_meta is not None

    shape = payload.shape_stats()
    pop = payload.population_stats()

    assert shape["ok"] is True
    assert pop["ok"] is True


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

    calls: dict[str, dict] = {"overlap": {}, "correlations": {}, "leakage": {}, "pearson": {}}

    def fake_overlap(z, bins, **kw):
        """Fake overlap matrix that returns a constant matrix."""
        calls["overlap"] = {"z": np.asarray(z), "bins": bins, "kw": dict(kw)}
        return np.eye(len(bins))

    def fake_pairs(z, bins, **kw):
        """Fake correlations matrix that returns a constant matrix."""
        calls["correlations"] = {"z": np.asarray(z), "bins": bins, "kw": dict(kw)}
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

    assert set(out.keys()) == {"overlap", "correlations", "leakage", "pearson"}
    assert out["overlap"].shape == (2, 2)
    assert out["leakage"].shape == (2, 2)

    assert np.allclose(calls["overlap"]["z"], payload.z)
    assert calls["overlap"]["bins"] == t.bins
    assert calls["overlap"]["kw"] == {"metric": "l1"}

    assert calls["correlations"]["kw"] == {"threshold": 0.1}
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


def test_bins_property_raises_before_build():
    """Tests that bins property raises before build_bins is called."""
    t = NZTomography()
    with pytest.raises(ValueError, match=r"Call build_bins"):
        _ = t.bins


def test_bin_keys_property_raises_before_build():
    """Tests that bin_keys property raises before build_bins is called."""
    t = NZTomography()
    with pytest.raises(ValueError, match=r"Call build_bins"):
        _ = t.bin_keys


def test_bins_and_bin_keys_return_cached_mapping(monkeypatch):
    """Tests that bins and bin_keys expose the cached bin mapping and order."""
    _install_fake_builder_modules(monkeypatch)
    monkeypatch.setattr(
        cu, "_parse_entry", lambda e: {**dict(e), "role": "source", "year": "1"}, raising=True
    )
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 3}, raising=True)

    t = NZTomography()
    z = np.linspace(0, 1, 11)
    nz = np.ones_like(z)

    payload = t.build_bins(z=z, nz=nz, tomo_spec={"kind": "photoz", "bins": {"n_bins": 3}})
    assert t.bins == payload.bins
    assert t.bin_keys == [0, 1, 2]


def test_z_property_raises_before_build():
    """Tests that z property raises before any build is performed."""
    t = NZTomography()
    with pytest.raises(ValueError, match=r"Call build_bins"):
        _ = t.z


def test_nz_property_raises_before_build():
    """Tests that nz property raises before any build is performed."""
    t = NZTomography()
    with pytest.raises(ValueError, match=r"Call build_bins"):
        _ = t.nz


def test_available_metric_kernels_delegates(monkeypatch):
    """Tests that available_metric_kernels delegates to the registry helper."""
    monkeypatch.setattr(
        "binny.api.nz_tomography._available_metric_kernels",
        lambda: ["m0", "m1"],
        raising=True,
    )
    from binny.api.nz_tomography import available_metric_kernels

    assert available_metric_kernels() == ["m0", "m1"]


def test_register_metric_kernel_delegates(monkeypatch):
    """Tests that register_metric_kernel delegates to the registry helper."""
    called = {"name": None, "func": None}

    def fake_register(name, func):
        called["name"] = name
        called["func"] = func

    monkeypatch.setattr(
        "binny.api.nz_tomography._register_metric_kernel",
        fake_register,
        raising=True,
    )

    from binny.api.nz_tomography import register_metric_kernel

    def k(*curves) -> float:
        _ = curves
        return 0.0

    register_metric_kernel("my_kernel", k)
    assert called["name"] == "my_kernel"
    assert called["func"] is k


def test_bin_combo_filter_delegates_to_bincombofilter_select(monkeypatch):
    """Tests that NZTomography.bin_combo_filter delegates to BinComboFilter.select."""
    _install_fake_builder_modules(monkeypatch)
    monkeypatch.setattr(
        cu, "_parse_entry", lambda e: {**dict(e), "role": "source", "year": "1"}, raising=True
    )
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 2}, raising=True)

    # Build bins in the usual way.
    t = NZTomography()
    z = np.linspace(0, 1, 11)
    nz = np.ones_like(z)
    t.build_bins(z=z, nz=nz, tomo_spec={"kind": "photoz", "bins": {"n_bins": 2}})

    # Patch BinComboFilter.select so we can prove it was called and control output.
    called = {"spec": None}

    class _FakeSelection:
        def values(self):
            return [(0, 1), (1, 0)]

    def fake_select(self, spec):
        called["spec"] = dict(spec)
        return _FakeSelection()

    monkeypatch.setattr(BinComboFilter, "select", fake_select, raising=True)

    spec = {"topology": {"name": "pairs_all"}, "filters": [{"name": "noop"}]}
    out = t.bin_combo_filter(spec)

    assert called["spec"] == spec
    assert out == [(0, 1), (1, 0)]


def test_make_bin_combo_filter_uses_other_and_checks_shared_z(monkeypatch):
    """Tests that cross-sample combo filtering requires identical z grids."""
    _install_fake_builder_modules(monkeypatch)
    monkeypatch.setattr(
        cu, "_parse_entry", lambda e: {**dict(e), "role": "source", "year": "1"}, raising=True
    )
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 2}, raising=True)

    t0 = NZTomography()
    t1 = NZTomography()

    z0 = np.linspace(0, 1, 11)
    z1 = np.linspace(0, 1, 12)  # different shape -> must fail
    nz0 = np.ones_like(z0)
    nz1 = np.ones_like(z1)

    t0.build_bins(z=z0, nz=nz0, tomo_spec={"kind": "photoz", "bins": {"n_bins": 2}})
    t1.build_bins(z=z1, nz=nz1, tomo_spec={"kind": "photoz", "bins": {"n_bins": 2}})

    with pytest.raises(ValueError, match=r"shared z grid"):
        _ = t0._make_bin_combo_filter(t1)

    # Now build t1 on the same grid and it should succeed.
    t1.clear()
    t1.build_bins(z=z0, nz=nz0, tomo_spec={"kind": "photoz", "bins": {"n_bins": 2}})

    f = t0._make_bin_combo_filter(t1)
    assert isinstance(f, BinComboFilter)


def test_make_bin_combo_filter_curves_argument_overrides_default(monkeypatch):
    """Tests that explicit curves bypasses self/other wiring."""
    _install_fake_builder_modules(monkeypatch)

    t = NZTomography()
    t._parent = {"z": np.linspace(0, 1, 5), "nz": np.ones(5), "survey_meta": None}
    t._state = {
        "tomo_spec": {"kind": "photoz", "bins": {}},
        "bins": {0: np.ones(5)},
        "tomo_meta": None,
    }

    z = t._parent["z"]
    curves = [{0: np.zeros_like(z)}, {0: np.ones_like(z)}]

    f = t._make_bin_combo_filter(curves=curves)
    assert isinstance(f, BinComboFilter)


def test_build_survey_bins_unknown_preset_raises(monkeypatch):
    """Tests build_survey_bins raises nicely when preset is unknown."""
    monkeypatch.setattr(
        cu, "config_path", lambda filename: (_ for _ in ()).throw(FileNotFoundError()), raising=True
    )
    monkeypatch.setattr(cu, "list_configs", lambda: ["hsc_survey_specs.yaml"], raising=True)

    t = NZTomography()
    with pytest.raises(FileNotFoundError, match=r"Unknown shipped survey preset"):
        t.build_survey_bins("not-a-real-survey")


def test_calibrate_smail_from_mock_delegates(monkeypatch):
    """Tests that calibrate_smail_from_mock delegates to calibration helper."""
    called = {}

    def fake_calibrate_depth_smail_from_mock(
        *,
        z_true,
        mag,
        maglims,
        area_deg2,
        infer_alpha_beta_from,
        alpha_beta_maglim,
        z_max,
    ):
        called["z_true"] = z_true
        called["mag"] = mag
        called["maglims"] = maglims
        called["area_deg2"] = area_deg2
        called["infer_alpha_beta_from"] = infer_alpha_beta_from
        called["alpha_beta_maglim"] = alpha_beta_maglim
        called["z_max"] = z_max
        return {"ok": True, "source": "fake"}

    monkeypatch.setattr(
        "binny.api.nz_tomography._calibrate_depth_smail_from_mock",
        fake_calibrate_depth_smail_from_mock,
        raising=True,
    )

    z_true = np.array([0.1, 0.2, 0.3])
    mag = np.array([24.1, 24.5, 25.0])
    maglims = np.array([24.5, 25.0, 25.5])

    out = NZTomography.calibrate_smail_from_mock(
        z_true=z_true,
        mag=mag,
        maglims=maglims,
        area_deg2=5.0,
        infer_alpha_beta_from="deep_cut",
        alpha_beta_maglim=25.5,
        z_max=3.0,
    )

    assert out == {"ok": True, "source": "fake"}
    assert np.allclose(called["z_true"], z_true)
    assert np.allclose(called["mag"], mag)
    assert np.allclose(called["maglims"], maglims)
    assert called["area_deg2"] == 5.0
    assert called["infer_alpha_beta_from"] == "deep_cut"
    assert called["alpha_beta_maglim"] == 25.5
    assert called["z_max"] == 3.0


def test_calibrate_smail_from_mock_uses_defaults(monkeypatch):
    """Tests that calibrate_smail_from_mock forwards default optional arguments."""
    called = {}

    def fake_calibrate_depth_smail_from_mock(**kwargs):
        called.update(kwargs)
        return {"ok": True}

    monkeypatch.setattr(
        "binny.api.nz_tomography._calibrate_depth_smail_from_mock",
        fake_calibrate_depth_smail_from_mock,
        raising=True,
    )

    z_true = np.array([0.1, 0.2])
    mag = np.array([24.0, 24.3])
    maglims = np.array([24.5, 25.0])

    out = NZTomography.calibrate_smail_from_mock(
        z_true=z_true,
        mag=mag,
        maglims=maglims,
        area_deg2=1.5,
    )

    assert out == {"ok": True}
    assert called["infer_alpha_beta_from"] == "deep_cut"
    assert called["alpha_beta_maglim"] is None
    assert called["z_max"] is None


def test_between_sample_stats_raises_before_build():
    """Tests that between_sample_stats raises before build_bins is called."""
    t0 = NZTomography()
    t1 = NZTomography()

    with pytest.raises(ValueError, match=r"Call build_bins"):
        t0.between_sample_stats(t1, overlap={})


def test_between_sample_stats_skips_all_when_all_none(monkeypatch):
    """Tests that between_sample_stats returns an empty dict when all inputs are None."""
    _install_fake_builder_modules(monkeypatch)
    monkeypatch.setattr(
        cu, "_parse_entry", lambda e: {**dict(e), "role": "source", "year": "1"}, raising=True
    )
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 2}, raising=True)

    t0 = NZTomography()
    t1 = NZTomography()

    z = np.linspace(0, 1, 11)
    nz0 = np.ones_like(z)
    nz1 = np.linspace(1.0, 2.0, z.size)

    t0.build_bins(z=z, nz=nz0, tomo_spec={"kind": "photoz", "bins": {"n_bins": 2}})
    t1.build_bins(z=z, nz=nz1, tomo_spec={"kind": "photoz", "bins": {"n_bins": 2}})

    out = t0.between_sample_stats(t1)
    assert out == {}


def test_between_sample_stats_requires_shared_z_grid(monkeypatch):
    """Tests that between_sample_stats requires both instances to share the same z grid."""
    _install_fake_builder_modules(monkeypatch)
    monkeypatch.setattr(
        cu, "_parse_entry", lambda e: {**dict(e), "role": "source", "year": "1"}, raising=True
    )
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 2}, raising=True)

    t0 = NZTomography()
    t1 = NZTomography()

    z0 = np.linspace(0, 1, 11)
    z1 = np.linspace(0, 1, 12)
    nz0 = np.ones_like(z0)
    nz1 = np.ones_like(z1)

    t0.build_bins(z=z0, nz=nz0, tomo_spec={"kind": "photoz", "bins": {"n_bins": 2}})
    t1.build_bins(z=z1, nz=nz1, tomo_spec={"kind": "photoz", "bins": {"n_bins": 2}})

    with pytest.raises(ValueError, match=r"share the same z grid"):
        t0.between_sample_stats(t1, overlap={})


def test_between_sample_stats_interval_mass_requires_target_edges(monkeypatch):
    """Tests that between_sample_stats requires target_edges for interval_mass."""
    _install_fake_builder_modules(monkeypatch)
    monkeypatch.setattr(
        cu, "_parse_entry", lambda e: {**dict(e), "role": "source", "year": "1"}, raising=True
    )
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 2}, raising=True)

    t0 = NZTomography()
    t1 = NZTomography()

    z = np.linspace(0, 1, 11)
    nz0 = np.ones_like(z)
    nz1 = np.linspace(1.0, 2.0, z.size)

    t0.build_bins(z=z, nz=nz0, tomo_spec={"kind": "photoz", "bins": {"n_bins": 2}})
    t1.build_bins(z=z, nz=nz1, tomo_spec={"kind": "photoz", "bins": {"n_bins": 2}})

    with pytest.raises(ValueError, match=r"interval_mass requires interval_mass=\{'target_edges':"):
        t0.between_sample_stats(t1, interval_mass={"normalize": True})


def test_between_sample_stats_delegates_to_between_sample_metrics(monkeypatch):
    """Tests that between_sample_stats delegates to between_sample_metrics."""
    _install_fake_builder_modules(monkeypatch)
    monkeypatch.setattr(
        cu, "_parse_entry", lambda e: {**dict(e), "role": "source", "year": "1"}, raising=True
    )
    monkeypatch.setattr(cu, "_builder_kwargs_from_spec", lambda spec: {"n_bins": 2}, raising=True)

    calls: dict[str, dict] = {
        "overlap": {},
        "correlations": {},
        "interval_mass": {},
        "pearson": {},
    }

    def fake_between_overlap(z, bins_a, bins_b, **kw):
        """Fake between-sample overlap matrix that returns a constant matrix."""
        calls["overlap"] = {
            "z": np.asarray(z),
            "bins_a": bins_a,
            "bins_b": bins_b,
            "kw": dict(kw),
        }
        return np.eye(len(bins_a))

    def fake_between_pairs(z, bins_a, bins_b, **kw):
        """Fake between-sample pair summaries that return a simple list."""
        calls["correlations"] = {
            "z": np.asarray(z),
            "bins_a": bins_a,
            "bins_b": bins_b,
            "kw": dict(kw),
        }
        return [(0, 0), (1, 1)]

    def fake_between_interval_mass(z, bins_a, target_edges, **kw):
        """Fake between-sample interval-mass matrix that returns a constant matrix."""
        calls["interval_mass"] = {
            "z": np.asarray(z),
            "bins_a": bins_a,
            "target_edges": target_edges,
            "kw": dict(kw),
        }
        return np.ones((len(bins_a), len(target_edges) - 1))

    def fake_between_pearson(z, bins_a, bins_b, **kw):
        """Fake between-sample pearson matrix that returns a constant matrix."""
        calls["pearson"] = {
            "z": np.asarray(z),
            "bins_a": bins_a,
            "bins_b": bins_b,
            "kw": dict(kw),
        }
        return np.zeros((len(bins_a), len(bins_b)))

    monkeypatch.setattr(
        "binny.api.nz_tomography._between_metrics.between_bin_overlap",
        fake_between_overlap,
        raising=True,
    )
    monkeypatch.setattr(
        "binny.api.nz_tomography._between_metrics.between_overlap_pairs",
        fake_between_pairs,
        raising=True,
    )
    monkeypatch.setattr(
        "binny.api.nz_tomography._between_metrics.between_interval_mass_matrix",
        fake_between_interval_mass,
        raising=True,
    )
    monkeypatch.setattr(
        "binny.api.nz_tomography._between_metrics.between_pearson_matrix",
        fake_between_pearson,
        raising=True,
    )

    t0 = NZTomography()
    t1 = NZTomography()

    z = np.linspace(0, 1, 11)
    nz0 = np.ones_like(z)
    nz1 = np.linspace(1.0, 2.0, z.size)

    payload0 = t0.build_bins(
        z=z,
        nz=nz0,
        tomo_spec={"kind": "photoz", "bins": {"scheme": "x", "n_bins": 2}},
    )
    payload1 = t1.build_bins(
        z=z,
        nz=nz1,
        tomo_spec={"kind": "photoz", "bins": {"scheme": "x", "n_bins": 2}},
    )

    out = t0.between_sample_stats(
        t1,
        overlap={"metric": "l1"},
        pairs={"threshold": 0.2},
        interval_mass={"target_edges": [0.0, 0.5, 1.0], "normalize": True},
        pearson={"clip": True},
    )

    assert set(out.keys()) == {"overlap", "correlations", "interval_mass", "pearson"}
    assert out["overlap"].shape == (2, 2)
    assert out["interval_mass"].shape == (2, 2)
    assert out["pearson"].shape == (2, 2)

    assert np.allclose(calls["overlap"]["z"], payload0.z)
    assert calls["overlap"]["bins_a"] == t0.bins
    assert calls["overlap"]["bins_b"] == t1.bins
    assert calls["overlap"]["kw"] == {"metric": "l1"}

    assert np.allclose(calls["correlations"]["z"], payload1.z)
    assert calls["correlations"]["kw"] == {"threshold": 0.2}

    assert calls["interval_mass"]["target_edges"] == [0.0, 0.5, 1.0]
    assert calls["interval_mass"]["bins_a"] == t0.bins
    assert calls["interval_mass"]["kw"] == {"normalize": True}

    assert calls["pearson"]["bins_a"] == t0.bins
    assert calls["pearson"]["bins_b"] == t1.bins
    assert calls["pearson"]["kw"] == {"clip": True}
