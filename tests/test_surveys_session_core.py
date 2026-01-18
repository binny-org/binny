"""Unit tests for ``binny.surveys.session_core`` module."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import binny.surveys.session_core as sc


def test_resolve_tomo_builder_photoz_and_specz() -> None:
    """Tests that resolve_tomo_builder returns callables for photoz and specz."""
    b1 = sc.resolve_tomo_builder("photoz")
    b2 = sc.resolve_tomo_builder("specz")
    assert callable(b1)
    assert callable(b2)


def test_resolve_tomo_builder_rejects_unknown_kind() -> None:
    """Tests that resolve_tomo_builder rejects unknown kind strings."""
    with pytest.raises(ValueError, match=r"Unknown kind"):
        sc.resolve_tomo_builder("nope")


def test_make_parent_from_arrays_without_meta() -> None:
    """Tests that make_parent_from_arrays stores arrays and meta=None."""
    z = [0.0, 1.0]
    nz = [1.0, 2.0]
    parent = sc.make_parent_from_arrays(z=z, nz=nz, survey_meta=None)

    assert parent["source"] == "arrays"
    assert np.allclose(parent["z"], np.asarray(z, dtype=float))
    assert np.allclose(parent["nz"], np.asarray(nz, dtype=float))
    assert parent["survey_meta"] is None
    assert parent["cfg"] is None
    assert parent["config_file"] is None
    assert parent["key"] is None


def test_make_parent_from_arrays_with_meta_copied() -> None:
    """Tests that make_parent_from_arrays copies provided survey_meta."""
    parent = sc.make_parent_from_arrays(
        z=[0.0, 1.0],
        nz=[1.0, 2.0],
        survey_meta={"a": 1},
    )
    assert parent["survey_meta"] == {"a": 1}


def test_load_entry_from_mapping_wires_parent_and_last(monkeypatch) -> None:
    """Tests that load_entry_from_mapping populates parent/last from survey_from_mapping."""
    z = np.linspace(0.0, 1.0, 5)
    nz = np.exp(-z)
    spec = {
        "role": "lens",
        "year": "1",
        "kind": "photoz",
        "bins": {"scheme": "eq", "n_bins": 2},
    }
    meta = {"survey": "x"}

    def fake_survey_from_mapping(**kwargs: Any):
        return z, nz, spec, meta

    monkeypatch.setattr(sc, "survey_from_mapping", fake_survey_from_mapping)

    parent, last = sc.load_entry_from_mapping(
        cfg={"tomography": []},
        key="survey",
        role="lens",
        year="1",
        include_survey_metadata=True,
    )

    assert parent["source"] == "mapping"
    assert parent["cfg"] == {"tomography": []}
    assert np.allclose(parent["z"], z)
    assert np.allclose(parent["nz"], nz)
    assert parent["survey_meta"] == meta
    assert parent["config_file"] is None
    assert parent["key"] == "survey"

    assert last["bins"] is None
    assert last["tomo_meta"] is None
    assert last["kind"] == "photoz"
    assert last["tomo_spec"]["role"] == "lens"


def test_load_entry_from_config_wires_parent_and_last(monkeypatch, tmp_path) -> None:
    """Tests that load_entry_from_config populates parent/last from survey_from_config."""
    z = np.linspace(0.0, 1.0, 5)
    nz = np.exp(-z)
    spec = {
        "role": "lens",
        "year": "1",
        "kind": "specz",
        "bins": {"scheme": "eq", "n_bins": 2},
    }
    meta = {"survey": "x"}

    def fake_survey_from_config(**kwargs: Any):
        return z, nz, spec, meta

    monkeypatch.setattr(sc, "survey_from_config", fake_survey_from_config)

    p = tmp_path / "cfg.yaml"
    p.write_text("name: x\ntomography: []\n", encoding="utf-8")

    parent, last = sc.load_entry_from_config(
        config_file=p,
        role="lens",
        year="1",
        include_survey_metadata=False,
    )

    assert parent["source"] == "config"
    assert np.allclose(parent["z"], z)
    assert np.allclose(parent["nz"], nz)
    assert parent["survey_meta"] is None  # because include_survey_metadata=False
    assert parent["config_file"] == p
    assert parent["key"] is None

    assert last["kind"] == "specz"


def test_build_bins_from_state_no_metadata(monkeypatch) -> None:
    """Tests that build_bins_from_state returns bins when include_metadata is False."""
    parent = sc.make_parent_from_arrays(z=[0.0, 1.0], nz=[1.0, 2.0])
    last = {
        "tomo_spec": {"kind": "photoz", "bins": {}},
        "bins": None,
        "tomo_meta": None,
        "kind": "photoz",
    }

    def fake_resolve(kind: str):
        assert kind == "photoz"
        return object()  # builder placeholder

    def fake_build_from_arrays(**kwargs: Any):
        # include_tomo_metadata must be False here
        assert kwargs["include_tomo_metadata"] is False
        return {0: np.asarray(kwargs["nz"], dtype=float)}

    monkeypatch.setattr(sc, "resolve_tomo_builder", fake_resolve)
    monkeypatch.setattr(sc, "build_from_arrays", fake_build_from_arrays)

    last2, out = sc.build_bins_from_state(parent=parent, last=last, include_metadata=False)

    assert isinstance(out, dict)
    assert 0 in out
    assert last2["bins"] == out
    assert last2["tomo_meta"] is None


def test_build_bins_from_state_with_metadata_and_overrides(monkeypatch) -> None:
    """Tests that build_bins_from_state returns (bins, meta) and applies overrides."""
    parent = sc.make_parent_from_arrays(z=[0.0, 1.0], nz=[1.0, 2.0], survey_meta={"a": 1})
    last = {
        "tomo_spec": {"kind": "photoz", "bins": {}, "x": 1},
        "bins": None,
        "tomo_meta": None,
        "kind": "photoz",
    }

    def fake_resolve(kind: str):
        # kind override should force specz here
        assert kind == "specz"
        return object()

    def fake_build_from_arrays(**kwargs: Any):
        assert kwargs["kind"] == "specz"
        assert kwargs["include_tomo_metadata"] is True
        # overrides should have updated tomo_spec
        assert kwargs["tomo_spec"]["x"] == 999
        bins = {0: np.asarray(kwargs["nz"], dtype=float)}
        meta = {"tomo": "meta"}
        return bins, meta

    monkeypatch.setattr(sc, "resolve_tomo_builder", fake_resolve)
    monkeypatch.setattr(sc, "build_from_arrays", fake_build_from_arrays)

    last2, out = sc.build_bins_from_state(
        parent=parent,
        last=last,
        include_metadata=True,
        kind="specz",
        overrides={"x": 999},
    )

    assert isinstance(out, tuple)
    bins_out, meta_out = out
    assert isinstance(bins_out, dict)
    assert meta_out == {"tomo": "meta"}
    assert last2["kind"] == "specz"
    assert last2["tomo_spec"]["x"] == 999
    assert last2["tomo_meta"] == {"tomo": "meta"}
