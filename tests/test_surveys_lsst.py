"""Tests that LSST convenience wrapper validates presets and return shapes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

import binny.surveys.lsst as lsst_mod


def _good_cfg(
    *,
    sample: str,
    year_key: str,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds a minimal LSST YAML-like mapping for tests."""
    block: dict[str, Any] = {
        "n_tomo_bins": 4,
        "smail": {"z0": 0.3, "alpha": 2.0, "beta": 1.0},
        "photoz": {"scatter_scale": 0.05, "mean_offset": 0.0},
        # "binning": optional
    }
    if overrides:
        # shallow merge at the block level is enough for these tests
        block.update(dict(overrides))

    cfg_lsst: dict[str, Any] = {
        "name": "lsst",
        "grid": {
            "method": "linear",
            "params": {"x_min": 0.0, "x_max": 3.0, "n": 11},
        },
        "footprint": {"area_deg2": 18000.0},
        "lens_sample": {},
        "source_sample": {},
    }
    cfg_lsst[f"{sample}_sample"][year_key] = block

    return {"lsst": cfg_lsst}


def test_year_key_accepts_only_1_or_10():
    """Tests that _year_key accepts only year 1 or 10."""
    assert lsst_mod._year_key(1) == "y1"
    assert lsst_mod._year_key(10) == "y10"

    with pytest.raises(ValueError, match=r"year must be 1 or 10"):
        lsst_mod._year_key(2)


def test_raises_if_top_level_lsst_missing_or_not_mapping(monkeypatch):
    """Tests that lsst_tomography rejects a preset without top-level lsst mapping."""
    monkeypatch.setattr(lsst_mod, "load_yaml", lambda *a, **k: {"lsst": 123})

    with pytest.raises(ValueError, match=r"top-level mapping 'lsst'"):
        lsst_mod.lsst_tomography(year=1, sample="lens")


def test_raises_if_sample_block_missing(monkeypatch):
    """Tests that lsst_tomography raises if lsst.lens_sample or lsst.source_sample is missing."""
    cfg = {"lsst": {"grid": {}}}  # missing lens_sample/source_sample
    monkeypatch.setattr(lsst_mod, "load_yaml", lambda *a, **k: cfg)

    with pytest.raises(ValueError, match=r"Preset missing mapping lsst\.lens_sample"):
        lsst_mod.lsst_tomography(year=1, sample="lens")


def test_raises_if_year_block_missing(monkeypatch):
    """Tests that lsst_tomography raises if lsst.<sample>_sample.<year> is missing."""
    cfg = {"lsst": {"grid": {}, "lens_sample": {}, "source_sample": {}}}
    monkeypatch.setattr(lsst_mod, "load_yaml", lambda *a, **k: cfg)

    with pytest.raises(ValueError, match=r"lsst\.lens_sample\.y1"):
        lsst_mod.lsst_tomography(year=1, sample="lens")


def test_raises_if_grid_missing(monkeypatch):
    """Tests that lsst_tomography raises if lsst.grid is missing."""
    cfg = {"lsst": {"lens_sample": {"y1": {}}, "source_sample": {"y1": {}}}}
    monkeypatch.setattr(lsst_mod, "load_yaml", lambda *a, **k: cfg)

    with pytest.raises(ValueError, match=r"Preset missing mapping lsst\.grid"):
        lsst_mod.lsst_tomography(year=1, sample="lens")


def test_raises_if_smail_or_photoz_missing(monkeypatch):
    """Tests that lsst_tomography raises if smail or photoz blocks are missing."""
    cfg = _good_cfg(sample="lens", year_key="y1", overrides={"smail": None})
    monkeypatch.setattr(lsst_mod, "load_yaml", lambda *a, **k: cfg)

    with pytest.raises(ValueError, match=r"lsst\.lens_sample\.y1\.smail"):
        lsst_mod.lsst_tomography(year=1, sample="lens")

    cfg2 = _good_cfg(sample="lens", year_key="y1", overrides={"photoz": None})
    monkeypatch.setattr(lsst_mod, "load_yaml", lambda *a, **k: cfg2)

    with pytest.raises(ValueError, match=r"lsst\.lens_sample\.y1\.photoz"):
        lsst_mod.lsst_tomography(year=1, sample="lens")


def test_raises_if_required_key_missing(monkeypatch):
    """Tests that lsst_tomography raises with the missing required key name."""
    cfg = _good_cfg(sample="lens", year_key="y1")
    # Remove a required key nested in smail.
    cfg["lsst"]["lens_sample"]["y1"]["smail"].pop("z0")
    monkeypatch.setattr(lsst_mod, "load_yaml", lambda *a, **k: cfg)

    with pytest.raises(ValueError, match=r"Preset missing required key: 'z0'"):
        lsst_mod.lsst_tomography(year=1, sample="lens")


def test_defaults_and_binning_scheme_lowercased(monkeypatch):
    """Tests that default binning scheme differs for lens/source and is lowercased."""
    seen: dict[str, Any] = {}

    def fake_survey_from_mapping(*, cfg, key, z, include_survey_metadata):
        # capture the constructed entry
        seen["cfg_entry"] = cfg
        # return minimal objects consistent with downstream
        z_arr = [0.0, 1.0]
        nz_arr = [1.0, 1.0]
        tomo = {
            "bin_edges": [0.0, 0.5, 1.0],
            "binning_scheme": cfg["tomo"]["binning_scheme"],
            "params": cfg["tomo"]["params"],
        }
        survey_meta = {"ok": True}
        return z_arr, nz_arr, tomo, survey_meta

    def fake_build_photoz_bins(*, z, nz, bin_edges, binning_scheme, include_metadata, **params):
        if include_metadata:
            return {"0": [1.0, 0.0]}, {"meta": True}
        return {"0": [1.0, 0.0]}

    monkeypatch.setattr(lsst_mod, "survey_from_mapping", fake_survey_from_mapping)
    monkeypatch.setattr(lsst_mod, "build_photoz_bins", fake_build_photoz_bins)

    # lens default => equidistant
    cfg_lens = _good_cfg(sample="lens", year_key="y1")
    monkeypatch.setattr(lsst_mod, "load_yaml", lambda *a, **k: cfg_lens)
    out = lsst_mod.lsst_tomography(year=1, sample="lens")
    assert isinstance(out, dict)
    assert seen["cfg_entry"]["tomo"]["binning_scheme"] == "equidistant"

    # source default => equal_number
    cfg_src = _good_cfg(sample="source", year_key="y1")
    monkeypatch.setattr(lsst_mod, "load_yaml", lambda *a, **k: cfg_src)
    out2 = lsst_mod.lsst_tomography(year=1, sample="source")
    assert isinstance(out2, dict)
    assert seen["cfg_entry"]["tomo"]["binning_scheme"] == "equal_number"

    # explicit binning should be lowercased
    cfg_src2 = _good_cfg(sample="source", year_key="y1", overrides={"binning": "EqUiDiStAnT"})
    monkeypatch.setattr(lsst_mod, "load_yaml", lambda *a, **k: cfg_src2)
    _ = lsst_mod.lsst_tomography(year=1, sample="source")
    assert seen["cfg_entry"]["tomo"]["binning_scheme"] == "equidistant"


@pytest.mark.parametrize(
    "include_survey_metadata, include_tomo_metadata, expected_len",
    [
        (False, False, 0),
        (True, False, 1),
        (False, True, 1),
        (True, True, 2),
    ],
)
def test_return_shapes_follow_metadata_flags(
    monkeypatch,
    include_survey_metadata: bool,
    include_tomo_metadata: bool,
    expected_len: int,
):
    """Tests that lsst_tomography returns bins plus optional metadata in the right combinations."""

    def fake_survey_from_mapping(*, cfg, key, z, include_survey_metadata):
        z_arr = [0.0, 1.0]
        nz_arr = [1.0, 1.0]
        tomo = {
            "bin_edges": [0.0, 0.5, 1.0],
            "binning_scheme": cfg["tomo"]["binning_scheme"],
            "params": cfg["tomo"]["params"],
        }
        survey_meta = {"survey": "meta"}
        return z_arr, nz_arr, tomo, survey_meta

    def fake_build_photoz_bins(*, z, nz, bin_edges, binning_scheme, include_metadata, **params):
        bins = {"0": [1.0, 0.0]}
        if include_metadata:
            return bins, {"tomo": "meta"}
        return bins

    monkeypatch.setattr(lsst_mod, "survey_from_mapping", fake_survey_from_mapping)
    monkeypatch.setattr(lsst_mod, "build_photoz_bins", fake_build_photoz_bins)
    monkeypatch.setattr(
        lsst_mod,
        "load_yaml",
        lambda *a, **k: _good_cfg(sample="lens", year_key="y1"),
    )

    out = lsst_mod.lsst_tomography(
        year=1,
        sample="lens",
        include_survey_metadata=include_survey_metadata,
        include_tomo_metadata=include_tomo_metadata,
    )

    if expected_len == 0:
        assert isinstance(out, dict)
    else:
        assert isinstance(out, tuple)
        assert len(out) == 1 + expected_len  # bins + optional meta(s)
        assert isinstance(out[0], dict)
