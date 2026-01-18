"""Unit tests for :class:`NZTomography`."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

import binny.api.nz_tomography as nzmod
from binny.api.nz_tomography import NZTomography


@pytest.fixture
def z() -> np.ndarray:
    """Tests that a standard true-z grid is available for NZTomography tests."""
    return np.linspace(0.0, 3.0, 301, dtype=np.float64)


@pytest.fixture
def nz(z: np.ndarray) -> np.ndarray:
    """Tests that a finite parent n(z) is available for NZTomography tests."""
    out = z**2 * np.exp(-z)
    out[0] = 0.0
    return out.astype(np.float64)


@pytest.fixture
def tomo_spec_photoz() -> dict[str, Any]:
    """Tests that a minimal photo-z tomo_spec mapping is available."""
    return {
        "kind": "photoz",
        "bin_edges": [0.0, 0.5, 1.0, 1.5],
        "params": {"scatter_scale": 0.05, "mean_offset": 0.01},
    }


def test_init_starts_empty() -> None:
    """Tests that a new NZTomography session starts with no cached state."""
    t = NZTomography()
    assert t.has_parent() is False
    assert t.has_entry() is False
    assert t.has_bins() is False
    assert t.parent_source() == "none"


def test_clear_resets_state(z: np.ndarray, nz: np.ndarray) -> None:
    """Tests that clear resets cached parent and cached last state."""
    t = NZTomography()
    t.set_parent_from_arrays(z=z, nz=nz)
    assert t.has_parent() is True

    t.clear()
    assert t.has_parent() is False
    assert t.has_entry() is False
    assert t.has_bins() is False
    assert t.parent_source() == "none"


def test_accessors_require_parent() -> None:
    """Tests that z/nz/survey_meta accessors raise before parent is cached."""
    t = NZTomography()
    with pytest.raises(ValueError, match=r"No parent"):
        _ = t.z()
    with pytest.raises(ValueError, match=r"No parent"):
        _ = t.nz()
    with pytest.raises(ValueError, match=r"No parent"):
        _ = t.survey_meta()


def test_set_parent_from_arrays_sets_parent_and_clears_last(
    z: np.ndarray, nz: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that set_parent_from_arrays caches parent and clears cached entry."""

    def _fake_make_parent_from_arrays(
        *, z: Any, nz: Any, survey_meta: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        return {
            "source": "arrays",
            "z": np.asarray(z, dtype=np.float64),
            "nz": np.asarray(nz, dtype=np.float64),
            "survey_meta": survey_meta,
        }

    monkeypatch.setattr(nzmod, "_make_parent_from_arrays", _fake_make_parent_from_arrays)

    t = NZTomography()
    t._last = {"tomo_spec": {"kind": "photoz"}, "bins": {"x": 1}}  # type: ignore[assignment]
    t.set_parent_from_arrays(z=z, nz=nz, survey_meta={"name": "x"})

    assert t.has_parent() is True
    assert t.has_entry() is False
    assert t.has_bins() is False
    assert t.parent_source() == "arrays"
    assert t.z().shape == z.shape
    assert t.nz().shape == nz.shape
    assert dict(t.survey_meta() or {}) == {"name": "x"}


def test_kind_defaults_when_no_entry() -> None:
    """Tests that kind returns the provided default when no entry is cached."""
    t = NZTomography()
    assert t.kind() == "photoz"
    assert t.kind(default="specz") == "specz"


def test_load_entry_from_mapping_caches_parent_and_entry(
    z: np.ndarray,
    nz: np.ndarray,
    tomo_spec_photoz: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that load_entry_from_mapping caches parent and tomo_spec."""
    parent = {"source": "mapping", "z": z, "nz": nz, "survey_meta": {"k": "v"}}
    last = {
        "tomo_spec": dict(tomo_spec_photoz),
        "bins": None,
        "tomo_meta": None,
    }

    def _fake_load_entry_from_mapping(
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return dict(parent), dict(last)

    monkeypatch.setattr(nzmod, "_load_entry_from_mapping", _fake_load_entry_from_mapping)

    t = NZTomography()
    t.load_entry_from_mapping(cfg={"survey": {}}, key="survey", role="lens", year=1)

    assert t.has_parent() is True
    assert t.has_entry() is True
    assert t.has_bins() is False
    assert t.parent_source() == "mapping"
    assert t.kind() == "photoz"
    assert dict(t.tomo_spec())["kind"] == "photoz"


def test_build_requires_parent_and_entry() -> None:
    """Tests that build raises unless both parent and tomo_spec are cached."""
    t = NZTomography()
    with pytest.raises(ValueError, match=r"No parent"):
        _ = t.build()

    t._parent = {"source": "arrays", "z": np.arange(3), "nz": np.arange(3)}  # type: ignore[assignment]
    with pytest.raises(ValueError, match=r"No tomo_spec"):
        _ = t.build()


def test_build_calls_core_and_caches_bins(
    z: np.ndarray,
    nz: np.ndarray,
    tomo_spec_photoz: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that build forwards args to core builder and caches bins/meta."""
    t = NZTomography()
    t._parent = {"source": "arrays", "z": z, "nz": nz}  # type: ignore[assignment]
    t._last = {
        "tomo_spec": dict(tomo_spec_photoz),
        "bins": None,
        "tomo_meta": None,
    }  # type: ignore[assignment]

    calls: list[dict[str, Any]] = []

    def _fake_build_bins_from_state(
        *,
        parent: Mapping[str, Any],
        last: Mapping[str, Any],
        include_metadata: bool,
        kind: str | None,
        overrides: Mapping[str, Any] | None,
    ) -> tuple[dict[str, Any], Any]:
        calls.append(
            {
                "include_metadata": include_metadata,
                "kind": kind,
                "overrides": overrides,
                "parent_source": parent.get("source"),
                "spec_kind": (last.get("tomo_spec") or {}).get("kind"),
            }
        )
        new_last = dict(last)
        new_last["bins"] = {
            0: np.ones_like(z),
            1: np.zeros_like(z),
        }
        new_last["tomo_meta"] = {"ok": True} if include_metadata else None
        return new_last, new_last["bins"]

    monkeypatch.setattr(nzmod, "_build_bins_from_state", _fake_build_bins_from_state)

    out = t.build(include_metadata=True, kind="photoz", overrides={"x": 1})

    assert calls and calls[0]["include_metadata"] is True
    assert calls[0]["kind"] == "photoz"
    assert dict(calls[0]["overrides"] or {}) == {"x": 1}
    assert t.has_bins() is True
    assert isinstance(out, dict)
    assert set(t.bins().keys()) == {0, 1}
    assert dict(t.tomo_meta() or {}) == {"ok": True}


def test_build_from_arrays_sets_state_then_builds(
    z: np.ndarray,
    nz: np.ndarray,
    tomo_spec_photoz: Mapping[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that build_from_arrays caches arrays, caches spec, then builds."""

    def _fake_make_parent_from_arrays(
        *, z: Any, nz: Any, survey_meta: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        return {"source": "arrays", "z": np.asarray(z), "nz": np.asarray(nz)}

    def _fake_build_bins_from_state(
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Any]:
        last = dict(kwargs["last"])
        z_arr = np.asarray(kwargs["parent"]["z"])
        last["bins"] = {0: np.ones_like(z_arr)}
        last["tomo_meta"] = {"ok": True} if kwargs["include_metadata"] else None
        return last, last["bins"]

    monkeypatch.setattr(nzmod, "_make_parent_from_arrays", _fake_make_parent_from_arrays)
    monkeypatch.setattr(nzmod, "_build_bins_from_state", _fake_build_bins_from_state)

    t = NZTomography()
    out = t.build_from_arrays(z=z, nz=nz, tomo_spec=tomo_spec_photoz, include_metadata=True)

    assert t.parent_source() == "arrays"
    assert t.has_entry() is True
    assert t.has_bins() is True
    assert isinstance(out, dict)
    assert set(t.bins().keys()) == {0}
    assert dict(t.tomo_meta() or {}) == {"ok": True}


def test_shape_stats_forwards_cached_z_and_bins(
    z: np.ndarray, nz: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that shape_stats forwards cached z and bins to bin_stats.shape_stats."""
    t = NZTomography()
    t._parent = {"source": "arrays", "z": z, "nz": nz}  # type: ignore[assignment]
    t._last = {"tomo_spec": {"kind": "photoz"}, "bins": {0: np.ones_like(z)}}  # type: ignore[assignment]

    seen: dict[str, Any] = {}

    def _fake_shape_stats(*, z: Any, bins: Any, **kwargs: Any) -> dict[str, Any]:
        seen["z"] = z
        seen["bins"] = bins
        seen["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(nzmod, "_shape_stats", _fake_shape_stats)

    out = t.shape_stats(foo=1)
    assert dict(out) == {"ok": True}
    assert np.asarray(seen["z"]).shape == z.shape
    assert set(dict(seen["bins"]).keys()) == {0}
    assert dict(seen["kwargs"]) == {"foo": 1}


def test_population_stats_requires_cached_tomo_meta(z: np.ndarray, nz: np.ndarray) -> None:
    """Tests that population_stats raises if no cached tomography metadata exists."""
    t = NZTomography()
    t._parent = {"source": "arrays", "z": z, "nz": nz}  # type: ignore[assignment]
    t._last = {"tomo_spec": {"kind": "photoz"}, "bins": {0: np.ones_like(z)}}  # type: ignore[assignment]

    with pytest.raises(ValueError, match=r"No tomo metadata cached"):
        _ = t.population_stats()


def test_population_stats_forwards_bins_and_metadata(
    z: np.ndarray, nz: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tests that population_stats forwards bins and cached metadata correctly."""
    parent = {"source": "mapping", "z": z, "nz": nz, "survey_meta": None}
    last = {"tomo_spec": {"kind": "photoz"}, "bins": None, "tomo_meta": None}

    def _fake_load_entry_from_mapping(
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return dict(parent), dict(last)

    def _fake_build_bins_from_state(
        **kwargs: Any,
    ) -> tuple[dict[str, Any], Any]:
        new_last = dict(kwargs["last"])
        new_last["bins"] = {0: np.ones_like(z)}
        new_last["tomo_meta"] = {"a": 1}
        return new_last, new_last["bins"]

    seen: dict[str, Any] = {}

    def _fake_population_stats(*, bins: Any, metadata: Any, **kwargs: Any) -> dict[str, Any]:
        seen["bins"] = bins
        seen["metadata"] = metadata
        seen["kwargs"] = kwargs
        return {"ok": True}

    monkeypatch.setattr(nzmod, "_load_entry_from_mapping", _fake_load_entry_from_mapping)
    monkeypatch.setattr(nzmod, "_build_bins_from_state", _fake_build_bins_from_state)
    monkeypatch.setattr(nzmod, "_population_stats", _fake_population_stats)

    t = NZTomography()
    t.load_entry_from_mapping(cfg={"survey": {}}, key="survey", role="lens", year=1)

    _ = t.build(include_metadata=True)

    out = t.population_stats(bar=2)
    assert dict(out) == {"ok": True}
    assert set(dict(seen["bins"]).keys()) == {0}
    assert dict(seen["metadata"]) == {"a": 1}
    assert dict(seen["kwargs"]) == {"bar": 2}


def test_from_mapping_calls_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests that from_mapping constructs a session and calls load_entry_from_mapping."""
    seen: dict[str, Any] = {}

    def _fake_load_entry_from_mapping(self: NZTomography, **kwargs: Any) -> None:
        seen.update(kwargs)
        # minimal state to look like a loaded session
        self._parent = {
            "source": "mapping",
            "z": np.arange(3),
            "nz": np.arange(3),
        }  # type: ignore[assignment]
        self._last = {
            "tomo_spec": {"kind": "photoz"},
            "bins": None,
            "tomo_meta": None,
        }  # type: ignore[assignment]

    monkeypatch.setattr(NZTomography, "load_entry_from_mapping", _fake_load_entry_from_mapping)

    t = NZTomography.from_mapping(
        {"survey": {"x": 1}},
        key="survey",
        role="lens",
        year=1,
        z=np.linspace(0.0, 1.0, 5),
        include_survey_metadata=True,
    )

    assert isinstance(t, NZTomography)
    assert t.parent_source() == "mapping"
    assert seen["key"] == "survey"
    assert seen["role"] == "lens"
    assert seen["year"] == 1
    assert np.asarray(seen["z"]).shape == (5,)
    assert seen["include_survey_metadata"] is True


def test_from_config_calls_loader(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Tests that from_config constructs a session and calls load_entry_from_config."""
    seen: dict[str, Any] = {}

    def _fake_load_entry_from_config(self: NZTomography, **kwargs: Any) -> None:
        seen.update(kwargs)
        self._parent = {
            "source": "config",
            "z": np.arange(4),
            "nz": np.arange(4),
        }  # type: ignore[assignment]
        self._last = {
            "tomo_spec": {"kind": "specz"},
            "bins": None,
            "tomo_meta": None,
        }  # type: ignore[assignment]

    monkeypatch.setattr(NZTomography, "load_entry_from_config", _fake_load_entry_from_config)

    cfg_file = tmp_path / "survey.yaml"
    cfg_file.write_text("survey: {}\n", encoding="utf-8")

    t = NZTomography.from_config(
        cfg_file,
        key="survey",
        role="source",
        year="y1",
        z=np.linspace(0.0, 2.0, 7),
        include_survey_metadata=False,
    )

    assert isinstance(t, NZTomography)
    assert t.parent_source() == "config"
    assert str(seen["config_file"]).endswith("survey.yaml")
    assert seen["key"] == "survey"
    assert seen["role"] == "source"
    assert seen["year"] == "y1"
    assert np.asarray(seen["z"]).shape == (7,)
    assert seen["include_survey_metadata"] is False


def test_load_entry_from_config_caches_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests that load_entry_from_config caches parent and entry from the core loader."""
    parent = {
        "source": "config",
        "z": np.arange(3, dtype=float),
        "nz": np.ones(3),
    }
    last = {"tomo_spec": {"kind": "photoz"}, "bins": None, "tomo_meta": None}

    def _fake_load_entry_from_config(
        **kwargs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return dict(parent), dict(last)

    monkeypatch.setattr(nzmod, "_load_entry_from_config", _fake_load_entry_from_config)

    t = NZTomography()
    t.load_entry_from_config(config_file="dummy.yaml", key="survey")

    assert t.has_parent() is True
    assert t.has_entry() is True
    assert t.has_bins() is False
    assert t.parent_source() == "config"
    assert t.kind() == "photoz"


def test_kind_handles_non_mapping_spec() -> None:
    """Tests that kind falls back to default if cached tomo_spec is not a mapping."""
    t = NZTomography()
    t._last = {"tomo_spec": "nope"}  # type: ignore[assignment]
    assert t.kind() == "photoz"
    assert t.kind(default="specz") == "specz"


def test_kind_strips_and_lowers_when_present() -> None:
    """Tests that kind normalizes whitespace/case from cached tomo_spec.kind."""
    t = NZTomography()
    t._last = {"tomo_spec": {"kind": "  SpEcZ  "}}  # type: ignore[assignment]
    assert t.kind() == "specz"


def test_bins_accessor_requires_build(z: np.ndarray, nz: np.ndarray) -> None:
    """Tests that bins() raises if tomo_spec is cached but no bins have been built."""
    t = NZTomography()
    t._parent = {"source": "arrays", "z": z, "nz": nz}  # type: ignore[assignment]
    t._last = {"tomo_spec": {"kind": "photoz"}, "bins": None}  # type: ignore[assignment]

    with pytest.raises(ValueError, match=r"No bins cached"):
        _ = t.bins()


def test_tomo_meta_requires_entry(z: np.ndarray, nz: np.ndarray) -> None:
    """Tests that tomo_meta() raises if no tomo_spec is cached."""
    t = NZTomography()
    t._parent = {"source": "arrays", "z": z, "nz": nz}  # type: ignore[assignment]

    with pytest.raises(ValueError, match=r"No tomo_spec cached"):
        _ = t.tomo_meta()
