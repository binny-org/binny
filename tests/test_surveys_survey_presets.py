"""Unit tests for ``binny.surveys.survey_presets``."""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from binny.surveys import survey_presets
from binny.surveys.survey_presets import (
    list_survey_configs,
    load_survey_config,
    show_survey_config,
)


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
