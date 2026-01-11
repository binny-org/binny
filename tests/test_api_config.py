"""Unit tests for binny.api.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from binny.api.config import SurveyConfig, load_config


def test_load_config_happy_path(tmp_path: Path) -> None:
    """Tests that load_config returns a SurveyConfig instance."""
    p = tmp_path / "config.yaml"
    p.write_text("survey: demo\nn_bins: 5\n", encoding="utf-8")

    cfg = load_config(p)

    assert isinstance(cfg, SurveyConfig)
    assert cfg.raw["survey"] == "demo"
    assert cfg.raw["n_bins"] == 5


def test_load_config_accepts_str_path(tmp_path: Path) -> None:
    """Tests that load_config accepts a str path."""
    p = tmp_path / "config.yaml"
    p.write_text("a: 1\n", encoding="utf-8")

    cfg = load_config(str(p))

    assert cfg.raw == {"a": 1}


def test_load_config_rejects_non_mapping_root(tmp_path: Path) -> None:
    """Tests that load_config rejects non-mapping root."""
    p = tmp_path / "config.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Config root must be a mapping\."):
        load_config(p)


def test_load_config_rejects_empty_file(tmp_path: Path) -> None:
    """Tests that load_config rejects empty file."""
    p = tmp_path / "config.yaml"
    p.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Config root must be a mapping\."):
        load_config(p)
