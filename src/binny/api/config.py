"""Module for loading survey configuration from a YAML file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SurveyConfig:
    raw: dict[str, Any]


def load_config(path: str | Path) -> SurveyConfig:
    p = Path(path)
    with p.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    return SurveyConfig(raw=data)
