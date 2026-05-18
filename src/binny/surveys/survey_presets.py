"""Helpers for working with built-in Binny survey configurations.

This module provides a small public interface for discovering, loading, and
displaying the survey YAML files distributed with Binny. These helpers let users
access survey presets such as LSST, DESI, or Roman by name, without needing to
know where the YAML files live inside the installed package.

The returned configurations use the same dictionary structure as the YAML files
and can be passed directly to :class:`binny.NZTomography`.
"""

from __future__ import annotations

from typing import Any

import yaml

from binny.surveys.config_utils import list_configs, load_config


def load_survey_config(survey: str) -> dict[str, Any]:
    """Load a built-in survey configuration by survey name.

    Args:
        survey: Survey preset name, such as ``"lsst"``, ``"desi"``, or
            ``"roman"``.

    Returns:
        Parsed survey configuration dictionary.
    """
    name = str(survey).strip().lower()
    return dict(load_config(f"{name}_survey_specs.yaml"))


def list_survey_configs() -> list[str]:
    """List built-in survey configuration names."""
    surveys = []

    for filename in list_configs():
        if filename.endswith("_survey_specs.yaml"):
            surveys.append(filename.removesuffix("_survey_specs.yaml"))

    return sorted(set(surveys))


def show_survey_config(
    survey: str,
    *,
    print_output: bool = True,
) -> str:
    """Show a built-in survey configuration as YAML text.

    Args:
        survey: Survey preset name, such as ``"lsst"``, ``"desi"``, or
            ``"roman"``.
        print_output: Whether to print the YAML text before returning it.

    Returns:
        The selected survey configuration formatted as YAML.
    """
    cfg = load_survey_config(survey)
    text = yaml.safe_dump(cfg, sort_keys=False)

    if print_output:
        print(text)

    return text
