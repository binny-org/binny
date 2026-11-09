"""Helpers for working with built-in Binny survey configurations.

This module provides a small public interface for discovering, loading, and
displaying the survey YAML files distributed with Binny. These helpers let users
access survey presets such as LSST, DESI, or Roman by name, without needing to
know where the YAML files live inside the installed package.

The returned configurations use the same dictionary structure as the YAML files
and can be used with ``build_survey_tomography``.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

import binny.surveys.config_utils as cu
from binny.surveys.config_utils import list_configs, load_config
from binny.tomography.builders import build_tomography_from_spec
from binny.tomography.core import Tomography


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


def build_survey_tomography(
    survey: str | None = None,
    *,
    config_file: str | Path | None = None,
    cfg: Mapping[Any, Any] | None = None,
    z: Any | None = None,
    role: str | None = None,
    year: Any | None = None,
    scenario: Any | None = None,
    sample: Any | None = None,
    overrides: Mapping[str, Any] | None = None,
    include_survey_metadata: bool = False,
    include_tomo_metadata: bool = False,
    include_photoz_diagnostics: bool = False,
) -> Tomography:
    """Build tomography from a survey preset or configuration.

    Args:
        survey: Built-in survey name, such as ``"lsst"`` or ``"desi"``.
        config_file: Path to a custom YAML file or shipped configuration.
        cfg: Optional in-memory survey configuration.
        z: Optional redshift grid overriding the configured grid.
        role: Optional tomography role selector.
        year: Optional survey year selector.
        scenario: Optional scenario selector.
        sample: Optional sample selector.
        overrides: Optional tomography specification overrides. Nested
            mappings are merged one level deep.
        include_survey_metadata: Whether to attach survey metadata.
        include_tomo_metadata: Whether to include tomography metadata.
        include_photoz_diagnostics: Whether to include photo-z diagnostics.

    Returns:
        Tomography containing the parent redshift distribution, bins,
        resolved specification, and requested metadata.

    Raises:
        ValueError: If configuration sources or tomography selection are invalid.
        FileNotFoundError: If the requested configuration cannot be found.
    """
    if sum(value is not None for value in (survey, config_file, cfg)) != 1:
        raise ValueError("Provide exactly one of survey, config_file, or cfg.")

    if survey is not None:
        name = str(survey).strip().lower()
        config_file = f"{name}_survey_specs.yaml"

    resolved_key = "survey"

    if config_file is not None:
        cfg, resolved_key = cu._resolve_config_entry(config_file=config_file)

    cfg = cu._require_mapping(cfg, what="cfg")

    entry = cu._require_single(
        cu._select_entries(
            cu._iter_tomography_entries(cfg),
            role=role,
            year=year,
            scenario=scenario,
            sample=sample,
        ),
        what="tomography entry",
    )

    spec = dict(entry)

    if overrides:
        for key, value in overrides.items():
            if isinstance(value, Mapping) and isinstance(spec.get(key), Mapping):
                spec[key] = {**spec[key], **value}
            else:
                spec[key] = value

    spec = cu._parse_entry(spec)

    if spec["kind"] == "predefined":
        z_arr = None
        nz_arr = None
    else:
        z_arr = cu._extract_z_grid(cfg, z)
        nz_arr = cu._build_parent_nz(spec, z_arr)

    result = build_tomography_from_spec(
        z=z_arr,
        nz=nz_arr,
        tomo_spec=spec,
        include_tomo_metadata=include_tomo_metadata,
        include_photoz_diagnostics=include_photoz_diagnostics,
    )

    if include_survey_metadata:
        result.survey_metadata = cu._survey_meta(
            cfg=cfg,
            resolved_key=resolved_key,
            role=spec["role"],
            year=spec["year"],
            scenario=spec["scenario"],
            sample=spec["sample"],
        )

    return result
