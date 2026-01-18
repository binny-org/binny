"""Survey config parsing utilities (new schema).

This module provides small, reusable helpers for reading survey YAML files and
selecting entries from a flat tomography list schema.

Schema (new)::

    name: <optional str>
    survey_meta: <optional mapping>            # ignored unless requested
    z_grid: {start: float, stop: float, n: int}  # optional; default used if missing
    nz: {model: str, params: {…}}             # required parent distribution
    tomography:                               # required
      - role: <optional str>
        year: <optional str>
        n_gal_arcmin2: <optional float>
        kind: photoz|specz                    # optional; defaults to photoz
        bins:
          edges: [...]                        # explicit edges
          # OR
          scheme: <str>
          n_bins: <int>
          range: [z_min, z_max]               # optional
        uncertainties: {…}                    # optional kwargs passed to builders
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from binny.surveys.config_utils import (
    _build_parent_nz,
    _extract_z_grid,
    _iter_tomography_entries,
    _parse_entry,
    _require_mapping,
    _require_single,
    _resolve_config_entry,
    _select_entries,
    _survey_meta,
)

__all__ = [
    "survey_from_config",
    "survey_from_mapping",
    "spec_from_config",
    "spec_from_mapping",
    "build_from_config",
    "build_from_mapping",
    "build_from_arrays",
]

_CONFIG_PKG = "binny.surveys.configs"


def survey_from_config(
    *,
    config_file: str | Path,
    key: str | None = None,
    role: str | None = None,
    year: Any | None = None,
    z: Any | None = None,
    include_survey_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any] | None]:
    """Loads a config and returns (z, nz, tomo_spec, optional meta)."""
    cfg, resolved_key = _resolve_config_entry(config_file=config_file, key=None)
    return survey_from_mapping(
        cfg=cfg,
        key=resolved_key,
        role=role,
        year=year,
        z=z,
        include_survey_metadata=include_survey_metadata,
    )


def survey_from_mapping(
    *,
    cfg: Mapping[Any, Any],
    key: str = "survey",
    role: str | None = None,
    year: Any | None = None,
    z: Any | None = None,
    include_survey_metadata: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any] | None]:
    """Parses a mapping and returns (z, nz, tomo_spec, optional meta)."""
    cfg = _require_mapping(cfg, what="cfg")

    z_arr = _extract_z_grid(cfg, z)

    entries = _iter_tomography_entries(cfg)
    matches = _select_entries(entries, role=role, year=year)
    entry = _require_single(matches, what="tomography entry")

    spec = _parse_entry(entry)
    nz_arr = _build_parent_nz(entry, z_arr)

    meta = (
        _survey_meta(cfg=cfg, resolved_key=str(key), role=spec["role"], year=spec["year"])
        if include_survey_metadata
        else None
    )
    return z_arr, nz_arr, spec, meta


def spec_from_config(
    *,
    kind: str,
    config_file: str | Path,
    key: str | None = None,
    role: str | None = None,
    year: Any | None = None,
    z: Any | None = None,
    include_survey_metadata: bool = False,
):
    """Returns the parsed spec (z, nz, spec[, meta]) for a requested kind."""
    z_arr, nz_arr, spec, meta = survey_from_config(
        config_file=config_file,
        key=key,
        role=role,
        year=year,
        z=z,
        include_survey_metadata=include_survey_metadata,
    )
    if spec["kind"] != str(kind).strip().lower():
        raise ValueError(f"kind must be {str(kind).strip().lower()!r} for this helper.")
    return (z_arr, nz_arr, spec, meta) if include_survey_metadata else (z_arr, nz_arr, spec)


def spec_from_mapping(
    *,
    kind: str,
    cfg: Mapping[Any, Any],
    key: str = "survey",
    role: str | None = None,
    year: Any | None = None,
    z: Any | None = None,
    include_survey_metadata: bool = False,
):
    """Returns the parsed spec (z, nz, spec[, meta]) for a requested kind."""
    z_arr, nz_arr, spec, meta = survey_from_mapping(
        cfg=cfg,
        key=key,
        role=role,
        year=year,
        z=z,
        include_survey_metadata=include_survey_metadata,
    )
    if spec["kind"] != str(kind).strip().lower():
        raise ValueError(f"kind must be {str(kind).strip().lower()!r} for this helper.")
    return (z_arr, nz_arr, spec, meta) if include_survey_metadata else (z_arr, nz_arr, spec)


def build_from_config(
    *,
    kind: str,
    builder,
    config_file: str | Path,
    key: str | None = None,
    role: str | None = None,
    year: Any | None = None,
    z: Any | None = None,
    include_survey_metadata: bool = False,
    include_tomo_metadata: bool = False,
):
    """Builds tomographic bins from a YAML config using a provided builder."""
    z_arr, nz_arr, spec, survey_meta = survey_from_config(
        config_file=config_file,
        key=key,
        role=role,
        year=year,
        z=z,
        include_survey_metadata=include_survey_metadata,
    )
    if spec["kind"] != str(kind).strip().lower():
        raise ValueError(f"kind must be {str(kind).strip().lower()!r} for this helper.")

    bins = spec["bins"]
    if "edges" in bins:
        bin_edges = bins["edges"]
        binning_scheme = None
        n_bins = None
    else:
        bin_edges = None
        binning_scheme = bins["scheme"]
        n_bins = bins["n_bins"]

    params: dict[str, Any] = dict(spec.get("uncertainties") or {})
    if "range" in bins:
        # many bin builders accept bin_range; normalize naming here if you want
        params.setdefault("bin_range", bins["range"])

    out = builder(
        z=z_arr,
        nz=nz_arr,
        bin_edges=bin_edges,
        binning_scheme=binning_scheme,
        n_bins=n_bins,
        include_metadata=include_tomo_metadata,
        **params,
    )

    if include_tomo_metadata:
        bins_out, tomo_meta = out
    else:
        bins_out, tomo_meta = out, None

    if include_survey_metadata and include_tomo_metadata:
        return bins_out, survey_meta, tomo_meta
    if include_survey_metadata:
        return bins_out, survey_meta
    if include_tomo_metadata:
        return bins_out, tomo_meta
    return bins_out


def build_from_mapping(
    *,
    kind: str,
    builder,
    cfg: Mapping[Any, Any],
    key: str = "survey",
    role: str | None = None,
    year: Any | None = None,
    z: Any | None = None,
    include_survey_metadata: bool = False,
    include_tomo_metadata: bool = False,
):
    """Builds tomographic bins from an in-memory mapping using a provided builder."""
    z_arr, nz_arr, spec, survey_meta = survey_from_mapping(
        cfg=cfg,
        key=key,
        role=role,
        year=year,
        z=z,
        include_survey_metadata=include_survey_metadata,
    )
    if spec["kind"] != str(kind).strip().lower():
        raise ValueError(f"kind must be {str(kind).strip().lower()!r} for this helper.")

    bins = spec["bins"]
    if "edges" in bins:
        bin_edges = bins["edges"]
        binning_scheme = None
        n_bins = None
    else:
        bin_edges = None
        binning_scheme = bins["scheme"]
        n_bins = bins["n_bins"]

    params: dict[str, Any] = dict(spec.get("uncertainties") or {})
    if "range" in bins:
        params.setdefault("bin_range", bins["range"])

    out = builder(
        z=z_arr,
        nz=nz_arr,
        bin_edges=bin_edges,
        binning_scheme=binning_scheme,
        n_bins=n_bins,
        include_metadata=include_tomo_metadata,
        **params,
    )

    if include_tomo_metadata:
        bins_out, tomo_meta = out
    else:
        bins_out, tomo_meta = out, None

    if include_survey_metadata and include_tomo_metadata:
        return bins_out, survey_meta, tomo_meta
    if include_survey_metadata:
        return bins_out, survey_meta
    if include_tomo_metadata:
        return bins_out, tomo_meta
    return bins_out


def build_from_arrays(
    *,
    kind: str,
    builder,
    z: Any,
    nz: Any,
    tomo_spec: Mapping[str, Any],
    include_tomo_metadata: bool = False,
):
    """Builds tomographic bins from in-memory (z, nz, entry_spec)."""
    z_arr = np.asarray(z, dtype=float)
    nz_arr = np.asarray(nz, dtype=float)

    spec = _parse_entry(tomo_spec)
    if spec["kind"] != str(kind).strip().lower():
        raise ValueError(f"kind must be {str(kind).strip().lower()!r} for this helper.")

    bins = spec["bins"]
    if "edges" in bins:
        bin_edges = bins["edges"]
        binning_scheme = None
        n_bins = None
    else:
        bin_edges = None
        binning_scheme = bins["scheme"]
        n_bins = bins["n_bins"]

    params: dict[str, Any] = dict(spec.get("uncertainties") or {})
    if "range" in bins:
        params.setdefault("bin_range", bins["range"])

    out = builder(
        z=z_arr,
        nz=nz_arr,
        bin_edges=bin_edges,
        binning_scheme=binning_scheme,
        n_bins=n_bins,
        include_metadata=include_tomo_metadata,
        **params,
    )

    if include_tomo_metadata:
        bins_out, tomo_meta = out
        return bins_out, tomo_meta
    return out
