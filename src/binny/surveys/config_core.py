from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from binny.surveys.config_utils import (
    _build_parent_nz,
    _builder_kwargs_from_spec,
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
    "build_bins_from_config",
    "build_bins_from_mapping",
    "build_bins_from_arrays",
]


def _norm_kind(kind: str) -> str:
    return str(kind).strip().lower()


def _require_kind(spec: Mapping[str, Any], *, kind: str) -> None:
    k = _norm_kind(kind)
    if _norm_kind(spec["kind"]) != k:
        raise ValueError(f"kind must be {k!r} for this helper.")


def _call_builder(
    *,
    builder,
    z_arr: np.ndarray,
    nz_arr: np.ndarray,
    spec: Mapping[str, Any],
    include_tomo_metadata: bool,
) -> tuple[Any, Any | None]:
    kwargs = _builder_kwargs_from_spec(spec)

    out = builder(
        z=z_arr,
        nz=nz_arr,
        include_metadata=include_tomo_metadata,
        **kwargs,
    )

    if include_tomo_metadata:
        bins_out, tomo_meta = out
        return bins_out, tomo_meta

    return out, None


def _shape_build_return(
    *,
    bins_out: Any,
    survey_meta: dict[str, Any] | None,
    tomo_meta: Any | None,
    include_survey_metadata: bool,
    include_tomo_metadata: bool,
):
    if include_survey_metadata and include_tomo_metadata:
        return bins_out, survey_meta, tomo_meta
    if include_survey_metadata:
        return bins_out, survey_meta
    if include_tomo_metadata:
        return bins_out, tomo_meta
    return bins_out


def build_bins_from_mapping(
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
    """Build tomographic bins from an in-memory mapping using a provided builder."""
    cfg = _require_mapping(cfg, what="cfg")

    z_arr = _extract_z_grid(cfg, z)

    entries = _iter_tomography_entries(cfg)
    matches = _select_entries(entries, role=role, year=year)
    entry = _require_single(matches, what="tomography entry")

    spec = _parse_entry(entry)
    _require_kind(spec, kind=kind)

    nz_arr = _build_parent_nz(entry, z_arr)

    survey_meta = (
        _survey_meta(cfg=cfg, resolved_key=str(key), role=spec["role"], year=spec["year"])
        if include_survey_metadata
        else None
    )

    bins_out, tomo_meta = _call_builder(
        builder=builder,
        z_arr=z_arr,
        nz_arr=nz_arr,
        spec=spec,
        include_tomo_metadata=include_tomo_metadata,
    )

    return _shape_build_return(
        bins_out=bins_out,
        survey_meta=survey_meta,
        tomo_meta=tomo_meta,
        include_survey_metadata=include_survey_metadata,
        include_tomo_metadata=include_tomo_metadata,
    )


def build_bins_from_config(
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
    """Build tomographic bins from YAML using a provided builder."""
    cfg, resolved_key = _resolve_config_entry(config_file=config_file, key=key)
    return build_bins_from_mapping(
        kind=kind,
        builder=builder,
        cfg=cfg,
        key=resolved_key,
        role=role,
        year=year,
        z=z,
        include_survey_metadata=include_survey_metadata,
        include_tomo_metadata=include_tomo_metadata,
    )


def build_bins_from_arrays(
    *,
    kind: str,
    builder,
    z: Any,
    nz: Any,
    tomo_spec: Mapping[str, Any],
    include_tomo_metadata: bool = False,
):
    """Build tomographic bins from in-memory (z, nz, tomo_spec)."""
    z_arr = np.asarray(z, dtype=float)
    nz_arr = np.asarray(nz, dtype=float)

    spec = _parse_entry(tomo_spec)
    _require_kind(spec, kind=kind)

    bins_out, tomo_meta = _call_builder(
        builder=builder,
        z_arr=z_arr,
        nz_arr=nz_arr,
        spec=spec,
        include_tomo_metadata=include_tomo_metadata,
    )

    return (bins_out, tomo_meta) if include_tomo_metadata else bins_out
