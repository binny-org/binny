"""Internal session helpers for NZTomography.

These functions implement the non-user-facing plumbing:
- resolve tomo builders from a kind string
- normalize and cache parent state (z, nz, meta)
- load a single tomography entry from config/mapping
- run the shared build_from_arrays wiring

This module is intentionally not part of the public API surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from binny.surveys.config_core import (
    build_from_arrays,
    survey_from_config,
    survey_from_mapping,
)


def resolve_tomo_builder(kind: str):
    """Resolve a tomo builder callable for a requested kind."""
    k = str(kind).strip().lower()
    if k == "photoz":
        from binny.nz_tomo.photoz import build_photoz_bins as builder  # type:
        # ignore

        return builder
    if k == "specz":
        from binny.nz_tomo.specz import build_specz_bins as builder  # type:
        # ignore

        return builder
    raise ValueError(f"Unknown kind {k!r}. Expected 'photoz' or 'specz'.")


def make_parent_from_arrays(
    *,
    z: Any,
    nz: Any,
    survey_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a normalized parent-state dict from in-memory arrays."""
    return {
        "source": "arrays",
        "z": np.asarray(z, dtype=float),
        "nz": np.asarray(nz, dtype=float),
        "survey_meta": dict(survey_meta) if survey_meta is not None else None,
        "cfg": None,
        "config_file": None,
        "key": None,
    }


def load_entry_from_config(
    *,
    config_file: str | Path,
    key: str | None = None,
    role: str | None = None,
    year: Any | None = None,
    z: Any | None = None,
    include_survey_metadata: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (parent_state, last_state) by reading a single entry from YAML."""
    z_arr, nz_arr, spec, survey_meta = survey_from_config(
        config_file=config_file,
        key=key,
        role=role,
        year=year,
        z=z,
        include_survey_metadata=include_survey_metadata,
    )

    parent = {
        "source": "config",
        "z": z_arr,
        "nz": nz_arr,
        "survey_meta": survey_meta if include_survey_metadata else None,
        "cfg": None,
        "config_file": config_file,
        "key": key,
    }
    last = {
        "tomo_spec": dict(spec),
        "bins": None,
        "tomo_meta": None,
        "kind": str(spec.get("kind", "photoz")).strip().lower(),
    }
    return parent, last


def load_entry_from_mapping(
    *,
    cfg: Mapping[Any, Any],
    key: str = "survey",
    role: str | None = None,
    year: Any | None = None,
    z: Any | None = None,
    include_survey_metadata: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (parent_state, last_state) by reading a single entry from a mapping."""
    z_arr, nz_arr, spec, survey_meta = survey_from_mapping(
        cfg=cfg,
        key=key,
        role=role,
        year=year,
        z=z,
        include_survey_metadata=include_survey_metadata,
    )

    parent = {
        "source": "mapping",
        "z": z_arr,
        "nz": nz_arr,
        "survey_meta": survey_meta if include_survey_metadata else None,
        "cfg": cfg,
        "config_file": None,
        "key": key,
    }
    last = {
        "tomo_spec": dict(spec),
        "bins": None,
        "tomo_meta": None,
        "kind": str(spec.get("kind", "photoz")).strip().lower(),
    }
    return parent, last


def build_bins_from_state(
    *,
    parent: dict[str, Any],
    last: dict[str, Any],
    include_metadata: bool,
    kind: str | None = None,
    overrides: Mapping[str, Any] | None = None,
):
    """Run build_from_arrays using cached parent/spec and return updated last + output."""
    spec = dict(last["tomo_spec"])
    if overrides:
        spec.update(dict(overrides))

    kind_use = (
        str(kind).strip().lower()
        if kind is not None
        else str(spec.get("kind", "photoz")).strip().lower()
    )

    builder = resolve_tomo_builder(kind_use)

    out = build_from_arrays(
        kind=kind_use,
        builder=builder,
        z=parent["z"],
        nz=parent["nz"],
        tomo_spec=spec,
        include_tomo_metadata=include_metadata,
    )

    if include_metadata:
        bins_out, tomo_meta = out
    else:
        bins_out, tomo_meta = out, None

    last2 = {
        "kind": kind_use,
        "tomo_spec": spec,
        "bins": bins_out,
        "tomo_meta": tomo_meta,
    }
    return last2, (bins_out, tomo_meta) if include_metadata else bins_out
