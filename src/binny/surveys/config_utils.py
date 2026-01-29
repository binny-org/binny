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

from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import yaml

from binny.axes.grids import linear_grid
from binny.nz.registry import nz_model

__all__ = [
    "SurveyFootprint",
    "list_configs",
    "config_path",
]

_CONFIG_PKG = "binny.surveys.configs"


class SurveyFootprint(TypedDict, total=False):
    """Typed footprint metadata returned from survey configs."""

    survey_area: float
    frac_sky: float


def _load_yaml_mapping(path: str | Path) -> Mapping[Any, Any]:
    """Loads a YAML file and requires the root document to be a mapping."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise ValueError("Config root must be a mapping.")
    return data


def _require_mapping(obj: Any, *, what: str) -> Mapping[Any, Any]:
    """Validates that an object is a mapping and returns it."""
    if not isinstance(obj, Mapping):
        raise ValueError(f"{what} must be a mapping.")
    return obj


def list_configs() -> list[str]:
    """Lists shipped YAML configuration files."""
    root = resources.files(_CONFIG_PKG)
    return sorted(
        p.name for p in root.iterdir() if p.is_file() and p.name.endswith((".yaml", ".yml"))
    )


def config_path(filename: str) -> Path:
    """Resolves a shipped configuration filename to a local filesystem path."""
    root = resources.files(_CONFIG_PKG)
    p = root / filename
    if not p.is_file():
        raise FileNotFoundError(
            f"No shipped config named {filename!r}. Available: {list_configs()}"
        )
    with resources.as_file(p) as real_path:
        return Path(real_path)


def _resolve_config_entry(
    *,
    config_file: str | Path,
    key: str | None = None,
) -> tuple[Mapping[Any, Any], str]:
    """Loads a survey config file for the new flat schema.

    The config file MUST be a single mapping with keys like:
    name, survey_meta, z_grid, nz, tomography.

    Args:
        config_file: Path to a YAML file, or the filename of a shipped config.
        key: Not supported in the new schema. Must be None.

    Returns:
        Tuple of (cfg mapping, resolved_key_label).

    Raises:
        ValueError: If key is provided or required blocks are missing.
    """
    if key is not None:
        raise ValueError(
            "This config schema does not support top-level keys. "
            "Pass a YAML file whose root is the config mapping, and omit key=..."
        )

    p = Path(config_file)
    if not p.exists():
        p = config_path(str(config_file))

    cfg = _load_yaml_mapping(p)

    # Light schema sanity checks (cheap and helpful)
    if "tomography" not in cfg:
        raise ValueError("Config must contain a 'tomography' list.")

    resolved = str(cfg.get("name") or p.stem)
    return cfg, resolved


def _normalize_label(x: Any | None) -> str | None:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _extract_z_grid(cfg: Mapping[Any, Any], z: Any | None) -> np.ndarray:
    """Returns true-z grid from override, cfg.z_grid, else a package default."""
    if z is not None:
        return np.asarray(z, dtype=np.float64)

    zspec = cfg.get("z_grid")
    if zspec is None:
        # Package default; document it in schema docs.
        return linear_grid(0.0, 3.0, 301)

    zspec = _require_mapping(zspec, what="Config.z_grid")
    try:
        z_min = float(zspec["start"])
        z_max = float(zspec["stop"])
        n = int(zspec["n"])
    except KeyError as e:
        raise ValueError("z_grid must contain keys: start, stop, n") from e

    return linear_grid(z_min, z_max, n)


def _extract_survey_meta(cfg: Mapping[Any, Any]) -> dict[str, Any] | None:
    """Returns cfg.survey_meta if present."""
    meta = cfg.get("survey_meta")
    if meta is None:
        return None
    return dict(_require_mapping(meta, what="Config.survey_meta"))


def _build_parent_nz(entry: Mapping[Any, Any], z: np.ndarray) -> np.ndarray:
    """Builds the parent/underlying n(z) from entry.nz."""
    nz_cfg = _require_mapping(entry.get("nz"), what="tomography entry.nz")
    try:
        model = str(nz_cfg["model"])
    except KeyError as e:
        raise ValueError("tomography entry.nz must contain a 'model' field.") from e

    params = nz_cfg.get("params") or {}
    if not isinstance(params, Mapping):
        raise ValueError("tomography entry.nz.params must be a mapping.")
    return nz_model(model, z, **dict(params))


def _iter_tomography_entries(cfg: Mapping[Any, Any]) -> list[Mapping[Any, Any]]:
    tomo = cfg.get("tomography")
    if tomo is None:
        raise ValueError("Config must contain a 'tomography' list.")
    if not isinstance(tomo, Sequence) or isinstance(tomo, str | bytes):
        raise ValueError("tomography must be a list of mappings.")
    out: list[Mapping[Any, Any]] = []
    for i, item in enumerate(tomo):
        out.append(_require_mapping(item, what=f"tomography[{i}]"))
    return out


def _select_entries(
    entries: list[Mapping[Any, Any]],
    *,
    role: Any | None,
    year: Any | None,
) -> list[Mapping[Any, Any]]:
    """Filters entries by optional role/year. If neither provided, returns all."""
    role_s = _normalize_label(role)
    year_s = _normalize_label(year)

    if role_s is None and year_s is None:
        return entries

    def match(e: Mapping[Any, Any]) -> bool:
        if role_s is not None and _normalize_label(e.get("role")) != role_s:
            return False
        if year_s is not None and _normalize_label(e.get("year")) != year_s:
            return False
        return True

    return [e for e in entries if match(e)]


def _require_single(entries: list[Mapping[Any, Any]], *, what: str) -> Mapping[Any, Any]:
    if len(entries) != 1:
        raise ValueError(f"{what} is ambiguous; matched {len(entries)} entries.")
    return entries[0]


def _parse_bins(bins_block: Any) -> dict[str, Any]:
    """Normalizes bins to either {'edges': array} or {'scheme': str, 'n_bins': int, ...}."""
    bins = _require_mapping(bins_block, what="bins")

    edges = bins.get("edges")
    scheme = bins.get("scheme")
    n_bins = bins.get("n_bins")
    z_range = bins.get("range")

    if edges is not None:
        if scheme is not None or n_bins is not None or z_range is not None:
            raise ValueError("bins: if 'edges' is provided, do not provide scheme/n_bins/range.")
        edges_arr = np.asarray(edges, dtype=np.float64)
        if edges_arr.ndim != 1 or edges_arr.size < 2:
            raise ValueError("bins.edges must be a 1D sequence with at least 2 values.")
        return {"edges": edges_arr}

    if scheme is None:
        raise ValueError("bins must provide either 'edges' or 'scheme' + 'n_bins'.")
    if n_bins is None:
        raise ValueError("bins.n_bins is required when using bins.scheme.")

    out: dict[str, Any] = {"scheme": str(scheme), "n_bins": int(n_bins)}
    if z_range is not None:
        zlo, zhi = map(float, z_range)
        out["range"] = (zlo, zhi)
    return out


def _parse_entry(entry: Mapping[Any, Any]) -> dict[str, Any]:
    """Returns a normalized spec dict for a single tomography entry."""
    kind = str(entry.get("kind", "photoz")).strip().lower()
    if kind not in {"photoz", "specz"}:
        raise ValueError("tomography entry kind must be 'photoz' or 'specz'.")

    nz_block = entry.get("nz")
    if nz_block is None:
        raise ValueError("tomography entry must contain an 'nz' mapping.")
    nz_block = _require_mapping(nz_block, what="tomography entry.nz")

    bins = _parse_bins(entry.get("bins"))

    unc = entry.get("uncertainties") or {}
    if not isinstance(unc, Mapping):
        raise ValueError("uncertainties must be a mapping if provided.")
    unc = dict(unc)

    return {
        "role": _normalize_label(entry.get("role")),
        "year": _normalize_label(entry.get("year")),
        "name": _normalize_label(entry.get("name")),
        "kind": kind,
        "nz": dict(nz_block),
        "bins": bins,
        "uncertainties": unc,
        "n_gal_arcmin2": (
            float(entry["n_gal_arcmin2"]) if entry.get("n_gal_arcmin2") is not None else None
        ),
    }


def _survey_meta(
    *,
    cfg: Mapping[Any, Any],
    resolved_key: str,
    role: str | None,
    year: str | None,
) -> dict[str, Any]:
    """Builds standardized survey metadata for the selection."""
    return {
        "survey": str(cfg.get("name", resolved_key)),
        "key": resolved_key,
        "role": role,
        "year": year,
        "survey_meta": _extract_survey_meta(cfg),
    }


def _builder_kwargs_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """
    Translate schema-facing bins/uncertainties -> builder-facing kwargs.

    Schema bins:
      - edges: [...]
        OR
      - scheme: str
        n_bins: int
      - range: [zmin, zmax] (optional)

    Builder kwargs:
      - bin_edges
      - binning_scheme
      - n_bins
      - bin_range (optional)
      + uncertainties passthrough
    """
    if "bins" not in spec or not isinstance(spec["bins"], Mapping):
        raise ValueError("tomo_spec must contain a 'bins' mapping.")

    bins = spec["bins"]

    # bin definition: explicit edges OR scheme+n_bins
    if "edges" in bins:
        kw: dict[str, Any] = {
            "bin_edges": bins["edges"],
            "binning_scheme": None,
            "n_bins": None,
        }
    else:
        kw = {
            "bin_edges": None,
            "binning_scheme": bins["scheme"],
            "n_bins": bins["n_bins"],
        }

    # passthrough builder kwargs
    params: dict[str, Any] = dict(spec.get("uncertainties") or {})

    # normalize range naming (optional)
    if "range" in bins:
        params.setdefault("bin_range", bins["range"])

    kw.update(params)
    return kw
