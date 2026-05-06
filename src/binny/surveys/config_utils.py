"""Survey config parsing utilities.

This module provides small, reusable helpers for reading survey YAML files and
selecting entries from a flat tomography list schema.

Schema::

    name: <optional str>
    survey_meta: <optional mapping>  # ignored unless requested
    z_grid: {start: float, stop: float, n: int}  # optional
    nz: {model: str, params: {…}}  # required parent distribution
    tomography:  # required
      - role: <optional str>
        year: <optional str>
        n_gal_arcmin2: <optional float>
        kind: photoz|specz  # optional; defaults to photoz
        bins:
          edges: [...]  # explicit edges
          # OR
          scheme: <str>
          n_bins: <int>
          range: [z_min, z_max]  # optional
        uncertainties: {…}  # optional kwargs passed to builders
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
    """Loads a YAML file and require a mapping at the document root.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed YAML content as a mapping.

    Raises:
        ValueError: If the YAML root document is not a mapping.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, Mapping):
        raise ValueError("Config root must be a mapping.")
    return data


def _require_mapping(obj: Any, *, what: str) -> Mapping[Any, Any]:
    """Validates that an object is a mapping.

    This is a small schema helper used to enforce that nested YAML blocks
    (e.g., ``bins``, ``nz``, ``survey_meta``) are mappings before they are
    accessed.

    Args:
        obj: Object to validate.
        what: Human-readable label used in error messages.

    Returns:
        The input object, typed as a mapping.

    Raises:
        ValueError: If ``obj`` is not a mapping.
    """
    if not isinstance(obj, Mapping):
        raise ValueError(f"{what} must be a mapping.")
    return obj


def list_configs() -> list[str]:
    """Lists shipped survey configuration filenames.

    Returns:
        Sorted list of YAML filenames shipped in the ``binny.surveys.configs``
        package directory.
    """
    root = resources.files(_CONFIG_PKG)
    return sorted(
        p.name for p in root.iterdir() if p.is_file() and p.name.endswith((".yaml", ".yml"))
    )


def config_path(filename: str) -> Path:
    """Resolves a shipped config filename to a concrete filesystem path.

    This helper locates configuration files bundled with the package and
    returns a usable local path (via ``importlib.resources.as_file``).

    Args:
        filename: Shipped YAML filename (``.yaml`` or ``.yml``).

    Returns:
        Local filesystem path to the shipped config.

    Raises:
        FileNotFoundError: If no shipped config matches ``filename``.
    """
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
            "Pass a YAML file whose root is the config mapping, "
            "and omit key=..."
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
    """Normalizes a label-like field from config input.

    This trims whitespace and treats empty strings as missing. It is used for
    optional selector fields such as ``role``, ``year``, and ``name``.

    Args:
        x: Input value from YAML (or None).

    Returns:
        A stripped non-empty string, or None if missing/empty.
    """
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def _extract_z_grid(cfg: Mapping[Any, Any], z: Any | None) -> np.ndarray:
    """Selects the common true-redshift grid for all outputs.

    Priority order:
      1) explicit override ``z``
      2) ``cfg["z_grid"]`` block
      3) package default grid

    Args:
        cfg: Parsed config mapping.
        z: Optional override grid.

    Returns:
        One-dimensional true-redshift grid as ``float64``.

    Raises:
        ValueError: If ``cfg["z_grid"]`` is present but missing required keys.
    """
    if z is not None:
        return np.asarray(z, dtype=np.float64)

    zspec = cfg.get("z_grid")
    if zspec is None:
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
    """Extracts optional survey metadata passthrough.

    Args:
        cfg: Parsed config mapping.

    Returns:
        ``cfg["survey_meta"]`` as a plain dict if present, otherwise None.

    Raises:
        ValueError: If ``survey_meta`` is present but not a mapping.
    """
    meta = cfg.get("survey_meta")
    if meta is None:
        return None
    return dict(_require_mapping(meta, what="Config.survey_meta"))


def _build_parent_nz(entry: Mapping[Any, Any], z: np.ndarray) -> np.ndarray:
    """Builds the parent n(z) on the provided true-redshift grid.

    This reads the ``nz`` block and evaluates the registered n(z) model on
    the common grid. For tabulated distributions, this also supports either
    inline arrays or a file-backed ``source`` block.
    """
    nz_cfg = _require_mapping(entry.get("nz"), what="tomography entry.nz")
    try:
        model = str(nz_cfg["model"])
    except KeyError as e:
        raise ValueError("tomography entry.nz must contain a 'model' field.") from e

    params = nz_cfg.get("params") or {}
    if not isinstance(params, Mapping):
        raise ValueError("tomography entry.nz.params must be a mapping.")

    model_params = dict(params)

    if model == "tabulated":
        model_params.update(_tabulated_params_from_config(nz_cfg))

    return nz_model(model, z, **model_params)


def _tabulated_params_from_config(nz_cfg: Mapping[Any, Any]) -> dict[str, Any]:
    """Extract tabulated n(z) inputs from inline arrays or a source file."""
    if "z_input" in nz_cfg or "nz_input" in nz_cfg:
        if "z_input" not in nz_cfg or "nz_input" not in nz_cfg:
            raise ValueError(
                "Tabulated n(z) requires both 'z_input' and 'nz_input' when using inline arrays."
            )
        return {
            "z_input": nz_cfg["z_input"],
            "nz_input": nz_cfg["nz_input"],
        }

    source = nz_cfg.get("source")
    if source is None:
        return {}

    source = _require_mapping(source, what="tomography entry.nz.source")

    try:
        path = Path(source["path"])
    except KeyError as e:
        raise ValueError("tomography entry.nz.source must contain a 'path' field.") from e

    if not path.is_absolute():
        path = resources.files("binny.surveys.data") / str(path)

    with resources.as_file(path) as real_path:
        table = np.loadtxt(
            real_path,
            skiprows=int(source.get("skiprows", 0)),
        )

    z_col = int(source.get("z_col", 0))
    nz_col = int(source.get("nz_col", 1))

    return {
        "z_input": table[:, z_col],
        "nz_input": table[:, nz_col],
    }


def _iter_tomography_entries(cfg: Mapping[Any, Any]) -> list[Mapping[Any, Any]]:
    """Returns the validated tomography list from a config mapping.

    Args:
        cfg: Parsed config mapping.

    Returns:
        List of validated tomography entry mappings, in file order.

    Raises:
        ValueError: If ``cfg["tomography"]`` is missing or not a list of
            mappings.
    """
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
    """Filters entries by optional role/year.

    Args:
        entries: List of tomography entries.
        role: Optional role label.
        year: Optional year label.

    Returns:
        A subset of entries matching the given role/year labels.
    """
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
    """Validates that exactly one entry matches the filter.

    Args:
        entries: List of tomography entries.
        what: Description of the entries for error messages.

    Returns:
        A single matching entry.
    """
    if len(entries) != 1:
        raise ValueError(f"{what} is ambiguous; matched {len(entries)} entries.")
    return entries[0]


def _parse_bins(bins_block: Any) -> dict[str, Any]:
    """Parses and normalize a bin specification block.

    The bin specification must define *either* explicit bin edges *or* a
    binning scheme with a number of bins. Mixed specifications are rejected.

    Supported inputs:
      - Explicit edges via ``bins["edges"]``.
      - Parametric binning via ``bins["scheme"]`` and ``bins["n_bins"]``,
        optionally with a redshift range ``bins["range"]``.

    Args:
        bins_block: Mapping describing the bin configuration.

    Returns:
        A normalized dictionary describing the binning configuration.
        This contains either:
          - ``{"edges": np.ndarray}``, or
          - ``{"scheme": str, "n_bins": int, "range": (float, float)}``
            if a range is provided.

    Raises:
        ValueError: If the bin specification is missing required fields,
            mixes incompatible options, or contains invalid values.
    """
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
    """Parses and normalize a single tomography entry.

    This normalizes label-like fields (role/year/name), validates the entry
    kind, and ensures required nested blocks (nz, bins) are present.

    Args:
        entry: Tomography entry mapping from ``cfg["tomography"]``.

    Returns:
        Normalized entry dictionary with keys:
        ``role``, ``year``, ``name``, ``kind``, ``nz``, ``bins``,
        ``uncertainties``, and ``n_gal_arcmin2``.

    Raises:
        ValueError: If required blocks are missing, the kind is unsupported,
            or nested blocks are invalid.
    """
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

    entry_dict = {
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

    return entry_dict


def _survey_meta(
    *,
    cfg: Mapping[Any, Any],
    resolved_key: str,
    role: str | None,
    year: str | None,
) -> dict[str, Any]:
    """Build standardized metadata for a config selection.

    Args:
        cfg: Parsed config mapping.
        resolved_key: Resolved label for the config (e.g., cfg.name or stem).
        role: Optional role filter applied to tomography entries.
        year: Optional year filter applied to tomography entries.

    Returns:
        Metadata dictionary including the survey label, selection filters,
        and optional ``survey_meta`` passthrough.
    """
    return {
        "survey": str(cfg.get("name", resolved_key)),
        "key": resolved_key,
        "role": role,
        "year": year,
        "survey_meta": _extract_survey_meta(cfg),
    }


def _builder_kwargs_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Translates schema-facing bin/uncertainty fields into builder kwargs.

    This converts a normalized tomography spec into the keyword arguments
    expected by the tomo builders.

    Args:
        spec: Normalized tomography spec containing ``bins`` and optional
            ``uncertainties``.

    Returns:
        Dictionary of builder kwargs. This always includes:
        ``bin_edges``, ``binning_scheme``, and ``n_bins``. If a bin range is
        present in the schema, this adds ``bin_range`` (unless already
        provided in uncertainties). All uncertainty entries are passed through.

    Raises:
        ValueError: If ``spec["bins"]`` is missing or not a mapping.
    """
    if "bins" not in spec or not isinstance(spec["bins"], Mapping):
        raise ValueError("tomo_spec must contain a 'bins' mapping.")

    bins = spec["bins"]

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
