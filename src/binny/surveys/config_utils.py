"""Survey config parsing utilities.

This module reads survey YAML files using a flat tomography-list schema. Each
tomography entry defines one sample, its parent redshift distribution, its bin
definition, and optional observational sample metadata.

Expected schema::

    name: <optional str>

    survey_meta:
      description: <optional str>
      footprint:
        nominal:
          survey_area: <optional float>
          frac_sky: <optional float>

    z_grid:
      start: <float>
      stop: <float>
      n: <int>

    tomography:
      - role: <optional str>
        year: <optional str>
        sample: <optional str>
        scenario: <optional str>
        name: <optional str>
        kind: photoz|specz

        nz:
          model: <str>
          params: <optional mapping>
          # For tabulated n(z), either inline arrays:
          z_input: <optional list[float]>
          nz_input: <optional list[float]>
          # Or a packaged source file:
          source:
            path: <str>
            skiprows: <optional int>
            z_col: <optional int>
            nz_col: <optional int>

        sample_properties:
          number_density:
            n_gal_arcmin2: <optional float>
            n_gal_comoving_h3_mpc3: <optional float | list[float]>
            dndz_deg2: <optional list[float]>
          shape_noise:
            sigma_e: <optional float>
          shot_noise:
            model: <optional str>
          volume:
            gpc3_hminus3: <optional float | list[float]>
          footprint:
            survey_area: <optional float>
            frac_sky: <optional float>

        bins:
          edges: <list[float]>
          # OR
          scheme: <str>
          n_bins: <int>
          range: <optional [float, float]>

        uncertainties:
          <builder kwargs>
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from binny.axes.grids import linear_grid
from binny.nz.registry import nz_model

__all__ = [
    "config_path",
    "list_configs",
]

_CONFIG_PKG = "binny.surveys.configs"
_DATA_PKG = "binny.surveys.data"


def list_configs() -> list[str]:
    """List shipped survey configuration filenames.

    Returns:
        Sorted YAML filenames shipped in the survey config package.
    """
    root = resources.files(_CONFIG_PKG)
    return sorted(
        p.name for p in root.iterdir() if p.is_file() and p.name.endswith((".yaml", ".yml"))
    )


def config_path(filename: str) -> Path:
    """Resolve a shipped survey config filename.

    Args:
        filename: Name of a bundled YAML config file.

    Returns:
        Filesystem path to the bundled config.

    Raises:
        FileNotFoundError: If no bundled config has the requested filename.
    """
    root = resources.files(_CONFIG_PKG)
    path = root / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"No shipped config named {filename!r}. Available: {list_configs()}"
        )

    with resources.as_file(path) as real_path:
        return Path(real_path)


def _load_yaml_mapping(path: str | Path) -> Mapping[Any, Any]:
    """Load a YAML file whose root must be a mapping.

    Args:
        path: YAML file path.

    Returns:
        Parsed YAML mapping.

    Raises:
        ValueError: If the YAML root is not a mapping.
    """
    p = Path(path)

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, Mapping):
        raise ValueError("Config root must be a mapping.")

    return data


def _require_mapping(obj: Any, *, what: str) -> Mapping[Any, Any]:
    """Require an object to be a mapping.

    Args:
        obj: Object to validate.
        what: Human-readable field name for error messages.

    Returns:
        ``obj`` typed as a mapping.

    Raises:
        ValueError: If ``obj`` is not a mapping.
    """
    if not isinstance(obj, Mapping):
        raise ValueError(f"{what} must be a mapping.")

    return obj


def _resolve_config_entry(
    *,
    config_file: str | Path,
    key: str | None = None,
) -> tuple[Mapping[Any, Any], str]:
    """Load a survey config using the flat tomography-list schema.

    Args:
        config_file: Path to a YAML file or filename of a shipped config.
        key: Unsupported legacy selector. Must be None.

    Returns:
        Pair of parsed config mapping and resolved config label.

    Raises:
        ValueError: If ``key`` is provided or the config has no tomography list.
    """
    if key is not None:
        raise ValueError(
            "This config schema does not support top-level keys. "
            "Pass a YAML file whose root is the config mapping and omit key=..."
        )

    path = Path(config_file)
    if not path.exists():
        path = config_path(str(config_file))

    cfg = _load_yaml_mapping(path)

    if "tomography" not in cfg:
        raise ValueError("Config must contain a 'tomography' list.")

    resolved = str(cfg.get("name") or path.stem)
    return cfg, resolved


def _normalize_label(x: Any | None) -> str | None:
    """Normalize optional selector labels.

    Args:
        x: Label-like value from the config.

    Returns:
        Stripped string, or None if the value is missing or empty.
    """
    if x is None:
        return None

    s = str(x).strip()
    return s if s else None


def _extract_z_grid(cfg: Mapping[Any, Any], z: Any | None) -> np.ndarray:
    """Return the redshift grid used to evaluate all distributions.

    Priority order:
        1. Explicit ``z`` override.
        2. ``cfg["z_grid"]``.
        3. Default grid from 0 to 3 with 301 points.

    Args:
        cfg: Parsed survey config.
        z: Optional explicit grid override.

    Returns:
        One-dimensional redshift grid.

    Raises:
        ValueError: If the config z-grid block is malformed.
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
        raise ValueError("z_grid must contain keys: start, stop, n.") from e

    return linear_grid(z_min, z_max, n)


def _extract_survey_meta(cfg: Mapping[Any, Any]) -> dict[str, Any] | None:
    """Return optional top-level survey metadata.

    Args:
        cfg: Parsed survey config.

    Returns:
        Plain dictionary containing ``survey_meta``, or None if absent.

    Raises:
        ValueError: If ``survey_meta`` is present but is not a mapping.
    """
    meta = cfg.get("survey_meta")

    if meta is None:
        return None

    return dict(_require_mapping(meta, what="Config.survey_meta"))


def _iter_tomography_entries(cfg: Mapping[Any, Any]) -> list[Mapping[Any, Any]]:
    """Return validated tomography entries.

    Args:
        cfg: Parsed survey config.

    Returns:
        List of tomography entry mappings in file order.

    Raises:
        ValueError: If ``tomography`` is missing or is not a list of mappings.
    """
    tomo = cfg.get("tomography")

    if tomo is None:
        raise ValueError("Config must contain a 'tomography' list.")

    if not isinstance(tomo, Sequence) or isinstance(tomo, str | bytes):
        raise ValueError("tomography must be a list of mappings.")

    return [_require_mapping(item, what=f"tomography[{i}]") for i, item in enumerate(tomo)]


def _select_entries(
    entries: list[Mapping[Any, Any]],
    *,
    role: Any | None = None,
    year: Any | None = None,
    scenario: Any | None = None,
    sample: Any | None = None,
) -> list[Mapping[Any, Any]]:
    """Select tomography entries by optional labels.

    Args:
        entries: Tomography entries.
        role: Optional role selector.
        year: Optional year selector.
        scenario: Optional scenario selector.
        sample: Optional sample selector.

    Returns:
        Entries matching all provided selectors.
    """
    role_s = _normalize_label(role)
    year_s = _normalize_label(year)
    scenario_s = _normalize_label(scenario)
    sample_s = _normalize_label(sample)

    if role_s is None and year_s is None and scenario_s is None and sample_s is None:
        return entries

    def match(entry: Mapping[Any, Any]) -> bool:
        return (
            (role_s is None or _normalize_label(entry.get("role")) == role_s)
            and (year_s is None or _normalize_label(entry.get("year")) == year_s)
            and (scenario_s is None or _normalize_label(entry.get("scenario")) == scenario_s)
            and (sample_s is None or _normalize_label(entry.get("sample")) == sample_s)
        )

    return [entry for entry in entries if match(entry)]


def _require_single(
    entries: list[Mapping[Any, Any]],
    *,
    what: str,
) -> Mapping[Any, Any]:
    """Require a selection to contain exactly one entry.

    Args:
        entries: Selected entries.
        what: Human-readable label used in the error message.

    Returns:
        The single selected entry.

    Raises:
        ValueError: If zero or multiple entries match.
    """
    if len(entries) != 1:
        raise ValueError(f"{what} is ambiguous; matched {len(entries)} entries.")

    return entries[0]


def _parse_bins(bins_block: Any) -> dict[str, Any]:
    """Parse a bin specification.

    A bin block must define either explicit edges or a named binning scheme with
    a number of bins. Mixed definitions are rejected.

    Args:
        bins_block: Mapping describing bin edges or a binning scheme.

    Returns:
        Normalized bin specification.

    Raises:
        ValueError: If the bin block is missing, malformed, or ambiguous.
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

    out: dict[str, Any] = {
        "scheme": str(scheme),
        "n_bins": int(n_bins),
    }

    if z_range is not None:
        zlo, zhi = map(float, z_range)
        out["range"] = (zlo, zhi)

    return out


def _parse_sample_properties(entry: Mapping[Any, Any]) -> dict[str, Any]:
    """Parse sample-level observational metadata.

    This preserves metadata such as number density, shape noise, shot noise,
    volume, and footprint. These quantities describe the observed sample and are
    intentionally separate from the redshift-distribution model.

    Args:
        entry: Tomography entry mapping.

    Returns:
        Plain dictionary of sample-level metadata.

    Raises:
        ValueError: If ``sample_properties`` or a known nested block is not a
        mapping.
    """
    props = entry.get("sample_properties") or {}

    if not isinstance(props, Mapping):
        raise ValueError("sample_properties must be a mapping if provided.")

    out = dict(props)

    for key in (
        "number_density",
        "shape_noise",
        "shot_noise",
        "volume",
        "footprint",
    ):
        value = out.get(key)

        if value is None:
            continue

        if not isinstance(value, Mapping):
            raise ValueError(f"sample_properties.{key} must be a mapping if provided.")

        out[key] = dict(value)

    return out


def _parse_entry(entry: Mapping[Any, Any]) -> dict[str, Any]:
    """Parse one tomography entry.

    This validates the tomography kind, redshift-distribution block, bin
    definition, optional uncertainty settings, and sample-level observational
    metadata.

    Args:
        entry: Raw tomography entry from the config.

    Returns:
        Normalized tomography specification.

    Raises:
        ValueError: If the tomography entry is malformed.
    """
    kind = str(entry.get("kind", "photoz")).strip().lower()

    if kind not in {"photoz", "specz"}:
        raise ValueError("tomography entry kind must be 'photoz' or 'specz'.")

    nz_block = entry.get("nz")

    if nz_block is None:
        raise ValueError("tomography entry must contain an 'nz' mapping.")

    nz_block = _require_mapping(nz_block, what="tomography entry.nz")

    uncertainties = entry.get("uncertainties") or {}
    if not isinstance(uncertainties, Mapping):
        raise ValueError("uncertainties must be a mapping if provided.")

    normalize_bins = entry.get("normalize_bins", True)

    if not isinstance(normalize_bins, bool):
        raise ValueError("normalize_bins must be a boolean if provided.")

    return {
        "role": _normalize_label(entry.get("role")),
        "year": _normalize_label(entry.get("year")),
        "scenario": _normalize_label(entry.get("scenario")),
        "sample": _normalize_label(entry.get("sample")),
        "name": _normalize_label(entry.get("name")),
        "kind": kind,
        "nz": dict(nz_block),
        "bins": _parse_bins(entry.get("bins")),
        "uncertainties": dict(uncertainties),
        "sample_properties": _parse_sample_properties(entry),
        "normalize_bins": normalize_bins,
    }


def _build_parent_nz(entry: Mapping[Any, Any], z: np.ndarray) -> np.ndarray:
    """Evaluate a tomography entry's parent redshift distribution.

    Args:
        entry: Tomography entry containing an ``nz`` block.
        z: Redshift grid on which to evaluate the parent distribution.

    Returns:
        Parent redshift distribution evaluated on ``z``.

    Raises:
        ValueError: If the ``nz`` block is missing required fields.
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
    """Return tabulated n(z) inputs from inline arrays or a source file.

    Args:
        nz_cfg: Redshift-distribution config block.

    Returns:
        Keyword arguments containing ``z_input`` and ``nz_input`` when tabulated
        data are provided. Returns an empty dictionary if no tabulated input is
        specified.

    Raises:
        ValueError: If tabulated input is incomplete or malformed.
    """
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
        path = resources.files(_DATA_PKG) / str(path)

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


def _survey_meta(
    *,
    cfg: Mapping[Any, Any],
    resolved_key: str,
    role: str | None = None,
    year: str | None = None,
    scenario: str | None = None,
    sample: str | None = None,
) -> dict[str, Any]:
    """Build metadata for a selected survey config entry.

    Args:
        cfg: Parsed survey config.
        resolved_key: Resolved config label.
        role: Optional role selector.
        year: Optional year selector.
        scenario: Optional scenario selector.
        sample: Optional sample selector.

    Returns:
        Metadata dictionary containing the survey label, selectors, and optional
        top-level survey metadata.
    """
    return {
        "survey": str(cfg.get("name", resolved_key)),
        "key": resolved_key,
        "role": role,
        "year": year,
        "scenario": scenario,
        "sample": sample,
        "survey_meta": _extract_survey_meta(cfg),
    }


def _builder_kwargs_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a normalized tomography spec into builder keyword arguments.

    Args:
        spec: Normalized tomography spec containing ``bins`` and optional
            ``uncertainties``.

    Returns:
        Keyword arguments expected by the tomography builders.

    Raises:
        ValueError: If the normalized spec does not contain a valid ``bins``
        mapping.
    """
    bins = spec.get("bins")

    if not isinstance(bins, Mapping):
        raise ValueError("tomo_spec must contain a 'bins' mapping.")

    if "edges" in bins:
        kwargs: dict[str, Any] = {
            "bin_edges": bins["edges"],
            "binning_scheme": None,
            "n_bins": None,
        }
    else:
        kwargs = {
            "bin_edges": None,
            "binning_scheme": bins["scheme"],
            "n_bins": bins["n_bins"],
        }

    params: dict[str, Any] = dict(spec.get("uncertainties") or {})

    if "range" in bins:
        params.setdefault("bin_range", bins["range"])

    kwargs.update(params)
    return kwargs
