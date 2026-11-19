"""Survey configuration parsing utilities.

This module reads survey YAML files using a flat tomography-list schema.
Each tomography entry defines one sample, its parent redshift distribution,
its binning prescription, and optional observational sample metadata.

The parent redshift distribution is evaluated on the full redshift grid.
Tomographic bin boundaries can cover the full grid or a restricted interval
without changing the underlying parent distribution.

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
        kind: photoz|specz|predefined

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
          # Exactly one binning mode must be specified.
          # See the alternatives below.

        uncertainties:
          <optional builder keyword arguments>

        normalize_bins: <optional bool>

Binning modes
-------------

The ``bins`` mapping supports four mutually exclusive configurations.

1. Generated bin edges
~~~~~~~~~~~~~~~~~~~~~~

Generate bin boundaries using a registered binning scheme::

    bins:
      scheme: equidistant
      n_bins: 5

An optional ``range`` restricts the binning interval while retaining
the original redshift grid and parent distribution::

    bins:
      scheme: equal_number
      n_bins: 5
      range: [0.3, 2.4]

Scheme-specific parameters can be supplied through ``params``::

    bins:
      scheme: generalized
      n_bins: 5
      range: [0.3, 2.4]
      params:
        alpha: 1.0
        beta: 0.0

The ``params`` mapping belongs to the binning scheme and is independent
of ``nz.params``, which configures the parent redshift distribution.

2. Explicit bin edges
~~~~~~~~~~~~~~~~~~~~~

Supply the bin boundaries directly::

    bins:
      edges: [0.25, 0.35, 0.65, 1.10, 1.63, 2.0]

The number of bins is inferred from the number of edges.
Do not combine ``edges`` with ``scheme``, ``n_bins``, ``range``,
or ``segments``.

3. Mixed binning
~~~~~~~~~~~~~~~~

Combine multiple binning schemes over adjacent intervals::

    bins:
      segments:
        - scheme: equidistant
          n_bins: 2
          x_min: 0.4
          x_max: 0.9
        - scheme: equal_number
          n_bins: 3
          x_min: 0.9
          x_max: 2.4

Each segment specifies its own scheme, number of bins, and interval.
Segments must be ordered and adjacent, with no gaps or overlaps.

Scheme-specific ``params`` may also be provided for individual segments.

The total number of bins is the sum of the bins in all segments.
Do not combine ``segments`` with top-level ``scheme``, ``n_bins``,
``range``, or ``edges``.

4. Predefined tomography
~~~~~~~~~~~~~~~~~~~~~~~~

Load externally supplied tomographic redshift distributions::

    kind: predefined

    bins:
      scheme: predefined
      source:
        path: mock_predefined_bins.txt
        z_col: z
        bin_cols:
          0: bin_0
          1: bin_1
      n_bins: 2
      normalize_bins: false
      norm_method: trapezoid

For predefined tomography, the supplied bin distributions are loaded
directly rather than generated from a parent distribution.

The ``nz`` block is not required for predefined tomography.

The ``n_bins`` field is optional and, when supplied, is checked against
the number of loaded distributions.

The ``normalize_bins`` and ``norm_method`` fields control normalization
of the predefined distributions.

General rules
-------------

- The ``z_grid`` defines the full redshift grid used for generated
  tomography and is independent of the binning interval.
- Generated binning uses the full grid unless ``bins.range`` is given.
- Explicit binning uses the supplied ``bins.edges``.
- Mixed binning uses the intervals specified by ``bins.segments``.
- The binning modes are mutually exclusive; conflicting settings
  must raise an error rather than silently override one another.
- ``nz.params`` configures the parent redshift-distribution model.
- ``bins.params`` configures the bin-edge generation scheme.
- ``uncertainties`` configures the photo-z or spec-z assignment model.
- ``sample_properties`` stores observational metadata independently
  of the redshift-distribution and binning models.
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
from binny.utils.validators import _normalize_label, _require_mapping

__all__ = [
    "config_path",
    "list_configs",
    "load_config",
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


def _load_tabulated_source_table(source: Mapping[Any, Any]) -> np.ndarray:
    """Load a packaged or absolute tabulated survey data source."""
    source = _require_mapping(source, what="tomography entry.nz.source")

    try:
        path = Path(source["path"])
    except KeyError as e:
        raise ValueError("tomography entry.nz.source must contain a 'path' field.") from e

    if not path.is_absolute():
        path = resources.files(_DATA_PKG) / str(path)

    with resources.as_file(path) as real_path:
        return np.loadtxt(
            real_path,
            skiprows=int(source.get("skiprows", 0)),
        )


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
    table = _load_tabulated_source_table(source)

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


def load_config(path: str | Path) -> Mapping[Any, Any]:
    """Load a survey configuration from a YAML file.

    Args:
        path: Path to a YAML file, or the filename of a shipped Binny survey
            configuration.

    Returns:
        Parsed survey configuration mapping.

    Raises:
        ValueError: If the YAML root is not a mapping or the config has no
            tomography list.
        FileNotFoundError: If ``path`` is neither an existing file nor a
            shipped config filename.
    """
    p = Path(path)

    if not p.exists():
        p = config_path(str(path))

    cfg = _load_yaml_mapping(p)

    if "tomography" not in cfg:
        raise ValueError("Config must contain a 'tomography' list.")

    return cfg
