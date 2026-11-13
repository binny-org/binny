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
from binny.tomography.predefined import load_predefined_bins
from binny.utils.types import FloatArray

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


def _parse_n_bins(value: Any, *, predefined: bool = False) -> int:
    """Validate and return the number of tomographic bins."""
    label = "Predefined bins.n_bins" if predefined else "bins.n_bins"

    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")

    try:
        n_bins = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be an integer.") from error

    if isinstance(value, float | np.floating) and (not np.isfinite(value) or value != n_bins):
        raise ValueError(f"{label} must be an integer.")

    if n_bins < 1:
        raise ValueError(f"{label} must be positive.")

    return n_bins


def _parse_explicit_bins(bins: Mapping[Any, Any]) -> dict[str, Any]:
    """Parse explicitly supplied tomographic bin edges."""
    if any(key != "edges" for key in bins):
        raise ValueError("bins: if 'edges' is provided, do not provide other bin settings.")

    edges = np.asarray(bins["edges"], dtype=np.float64)

    if edges.ndim != 1 or edges.size < 2:
        raise ValueError("bins.edges must be a 1D sequence with at least 2 values.")

    return {"edges": edges}


def _parse_predefined_bins(bins: Mapping[Any, Any]) -> dict[str, Any]:
    """Parse externally supplied predefined tomographic bins."""
    source = _require_mapping(bins.get("source"), what="bins.source")

    if bins.get("range") is not None:
        raise ValueError("Predefined bins must not provide bins.range.")

    normalize_bins = bins.get("normalize_bins", False)

    if not isinstance(normalize_bins, bool):
        raise ValueError("Predefined bins.normalize_bins must be a boolean.")

    norm_method = str(bins.get("norm_method", "trapezoid")).strip().lower()

    if norm_method not in {"trapezoid", "simpson"}:
        raise ValueError("Predefined bins.norm_method must be 'trapezoid' or 'simpson'.")

    out: dict[str, Any] = {
        "scheme": "predefined",
        "source": dict(source),
        "normalize_bins": normalize_bins,
        "norm_method": norm_method,
    }

    if bins.get("n_bins") is not None:
        out["n_bins"] = _parse_n_bins(bins["n_bins"], predefined=True)

    return out


def _parse_generated_bins(bins: Mapping[Any, Any]) -> dict[str, Any]:
    """Parse a registered tomographic bin-edge generation scheme."""
    allowed = {"scheme", "n_bins", "range", "params"}

    unknown = set(bins) - allowed

    if unknown:
        raise ValueError(f"Unknown generated bin settings: {sorted(unknown)}")

    scheme = bins.get("scheme")

    if scheme is None:
        raise ValueError("bins must provide either 'edges' or 'scheme'.")

    scheme_name = str(scheme).strip()

    if not scheme_name:
        raise ValueError("bins.scheme must not be empty.")

    if bins.get("n_bins") is None:
        raise ValueError("bins.n_bins is required when using a generated bins.scheme.")

    out: dict[str, Any] = {
        "scheme": scheme_name,
        "n_bins": _parse_n_bins(bins["n_bins"]),
    }

    if bins.get("range") is not None:
        bin_range = np.asarray(bins["range"], dtype=np.float64)

        if (
            bin_range.shape != (2,)
            or not np.all(np.isfinite(bin_range))
            or bin_range[0] >= bin_range[1]
        ):
            raise ValueError("bins.range must contain two finite, strictly increasing values.")

        out["range"] = tuple(bin_range)

    scheme_params = bins.get("params", {})

    if not isinstance(scheme_params, Mapping):
        raise ValueError("bins.params must be a mapping.")

    if scheme_params:
        out["params"] = dict(scheme_params)

    return out


def _parse_mixed_bins(bins: Mapping[Any, Any]) -> dict[str, Any]:
    """Parse a mixed tomographic binning specification."""
    if set(bins) != {"segments"}:
        raise ValueError("Mixed bins must contain only 'segments'.")

    segments = bins["segments"]

    if not isinstance(segments, Sequence) or isinstance(segments, str | bytes) or not segments:
        raise ValueError("bins.segments must be a non-empty sequence.")

    parsed = []

    for i, segment in enumerate(segments):
        segment = _require_mapping(
            segment,
            what=f"bins.segments[{i}]",
        )

        allowed = {"scheme", "n_bins", "x_min", "x_max", "params"}

        if set(segment) - allowed:
            raise ValueError(
                f"Unknown fields in bins.segments[{i}]: {sorted(set(segment) - allowed)}"
            )

        required = {"scheme", "n_bins", "x_min", "x_max"}

        if not required <= set(segment):
            raise ValueError(
                f"bins.segments[{i}] requires 'scheme', 'n_bins', 'x_min', and 'x_max'."
            )

        lo = float(segment["x_min"])
        hi = float(segment["x_max"])

        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError(f"bins.segments[{i}] requires finite x_min < x_max.")

        scheme = str(segment["scheme"]).strip()

        if not scheme:
            raise ValueError(f"bins.segments[{i}].scheme must not be empty.")

        params = segment.get("params", {})

        if not isinstance(params, Mapping):
            raise ValueError(f"bins.segments[{i}].params must be a mapping.")

        parsed_segment = {
            "scheme": scheme,
            "n_bins": _parse_n_bins(segment["n_bins"]),
            "x_min": lo,
            "x_max": hi,
        }

        if params:
            parsed_segment["params"] = dict(params)

        parsed.append(parsed_segment)

    for previous, current in zip(parsed[:-1], parsed[1:], strict=True):
        if not np.isclose(
            previous["x_max"],
            current["x_min"],
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("Mixed binning segments must be adjacent.")

    return {"segments": parsed}


def _parse_bins(bins_block: Any) -> dict[str, Any]:
    """Parse mutually exclusive tomographic binning specifications."""
    bins = _require_mapping(bins_block, what="bins")

    modes = [key for key in ("edges", "scheme", "segments") if key in bins]

    if len(modes) != 1:
        raise ValueError("bins must specify exactly one of 'edges', 'scheme', or 'segments'.")

    if "edges" in bins:
        return _parse_explicit_bins(bins)

    if "segments" in bins:
        return _parse_mixed_bins(bins)

    if str(bins["scheme"]).strip().lower() == "predefined":
        return _parse_predefined_bins(bins)

    return _parse_generated_bins(bins)


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

    This validates the tomography kind, bin definition, optional parent
    redshift-distribution block, uncertainty settings, and sample metadata.

    Predefined tomography entries do not require an ``nz`` block because
    their tomographic distributions are loaded directly from a file.

    Args:
        entry: Raw tomography entry from the config.

    Returns:
        Normalized tomography specification.

    Raises:
        ValueError: If the tomography entry is malformed.
    """
    kind = (
        str(
            entry.get(
                "kind",
                "photoz",
            )
        )
        .strip()
        .lower()
    )

    if kind not in {
        "photoz",
        "specz",
        "predefined",
    }:
        raise ValueError("tomography entry kind must be 'photoz', 'specz', or 'predefined'.")

    bins = _parse_bins(entry.get("bins"))

    bin_scheme = (
        str(
            bins.get(
                "scheme",
                "",
            )
        )
        .strip()
        .lower()
    )

    if kind == "predefined" and bin_scheme != "predefined":
        raise ValueError(
            "A tomography entry with kind='predefined' must use bins.scheme='predefined'."
        )

    if bin_scheme == "predefined":
        kind = "predefined"

    if kind == "predefined":
        nz_block: dict[str, Any] | None = None
    else:
        raw_nz_block = entry.get("nz")

        if raw_nz_block is None:
            raise ValueError("Photo-z and spec-z tomography entries must contain an 'nz' mapping.")

        nz_block = dict(
            _require_mapping(
                raw_nz_block,
                what="tomography entry.nz",
            )
        )

    uncertainties = entry.get("uncertainties") or {}

    if not isinstance(
        uncertainties,
        Mapping,
    ):
        raise ValueError("uncertainties must be a mapping if provided.")

    if kind == "predefined":
        normalize_bins = bool(
            bins.get(
                "normalize_bins",
                False,
            )
        )
    else:
        normalize_bins = entry.get(
            "normalize_bins",
            True,
        )

        if not isinstance(
            normalize_bins,
            bool,
        ):
            raise ValueError("normalize_bins must be a boolean if provided.")

    return {
        "role": _normalize_label(entry.get("role")),
        "year": _normalize_label(entry.get("year")),
        "scenario": _normalize_label(entry.get("scenario")),
        "sample": _normalize_label(entry.get("sample")),
        "name": _normalize_label(entry.get("name")),
        "kind": kind,
        "nz": nz_block,
        "bins": bins,
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


def _load_predefined_bin_source(
    bins_spec: Mapping[str, Any],
) -> tuple[FloatArray, dict[int, FloatArray]]:
    """Load predefined tomographic bins from a configured source file.

    Relative paths are resolved inside the packaged survey-data directory.

    Args:
        bins_spec: Parsed ``bins`` configuration mapping.

    Returns:
        Shared redshift grid and predefined tomographic bin curves.

    Raises:
        ValueError: If the source configuration is missing or malformed, or
            if ``n_bins`` disagrees with the number of loaded bins.
    """
    bins_spec = _require_mapping(
        bins_spec,
        what="predefined bins",
    )

    source = _require_mapping(
        bins_spec.get("source"),
        what="predefined bins.source",
    )

    try:
        path = Path(source["path"])
    except KeyError as error:
        raise ValueError("Predefined bins.source must contain a 'path' field.") from error

    if not path.is_absolute():
        path = resources.files(_DATA_PKG) / str(path)

    z_col = source.get("z_col", "z")
    bin_cols = source.get("bin_cols")

    if bin_cols is not None and not isinstance(bin_cols, Mapping):
        raise ValueError("Predefined bins.source.bin_cols must be a mapping.")

    with resources.as_file(path) as real_path:
        z, bins = load_predefined_bins(
            real_path,
            z_col=z_col,
            bin_cols=bin_cols,
        )

    configured_n_bins = bins_spec.get("n_bins")

    if configured_n_bins is not None:
        if isinstance(configured_n_bins, bool):
            raise ValueError("Predefined bins.n_bins must be an integer.")

        try:
            n_bins = int(configured_n_bins)
        except (TypeError, ValueError) as error:
            raise ValueError("Predefined bins.n_bins must be an integer.") from error

        if n_bins != len(bins):
            raise ValueError(
                f"Configured n_bins={n_bins} does not match the "
                f"{len(bins)} predefined bins loaded from {path}."
            )

    return z, bins


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


def _builder_kwargs_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a tomography specification into builder keyword arguments.

    Args:
        spec: Normalized tomography specification containing ``bins``
            and optional ``uncertainties``.

    Returns:
        Keyword arguments for the tomography builders.

    Raises:
        ValueError: If the specification lacks a valid ``bins`` mapping.
    """
    bins = spec.get("bins")

    if not isinstance(bins, Mapping):
        raise ValueError("tomo_spec must contain a 'bins' mapping.")

    kwargs = dict(spec.get("uncertainties") or {})
    kwargs["bins"] = dict(bins)

    return kwargs


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
