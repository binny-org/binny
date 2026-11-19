"""Parsing and preparation of tomography specifications."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from binny.tomography.predefined import load_predefined_bins
from binny.utils.types import FloatArray
from binny.utils.validators import _normalize_label, _require_mapping

_DATA_PKG = "binny.surveys.data"


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


def _builder_kwargs_from_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a tomography specification into generic builder arguments.

    Separates construction options from assignment-model parameters.

    Args:
        spec: Normalized tomography specification containing ``bins``
            and optional ``uncertainties``.

    Returns:
        Keyword arguments for build_tomography.

    Raises:
        ValueError: If the specification lacks a valid ``bins`` mapping.
    """
    bins = spec.get("bins")

    if not isinstance(bins, Mapping):
        raise ValueError("tomo_spec must contain a 'bins' mapping.")

    params = dict(spec.get("uncertainties") or {})

    kwargs: dict[str, Any] = {
        "bins": dict(bins),
        "normalize_bins": spec.get("normalize_bins", True),
    }

    construction_options = (
        "normalize_input",
        "normalize_bins",
        "norm_method",
        "save_metadata_path",
    )

    for name in construction_options:
        if name in params:
            kwargs[name] = params.pop(name)

    if spec.get("kind") == "photoz":
        for name in ("z_ph", "nz_ph"):
            if name in params:
                kwargs[name] = params.pop(name)

    kwargs["params"] = params

    return kwargs
