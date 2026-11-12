"""Validate dictionary specifications for one-dimensional bin construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

import numpy as np

from binny.utils.types import FloatArray
from binny.utils.validators import validate_axis_and_weights, validate_n_bins

BinMode = Literal["explicit", "generated", "mixed"]

__all__ = [
    "BinMode",
    "BIN_MODES",
    "validate_bins",
    "restrict_bin_range",
]

BIN_MODES: dict[str, dict[str, frozenset[str]]] = {
    "explicit": {
        "required": frozenset({"edges"}),
        "allowed": frozenset({"edges"}),
    },
    "generated": {
        "required": frozenset({"scheme", "n_bins"}),
        "allowed": frozenset({"scheme", "n_bins", "range", "params"}),
    },
    "mixed": {
        "required": frozenset({"segments"}),
        "allowed": frozenset({"segments"}),
    },
}

MODE_KEYS = {
    "edges": "explicit",
    "scheme": "generated",
    "segments": "mixed",
}

_SEGMENT_KEYS = frozenset(
    {
        "scheme",
        "n_bins",
        "range",
        "x_min",
        "x_max",
        "z_min",
        "z_max",
        "params",
    }
)

_RESERVED_PARAMS = frozenset(
    {
        "scheme",
        "n_bins",
        "range",
        "bin_range",
    }
)


def _validate_range(
    value: Any,
    *,
    name: str,
) -> tuple[float, float]:
    """Validate an interval and return its numerical endpoints.

    Args:
        value: Candidate interval containing two numerical endpoints.
        name: Parameter name used in validation messages.

    Returns:
        Pair of finite, strictly increasing interval endpoints.

    Raises:
        ValueError: If the interval does not contain two finite,
            strictly increasing numerical values.
    """
    try:
        bounds = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain two finite numbers.") from exc

    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError(f"{name} must contain two finite, increasing numbers.")

    return float(bounds[0]), float(bounds[1])


def _validate_params(
    value: Any,
    *,
    name: str,
) -> dict[str, Any]:
    """Validate a mapping of scheme-specific parameters.

    Args:
        value: Candidate parameter mapping.
        name: Parameter name used in validation messages.

    Returns:
        Shallow copy of the validated parameter mapping.

    Raises:
        ValueError: If the value is not a mapping or contains
            reserved binning specification keys.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")

    params = dict(value)
    reserved = set(params) & _RESERVED_PARAMS

    if reserved:
        raise ValueError(f"Reserved keys in {name}: {sorted(reserved)}.")

    return params


def validate_bins(
    bins: Mapping[str, Any],
) -> tuple[BinMode, dict[str, Any]]:
    """Validate a binning specification and identify its construction mode.

    Exactly one of ``edges``, ``scheme``, or ``segments`` selects
    the binning mode.

    Explicit binning accepts supplied boundaries. Generated binning
    accepts a registered scheme, a number of bins, and an optional
    coordinate interval. Mixed binning combines independently
    specified schemes over adjacent intervals.

    Scheme-specific parameters are checked by the registered
    scheme resolver rather than by this function.

    Args:
        bins: Mapping describing explicit, generated, or mixed binning.

    Returns:
        Pair containing the selected binning mode and a validated
        shallow copy of the specification.

    Raises:
        TypeError: If ``bins`` is not a mapping.
        ValueError: If the specification selects multiple modes,
            contains unsupported keys, or has invalid parameters.
    """
    if not isinstance(bins, Mapping):
        raise TypeError("bins must be a mapping.")

    selected = {MODE_KEYS[key] for key in bins.keys() & MODE_KEYS.keys()}

    if len(selected) != 1:
        raise ValueError("Specify exactly one of 'edges', 'scheme', or 'segments'.")

    mode = cast(BinMode, selected.pop())
    rules = BIN_MODES[mode]

    missing = rules["required"] - bins.keys()
    extra = bins.keys() - rules["allowed"]

    if missing or extra:
        raise ValueError(
            f"Invalid {mode} bins: missing={sorted(missing)}, unsupported={sorted(extra)}."
        )

    spec = dict(bins)

    if mode == "explicit":
        _validate_explicit(spec)
    elif mode == "generated":
        _validate_generated(spec)
    else:
        _validate_mixed(spec)

    return mode, spec


def _validate_explicit(spec: dict[str, Any]) -> None:
    """Validate the structure of an explicit boundary specification.

    Args:
        spec: Explicit binning specification.

    Raises:
        ValueError: If the boundary array is missing.

    Notes:
        Numerical boundary validation is performed by
        ``validate_bin_edges`` during resolution.
    """
    if spec["edges"] is None:
        raise ValueError("bins.edges cannot be None.")


def _validate_generated(spec: dict[str, Any]) -> None:
    """Validate the structure of a generated binning specification.

    Args:
        spec: Generated binning specification.

    Raises:
        ValueError: If the scheme name, number of bins, interval,
            or parameter mapping is invalid.
    """
    scheme = spec["scheme"]

    if not isinstance(scheme, str) or not scheme.strip():
        raise ValueError("bins.scheme must be a nonempty string.")

    if spec["n_bins"] is None:
        raise ValueError("You must provide n_bins for the selected binning scheme.")

    validate_n_bins(spec["n_bins"])

    if spec.get("range") is not None:
        spec["range"] = _validate_range(
            spec["range"],
            name="bins.range",
        )

    spec["params"] = _validate_params(
        spec.get("params", {}),
        name="bins.params",
    )


def _validate_mixed(spec: dict[str, Any]) -> None:
    """Validate the structure of a mixed binning specification.

    Each segment must define a registered scheme name, a number of
    bins, and an interval. The interval may be supplied through
    ``range``, generic ``x_min``/``x_max`` boundaries, or legacy
    ``z_min``/``z_max`` boundaries.

    Adjacent intervals are checked for continuity when their
    constructed boundaries are combined.

    Args:
        spec: Mixed binning specification containing ``segments``.

    Raises:
        ValueError: If the segment sequence is empty, a segment
            is malformed, or its interval is missing or invalid.
    """
    segments = spec["segments"]

    if not isinstance(segments, Sequence) or isinstance(segments, str | bytes):
        raise ValueError("bins.segments must be a sequence of segment mappings.")

    if not segments or not all(isinstance(segment, Mapping) for segment in segments):
        raise ValueError("bins.segments must contain segment mappings.")

    validated: list[dict[str, Any]] = []

    for index, segment in enumerate(segments):
        extra = set(segment) - _SEGMENT_KEYS

        if extra:
            raise ValueError(f"Segment {index} has unsupported keys: {sorted(extra)}.")

        if "scheme" not in segment or "n_bins" not in segment:
            raise ValueError(f"Segment {index} requires 'scheme' and 'n_bins'.")

        scheme = segment["scheme"]

        if not isinstance(scheme, str) or not scheme.strip():
            raise ValueError(f"Segment {index} requires a nonempty scheme name.")

        validate_n_bins(segment["n_bins"])

        params = _validate_params(
            segment.get("params", {}),
            name=f"bins.segments[{index}].params",
        )

        has_range = segment.get("range") is not None
        has_x = "x_min" in segment or "x_max" in segment
        has_z = "z_min" in segment or "z_max" in segment

        if sum((has_range, has_x, has_z)) != 1:
            raise ValueError(
                f"Segment {index} must specify exactly one interval "
                "using 'range', 'x_min'/'x_max', or "
                "'z_min'/'z_max'."
            )

        if has_range:
            interval = _validate_range(
                segment["range"],
                name=f"bins.segments[{index}].range",
            )
        elif has_x:
            if "x_min" not in segment or "x_max" not in segment:
                raise ValueError(f"Segment {index} requires both x_min and x_max.")

            interval = _validate_range(
                (segment["x_min"], segment["x_max"]),
                name=f"bins.segments[{index}] interval",
            )
        else:
            if "z_min" not in segment or "z_max" not in segment:
                raise ValueError(f"Segment {index} requires both z_min and z_max.")

            interval = _validate_range(
                (segment["z_min"], segment["z_max"]),
                name=f"bins.segments[{index}] interval",
            )

        validated.append(
            {
                "scheme": scheme,
                "n_bins": segment["n_bins"],
                "range": interval,
                "params": params,
            }
        )

    spec["segments"] = validated


def restrict_bin_range(
    z: FloatArray,
    nz: FloatArray,
    bin_range: tuple[float, float] | None,
) -> tuple[FloatArray, FloatArray]:
    """Restrict sampled data to an interval, interpolating its endpoints.

    The sampling coordinates and their associated density are
    restricted to the requested interval. If an endpoint does not
    coincide with an existing sample, its density is obtained
    by linear interpolation.

    The original arrays are not modified.

    Args:
        z: Strictly increasing one-dimensional sampling coordinates.
        nz: Density or weights evaluated on ``z``.
        bin_range: Optional interval containing the lower and upper
            boundaries. If omitted, the complete arrays are returned.

    Returns:
        Pair containing the restricted sampling coordinates and
        corresponding interpolated density values.

    Raises:
        ValueError: If the sampling arrays are invalid, the interval
            is malformed, or its boundaries lie outside the grid.
    """
    z_arr, nz_arr = validate_axis_and_weights(z, nz)

    if bin_range is None:
        return z_arr, nz_arr

    lo, hi = _validate_range(
        bin_range,
        name="bins.range",
    )

    if lo < z_arr[0] or hi > z_arr[-1]:
        raise ValueError("bins.range must lie within the binning grid.")

    interior = z_arr[(z_arr > lo) & (z_arr < hi)]
    restricted_z = np.concatenate(([lo], interior, [hi]))

    restricted_nz = np.interp(
        restricted_z,
        z_arr,
        nz_arr,
    )

    return restricted_z, restricted_nz
