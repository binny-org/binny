"""Dictionary specifications for tomographic bin construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

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

MODE_KEYS = {"edges": "explicit", "scheme": "generated", "segments": "mixed"}


def validate_bins(bins: Mapping[str, Any]) -> tuple[BinMode, dict[str, Any]]:
    """Validate a bin specification and return its mode and a shallow copy.

    Exactly one of ``edges``, ``scheme`` or ``segments`` selects the mode.
    Explicit edges cannot be combined with a range or a generating scheme.
    Generated schemes accept ``range`` and nested scheme-specific ``params``.
    Mixed binning defines counts and ranges inside each segment.
    """
    if not isinstance(bins, Mapping):
        raise TypeError("bins must be a mapping.")

    selected = {MODE_KEYS[key] for key in bins.keys() & MODE_KEYS.keys()}
    if len(selected) != 1:
        raise ValueError("Specify exactly one of 'edges', 'scheme', or 'segments'.")

    mode = selected.pop()
    rules = BIN_MODES[mode]
    missing = rules["required"] - bins.keys()
    extra = bins.keys() - rules["allowed"]
    if missing or extra:
        raise ValueError(
            f"Invalid {mode} bins: missing={sorted(missing)}, unsupported={sorted(extra)}."
        )

    spec = dict(bins)
    validators = {
        "explicit": _validate_explicit,
        "generated": _validate_generated,
        "mixed": _validate_mixed,
    }
    validators[mode](spec)
    return mode, spec  # type: ignore[return-value]


def _validate_explicit(spec: dict[str, Any]) -> None:
    # Numerical edge validation is performed by validate_bin_edges at resolution.
    if spec["edges"] is None:
        raise ValueError("bins.edges cannot be None.")


def _validate_generated(spec: dict[str, Any]) -> None:
    if not isinstance(spec["scheme"], str) or not spec["scheme"].strip():
        raise ValueError("bins.scheme must be a nonempty string.")
    if spec["n_bins"] is None:
        raise ValueError("You must provide n_bins for the selected binning scheme.")
    validate_n_bins(spec["n_bins"])
    params = spec.get("params", {})
    if not isinstance(params, Mapping):
        raise ValueError("bins.params must be a mapping.")
    reserved = set(params) & {"scheme", "n_bins", "range", "bin_range"}
    if reserved:
        raise ValueError(f"Reserved keys in bins.params: {sorted(reserved)}.")
    spec["params"] = dict(params)


def _validate_mixed(spec: dict[str, Any]) -> None:
    segments = spec["segments"]
    if not isinstance(segments, Sequence) or isinstance(segments, str | bytes):
        raise ValueError("bins.segments must be a sequence of segment mappings.")
    if not segments or not all(isinstance(item, Mapping) for item in segments):
        raise ValueError("bins.segments must contain segment mappings.")


def restrict_bin_range(
    z: FloatArray,
    nz: FloatArray,
    bin_range: tuple[float, float] | None,
) -> tuple[FloatArray, FloatArray]:
    """Restrict sampled data to a range, inserting interpolated endpoints.

    The full input grid and distribution are never modified.
    """
    z_arr, nz_arr = validate_axis_and_weights(z, nz)
    if bin_range is None:
        return z_arr, nz_arr

    try:
        bounds = np.asarray(bin_range, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("bins.range must contain two finite numbers.") from error
    if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
        raise ValueError("bins.range must contain two finite, increasing numbers.")
    lo, hi = bounds
    if lo < z_arr[0] or hi > z_arr[-1]:
        raise ValueError("bins.range must lie within the binning grid.")

    interior = z_arr[(z_arr > lo) & (z_arr < hi)]
    restricted_z = np.concatenate(([lo], interior, [hi]))
    return restricted_z, np.interp(restricted_z, z_arr, nz_arr)
