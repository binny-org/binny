"""Resolve explicit, generated, and mixed tomographic bin specifications."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from binny.axes.bin_edges import (
    equal_number_edges,
    equidistant_edges,
    generalized_tomographic_edges,
)
from binny.axes.mixed_edges import mixed_edges as mixed_edges_axes
from binny.binning.scheme_registry import get_binning_scheme, register_binning_scheme
from binny.binning.specs import restrict_bin_range, validate_bins
from binny.utils.normalization import normalize_1d
from binny.utils.types import FloatArray
from binny.utils.validators import validate_axis_and_weights, validate_n_bins

__all__ = [
    "resolve_bin_edges",
    "validate_bin_edges",
    "photoz_segments_to_axes",
    "mixed_edges_from_segments",
]


@register_binning_scheme("equidistant", aliases=("eq", "linear"))
def _equidistant_scheme(z: FloatArray, nz: FloatArray, n_bins: int) -> FloatArray:
    """Generate equally spaced edges on the supplied grid."""
    return equidistant_edges(float(z[0]), float(z[-1]), n_bins)


@register_binning_scheme("equal_number", aliases=("equipopulated", "en"))
def _equal_number_scheme(
    z: FloatArray,
    nz: FloatArray,
    n_bins: int,
    *,
    equal_number_axis: Any | None = None,
    equal_number_weights: Any | None = None,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    normalize_equal_number_weights: bool = True,
) -> FloatArray:
    """Generate equal-population edges, optionally from separate sampled weights."""
    if (equal_number_axis is None) ^ (equal_number_weights is None):
        raise ValueError("Provide both equal_number_axis and equal_number_weights, or neither.")
    if equal_number_axis is None:
        x, w = z, nz
    else:
        x, w = validate_axis_and_weights(equal_number_axis, equal_number_weights)
    if normalize_equal_number_weights:
        total = float(np.trapezoid(w, x=x))
        if total > 0.0 and not np.isclose(total, 1.0, rtol=1e-3, atol=1e-4):
            w = normalize_1d(x, w, method=norm_method)
    return equal_number_edges(x, w, n_bins)


@register_binning_scheme("generalized")
def _generalized_scheme(
    z: FloatArray,
    nz: FloatArray,
    n_bins: int,
    *,
    alpha: float = 0.0,
    beta: float = 0.0,
    comoving_distance: Any | None = None,
) -> FloatArray:
    """Generate generalized edges using n(z)^alpha and (dchi/dz)^beta."""
    return generalized_tomographic_edges(
        z,
        nz,
        n_bins,
        alpha=alpha,
        beta=beta,
        comoving_distance=comoving_distance,
    )


def _resolve_explicit(
    spec: Mapping[str, Any], z: FloatArray, nz: FloatArray, options: dict[str, Any]
) -> FloatArray:
    return validate_bin_edges(spec["edges"])


def _resolve_generated(
    spec: Mapping[str, Any], z: FloatArray, nz: FloatArray, options: dict[str, Any]
) -> FloatArray:
    scheme = get_binning_scheme(spec["scheme"])
    original_z = z
    z, nz = restrict_bin_range(z, nz, spec.get("range"))
    params = dict(spec.get("params", {}))
    available = dict(options)
    available.update(params)

    # Resample a supplied chi(z) only when it corresponds to the original grid.
    if "comoving_distance" in available and available["comoving_distance"] is not None:
        chi = np.asarray(available["comoving_distance"], dtype=float)
        if chi.shape != original_z.shape or not np.all(np.isfinite(chi)):
            raise ValueError("comoving_distance must be finite and match the original z grid.")
        if np.any(np.diff(chi) <= 0):
            raise ValueError("comoving_distance must be strictly increasing.")
        if spec.get("range") is not None:
            available["comoving_distance"] = np.interp(z, original_z, chi)

    # An independent equal-number distribution takes precedence over parent n(z).
    # Restrict it to the same requested interval, on its own grid.
    separate_axis = available.get("equal_number_axis")
    separate_weights = available.get("equal_number_weights")
    if (separate_axis is None) ^ (separate_weights is None):
        raise ValueError("Provide both equal_number_axis and equal_number_weights, or neither.")
    if separate_axis is not None and spec.get("range") is not None:
        x, w = restrict_bin_range(separate_axis, separate_weights, spec["range"])
        available["equal_number_axis"] = x
        available["equal_number_weights"] = w

    signature = inspect.signature(scheme)
    accepted = set(signature.parameters)
    reserved = {"z", "nz", "n_bins", "bin_range"}
    unknown = set(params) - accepted
    if unknown:
        raise ValueError(f"Unsupported parameters for {spec['scheme']!r}: {sorted(unknown)}.")
    if set(params) & reserved:
        raise ValueError("Scheme parameters cannot replace z, nz, n_bins, or bin_range.")
    kwargs = {
        key: value for key, value in available.items() if key in accepted and key not in reserved
    }
    return scheme(z, nz, spec["n_bins"], **kwargs)


def _resolve_mixed(
    spec: Mapping[str, Any], z: FloatArray, nz: FloatArray, options: dict[str, Any]
) -> FloatArray:
    return mixed_edges_from_segments(
        spec["segments"],
        z_axis=z,
        nz_axis=nz,
        z_ph=options.get("z_ph"),
        nz_ph=options.get("nz_ph"),
    )


_RESOLVERS = {
    "explicit": _resolve_explicit,
    "generated": _resolve_generated,
    "mixed": _resolve_mixed,
}


def resolve_bin_edges(
    *,
    z_axis: FloatArray,
    nz_axis: FloatArray,
    bins: Mapping[str, Any],
    equal_number_axis: Any | None = None,
    equal_number_weights: Any | None = None,
    z_ph: Any | None = None,
    nz_ph: Any | None = None,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    normalize_equal_number_weights: bool = True,
) -> FloatArray:
    """Resolve a dictionary bin specification into numerical edges.

    ``bins`` supports exactly one of:

    - ``{"edges": [...]}`` for explicit edges.
    - ``{"scheme": name, "n_bins": n, "range": (lo, hi), "params": {...}}``
      for generated edges.
    - ``{"segments": [...]}`` for mixed binning.

    The optional generated ``range`` restricts edge construction, not the
    parent evaluation grid.
    """
    mode, spec = validate_bins(bins)
    z, nz = validate_axis_and_weights(z_axis, nz_axis)

    options = {
        "equal_number_axis": equal_number_axis,
        "equal_number_weights": equal_number_weights,
        "norm_method": norm_method,
        "normalize_equal_number_weights": normalize_equal_number_weights,
        "z_ph": z_ph,
        "nz_ph": nz_ph,
    }

    return _RESOLVERS[mode](spec, z, nz, options)


def validate_bin_edges(
    bin_edges: Any,
    *,
    require_within: tuple[float, float] | None = None,
) -> FloatArray:
    """Validates a bin-edge array.

    Args:
        bin_edges: Candidate bin edges.
        require_within: Optional `(min, max)` interval that the edges must lie
            within.

    Returns:
        A validated 1D array of bin edges as ``float64``.

    Raises:
        ValueError: If the edges are not 1D, not finite, not strictly
            increasing, or fall outside `require_within`.
    """
    be = np.asarray(bin_edges, dtype=float)

    if be.ndim != 1:
        raise ValueError("bin_edges must be 1D.")
    if be.size < 2:
        raise ValueError("bin_edges must have at least two entries.")
    if not np.all(np.isfinite(be)):
        raise ValueError("bin_edges must contain only finite values.")
    if not np.all(np.diff(be) > 0):
        raise ValueError("bin_edges must be strictly increasing.")

    if require_within is not None:
        lo, hi = float(require_within[0]), float(require_within[1])
        if be[0] < lo or be[-1] > hi:
            raise ValueError(f"bin_edges must lie within [{lo}, {hi}], got [{be[0]}, {be[-1]}].")

    n_bins_eff = int(be.size - 1)
    validate_n_bins(n_bins_eff)
    return be.astype(np.float64, copy=False)


def photoz_segments_to_axes(
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Converts photo-z segment specifications to the mixed-binning format.

    This helper translates a list of segment mappings (each describing a binning
    scheme and redshift range) into the format expected by
    :func:`binny.axes.mixed_edges.mixed_edges`.

    Each input segment must provide:
        - ``"scheme"``: binning method name (aliases resolved downstream).
        - ``"n_bins"``: number of bins in this segment.
        - ``"z_min"``: lower redshift bound.
        - ``"z_max"``: upper redshift bound.

    Args:
        segments: Sequence of segment specification mappings.

    Returns:
        List of axes-style segment dictionaries with keys ``"method"``,
        ``"n_bins"``, and ``"params"`` (containing ``"x_min"`` and ``"x_max"``).

    Raises:
        ValueError: If any required key is missing from a segment.
    """
    out: list[dict[str, Any]] = []
    for seg in segments:
        try:
            scheme = str(seg["scheme"])
            n_bins = int(seg["n_bins"])
            z_min = float(seg["z_min"])
            z_max = float(seg["z_max"])
        except KeyError as e:
            raise ValueError(f"Missing key in mixed bin spec: {e.args[0]!r}") from e

        out.append(
            {
                "method": scheme,  # axes will resolve aliases
                "n_bins": n_bins,
                "params": {"x_min": z_min, "x_max": z_max},
            }
        )
    return out


def mixed_edges_from_segments(
    segments: Sequence[Mapping[str, Any]],
    *,
    z_axis: Any,
    nz_axis: Any,
    z_ph: Any | None,
    nz_ph: Any | None,
) -> np.ndarray:
    """Returns bin edges for a mixed segment specification.

    This function creates a single 1D array of bin edges by stitching together
    multiple binning segments. Each segment defines a redshift interval and a
    binning scheme (e.g. equidistant, log, equal-number).

    For equal-number segments, the function needs an axis and weight curve that
    represent the population being split. You may provide a photo-z proxy
    ``(z_ph, nz_ph)``. If not provided, the function uses ``(z_axis, nz_axis)``
    as a proxy.

    Args:
        segments: Sequence of segment specification mappings. Each segment must
            include ``scheme``, ``n_bins``, ``z_min``, and ``z_max``.
        z_axis: Redshift axis for the parent distribution (proxy for photo-z if
            ``z_ph`` is not provided).
        nz_axis: Weights or distribution values evaluated on ``z_axis``.
        z_ph: Optional photo-z axis used for equal-number binning decisions.
        nz_ph: Optional photo-z weights evaluated on ``z_ph``.

    Returns:
        One-dimensional NumPy array of bin edges of length ``(N_total + 1)``,
        where ``N_total`` is the sum of ``n_bins`` over segments.

    Raises:
        ValueError: If exactly one of ``z_ph`` or ``nz_ph`` is provided.
        ValueError: If any segment is missing required keys.
        ValueError: If the provided axis/weights are invalid (propagated from
            validation utilities or downstream edge builders).
    """
    if (z_ph is None) ^ (nz_ph is None):
        raise ValueError(
            "Provide both z_ph and nz_ph, or neither. "
            "If neither is provided, (z, nz) is used as a proxy for (z_ph, nz_ph)."
        )

    if z_ph is not None:
        x_full, w_full = validate_axis_and_weights(z_ph, nz_ph)
    else:
        x_full, w_full = validate_axis_and_weights(z_axis, nz_axis)

    segments_axes = photoz_segments_to_axes(segments)
    total = sum(int(s["n_bins"]) for s in segments_axes)

    return mixed_edges_axes(
        segments_axes,
        x=x_full,
        weights=w_full,
        total_n_bins=total,
    )
