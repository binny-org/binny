"""Resolve bin-edge specifications for one-dimensional sampled densities."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from binny.axes.bin_edges import (
    equal_information_edges,
    equal_number_edges,
    equidistant_edges,
    generalized_tomographic_edges,
    geometric_edges,
    log_edges,
)
from binny.axes.mixed_edges import mixed_edges as mixed_edges_axes
from binny.binning.scheme_registry import (
    get_binning_scheme,
    register_binning_scheme,
)
from binny.binning.specs import restrict_bin_range, validate_bins
from binny.utils.normalization import normalize_1d
from binny.utils.types import FloatArray
from binny.utils.validators import validate_axis_and_weights, validate_n_bins

__all__ = [
    "resolve_bin_edges",
    "validate_bin_edges",
    "segments_to_axes",
    "photoz_segments_to_axes",
    "mixed_edges_from_segments",
]


@register_binning_scheme("equidistant", aliases=("eq", "linear"))
def _equidistant_scheme(
    axis: FloatArray,
    density: FloatArray,
    n_bins: int,
) -> FloatArray:
    """Construct uniformly spaced boundaries over the input interval."""
    return equidistant_edges(float(axis[0]), float(axis[-1]), n_bins)


@register_binning_scheme("logarithmic", aliases=("log",))
def _logarithmic_scheme(
    axis: FloatArray,
    density: FloatArray,
    n_bins: int,
) -> FloatArray:
    """Construct logarithmically spaced boundaries over the input interval."""
    return log_edges(float(axis[0]), float(axis[-1]), n_bins)


@register_binning_scheme("geometric", aliases=("geom",))
def _geometric_scheme(
    axis: FloatArray,
    density: FloatArray,
    n_bins: int,
) -> FloatArray:
    """Construct geometrically spaced boundaries over the input interval."""
    return geometric_edges(float(axis[0]), float(axis[-1]), n_bins)


@register_binning_scheme("equal_information")
def _equal_information_scheme(
    axis: FloatArray,
    density: FloatArray,
    n_bins: int,
    *,
    info_density: Any | None = None,
) -> FloatArray:
    """Construct boundaries enclosing equal integrated information."""
    if info_density is None:
        raise ValueError("equal_information requires bins.params.info_density.")

    return equal_information_edges(axis, info_density, n_bins)


@register_binning_scheme("equal_number", aliases=("equipopulated", "en"))
def _equal_number_scheme(
    axis: FloatArray,
    density: FloatArray,
    n_bins: int,
    *,
    equal_number_axis: Any | None = None,
    equal_number_weights: Any | None = None,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    normalize_equal_number_weights: bool = True,
) -> FloatArray:
    """Construct boundaries enclosing equal integrated population weights."""
    if (equal_number_axis is None) ^ (equal_number_weights is None):
        raise ValueError("Provide both equal_number_axis and equal_number_weights, or neither.")

    if equal_number_axis is None:
        x, weights = axis, density
    else:
        x, weights = validate_axis_and_weights(
            equal_number_axis,
            equal_number_weights,
        )

    if normalize_equal_number_weights:
        total = float(np.trapezoid(weights, x=x))

        if total > 0.0 and not np.isclose(
            total,
            1.0,
            rtol=1e-3,
            atol=1e-4,
        ):
            weights = normalize_1d(x, weights, method=norm_method)

    return equal_number_edges(x, weights, n_bins)


@register_binning_scheme("generalized")
def _generalized_scheme(
    axis: FloatArray,
    density: FloatArray,
    n_bins: int,
    *,
    alpha: float = 0.0,
    beta: float = 0.0,
    comoving_distance: Any | None = None,
    derivative_method: Literal["analytic", "numerical"] = "analytic",
) -> FloatArray:
    """Construct density- and distance-weighted generalized boundaries."""
    return generalized_tomographic_edges(
        axis,
        density,
        n_bins,
        alpha=alpha,
        beta=beta,
        comoving_distance=comoving_distance,
        derivative_method=derivative_method,
    )


def _resolve_explicit(
    spec: Mapping[str, Any],
    axis: FloatArray,
    density: FloatArray,
    options: dict[str, Any],
) -> FloatArray:
    """Validate and return explicitly supplied bin boundaries."""
    return validate_bin_edges(spec["edges"])


def _resolve_generated(
    spec: Mapping[str, Any],
    axis: FloatArray,
    density: FloatArray,
    options: dict[str, Any],
) -> FloatArray:
    """Resolve a registered scheme and its optional coordinate restriction."""
    scheme = get_binning_scheme(spec["scheme"])

    original_axis = axis
    axis, density = restrict_bin_range(
        axis,
        density,
        spec.get("range"),
    )

    params = dict(spec.get("params", {}))
    available = {**options, **params}

    # Restrict distance samples consistently with the primary axis.
    chi_input = available.get("comoving_distance")

    if chi_input is not None:
        chi = np.asarray(chi_input, dtype=float)

        if chi.shape != original_axis.shape or not np.all(np.isfinite(chi)):
            raise ValueError("comoving_distance must be finite and match the original axis grid.")

        if np.any(np.diff(chi) <= 0):
            raise ValueError("comoving_distance must be strictly increasing.")

        if spec.get("range") is not None:
            available["comoving_distance"] = np.interp(
                axis,
                original_axis,
                chi,
            )

    # Independent equal-number weights use their own sampling axis.
    separate_axis = available.get("equal_number_axis")
    separate_weights = available.get("equal_number_weights")

    if (separate_axis is None) ^ (separate_weights is None):
        raise ValueError("Provide both equal_number_axis and equal_number_weights, or neither.")

    if separate_axis is not None and spec.get("range") is not None:
        x, weights = restrict_bin_range(
            separate_axis,
            separate_weights,
            spec["range"],
        )

        available["equal_number_axis"] = x
        available["equal_number_weights"] = weights

    accepted = set(inspect.signature(scheme).parameters)
    reserved = {"axis", "density", "n_bins", "bin_range"}

    unknown = set(params) - accepted

    if unknown:
        raise ValueError(f"Unsupported parameters for {spec['scheme']!r}: {sorted(unknown)}.")

    if set(params) & reserved:
        raise ValueError("Scheme parameters cannot replace axis, density, n_bins, or bin_range.")

    kwargs = {
        key: value for key, value in available.items() if key in accepted and key not in reserved
    }

    return scheme(axis, density, spec["n_bins"], **kwargs)


def _resolve_mixed(
    spec: Mapping[str, Any],
    axis: FloatArray,
    density: FloatArray,
    options: dict[str, Any],
) -> FloatArray:
    """Resolve mixed segments using the optional selection distribution."""
    return mixed_edges_from_segments(
        spec["segments"],
        axis=axis,
        density=density,
        selection_axis=options.get("selection_axis"),
        selection_density=options.get("selection_density"),
    )


_RESOLVERS = {
    "explicit": _resolve_explicit,
    "generated": _resolve_generated,
    "mixed": _resolve_mixed,
}


def resolve_bin_edges(
    *,
    axis: FloatArray,
    density: FloatArray,
    bins: Mapping[str, Any],
    equal_number_axis: Any | None = None,
    equal_number_weights: Any | None = None,
    selection_axis: Any | None = None,
    selection_density: Any | None = None,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    normalize_equal_number_weights: bool = True,
) -> FloatArray:
    """Construct bin boundaries from a declarative binning specification.

    Supports explicit boundaries, registered binning schemes, and combinations
    of different schemes over adjacent intervals. The coordinate can represent
    redshift, magnitude, luminosity, mass, or another numerical quantity.

    Density-dependent schemes use the supplied density by default. An
    independent distribution can be provided for equal-number binning or
    mixed-segment selection without modifying the primary distribution.

    Args:
        axis: Strictly increasing one-dimensional sampling coordinates.
        density: Nonnegative density sampled on ``axis``. Normalization is
            not required.
        bins: Binning specification. Exactly one of the following forms
            must be supplied:

            - ``{"edges": [...]}`` for explicit boundaries.
            - ``{"scheme": name, "n_bins": n, "range": (lo, hi),
              "params": {...}}`` for generated boundaries.
            - ``{"segments": [...]}`` for mixed binning schemes.

            The optional ``range`` restricts the interval over which
            generated boundaries are calculated.
        equal_number_axis: Optional independent sampling coordinates for
            equal-number binning. Requires ``equal_number_weights``.
        equal_number_weights: Population weights sampled on
            ``equal_number_axis``.
        selection_axis: Optional independent sampling coordinates used to
            determine mixed-segment boundaries. Requires
            ``selection_density``.
        selection_density: Density sampled on ``selection_axis``. If no
            selection distribution is supplied, mixed binning uses the
            primary ``axis`` and ``density``.
        norm_method: Integration method for normalizing equal-number weights.
            Either ``"trapezoid"`` or ``"simpson"``.
        normalize_equal_number_weights: Whether to normalize equal-number
            weights before constructing boundaries.

    Returns:
        FloatArray: Finite, strictly increasing bin boundaries.

    Raises:
        ValueError: If the sampling arrays, binning specification, scheme
            parameters, or resulting boundaries are invalid.
    """
    mode, spec = validate_bins(bins)
    axis, density = validate_axis_and_weights(axis, density)

    options = {
        "equal_number_axis": equal_number_axis,
        "equal_number_weights": equal_number_weights,
        "selection_axis": selection_axis,
        "selection_density": selection_density,
        "norm_method": norm_method,
        "normalize_equal_number_weights": normalize_equal_number_weights,
    }

    return _RESOLVERS[mode](spec, axis, density, options)


def validate_bin_edges(
    bin_edges: Any,
    *,
    require_within: tuple[float, float] | None = None,
) -> FloatArray:
    """Validate a sequence of one-dimensional bin boundaries.

    Ensures that the boundaries are finite, strictly increasing, and define
    at least one bin. An optional interval can constrain the allowed range.

    Args:
        bin_edges: Candidate bin boundaries.
        require_within: Optional inclusive ``(minimum, maximum)`` interval
            containing all boundaries.

    Returns:
        FloatArray: Validated one-dimensional ``float64`` array.

    Raises:
        ValueError: If the boundaries are not one-dimensional, contain fewer
            than two values, are non-finite, are not strictly increasing,
            or extend outside ``require_within``.
    """
    edges = np.asarray(bin_edges, dtype=float)

    if edges.ndim != 1:
        raise ValueError("bin_edges must be 1D.")

    if edges.size < 2:
        raise ValueError("bin_edges must have at least two entries.")

    if not np.all(np.isfinite(edges)):
        raise ValueError("bin_edges must contain only finite values.")

    if not np.all(np.diff(edges) > 0):
        raise ValueError("bin_edges must be strictly increasing.")

    if require_within is not None:
        lo, hi = map(float, require_within)

        if edges[0] < lo or edges[-1] > hi:
            raise ValueError(
                f"bin_edges must lie within [{lo}, {hi}], got [{edges[0]}, {edges[-1]}]."
            )

    validate_n_bins(int(edges.size - 1))

    return edges.astype(np.float64, copy=False)


def segments_to_axes(
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert declarative binning segments to axis-level specifications.

    Each segment describes a binning strategy and the coordinate interval
    over which it applies. Generic ``x_min`` and ``x_max`` boundaries are
    preferred, while ``z_min`` and ``z_max`` remain supported for existing
    redshift-based applications.

    The returned specifications are compatible with
    :func:`binny.axes.mixed_edges.mixed_edges`.

    Args:
        segments: Ordered segment mappings. Each must contain ``scheme``,
            ``n_bins``, and either ``x_min``/``x_max`` or
            ``z_min``/``z_max``.

    Returns:
        list[dict[str, Any]]: Segment specifications containing ``method``,
        ``n_bins``, and ``params`` with generic interval boundaries.

    Raises:
        ValueError: If a segment is missing a required field or has an
            incomplete interval.
    """
    result: list[dict[str, Any]] = []

    for segment in segments:
        try:
            scheme = str(segment["scheme"])
            n_bins = int(segment["n_bins"])
        except KeyError as exc:
            raise ValueError(f"Missing key in mixed bin spec: {exc.args[0]!r}") from exc

        x_min = segment.get("x_min", segment.get("z_min"))
        x_max = segment.get("x_max", segment.get("z_max"))

        if x_min is None or x_max is None:
            raise ValueError("Each mixed segment requires x_min and x_max (or z_min and z_max).")

        params = dict(segment.get("params", {}) or {})
        params.update(
            x_min=float(x_min),
            x_max=float(x_max),
        )

        result.append(
            {
                "method": scheme,
                "n_bins": n_bins,
                "params": params,
            }
        )

    return result


def photoz_segments_to_axes(
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert legacy photo-z segment specifications to axis-level segments.

    This compatibility function accepts the same segment specifications as
    :func:`segments_to_axes`. New code should use that function directly.

    Args:
        segments: Ordered segment specifications.

    Returns:
        list[dict[str, Any]]: Axis-level segment specifications.

    Raises:
        ValueError: If a segment specification is invalid.
    """
    return segments_to_axes(segments)


def mixed_edges_from_segments(
    segments: Sequence[Mapping[str, Any]],
    *,
    axis: Any,
    density: Any,
    selection_axis: Any | None = None,
    selection_density: Any | None = None,
    info_density: Any | None = None,
) -> np.ndarray:
    """Construct bin boundaries from adjacent, independently defined segments.

    Each segment specifies a coordinate interval, a binning scheme, and the
    number of bins to construct within that interval. Different segments may
    use different strategies, such as uniform spacing, logarithmic spacing,
    equal population, or equal information.

    Density-dependent segments use the primary distribution unless an
    independent selection distribution is supplied. Adjacent segments must
    share a boundary, and their combined edges form one increasing array.

    Args:
        segments: Ordered segment specifications containing ``scheme``,
            ``n_bins``, and interval boundaries ``x_min``/``x_max`` or
            ``z_min``/``z_max``.
        axis: Strictly increasing primary sampling coordinates.
        density: Nonnegative primary density sampled on ``axis``.
        selection_axis: Optional independent sampling coordinates used for
            equal-number segments. Requires ``selection_density``.
        selection_density: Population weights sampled on ``selection_axis``.
            If omitted, equal-number segments use the primary distribution.
        info_density: Optional information-density samples on the selected
            sampling axis, used by equal-information segments.

    Returns:
        np.ndarray: Strictly increasing boundaries defining the combined
        segments, with length equal to the total number of bins plus one.

    Raises:
        ValueError: If the selection arrays are incomplete, a segment is
            invalid, or the resulting boundaries cannot be combined.
    """
    if (selection_axis is None) ^ (selection_density is None):
        raise ValueError("Provide both selection_axis and selection_density, or neither.")

    if selection_axis is None:
        x, weights = validate_axis_and_weights(axis, density)
    else:
        x, weights = validate_axis_and_weights(
            selection_axis,
            selection_density,
        )

    converted = segments_to_axes(segments)
    total = sum(int(segment["n_bins"]) for segment in converted)

    return mixed_edges_axes(
        converted,
        x=x,
        weights=weights,
        info_density=info_density,
        total_n_bins=total,
    )
