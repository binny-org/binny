"""Resolve bin-edge specifications for one-dimensional sampled densities."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import numpy as np

from binny.axes.bin_edges import (
    chebyshev_edges,
    equal_information_edges,
    equal_number_edges,
    equidistant_edges,
    generalized_tomographic_edges,
    geometric_edges,
    log_edges,
    power_edges,
    sinh_edges,
)
from binny.axes.mixed_edges import mixed_edges as combine_mixed_edges
from binny.binning.scheme_registry import (
    get_binning_scheme,
    register_binning_scheme,
)
from binny.binning.specs import restrict_bin_range, validate_bins
from binny.utils.normalization import normalize_1d
from binny.utils.types import FloatArray
from binny.utils.validators import (
    validate_axis_and_weights,
    validate_n_bins,
)

__all__ = [
    "resolve_bin_edges",
    "validate_bin_edges",
    "segments_to_axes",
    "photoz_segments_to_axes",
    "mixed_edges_from_segments",
]


@register_binning_scheme("equidistant", aliases=("eq", "eq_dist", "linear"))
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


@register_binning_scheme("power")
def _power_scheme(
    axis: FloatArray,
    density: FloatArray,
    n_bins: int,
    *,
    power: float = 2.0,
) -> FloatArray:
    """Construct power-law spaced boundaries over the input interval."""
    return power_edges(
        float(axis[0]),
        float(axis[-1]),
        n_bins,
        power=power,
    )


@register_binning_scheme("chebyshev")
def _chebyshev_scheme(
    axis: FloatArray,
    density: FloatArray,
    n_bins: int,
) -> FloatArray:
    """Construct Chebyshev-Lobatto boundaries over the input interval."""
    return chebyshev_edges(float(axis[0]), float(axis[-1]), n_bins)


@register_binning_scheme("sinh")
def _sinh_scheme(
    axis: FloatArray,
    density: FloatArray,
    n_bins: int,
    *,
    strength: float = 3.0,
    concentration: Literal["midpoint", "endpoints"] = "midpoint",
) -> FloatArray:
    """Construct sinh-stretched boundaries over the input interval."""
    return sinh_edges(
        float(axis[0]),
        float(axis[-1]),
        n_bins,
        strength=strength,
        concentration=concentration,
    )


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


@register_binning_scheme(
    "equal_number",
    aliases=("equipopulated", "en"),
)
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
            weights = normalize_1d(
                x,
                weights,
                method=norm_method,
            )

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
    """Resolve a registered scheme over an optionally restricted interval."""
    scheme = get_binning_scheme(spec["scheme"])

    original_axis = axis
    bin_range = spec.get("range")

    axis, density = restrict_bin_range(
        axis,
        density,
        bin_range,
    )

    params = dict(spec.get("params", {}) or {})
    available = {**options, **params}

    # Restrict distance samples to the selected interval.
    chi_input = available.get("comoving_distance")

    if chi_input is not None:
        chi = np.asarray(chi_input, dtype=float)

        if chi.shape != original_axis.shape:
            raise ValueError("comoving_distance must match the original axis grid.")

        if not np.all(np.isfinite(chi)):
            raise ValueError("comoving_distance must be finite.")

        if np.any(np.diff(chi) <= 0):
            raise ValueError("comoving_distance must be strictly increasing.")

        if bin_range is not None:
            available["comoving_distance"] = np.interp(
                axis,
                original_axis,
                chi,
            )

    # Independent equal-number weights may use another sampling grid.
    separate_axis = available.get("equal_number_axis")
    separate_weights = available.get("equal_number_weights")

    if (separate_axis is None) ^ (separate_weights is None):
        raise ValueError("Provide both equal_number_axis and equal_number_weights, or neither.")

    if separate_axis is not None and bin_range is not None:
        x, weights = restrict_bin_range(
            separate_axis,
            separate_weights,
            bin_range,
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
    """Resolve mixed segments using the registered binning schemes."""
    return mixed_edges_from_segments(
        spec["segments"],
        axis=axis,
        density=density,
        selection_axis=options.get("selection_axis"),
        selection_density=options.get("selection_density"),
        equal_number_axis=options.get("equal_number_axis"),
        equal_number_weights=options.get("equal_number_weights"),
        norm_method=options.get("norm_method", "trapezoid"),
        normalize_equal_number_weights=options.get(
            "normalize_equal_number_weights",
            True,
        ),
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

    Supports explicitly supplied boundaries, registered binning schemes,
    and combinations of different schemes over adjacent intervals.

    The sampling coordinate may represent redshift, magnitude, mass,
    luminosity, or another numerical quantity.

    Density-dependent schemes use the supplied density by default.
    Independent distributions can be provided for equal-number binning
    or mixed-segment selection without modifying the primary density.

    Args:
        axis: Strictly increasing one-dimensional sampling coordinates.
        density: Nonnegative density sampled on ``axis``. Normalization
            is not required.
        bins: Binning specification. Supported forms are
            ``{"edges": [...]}`` for explicit boundaries,
            ``{"scheme": name, "n_bins": n, "range": (lo, hi),
            "params": {...}}`` for generated boundaries, and
            ``{"segments": [...]}`` for mixed binning.
        equal_number_axis: Optional independent coordinates for
            equal-number binning. Requires ``equal_number_weights``.
        equal_number_weights: Population weights sampled on
            ``equal_number_axis``.
        selection_axis: Optional independent coordinates for mixed
            binning. Requires ``selection_density``.
        selection_density: Density sampled on ``selection_axis``.
        norm_method: Integration method for normalizing equal-number
            weights, either ``"trapezoid"`` or ``"simpson"``.
        normalize_equal_number_weights: Whether to normalize
            equal-number weights before constructing boundaries.

    Returns:
        Finite, strictly increasing one-dimensional bin boundaries.

    Raises:
        ValueError: If the sampling arrays, specification, scheme
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

    The boundaries must be finite, strictly increasing, and define
    at least one bin. An optional interval can constrain their range.

    Args:
        bin_edges: Candidate bin boundaries.
        require_within: Optional inclusive interval containing all
            boundaries, specified as ``(minimum, maximum)``.

    Returns:
        Validated one-dimensional float64 array.

    Raises:
        ValueError: If the boundaries are not one-dimensional, contain
            fewer than two values, are non-finite, are not strictly
            increasing, or lie outside the required interval.
    """
    edges = np.asarray(bin_edges, dtype=np.float64)

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

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError("require_within must contain a valid finite interval.")

        if edges[0] < lo or edges[-1] > hi:
            raise ValueError(
                f"bin_edges must lie within [{lo}, {hi}], got [{edges[0]}, {edges[-1]}]."
            )

    validate_n_bins(int(edges.size - 1))

    return edges


def segments_to_axes(
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert mixed-segment specifications to generic axis descriptions.

    Each segment specifies a registered binning scheme, a number of
    bins, and a coordinate interval. Generic ``x_min`` and ``x_max``
    boundaries are preferred, while legacy ``z_min`` and ``z_max``
    keys remain supported.

    The returned specifications use the same numerical coordinate
    convention regardless of the physical meaning of the axis.

    Args:
        segments: Ordered segment mappings containing ``scheme``,
            ``n_bins``, and interval boundaries. Optional scheme
            parameters may be supplied through ``params``.

    Returns:
        List of segment dictionaries containing ``method``,
        ``n_bins``, and ``params`` with generic interval boundaries.

    Raises:
        ValueError: If a segment is missing a required field, has
            invalid boundaries, or contains an incomplete interval.
    """
    result: list[dict[str, Any]] = []

    for index, segment in enumerate(segments):
        try:
            scheme = str(segment["scheme"])
            n_bins = int(segment["n_bins"])
        except KeyError as exc:
            raise ValueError(f"Segment {index} is missing {exc.args[0]!r}.") from exc

        validate_n_bins(n_bins)

        params = dict(segment.get("params", {}) or {})

        if "range" in segment:
            x_min, x_max = segment["range"]
        else:
            x_min = segment.get(
                "x_min",
                segment.get("z_min", params.get("x_min", params.get("z_min"))),
            )
            x_max = segment.get(
                "x_max",
                segment.get("z_max", params.get("x_max", params.get("z_max"))),
            )

        if x_min is None or x_max is None:
            raise ValueError(f"Segment {index} requires x_min and x_max (or z_min and z_max).")

        lo, hi = float(x_min), float(x_max)

        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Segment {index} has an invalid interval: ({lo}, {hi}).")

        for key in ("x_min", "x_max", "z_min", "z_max"):
            params.pop(key, None)

        result.append(
            {
                "method": scheme,
                "n_bins": n_bins,
                "range": (lo, hi),
                "params": params,
            }
        )

    return result


def photoz_segments_to_axes(
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert legacy photo-z segments to generic axis descriptions.

    This compatibility function accepts the same segment specifications
    as :func:`segments_to_axes`. New code should call that function
    directly.

    Args:
        segments: Ordered segment specifications.

    Returns:
        List of generic axis-level segment descriptions.

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
    equal_number_axis: Any | None = None,
    equal_number_weights: Any | None = None,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    normalize_equal_number_weights: bool = True,
) -> np.ndarray:
    """Construct bin boundaries from adjacent, independently defined segments.

    Each segment specifies a coordinate interval, a registered binning
    scheme, and the number of bins to construct within that interval.

    Every segment is resolved through the same scheme registry used
    for ordinary binning. The resulting boundaries are combined using
    :func:`binny.axes.mixed_edges.mixed_edges`.

    Density-dependent schemes use the primary distribution unless an
    independent selection distribution is provided. Equal-number
    segments may also use explicitly supplied independent population
    weights.

    Adjacent segments must share a boundary. The shared boundary is
    included only once in the final result.

    Args:
        segments: Ordered segment specifications containing ``scheme``,
            ``n_bins``, interval boundaries, and optional ``params``.
        axis: Strictly increasing primary sampling coordinates.
        density: Nonnegative primary density sampled on ``axis``.
        selection_axis: Optional independent sampling coordinates for
            density-dependent mixed segments. Requires
            ``selection_density``.
        selection_density: Density sampled on ``selection_axis``.
            If omitted, the primary distribution is used.
        info_density: Optional information-density samples evaluated
            on the selected sampling axis.
        equal_number_axis: Optional independent coordinates for
            equal-number binning. Requires ``equal_number_weights``.
        equal_number_weights: Population weights sampled on
            ``equal_number_axis``.
        norm_method: Integration method used when normalizing
            equal-number weights.
        normalize_equal_number_weights: Whether to normalize
            equal-number weights before constructing boundaries.

    Returns:
        Strictly increasing array of combined bin boundaries, with
        length equal to the total number of bins plus one.

    Raises:
        ValueError: If the sampling arrays, segment specifications,
            scheme parameters, or combined boundaries are invalid.
    """
    if (selection_axis is None) ^ (selection_density is None):
        raise ValueError("Provide both selection_axis and selection_density, or neither.")

    if (equal_number_axis is None) ^ (equal_number_weights is None):
        raise ValueError("Provide both equal_number_axis and equal_number_weights, or neither.")

    primary_axis, primary_density = validate_axis_and_weights(
        axis,
        density,
    )

    if selection_axis is None:
        selected_axis = primary_axis
        selected_density = primary_density
    else:
        selected_axis, selected_density = validate_axis_and_weights(
            selection_axis,
            selection_density,
        )

    if info_density is not None:
        _, info_density = validate_axis_and_weights(
            selected_axis,
            info_density,
        )

    converted = segments_to_axes(segments)
    segment_edges: list[np.ndarray] = []

    for segment in converted:
        scheme = segment["method"]
        bin_range = segment["range"]
        params = dict(segment["params"])

        # Selection coordinates determine density-dependent boundaries.
        segment_axis = selected_axis
        segment_density = selected_density

        # Information density must be restricted to the same interval.
        if info_density is not None and "info_density" not in params:
            _, restricted_info = restrict_bin_range(
                selected_axis,
                info_density,
                bin_range,
            )
            params["info_density"] = restricted_info

        # Explicit population weights take precedence over selection data.
        if (
            equal_number_axis is not None
            and "equal_number_axis" not in params
            and "equal_number_weights" not in params
        ):
            params["equal_number_axis"] = equal_number_axis
            params["equal_number_weights"] = equal_number_weights

        spec = {
            "scheme": scheme,
            "n_bins": segment["n_bins"],
            "range": bin_range,
            "params": params,
        }

        edges = _resolve_generated(
            spec,
            segment_axis,
            segment_density,
            {
                "norm_method": norm_method,
                "normalize_equal_number_weights": (normalize_equal_number_weights),
            },
        )

        segment_edges.append(validate_bin_edges(edges))

    total_n_bins = sum(segment["n_bins"] for segment in converted)

    return combine_mixed_edges(
        segment_edges,
        total_n_bins=total_n_bins,
    )
