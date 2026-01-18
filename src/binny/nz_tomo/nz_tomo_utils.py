"""Utility functions for ``binny.nz_tomo`` module."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from binny.axes.mixed_edges import mixed_edges as mixed_edges_axes
from binny.utils.normalization import as_float_array
from binny.utils.validators import validate_axis_and_weights

__all__ = [
    "photoz_segments_to_axes",
    "mixed_edges_from_segments",
    "extract_bin_edges_from_meta",
    "resolve_n_bins_for_builder",
    "resolve_bin_edges_for_leakage",
]


def photoz_segments_to_axes(
    segments: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert a photo-z-style mixed-bin spec into an axes-style segment spec.

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
    """Build mixed bin edges from a segmented binning specification.

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
        One-dimensional numpy array of bin edges of length ``(N_total + 1)``,
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


def extract_bin_edges_from_meta(
    meta: Mapping[str, Any],
) -> NDArray[np.float64] | None:
    """Extract bin edges from builder metadata.

    This helper pulls out the ``bin_edges`` entry from tomography builder metadata
    when present. It is used to support downstream diagnostics (e.g., leakage
    matrices) without forcing users to thread bin edges through manually.

    Args:
        meta: Builder metadata mapping.

    Returns:
        A float64 array of bin edges if present, otherwise ``None``.
    """
    be = meta.get("bin_edges", None)
    if be is None:
        return None
    return as_float_array(be, name="bin_edges")


def resolve_n_bins_for_builder(*, bin_edges: Any | None, n_bins: int | None) -> int | None:
    """Resolve the effective ``n_bins`` argument for bin builders.

    This helper enforces a single rule: when explicit ``bin_edges`` are provided,
    builders should not also receive ``n_bins``. It keeps the logic consistent
    across config-driven and programmatic entry points.

    Args:
        bin_edges: Optional explicit bin edge specification.
        n_bins: Optional number of bins.

    Returns:
        ``None`` if ``bin_edges`` is provided, otherwise ``n_bins``.
    """
    return None if bin_edges is not None else n_bins


def resolve_bin_edges_for_leakage(
    *, bin_edges: Any | None, cached_bin_edges: Any | None
) -> NDArray[np.float64]:
    """Resolve bin edges for leakage-style diagnostics.

    This helper supports two user workflows: providing explicit bin edges at call
    time, or relying on cached bin edges from the most recent tomography build.
    It exists to keep error messages consistent and avoid duplicating checks.

    Args:
        bin_edges: Explicit bin edges provided by the caller.
        cached_bin_edges: Cached bin edges stored on a wrapper instance.

    Returns:
        Bin edges as a float64 NumPy array.

    Raises:
        ValueError: If neither explicit nor cached bin edges are available.
    """
    if bin_edges is None:
        if cached_bin_edges is None:
            raise ValueError("bin_edges is required (not provided and not cached).")
        return as_float_array(cached_bin_edges, name="cached_bin_edges")
    return as_float_array(bin_edges, name="bin_edges")
