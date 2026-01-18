r"""Agnostic core utilities for building tomographic redshift bins.

This module factors out the shared mechanics of tomography builders:

- resolving bin edges from explicit edges, simple schemes, or mixed segments
- validating edges consistently
- constructing per-bin curves via a callback on a common true-z grid
- optional post-processing of raw bins (e.g. response-matrix mixing)
- per-bin normalization and metadata bookkeeping

It does **not** implement any survey- or model-specific physics. Photo-z and
spec-z builders provide the bin-construction callback(s) and any extra modeling
steps.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from binny.axes.bin_edges import equal_number_edges, equidistant_edges
from binny.nz_tomo.nz_tomo_utils import mixed_edges_from_segments
from binny.utils.metadata import build_tomo_bins_metadata, save_metadata_txt
from binny.utils.normalization import normalize_1d
from binny.utils.validators import validate_axis_and_weights, validate_n_bins

FloatArray: TypeAlias = NDArray[np.float64]
BinningScheme: TypeAlias = str | Sequence[Mapping[str, Any]] | Mapping[str, Any]

__all__ = [
    "resolve_bin_edges",
    "validate_bin_edges",
    "build_bins_on_edges",
    "finalize_tomo_metadata",
]


def resolve_bin_edges(
    *,
    z_axis: FloatArray,
    nz_axis: FloatArray,
    bin_edges: Any | None,
    binning_scheme: BinningScheme | None,
    n_bins: int | None,
    bin_range: tuple[float, float] | None = None,
    # equal-number needs some axis/weights; default is (z_axis, nz_axis)
    equal_number_axis: Any | None = None,
    equal_number_weights: Any | None = None,
    # mixed edges may also want photo-z proxies
    z_ph: Any | None = None,
    nz_ph: Any | None = None,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    normalize_equal_number_weights: bool = True,
) -> FloatArray:
    """Resolves bin edges from explicit edges, a simple scheme, or a mixed spec.

    Args:
        z_axis: Default axis used for equidistant edges and as a fallback for
            equal-number edges.
        nz_axis: Default weights used as a fallback for equal-number edges.
        bin_edges: Explicit edges (mutually exclusive with scheme).
        binning_scheme: String scheme name or mixed segments spec.
        n_bins: Number of bins for string schemes.
        bin_range: Range for equidistant edges (mutually exclusive with scheme).
        equal_number_axis: Axis used for equal-number binning if provided.
        equal_number_weights: Weights used for equal-number binning if provided.
        z_ph: Optional photo-z axis used by mixed segments helper.
        nz_ph: Optional photo-z weights used by mixed segments helper.
        norm_method: Normalization method for :func:`normalize_1d` when
            normalizing equal-number weights.
        normalize_equal_number_weights: If True, normalize the weights used to
            compute equal-number edges.

    Returns:
        One-dimensional float array of edges.

    Raises:
        ValueError: If inputs are inconsistent (e.g. both edges and scheme).
        ValueError: If scheme is unsupported or malformed.
    """
    z_arr, nz_arr = validate_axis_and_weights(z_axis, nz_axis)

    if bin_edges is not None:
        if (binning_scheme is not None) or (n_bins is not None):
            raise ValueError("Provide either bin_edges or (binning_scheme, n_bins), not both.")
        return np.asarray(bin_edges, dtype=float)

    if binning_scheme is None:
        raise ValueError("bin_edges is None. You must provide binning_scheme.")

    # --- string scheme
    if isinstance(binning_scheme, str):
        if n_bins is None:
            raise ValueError(
                "bin_edges is None and binning_scheme is a string. You must provide n_bins."
            )
        validate_n_bins(n_bins)

        scheme = binning_scheme.lower()
        if scheme in {"equidistant", "eq", "linear"}:
            if bin_range is not None:
                lo, hi = float(bin_range[0]), float(bin_range[1])
                if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
                    raise ValueError("bin_range must be finite with bin_range[0] < bin_range[1].")
                return equidistant_edges(lo, hi, n_bins)
            return equidistant_edges(float(z_arr[0]), float(z_arr[-1]), n_bins)

        if scheme in {"equal_number", "equipopulated", "en"}:
            if (equal_number_axis is None) ^ (equal_number_weights is None):
                raise ValueError(
                    "Provide both equal_number_axis and equal_number_weights, or neither."
                )

            if equal_number_axis is None:
                x, w = z_arr, nz_arr
            else:
                x, w = validate_axis_and_weights(equal_number_axis, equal_number_weights)

            if normalize_equal_number_weights:
                total = float(np.trapezoid(w, x=x))
                if total > 0.0 and not np.isclose(total, 1.0, rtol=1e-3, atol=1e-4):
                    w = normalize_1d(x, w, method=norm_method)

            return equal_number_edges(x, w, n_bins)

        raise ValueError(
            "Unsupported binning_scheme. Supported: "
            "'equidistant' (eq/linear) and 'equal_number' (equipopulated/en)."
        )

    # --- mixed segments (sequence or mapping with 'segments')
    if n_bins is not None:
        raise ValueError("In mixed binning mode, set n_bins per segment and leave n_bins=None.")

    if isinstance(binning_scheme, Mapping):
        if "segments" not in binning_scheme:
            raise ValueError(
                "Mixed binning dict must contain key 'segments' (e.g. {'segments': [...] })."
            )
        segments = binning_scheme["segments"]
    else:
        segments = binning_scheme

    if not isinstance(segments, Sequence) or isinstance(segments, str | bytes):
        raise ValueError("Mixed binning requires a sequence of segment dicts.")

    return mixed_edges_from_segments(
        segments,
        z_axis=z_arr,
        nz_axis=nz_arr,
        z_ph=z_ph,
        nz_ph=nz_ph,
    )


def validate_bin_edges(
    bin_edges: Any,
    *,
    require_within: tuple[float, float] | None = None,
) -> FloatArray:
    """Validates bin edges for tomography builders.

    Args:
        bin_edges: Candidate bin edge array.
        require_within: Optional (min, max) interval that the edges must lie
            within. Use this for true-z binning; leave None for photo-z edges.

    Returns:
        Validated bin edge array as float64.

    Raises:
        ValueError: If edges are not 1D, finite, strictly increasing, or outside
            the required interval.
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


def build_bins_on_edges(
    *,
    z: FloatArray,
    nz_parent_for_meta: FloatArray,
    bin_edges: FloatArray,
    raw_bin_for_edge: Callable[[int, float, float], FloatArray],
    normalize_bins: bool,
    norm_method: Literal["trapezoid", "simpson"],
    mixer: Callable[[dict[int, FloatArray]], dict[int, FloatArray]] | None = None,
    need_meta: bool,
) -> tuple[dict[int, FloatArray], dict[int, float] | None, float | None]:
    """Builds tomographic bins from edges using a raw-bin callback.

    The callback must return the *raw* (pre-normalization) bin curve on the same
    ``z`` grid.

    Args:
        z: True-z grid.
        nz_parent_for_meta: Parent curve used for metadata bookkeeping
            (typically the original nz passed by the user, not per-builder
            normalized variants).
        bin_edges: Validated 1D edge array.
        raw_bin_for_edge: Callback ``(i, zmin, zmax) -> raw_bin(z)``.
        normalize_bins: Whether to normalize each returned bin.
        norm_method: Normalization method.
        mixer: Optional post-processing step applied to the dict of raw bins
            *before* norms/normalization (e.g. response-matrix mixing).
        need_meta: Whether to compute parent_norm and bins_norms.

    Returns:
        (bins, bins_norms, parent_norm)
    """
    z_arr = np.asarray(z, dtype=float)
    parent_arr = np.asarray(nz_parent_for_meta, dtype=float)

    n_bins = int(bin_edges.size - 1)

    # Build raw bins first
    raw_bins: dict[int, FloatArray] = {}
    for i, (a, b) in enumerate(zip(bin_edges[:-1], bin_edges[1:], strict=False)):
        raw = raw_bin_for_edge(i, float(a), float(b))
        arr = np.asarray(raw, dtype=np.float64)
        if arr.shape != z_arr.shape:
            raise ValueError(
                f"raw_bin_for_edge returned shape {arr.shape}, expected {z_arr.shape}."
            )
        raw_bins[i] = arr

    # Optional mixing (spec-z)
    if mixer is not None:
        raw_bins = mixer(raw_bins)

    bins_norms: dict[int, float] | None = {} if need_meta else None
    parent_norm: float | None = float(np.trapezoid(parent_arr, x=z_arr)) if need_meta else None

    # Track norms and optionally normalize
    out: dict[int, FloatArray] = {}
    for i in range(n_bins):
        arr = raw_bins[i]

        area = float(np.trapezoid(arr, x=z_arr))
        if bins_norms is not None:
            bins_norms[i] = area

        if normalize_bins:
            if np.isclose(area, 0.0, atol=1e-12):
                out[i] = arr
                continue
            arr = normalize_1d(z_arr, arr, method=norm_method)

        out[i] = arr

    return out, bins_norms, parent_norm


def finalize_tomo_metadata(
    *,
    kind: str,
    z: FloatArray,
    parent_nz: FloatArray,
    bin_edges: FloatArray,
    bins: Mapping[int, FloatArray],
    inputs: Mapping[str, Any],
    parent_norm: float | None,
    bins_norms: Mapping[int, float] | None,
    include_metadata: bool,
    save_metadata_path: str | None,
) -> dict[str, Any] | None:
    """Build and optionally save tomography metadata.

    Returns None if neither include_metadata nor save_metadata_path is set.
    """
    need_meta = include_metadata or (save_metadata_path is not None)
    if not need_meta:
        return None

    frac_per_bin: dict[int, float] | None
    if parent_norm is None or parent_norm == 0.0 or bins_norms is None:
        frac_per_bin = None
    else:
        frac_per_bin = {i: float(bins_norms[i] / parent_norm) for i in bins_norms}

    meta = build_tomo_bins_metadata(
        kind=kind,
        z=np.asarray(z, dtype=float),
        parent_nz=np.asarray(parent_nz, dtype=float),
        bin_edges=np.asarray(bin_edges, dtype=float),
        bins_returned=dict(bins),
        inputs=dict(inputs),
        parent_norm=parent_norm,
        bins_norms=dict(bins_norms) if bins_norms is not None else None,
        frac_per_bin=frac_per_bin,
    )

    if save_metadata_path is not None:
        save_metadata_txt(meta, save_metadata_path)

    return meta if include_metadata else None
