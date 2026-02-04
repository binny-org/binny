"""Shared utilities for tomographic redshift binning.

This module contains the shared building blocks for creating tomographic bins
from a parent redshift distribution. Photo-z and spec-z builders use these
helpers to keep behavior consistent across binning methods.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from binny.axes.bin_edges import equal_number_edges, equidistant_edges
from binny.nz_tomo.nz_tomo_utils import mixed_edges_from_segments
from binny.utils.metadata import build_tomo_bins_metadata, save_metadata_txt
from binny.utils.normalization import normalize_1d
from binny.utils.types import BinningScheme, FloatArray
from binny.utils.validators import validate_axis_and_weights, validate_n_bins

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
    equal_number_axis: Any | None = None,
    equal_number_weights: Any | None = None,
    z_ph: Any | None = None,
    nz_ph: Any | None = None,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    normalize_equal_number_weights: bool = True,
) -> FloatArray:
    """Returns tomographic bin edges.

    Args:
        z_axis: Default axis used when edges are derived from the input range.
        nz_axis: Default weights used when edges are derived from weights.
        bin_edges: Explicit bin edges. Mutually exclusive with `binning_scheme`
            and `n_bins`.
        binning_scheme: Binning description. May be a scheme name (e.g.
            `"equidistant"`, `"equal_number"`) or a mixed-segment specification.
        n_bins: Number of bins for simple scheme names.
        bin_range: Optional `(min, max)` range for equidistant binning.
        equal_number_axis: Optional axis used for equal-number binning.
        equal_number_weights: Optional weights used for equal-number binning.
        z_ph: Optional axis used by mixed-segment specifications.
        nz_ph: Optional weights used by mixed-segment specifications.
        norm_method: Normalization method used when normalizing weights.
        normalize_equal_number_weights: Whether to normalize the weights used
            for equal-number binning.

    Returns:
        A 1D array of bin edges.

    Raises:
        ValueError: If inputs are inconsistent or the binning specification is
            not supported.
    """
    z_arr, nz_arr = validate_axis_and_weights(z_axis, nz_axis)

    if bin_edges is not None:
        if (binning_scheme is not None) or (n_bins is not None):
            raise ValueError("Provide either bin_edges or (binning_scheme, n_bins), not both.")
        return np.asarray(bin_edges, dtype=float)

    if binning_scheme is None:
        raise ValueError("bin_edges is None. You must provide binning_scheme.")

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
    """Builds tomographic bins on fixed edges.

    Args:
        z: Redshift grid for all returned curves.
        nz_parent_for_meta: Parent distribution used for metadata.
        bin_edges: Validated 1D array of bin edges.
        raw_bin_for_edge: Callable that returns the raw bin curve for a bin
            index and edge pair.
        normalize_bins: Whether to normalize each bin curve.
        norm_method: Normalization method used when `normalize_bins=True`.
        mixer: Optional transformation applied to the set of raw bins.
        need_meta: Whether to compute per-bin norms and the parent norm.

    Returns:
        A tuple `(bins, bins_norms, parent_norm)`, where `bins` is a mapping
        of bin index to bin curve, `bins_norms` is the per-bin integral (or
        `None`), and `parent_norm` is the parent integral (or `None`).

    Raises:
        ValueError: If a raw bin curve does not match the shape of `z`.
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
    """Builds tomography metadata and optionally writes it to disk.

    Args:
        kind: Tomography kind label (e.g. `"photoz"`, `"specz"`).
        z: Redshift grid.
        parent_nz: Parent redshift distribution.
        bin_edges: Bin-edge array used to construct the bins.
        bins: Mapping of bin index to bin curve.
        inputs: Input settings to record in the metadata.
        parent_norm: Integral of the parent distribution, if available.
        bins_norms: Per-bin integrals, if available.
        include_metadata: Whether to return the metadata dictionary.
        save_metadata_path: Optional path for writing a text metadata file.

    Returns:
        Metadata dictionary if requested, otherwise `None`.
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
