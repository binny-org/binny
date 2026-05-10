"""Metadata helpers for tomographic redshift-bin products.

This module builds and writes metadata for generated tomography outputs. The
metadata records the redshift grid, parent distribution, returned bin curves,
bin-population summaries, and the user-facing inputs used to produce the bins.

The metadata is intentionally descriptive only: this module does not decide how
number densities, bin fractions, counts, shape noise, or shot noise should be
computed. Callers compute those quantities under their own conventions and pass
them in explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

__all__ = [
    "build_tomo_bins_metadata",
    "round_floats",
    "save_metadata_txt",
]


def build_tomo_bins_metadata(
    *,
    kind: Literal["photoz", "specz"],
    z: Any,
    parent_nz: Any,
    bin_edges: Any,
    bins_returned: Mapping[int, Any],
    inputs: Mapping[str, Any],
    parent_norm: float | None = None,
    bins_norms: Mapping[int, float] | None = None,
    frac_per_bin: Mapping[int, float] | None = None,
    density_per_bin: Mapping[int, float] | None = None,
    count_per_bin: Mapping[int, float] | None = None,
    normalize_bins: bool | None = None,
    notes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata for tomographic redshift-bin outputs.

    The returned dictionary is designed to be saved alongside generated bin
    curves. It records the common redshift grid, the parent redshift
    distribution, the nominal bin definition, the returned per-bin curves, and
    optional summaries such as per-bin fractions, number densities, and counts.

    The bin curves may be normalized per bin or may carry population
    information, depending on the calling convention. This function only records
    what it is given.

    Args:
        kind: Tomography mode. Use ``"photoz"`` for photo-z selected bins and
            ``"specz"`` for spectroscopic/redshift-sharp bins.
        z: Redshift grid shared by the parent distribution and all bin curves.
        parent_nz: Parent redshift distribution evaluated on ``z``.
        bin_edges: Nominal tomographic bin edges.
        bins_returned: Mapping from bin index to the returned per-bin curve
            ``n_i(z)`` evaluated on ``z``.
        inputs: User-facing configuration used to generate the bins.
        parent_norm: Optional scalar normalization associated with the parent
            distribution.
        bins_norms: Optional mapping from bin index to per-bin normalization
            values.
        frac_per_bin: Optional mapping from bin index to the fraction of the
            parent sample in that bin.
        density_per_bin: Optional mapping from bin index to a number density
            associated with that bin. The unit convention should be documented
            in ``inputs``.
        count_per_bin: Optional mapping from bin index to galaxy counts.
        normalize_bins: Optional flag recording whether returned bin curves were
            normalized to unit integral.
        notes: Optional free-form annotations to store verbatim.

    Returns:
        A nested dictionary suitable for deterministic text or JSON output.
    """
    z_arr = np.asarray(z, dtype=float)
    parent_arr = np.asarray(parent_nz, dtype=float)
    edges_arr = np.asarray(bin_edges, dtype=float)

    bins_out: dict[int, list[float]] = {
        int(k): np.asarray(v, dtype=float).tolist() for k, v in bins_returned.items()
    }

    truez_summary = _compute_effective_truez(z_arr, bins_returned)

    meta: dict[str, Any] = {
        "kind": kind,
        "grid": {
            "z": z_arr.tolist(),
            "z_min": float(z_arr[0]) if z_arr.size else None,
            "z_max": float(z_arr[-1]) if z_arr.size else None,
            "n": int(z_arr.size),
        },
        "parent_nz": {
            "values": parent_arr.tolist(),
            "norm": None if parent_norm is None else float(parent_norm),
        },
        "bins": {
            "indices": sorted(int(i) for i in bins_returned.keys()),
            "n_bins": int(len(bins_returned)),
            "bin_edges": edges_arr.tolist(),
            "normalize_bins": normalize_bins,
            "bins_returned": bins_out,
            "bins_norms": None
            if bins_norms is None
            else {int(k): float(v) for k, v in bins_norms.items()},
            "frac_per_bin": None
            if frac_per_bin is None
            else {int(k): float(v) for k, v in frac_per_bin.items()},
            "density_per_bin": None
            if density_per_bin is None
            else {int(k): float(v) for k, v in density_per_bin.items()},
            "count_per_bin": None
            if count_per_bin is None
            else {int(k): float(v) for k, v in count_per_bin.items()},
            "truez_summary": truez_summary,
        },
        "inputs": dict(inputs),
        "description": _metadata_description(),
    }

    if notes is not None:
        meta["notes"] = dict(notes)

    return meta


def round_floats(obj: Any, decimal_places: int | None) -> Any:
    """Round floating-point values in nested metadata.

    Args:
        obj: Metadata object to process. May be a scalar, mapping, list, tuple,
            NumPy scalar, or nested combination of these.
        decimal_places: Number of decimal places to keep. If None, ``obj`` is
            returned unchanged.

    Returns:
        A copy of ``obj`` with floating-point values rounded where applicable.
    """
    if decimal_places is None:
        return obj

    if isinstance(obj, float):
        return float(round(obj, decimal_places))

    if isinstance(obj, np.floating):
        return float(round(float(obj), decimal_places))

    if isinstance(obj, Mapping):
        return {k: round_floats(v, decimal_places) for k, v in obj.items()}

    if isinstance(obj, list):
        return [round_floats(v, decimal_places) for v in obj]

    if isinstance(obj, tuple):
        return tuple(round_floats(v, decimal_places) for v in obj)

    return obj


def save_metadata_txt(
    meta: Mapping[str, Any],
    path: str | Path,
    *,
    decimal_places: int | None = 2,
) -> Path:
    """Write metadata to a deterministic UTF-8 text file.

    Args:
        meta: Metadata mapping to write.
        path: Output text-file path.
        decimal_places: Optional number of decimal places used when writing
            floating-point values. If None, values are written without rounding.

    Returns:
        Path to the written file.
    """
    p = Path(path)
    rounded = round_floats(dict(meta), decimal_places)
    p.write_text(_format(rounded) + "\n", encoding="utf-8")
    return p


def _format(meta: Any, indent: int = 0) -> str:
    """Format nested metadata as deterministic, human-readable text."""
    pad = "  " * indent

    if isinstance(meta, Mapping):
        lines: list[str] = []
        for key in sorted(meta):
            value = meta[key]
            if isinstance(value, Mapping | list | tuple):
                lines.append(f"{pad}{key}:")
                lines.append(_format(value, indent + 1))
            else:
                lines.append(f"{pad}{key}: {value}")
        return "\n".join(lines)

    if isinstance(meta, list | tuple):
        lines = []
        for item in meta:
            if isinstance(item, Mapping | list | tuple):
                lines.append(f"{pad}-")
                lines.append(_format(item, indent + 1))
            else:
                lines.append(f"{pad}- {item}")
        return "\n".join(lines)

    return f"{pad}{meta}"


def _weighted_quantile(
    z: np.ndarray,
    pdf: np.ndarray,
    q: float,
) -> float | None:
    """Return a quantile of a one-dimensional distribution on a grid.

    Args:
        z: Monotonic one-dimensional grid.
        pdf: Distribution values evaluated on ``z``. The distribution does not
            need to be normalized.
        q: Quantile in the interval [0, 1].

    Returns:
        Redshift value where the cumulative probability reaches ``q``.
        Returns None if the grid is empty or the distribution has zero total
        weight.
    """
    if z.size == 0:
        return None

    area = float(np.trapezoid(pdf, z))
    if area <= 0.0:
        return None

    pdf_norm = pdf / area

    if z.size == 1:
        return float(z[0])

    dz = np.diff(z)
    cdf = np.empty_like(z, dtype=float)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (pdf_norm[:-1] + pdf_norm[1:]) * dz)

    total = cdf[-1]
    if total <= 0.0:
        return None

    cdf = cdf / total
    return float(np.interp(q, cdf, z))


def _compute_effective_truez(
    z: Any,
    bins_returned: Mapping[int, Any],
) -> dict[int, dict[str, float | None]]:
    """Summarize the effective true-redshift distribution of each bin.

    For each returned bin curve ``n_i(z)``, this computes location and width
    summaries of the bin in true-redshift space. These summaries are useful for
    photo-z bins because the nominal photo-z bin edges do not correspond to
    sharp true-redshift boundaries.

    Args:
        z: Redshift grid shared by all bin curves.
        bins_returned: Mapping from bin index to per-bin distributions evaluated
            on ``z``.

    Returns:
        Mapping from bin index to summary statistics. Empty or zero-weight bins
        receive None-valued summaries.
    """
    z_arr = np.asarray(z, dtype=float)

    out: dict[int, dict[str, float | None]] = {}

    for i, n_bin in bins_returned.items():
        pdf = np.asarray(n_bin, dtype=float)
        area = float(np.trapezoid(pdf, z_arr))

        if area <= 0.0:
            out[int(i)] = _empty_truez_summary()
            continue

        pdf_norm = pdf / area

        out[int(i)] = {
            "z_mean": float(np.trapezoid(z_arr * pdf_norm, z_arr)),
            "z_median": _weighted_quantile(z_arr, pdf_norm, 0.50),
            "z_mode": float(z_arr[np.argmax(pdf_norm)]),
            "z_lo_68": _weighted_quantile(z_arr, pdf_norm, 0.16),
            "z_hi_68": _weighted_quantile(z_arr, pdf_norm, 0.84),
            "z_lo_95": _weighted_quantile(z_arr, pdf_norm, 0.025),
            "z_hi_95": _weighted_quantile(z_arr, pdf_norm, 0.975),
            "z_q05": _weighted_quantile(z_arr, pdf_norm, 0.05),
            "z_q25": _weighted_quantile(z_arr, pdf_norm, 0.25),
            "z_q75": _weighted_quantile(z_arr, pdf_norm, 0.75),
            "z_q95": _weighted_quantile(z_arr, pdf_norm, 0.95),
        }

    return out


def _empty_truez_summary() -> dict[str, float | None]:
    """Return the true-redshift summary used for empty bins."""
    return {
        "z_mean": None,
        "z_median": None,
        "z_mode": None,
        "z_lo_68": None,
        "z_hi_68": None,
        "z_lo_95": None,
        "z_hi_95": None,
        "z_q05": None,
        "z_q25": None,
        "z_q75": None,
        "z_q95": None,
    }


def _metadata_description() -> dict[str, Any]:
    """Return descriptions of the metadata fields."""
    return {
        "kind": "Tomography type: 'photoz' or 'specz'.",
        "grid": {
            "z": "Redshift grid used for all distributions.",
            "z_min": "Minimum redshift of the grid.",
            "z_max": "Maximum redshift of the grid.",
            "n": "Number of grid points.",
        },
        "parent_nz": {
            "values": "Parent redshift distribution evaluated on z.",
            "norm": "Optional normalization of the parent distribution.",
        },
        "bins": {
            "indices": "Indices of tomographic bins.",
            "n_bins": "Total number of bins.",
            "bin_edges": "Nominal bin edges.",
            "normalize_bins": (
                "Whether returned per-bin curves were normalized to unit integral. "
                "None means the convention was not recorded."
            ),
            "bins_returned": "Per-bin distributions n_i(z) evaluated on z.",
            "bins_norms": "Optional per-bin normalization values.",
            "frac_per_bin": "Optional fraction of galaxies per bin.",
            "density_per_bin": (
                "Optional number density per bin. The unit convention should "
                "be documented in inputs."
            ),
            "count_per_bin": "Optional galaxy counts per bin.",
            "truez_summary": (
                "Summary statistics of the effective true-redshift distribution in each bin."
            ),
        },
        "inputs": {
            "description": "User-provided configuration used to generate the bins.",
            "sample_properties": (
                "Optional sample-level observational metadata, such as number "
                "density, shape noise, shot noise, footprint, or volume. These "
                "quantities describe the sample and are not part of the redshift "
                "distribution model."
            ),
        },
        "notes": "Optional user-provided annotations.",
    }
