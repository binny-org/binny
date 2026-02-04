"""Metadata utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

__all__ = ["build_tomo_bins_metadata", "save_metadata_txt", "round_floats"]


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
    notes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds metadata for tomographic redshift-bin products.

    This function packages tomographic bin outputs into a self-describing,
    serializable dictionary that can be saved alongside generated bin curves.
    It records the redshift grid, the parent distribution, the bin definition,
    the returned per-bin curves, and any optional population summaries supplied
    by the caller.

    The function is intentionally non-opinionated: it does not compute
    normalization constants, per-bin fractions, number densities, or counts.
    Callers that need population summaries should compute them under their own
    conventions and pass them in explicitly.

    Args:
      kind: Tomography mode label. Supported values are ``"photoz"`` and
        ``"specz"``.
      z: Redshift grid shared by the parent distribution and all bin curves.
      parent_nz: Parent redshift distribution evaluated on ``z``. This may be
        normalized or unnormalized.
      bin_edges: Bin edges that define the tomographic selection (edge convention
        is controlled by the caller).
      bins_returned: Mapping from bin index to the per-bin curve ``n_i(z)``
        evaluated on ``z``. These curves may be normalized per bin or may carry
        population information, depending on the caller.
      inputs: User-facing configuration that fully specifies how the bins were
        generated (e.g., binning scheme, scatter model parameters, completeness,
        leakage settings).
      parent_norm: Optional scalar associated with the parent distribution under
        the caller's convention (for example, the integral of the unnormalized
        parent curve on ``z``).
      bins_norms: Optional mapping from bin index to a population-carrying scalar
        under the caller's convention (for example, integrals of raw bin curves).
      frac_per_bin: Optional mapping from bin index to population fraction.
      density_per_bin: Optional mapping from bin index to number density.
      count_per_bin: Optional mapping from bin index to counts.
      notes: Optional free-form annotations to store verbatim.

    Returns:
      A nested dictionary containing metadata suitable for deterministic text or
      JSON dumps.
    """
    z_arr = np.asarray(z, dtype=float)
    parent_arr = np.asarray(parent_nz, dtype=float)
    edges_arr = np.asarray(bin_edges, dtype=float)

    # Store returned bins as lists for deterministic, text/JSON-friendly dumps.
    bins_out: dict[int, list[float]] = {
        int(k): np.asarray(v, dtype=float).tolist() for k, v in bins_returned.items()
    }

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
        },
        "inputs": dict(inputs),
    }

    if notes is not None:
        meta["notes"] = dict(notes)

    return meta


def _format(meta: Any, indent: int = 0) -> str:
    """Format metadata as deterministic, human-readable text."""
    pad = "  " * indent
    if isinstance(meta, Mapping):
        lines: list[str] = []
        for k in sorted(meta):
            v = meta[k]
            if isinstance(v, Mapping | list | tuple):
                lines.append(f"{pad}{k}:")
                lines.append(_format(v, indent + 1))
            else:
                lines.append(f"{pad}{k}: {v}")
        return "\n".join(lines)
    if isinstance(meta, (list | tuple)):
        lines: list[str] = []
        for item in meta:
            if isinstance(item, (Mapping | list | tuple)):
                lines.append(f"{pad}-")
                lines.append(_format(item, indent + 1))
            else:
                lines.append(f"{pad}- {item}")
        return "\n".join(lines)
    return f"{pad}{meta}"


def round_floats(obj: Any, decimal_places: int | None) -> Any:
    """Recursively rounds floats in nested metadata."""
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
    """Writes metadata to a UTF-8 text file."""
    p = Path(path)
    rounded = round_floats(dict(meta), decimal_places)
    p.write_text(_format(rounded) + "\n", encoding="utf-8")
    return p
