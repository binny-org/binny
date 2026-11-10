"""Tomography from externally supplied redshift distributions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

from binny.tomography.core import Tomography
from binny.utils.normalization import normalize_1d
from binny.utils.types import FloatArray
from binny.utils.validators import validate_axis_and_weights

__all__ = ["build_predefined_tomography", "load_predefined_bins"]


def load_predefined_bins(
    path: str | Path,
    *,
    z_col: int | str = "z",
    bin_cols: Mapping[int, int | str] | None = None,
) -> tuple[FloatArray, dict[int, FloatArray]]:
    """Load predefined bins from a text file.

    Supports named columns or zero-based integer column positions.
    Named columns matching ``bin_<integer>`` are detected automatically.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Predefined-bin file does not exist: {path}")

    if isinstance(z_col, bool) or not isinstance(z_col, int | np.integer | str):
        raise TypeError("z_col must be a column name or integer position.")

    named = isinstance(z_col, str)

    if bin_cols is None:
        if not named:
            raise ValueError("bin_cols is required for positional columns.")
        columns: dict[int, int | str] = {}
    else:
        if not isinstance(bin_cols, Mapping) or not bin_cols:
            raise ValueError("bin_cols must be a non-empty mapping.")
        columns = {}
        for key, value in bin_cols.items():
            if isinstance(key, bool) or not isinstance(key, int | np.integer):
                raise TypeError("Bin indices must be integers.")
            if isinstance(value, bool) or not isinstance(value, int | np.integer | str):
                raise TypeError("Column selectors must be names or integers.")
            if isinstance(value, str) != named:
                raise ValueError(
                    "z_col and all bin_cols must use the same selector type "
                    "(same type required: either all names or all integer positions)."
                )
            if int(key) in columns:
                raise ValueError(f"Duplicate bin index {int(key)}.")
            columns[int(key)] = value

    if named:
        table = np.genfromtxt(path, names=True, dtype=np.float64, ndmin=1)
        names = table.dtype.names
        if names is None or z_col not in names:
            raise ValueError(f"Missing redshift column {z_col!r}.")

        if bin_cols is None:
            for name in names:
                if name.startswith("bin_"):
                    try:
                        index = int(name.removeprefix("bin_"))
                    except ValueError as error:
                        raise ValueError(f"Invalid predefined column {name!r}.") from error
                    if index in columns:
                        raise ValueError(f"Duplicate bin index {index}.")
                    columns[index] = name

        if not columns:
            raise ValueError("No predefined bin columns found.")

        for column in columns.values():
            if column not in names:
                raise ValueError(f"Missing column {column!r}.")

        z = np.asarray(table[z_col], dtype=np.float64)
        bins = {
            index: np.asarray(table[column], dtype=np.float64)
            for index, column in sorted(columns.items())
        }
    else:
        table = np.loadtxt(path, dtype=np.float64, ndmin=2)
        n_columns = table.shape[1]
        selectors = [int(z_col), *(int(v) for v in columns.values())]
        if any(index < 0 or index >= n_columns for index in selectors):
            raise ValueError("Column index is outside the available range.")

        z = np.asarray(table[:, int(z_col)], dtype=np.float64)
        bins = {
            index: np.asarray(table[:, int(column)], dtype=np.float64)
            for index, column in sorted(columns.items())
        }

    if z.ndim != 1 or z.size < 2:
        raise ValueError("z must be one-dimensional with at least two values.")
    if not np.all(np.isfinite(z)) or not np.all(np.diff(z) > 0):
        raise ValueError("z must be finite and strictly increasing.")

    for index, curve in bins.items():
        if curve.shape != z.shape or not np.all(np.isfinite(curve)):
            raise ValueError(f"Invalid predefined bin {index}.")

    return z, bins


def build_predefined_tomography(
    z: Any,
    bins: Mapping[int, Any],
    *,
    nz: Any | None = None,
    normalize: bool = False,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    include_metadata: bool = False,
) -> Tomography:
    """Build tomography directly from supplied bin distributions.

    The parent n(z) is the sum of the raw bins unless explicitly provided.
    Optional normalization affects the returned bins, not the parent.
    """
    if not isinstance(bins, Mapping) or not bins:
        raise ValueError("bins must be a non-empty mapping.")

    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("z must be one-dimensional with at least two values.")
    if not np.all(np.isfinite(z)) or not np.all(np.diff(z) > 0):
        raise ValueError("z must be finite and strictly increasing.")

    raw_bins: dict[int, FloatArray] = {}
    bin_norms: dict[int, float] = {}

    for key, curve in bins.items():
        if isinstance(key, bool) or not isinstance(key, int | np.integer):
            raise TypeError("Predefined bin keys must be integers.")

        index = int(key)
        if index in raw_bins:
            raise ValueError(f"Duplicate bin index {index}.")

        _, values = validate_axis_and_weights(z, curve)
        values = values.astype(np.float64, copy=True)

        area = float(np.trapezoid(values, x=z))
        if np.isclose(area, 0.0, atol=1e-12):
            raise ValueError(f"Predefined bin {index} has zero integral.")

        raw_bins[index] = values
        bin_norms[index] = area

    raw_bins = dict(sorted(raw_bins.items()))
    bin_norms = dict(sorted(bin_norms.items()))

    if nz is None:
        parent = np.sum(np.stack(list(raw_bins.values())), axis=0)
        parent_source = "sum_of_bins"
    else:
        _, parent = validate_axis_and_weights(z, nz)
        parent = parent.astype(np.float64, copy=True)
        parent_source = "provided"

    output_bins = {
        index: (normalize_1d(z, curve, method=norm_method) if normalize else curve.copy())
        for index, curve in raw_bins.items()
    }

    metadata: dict[str, Any] = {}
    if include_metadata:
        parent_norm = float(np.trapezoid(parent, x=z))
        fractions = (
            None
            if np.isclose(parent_norm, 0.0, atol=1e-12)
            else {index: float(area / parent_norm) for index, area in bin_norms.items()}
        )
        metadata = {
            "kind": "predefined",
            "parent_norm": parent_norm,
            "bins_norms": bin_norms,
            "frac_per_bin": fractions,
            "inputs": {
                "normalize_bins": normalize,
                "norm_method": norm_method,
                "parent_source": parent_source,
            },
        }

    return Tomography(
        z=z,
        nz=parent,
        bins=output_bins,
        metadata=metadata,
    )
