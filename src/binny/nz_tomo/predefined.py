"""Utilities for externally supplied tomographic redshift distributions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

from binny.utils.normalization import normalize_1d
from binny.utils.types import FloatArray
from binny.utils.validators import validate_axis_and_weights

__all__ = [
    "load_predefined_bins",
    "prepare_predefined_bins",
]


def load_predefined_bins(
    path: str | Path,
    *,
    z_col: int | str = "z",
    bin_cols: Mapping[int, int | str] | None = None,
) -> tuple[FloatArray, dict[int, FloatArray]]:
    """Load externally supplied tomographic bin curves from a text file.

    Column selectors may be either integer column positions or named columns.

    If ``bin_cols`` is omitted, the file must have named columns and columns
    named ``bin_<integer>`` are detected automatically.

    This function only reads and validates the supplied table. It does not
    normalize the curves, construct a parent n(z), split bins, or re-bin them.

    Args:
        path: Path to the input text file.
        z_col: Redshift column name or zero-based column position.
        bin_cols: Optional mapping from integer bin index to column name or
            zero-based column position.

    Returns:
        Shared redshift grid and mapping from integer bin index to n_i(z).

    Raises:
        FileNotFoundError: If the input file does not exist.
        TypeError: If bin indices or column selectors have invalid types.
        ValueError: If columns are missing, arrays have invalid shapes,
            values are non-finite, or the redshift grid is not increasing.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Predefined-bin file does not exist: {path}")

    if isinstance(z_col, bool) or not isinstance(
        z_col,
        int | np.integer | str,
    ):
        raise TypeError("z_col must be a column name or integer column position.")

    uses_named_columns = isinstance(z_col, str)

    if bin_cols is not None:
        if not isinstance(bin_cols, Mapping) or not bin_cols:
            raise ValueError("bin_cols must be a non-empty mapping when provided.")

        validated_bin_cols: dict[int, int | str] = {}

        for key, column in bin_cols.items():
            if isinstance(key, bool) or not isinstance(
                key,
                int | np.integer,
            ):
                raise TypeError("Predefined bin-column keys must be integers.")

            bin_index = int(key)

            if bin_index in validated_bin_cols:
                raise ValueError(f"Duplicate predefined bin index {bin_index}.")

            if isinstance(column, bool) or not isinstance(
                column,
                int | np.integer | str,
            ):
                raise TypeError(
                    "Predefined bin columns must be column names or integer column positions."
                )

            if isinstance(column, str) != uses_named_columns:
                raise ValueError(
                    "z_col and all bin_cols must use the same selector type: "
                    "either all names or all integer positions."
                )

            validated_bin_cols[bin_index] = column
    else:
        validated_bin_cols = {}

        if not uses_named_columns:
            raise ValueError("bin_cols must be provided when z_col is an integer column position.")

    if uses_named_columns:
        table = np.genfromtxt(
            path,
            names=True,
            dtype=np.float64,
            ndmin=1,
        )

        if table.dtype.names is None:
            raise ValueError(f"Could not read named columns from {path}.")

        column_names = list(table.dtype.names)
        z_column_name = str(z_col)

        if z_column_name not in column_names:
            raise ValueError(
                f"Input file {path} does not contain redshift column "
                f"{z_column_name!r}. Available columns: {column_names}."
            )

        if bin_cols is None:
            for column_name in column_names:
                if not column_name.startswith("bin_"):
                    continue

                suffix = column_name.removeprefix("bin_")

                try:
                    bin_index = int(suffix)
                except ValueError as error:
                    raise ValueError(
                        f"Column {column_name!r} starts with 'bin_' "
                        "but does not end with an integer bin index."
                    ) from error

                if bin_index in validated_bin_cols:
                    raise ValueError(f"Multiple columns map to predefined bin index {bin_index}.")

                validated_bin_cols[bin_index] = column_name

        if not validated_bin_cols:
            raise ValueError("No predefined tomographic bin columns were provided or detected.")

        z = np.asarray(
            table[z_column_name],
            dtype=np.float64,
        )

        bins: dict[int, FloatArray] = {}

        for bin_index, column in sorted(validated_bin_cols.items()):
            column_name = str(column)

            if column_name not in column_names:
                raise ValueError(
                    f"Input file {path} does not contain configured "
                    f"bin column {column_name!r}. "
                    f"Available columns: {column_names}."
                )

            bins[bin_index] = np.asarray(
                table[column_name],
                dtype=np.float64,
            )

    else:
        table = np.loadtxt(
            path,
            dtype=np.float64,
            ndmin=2,
        )

        n_columns = table.shape[1]
        z_column_index = int(z_col)

        if z_column_index < 0 or z_column_index >= n_columns:
            raise ValueError(
                f"z_col={z_column_index} is outside the available "
                f"column range 0 to {n_columns - 1} in {path}."
            )

        z = np.asarray(
            table[:, z_column_index],
            dtype=np.float64,
        )

        bins = {}

        for bin_index, column in sorted(validated_bin_cols.items()):
            column_index = int(column)

            if column_index < 0 or column_index >= n_columns:
                raise ValueError(
                    f"Configured column {column_index} for bin "
                    f"{bin_index} is outside the available column "
                    f"range 0 to {n_columns - 1} in {path}."
                )

            bins[bin_index] = np.asarray(
                table[:, column_index],
                dtype=np.float64,
            )

    if z.ndim != 1:
        raise ValueError("The loaded redshift column must be one-dimensional.")

    if z.size < 2:
        raise ValueError("The loaded redshift grid must contain at least two values.")

    if not np.all(np.isfinite(z)):
        raise ValueError("The loaded redshift grid contains non-finite values.")

    if not np.all(np.diff(z) > 0.0):
        raise ValueError("The loaded redshift grid must be strictly increasing.")

    for bin_index, curve in bins.items():
        if curve.ndim != 1:
            raise ValueError(f"Loaded predefined bin {bin_index} must be one-dimensional.")

        if curve.shape != z.shape:
            raise ValueError(
                f"Loaded predefined bin {bin_index} has shape "
                f"{curve.shape}, but z has shape {z.shape}."
            )

        if not np.all(np.isfinite(curve)):
            raise ValueError(f"Loaded predefined bin {bin_index} contains non-finite values.")

    return z, bins


def prepare_predefined_bins(
    z: Any,
    bins: Mapping[int, Any],
    *,
    nz: Any | None = None,
    normalize_bins: bool = False,
    norm_method: Literal["trapezoid", "simpson"] = "trapezoid",
    include_metadata: bool = False,
) -> tuple[
    FloatArray,
    FloatArray,
    dict[int, FloatArray],
    dict[str, Any] | None,
]:
    """Validate and prepare externally supplied tomographic bin curves.

    Args:
        z: Shared redshift grid.
        bins: Mapping from bin index to externally supplied n_i(z).
        nz: Optional parent n(z). If omitted, the raw bin curves are summed.
        normalize_bins: Whether to normalize each returned bin curve.
        norm_method: Numerical integration method used for normalization.
        include_metadata: Whether to return population metadata.

    Returns:
        Validated redshift grid, parent n(z), prepared bin curves, and optional
        metadata.
    """
    if not isinstance(bins, Mapping) or not bins:
        raise ValueError("bins must be a non-empty mapping.")

    z_arr = np.asarray(
        z,
        dtype=np.float64,
    )

    if z_arr.ndim != 1:
        raise ValueError("z must be one-dimensional.")

    if z_arr.size < 2:
        raise ValueError("z must contain at least two values.")

    if not np.all(np.isfinite(z_arr)):
        raise ValueError("z must contain only finite values.")

    if not np.all(np.diff(z_arr) > 0.0):
        raise ValueError("z must be strictly increasing.")

    raw_bins: dict[int, FloatArray] = {}
    bins_norms: dict[int, float] = {}

    for key, curve in bins.items():
        if isinstance(key, bool) or not isinstance(
            key,
            int | np.integer,
        ):
            raise TypeError("Predefined bin keys must be integers.")

        bin_index = int(key)

        if bin_index in raw_bins:
            raise ValueError(f"Duplicate bin index {bin_index}.")

        _, curve_arr = validate_axis_and_weights(
            z_arr,
            curve,
        )

        curve_arr = curve_arr.astype(
            np.float64,
            copy=True,
        )

        area = float(
            np.trapezoid(
                curve_arr,
                x=z_arr,
            )
        )

        if np.isclose(
            area,
            0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"Predefined bin {bin_index} has zero integral.")

        raw_bins[bin_index] = curve_arr
        bins_norms[bin_index] = area

    raw_bins = dict(sorted(raw_bins.items()))

    bins_norms = dict(sorted(bins_norms.items()))

    if nz is None:
        parent_arr = np.sum(
            np.stack(
                list(raw_bins.values()),
                axis=0,
            ),
            axis=0,
        )

        parent_source = "sum_of_bins"
    else:
        _, parent_arr = validate_axis_and_weights(
            z_arr,
            nz,
        )

        parent_arr = parent_arr.astype(
            np.float64,
            copy=True,
        )

        parent_source = "provided"

    output_bins: dict[int, FloatArray] = {}

    for bin_index, curve_arr in raw_bins.items():
        if normalize_bins:
            output_bins[bin_index] = normalize_1d(
                z_arr,
                curve_arr,
                method=norm_method,
            )
        else:
            output_bins[bin_index] = curve_arr.copy()

    metadata: dict[str, Any] | None = None

    if include_metadata:
        parent_norm = float(
            np.trapezoid(
                parent_arr,
                x=z_arr,
            )
        )

        if np.isclose(
            parent_norm,
            0.0,
            atol=1.0e-12,
        ):
            frac_per_bin = None
        else:
            frac_per_bin = {key: float(value / parent_norm) for key, value in bins_norms.items()}

        metadata = {
            "kind": "predefined",
            "parent_norm": parent_norm,
            "bins_norms": bins_norms,
            "frac_per_bin": frac_per_bin,
            "inputs": {
                "normalize_bins": normalize_bins,
                "norm_method": norm_method,
                "parent_source": parent_source,
            },
        }

    return (
        z_arr,
        parent_arr,
        output_bins,
        metadata,
    )
