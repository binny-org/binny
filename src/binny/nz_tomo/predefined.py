"""Backward-compatible utilities for predefined tomographic redshift bins."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from binny.tomography.predefined import (
    build_predefined_tomography,
)
from binny.tomography.predefined import (
    load_predefined_bins as _load_predefined_bins,
)
from binny.utils.types import FloatArray

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
    """Load predefined bins using the functional implementation."""
    return _load_predefined_bins(
        path,
        z_col=z_col,
        bin_cols=bin_cols,
    )


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
    """Prepare predefined bins using the functional implementation."""
    result = build_predefined_tomography(
        z=z,
        bins=bins,
        nz=nz,
        normalize=normalize_bins,
        norm_method=norm_method,
        include_metadata=include_metadata,
    )

    return (
        result.z,
        result.nz,
        result.bins,
        result.metadata,
    )
