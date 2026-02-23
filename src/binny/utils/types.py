"""Shared typing aliases for Binny."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

# Scalars (inputs)
ScalarLike: TypeAlias = float | int | bool | np.floating | np.integer | np.bool_


# ndarrays (outputs/internal)
FloatArray: TypeAlias = NDArray[np.float64]
FloatArray1D: TypeAlias = NDArray[np.float64]
FloatArray2D: TypeAlias = NDArray[np.float64]

IntArray: TypeAlias = NDArray[np.int64]
IntArray1D: TypeAlias = NDArray[np.int64]
IntArray2D: TypeAlias = NDArray[np.int64]

BoolArray: TypeAlias = NDArray[np.bool_]
BoolArray1D: TypeAlias = NDArray[np.bool_]


# Array-ish inputs
FloatLike1D: TypeAlias = Sequence[float] | NDArray[np.floating]
FloatLike2D: TypeAlias = Sequence[Sequence[float]] | NDArray[np.floating]

IntLike1D: TypeAlias = Sequence[int] | NDArray[np.integer]
IntLike2D: TypeAlias = Sequence[Sequence[int]] | NDArray[np.integer]

# “Anything np.asarray accepts” (no ArrayLike unions; PyCharm-friendly)
Arrayish: TypeAlias = (
    ScalarLike | Sequence[object] | Sequence[Sequence[object]] | NDArray[np.generic]
)


# Common structures
IndexSeq: TypeAlias = Sequence[int]

EdgePair: TypeAlias = tuple[float, float]
EdgeMap: TypeAlias = Mapping[int, EdgePair]
EdgesLike: TypeAlias = EdgeMap | Sequence[float] | NDArray[np.floating]


# Binning schemes
BinningScheme: TypeAlias = str | Sequence[Mapping[str, Any]] | Mapping[str, Any]


# Correlation pairs
Pair = tuple[int, int]
Pairs = list[Pair]

IndexTuple = tuple[int, ...]
IndexTuples = list[IndexTuple]
