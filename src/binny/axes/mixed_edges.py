"""Utilities for combining adjacent arrays of bin boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["mixed_edges"]


def mixed_edges(
    segments: Sequence[Any],
    *,
    total_n_bins: int | None = None,
    atol: float = 1e-12,
) -> np.ndarray:
    """Combine adjacent sets of bin boundaries into one array.

    Each segment contains an independently constructed, strictly
    increasing array of boundaries. Adjacent segments must share
    a common endpoint. The shared boundary is included only once
    in the combined result.

    This function does not construct boundaries or interpret binning
    schemes. Any edge-construction method can be used to generate
    the input segments.

    Args:
        segments: Ordered sequence of one-dimensional arrays of bin
            boundaries. Each array must contain at least two values.
        total_n_bins: Optional expected number of bins across all
            segments. If provided, it must match the combined result.
        atol: Absolute tolerance used to compare adjacent segment
            endpoints. Must be finite and nonnegative.

    Returns:
        One-dimensional float64 array of strictly increasing bin
        boundaries. Its length is the total number of bins plus one.

    Raises:
        ValueError: If no segments are supplied, a segment contains
            invalid boundaries, adjacent segments do not share an
            endpoint, or the total number of bins is inconsistent.
    """
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be finite and nonnegative.")

    if not segments:
        raise ValueError("At least one segment is required.")

    if total_n_bins is not None:
        if (
            isinstance(total_n_bins, bool | np.bool_)
            or not isinstance(total_n_bins, int | np.integer)
            or total_n_bins < 1
        ):
            raise ValueError("total_n_bins must be a positive integer.")

    combined: list[np.ndarray] = []
    previous_right: float | None = None
    number_of_bins = 0

    for index, segment in enumerate(segments):
        edges = np.asarray(segment, dtype=np.float64)

        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(
                f"Segment {index} must contain at least two one-dimensional boundaries."
            )

        if not np.all(np.isfinite(edges)):
            raise ValueError(f"Segment {index} contains non-finite boundaries.")

        if not np.all(np.diff(edges) > 0):
            raise ValueError(f"Segment {index} boundaries must be strictly increasing.")

        # Count bins before removing the shared boundary.
        number_of_bins += edges.size - 1

        if previous_right is not None:
            if not np.isclose(
                edges[0],
                previous_right,
                rtol=0.0,
                atol=atol,
            ):
                raise ValueError(
                    f"Segment {index} starts at {edges[0]}, "
                    f"but the previous segment ends at {previous_right}."
                )

            # Preserve the previous segment's shared boundary.
            edges = edges[1:]

        combined.append(edges)
        previous_right = float(edges[-1])

    if total_n_bins is not None and number_of_bins != total_n_bins:
        raise ValueError(f"Expected {total_n_bins} bins, got {number_of_bins}.")

    result = np.concatenate(combined)

    if not np.all(np.diff(result) > 0):
        raise ValueError("Combined bin boundaries must be strictly increasing.")

    return result
