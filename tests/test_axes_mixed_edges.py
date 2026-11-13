"""Unit tests for combining adjacent arrays of bin boundaries."""

import numpy as np
import pytest

from binny.axes.mixed_edges import mixed_edges


def test_mixed_edges_combines_adjacent_segments():
    """Tests that adjacent segments are combined without duplicate boundaries."""
    segments = [
        [0.0, 1.0, 2.0],
        [2.0, 3.0, 4.0],
    ]

    result = mixed_edges(segments, total_n_bins=4)

    np.testing.assert_array_equal(result, [0.0, 1.0, 2.0, 3.0, 4.0])


def test_mixed_edges_accepts_single_segment():
    """Tests that a single segment preserves its original boundaries."""
    result = mixed_edges([[0.0, 0.5, 1.0]])

    np.testing.assert_array_equal(result, [0.0, 0.5, 1.0])


def test_mixed_edges_accepts_numpy_arrays():
    """Tests that NumPy arrays can be combined into one boundary array."""
    segments = [
        np.array([0.0, 0.5, 1.0]),
        np.array([1.0, 1.5, 2.0]),
    ]

    result = mixed_edges(segments)

    np.testing.assert_array_equal(
        result,
        [0.0, 0.5, 1.0, 1.5, 2.0],
    )


def test_mixed_edges_accepts_matching_boundaries_within_tolerance():
    """Tests that nearly identical shared boundaries are accepted."""
    segments = [
        [0.0, 1.0, 2.0],
        [2.0 + 1e-13, 3.0, 4.0],
    ]

    result = mixed_edges(segments, atol=1e-12)

    np.testing.assert_array_equal(
        result,
        [0.0, 1.0, 2.0, 3.0, 4.0],
    )


def test_mixed_edges_rejects_boundary_mismatch():
    """Tests that nonadjacent segments raise a ValueError."""
    segments = [
        [0.0, 1.0, 2.0],
        [2.1, 3.0, 4.0],
    ]

    with pytest.raises(ValueError, match="previous segment ends"):
        mixed_edges(segments)


def test_mixed_edges_rejects_overlapping_segments():
    """Tests that overlapping segment intervals are rejected."""
    segments = [
        [0.0, 1.0, 2.0],
        [1.5, 2.5, 3.0],
    ]

    with pytest.raises(ValueError, match="previous segment ends"):
        mixed_edges(segments)


def test_mixed_edges_rejects_empty_segments():
    """Tests that an empty segment collection is rejected."""
    with pytest.raises(ValueError, match="At least one segment"):
        mixed_edges([])


@pytest.mark.parametrize(
    "segment",
    [
        [],
        [1.0],
        [[0.0, 1.0], [2.0, 3.0]],
    ],
)
def test_mixed_edges_rejects_invalid_dimensions(segment):
    """Tests that segments must be one-dimensional with two or more boundaries."""
    with pytest.raises(ValueError, match="at least two"):
        mixed_edges([segment])


@pytest.mark.parametrize(
    "segment",
    [
        [0.0, np.nan, 2.0],
        [0.0, np.inf, 2.0],
        [0.0, -np.inf, 2.0],
    ],
)
def test_mixed_edges_rejects_nonfinite_boundaries(segment):
    """Tests that segments containing nonfinite boundaries are rejected."""
    with pytest.raises(ValueError, match="non-finite"):
        mixed_edges([segment])


@pytest.mark.parametrize(
    "segment",
    [
        [0.0, 1.0, 1.0],
        [0.0, 2.0, 1.0],
        [2.0, 1.0, 0.0],
    ],
)
def test_mixed_edges_rejects_nonincreasing_boundaries(segment):
    """Tests that segment boundaries must be strictly increasing."""
    with pytest.raises(ValueError, match="strictly increasing"):
        mixed_edges([segment])


def test_mixed_edges_rejects_total_bin_count_mismatch():
    """Tests that the expected number of bins must match the actual count."""
    segments = [
        [0.0, 1.0, 2.0],
        [2.0, 3.0, 4.0],
    ]

    with pytest.raises(ValueError, match="Expected 5 bins, got 4"):
        mixed_edges(segments, total_n_bins=5)


@pytest.mark.parametrize(
    "total_n_bins",
    [0, -1, 2.5, True, None],
)
def test_mixed_edges_validates_total_bin_count(total_n_bins):
    """Tests that bin counts are positive integers or omitted."""
    segments = [[0.0, 1.0, 2.0]]

    if total_n_bins is None:
        result = mixed_edges(segments, total_n_bins=total_n_bins)
        assert result.size == 3
    else:
        with pytest.raises(ValueError, match="positive integer"):
            mixed_edges(segments, total_n_bins=total_n_bins)


@pytest.mark.parametrize(
    "atol",
    [-1.0, np.nan, np.inf],
)
def test_mixed_edges_rejects_invalid_tolerance(atol):
    """Tests that the absolute tolerance must be finite and nonnegative."""
    with pytest.raises(ValueError, match="finite and nonnegative"):
        mixed_edges([[0.0, 1.0]], atol=atol)


def test_mixed_edges_rejects_nonincreasing_combined_boundaries():
    """Tests that combined boundaries must remain strictly increasing."""
    segments = [
        [0.0, 1.0, 2.0],
        [2.0 - 1e-13, 2.0 - 5e-14, 3.0],
    ]

    with pytest.raises(ValueError, match="Combined bin boundaries"):
        mixed_edges(segments, atol=1e-12)


def test_mixed_edges_preserves_total_number_of_bins():
    """Tests that combining segments preserves the sum of their bin counts."""
    segments = [
        [0.0, 0.5, 1.0],
        [1.0, 1.5, 2.0, 2.5],
        [2.5, 3.0],
    ]

    result = mixed_edges(segments, total_n_bins=6)

    assert result.size == 7
    assert np.all(np.diff(result) > 0)
